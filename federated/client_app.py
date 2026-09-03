import json
import os
import time

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from common import build_tokenized_loader
from federated.config import load_run_config
from federated.runtime import (
    BaseModelCache,
    build_fake_args,
    get_lora_ndarrays,
    load_accountant_history,
    load_client_shard,
    save_accountant_history,
    save_client_privacy_report,
    set_lora_ndarrays,
    train_lock,
)
from train import train_dp_sgd

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "partitions", "manifest.json")


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


class FlowerLoRAClient(NumPyClient):
    """One simulated FL client: local QLoRA + Opacus DP-SGD on its own
    MIMIC-IV-Note shard, wrapped by secaggplus_mod (see ClientApp below) so
    its parameter update is never sent to the server in the clear."""

    def __init__(self, client_id, peft_model, tokenizer, shard, run_config):
        self.client_id = client_id
        self.peft_model = peft_model
        self.tokenizer = tokenizer
        self.shard = shard
        self.run_config = run_config

    def fit(self, parameters, config):
        # config carries this round's server_round (see server_app.py's
        # on_fit_config_fn); run_config carries the static, run-wide settings
        # from pyproject.toml. fit()-level config wins on overlap.
        fake_args = build_fake_args({**self.run_config, **config})

        loader = build_tokenized_loader(
            self.shard,
            self.tokenizer,
            fake_args.text_column,
            fake_args.max_length,
            fake_args.logical_batch_size_dp_sgd,
            shuffle=True,
        )

        target_delta = 1.0 / (10 * len(self.shard))
        prior_history = load_accountant_history(self.client_id)

        # self.peft_model is BaseModelCache's single shared object -- see
        # train_lock()'s docstring in runtime.py for why this whole
        # load-weights -> train -> read-weights sequence must be exclusive.
        # The acquire/release prints are a deliberate, cheap tripwire: if two
        # clients' prints ever interleave without a "released" between them,
        # that's proof the lock isn't actually exclusive.
        print(f"[client {self.client_id}] waiting for train_lock at {time.time():.2f}")
        with train_lock():
            print(f"[client {self.client_id}] ACQUIRED train_lock at {time.time():.2f}")
            _, keys = get_lora_ndarrays(self.peft_model)
            set_lora_ndarrays(self.peft_model, parameters, keys)
            final_eps, history = train_dp_sgd(
                self.peft_model, loader, target_delta, fake_args, accountant_history=prior_history
            )
            new_ndarrays, _ = get_lora_ndarrays(self.peft_model)
            print(f"[client {self.client_id}] RELEASING train_lock at {time.time():.2f}")

        save_accountant_history(self.client_id, history)

        server_round = config.get("server_round", 0)
        save_client_privacy_report(
            self.client_id, server_round, final_eps, fake_args.target_epsilon, target_delta, fake_args.max_grad_norm
        )

        # num_examples is SecAgg+'s weighting factor `w` (each client uploads
        # "[w, w*params]") -- must stay below pyproject.toml's
        # secagg-max-weight or it gets silently clipped.
        metrics = {"achieved_epsilon": final_eps, "client_id": self.client_id, "round": server_round}
        return new_ndarrays, len(self.shard), metrics


def client_fn(context: Context):
    # node_config (partition-id) is set correctly by the simulation runtime
    # regardless of run_config plumbing -- see federated/config.py for why
    # run_config itself is loaded from pyproject.toml directly instead.
    client_id = int(context.node_config["partition-id"])
    run_config = load_run_config()

    manifest = load_manifest()
    client_entry = next(c for c in manifest["clients"] if c["client_id"] == client_id)

    fake_args = build_fake_args(run_config)
    # Loaded once per process and reused across every round this client
    # activates for -- see BaseModelCache's docstring for why this matters.
    peft_model, tokenizer = BaseModelCache.get_or_load(fake_args)

    shard = load_client_shard(
        manifest["dataset"],
        client_entry["subject_ids"],
        manifest["eval_fraction"],
        manifest["seed"],
        one_note_per_subject=bool(run_config.get("one-note-per-subject", True)),
    )

    return FlowerLoRAClient(client_id, peft_model, tokenizer, shard, run_config).to_client()


# secaggplus_mod intercepts this ClientApp's TRAIN messages and does the
# SecAgg+ masking/secret-sharing dance around FlowerLoRAClient.fit()'s
# returned parameters -- fit() itself is unaware of SecAgg+ entirely.
app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])
