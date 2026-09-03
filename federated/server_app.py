import json
import os

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, SimpleClientManager
from flwr.server.compat import LegacyContext
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from common import build_eval_loader, build_model_and_tokenizer, perplexity
from federated.client_app import load_manifest
from federated.config import load_run_config
from federated.runtime import LatestParamsHolder, build_fake_args, get_lora_ndarrays, set_lora_ndarrays
from federated.privacy_report import aggregate_privacy_reports

GLOBAL_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "global-adapter")

app = ServerApp()


@app.main()
def main(grid, context: Context) -> None:
    # See federated/config.py: run_config is loaded from pyproject.toml
    # rather than from context.run_config.
    run_config  = load_run_config()
    num_clients = int(run_config["num-clients"])
    num_rounds  = int(run_config["num-server-rounds"])

    manifest = load_manifest()
    # The one thing that can silently desync client and server: the manifest
    # partition_data.py wrote must actually have num_clients shards, or some
    # clients would be training on a shard that doesn't exist / colliding
    # with another client's supposed shard.
    assert len(manifest["clients"]) == num_clients, (
        f"pyproject.toml's num-clients={num_clients} does not match "
        f"manifest.json's {len(manifest['clients'])} client shards - rerun partition_data.py"
    )
    # SecAggPlusWorkflow.setup_stage() hard-requires at least 2 sampled clients
    # (pairwise masking has no counterpart to cancel against with only one) --
    # with fewer, it logs an error and returns False, which just makes
    # DefaultWorkflow silently skip every round's fit with no exception raised.
    # Centralized eval still runs each round regardless, so the run "succeeds"
    # end-to-end while saving an untrained (zero-init LoRA) adapter. Fail loudly
    # here instead, before paying for a 7B model load.
    assert num_clients >= 2, (
        f"num-clients={num_clients}, but SecAgg+ requires at least 2 clients per "
        f"round (see flwr's SecAggPlusWorkflow.setup_stage) -- with fewer, fit "
        f"silently no-ops every round instead of raising"
    )
    # Also validated here (not just inside the workflow later) so a bad config
    # fails before the 7B model load below, not after.
    secagg_max_weight = float(run_config["secagg-max-weight"])
    max_shard_rows = max(c["num_rows"] for c in manifest["clients"])
    assert secagg_max_weight > max_shard_rows, (
        f"secagg-max-weight={secagg_max_weight} must exceed the largest shard's "
        f"row count ({max_shard_rows}), or SecAgg+ silently clips client weights"
    )

    fake_args = build_fake_args(run_config)
    fake_args.dataset = manifest["dataset"]

    # Built once, server-side only: gives a shape-correct, zero-initialized
    # (lora_B is zero-init by construction) starting adapter for
    # initial_parameters, and is reused after the run to hold + save the
    # final aggregated LoRA weights and to run centralized held-out eval.
    peft_model, tokenizer = build_model_and_tokenizer(fake_args)
    initial_ndarrays, keys = get_lora_ndarrays(peft_model)
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    eval_loader = build_eval_loader(tokenizer, fake_args)
    eval_max_batches = run_config.get("eval-max-batches", 0) or None
    device = torch.device("cuda")

    def evaluate_fn(server_round, parameters_ndarrays, config):
        set_lora_ndarrays(peft_model, parameters_ndarrays, keys)
        loss, ppl = perplexity(peft_model, eval_loader, device, eval_max_batches)
        print(f"[server_app] round {server_round}: held-out loss={loss:.4f} perplexity={ppl:.2f}")
        LatestParamsHolder.update(server_round, parameters_ndarrays)
        return loss, {"perplexity": ppl}

    def on_fit_config_fn(server_round):
        return {"server_round": server_round}

    # Deliberately NOT cast to int.
    # TOML gives these back as whatever numeric type was authored (int or float)
    # SecAggPlusWorkflow treats the two differently:
    # A float is a *proportion* of participating clients (see
    # pyproject.toml's comment), so e.g. int(0.6) silently truncating to 0
    # would corrupt the intended 60% threshold into "no shares needed".
    secagg_num_shares = run_config["secagg-num-shares"]
    secagg_threshold = run_config["secagg-reconstruction-threshold"]
    # secagg_max_weight/max_shard_rows already validated above, before the
    # model load.

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # no federated/client-side eval -- centralized evaluate_fn only
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
    )

    legacy_context = LegacyContext(
        context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_manager=SimpleClientManager(),
    )

    secagg_timeout = run_config["secagg-timeout"]
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=secagg_num_shares,
            reconstruction_threshold=secagg_threshold,
            max_weight=secagg_max_weight,
            clipping_range=float(run_config["secagg-clipping-range"]),
            quantization_range=int(run_config["secagg-quantization-range"]),
            timeout=None if not secagg_timeout else float(secagg_timeout),
        )
    )

    workflow(grid, legacy_context)

    assert LatestParamsHolder.ndarrays is not None, "no round ever completed evaluation"
    set_lora_ndarrays(peft_model, LatestParamsHolder.ndarrays, keys)
    peft_model.save_pretrained(GLOBAL_ADAPTER_DIR)
    tokenizer.save_pretrained(GLOBAL_ADAPTER_DIR)
    print(f"Saved federated global adapter to {GLOBAL_ADAPTER_DIR} (final round={LatestParamsHolder.round})")

    privacy_report = aggregate_privacy_reports(num_clients=num_clients, num_rounds=num_rounds)
    with open(os.path.join(GLOBAL_ADAPTER_DIR, "privacy_report.json"), "w") as f:
        json.dump(privacy_report, f, indent=2)
    print(f"Saved privacy_report.json: system_worst_case_epsilon={privacy_report['system_worst_case_epsilon']}")
