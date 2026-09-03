import contextlib
import fcntl
import json
import os
from types import SimpleNamespace

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from common import build_model_and_tokenizer, load_split, select_one_note_per_subject

CLIENT_STATE_DIR = os.path.join(os.path.dirname(__file__), "client_state")
_TRAIN_LOCK_PATH = os.path.join(CLIENT_STATE_DIR, ".train.lock")


@contextlib.contextmanager
def train_lock():
    """Cross-task mutual exclusion around BaseModelCache's shared PeftModel.

    Flower's simulation backend can dispatch multiple sampled clients' fit()
    calls with overlapping execution, even with client-resources
    num_gpus=1.0 requesting the whole GPU per activation (confirmed
    empirically: with 5 clients, most fit() calls raced and hit Opacus's
    "Trying to add hooks twice to the same model" -- num_gpus governs Ray's
    scheduling weight, not a hard mutual-exclusion lock on client task
    execution). A plain in-process threading.Lock was tried first and did
    NOT reliably prevent this -- it's unclear whether Ray's task dispatch
    for this backend genuinely uses separate OS threads, separate
    (sub)processes sharing a reported pid, or something else, so an
    in-memory Python lock's shared-object assumption can't be trusted. An
    OS-level flock() on a real file works regardless of which of those is
    actually true, since the kernel enforces it independent of any Python
    object identity.
    """
    os.makedirs(CLIENT_STATE_DIR, exist_ok=True)
    with open(_TRAIN_LOCK_PATH, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def get_lora_ndarrays(peft_model):
    # Sorted key order is the single source of truth both sides agree on --
    # client and server never need to exchange key names, only the ndarray
    # list itself (Flower's Parameters are unlabeled).
    state_dict = get_peft_model_state_dict(peft_model)
    keys = sorted(state_dict.keys())
    ndarrays = [state_dict[k].detach().cpu().float().numpy() for k in keys]
    return ndarrays, keys


def set_lora_ndarrays(peft_model, ndarrays, keys):
    state_dict = {k: torch.from_numpy(v) for k, v in zip(keys, ndarrays)}
    set_peft_model_state_dict(peft_model, state_dict)


def build_fake_args(run_config):
    # Maps Flower's flat run_config dict (TOML scalars only: bool/int/float/
    # str) onto the exact attribute names build_model_and_tokenizer(),
    # build_tokenized_loader(), and train_dp_sgd() already expect, so those
    # functions can be called completely unmodified from FL client code.
    max_steps = run_config.get("max-steps", 0)
    return SimpleNamespace(
        model_id=run_config["model-id"],
        dataset=run_config.get("dataset-path", ""),
        text_column=run_config.get("text-column", "text"),
        max_length=int(run_config.get("max-length", 128)),
        eval_fraction=float(run_config["eval-fraction"]),
        seed=int(run_config["seed"]),
        lora_r=int(run_config["lora-r"]),
        lora_alpha=int(run_config["lora-alpha"]),
        lora_dropout=float(run_config["lora-dropout"]),
        epochs=int(run_config.get("local-epochs", 1)),
        lr=float(run_config.get("lr", 2e-4)),
        batch_size=int(run_config["local-batch-size"]),
        logical_batch_size_dp_sgd=int(run_config["local-logical-batch-size-dp-sgd"]),
        target_epsilon=float(run_config["target-epsilon-per-round"]),
        max_grad_norm=float(run_config["max-grad-norm"]),
        max_steps=None if not max_steps else int(max_steps),
        log_every=int(run_config.get("log-every", 5)),
    )


def load_client_shard(dataset_path, subject_ids, eval_fraction, seed, one_note_per_subject=True):
    # one_note_per_subject on by default, so every patient contributes exactly
    # one training example -- pass False (or set pyproject.toml's
    # one-note-per-subject to false) to opt out and let a client's shard keep
    # ALL of its subjects' notes instead (a subject with many notes then
    # contributes proportionally more training examples). See
    # common.select_one_note_per_subject() (shared with the single-machine
    # --one-note-per-subject path in centralized/qlora_finetune.py) for why
    # this is what turns the record-level DP-SGD guarantee into a per-patient
    # one.
    dataset = load_split(dataset_path, "train", eval_fraction, seed)
    subject_id_set = set(subject_ids)
    shard = dataset.filter(lambda row: row["subject_id"] in subject_id_set)
    if one_note_per_subject:
        shard = select_one_note_per_subject(shard, seed)
    return shard


class BaseModelCache:
    """Process-global cache for the loaded 4-bit base model + PEFT wrapper.

    Flower's simulation backend reuses the same Ray actor process across a
    client's activations round to round (with client-resources configured
    for one activation at a time -- see pyproject.toml), so caching here
    means the 7B checkpoint is quantized/loaded once per process, not once
    per (client, round) pair. Round-to-round reuse of the same PeftModel
    object works because train.py's train_dp_sgd() cleans up Opacus's
    per-sample-gradient hooks after each call -- without that, a second
    train_dp_sgd() call on the same object would raise "Trying to add hooks
    twice to the same model".
    """

    _model = None
    _tokenizer = None
    _load_count = 0

    @classmethod
    def get_or_load(cls, args):
        # Guards against two concurrently-dispatched clients both seeing
        # cls._model is None and racing to load a second 7B copy. Reuses the
        # same OS-level file lock as train_lock() for the same robustness
        # reason (see its docstring).
        with train_lock():
            if cls._model is None:
                cls._load_count += 1
                print(f"[BaseModelCache] loading base model (load #{cls._load_count}, pid={os.getpid()})")
                cls._model, cls._tokenizer = build_model_and_tokenizer(args)
        return cls._model, cls._tokenizer

    @classmethod
    def load_count(cls):
        return cls._load_count


def _history_path(client_id):
    return os.path.join(CLIENT_STATE_DIR, f"client_{client_id}_privacy.json")


def _report_path(client_id):
    return os.path.join(CLIENT_STATE_DIR, f"client_{client_id}_privacy_report.json")


def load_accountant_history(client_id):
    path = _history_path(client_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        # Opacus's accountant.history is a list of (noise_multiplier,
        # sample_rate, num_steps) tuples; JSON round-trips them as lists,
        # which unpack identically (a, b, c = [x, y, z] works the same as
        # a, b, c = (x, y, z)), so no conversion back to tuples is needed.
        return json.load(f)


def save_accountant_history(client_id, history):
    os.makedirs(CLIENT_STATE_DIR, exist_ok=True)
    with open(_history_path(client_id), "w") as f:
        json.dump(history, f)


def save_client_privacy_report(client_id, server_round, achieved_epsilon, target_epsilon, delta, max_grad_norm):
    os.makedirs(CLIENT_STATE_DIR, exist_ok=True)
    report = {
        "dp_enabled": True,
        "client_id": client_id,
        "round": server_round,
        "target_epsilon": target_epsilon,
        "achieved_epsilon": achieved_epsilon,
        "delta": delta,
        "max_grad_norm": max_grad_norm,
    }
    with open(_report_path(client_id), "w") as f:
        json.dump(report, f, indent=2)
    return report


class LatestParamsHolder:
    """Mutable box the server's evaluate_fn stashes each round's aggregated
    LoRA ndarrays into, so the final global state is retrievable after
    SecAggPlusWorkflow finishes without digging into Strategy internals."""

    round = None
    ndarrays = None

    @classmethod
    def update(cls, server_round, ndarrays):
        cls.round = server_round
        cls.ndarrays = ndarrays
