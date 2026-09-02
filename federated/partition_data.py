import argparse
import json
import os
import random
from collections import Counter

from common import add_data_args, load_split
from federated.config import load_run_config

DEFAULT_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "partitions", "manifest.json")


def parse_args():
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    parser.add_argument("--output", default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    # No --num-clients flag on purpose: always taken from pyproject.toml, so
    # the manifest this script writes can never disagree with what
    # server_app.py expects (see its assert on manifest["clients"] length).
    num_clients = load_run_config()["num-clients"]
    # Matches server_app.py's own num_clients >= 2 assert (SecAgg+ requires at
    # least 2 clients per round -- with fewer, Flower's SecAggPlusWorkflow
    # silently no-ops every round instead of raising). Checked here too, before
    # the full-dataset load_split() below, so a bad pyproject.toml config fails
    # in seconds instead of after partitioning.
    assert num_clients >= 2, (
        f"pyproject.toml's num-clients={num_clients}, but SecAgg+ requires at "
        f"least 2 clients per round -- see server_app.py's matching check"
    )

    # Identical calls to what centralized/qlora_finetune.py and
    # centralized/evaluate.py already use, so this partition is carved from
    # the exact same train/eval split -- a client shard can never contain a
    # row from the held-out eval set.
    train_split = load_split(args.dataset, "train", args.eval_fraction, args.seed)
    eval_split = load_split(args.dataset, "test", args.eval_fraction, args.seed)

    train_subjects = set(train_split["subject_id"])
    eval_subjects = set(eval_split["subject_id"])
    # load_split() (common.py) splits at the row (note) level, not the subject level
    # -- a subject_id's other notes can legitimately land on either side.
    # What we DO guarantee below is the FL-specific invariant:
    # No single row is assigned to two different clients, and
    # No subject's notes are split across two clients (simulating
    # siloed-by-patient hospital data).
    subject_overlap = train_subjects & eval_subjects
    print(
        f"Note: the split is done at the note level, not the patient level. "
        f"So, a patient (subject_id) can appear in both train and test. "
        f"A patient usually has multiple notes. "
        f"({100 * len(subject_overlap) / len(eval_subjects):.1f}%) of test-split "
        f"patients also have at least one note in the train split."
    )

    # Count the number of notes per subject_id.
    # Per-client row counts are needed for SecAgg+'s max_weight check below.
    # (Local DP-SGD itself runs entirely on the client side.)
    note_counts = Counter(train_split["subject_id"])

    subject_ids = sorted(train_subjects)
    random.Random(args.seed).shuffle(subject_ids)
    # Round-robin over the sorted-then-shuffled subject list.
    # This balances shard sizes by subject count, not by note count.
    # clients can still end up with different numbers of notes.
    # That's why client_app.py reports each client's true note count as num_examples:
    # FedAvg/SecAgg+ weight the aggregate by it, keeping the imbalance fair.
    shards = [subject_ids[i :: num_clients] for i in range(num_clients)]

    # Round-robin slicing is disjoint by construction.
    # No subject appears in two clients' data.
    seen = set()
    for shard in shards:
        overlap = seen & set(shard)
        assert not overlap, f"{len(overlap)} subject_ids duplicated across shards"
        seen |= set(shard)

    clients = []
    for client_id, shard in enumerate(shards):
        num_rows = sum(note_counts[sid] for sid in shard)
        clients.append({"client_id": client_id, "subject_ids": shard, "num_rows": num_rows})

    manifest = {
        "num_clients": num_clients,
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "dataset": args.dataset,
        "clients": clients,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    row_counts = [c["num_rows"] for c in clients]
    print(f"Wrote manifest for {num_clients} clients to {args.output}")
    print(f"Row counts per client: {row_counts} (min={min(row_counts)}, max={max(row_counts)})")
    print(f"Total rows across shards: {sum(row_counts)} (train split has {len(train_split)})")
    print("Self-check passed: zero subject_id overlap between shards (no patient split across clients).")


if __name__ == "__main__":
    main()
