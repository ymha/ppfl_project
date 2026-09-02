import argparse
import json
import os

from common import (
    DEFAULT_ADAPTER_DIR,
    add_data_args,
    build_model_and_tokenizer,
    build_tokenized_loader,
    load_split,
    select_one_note_per_subject,
)
from train import train_adamw, train_dp_sgd


def parse_args():
    # Every tunable knob for this script lives here, so nothing needs to be
    # hardcoded elsewhere. Run `python -m centralized.qlora_finetune --help`
    # to see all of them.
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    # Only meaningful with --dp-sgd: sets the Poisson sample rate that
    # Opacus's privacy accounting is based on (the *logical* batch), and can
    # safely be far larger than the GPU can fit -- see --batch-size, which is
    # what actually runs on the GPU in either mode.
    parser.add_argument("--logical-batch-size-dp-sgd", type=int, default=8)
    # The actual per-step GPU batch. With --adamw, this is used directly as
    # the training batch size. With --dp-sgd, Opacus's BatchMemoryManager
    # uses it to cap memory by splitting each logical batch
    # (--logical-batch-size-dp-sgd) into physical chunks of at most this size
    # before the forward/backward pass; it still only clips+noises+steps once
    # per logical batch, so the privacy accounting is unaffected by this
    # value, only the peak GPU memory. Measured on a 24GB RTX 4090 with this
    # model/LoRA config: 8 peaks at ~9GiB, 16 at ~19GiB, 32 OOMs -- so keep
    # this at 8-12 regardless of how large --logical-batch-size-dp-sgd is set.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # Differential privacy budget. Opacus will pick the noise level needed to
    # stay within this (epsilon, delta) budget over the given number of epochs.
    parser.add_argument("--target-epsilon", type=float, default=8.0)
    parser.add_argument("--target-delta", type=float, default=None, help="defaults to 1/(10*len(dataset))")
    # Per-sample gradient clipping threshold (the "C" in DP-SGD's clip-then-noise step).
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # Exactly one of these picks the training mode -- there is no implicit
    # default, so it's always explicit which one a given run used.
    mode_group = parser.add_mutually_exclusive_group(required=True)
    # Opacus DP-SGD: per-sample gradient clipping + Gaussian noise, spends a
    # (epsilon, delta) privacy budget. See train_dp_sgd() in train.py.
    mode_group.add_argument("--dp-sgd", action="store_true", help="train with Opacus DP-SGD (private)")
    # Plain QLoRA fine-tune, a normal AdamW step with no per-sample clipping
    # or noise. This is the non-private utility upper bound: train it into a
    # separate --output-dir and pass both adapters to evaluate.py to see
    # exactly what DP-SGD's clip+noise step costs. See train_adamw() in train.py.
    mode_group.add_argument("--adamw", action="store_true", help="train without DP (non-private baseline)")
    # Stop after this many optimizer steps regardless of epochs -- for
    # quickly smoke-testing the train -> save -> evaluate pipeline before
    # committing to a full run over the whole dataset.
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--output-dir", default=DEFAULT_ADAPTER_DIR)
    # Off by default, so existing single-machine runs (e.g. the ones behind
    # RESULTS.md) are unaffected unless explicitly opted into. When set,
    # collapses the training split to one (seeded-random) note per
    # subject_id before training -- see common.select_one_note_per_subject()
    # for why this turns the reported (epsilon, delta) from a per-note into a
    # genuine per-patient (user-level) guarantee: no subject then contributes
    # more than one training example, so a neighboring dataset differing in
    # one record is exactly one differing in one patient's entire
    # contribution. Matches federated/'s default behavior (see
    # pyproject.toml's one-note-per-subject) for a single-machine equivalent
    # experiment. Only affects the training split -- the held-out eval split
    # is untouched, since it's never part of the DP mechanism.
    parser.add_argument(
        "--one-note-per-subject",
        action="store_true",
        help="collapse the training set to one note per subject_id, for a user-level "
        "(per-patient) DP guarantee instead of a per-note one",
    )
    return parser.parse_args()


def build_data_loader(tokenizer, args, batch_size):
    # Only the "train" side is used here; evaluate.py redoes this same split
    # (same seed/fraction) and reads the "test" side.
    dataset = load_split(args.dataset, "train", args.eval_fraction, args.seed)
    if args.one_note_per_subject:
        # Must happen before dataset_size is read off below, so target_delta
        # (1/(10*dataset_size)) and Opacus's sample_rate (1/len(data_loader))
        # both reflect the actual (smaller, per-patient) training set size.
        dataset = select_one_note_per_subject(dataset, args.seed)
    # NOTE: for --dp-sgd, this plain DataLoader gets replaced by Opacus with a
    # Poisson-sampling DataLoader inside train_dp_sgd() -- the privacy
    # accounting depends on that.
    data_loader = build_tokenized_loader(
        dataset, tokenizer, args.text_column, args.max_length, batch_size, shuffle=True
    )
    return data_loader, len(dataset)


def main():
    args = parse_args()
    peft_model, tokenizer = build_model_and_tokenizer(args)
    # --dp-sgd needs the loader built at the *logical* batch size so Opacus
    # derives the right Poisson sample rate; --adamw has no such distinction
    # and just trains at --batch-size directly.
    batch_size = args.logical_batch_size_dp_sgd if args.dp_sgd else args.batch_size
    data_loader, dataset_size = build_data_loader(tokenizer, args, batch_size)
    # Standard rule of thumb: delta should be much smaller than 1/dataset_size,
    # otherwise the privacy guarantee becomes vacuous (allows leaking a
    # non-negligible fraction of individual records with "certainty").
    target_delta = args.target_delta or 1.0 / (10 * dataset_size)

    if args.adamw:
        train_adamw(peft_model, data_loader, args)
        final_eps = None
    else:
        # train_dp_sgd() wraps peft_model in Opacus's GradSampleModule in
        # place; peft_model itself keeps updating and still exposes
        # save_pretrained() afterwards.
        final_eps, _ = train_dp_sgd(peft_model, data_loader, target_delta, args)

    # peft_model.save_pretrained only writes the small LoRA adapter weights
    # (a few hundred MB), not the full frozen 7B base model.
    peft_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Record the actual privacy guarantee this adapter was trained under, so
    # it can be audited later without re-running training. For a --adamw run
    # there is no guarantee to record, so the epsilon/delta/clip fields are
    # left null and evaluate.py treats that as the non-private baseline.
    privacy_report = {
        "dp_enabled": args.dp_sgd,
        "target_epsilon": args.target_epsilon if args.dp_sgd else None,
        "achieved_epsilon": final_eps,
        "delta": target_delta if args.dp_sgd else None,
        "max_grad_norm": args.max_grad_norm if args.dp_sgd else None,
        "epochs": args.epochs,
        # "user" here means the reported (epsilon, delta) is a per-patient
        # guarantee (--one-note-per-subject: no subject contributes more than
        # one training example, so a neighboring-dataset record is exactly a
        # neighboring-dataset patient). "record" means the same numbers are
        # only a per-note guarantee -- a patient with N notes in the training
        # set gets a weaker (roughly N-fold, via group privacy) actual
        # guarantee than the headline epsilon suggests.
        "guarantee_level": "user" if args.one_note_per_subject else "record",
    }
    with open(os.path.join(args.output_dir, "privacy_report.json"), "w") as f:
        json.dump(privacy_report, f, indent=2)

    print(f"Done. LoRA adapter + privacy_report.json saved to {args.output_dir}")


if __name__ == "__main__":
    main()
