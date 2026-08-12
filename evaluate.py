import argparse
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common import DEFAULT_ADAPTER_DIR, add_data_args, build_bnb_config, build_tokenized_loader, load_split


def parse_args():
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="also report the un-adapted base model's perplexity, for a before/after utility comparison",
    )
    parser.add_argument(
        "--compare-adapter-dir",
        default=None,
        help="a second, already-trained adapter (e.g. a --adamw run) to evaluate in the same pass, "
        "for a DP vs non-DP utility comparison",
    )
    # Held-out sets can be tens of thousands of rows; cap batches for a quick
    # smoke test of the eval pipeline before running the full pass.
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def build_eval_loader(tokenizer, args):
    dataset = load_split(args.dataset, "test", args.eval_fraction, args.seed)
    return build_tokenized_loader(
        dataset, tokenizer, args.text_column, args.max_length, args.batch_size, shuffle=False
    )


@torch.no_grad()
def perplexity(model, data_loader, device, max_batches=None):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(data_loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        # out.loss is already averaged over this batch's non-masked (-100)
        # label positions, so weight by that count before combining batches
        # into one corpus-level average -- otherwise a half-empty last batch
        # would count as much as a full one.
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss)


def print_privacy(adapter_dir, label):
    # privacy_report.json is written by qlora_finetune.py's main(); a --adamw
    # run still writes one, just with dp_enabled=False and null epsilon/delta.
    path = os.path.join(adapter_dir, "privacy_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        privacy = json.load(f)
    if privacy.get("dp_enabled", True) and privacy.get("achieved_epsilon") is not None:
        print(f"{label} trained under epsilon={privacy['achieved_epsilon']:.2f}, delta={privacy['delta']:.2e}")
    else:
        print(f"{label} trained without DP (--adamw baseline)")
    return privacy


def main():
    args = parse_args()
    bnb_config = build_bnb_config()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir, adapter_name="primary")

    data_loader = build_eval_loader(tokenizer, args)
    device = torch.device("cuda")

    loss, ppl = perplexity(model, data_loader, device, args.max_batches)
    print(f"Adapter ({args.adapter_dir}): loss={loss:.4f} perplexity={ppl:.2f}")
    results = {"adapter_dir": args.adapter_dir, "adapter_loss": loss, "adapter_perplexity": ppl}

    if args.compare_base:
        # disable_adapter() routes forward() through base_layer only, so this
        # runs the frozen, un-tuned base model on the same held-out batches --
        # a fair before/after utility comparison.
        with model.disable_adapter():
            base_loss, base_ppl = perplexity(model, data_loader, device, args.max_batches)
        print(f"Base model (no adapter): loss={base_loss:.4f} perplexity={base_ppl:.2f}")
        results["base_loss"] = base_loss
        results["base_perplexity"] = base_ppl

    if args.compare_adapter_dir:
        # Loads a second LoRA adapter onto the same base model and switches
        # the active one, so e.g. a DP-SGD run and a --adamw run (both
        # evaluated on the identical held-out split) can be compared in one
        # pass without loading the 7B base model twice.
        model.load_adapter(args.compare_adapter_dir, adapter_name="compare")
        model.set_adapter("compare")
        cmp_loss, cmp_ppl = perplexity(model, data_loader, device, args.max_batches)
        print(f"Adapter ({args.compare_adapter_dir}): loss={cmp_loss:.4f} perplexity={cmp_ppl:.2f}")
        model.set_adapter("primary")
        results["compare_adapter_dir"] = args.compare_adapter_dir
        results["compare_adapter_loss"] = cmp_loss
        results["compare_adapter_perplexity"] = cmp_ppl

    primary_privacy = print_privacy(args.adapter_dir, "Primary adapter")
    if primary_privacy:
        results["achieved_epsilon"] = primary_privacy["achieved_epsilon"]
        results["delta"] = primary_privacy["delta"]

    if args.compare_adapter_dir:
        compare_privacy = print_privacy(args.compare_adapter_dir, "Compare adapter")
        if compare_privacy:
            results["compare_achieved_epsilon"] = compare_privacy["achieved_epsilon"]
            results["compare_delta"] = compare_privacy["delta"]

    with open(os.path.join(args.adapter_dir, "eval_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved eval_report.json to {args.adapter_dir}")


if __name__ == "__main__":
    main()
