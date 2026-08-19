import argparse
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from opacus.validators import ModuleValidator

from common import DEFAULT_ADAPTER_DIR, add_data_args, build_bnb_config, build_tokenized_loader, load_split
from train import train_adamw, train_dp_sgd

# Linear layers LoRA adapters get attached to: the 4 attention projections
# (query/key/value/output) and the 3 MLP projections (gate/up/down). These are
# the actual nn.Linear submodule names inside OLMo3's transformer blocks.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    # Every tunable knob for this script lives here, so nothing needs to be
    # hardcoded elsewhere. Run `python qlora_finetune.py --help` to see all of them.
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
    return parser.parse_args()


def build_model_and_tokenizer(args):
    # This config only quantizes nn.Linear submodules (attention
    # q/k/v/o and MLP gate/up/down here) -- transformers replaces each of
    # those with a 4-bit Linear4bit layer at load time. Every other module
    # type (LayerNorm/RMSNorm, embeddings) is left as a regular nn.Parameter
    # in its original dtype; see the prepare_model_for_kbit_training call
    # below for why that matters.
    bnb_config = build_bnb_config()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        # OLMo3 base has no dedicated pad token; reuse EOS for batch padding.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    # Freezes all base weights (requires_grad=False) and upcasts any
    # remaining fp16/bf16 params (in practice, the LayerNorm/RMSNorm weights)
    # to fp32 for training stability.
    #
    # This does NOT overlap with bnb_4bit_compute_dtype above: that setting
    # only ever applies to the quantized `Linear4bit` submodules (attention
    # q/k/v/o and MLP gate/up/down), telling bitsandbytes what dtype to
    # dequantize *those* weights into for a matmul. BitsAndBytesConfig never
    # touches LayerNorm/RMSNorm in the first place -- it only replaces
    # nn.Linear layers with 4-bit ones, so norm layers are loaded as plain
    # nn.Parameter tensors in the checkpoint's native dtype (bf16 here) and
    # are left completely untouched by compute_dtype. Without this explicit
    # upcast, those norm params would just stay in bf16 with no fp32 boost at
    # all, since nothing else in the pipeline ever promotes them.
    model = prepare_model_for_kbit_training(model)

    # lora_r (LoRA rank): size of the low-rank matrices A (r x d_in) and B (d_out x r)
    # that approximate the weight update B@A, instead of learning a full
    # d_out x d_in delta directly.

    # lora_alpha: scaling factor applied to the LoRA update before it's added to the
    # frozen base output: out = base_layer(x) + lora_B(lora_A(x)) * (lora_alpha / r).
    # Raising r alone changes the typical magnitude of B@A, so alpha/r exists
    # to keep that output scale roughly stable as r is swept -- alpha acts
    # like a second, independent knob for how strongly the adapter's update
    # gets blended in, separate from r's role of controlling how expressive
    # (how high-rank) that update can be. Note lora_B is zero-initialized, so
    # at step 0, B@A = 0 regardless of alpha -- the scaling only starts to
    # matter once B moves away from zero during training. 2*r is a common
    # default ratio (used here: r=16, alpha=32 -> scaling=2.0).
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    # Walks the model, finds every submodule whose name matches
    # LORA_TARGET_MODULES, and replaces each one in place with a LoRA wrapper
    # layer: the original (quantized) layer is kept inside it as `base_layer`,
    # with new lora_A/lora_B nn.Linear submodules added alongside it. That
    # wrapper's forward is what actually computes
    #   base_layer(x) + lora_B(lora_A(dropout(x))) * scaling
    # i.e. get_peft_model is the step that installs the LoRA math described
    # above into the model, rather than just applying it in the abstract.
    # Only the newly added lora_A/lora_B params end up with requires_grad=True;
    # everything else (frozen by prepare_model_for_kbit_training above) stays
    # frozen. Because task_type="CAUSAL_LM", the returned object is wrapped as
    # a PeftModelForCausalLM, which is what makes save_pretrained() persist
    # only the adapter weights (not the full base model) and keeps generate()
    # working the same way as the underlying causal LM.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Opacus needs per-sample gradients for every trainable submodule. This
    # checks the model doesn't contain layer types it can't compute those for
    # (e.g. BatchNorm, which mixes information across samples in a batch).
    # Not an issue here since transformers use LayerNorm/RMSNorm instead.
    errors = ModuleValidator.validate(model, strict=False)
    print(f"ModuleValidator errors: {errors}")

    return model, tokenizer


def build_data_loader(tokenizer, args, batch_size):
    # Only the "train" side is used here; evaluate.py redoes this same split
    # (same seed/fraction) and reads the "test" side.
    dataset = load_split(args.dataset, "train", args.eval_fraction, args.seed)
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
    }
    with open(os.path.join(args.output_dir, "privacy_report.json"), "w") as f:
        json.dump(privacy_report, f, indent=2)

    print(f"Done. LoRA adapter + privacy_report.json saved to {args.output_dir}")


if __name__ == "__main__":
    main()
