import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# Base model to fine-tune. This is the pretrained (non-instruct) OLMo 3 7B checkpoint.
MODEL_ID = "allenai/Olmo-3-1025-7B"

# Linear layers LoRA adapters get attached to: the 4 attention projections
# (query/key/value/output) and the 3 MLP projections (gate/up/down). These are
# the actual nn.Linear submodule names inside OLMo3's transformer blocks.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    # Every tunable knob for this script lives here, so nothing needs to be
    # hardcoded elsewhere. Run `python qlora_dpsgd.py --help` to see all of them.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", default="Abirate/english_quotes")
    parser.add_argument("--text-column", default="quote")
    parser.add_argument("--max-length", type=int, default=128)
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
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--output-dir", default="./qlora-dpsgd-adapter")
    return parser.parse_args()


def build_model_and_tokenizer(args):
    # NF4 (NormalFloat4) with double quantization is the exact quantization
    # scheme from the QLoRA paper: weights are stored in 4 bits, and are only
    # dequantized to bf16 on the fly for each matmul during forward/backward.
    #
    # This config only quantizes nn.Linear submodules (attention
    # q/k/v/o and MLP gate/up/down here) -- transformers replaces each of
    # those with a 4-bit Linear4bit layer at load time. Every other module
    # type (LayerNorm/RMSNorm, embeddings) is left as a regular nn.Parameter
    # in its original dtype; see the prepare_model_for_kbit_training call
    # below for why that matters.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

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


def build_data_loader(tokenizer, args):
    # Small public dataset used purely to exercise the training loop end to
    # end, not to produce a meaningful fine-tuned model.
    dataset = load_dataset(args.dataset, split="train")

    def tokenize(batch):
        out = tokenizer(
            batch[args.text_column],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        # Causal LM training predicts the next token at every position, so the
        # labels are just the input ids themselves (shifted internally by the
        # model's loss computation).
        out["labels"] = out["input_ids"].copy()
        return out

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # NOTE: this plain DataLoader gets replaced by Opacus with a Poisson-sampling
    # DataLoader inside train() -- the privacy accounting depends on that.
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    return data_loader, len(dataset)


def train(peft_model, data_loader, target_delta, args):
    # Only optimize the LoRA A/B matrices -- everything else is frozen.
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    privacy_engine = PrivacyEngine()
    # make_private_with_epsilon wraps (model, optimizer, data_loader) into
    # DP-SGD-aware versions:
    #   - dp_model: a GradSampleModule that hooks every trainable nn.Linear
    #     (here, only the LoRA layers) to compute per-sample gradients.
    #   - dp_optimizer: a DPOptimizer that clips each sample's gradient to
    #     max_grad_norm, sums them, adds Gaussian noise, then does a normal step.
    #   - dp_data_loader: a Poisson-sampling loader required for the privacy
    #     accountant's math to be valid.
    # The noise multiplier is solved for automatically so that, after `epochs`
    # passes over the data, the privacy spend equals target_epsilon.
    dp_model, dp_optimizer, dp_data_loader = privacy_engine.make_private_with_epsilon(
        module=peft_model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=args.target_epsilon,
        target_delta=target_delta,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
    )
    print(f"Using noise_multiplier={dp_optimizer.noise_multiplier:.4f} for target epsilon={args.target_epsilon}")

    device = torch.device("cuda")
    dp_model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(dp_data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            dp_optimizer.zero_grad()
            # Forward/backward here trigger Opacus's hooks: per-sample grads
            # are captured for the LoRA layers, then clipped + noised inside
            # dp_optimizer.step() below (not a plain AdamW step).
            loss = dp_model(**batch).loss
            loss.backward()
            dp_optimizer.step()

            if step % args.log_every == 0:
                # Privacy spend only grows with the number of steps taken, so
                # it can be queried at any point during training.
                eps = privacy_engine.get_epsilon(delta=target_delta)
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f} eps={eps:.2f}")

    final_eps = privacy_engine.get_epsilon(delta=target_delta)
    print(f"Final: epsilon={final_eps:.2f}, delta={target_delta:.2e}")
    return final_eps


def main():
    args = parse_args()
    peft_model, tokenizer = build_model_and_tokenizer(args)
    data_loader, dataset_size = build_data_loader(tokenizer, args)
    # Standard rule of thumb: delta should be much smaller than 1/dataset_size,
    # otherwise the privacy guarantee becomes vacuous (allows leaking a
    # non-negligible fraction of individual records with "certainty").
    target_delta = args.target_delta or 1.0 / (10 * dataset_size)

    # train() wraps peft_model in Opacus's GradSampleModule in place; peft_model
    # itself keeps updating and still exposes save_pretrained() afterwards.
    final_eps = train(peft_model, data_loader, target_delta, args)

    # peft_model.save_pretrained only writes the small LoRA adapter weights
    # (a few hundred MB), not the full frozen 7B base model.
    peft_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Record the actual privacy guarantee this adapter was trained under, so
    # it can be audited later without re-running training.
    privacy_report = {
        "target_epsilon": args.target_epsilon,
        "achieved_epsilon": final_eps,
        "delta": target_delta,
        "max_grad_norm": args.max_grad_norm,
        "epochs": args.epochs,
    }
    with open(os.path.join(args.output_dir, "privacy_report.json"), "w") as f:
        json.dump(privacy_report, f, indent=2)

    print(f"Done. LoRA adapter + privacy_report.json saved to {args.output_dir}")


if __name__ == "__main__":
    main()
