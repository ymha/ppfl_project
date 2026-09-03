import math
import os
import random

import torch
from datasets import load_dataset
from opacus.validators import ModuleValidator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)

# Base model to fine-tune. This is the pretrained (non-instruct) OLMo 3 7B checkpoint.
MODEL_ID = "allenai/Olmo-3-1025-7B"

# Linear layers LoRA adapters get attached to: the 4 attention projections
# (query/key/value/output) and the 3 MLP projections (gate/up/down). These are
# the actual nn.Linear submodule names inside OLMo3's transformer blocks.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# MIMIC-IV-Note discharge summaries. Local gzipped CSV (note_id, subject_id,
# hadm_id, note_type, note_seq, charttime, storetime, text) -- one free-text
# clinical note per row. Real PHI-adjacent data (MIMIC only de-identifies
# direct identifiers), which is exactly the kind of dataset DP-SGD's
# per-sample clipping + noise is meant to protect.
DEFAULT_DATASET_PATH = os.path.expanduser(
    "~/sources/datasets/physionet.org/files/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
)

# Default location for a trained LoRA adapter -- shared so qlora_finetune.py's
# --output-dir and evaluate.py's --adapter-dir agree without being told.
DEFAULT_ADAPTER_DIR = "./qlora-adapter"


def add_data_args(parser):
    # Shared between qlora_finetune.py and evaluate.py: both need to load and
    # tokenize the same dataset the same way to reconstruct matching splits.
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-length", type=int, default=128)
    # The 10% held out here is never seen during training, so evaluate.py can
    # measure perplexity on genuinely unseen notes. --seed must match between
    # the two scripts for both to reconstruct the same held-out rows.
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)


def build_bnb_config():
    # NF4 (NormalFloat4) with double quantization is the exact quantization
    # scheme from the QLoRA paper: weights are stored in 4 bits, and are only
    # dequantized to bf16 on the fly for each matmul during forward/backward.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def build_model_and_tokenizer(args):
    # Shared by centralized/qlora_finetune.py and federated/ (server_app.py's
    # server-side model, and runtime.BaseModelCache's per-process cached
    # client model) -- both need the identical 4-bit QLoRA + LoRA-wrapped
    # model, just with different LoRA weights loaded in afterwards.
    #
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


def load_split(dataset_path, split, eval_fraction, seed):
    # dataset_path is the local MIMIC-IV-Note CSV path. Both qlora_finetune.py
    # and evaluate.py must call this with the same eval_fraction/seed so they
    # reconstruct the exact same train/test partition -- that's what makes
    # evaluate.py's "test" split genuinely unseen by training's "train" split.
    dataset = load_dataset("csv", data_files=dataset_path, split="train")
    if split == "train":
        if eval_fraction > 0:
            dataset = dataset.train_test_split(test_size=eval_fraction, seed=seed)["train"]
        return dataset
    return dataset.train_test_split(test_size=eval_fraction, seed=seed)["test"]


def select_one_note_per_subject(dataset, seed):
    """Keep exactly one (seeded-random) note per subject_id.

    Standard "one record per user" trick for turning a record-level DP-SGD
    guarantee into a genuine per-patient (user-level) one: if no subject
    contributes more than one training example, a neighboring dataset
    differing in one record is exactly a neighboring dataset differing in one
    subject's entire contribution, so the record-level (epsilon, delta)
    already computed applies at the subject level too, with no change to the
    training loop or privacy accounting itself -- only which rows are trained
    on. Shared with federated/runtime.py, which applies the same selection
    per client shard (see pyproject.toml's one-note-per-subject).
    """
    indices_by_subject = {}
    for i, subject_id in enumerate(dataset["subject_id"]):
        indices_by_subject.setdefault(subject_id, []).append(i)

    rng = random.Random(seed)
    chosen_indices = sorted(rng.choice(indices) for indices in indices_by_subject.values())
    return dataset.select(chosen_indices)


def build_tokenized_loader(dataset, tokenizer, text_column, max_length, batch_size, shuffle):
    def tokenize(batch):
        out = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def build_eval_loader(tokenizer, args):
    # Shared by centralized/evaluate.py and federated/server_app.py's
    # centralized held-out evaluation -- both read the "test" side of the
    # exact same split load_split() reconstructs from --eval-fraction/--seed.
    dataset = load_split(args.dataset, "test", args.eval_fraction, args.seed)
    return build_tokenized_loader(
        dataset, tokenizer, args.text_column, args.max_length, args.batch_size, shuffle=False
    )


@torch.no_grad()
def perplexity(model, data_loader, device, max_batches=None):
    # Shared by centralized/evaluate.py and federated/server_app.py's
    # per-round centralized evaluation.
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
