import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling

# Base model to fine-tune. This is the pretrained (non-instruct) OLMo 3 7B checkpoint.
MODEL_ID = "allenai/Olmo-3-1025-7B"

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
