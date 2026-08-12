# ppfl_project

QLoRA fine-tuning of a 7B LLM on MIMIC-IV-Note clinical notes, comparing
Opacus DP-SGD (differentially private) against plain AdamW (non-private) to
measure the utility cost of privacy on real PHI-adjacent text.

See [RESULTS.md](RESULTS.md) for the actual experiment writeup and numbers.

## Files

| File | Purpose |
|---|---|
| `common.py` | Shared constants/helpers: model id, dataset path, adapter dir default, 4-bit quantization config, dataset split, tokenized `DataLoader` construction |
| `train.py` | `train_dp_sgd()` and `train_adamw()` — the two training loops |
| `qlora_finetune.py` | CLI entry point: builds the model/data, dispatches to one of the two training loops, saves the LoRA adapter + `privacy_report.json` |
| `evaluate.py` | CLI entry point: loads a trained adapter (and optionally a second one, and/or the un-adapted base model) and reports perplexity on the held-out split |

## Requirements

- CUDA GPU (developed/tested on a 24GB RTX 4090)
- `torch`, `transformers`, `peft`, `opacus`, `bitsandbytes`, `datasets`

## Dataset

Expects a local MIMIC-IV-Note discharge-summary CSV (gzip ok) with a `text`
column, one clinical note per row. Default path is set in `common.py`
(`DEFAULT_DATASET_PATH`); override with `--dataset` if yours lives elsewhere.
Access to MIMIC-IV-Note requires a completed PhysioNet credentialing process.

## Usage

### 1. Train

Exactly one of `--dp-sgd` / `--adamw` is required — there is no implicit
default, so it's always explicit which mode a given adapter was trained with.

```bash
# Differentially private (Opacus DP-SGD)
python qlora_finetune.py --dp-sgd \
  --target-epsilon 8.0 \
  --output-dir ./qlora-adapter-dpsgd

# Non-private baseline (plain AdamW)
python qlora_finetune.py --adamw \
  --output-dir ./qlora-adapter-adamw
```

Each run saves the LoRA adapter (`adapter_config.json` +
`adapter_model.safetensors`), the tokenizer, and a `privacy_report.json`
(achieved epsilon/delta for `--dp-sgd`, or `dp_enabled: false` for `--adamw`)
into `--output-dir`.

For a quick end-to-end smoke test before committing to a full run over the
dataset, add `--max-steps 3`.

### 2. Evaluate

```bash
python evaluate.py \
  --adapter-dir ./qlora-adapter-dpsgd \
  --compare-adapter-dir ./qlora-adapter-adamw \
  --compare-base
```

Loads the base model once, then attaches/detaches/switches LoRA adapters on
that same instance to measure perplexity on the identical held-out split for
up to three conditions: the primary adapter, the un-adapted base model
(`--compare-base`), and a second adapter (`--compare-adapter-dir`) — e.g. a
DP-SGD run vs. an AdamW run. Results (plus each adapter's privacy report) are
printed and saved to `<--adapter-dir>/eval_report.json`.

## Key flags

| Flag | Script | Meaning |
|---|---|---|
| `--dp-sgd` / `--adamw` | `qlora_finetune.py` | Training mode (mutually exclusive, one required) |
| `--logical-batch-size-dp-sgd` | `qlora_finetune.py` | DP-SGD only: sets Opacus's Poisson sample rate (the privacy-accounting batch) |
| `--batch-size` | `qlora_finetune.py` | The actual per-step GPU batch in both modes — for `--adamw` it's the training batch directly; for `--dp-sgd`, Opacus's `BatchMemoryManager` uses it to cap memory without changing the privacy accounting |
| `--target-epsilon` / `--target-delta` | `qlora_finetune.py` | DP-SGD privacy budget (delta defaults to `1/(10*len(dataset))`) |
| `--eval-fraction` / `--seed` | both | Must match between training and evaluation runs so the held-out split is reconstructed identically and stays genuinely unseen |
| `--max-steps` | `qlora_finetune.py` | Cap optimizer steps, for smoke-testing the pipeline |
| `--compare-base` / `--compare-adapter-dir` | `evaluate.py` | Additional conditions to evaluate in the same pass |

Run `python qlora_finetune.py --help` or `python evaluate.py --help` for the
full list.
