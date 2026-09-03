# ppfl_project

QLoRA fine-tuning of a 7B LLM on MIMIC-IV-Note clinical notes, comparing
Opacus DP-SGD (differentially private) against plain AdamW (non-private) to
measure the utility cost of privacy on real PHI-adjacent text. The
single-machine ("centralized") implementation lives in `centralized/` (see
below). Also includes a federated learning variant (`federated/`, see
below): multiple simulated clients each hold their own shard of notes,
fine-tune locally with DP-SGD, and their updates are combined via Flower's
Secure Aggregation (SecAgg+) so the server never sees any individual
client's update in the clear.

See [RESULTS.md](RESULTS.md) for the actual experiment writeup and numbers.

## Shared code

`common.py` and `train.py` live at the repo root, above both `centralized/`
and `federated/`, because both packages import them directly (not through
each other) for the parts of the pipeline that are identical either way:
model/tokenizer construction, dataset loading and splitting, the DP-SGD and
AdamW training loops, and held-out perplexity evaluation.

| File | Purpose |
|---|---|
| `common.py` | Constants (model id, dataset path, adapter dir default), 4-bit quantization + LoRA model construction (`build_model_and_tokenizer`), dataset split, one-note-per-subject selection, tokenized `DataLoader` construction, held-out eval loader, and `perplexity()` |
| `train.py` | `train_dp_sgd()` and `train_adamw()` — the two training loops |

## Centralized (single-machine)

`centralized/` holds the single-machine CLI entry points, built on top of
the shared code above.

| File | Purpose |
|---|---|
| `centralized/qlora_finetune.py` | CLI entry point: builds the model/data, dispatches to one of the two training loops, saves the LoRA adapter + `privacy_report.json` |
| `centralized/evaluate.py` | CLI entry point: loads a trained adapter (and optionally a second one, and/or the un-adapted base model) and reports perplexity on the held-out split |

## Requirements

- CUDA GPU (developed/tested on a 24GB RTX 4090)
- `torch`, `transformers`, `peft`, `opacus`, `bitsandbytes`, `datasets`
- `flwr[simulation]` for the federated variant only (pinned in
  `pyproject.toml`)

## Dataset

Expects a local MIMIC-IV-Note discharge-summary CSV (gzip ok) with a `text`
column, one clinical note per row. Default path is set in `common.py`
(`DEFAULT_DATASET_PATH`); override with `--dataset` if yours lives
elsewhere. Access to MIMIC-IV-Note requires a completed PhysioNet
credentialing process.

## Usage

### 1. Train

Exactly one of `--dp-sgd` / `--adamw` is required — there is no implicit
default, so it's always explicit which mode a given adapter was trained with.

```bash
# Differentially private (Opacus DP-SGD)
python -m centralized.qlora_finetune --dp-sgd \
  --target-epsilon 8.0 \
  --output-dir ./qlora-adapter-dpsgd

# Non-private baseline (plain AdamW)
python -m centralized.qlora_finetune --adamw \
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
python -m centralized.evaluate \
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
| `--dp-sgd` / `--adamw` | `centralized/qlora_finetune.py` | Training mode (mutually exclusive, one required) |
| `--logical-batch-size-dp-sgd` | `centralized/qlora_finetune.py` | DP-SGD only: sets Opacus's Poisson sample rate (the privacy-accounting batch) |
| `--batch-size` | `centralized/qlora_finetune.py` | The actual per-step GPU batch in both modes — for `--adamw` it's the training batch directly; for `--dp-sgd`, Opacus's `BatchMemoryManager` uses it to cap memory without changing the privacy accounting |
| `--target-epsilon` / `--target-delta` | `centralized/qlora_finetune.py` | DP-SGD privacy budget (delta defaults to `1/(10*len(dataset))`) |
| `--one-note-per-subject` | `centralized/qlora_finetune.py` | Off by default. Collapses the training set to one (seeded-random) note per `subject_id` before training, turning the reported (epsilon, delta) into a genuine per-patient (user-level) guarantee instead of a per-note one — since no subject then contributes more than one training example, a neighboring-dataset record is exactly a neighboring-dataset patient. Recorded as `guarantee_level` (`"user"` vs `"record"`) in `privacy_report.json`. Matches `federated/`'s default (`pyproject.toml`'s `one-note-per-subject`) for a single-machine equivalent experiment |
| `--eval-fraction` / `--seed` | both | Must match between training and evaluation runs so the held-out split is reconstructed identically and stays genuinely unseen |
| `--max-steps` | `centralized/qlora_finetune.py` | Cap optimizer steps, for smoke-testing the pipeline |
| `--compare-base` / `--compare-adapter-dir` | `centralized/evaluate.py` | Additional conditions to evaluate in the same pass |

Run `python -m centralized.qlora_finetune --help` or
`python -m centralized.evaluate --help` for the full list.

## Federated Learning (Flower + SecAgg+)

`federated/` extends the same shared QLoRA + DP-SGD training loop
(`train.py`/`common.py`, see above) into a simulated federated setting: N
clients each hold a disjoint shard of MIMIC-IV-Note (split by `subject_id`,
so no patient's notes cross a client boundary), fine-tune locally with the
same `train_dp_sgd()` used above, and their LoRA updates are combined
server-side via FedAvg wrapped in Flower's SecAgg+ protocol — the server
only ever recovers the aggregate, never an individual client's update. Runs
as a single-process Flower Simulation on one GPU (clients activate
sequentially, not on separate machines).

| File | Purpose |
|---|---|
| `federated/config.py` | Loads `pyproject.toml`'s `[tool.flwr.app.config]` table directly (see its comment for why, instead of Flower's own `Context.run_config`) |
| `federated/partition_data.py` | Offline CLI: splits the training set into N client shards by `subject_id`, writes `federated/partitions/manifest.json` |
| `federated/runtime.py` | LoRA weight (de)serialization, mapping `pyproject.toml`'s run config onto the CLI's argument shape (`build_fake_args`), per-client shard loading, the process-wide cached base model, cross-round DP accountant history I/O, and the `train_lock()` used to serialize client training on the shared model |
| `federated/client_app.py` | Flower `ClientApp` — each client's `fit()` loads the global LoRA weights, trains locally via `train_dp_sgd()`, returns the updated weights |
| `federated/server_app.py` | Flower `ServerApp` — wires FedAvg through Flower's `SecAggPlusWorkflow`, runs centralized held-out evaluation each round, saves the final aggregated adapter |
| `federated/privacy_report.py` | Aggregates each client's per-round privacy report into a system-level report (worst-case cumulative epsilon across clients) |
| `federated/simulation.py` | Entry point: launches the Flower simulation with both apps above |

All federated run parameters (number of clients/rounds, local batch/step
caps, SecAgg+ parameters, etc.) live in `pyproject.toml`'s
`[tool.flwr.app.config]` table — see the comments there for what each one
means and its safe range.

`num-clients` must be at least 2: SecAgg+'s pairwise masking has no
counterpart to cancel against with a single client, and Flower's own
`SecAggPlusWorkflow` will otherwise abort the fit step every round with no
exception raised — `server_app.py` and `partition_data.py` both assert this
explicitly (and fail before the 7B model load) so a bad config errors out
immediately instead of silently producing an untrained adapter.

### 1. Partition the dataset

```bash
python -m federated.partition_data
```

`--num-clients` is not a CLI flag here — it's always read from
`pyproject.toml`'s `num-clients` (see the comment on `main()`), so the
manifest this writes can never disagree with what `server_app.py` expects.

### 2. Run the federated simulation

```bash
python -m federated.simulation --num-cpus 4 --num-gpus 1.0
```

Saves the aggregated adapter + `privacy_report.json` to
`federated/global-adapter/`, in the same shape as
`centralized/qlora_finetune.py`'s output — so it can be evaluated and
compared exactly like the single-machine adapters:

```bash
python -m centralized.evaluate \
  --adapter-dir federated/global-adapter \
  --compare-adapter-dir qlora-dpsgd-adapter-eps8-full \
  --compare-base
```

See [RESULTS.md](RESULTS.md) for the federated run's actual numbers and a
discussion of what SecAgg+ and DP-SGD each protect against (two distinct,
non-composable guarantees).
