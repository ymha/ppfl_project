# DP-SGD vs. Non-Private QLoRA Fine-Tuning on MIMIC-IV-Note

## Setup

- **Base model:** `allenai/Olmo-3-1025-7B` (pretrained, non-instruct), loaded in 4-bit NF4 (QLoRA)
- **Adapter:** LoRA, r=16, alpha=32, dropout=0.05, targeting `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` (~40M trainable params, 0.54% of the 7.34B total)
- **Dataset:** MIMIC-IV-Note discharge summaries (`discharge.csv.gz`), 331,793 notes total
  - 10% (33,180 notes) held out and never seen during training, used only for evaluation
  - Remaining 298,613 notes used for training
- **Training:** 1 epoch, batch size 8 (physical, GPU-resident) via Opacus `BatchMemoryManager` for the DP-SGD run
- **Hardware:** single RTX 4090 (24GB)

Three conditions were evaluated:

| Condition | Description |
|---|---|
| **Base** | Frozen pretrained model, no fine-tuning |
| **DP-SGD** | LoRA fine-tuned with Opacus DP-SGD, target epsilon=8.0, logical batch size 256, max grad norm 1.0 |
| **Plain QLoRA** | Same LoRA fine-tune, no privacy mechanism (`--adamw`) — standard AdamW, no per-sample clipping or noise |

## Results

Perplexity measured on the full 33,180-note held-out set (`exp(mean cross-entropy loss)`, lower is better):

| Model | Loss | Perplexity | Privacy guarantee |
|---|---|---|---|
| Base (no fine-tuning) | 2.964 | **19.37** | — |
| DP-SGD fine-tuned | 0.946 | **2.58** | ε ≈ 7.99, δ ≈ 3.35×10⁻⁷ |
| Plain QLoRA fine-tuned | 0.714 | **2.04** | none |

## Discussion

- **Fine-tuning helps enormously.** Both fine-tuned models cut perplexity by roughly 7–9x relative to the base model, confirming the LoRA adapters learn MIMIC's highly templated discharge-summary structure and vocabulary effectively.
- **Cost of privacy.** At ε≈8, DP-SGD's clip-and-noise mechanism costs about a 26% increase in perplexity relative to the non-private run (2.58 vs. 2.04). This is the concrete utility price paid for the (ε≈7.99, δ≈3.35×10⁻⁷)-DP guarantee against per-sample gradient leakage.
- **Practical takeaway.** Even under DP-SGD, the fine-tuned model remains far closer to the plain fine-tune than to the untouched base model — i.e., privacy-preserving fine-tuning on clinical notes at this scale is practically usable, not just theoretically possible.

## Artifacts

- `qlora-dpsgd-adapter-eps8-full/` — DP-SGD LoRA adapter, `privacy_report.json`, `eval_report.json`
- `qlora-plain-adapter-full/` — non-private LoRA adapter, `privacy_report.json`
