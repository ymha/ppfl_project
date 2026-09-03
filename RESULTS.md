# DP-SGD vs. Non-Private QLoRA Fine-Tuning on MIMIC-IV-Note

## Setup

- **Base model:** `allenai/Olmo-3-1025-7B` (pretrained, non-instruct), loaded in 4-bit NF4 (QLoRA)
- **Adapter:** LoRA, r=16, alpha=32, dropout=0.05, targeting `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` (~40M trainable params, 0.54% of the 7.34B total)
- **Dataset:** MIMIC-IV-Note discharge summaries (`discharge.csv.gz`), 331,793 notes total
  - 10% (33,180 notes) held out and never seen during training, used only for evaluation
  - Remaining 298,613 notes used for training
- **Training:** 1 epoch, batch size 8 (physical, GPU-resident) via Opacus `BatchMemoryManager` for the DP-SGD run
- **Hardware:** single RTX 4090 (24GB) for all four conditions, including the federated one — Flower's simulation engine runs all 5 clients' training sequentially on the one GPU (`federated/simulation.py`), not on separate machines
- **Privacy accounting:** Opacus `PrivacyEngine()` is instantiated with no explicit `accountant=` argument, so all reported epsilons (single-machine and federated) come from Opacus 1.6.0's default **PRV (Privacy loss Random Variable) accountant** — numerical composition via FFT convolution (Gopi et al., 2021), not the classical RDP (Rényi-DP) accountant. PRV is generally tighter than RDP for the same noise multiplier/step count. The cross-round history splicing in `train.py`/`federated/runtime.py` (`accountant.history`, a list of `(noise_multiplier, sample_rate, num_steps)` tuples) works identically regardless of accountant choice, since both implement the same `IAccountant` interface — only the epsilon math itself is PRV-based here.

Four conditions were evaluated:

| Condition | Description |
|---|---|
| **Base** | Frozen pretrained model, no fine-tuning |
| **DP-SGD** | LoRA fine-tuned with Opacus DP-SGD, target epsilon=8.0, logical batch size 256, max grad norm 1.0 |
| **Plain QLoRA** | Same LoRA fine-tune, no privacy mechanism (`--adamw`) — standard AdamW, no per-sample clipping or noise |
| **Federated DP-SGD + SecAgg+** | 5 simulated clients, each with its own IID shard (~60k notes, split by `subject_id` so no patient's notes cross client boundaries), each locally fine-tuning with Opacus DP-SGD (target ε=8.0/round, logical batch 256, physical batch 8, capped at 1000 physical steps/round ≈ 13% of one local epoch) for 3 FedAvg rounds. Client updates are never seen individually by the aggregator: Flower's SecAgg+ protocol (`num_shares`=100% of clients, `reconstruction_threshold`=60% of shares) masks and secret-shares each client's LoRA delta before summation, so the server only ever recovers the aggregate. See `federated/` for the implementation (`server_app.py`, `client_app.py`, `partition_data.py`, `simulation.py`) |

## Results

Perplexity measured on the full 33,180-note held-out set (`exp(mean cross-entropy loss)`, lower is better):

| Model                              | Loss  | Perplexity | Privacy guarantee                                       |
|-------------------------------------|-------|------------|-----------------------------------------------------------|
| Base (no fine-tuning)               | 2.964 | **19.37**  | none                                                       |
| DP-SGD fine-tuned                   | 0.946 | **2.58**   | ε ≈ 7.99, δ ≈ 3.35×10⁻⁷                                    |
| Plain QLoRA fine-tuned              | 0.714 | **2.04**   | none                                                       |
| Federated DP-SGD + SecAgg+ (5×3)    | 1.174 | **3.23**   | worst-case ε ≈ 6.76 across clients (δ ≈ 1.68×10⁻⁶); individual client updates never seen by the aggregator (SecAgg+) |

Federated perplexity by round (server-side centralized eval on the same held-out split, aggregated LoRA weights after each round): round 0 (initial, unfine-tuned) 19.44 → round 1 6.55 → round 2 3.64 → round 3 (final) 3.25, matching the 3.23 uncapped final number above.

## Discussion

- **Fine-tuning helps enormously.** All three fine-tuned models cut perplexity substantially relative to the base model, confirming the LoRA adapters learn MIMIC's highly templated discharge-summary structure and vocabulary effectively.
- **Cost of privacy.** At ε≈8, DP-SGD's clip-and-noise mechanism costs about a 26% increase in perplexity relative to the non-private run (2.58 vs. 2.04). This is the concrete utility price paid for the (ε≈7.99, δ≈3.35×10⁻⁷)-DP guarantee against per-sample gradient leakage.
- **Practical takeaway.** Even under DP-SGD, the fine-tuned model remains far closer to the plain fine-tune than to the untouched base model — i.e., privacy-preserving fine-tuning on clinical notes at this scale is practically usable, not just theoretically possible.
- **Federated result is not apples-to-apples with the single-machine DP-SGD run, by design.** The single-machine run sees the full 298,613-row training set once (one full epoch). Each federated client, by contrast, is capped at 1000 physical steps/round (of the ~7,449 needed for one full local epoch over its own ~60k-row shard) for three rounds — roughly 40% of one epoch's worth of gradient steps per client, cumulatively, and only over that client's own shard rather than the full dataset. The federated adapter's higher perplexity (3.23 vs. 2.58) mostly reflects this reduced training budget, not an inherent cost of federation or SecAgg+ — the round-by-round trend (19.44 → 6.55 → 3.64 → 3.25) shows the federated model was still improving steadily when the run ended, not converged.
- **Two distinct privacy guarantees, not one.** SecAgg+ and per-client DP-SGD protect against different things and their guarantees don't compose into a single epsilon. SecAgg+ bounds what an honest-but-curious aggregator can see about any individual client's update (a confidentiality guarantee against the *server*, independent of epsilon). The reported ε≈6.76 is the composed, cumulative per-client DP-SGD guarantee (via Opacus's PRV accountant, correctly composed across a client's 3 rounds using the accountant-history plumbing in `train.py`) — a record-level guarantee each client's own data enjoyed *before* secret-sharing, that holds regardless of what the aggregator does. Reported as the worst case (max) across the 5 clients, whose individual cumulative epsilons were tightly clustered (6.75–6.76), since all clients trained on similarly-sized shards under an identical per-round schedule.
- **Stopping early costs less privacy budget, honestly.** Each client's per-round target was ε=8.0, but capping training at 1000 (of ~7,449 needed) steps/round means each round spent less budget than a full round would have — hence 6.76 cumulative over 3 rounds, comfortably under 3×8=24. A full-epoch-per-round federated run would spend closer to the full per-round target each time.
- **`target-epsilon-per-round` is a per-round target, not a total privacy budget — there is no enforced cap on the cumulative epsilon.** Each round's `make_private_with_epsilon(target_epsilon=8.0, epochs=1, ...)` call (`runtime.build_fake_args()` → `train_dp_sgd()`) is calibrated independently and has no knowledge of how many rounds already ran or how many are still to come. The *cumulative* epsilon is computed — via the accountant-history splicing in `train.py`/`runtime.py`, aggregated across a client's rounds in `privacy_report.py` — but only as a post-hoc measurement of whatever composition falls out of those independent per-round calibrations, not as a quantity the noise level is solved to keep under some fixed ceiling. The 6.76 figure landing under the per-round target of 8.0 is an artifact of this run's aggressive `max-steps=1000` cap (each round spends well under its own 8.0 target before reaching it, per the point above), not a guarantee. Increasing `num-server-rounds`, or removing the `max-steps` cap so each round trains to completion, would each independently drive the cumulative epsilon higher with nothing in the code to stop it — including past 8.0. A methodologically stricter design for a fixed total-privacy-budget federated run would instead decide the total round count up front and calibrate against it directly (e.g., one `make_private_with_epsilon(target_epsilon=8.0, epochs=num_rounds, ...)`-equivalent solve, or an explicit composition-aware split of the total budget across rounds) so the composed total is bounded by construction — this codebase does not do that.

## Artifacts

- `qlora-dpsgd-adapter-eps8-full/` — DP-SGD LoRA adapter, `privacy_report.json`, `eval_report.json`
- `qlora-plain-adapter-full/` — non-private LoRA adapter, `privacy_report.json`
- `federated/global-adapter/` — federated DP-SGD + SecAgg+ aggregated LoRA adapter, `privacy_report.json` (system + per-client epsilons), `eval_report.json`
- `federated/client_state/` — per-client accountant history and privacy reports (one client's own record-level DP bookkeeping, independent of the aggregator)
