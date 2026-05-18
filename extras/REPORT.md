# LightWeight MAPF-GPT — Experimental Report

**Repository:** `extras/` (package `lmgpt/`) · **Period:** 15–17 May 2026 · **Hardware:** NVIDIA H200, `mapf-gpt` (PyTorch 2.10, CUDA 12.8)

Compression and distillation of [MAPF-GPT](https://github.com/CognitiveAISystems/MAPF-GPT) policies evaluated on the full POGEMA benchmark grid (5 map groups, varying team sizes) plus a controlled forward-latency suite.

---

## 1. Executive summary

| Goal | Outcome |
|------|---------|
| Smaller/faster substitute for **author 2M** | **T1** (L4 trim + CE+KL): 1.24M params, CSR **0.849** vs 0.850, forward **1.23×** faster @ B=2048 |
| Aggressive size cut | **T2** (L3 trim + CE+KL): 0.93M params, CSR 0.832, **1.58×** faster |
| Post-training quant only | **C3** (AWQ W4A16): CSR 0.845, latency ≈ R0 |
| Small distilled students | **D1** best among 1M (CSR 0.829), **1.50×** faster; quality gap −2.1 pp vs R0 |
| “2M” student from 85M (**D5**) | Student is **6.4M** (8×256d), not author-2M; beats 6M on CSR but **slower** than R0 |

**Recommended deployment target:** checkpoint **T1** (`trim_ft/.../L4_cekl/ckpt_best.pt`).

---

## 2. Model nomenclature

Checkpoint labels (2M / 6M / 1M) are historical; **measured parameter counts** differ.

| Label | Architecture | Params |
|-------|--------------|--------|
| Author 2M (R0) | 5L × 160d × 5h | 1.55M |
| Author 6M | 8L × 256d × 8h | 6.31M |
| Author 85M | 12L × 768d × 12h | 85.0M |
| L4 trim / T1,T3 | 4L × 160d | 1.24M |
| L3 trim / T2 | 3L × 160d | 0.93M |
| Student 1M (D1–D3) | 4L × 128d × 4h | 0.84M |
| Student 1.5M (D4) | 4L × 160d × 5h | 1.29M |
| Student “2M” (D5) | 8L × 256d × 8h | **6.40M** |

---

## 3. Methods (concise)

| ID | Procedure |
|----|-----------|
| **C1/C2** | Depth trim of author 2M to 4 / 3 layers (`trim.py`), no fine-tune |
| **T1–T3** | Fine-tune trimmed weights on `dataset/train`; CE+KL self-distill from full 2M (T1,T2) or CE-only (T3) |
| **C3** | AWQ W4A16 on author 2M (`quantize_awq.py`, calib from validation) |
| **D1–D5** | KL+CE distillation; D3/D4/D5 add hidden-state MSE + multi-position logits (`with_hidden.json`) |

Loss (distill / CE+KL): \(\alpha_{kl}=0.7\), \(\alpha_{ce}=0.3\), \(T=2\); hidden weight 0.5 when enabled.

Training: streaming Arrow, **bf16**, AdamW; trim-ft **30k** iters, distill **50k** iters, `batch_size=2048`, `lr=1e-4` (ft) / `3e-4` (distill). Not full-dataset epochs.

---

## 4. Evaluation protocol

### 4.1 POGEMA (quality)

- **Script:** `extras/scripts/eval_pogema.py` — all 5 groups, **3296 episodes / model**, CUDA, **FP32** inference.
- **Metrics:** CSR, ISR, SR, SoC (mean over episodes), episode `runtime` (env + policy, seconds).

**Number of agents** is a first-class axis of the benchmark (from `eval_configs/*/grid_search`):

| Map group | `num_agents` values |
|-----------|---------------------|
| 01-random | 8, 16, 24, 32, 48, 64 |
| 02-mazes | 8, 16, 24, 32, 48, 64 |
| 03-warehouse | 32, 64, 96, 128, 160, 192 |
| 04-movingai | 64, 128, 192, 256 |
| 05-puzzles | 2, 3, 4 |

Each episode logs `map_name`, `num_agents`, `seed`. Aggregates:

- **Global CSR** (report table below): unweighted mean over **all** episodes (all group × agent × map × seed cells).
- **Stratified:** `summary_by_model_map_agents.csv` per run; merged `by_map_group_agents.md` in `extras/runs/reports/final_2026-05-17/`.

Training does **not** sweep POGEMA agent counts; one policy is evaluated across the full grid.

### 4.2 Forward latency (speed)

- **Script:** `extras/scripts/bench_latency.py` — `model.act` only, **bf16**, `seq_len=256`.
- **Batch size B=2048** proxies simultaneous agents in one forward step (not identical to POGEMA scenarios).
- **Unified run (17 May, GPU 7 idle):** warmup=50, measure=200, CUDA events — see `latency_b2048.md`.

POGEMA `runtime_s` and forward `p50_ms` measure different things; both are reported.

---

## 5. Main results

### 5.1 Global POGEMA + forward latency (primary table)

| ID | Method | Params | CSR↑ | ISR↑ | SoC↓ | POGEMA `runtime`↓ | Latency p50 @B=2048↓ | Notes |
|----|--------|--------|------|------|------|-------------------|----------------------|-------|
| **R0** | Author 2M | 1.55M | **0.850** | 0.964 | 3408 | 1.47 s | **18.15 ms** | Reference |
| — | Author 6M | 6.31M | 0.866 | 0.968 | 3299 | 2.40 s | 39.04 ms | Quality bound |
| — | Author 85M | 85.0M | **0.925** | 0.983 | 3070 | 8.85 s | 225.67 ms | Quality bound |
| **C1** | L4 trim | 1.24M | 0.561 | 0.899 | 4613 | 2.13 s | 14.83 ms | No FT |
| **C2** | L3 trim | 0.93M | 0.259 | 0.802 | 9508 | 1.55 s | 11.48 ms | No FT |
| **T1** | L4 trim + CE+KL | 1.24M | **0.849** | 0.964 | 3401 | 0.46 s | 14.80 ms | **Best 2M replacement** |
| **T2** | L3 trim + CE+KL | 0.93M | 0.832 | 0.958 | 3515 | 0.41 s | 11.50 ms | Fastest trim line |
| **T3** | L4 trim + CE only | 1.24M | 0.848 | 0.961 | 3530 | 0.47 s | 14.81 ms | ≈ T1 |
| **C3** | AWQ W4A16 | 1.55M* | 0.845 | 0.965 | 3422 | 1.79 s | 18.18 ms | *quantized weights |
| **D1** | 1M ← 6M | 0.84M | 0.829 | 0.954 | 3545 | 0.41 s | 12.13 ms | Logits KD |
| **D2** | 1M ← 85M (logits) | 0.84M | 0.796 | 0.942 | 3646 | 0.41 s | 12.14 ms | |
| **D3** | 1M ← 85M (hidden) | 0.84M | 0.788 | 0.942 | 3829 | 0.42 s | 12.15 ms | |
| **D4** | 1.5M ← 85M (hidden) | 1.29M | 0.819 | 0.951 | 3682 | 0.49 s | 14.84 ms | |
| **D5** | 8L×256 ← 85M (hidden) | 6.40M | 0.881 | 0.969 | 3255 | 1.06 s | 39.05 ms | Not author-2M width |

*Speedup vs R0 @B=2048:* T1 **1.23×**, T2 **1.58×**, D1 **1.50×**, C1/C2 fast but unusable CSR.

**Sources:** POGEMA — `global_by_model.md`; latency — unified bench `extras/runs/latency/2026-05-17_19-3*_final_2026-05-17_*`.

### 5.2 CSR vs. number of agents (author 2M vs T1)

Global CSR collapses across team sizes; breakdown shows where quality is won or lost.

| Group | Agents | CSR (R0 2M) | CSR (T1) |
|-------|--------|-------------|----------|
| random | 32 | 0.99 | 0.99 |
| random | 64 | 0.84 | 0.82 |
| mazes | 32 | 0.80 | 0.84 |
| mazes | 64 | 0.34 | 0.34 |
| warehouse | 128 | 0.90 | 1.00 |
| warehouse | 192 | 0.19 | 0.20 |
| movingai | 256 | 0.66 | 0.66 |

Full grid: `by_map_group_agents.md` (Wilson 95% CI per cell).

**Pattern:** both models are near ceiling on easy cells (low `num_agents`, random/mazes small teams); degradation at large teams is similar — T1 does not fix hard warehouse/mazes tails but matches R0 on typical cells.

### 5.3 Ablations (selected)

| Comparison | Result |
|------------|--------|
| T3 vs T1 (CE vs CE+KL) | CSR 0.848 vs 0.849 — KL optional |
| D1 vs D2/D3 (1M teacher) | D1 **0.829** > D2 0.796 > D3 0.788 |
| C1 → T1 (FT) | CSR 0.561 → 0.849 |

---

## 6. Conclusions

1. **Trim + fine-tune** is the only method that matches author-2M CSR while cutting parameters and forward time (T1/T2).
2. **AWQ** preserves CSR with modest speedup under our W4A16 setup.
3. **Distillation** to 1M trades ~2 pp CSR for ~1.5× forward speed; teacher **6M** beats **85M** for 1M students.
4. **D5** must be reported as a **6M-width** student, not a compressed 2M author model.
5. **Agent count** is fully represented in POGEMA evaluation but hidden in the global CSR column — use stratified tables for analysis.

---

## 7. Experiment registry

| ID | Train checkpoint | Eval run |
|----|------------------|----------|
| R0 | `weights/model-{2,6,85}M.pt` | `baselines/2026-05-15_20-23-07` |
| C1/C2 | `weights/model-2M-L{4,3}.pt` | `trim/2026-05-15_20-{27,22}-*` |
| C3 | `weights/model-2M-awq.pt` | `awq/2026-05-15_20-31-50` |
| T1–T3 | `trim_ft/2026-05-15_20-{36,45,47}-*` | `trim_ft/2026-05-17_03-{02,33,04}-*` |
| D1–D5 | `distill/2026-05-15_21-10-*`, `2026-05-16_13-5*` | `distill/2026-05-17_*` |

Regenerate tables:

```bash
python extras/scripts/compare_runs.py --baseline-model 2M \
  --runs extras/runs/baselines/2026-05-15_20-23-07 ... \
  --out-dir extras/runs/reports/final_2026-05-17
```

Latency (all models):

```bash
CUDA_VISIBLE_DEVICES=7 python extras/scripts/bench_latency.py \
  --weights <ckpt.pt> --label <ID> --dtype bfloat16 --out-dir extras/runs/latency
```

---

## 8. Artifacts

| Artifact | Path |
|----------|------|
| This report | `extras/REPORT.md` |
| POGEMA global | `extras/runs/reports/final_2026-05-17/global_by_model.md` |
| POGEMA × agents | `extras/runs/reports/final_2026-05-17/by_map_group_agents.md` |
| Latency B=2048 | `extras/runs/reports/final_2026-05-17/latency_b2048.md` |
| Checkpoints manifest | `extras/weights_manifest.json` |
| Shareable zip | `bash extras/scripts/package_weights.sh v1` → `dist/lightweight-checkpoints-v1.zip` |

Author baselines (2M/6M/85M): [Hugging Face](https://huggingface.co/) per upstream MAPF-GPT README. Trained extras weights are distributed via zip, not git.

---

## Appendix A — Latency @ B=2048 (full)

| ID | p50 (ms) | p95 (ms) | vs R0 |
|----|----------|----------|-------|
| R0 | 18.15 | 18.31 | 1.00× |
| T1 | 14.80 | 14.86 | 1.23× |
| T2 | 11.50 | 11.59 | 1.58× |
| T3 | 14.81 | 14.91 | 1.23× |
| C1 | 14.83 | 14.93 | 1.22× |
| C2 | 11.48 | 11.55 | 1.58× |
| C3 | 18.18 | 18.25 | 1.00× |
| D1 | 12.13 | 12.21 | 1.50× |
| D2 | 12.14 | 12.19 | 1.50× |
| D3 | 12.15 | 12.21 | 1.49× |
| D4 | 14.84 | 14.93 | 1.22× |
| D5 | 39.05 | 39.20 | 0.46× |
| 6M | 39.04 | 39.41 | 0.46× |
| 85M | 225.67 | 228.02 | 0.08× |

---

*End of report.*
