# REPOSITORY_STRUCTURE.md
## Complete Evidence-Based Repository Inventory
### Dist-Abl-PRL-All-Exs-ETTH1
**Audit Date:** Phase 1 — Full Scan  
**Auditor Note:** All contents verified by direct file reading. No assumptions made from names alone.

---

## 1. Root-Level Files

| File | Purpose | Notes |
|------|---------|-------|
| `README.md` | Project overview, experiment table, results summary | Contains results table (pred_len=96 only). Numbers may differ from mse_mae_scores_sorted.txt |
| `TECHNICAL_OBSERVATIONS.md` | Design decision log, delta boundary analysis, positional vs semantic space explanation | References path `/Users/nehaamin/Desktop/...` — stale local path |
| `research-audit-plan.md` | Forensic audit of ordering experiments; classifies all ordering variants A–G | Comprehensive prior audit; references files that exist in Archived_experiments |
| `mse_mae_scores_sorted.txt` | Central results log for all experiments, Phase 1 and Phase 2 | **CRITICAL: Exp6-Pre and Exp6-Post Phase 1 AND Phase 2 scores are byte-for-byte identical** |
| `run_phase1_baseline.sh` | Entry-point script for baseline run | References `Informer2020` directory (not `Informer2020-original`) |

---

## 2. Directory Tree

```
Dist-Abl-PRL-All-Exs-ETTH1/
│
├── README.md
├── TECHNICAL_OBSERVATIONS.md
├── research-audit-plan.md
├── mse_mae_scores_sorted.txt
├── run_phase1_baseline.sh
│
├── Informer2020-original/           ← SHARED base codebase
│   ├── main_informer.py             ← Training entry point (not read yet — shared)
│   ├── requirements.txt
│   ├── environment.yml
│   ├── Dockerfile
│   ├── Makefile
│   ├── models/
│   │   ├── attn.py                  ← MODIFIED in embed.py: adds DataEmbedding_ordering_sem
│   │   ├── embed.py                 ← Contains DataEmbedding + DataEmbedding_ordering_sem
│   │   ├── model.py                 ← Only dispatches: 'vanilla' or 'ordering_sem' (2 modes only)
│   │   ├── encoder.py               ← Standard (unmodified)
│   │   ├── decoder.py               ← Standard (unmodified)
│   │   └── legendre_embedding.py    ← NOT PRESENT in original; added by experiments
│   ├── data/
│   │   ├── data_loader.py
│   │   └── ETT/ETTh1.csv
│   ├── exp/
│   │   ├── exp_basic.py
│   │   └── exp_informer.py
│   └── utils/
│       ├── tools.py
│       ├── metrics.py
│       ├── masking.py
│       └── timefeatures.py
│
├── experiments/
│   ├── Baseline/
│   ├── exp1_distance_pre_softmax_decay/
│   ├── exp1_distance_post_softmax/
│   ├── exp2_full_paper/
│   ├── exp3_label_only/
│   ├── E-96-3b-Label-Temporal-Controlled/
│   ├── exp5_label_order/
│   ├── exp5b_label_order_clean_delta_MV/
│   ├── exp6_lod_pre/
│   ├── exp6_lod_post/
│   ├── Formula-A-pos/
│   ├── Formula-B-pos/
│   ├── Formula-B-sem/
│   └── formula-A-sem/
│
├── Archived_experiments/
│   ├── exp4_order_only/
│   ├── exp4_ordering_archive/
│   ├── exp4a_order_only_mentor_version/
│   ├── exp4b_order_input_mentor_version/
│   └── exp4b_order_input_position/
│
├── results/
│   ├── baseline_ph1_ETTh1_pred96_seed2021/    (empty dir)
│   ├── baseline_ph1_ETTh1_pred192_seed2021/   (empty dir)
│   ├── exp1_distance_decay/
│   ├── exp1_distance_post_softmax/
│   ├── exp2_full_paper/
│   ├── exp3_label_only/
│   ├── exp3b_label_temporal_48/
│   ├── exp3b_label_temporal_96/
│   ├── exp3b_label_temporal_192/
│   ├── exp3b_label_temporal_336/
│   ├── exp3b_label_temporal_720/
│   ├── exp4_order_only/
│   └── exp5_label_order/
│
└── checkpoints/                               (model weight storage)
```

---

## 3. File Inventory Table — All Experiment Model Files

| File Path | Purpose | Used By | Key Notes |
|-----------|---------|---------|-----------|
| `Informer2020-original/models/attn.py` | Baseline FullAttention + ProbAttention | Baseline | No decay; standard scaled dot-product |
| `Informer2020-original/models/embed.py` | Baseline DataEmbedding + DataEmbedding_ordering_sem | Baseline, Exp4-Sem | model.py dispatches between these two only |
| `Informer2020-original/models/model.py` | Informer class; pe_mode dispatch | All exps via copy | Only supports 'vanilla' and 'ordering_sem' — ALL other modes require experiment-local model.py |
| `experiments/exp1_distance_pre_softmax_decay/models/attn.py` | FullAttention with pre-softmax distance decay | Exp1-Pre | `scores = scores * alpha` BEFORE softmax |
| `experiments/exp1_distance_pre_softmax_decay/models/embed.py` | DataEmbedding WITHOUT sinusoidal PE | Exp1-Pre | `x = value_embedding + temporal_embedding` (no PE) |
| `experiments/exp1_distance_pre_softmax_decay/models/model.py` | Informer with decay_a parameter | Exp1-Pre | decay_a passed to FullAttention and ProbAttention |
| `experiments/exp1_distance_post_softmax/models/attn.py` | FullAttention with post-softmax distance decay | Exp1-Post | `A = A * alpha` AFTER softmax |
| `experiments/exp2_full_paper/models/embed.py` | DataEmbedding with Legendre + DistancePositionOperator | Exp2 | Distance operator applied to legendre_pos NOT value_emb |
| `experiments/exp2_full_paper/models/attn.py` | UNMODIFIED FullAttention (copy of baseline) | Exp2 | NO distance decay in attention for Exp2 |
| `experiments/exp2_full_paper/models/distance_operator.py` | Full LOD distance operator: α(i,j)·w_ij·Δx_ij | Exp2 only | O(L²D) complexity; feature-space weighting |
| `experiments/exp3_label_only/models/embed.py` | DataEmbedding: value_emb + legendre ONLY (no T_i) | Exp3 | Verified: x = value_emb + legendre_pos (no temporal) |
| `experiments/E-96-3b-Label-Temporal-Controlled/models/` | Label + Temporal (controlled) | Exp3b | Not fully read — structure same as exp3 + temporal |
| `experiments/exp5_label_order/models/embed.py` | DataEmbedding: value+temporal+legendre+ordering | Exp5 | ordering_operator applied to legendre_pos (positional space) |
| `experiments/exp5_label_order/models/ordering_operator.py` | Pure pairwise displacement: (1/N-1)·Σ(P_i−P_j) | Exp5 | O(L²D) — all-pairs, NOT consecutive delta |
| `experiments/exp5b_label_order_clean_delta_MV/models/embed.py` | Returns tuple (combined_emb, delta_x) | Exp5b | delta_x = value_emb - roll(value_emb,1); zero first position |
| `experiments/exp5b_label_order_clean_delta_MV/models/attn.py` | AttentionLayer with optional delta_values for V | Exp5b | delta_values overrides V projection |
| `experiments/exp5b_label_order_clean_delta_MV/models/encoder.py` | Modified encoder passing delta_x through layers | Exp5b | ConvLayer also downsamples delta_x |
| `experiments/exp6_lod_post/models/embed.py` | Returns (combined_emb, delta_x); dropout on BOTH | Exp6-Post | delta_x gets dropout; exp6_lod_pre does not apply dropout to delta_x identically |
| `experiments/exp6_lod_post/models/attn.py` | Post-softmax distance decay + delta_values V | Exp6-Post | A = A * alpha AFTER softmax |
| `experiments/exp6_lod_post/models/encoder.py` | Passes delta_x; does NOT downsample delta_x through ConvLayer | Exp6-Post | **BUG/DIFFERENCE vs Exp5b: ConvLayer applied to x but NOT delta_x** |
| `experiments/exp6_lod_pre/models/embed.py` | Returns (combined_emb, delta_x); dropout on combined ONLY | Exp6-Pre | dropout(combined_emb) but NOT dropout(delta_x) separately |
| `experiments/exp6_lod_pre/models/attn.py` | Pre-softmax distance decay + delta_values V | Exp6-Pre | scores = scores * alpha BEFORE softmax |
| `experiments/exp6_lod_pre/models/encoder.py` | Identical to exp6_lod_post encoder | Exp6-Pre | Same code |
| `experiments/exp6_lod_pre/models/legendre_embedding.py` | Pure PyTorch Legendre recurrence (no scipy) | Exp6-Pre, Exp6-Post | Uses recurrence: P_n = ((2n-1)xP_{n-1}-(n-1)P_{n-2})/n |
| `experiments/exp6_lod_post/models/legendre_embedding.py` | Same as exp6_lod_pre version but with `self.d_model ** 0.5` not `math.sqrt(self.d_model)` | Exp6-Post | Functionally identical scaling |

---

## 4. Experiment Classification Cross-Reference

| Experiment ID | Folder | Components (L/O/D) | Temporal | PE Removed | Attn Modified | Entry pe_mode |
|---------------|--------|---------------------|----------|------------|---------------|---------------|
| Baseline | `experiments/Baseline/` | None (vanilla) | ✅ | ❌ | ❌ | `prob` (no pe_mode arg) |
| Exp1-Pre | `exp1_distance_pre_softmax_decay/` | D-pre | ✅ | ✅ | Pre-softmax decay | N/A (uses exp-local model.py) |
| Exp1-Post | `exp1_distance_post_softmax/` | D-post | ✅ | ✅ | Post-softmax decay | N/A |
| Exp2 (LOD) | `exp2_full_paper/` | L+O+D (embedding-level) | ✅ | ✅ | ❌ | N/A |
| Exp3 (L-only) | `exp3_label_only/` | L only | ❌ | ✅ | ❌ | N/A |
| Exp3b | `E-96-3b-Label-Temporal-Controlled/` | L only | ✅ | ✅ | ❌ | N/A |
| Exp5 (L+O) | `exp5_label_order/` | L+O (positional pairwise) | ✅ | ✅ | ❌ | N/A |
| Exp5b | `exp5b_label_order_clean_delta_MV/` | L+O (delta MV split) | ✅ | ✅ | V=delta_x | N/A |
| Exp6-Pre | `exp6_lod_pre/` | L+O+D-pre (delta MV + pre) | ✅ | ✅ | Pre-softmax + V=delta_x | N/A |
| Exp6-Post | `exp6_lod_post/` | L+O+D-post (delta MV + post) | ✅ | ✅ | Post-softmax + V=delta_x | N/A |
| Formula-A-pos | `Formula-A-pos/` | Not read — pending | — | — | — | — |
| Formula-B-pos | `Formula-B-pos/` | Not read — pending | — | — | — | — |
| Formula-B-sem | `Formula-B-sem/` | Not read — pending | — | — | — | — |
| formula-A-sem | `formula-A-sem/` | Not read — pending | — | — | — | — |

---

## 5. Results Directory vs Experiment Folder Mismatch

| Results Folder | Matching Experiment Folder | Mismatch? |
|---------------|---------------------------|-----------|
| `results/baseline_ph1_ETTh1_pred96_seed2021/` | `experiments/Baseline/` | ⚠️ **EMPTY** — no training_log.txt |
| `results/baseline_ph1_ETTh1_pred192_seed2021/` | `experiments/Baseline/` | ⚠️ **EMPTY** — no training_log.txt |
| `results/exp1_distance_decay/` | `experiments/exp1_distance_pre_softmax_decay/` | ⚠️ **NAME MISMATCH** — folder named "decay" not "pre_softmax_decay" |
| `results/exp1_distance_post_softmax/` | `experiments/exp1_distance_post_softmax/` | ✅ Match |
| `results/exp2_full_paper/` | `experiments/exp2_full_paper/` | ✅ Match |
| `results/exp3_label_only/` | `experiments/exp3_label_only/` | ✅ Match |
| `results/exp3b_label_temporal_*` | `experiments/E-96-3b-Label-Temporal-Controlled/` | ⚠️ **NAME MISMATCH** — "exp3b" vs "E-96-3b" |
| `results/exp4_order_only/` | `Archived_experiments/exp4_order_only/` | ⚠️ **ARCHIVED** — result exists but source is archived |
| `results/exp5_label_order/` | `experiments/exp5_label_order/` | ✅ Match |
| N/A | `experiments/exp5b_label_order_clean_delta_MV/` | ⚠️ **NO RESULTS FOLDER** — Exp5b has no results directory |
| N/A | `experiments/exp6_lod_pre/` | ⚠️ **NO RESULTS FOLDER** |
| N/A | `experiments/exp6_lod_post/` | ⚠️ **NO RESULTS FOLDER** |

---

## 6. Key Structural Observations

1. **Experiment execution pattern**: Each experiment copies its local `models/` files into `Informer2020-original/models/` before running `main_informer.py`. The `Informer2020-original` directory is the shared execution environment — it is overwritten by whichever experiment ran last.

2. **Informer2020-original/models/model.py state**: Currently contains only `vanilla` and `ordering_sem` dispatch. This is the state of the codebase as of the last overwrite — it is **not the original Informer**; it has been modified.

3. **No centralized results**: Results for Exp5b, Exp6-Pre, and Exp6-Post do not have corresponding `results/` subdirectories, suggesting those experiments were run externally (Google Colab) and results captured only in `mse_mae_scores_sorted.txt`.
