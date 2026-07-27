# Distributed Ablation of Position-Relative Learning (PRL) — ETTh1

> **Ablation study on the Informer transformer** systematically isolating three novel positional encoding components — **Label (L)**, **Order (O)**, and **Distance (D)** — on the ETTh1 multivariate time-series forecasting benchmark.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Baseline Model](#baseline-model)
3. [PRL Components](#prl-components)
4. [Experiments](#experiments)
5. [Results Summary](#results-summary)
6. [Repository Structure](#repository-structure)
7. [Running Experiments](#running-experiments)
8. [Key Findings](#key-findings)

---

## Project Overview

This repository contains a complete ablation study for **Position-Relative Learning (PRL)** applied to the [Informer](https://arxiv.org/abs/2012.07436) (AAAI'21 Best Paper) architecture for long-sequence time-series forecasting.

The core question being answered:
> *Which positional encoding components — Label, Order, and Distance — contribute to forecasting quality, and how do they interact?*

**Dataset:** ETTh1 (Electricity Transformer Temperature — Hourly, 7 features, ~17,420 hourly observations)  
**Task:** Multivariate time-series forecasting at horizons 24 / 48 / 96 / 192 / 336 / 720  
**Base Configuration:** `seq_len=96`, `label_len=48`, `d_model=512`, `n_heads=8`, `e_layers=2`, `d_layers=1`

---

## Baseline Model

The **Informer** replaces quadratic self-attention with **ProbSparse attention** (O(L log L)), making it tractable for long sequences. Its embedding stack is:

```
X'_i = X_i + PE_i + T_i

  X_i  = TokenEmbedding (Conv1D)        — semantic content
  PE_i = Sinusoidal positional encoding — position signal
  T_i  = TemporalEmbedding (hour/day/month/weekday/holiday)
```

| Metric | pred_len=96 |
|--------|-------------|
| **MSE** | **0.519** |
| **MAE** | **0.513** |

---

## PRL Components

| Symbol | Name | Description | Formula |
|--------|------|-------------|---------|
| **L** | Label | Legendre polynomial basis — gives each position a **unique orthogonal fingerprint** | `P_i = [L_0(î), …, L_{d-1}(î)]`, î ∈ [-1,1] |
| **O** | Order | Signed feature-space displacements — encodes **directional relationships** among tokens | `O_i = (1/(N-1))·Σ_{j≠i}(X_i − X_j)` |
| **D** | Distance | Index-based proximity decay applied to attention scores — encodes **spatial locality** | `α(i,j) = 1/(1 + \|i−j\|^a)` |

---

## Experiments

### Baseline — Vanilla Informer
The unmodified Informer with sinusoidal PE + temporal embedding, serving as the reference for all ablations. All subsequent experiments replace or augment the positional encoding stack.

---

### Exp 1 — Distance Decay (Pre-Softmax)
Removes sinusoidal PE and injects index-based distance decay **before** the softmax in attention; tests whether proximity bias alone provides sufficient positional signal without any explicit embedding.

> **Change vs baseline:** Sinusoidal PE removed; `scores ← scores × α(i,j)` applied pre-softmax in `attn.py`. Temporal embedding retained.

---

### Exp 1-Post — Distance Decay (Post-Softmax)
Identical to Exp 1 but applies the same distance decay **after** softmax (`A ← softmax(scores) × α(i,j)`); designed to quantify whether the placement of decay relative to normalisation matters.

> **Change vs baseline:** Same as Exp 1 but decay is post-softmax, breaking probability normalisation.

---

### Exp 2 — Full LOD (Label + Order + Distance)
Combines all three PRL components simultaneously: Legendre labels in the embedding, signed-displacement ordering in the embedding, and distance decay in attention; tests whether the three signals are positively synergistic.

> **Change vs baseline:** Sinusoidal PE replaced by `P_i + O_i`; pre-softmax distance decay also active. All three components (L, O, D) active together.

---

### Exp 3 — Label Only (Legendre Polynomials)
Replaces sinusoidal PE with Legendre polynomial embeddings **and also removes temporal embedding**; isolates the pure orthogonal distinctiveness signal of Legendre labels with no other positional context.

> **Change vs baseline:** Sinusoidal PE replaced by Legendre labels; temporal embedding also removed, making this a stripped configuration `X'_i = X_i + P_i`.

---

### Exp 3b — Label + Temporal (Controlled)
Re-introduces temporal embedding alongside Legendre labels (unlike Exp 3); acts as a controlled counterpart to separate whether Exp 3's failure was caused by the label encoding or by the absence of temporal features.

> **Change vs baseline:** Sinusoidal PE replaced by Legendre labels; temporal embedding restored. Formula: `X'_i = X_i + T_i + P_i`.

---

### Exp 4 — Order Only (Signed Displacements)
Replaces sinusoidal PE with the ordering operator alone (uniform-weighted signed displacements in value-embedding space); isolates the directional relationship signal without any label distinctiveness or distance decay.

> **Change vs baseline:** Sinusoidal PE replaced by `O_i`; temporal embedding retained. Formula: `X'_i = X_i + T_i + O_i`.

---

### Exp 4a — Order Only (Mentor Version)
An alternate formulation of the ordering operator guided by mentor feedback, exploring a different normalisation strategy; used to validate robustness of the Order component implementation before inclusion in combined experiments.

> **Change vs baseline:** Same structural change as Exp 4 but with a revised ordering formula provided during project review.

---

### Exp 4b — Order in Positional Space
Computes the ordering operator using sequential **Legendre deltas** (positional-space differences) rather than value-embedding differences; tests whether ordering signals derived from the positional manifold outperform those from the semantic manifold.

> **Change vs baseline:** Sinusoidal PE replaced by positional-space ordering `O_i^pos = ΔP_i / (‖P‖_mean + ε)`; temporal embedding retained.

---

### Exp 4 (Sem) — Order in Semantic Space
Uses normalised sequential value-embedding deltas (`delta_i^val / x̄_val`) instead of pairwise displacements; tests a cleaner, causally structured formulation of the Order signal aligned to temporal transitions.

> **Change vs baseline:** Sinusoidal PE replaced by `O_i^sem`; temporal embedding retained. Flag: `--pe_mode ordering_sem`.

---

### Exp 5 — Label + Order (L+O Synergy)
Combines Legendre labels with uniform-weighted ordering (no distance decay); directly tests whether L and O are complementary and whether their combination can beat distance-only (Exp 1) without any explicit proximity bias.

> **Change vs baseline:** Sinusoidal PE replaced by `P_i + O_i`; temporal embedding retained; no distance decay. Formula: `X'_i = X_i + T_i + P_i + O_i`.

---

### Exp 5 (Pos) — L+O with Order in Positional Space
Variant of Exp 5 where the ordering operator is computed in positional (Legendre) space rather than semantic (value) space; tests whether keeping both L and O in the same positional manifold is beneficial or redundant.

> **Change vs baseline:** Same as Exp 5 but `O_i` is derived from Legendre deltas rather than value deltas.

---

### Exp 5b — Label + Order with Clean Delta (Semantic Separation)
A architectural innovation that separates positional and semantic signals across the attention mechanism: Q/K receive the combined `(X + T + P)` embedding while V receives only the clean semantic delta `Δx_i = X_i − X_{i-1}`; hypothesises that disentangling distinctiveness (L) from directionality (O) across Q/K/V improves forecasting.

> **Change vs baseline:** Sinusoidal PE replaced; `DataEmbedding` returns a tuple `(combined_emb, delta_x)`; encoder and attention modified so V uses `delta_x` only, while Q/K use `combined_emb`.

---

### Exp 6 Pre — LOD with Pre-Softmax Decay
Reconstructs the full LOD combination (Exp 2) but applies distance decay **pre-softmax** (like Exp 1) rather than post-softmax; tests whether the placement of distance decay within the attention pipeline reverses LOD's performance degradation relative to L+O.

> **Change vs baseline:** All three components (L, O, D) active; distance decay applied to attention logits before normalisation.

---

### Exp 6 Post — LOD with Post-Softmax Decay
Reconstructs full LOD but applies distance decay **post-softmax**; acts as the direct complement to Exp 6 Pre to determine whether soft re-weighting of probabilities is structurally safer than logit-level interference when complex embeddings are already present.

> **Change vs baseline:** All three components (L, O, D) active; distance decay applied to attention weights after softmax normalisation.

---

## Results Summary

All results on **ETTh1, pred_len = 96**, single seed (2021):

| Rank | Experiment | Components | Temporal PE | MSE ↓ | MAE ↓ | vs Baseline |
|------|-----------|-----------|:-----------:|-------|-------|-------------|
| 🥇 | **Baseline** | Sinusoidal | ✅ | **0.519** | **0.513** | — |
| 🥈 | **Exp 5 (L+O)** | Label + Order | ✅ | **0.719** | **0.635** | +38.5% |
| 🥉 | **Exp 1 (D pre)** | Distance pre | ✅ | **0.725** | **0.652** | +39.7% |
| 4 | Exp 2 (LOD) | L+O+D | ✅ | 0.804 | 0.710 | +54.9% |
| 5 | Exp 4 (O) | Order | ✅ | 0.835 | 0.720 | +60.9% |
| 6 | Exp 1-Post (D post) | Distance post | ✅ | 0.9072 | 0.7027 | +74.8% |
| 7 | Exp 3 (L) | Label | ❌ | 1.124 | 0.855 | +116.6% |

> **Lower MSE / MAE is better.**

---

## Repository Structure

```
Dist-Abl-PRL-All-Exs-ETTH1/
│
├── README.md                            ← This file
├── TECHNICAL_OBSERVATIONS.md           ← Design choices, delta boundary analysis
├── run_phase1_baseline.sh               ← Master script: baseline Phase 1 runs
│
├── Informer2020-original/              ← Original Informer codebase (base for all exps)
│   ├── main_informer.py                ← Training entry point
│   ├── requirements.txt / environment.yml / Dockerfile
│   ├── models/
│   │   ├── model.py                    ← Informer class (pe_mode dispatch)
│   │   ├── embed.py                    ← DataEmbedding variants
│   │   ├── attn.py                     ← FullAttention + ProbSparse + distance hooks
│   │   ├── encoder.py / decoder.py     ← Standard Informer stack
│   │   └── legendre_embedding.py       ← [ADDED] Legendre polynomial basis
│   ├── data/
│   │   ├── data_loader.py
│   │   └── ETT/ETTh1.csv              ← Raw dataset
│   ├── exp/
│   │   ├── exp_basic.py
│   │   └── exp_informer.py            ← Experiment runner
│   ├── utils/
│   │   ├── tools.py / metrics.py / masking.py / timefeatures.py
│   ├── scripts/
│   │   ├── ETTh1.sh / ETTh2.sh / ETTm1.sh / WTH.sh
│   └── checkpoints/                   ← Baseline model checkpoints
│
├── experiments/                        ← Per-experiment source code & configs
│   ├── Baseline/
│   │   ├── BaseLine_Model (1).ipynb
│   │   ├── COLAB_INSTRUCTIONS.md
│   │   ├── baseline_phase1.sh / baseline_phase2.sh
│   │   └── setup_dataset.sh
│   │
│   ├── exp1_distance_pre_softmax_decay/        ← Exp 1: D pre-softmax
│   │   ├── README-E1.md
│   │   ├── exp1_phase1_pre.ipynb
│   │   └── models/ (attn.py★ embed.py★ encoder.py decoder.py model.py)
│   │
│   ├── exp1_distance_post_softmax/             ← Exp 1-Post: D post-softmax
│   │   ├── README.md
│   │   ├── exp1_post.ipynb / Exp1_post_alpha0_5.ipynb
│   │   ├── exp1_alpha_0.5.sh / exp1_post_phase2_alpha0.5.sh
│   │   └── models/ (attn.py★ embed.py★ test_attn.py …)
│   │
│   ├── exp2_full_paper/                        ← Exp 2: L+O+D (Full LOD)
│   │   ├── README-E2.md
│   │   ├── exp2.ipynb / exp2_phase2_alpha0.5.sh
│   │   └── models/ (legendre_embedding.py★ distance_operator.py★ embed.py★ …)
│   │
│   ├── exp3_label_only/                        ← Exp 3: Label only (no temporal)
│   │   ├── README-E3.md
│   │   ├── exp3.ipynb
│   │   └── models/ (legendre_embedding.py★ embed.py★ …)
│   │
│   ├── E-96-3b-Label-Temporal-Controlled/      ← Exp 3b: Label + Temporal
│   │   ├── README.md
│   │   ├── exp3b.ipynb
│   │   └── models/ (legendre_embedding.py★ embed.py★ …)
│   │
│   ├── exp4_order_only/                        ← Exp 4: Order only
│   │   ├── README-E4.md / theory.md
│   │   ├── Exp4_order_only.ipynb
│   │   ├── exp4_order_only.sh / exp4_order_ony_phase2.sh
│   │   └── models/ (ordering_operator.py★ embed.py★ …)
│   │
│   ├── exp4a_order_only_mentor_version/        ← Exp 4a: Order (mentor revision)
│   │   ├── README-E4.md
│   │   ├── exp4a_order_only_mentor_version.ipynb
│   │   └── models/
│   │
│   ├── exp4b_order_input_mentor_version/       ← Exp 4b: Order in positional space
│   │   ├── README-E4b.md / theory.md
│   │   ├── exp4b_order_input_mentor_version_updated.ipynb
│   │   ├── exp4b_phase2_pred336.sh
│   │   └── models/ (ordering_operator_positional.py★ …)
│   │
│   ├── exp4b_order_input_position/             ← Exp 4b alt: positional-space Order
│   │   ├── README-E4b.md / theory.md
│   │   ├── exp4b_order_input_position.ipynb
│   │   └── models/ (ordering_operator_positional.py★ …)
│   │
│   ├── exp4_ordering_new_sem_space/            ← Exp 4 (Sem): semantic-space Order
│   │   ├── README.md
│   │   ├── exp4_ordering_new_sem_space_ph1.sh
│   │   └── models/
│   │
│   ├── exp5_label_order/                       ← Exp 5: L+O (best ablation)
│   │   ├── README-E5.md / theory.md
│   │   ├── Exp5_label_order.ipynb
│   │   ├── exp5_label_order_ph1.sh / exp5_label_order_ph2.sh
│   │   └── models/ (legendre_embedding.py★ ordering_operator.py★ embed.py★ …)
│   │
│   ├── exp5_ordering_new_pos_space/            ← Exp 5 (Pos): L+O positional-space
│   │   ├── README.md
│   │   ├── exp5_ordering_new_pos_space_ph1.sh / _ph2.sh
│   │   └── models/
│   │
│   ├── exp5b_label_order_clean_delta_MV/       ← Exp 5b: L+O clean-delta V
│   │   ├── README.md / theory.md
│   │   ├── exp5b_label_order_clean_delta_MV.ipynb
│   │   ├── e5b_lab_ord_clean_delta_mv_ph1.sh
│   │   └── models/ (legendre_embedding.py★ embed.py★ attn.py★ encoder.py★ model.py★)
│   │
│   ├── exp6_lod_pre/                           ← Exp 6 Pre: L+O+D pre-softmax
│   │   ├── README-E6-Pre.md
│   │   ├── Exp6_lod_pre-Gcolab.ipynb
│   │   ├── exp6_lod_pre_phase1.sh / exp6_lod_pre_phase2.sh
│   │   └── models/ (legendre_embedding.py★ embed.py★ attn.py★ …)
│   │
│   └── exp6_lod_post/                          ← Exp 6 Post: L+O+D post-softmax
│       ├── README-E6-POST.md
│       ├── Exp6_lod_post-Gcolab.ipynb
│       ├── exp6_lod_post_phase1.sh / exp6_lod_post_phase2.sh
│       └── models/ (legendre_embedding.py★ embed.py★ attn.py★ …)
│
├── results/                            ← Training logs & metrics (per-run)
│   ├── baseline_ph1_ETTh1_pred96_seed2021/   training_log.txt
│   ├── baseline_ph1_ETTh1_pred192_seed2021/  training_log.txt
│   ├── exp1_distance_decay/                  training_log.txt
│   ├── exp1_distance_post_softmax/           training_log.txt
│   ├── exp2_full_paper/                      training_log.txt
│   ├── exp3_label_only/                      training_log.txt
│   ├── exp3b_label_temporal_48/ … _720/      training_log.txt × 5
│   ├── exp4_order_only/                      training_log.txt
│   └── exp5_label_order/                     training_log.txt
│
├── checkpoints/                        ← Saved model weights (current experiments)
└── logs/                               ← Stdout logs (currently empty)
```

> `★` marks files that were modified relative to the vanilla Informer codebase.

---

## Running Experiments

### Prerequisites

```bash
# Create conda environment
conda env create -f Informer2020-original/environment.yml
conda activate informer

# Or use pip
pip install -r Informer2020-original/requirements.txt
```

### Run Baseline

```bash
bash run_phase1_baseline.sh
# or directly:
cd Informer2020-original
python main_informer.py \
  --model informer --data ETTh1 --attn full \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --pe_mode vanilla --seed 2021
```

### Run an Ablation Experiment

Each experiment folder ships with a shell script:

```bash
# Example: Exp 5 (Label + Order)
bash experiments/exp5_label_order/exp5_label_order_ph1.sh

# Example: Exp 1 with custom decay exponent
cd Informer2020-original
python main_informer.py \
  --model informer --data ETTh1 --attn full \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --pe_mode distance_pre --decay_a 1.0 --seed 2021
```

### `--pe_mode` Flag Reference

| `--pe_mode` value | Experiment |
|-------------------|-----------|
| `vanilla` | Baseline |
| `distance_pre` | Exp 1 |
| `distance_post` | Exp 1-Post |
| `lod` | Exp 2 (full LOD) |
| `label_only` | Exp 3 |
| `label_temporal` | Exp 3b |
| `order_only` | Exp 4 |
| `ordering_sem` | Exp 4 (Semantic) |
| `ordering_pos` | Exp 4b |
| `label_order` | Exp 5 |
| `label_order_pos` | Exp 5 (Pos) |
| `label_order_clean_delta` | Exp 5b |
| `lod_pre` | Exp 6 Pre |
| `lod_post` | Exp 6 Post |

---

## Key Findings

1. **Label + Order (Exp 5) is the best ablation** — MSE 0.719 outperforms Distance-only (0.725); orthogonal distinctiveness and directional relationships are positively synergistic.

2. **Distance hurts L+O** — Adding D to L+O (Exp 2: 0.804) is 11% worse than L+O alone (Exp 5: 0.719); distance decay introduces conflicting biases when both semantic signals are present.

3. **Pre-softmax placement is critical** — Exp 1 (pre: 0.725) vs Exp 1-Post (post: 0.9072); applying decay after normalisation breaks the probability simplex.

4. **Label alone is insufficient** — Exp 3 (no temporal, label only: 1.124) is the worst result; positional distinctiveness without temporal context fails entirely.

5. **Signal separation may help** — Exp 5b routes Label into Q/K and clean Order delta into V, hypothesising that orthogonal decomposition of positional and semantic signals improves attention quality (results pending).

6. **None of the ablations match vanilla** — The standard sinusoidal PE + temporal embedding (baseline: 0.519) remains the strongest configuration; all PRL variants are exploring whether a theoretically motivated replacement can close this gap.

---

## Citation

If you use this code, please cite the original Informer paper:

```bibtex
@inproceedings{zhou2021informer,
  title={Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle={AAAI},
  year={2021}
}
```
