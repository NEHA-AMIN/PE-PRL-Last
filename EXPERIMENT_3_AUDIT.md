# EXPERIMENT_3_AUDIT.md — Exp2 (Full LOD)
## Label + Order + Distance (Embedding-Level)
**Audit Status:** COMPLETE  
**Evidence Source:** `exp2_full_paper/models/`, `README-E2.md`, `mse_mae_scores_sorted.txt`

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Exp 2 — Full LOD (Label + Order + Distance) |
| Folder | `experiments/exp2_full_paper/` |
| Notebook | `exp2.ipynb` |
| Shell script | `exp2_phase2_alpha0.5.sh` |
| Result directory | `results/exp2_full_paper/` |
| Central results | `mse_mae_scores_sorted.txt` under "Exp2" |

---

## 2. Objective

Test all three PRL components simultaneously at the **embedding layer** using the original LOD formulation. Label provides distinctiveness, Order provides directional relationship, Distance provides proximity bias.

**Important note about naming:** Despite being called "LOD" (Label + Order + Distance), the "Order" and "Distance" in Exp2 are both embedded in a single `DistancePositionOperator` applied to Legendre embeddings. There is NO separate "Order" and "Distance" — the operator computes `Σ α(i,j)·w_ij·(P_i−P_j)`. The `D` (distance decay `α`) and `O` (ordering displacements) are fused.

---

## 3. Mathematical Formulation (Derived from Code)

**Source:** [`experiments/exp2_full_paper/models/embed.py:132-159`](experiments/exp2_full_paper/models/embed.py:132)

```
X'_i = X_i + T_i + P_i + O_i

where:
  X_i = TokenEmbedding(x_i)                    [semantic]
  T_i = TemporalEmbedding(t_i)                 [temporal]
  P_i = LegendrePositionEmbedding(i)            [label]
  O_i = DistancePositionOperator(P) * (1/√d)   [order+distance, applied to P not X]
```

### Distance Operator (from `distance_operator.py`):
```
O_i = (1/√d_model) · Σ_{j≠i} α(i,j) · w_ij · (P_i − P_j)

α(i,j) = 1/(1+|i−j|^a)                [index-based decay, a=1.0]
w_ij    = 1/(1+d_ij)                   [feature-space weighting]
d_ij    = ‖P_i − P_j‖₁                [L1 distance in Legendre space]
```

**Verified:** Code in `distance_operator.py:44-76` exactly implements this formula.

**CRITICAL OBSERVATION — What "Order" means in Exp2:**
- The `distance_operator` receives `legendre_pos` (positional space), NOT `value_emb` (semantic space)
- This means the signed displacements `P_i − P_j` are differences between **Legendre polynomial evaluations** — purely positional vectors
- The feature-space weighting `w_ij = 1/(1+d_ij)` uses L1 distance in **Legendre space** (not semantic value space)
- README says the operator "captures directional relationships among tokens" — this is technically the displacement between **position vectors**, not between **token values**

**Verified against README-E2.md:** README states `O_i = Σ_{j≠i} α(i,j)·(w_ij⊙Δx_ij)` where `Δx_ij = P_i − P_j`. ✅ Consistent with code.

---

## 4. Algorithm Flow

```
Input [B, 96, 7]
         ↓ TokenEmbedding [B, 96, 512]
         + TemporalEmbedding [B, 96, 512]
         + LegendrePositionEmbedding [B, 96, 512]          ← no scipy-free version; dynamic import
         + DistancePositionOperator(legendre_pos) / √512  ← O(L²) computation on Legendre
         ↓ Dropout
FullAttention Encoder (UNMODIFIED — no attention-level decay)
         ↓
Decoder (same embedding)
         ↓
Linear projection
```

**Key distinction from Exp6-Post/Pre:** Exp2 has **no distance decay in attention** — the distance signal is entirely in the embedding. Exp6 variants have distance decay in attention AND a clean delta in the V matrix.

---

## 5. Code Walkthrough

### 5.1 `embed.py` — Dynamic Imports (Lines 107-128)
```python
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from legendre_embedding import LegendrePositionEmbedding
from distance_operator import DistancePositionOperator
```
This import pattern means the experiment MUST be run from the same directory as these files, or the copy-to-Informer2020-original mechanism must include `distance_operator.py`. Shell scripts should copy this file too.

### 5.2 `distance_operator.py` — Memory Efficiency (Lines 66-75)
The operator avoids allocating `[B, L, L, D]` tensor:
```python
W_sum = W.sum(dim=2, keepdim=True)  # [B, L, 1]
term1 = X * W_sum                   # [B, L, D]
term2 = torch.bmm(W, X)             # [B, L, D]
O = term1 - term2                   # [B, L, D]
```
This is algebraically equivalent to `Σ_j W_ij·(X_i−X_j)`. ✅ Verified.

### 5.3 `attn.py` — UNMODIFIED
Exp2's `attn.py` is identical to the baseline `FullAttention` — no distance decay in attention. **This confirms that Exp2's "Distance" component is ONLY at the embedding level**, not in attention. The README description is correct.

---

## 6. Configuration Audit

| Parameter | README-E2.md | mse_mae_scores_sorted.txt | Consistent? |
|-----------|-------------|--------------------------|-------------|
| pred_len | 24 (!) | 96, 192 (Ph1); 48, 96, 192 (Ph2) | 🔴 MISMATCH (same issue as Exp1) |
| decay_a | 1.0 | 1.0 (fixed) | ✅ |
| distance_type | L1 | — | ✅ (from code) |
| attn | full | — | ✅ (from model.py) |

**🔴 Same pred_len=24 error as Exp1-Pre README.** README-E2.md states `Prediction Length: 24`. Actual runs use 48/96/192. This error is present in multiple experiment READMEs, suggesting they were generated from a common template with an outdated pred_len value.

---

## 7. Result Verification

### Phase 1 (from `mse_mae_scores_sorted.txt`):
| α | pred_len | MSE | MAE |
|---|---------|-----|-----|
| 1.0 | 96 | 0.8242 | 0.7279 |
| 1.0 | 192 | 0.9002 | 0.7511 |

### Phase 2 (from `mse_mae_scores_sorted.txt`):
| pred_len | seed | MSE | MAE |
|---------|------|-----|-----|
| 48 | 2021 | 0.9983 | 0.7945 |
| 48 | 2022 | 0.7760 | 0.6871 |
| 48 | 2023 | 0.9125 | 0.7590 |
| 96 | 2021 | 0.8476 | 0.7159 |
| 96 | 2022 | 0.8439 | 0.7219 |
| 96 | 2023 | 0.8688 | 0.7373 |
| 192 | 2021 | 1.1408 | 0.8648 |
| 192 | 2022 | 0.8643 | 0.7345 |
| 192 | 2023 | 0.8823 | 0.7523 |

**README-E2.md Results Table:**
| Source | MSE | MAE |
|--------|-----|-----|
| README-E2.md | 0.8036 | 0.7102 |
| mse_mae_scores_sorted.txt Phase 1 (pred=96) | 0.8242 | 0.7279 |
| mse_mae_scores_sorted.txt Phase 2 (pred=96, seed=2021) | 0.8476 | 0.7159 |

**README.md summary table:**
| Source | MSE | MAE |
|--------|-----|-----|
| README.md (main) | 0.804 | 0.710 |
| README-E2.md | 0.8036 | 0.7102 |
| Central results (Phase 1, pred=96) | 0.8242 | 0.7279 |

**⚠️ Inconsistency:** README.md reports 0.804/0.710, README-E2.md reports 0.8036/0.7102. These are close but not identical (0.8040 vs 0.8036; 0.710 vs 0.7102 — likely rounding). The central results file shows Phase 1 as 0.8242/0.7279 — significantly different from both READMEs. Again, the README values cannot be verified from logged results.

**Averages check (Phase 2, pred=96):** (0.8476+0.8439+0.8688)/3 = 2.5603/3 = 0.8534. Central file states 0.8534. ✅ Arithmetic correct.

---

## 8. Consistency Checks

### 8.1 Component Classification
- ✅ Legendre (Label) confirmed from code
- ✅ DistancePositionOperator (Order+Distance fused) confirmed from code
- ⚠️ README.md summary table says "Exp 2 (LOD): Components = L+O+D" — technically correct but misleading because O and D are fused in one operator, not separate ablatable components

### 8.2 Formula
- ✅ `O_i = Σ α·w·(P_i−P_j)` matches code exactly
- ✅ Applied to Legendre embeddings (positional space), NOT value_emb — confirmed
- ✅ `distance_scale = 1/√d_model` — confirmed in code

### 8.3 Results
- ⚠️ README-E2.md MSE=0.8036 vs README.md MSE=0.804 — rounding discrepancy
- 🔴 Neither README value matches central results file Phase 1 (0.8242)

---

## 9. Inconsistency Report

### Critical Issues
1. **README-E2 pred_len=24 error**: Same template error as Exp1.
2. **README MSE=0.8036/0.804 not in central results file**: Phase 1 shows 0.8242, Phase 2 seed=2021 shows 0.8476 — neither matches.

### Moderate Issues
3. **O and D are fused, not separate**: The "Order" and "Distance" components of Exp2's LOD are implemented as a single operator. The ablation therefore tests L+(O⊕D), not L+O+D independently.
4. **README.md rounding**: 0.8036 → 0.804 (main README rounds to 3 decimal places; E2 README uses 4).

### Minor Issues
5. Shell script `exp2_phase2_alpha0.5.sh` name implies α=0.5, but Phase 2 in central results uses only α=1.0.

---

## 10. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation confidence | 9/10 | Operator code is correct and verified |
| Documentation confidence | 3/10 | README pred_len wrong; MSE cannot be traced to logs |
| Result confidence | 4/10 | Central file averages check out; but specific result values conflict with READMEs |
