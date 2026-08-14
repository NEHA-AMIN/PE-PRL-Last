# EXPERIMENT_4_AUDIT.md — Exp3 (Label Only)
## Pure Legendre Embeddings, No Temporal
**Audit Status:** COMPLETE  
**Evidence Source:** `exp3_label_only/models/embed.py`, `README-E3.md`, `mse_mae_scores_sorted.txt`

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Exp 3 — Label Only (Legendre Polynomials) |
| Folder | `experiments/exp3_label_only/` |
| Notebook | `exp3.ipynb` |
| Result directory | `results/exp3_label_only/` |
| Central results | `mse_mae_scores_sorted.txt` under "Exp3" |
| Phase 2 status | **⚠️ NOT FOUND in notebook outputs** (per central results file) |

---

## 2. Objective

Isolate the effect of Legendre polynomial labels with ALL other components removed. No sinusoidal PE, no temporal embedding, no order signal, no distance decay.

---

## 3. Mathematical Formulation (Derived from Code)

**Source:** [`experiments/exp3_label_only/models/embed.py:130-134`](experiments/exp3_label_only/models/embed.py:130)
```python
value_emb = self.value_embedding(x)
legendre_pos = self.legendre_embedding(x)
x = value_emb + legendre_pos
```

**Formula:**
```
X'_i = X_i + P_i
  (NO T_i — temporal explicitly excluded)
  (NO sinusoidal PE)
  (NO ordering operator)
  (NO distance decay)
```

**Verified against README.md:** README states "X'_i = X_i + P_i" for Exp3 with "temporal embedding also removed." ✅ Confirmed from code.

**Verified against README-E3.md:** ✅ Consistent (README-E3.md states "NO temporal embedding, NO distance operator, NO ordering signal").

---

## 4. Legendre Embedding Implementation

This experiment uses a **dynamically imported** `LegendrePositionEmbedding`. The import is:
```python
sys.path.insert(0, os.path.dirname(__file__))
from legendre_embedding import LegendrePositionEmbedding
```

The `legendre_embedding.py` file in `experiments/exp3_label_only/models/` was not read directly. The implementation in `exp6_lod_pre/models/legendre_embedding.py` (which was read) implements:
- Pure PyTorch recurrence: `P_n = ((2n-1)·pos·P_{n-1}−(n-1)·P_{n-2})/n`
- Positions normalized to `[-1, 1]`
- Scaling by `1/√d_model`

The Exp2/Exp3/Exp5 Legendre files reference in `TECHNICAL_OBSERVATIONS.md` use scipy (`scipy.special.legendre`) based on the older design. There may be a difference between implementations — this is flagged as needing verification.

---

## 5. Result Verification

### Phase 1 (from `mse_mae_scores_sorted.txt`):
| pred_len | seed | MSE | MAE |
|---------|------|-----|-----|
| 96 | 2021 | 1.0989 | 0.8541 |
| 192 | 2021 | 1.4844 | 0.9973 |

**Phase 2:** Explicitly noted as "NOT FOUND in notebook outputs" in central results file.

**README.md summary table:**
| Source | MSE (pred=96) | MAE (pred=96) |
|--------|--------------|--------------|
| README.md | 1.124 | 0.855 |
| Central results (Phase 1, pred=96) | 1.0989 | 0.8541 |

**⚠️ Inconsistency:** README.md reports MSE=1.124/MAE=0.855. Central results file Phase 1 shows 1.0989/0.8541. These are close but not identical — README may have rounded differently, or used a different run. Given no Phase 2 results exist, the README value (1.124) cannot be corroborated.

---

## 6. Consistency Checks

### 6.1 Formula
- ✅ `X'_i = X_i + P_i` confirmed from code
- ✅ No temporal embedding confirmed from code (temporal object instantiated but never called in forward())
- **DEAD CODE:** `self.temporal_embedding` instantiated in `__init__` (line 102) but never used in `forward()`. `self.position_embedding` also instantiated but never used.

### 6.2 Results
- ⚠️ README 1.124 ≠ Central results 1.0989 (Δ = 0.025)
- 🔴 No Phase 2 results — experiment appears incomplete
- This is the worst-performing ablation; no reason was given for not completing Phase 2

### 6.3 Architecture
- ✅ No attention modification (standard FullAttention)
- ✅ Confirmed: removing temporal embedding makes this a significantly weaker model

---

## 7. Inconsistency Report

### Critical Issues
1. **Phase 2 incomplete**: No Phase 2 results exist. The experiment was not completed.
2. **README MSE (1.124) ≠ central results (1.0989)**: Source of 1.124 unknown.

### Moderate Issues
3. **Two dead-code objects**: `position_embedding` and `temporal_embedding` instantiated but never used.

---

## 8. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation confidence | 9/10 | Code is simple and clear |
| Documentation confidence | 4/10 | README value differs from logged value; incomplete documentation |
| Result confidence | 3/10 | Phase 2 missing; Phase 1 MSE differs from README |

---

# EXPERIMENT_5_AUDIT.md — Exp5 (Label + Order)
## Legendre Labels + Pairwise Ordering in Positional Space
**Audit Status:** COMPLETE  
**Evidence Source:** `exp5_label_order/models/`, `README-E5.md`, `mse_mae_scores_sorted.txt`

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Exp 5 — Label + Order (L+O Synergy) |
| Folder | `experiments/exp5_label_order/` |
| Notebook | `Exp5_label_order.ipynb` |
| Shell scripts | `exp5_label_order_ph1.sh`, `exp5_label_order_ph2.sh` |
| Theory doc | `theory.md` |
| Result directory | `results/exp5_label_order/training_log.txt` |
| Central results | `mse_mae_scores_sorted.txt` under "Exp5" |

---

## 2. Objective

Combine Legendre labels (L) with a pairwise ordering operator (O) in **positional space** (applied to Legendre embeddings, not token embeddings). Tests whether the combination beats distance-only (Exp1) without any explicit proximity bias.

---

## 3. Mathematical Formulation (Derived from Code)

**Source:** [`experiments/exp5_label_order/models/embed.py:135-151`](experiments/exp5_label_order/models/embed.py:135)

```
X'_i = X_i + T_i + P_i + O_i/√d

where:
  O_i = (1/(L-1)) · Σ_{j≠i} (P_i − P_j)    [pairwise all-to-all]
```

**From `ordering_operator.py:38-50`:**
```python
X_i = X.unsqueeze(2)  # [B, L, 1, D]
X_j = X.unsqueeze(1)  # [B, 1, L, D]
delta_x = X_i - X_j   # [B, L, L, D]
# mask diagonal
O = delta_x.sum(dim=2) / (L - 1)
```

**Algebraic simplification:** `(1/(L-1)) · Σ_{j≠i} (P_i − P_j) = P_i − P̄` where `P̄` is the mean Legendre vector. This is equivalent to **mean-centering** the Legendre embeddings. The ordering operator effectively computes how far from the mean each position's Legendre vector is.

**Verified against README-E5.md shell script comment:** `O_i = (1/L-1) * Σ_{j≠i} (P_i - P_j)` ✅

**Verified against README.md main description:** README says "Label + Order (Exp 5): `X'_i = X_i + T_i + P_i + O_i`". ✅ Consistent.

**⚠️ NOTE on ordering_operator formula:** The operator computes `O_i = Σ(P_i−P_j)/(L-1)` which mathematically equals `P_i − mean_P`. This means the "order" signal in Exp5 is actually **Legendre label minus Legendre mean** — a centered version of the label. This is NOT the consecutive-difference ordering signal used in Exp5b or Exp6.

---

## 4. Algorithm Flow

```
X'_i = X_i + T_i + P_i + (P_i − P̄)/√d    [all positions processed together]
         ↓
FullAttention (UNMODIFIED — no distance decay in attention)
         ↓
Standard encoder/decoder pipeline
```

---

## 5. Code Walkthrough

### 5.1 Memory Warning
`OrderingOperator` creates `[B, L, L, D]` tensor (`delta_x = X_i - X_j`). For B=32, L=96, D=512: 32×96×96×512×4 bytes ≈ **1.5 GB** per forward pass. This is the O(L²D) cost warned about in the shell script: "WARNING: O(L²) operator — expect ~50-100s per epoch."

### 5.2 Scaling
```python
ordering_pos = ordering_pos / math.sqrt(value_emb.size(-1))  # / √512
```
This is `ordering_pos / 22.6`. Not mentioned in README formula but present in code.

---

## 6. Result Verification

### Phase 1 (from `mse_mae_scores_sorted.txt`):
| pred_len | seed | MSE | MAE |
|---------|------|-----|-----|
| 96 | 2021 | 0.8519 | 0.7383 |
| 192 | 2021 | 0.9788 | 0.7980 |

### Phase 2 (from `mse_mae_scores_sorted.txt`):
| pred_len | seed | MSE | MAE |
|---------|------|-----|-----|
| 48 | 2021 | 0.8328 | 0.7210 |
| 48 | 2022 | 0.7529 | 0.6773 |
| 48 | 2023 | 0.8448 | 0.7343 |
| 96 | 2021 | 0.8972 | 0.7471 |
| 96 | 2022 | 0.8450 | 0.7387 |
| 96 | 2023 | 0.8545 | 0.7253 |
| 192 | 2021 | 1.1154 | 0.8443 |
| 192 | 2022 | 0.8733 | 0.7359 |
| 192 | 2023 | 0.9154 | 0.7534 |

**README.md summary table (pred_len=96):**
| Source | MSE | MAE |
|--------|-----|-----|
| README.md | 0.719 | 0.635 |
| Central results Phase 1 (pred=96, seed=2021) | 0.8519 | 0.7383 |
| Central results Phase 2 avg (pred=96) | 0.8655 | 0.7370 |

**🔴 CRITICAL INCONSISTENCY:** README.md reports Exp5 as MSE=0.719/MAE=0.635. Central results Phase 1 shows 0.8519/0.7383 and Phase 2 average shows 0.8655/0.7370. The README value 0.719 is **significantly lower** than any logged measurement (~0.13 MSE gap) and does not appear anywhere in the results files.

**Averages check (Phase 2, pred=96):** (0.8972+0.8450+0.8545)/3 = 2.5967/3 = 0.8655. Central file states 0.8655. ✅ Arithmetic correct.

---

## 7. Consistency Checks

### 7.1 Formula
- ✅ `X'_i = X_i + T_i + P_i + O_i` confirmed from code
- ✅ O applied to legendre_pos (positional space), NOT value_emb — confirmed
- ⚠️ `O_i = (1/(L-1))·Σ(P_i−P_j)` is mathematically equivalent to `P_i − P̄` — making the combined signal `X_i + T_i + P_i + (P_i − P̄)/√d = X_i + T_i + P_i(1+1/√d) − P̄/√d`. The L and O signals are NOT independent; they are derived from the same Legendre vectors.

### 7.2 Results
- 🔴 README MSE=0.719 is ~0.133 lower than any logged value — cannot be verified
- ✅ Phase 2 averages internally consistent

---

## 8. Inconsistency Report

### Critical Issues
1. **README MSE=0.719 has no source**: The central results file shows best Phase 1 as 0.8519 and Phase 2 average as 0.8655. The README value 0.719 cannot be traced to any log.
2. **L and O are not independent**: The ordering signal is `P_i − P̄`, derived from the same Legendre vectors as the label `P_i`. They are not two independent positional signals.

### Moderate Issues
3. **Memory footprint**: O(L²D) operator (~1.5 GB) not documented in main README as a limitation.
4. **Scaling factor `/√d` not in README formula**: Code applies `/√512` but formula in README omits this.

---

## 9. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation confidence | 8/10 | Code is clear; algebraic simplification confirms operator logic |
| Documentation confidence | 2/10 | README MSE (0.719) is dramatically lower than any logged value |
| Result confidence | 4/10 | Phase 2 averages internally consistent; README summary value unexplained |
