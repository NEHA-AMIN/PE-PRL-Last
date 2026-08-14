# EXPERIMENT 11 AUDIT — Formula-B-sem: Global Mean Deviation Ordering in Semantic Space

**Folder:** `experiments/Formula-B-sem/`  
**Audit Date:** 2025  
**Status:** No results logged. Best-documented experiment in repository.

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Formula B (Semantic Space) — Global Mean Deviation Ordering |
| Folder name | `Formula-B-sem` |
| Code class | `DataEmbedding_formula_b_sem` |
| pe_mode flag | `--pe_mode formula_b_sem` |
| README title | "Formula B — Global Mean Deviation Ordering in Semantic Space" |
| Shell scripts | `formula-B-sem-ph1.sh`, `formula-B-sem-ph2.sh` |
| Notebook | `Order_Forumla_B_sem.ipynb` (note: typo in filename "Forumla") |

---

## 2. Objective

Replace sinusoidal PE with a **global mean deviation ordering signal** in semantic (token embedding) space. Unlike Formula A (which encodes local change by comparing adjacent tokens), Formula B encodes **global position** by measuring how far each token sits from the sequence-level centroid:

```
Ordering_i = (μ − V_i) / (x̄ + ε)
```

where μ is the global mean of all token embeddings in the window, and x̄ is the mean per-token L2 norm (scalar normaliser).

Formula B is the **normalised** counterpart of Formula A. Formula A encodes local dynamics; Formula B encodes global deviation.

---

## 3. Mathematical Formulation

**Implemented formula** (verified from [`embed.py`](experiments/Formula-B-sem/models/embed.py) lines 253–269):

```
X'_i = V_i + T_i + Ordering_i
```

Where:
```
V_i          = TokenEmbedding(x_i)           [B, L, D]
μ            = (1/L) Σ_k V_k                 [B, 1, D]   global mean
Δ_i          = μ − V_i                       [B, L, D]   deviation
x̄           = (1/L) Σ_i ||V_i||_2           [B, 1, 1]   scalar normaliser
Ordering_i   = Δ_i / (x̄ + ε)               [B, L, D],  ε = 1e-8
```

| Symbol | Code | Location |
|--------|------|----------|
| V_i | `val = self.value_embedding(x)` | embed.py:253 |
| T_i | `temp = self.temporal_embedding(x_mark)` | embed.py:256 |
| μ | `global_mean = val.mean(dim=1, keepdim=True)` | embed.py:260 |
| Δ_i | `delta = global_mean - val` | embed.py:263 |
| x̄ | `val.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)` | embed.py:266 |
| Ordering_i | `ordering = delta / (x_bar + 1e-8)` | embed.py:269 |

**Formula verification against README:** README Section 4.1 states the formula precisely. Code implements it exactly. ✅

**Key properties verified from code:**
- **No sinusoidal PE** — `PositionalEmbedding` is present in file but NOT initialised in `DataEmbedding_formula_b_sem.__init__` ✅
- **No Legendre embedding** — not imported, not used ✅
- **Translation-invariant** — adding constant c to all V_i cancels in μ − V_i ✅ (verified algebraically)
- **Scale-invariant** — scaling all V_i by α → (αμ − αV_i)/(αx̄) = (μ − V_i)/x̄ ✅ (verified algebraically, ignoring ε)
- **No new parameters** — x̄ and μ computed from existing TokenEmbedding output ✅

---

## 4. Algorithm Flow

```
Input x [B, L, c_in]
↓
TokenEmbedding → val [B, L, D]                         ← V_i
TemporalEmbedding → temp [B, L, D]                     ← T_i
↓
global_mean = val.mean(dim=1, keepdim=True) [B, 1, D]  ← μ (broadcasts)
delta = global_mean - val              [B, L, D]        ← Δ_i = μ − V_i
x_bar = val.norm(...).mean(...) [B, 1, 1]               ← x̄ (scalar per sample)
ordering = delta / (x_bar + 1e-8)     [B, L, D]        ← Ordering_i
↓
output = dropout(val + temp + ordering)
↓
Standard FullAttention encoder (no distance decay, no Legendre)
↓
Standard decoder + linear projection
```

---

## 5. Code Walkthrough

### `embed.py` — `DataEmbedding_formula_b_sem`

- The most thoroughly documented embedding class in the repository.
- `DataEmbedding` (vanilla) is retained in the same file for `pe_mode='vanilla'` dispatch.
- Diagnostic step counter (`_diag_step`) prints 5 signal statistics every 100 training steps: `val_norm`, `ordering_norm`, `temp_norm`, `x_bar`, `ordering/val` ratio. This provides live monitoring of signal balance.
- No external dependencies — entirely self-contained.

### `attn.py` — Standard unmodified

Identical to baseline, Formula-A-sem, Formula-A-pos. No modifications. ✅

### `model.py` — dispatch

```python
if pe_mode == 'vanilla':       → DataEmbedding
elif pe_mode == 'formula_b_sem': → DataEmbedding_formula_b_sem
else:                          → raise ValueError
```
Both `Informer` and `InformerStack` implement this correctly. ✅

---

## 6. Formula-B-sem vs Formula-A-sem Comparison

| Dimension | Formula-A-sem | Formula-B-sem |
|-----------|---------------|---------------|
| Signal type | Local (consecutive diff) | Global (deviation from mean) |
| Formula | Δ(X_i) = X_i − X_{i−1} | Ordering_i = (μ − V_i) / (x̄ + ε) |
| Normalised | No | Yes |
| Boundary condition | Yes (i=0: Δ = X_0) | None (mean is symmetric) |
| Legendre | No | No |
| No. parameters added | 0 | 0 |

---

## 7. Mathematical Properties: Scale-Invariance Proof

For Formula-B-sem, let V'_i = α·V_i (scale all embeddings by α):
- μ' = α·μ
- Δ'_i = α·μ − α·V_i = α·Δ_i
- x̄' = α·x̄
- Ordering'_i = α·Δ_i / (α·x̄ + ε)

As α·x̄ >> ε (i.e., typical embedding scales >> 1e-8):
```
Ordering'_i ≈ Δ_i / x̄ = Ordering_i
```
Scale-invariant. ✅

---

## 8. Configuration Audit

From [`formula-B-sem-ph1.sh`](experiments/Formula-B-sem/formula-B-sem-ph1.sh):

| Parameter | Script value | README value | Match |
|-----------|-------------|-------------|-------|
| attn | full | full | ✅ |
| embed | timeF | timeF | ✅ |
| d_ff | 2048 | 2048 | ✅ |
| seq_len | 96 | 96 | ✅ |
| label_len | 48 | 48 | ✅ |
| pe_mode | formula_b_sem | formula_b_sem | ✅ |
| pred_len | 96, 192 (ph1) | 24–720 | ⚠️ ph1 subset |

Reference: same Exp1-Pre values used as threshold (MSE=0.8683 at pred=96).

---

## 9. Notebook Filename Typo

The notebook is named `Order_Forumla_B_sem.ipynb` — "Forumla" instead of "Formula". Minor naming error.

---

## 10. Result Verification

| Source | MSE | MAE |
|--------|-----|-----|
| mse_mae_scores_sorted.txt | Not present | Not present |
| Notebook | Not run | Not run |

**No results. Experiment not executed.**

---

## 11. Inconsistency Report

### Critical Issues
None.

### Moderate Issues
1. **No results** — experiment not executed

### Minor Issues
1. **Notebook filename typo** — `Order_Forumla_B_sem.ipynb`

---

## 12. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation | 10/10 | Best-documented embedding class in repository; formula matches README precisely; diagnostics included |
| Documentation | 9/10 | Excellent README with proofs and hypothesis framing; only a typo in notebook filename |
| Results | 0/10 | No results |
