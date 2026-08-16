# EXPERIMENT 12 AUDIT — Formula-B-pos: Global Mean Deviation Ordering in Positional Space

**Folder:** `experiments/Formula-B-pos/`  
**Audit Date:** 2025  
**Status:** No results logged. Implementation sound. Critical formula detail verified.

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Formula B (Positional Space) — Global Mean Deviation Ordering |
| Folder name | `Formula-B-pos` |
| Code class | `DataEmbedding_formula_b_pos` |
| pe_mode flag | `--pe_mode formula_b_pos` |
| README title | "Formula B (Positional Space) — Global Mean Deviation Ordering" |
| Shell scripts | `formula-B-pos-ph1.sh`, `formula-B-pos-ph2.sh` |
| Notebook | `Order_formula_b_pos (1).ipynb` |

---

## 2. Objective

Apply the same **global mean deviation ordering formula** used in Formula-B-sem but in **Legendre positional space** instead of semantic space:

```
Ordering_i = (μ_p − P_i) / (p̄ + ε)
```

where:
- `P_i` = Legendre polynomial positional embedding (content-independent, fixed buffer)
- `μ_p` = global mean of all Legendre vectors in the sequence
- `p̄` = mean per-token L2 norm of P (scalar normaliser)

This is the **positional-space counterpart of Formula-B-sem**: identical ordering formula, different input space (P vs V).

Key properties:
- **Content-independent** — V_i never enters the ordering computation
- **Deterministic** — P_i is a fixed buffer; μ_p is the same every forward pass for a given L
- **No boundary condition** — μ_p is symmetric over all L positions

---

## 3. Mathematical Formulation

**Implemented formula** (verified from [`embed.py`](experiments/Formula-B-pos/models/embed.py) lines 200–220):

```
X'_i = V_i + T_i + P_i + Ordering_i
```

Where:
```
P_i      = LegendrePositionEmbedding(x)        [B, L, D]   fixed buffer
μ_p      = (1/L) Σ_k P_k                      [B, 1, D]   global mean of P
Δ_p_i    = μ_p − P_i                          [B, L, D]   mean deviation
p̄       = (1/L) Σ_i ||P_i||_2                [B, 1, 1]   scalar normaliser
Ordering_i = Δ_p_i / (p̄ + ε)                [B, L, D],  ε = 1e-8
```

| Symbol | Code | Location |
|--------|------|----------|
| V_i | `val = self.value_embedding(x)` | embed.py:200 |
| T_i | `temp = self.temporal_embedding(x_mark)` | embed.py:201 |
| P_i | `leg = self.legendre_embedding(x)` | embed.py:202 |
| P_i detached | `leg_d = leg.detach()` | embed.py:206 |
| μ_p | `mu_p = leg_d.mean(dim=1, keepdim=True)` | embed.py:209 |
| Δ_p_i | `delta_p = mu_p - leg_d` | embed.py:212 |
| p̄ | `leg_d.norm(dim=-1).mean(dim=1,keepdim=True).unsqueeze(-1)` | embed.py:218 |
| Ordering_i | `ordering = delta_p / (p_bar + 1e-8)` | embed.py:220 |

**Formula verification against README:** README Section 4.1 and Section 6 give the mathematical derivation. Code implements it exactly. ✅

---

## 4. Algorithm Flow

```
Input x [B, L, c_in]
↓
TokenEmbedding → val [B, L, D]                         ← V_i
TemporalEmbedding → temp [B, L, D]                     ← T_i
LegendrePositionEmbedding → leg [B, L, D]              ← P_i (fixed buffer)
↓
leg_d = leg.detach()
mu_p = leg_d.mean(dim=1, keepdim=True)  [B, 1, D]     ← μ_p (broadcasts)
delta_p = mu_p - leg_d                  [B, L, D]      ← Δ_p_i
p_bar = mean(||P_i||_2)                [B, 1, 1]      ← scalar normaliser
ordering = delta_p / (p_bar + 1e-8)   [B, L, D]      ← Ordering_i
↓
output = dropout(val + temp + leg + ordering)
↓
Standard FullAttention encoder
↓
Standard decoder + linear projection
```

---

## 5. The Only Structural Difference from Formula-A-pos

**Two lines in embed.py differ:**

| | Formula-A-pos | Formula-B-pos |
|-|---------------|---------------|
| Code | `delta_p = zeros; delta_p[:, 1:, :] = P_i − P_{i−1}` | `mu_p = P.mean(dim=1,...); delta_p = mu_p − P` |
| Signal | Local (consecutive diff) | Global (mean deviation) |
| Boundary | Zero at i=0 | None (symmetric) |

Everything else — normaliser `p_bar`, `ordering`, output sum, all hyperparameters — is bit-for-bit identical. ✅

---

## 6. Code Walkthrough

### `embed.py` — `DataEmbedding_formula_b_pos`

- Both `DataEmbedding` (vanilla) and `DataEmbedding_formula_b_pos` coexist.
- Legendre imported via `sys.path.insert` + `from legendre_embedding import LegendrePositionEmbedding`
- Diagnostic step counter prints 5 statistics every 100 steps.
- No `scaling` argument passed to `LegendrePositionEmbedding` — uses default. Verify default is `scaling=False` or `True` in the Legendre file. Formula-A-pos also does not pass scaling.

### `model.py` — dispatch

```python
if pe_mode == 'vanilla':       → DataEmbedding
elif pe_mode == 'formula_b_pos': → DataEmbedding_formula_b_pos
else:                          → raise ValueError
```
Both `Informer` and `InformerStack` correct. ✅

---

## 7. Configuration Audit

From [`formula-B-pos-ph1.sh`](experiments/Formula-B-pos/formula-B-pos-ph1.sh):

| Parameter | Script value | README value | Match |
|-----------|-------------|-------------|-------|
| attn | full | full | ✅ |
| embed | timeF | timeF | ✅ |
| d_ff | 2048 | 2048 | ✅ |
| seq_len | 96 | 96 | ✅ |
| label_len | 48 | 48 | ✅ |
| pe_mode | formula_b_pos | formula_b_pos | ✅ |
| pred_len | 96, 192 (ph1) | 24–720 | ⚠️ ph1 subset |

Reference values identical to other Formula experiments (Exp1-Pre: MSE=0.8683 at pred=96). ✅

---

## 8. Result Verification

| Source | MSE | MAE |
|--------|-----|-----|
| mse_mae_scores_sorted.txt | Not present | Not present |
| Notebook | Not run | Not run |

**No results. Experiment not executed.**

---

## 9. Cross-Formula Comparison Table

| Experiment | Input space | Delta type | Label P_i added | Normalised |
|------------|-------------|-----------|----------------|-----------|
| Formula-A-sem | Semantic (V) | Consecutive | No | No |
| Formula-B-sem | Semantic (V) | Global mean | No | Yes |
| Formula-A-pos | Positional (P) | Consecutive | Yes | Yes (p_bar) |
| **Formula-B-pos** | **Positional (P)** | **Global mean** | **Yes** | **Yes (p_bar)** |

The 2×2 grid (A/B × sem/pos) is complete.

---

## 10. Inconsistency Report

### Critical Issues
None.

### Moderate Issues
1. **No results** — experiment not executed

### Minor Issues
1. **Notebook filename has space and parentheses** — `Order_formula_b_pos (1).ipynb` suggests a Colab copy artifact
2. **Dynamic import inside `__init__`** — same fragile pattern as Formula-A-pos

---

## 11. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation | 9/10 | Clean; formula matches README; dispatch correct |
| Documentation | 9/10 | Excellent README with mathematical derivation and cross-experiment comparison |
| Results | 0/10 | No results |
