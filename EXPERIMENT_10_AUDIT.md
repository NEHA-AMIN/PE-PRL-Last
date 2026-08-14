# EXPERIMENT 10 AUDIT — Formula-A-pos: Consecutive Legendre Delta Ordering in Positional Space

**Folder:** `experiments/Formula-A-pos/`  
**Audit Date:** 2025  
**Status:** No results logged. Implementation sound. Formula verified.

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Formula A (Positional Space) — Consecutive Legendre Delta |
| Folder name | `Formula-A-pos` |
| Code class | `DataEmbedding_ordering_pos` |
| pe_mode flag | `--pe_mode ordering_pos` |
| README title | "Experiment 5 Revised — Ordering in Positional Space" |
| Shell scripts | `formula-A-pos-ph1.sh`, `Formula-A-pos-ph2.sh` |

---

## 2. Objective

Compute a consecutive-delta ordering signal in **Legendre positional space** rather than semantic (token embedding) space. The delta is:

```
Δ_i^leg = 0              if i = 0    (zero-pad boundary)
           P_i − P_{i−1} if i ≥ 1
```

where `P_i` is the Legendre polynomial positional embedding (a fixed non-trainable buffer, content-independent). This is then normalised by the mean L2 norm of the positional vectors and added to the embedding alongside the label P_i itself.

**Key distinction from Formula-A-sem:** The delta is computed on `P` (fixed Legendre positions), NOT on `X` (content-dependent token embeddings). The signal is therefore completely content-independent.

---

## 3. Mathematical Formulation

**Implemented formula** (verified from [`embed.py`](experiments/Formula-A-pos/models/embed.py) lines 163–183):

```
X'_i = X_i + T_i + P_i + O_i^pos
```

Where:
```
P_i      = LegendrePositionEmbedding(x)          [B, L, D]   fixed buffer
delta_p_i = 0            if i = 0
             P_i − P_{i−1}  if i ≥ 1             [B, L, D]
p_bar    = (1/L) Σ_i ||P_i||_2                  [B, 1, 1]   scalar
O_i^pos  = delta_p_i / (p_bar + 1e-8)           [B, L, D]
```

| Symbol | Definition | Code | Location |
|--------|-----------|------|----------|
| X_i | TokenEmbedding | `val = self.value_embedding(x)` | embed.py:165 |
| T_i | TemporalEmbedding | `temp = self.temporal_embedding(x_mark)` | embed.py:166 |
| P_i | LegendrePositionEmbedding | `leg = self.legendre_embedding(x)` | embed.py:167 |
| delta_p | zeros, then P_i−P_{i−1} | lines 172–173 | embed.py |
| p_bar | mean L2 norm of P | `leg_d.norm(dim=-1).mean(dim=1)...` | embed.py:179 |
| O_i^pos | delta_p / (p_bar + 1e-8) | `ordering = delta_p / (p_bar + 1e-8)` | embed.py:181 |

**Formula verification against README:** README formula matches code exactly. ✅

**Legendre buffer is detached** (`leg_d = leg.detach()`) before computing delta and p_bar, ensuring no gradient flows through the positional buffer. ✅

---

## 4. Algorithm Flow

```
Input x [B, L, c_in]
↓
TokenEmbedding → val [B, L, D]                    ← X_i (semantic)
TemporalEmbedding → temp [B, L, D]                ← T_i (calendar)
LegendrePositionEmbedding → leg [B, L, D]         ← P_i (positional label, fixed)
↓
leg_d = leg.detach()
delta_p[:, 0, :] = 0                              ← zero boundary
delta_p[:, 1:, :] = leg_d[:, 1:, :] - leg_d[:, :-1, :]  ← consecutive Legendre delta
↓
p_bar = mean(||P_i||_2) → [B, 1, 1]              ← scalar normaliser
ordering = delta_p / (p_bar + 1e-8)              ← O_i^pos
↓
output = dropout(val + temp + leg + ordering)
↓
Standard FullAttention encoder
↓
Standard decoder + linear projection
```

---

## 5. Code Walkthrough

### `embed.py` — `DataEmbedding_ordering_pos`

- Legendre embedding is imported via `sys.path.insert` + `from legendre_embedding import LegendrePositionEmbedding` (dynamic import pattern used in earlier experiments)
- `scaling=True` is NOT passed (default). In earlier Legendre implementations, scaling divides by `sqrt(d_model)`. Need to verify Legendre file.
- Both `DataEmbedding` (vanilla) and `DataEmbedding_ordering_pos` are present in the same file.

### `model.py` — dispatch

Dispatch on `pe_mode`:
- `'vanilla'` → `DataEmbedding`  
- `'ordering_pos'` → `DataEmbedding_ordering_pos`  
- anything else → `raise ValueError`

Both `Informer` and `InformerStack` implement this correctly. ✅

### Legendre file

`experiments/Formula-A-pos/models/legendre_embedding.py` is present. Import inside `__init__` uses `sys.path.insert(0, os.path.dirname(__file__))` — this is the same dynamic import used across Exp3, Exp5, Exp5b, Exp6. The shell script explicitly copies it: `cp legendre_embedding.py ...`

---

## 6. Formula-A-pos vs Formula-A-sem Comparison

| Dimension | Formula-A-sem | Formula-A-pos |
|-----------|---------------|---------------|
| Delta input | X_i = TokenEmbedding (content-dependent) | P_i = Legendre (content-independent) |
| Legendre label added | No | Yes (P_i added directly) |
| Normaliser | Not normalised (raw delta) | p_bar = mean(||P_i||_2) |
| Formula | X'_i = X_i + Δ(X_i) + T_i | X'_i = X_i + T_i + P_i + O_i^pos |
| Boundary at i=0 | delta = X_0 (first token content) | delta = 0 (zero pad) |

The two experiments differ in both the input space of the delta and the presence of a normaliser and Legendre label. They are **not directly comparable** as single-variable ablations.

---

## 7. Configuration Audit

From [`formula-A-pos-ph1.sh`](experiments/Formula-A-pos/formula-A-pos-ph1.sh):

| Parameter | Script value | README value | Match |
|-----------|-------------|-------------|-------|
| attn | full | full | ✅ |
| embed | timeF | timeF | ✅ |
| seq_len | 96 | 96 | ✅ |
| label_len | 48 | 48 | ✅ |
| pred_len | 96, 192 (ph1) | 24–720 (full) | ⚠️ ph1 subset |
| d_ff | 2048 | 2048 | ✅ |
| pe_mode | ordering_pos | ordering_pos | ✅ |

Reference comparison values in script:
- `pred=96  MSE=0.8683` (Exp1-Pre)
- `pred=192 MSE=0.8463`

These are the same reference values used in ALL four Formula experiment scripts, providing a consistent decision threshold. ✅

---

## 8. Result Verification

| Source | MSE | MAE |
|--------|-----|-----|
| mse_mae_scores_sorted.txt | Not present | Not present |
| Notebooks | Not run | Not run |

**No results. Experiment not executed.**

---

## 9. Inconsistency Report

### Critical Issues
None.

### Moderate Issues
1. **README title wrong** — "Experiment 5 Revised — Ordering in Positional Space" (should be "Formula-A-pos" or consistent with the new naming scheme; "Exp 5 Revised" does not appear elsewhere)
2. **README refers to wrong folder** — "Folder: `experiments/exp5_ordering_new_pos_space/`" — this folder does not exist. Correct path is `experiments/Formula-A-pos/`
3. **No results** — experiment not executed

### Minor Issues
1. **Dynamic import inside `__init__`** — `sys.path.insert` inside a constructor is fragile. Works but unusual.

---

## 10. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation | 9/10 | Clean implementation; formula and dispatch are correct |
| Documentation | 4/10 | README folder path wrong; experiment name inconsistent |
| Results | 0/10 | No results |
