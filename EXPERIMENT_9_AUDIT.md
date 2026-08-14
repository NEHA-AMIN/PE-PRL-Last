# EXPERIMENT 9 AUDIT — Formula-A-sem: Consecutive Delta Ordering in Semantic Space

**Folder:** `experiments/formula-A-sem/`  
**Audit Date:** 2025  
**Status:** No results logged. Implementation internally consistent. Notable naming confusion.

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Formula A (Semantic Space) — Delta Ordering |
| Folder name | `formula-A-sem` (lowercase f, mixed case) |
| Code class | `DataEmbedding_delta_pos` |
| pe_mode flag | `--pe_mode delta_pos` |
| README title | "Experiment 4c — Delta Ordering in Positional Space" |
| Shell script name | `formula-A-sem-ph1.sh`, `formula-A-sem-ph2.sh` |

---

## 2. Objective

Replace sinusoidal PE with a **consecutive delta ordering signal** built from the token (value) embeddings. The signal is the first-order difference between adjacent token representations:

```
Δ(X_i) = X_i − X_{i−1}   for i ≥ 1
Δ(X_0) = X_0              for i = 0  (zero-pad boundary)
```

Research question: Can local directional change between consecutive embedding vectors replace absolute positional encoding?

---

## 3. Mathematical Formulation

**Implemented formula** (verified from [`embed.py`](experiments/formula-A-sem/models/embed.py) lines 248–250):

```
X'_i = X_i + Δ(X_i) + T_i
```

| Symbol | Definition | Implementation | Location |
|--------|-----------|---------------|----------|
| X_i | TokenEmbedding (Conv1D circular) | `val = self.value_embedding(x)` | embed.py:241 |
| T_i | TemporalEmbedding | `temp = self.temporal_embedding(x_mark)` | embed.py:244 |
| Δ(X_0) | X_0 (boundary) | `delta[:, 0, :] = val[:, 0, :]` | embed.py:249 |
| Δ(X_i) | X_i − X_{i−1} for i≥1 | `delta[:, 1:, :] = val[:, 1:, :] - val[:, :-1, :]` | embed.py:250 |

**Formula verification against README:**
README states: `X'_i = X_i + Δ(X_i) + T_i` ✅ MATCHES CODE

**Note on delta at i=0:** The boundary condition `Δ(X_0) = X_0` means the first token's positional signal equals its own content. README explicitly documents this as a known, bounded artefact.

---

## 4. Algorithm Flow

```
Input x [B, L, c_in]
↓
TokenEmbedding (Conv1D) → val [B, L, D]          ← X_i
↓
TemporalEmbedding → temp [B, L, D]               ← T_i
↓
delta = zeros_like(val)
delta[:, 0, :] = val[:, 0, :]                    ← Δ(X_0) = X_0
delta[:, 1:, :] = val[:, 1:, :] - val[:, :-1, :] ← Δ(X_i) = X_i − X_{i−1}
↓
output = dropout(val + delta + temp)
↓
Standard FullAttention encoder (no distance decay, no Legendre)
↓
Standard decoder + linear projection
```

---

## 5. Code Walkthrough

### `embed.py` — `DataEmbedding_delta_pos`

- No Legendre buffer, no distance operator. Entirely self-contained.
- Diagnostic step counter (`_diag_step`) prints signal magnitude ratios every 100 training steps to monitor whether delta dominates or vanishes relative to val. This is intentional monitoring for the scale-sensitivity hypothesis.
- The class `DataEmbedding` (vanilla, unchanged) is also present in the same file, preserved for `pe_mode='vanilla'` dispatch.

### `attn.py` — Standard unmodified FullAttention/ProbAttention

No modifications. Verified identical to Exp3b, Formula-B-sem, baseline.

### `model.py` — `pe_mode` dispatch

Imports: `DataEmbedding`, `DataEmbedding_delta_pos`  
Dispatch:
```python
if pe_mode == 'vanilla':     → DataEmbedding
elif pe_mode == 'delta_pos': → DataEmbedding_delta_pos
else:                        → raise ValueError
```
Both `Informer` and `InformerStack` implement this dispatch correctly. ✅

---

## 6. Critical Naming Inconsistency: Folder Name vs README vs Class Name

| Identifier | Value | Space indicated |
|-----------|-------|----------------|
| Folder name | `formula-A-sem` | Semantic space |
| README title | "Experiment 4c — Delta Ordering **in Positional Space**" | **Positional** space |
| Class name | `DataEmbedding_delta_pos` | **Positional** (by name) |
| pe_mode flag | `delta_pos` | **Positional** (by name) |
| Actual formula | `X_i + Δ(X_i) + T_i` where X_i = TokenEmbedding | **Semantic** (by operation) |

**Critical finding:** The class name, pe_mode flag, and README title all say "positional" but the actual operation is performed on **semantic (token) embeddings** (X_i = TokenEmbedding output, not Legendre). The folder name `formula-A-sem` is correct. All other identifiers are misleading.

This is a **persistent naming confusion** from earlier experiments. The formula operates in semantic space. The `_pos` suffix in class/flag names is a historical misnaming.

---

## 7. Configuration Audit

From [`formula-A-sem-ph1.sh`](experiments/formula-A-sem/formula-A-sem-ph1.sh):

| Parameter | Script value | README value | Match |
|-----------|-------------|-------------|-------|
| attn | full | full | ✅ |
| embed | timeF | timeF | ✅ |
| seq_len | 96 | 96 | ✅ |
| label_len | 48 | 48 | ✅ |
| pred_len | 96, 192 (ph1) | 24/48/96/192/336/720 | ⚠️ ph1 subset only |
| d_ff | 2048 | 2048 | ✅ |
| batch_size | 32 | 32 | ✅ |
| seed | 2021 | 2021 | ✅ |
| pe_mode | delta_pos | delta_pos | ✅ |

Phase 1 runs only pred_len 96 and 192. Full sweep in Phase 2 (separate script).

---

## 8. Result Verification

| Source | MSE | MAE |
|--------|-----|-----|
| mse_mae_scores_sorted.txt | Not present | Not present |
| Notebook | Not run | Not run |

**No results exist. Experiment not yet executed.**

---

## 9. Inconsistency Report

### Critical Issues
1. **Naming confusion: `_pos` suffix used for a semantic-space operation** — `DataEmbedding_delta_pos`, `pe_mode='delta_pos'`, and README title "in Positional Space" all contradict the folder name `formula-A-sem` and the actual operation (delta on TokenEmbedding = semantic content)

### Moderate Issues
1. **README title wrong** — "Experiment 4c — Delta Ordering in Positional Space" should read "in Semantic Space"
2. **No results** — experiment has not been run

### Minor Issues
1. **Shell script reference wrong in README** — README says `bash experiments/exp4c_delta_ordering_pos_space/exp4c_delta_ordering_pos_space_ph1.sh` but that path does not exist; correct path is `bash experiments/formula-A-sem/formula-A-sem-ph1.sh`

---

## 10. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation | 9/10 | Code is clean, formula is correctly implemented, dispatch works |
| Documentation | 4/10 | README title wrong (says positional, means semantic); path reference wrong |
| Results | 0/10 | No results exist |
