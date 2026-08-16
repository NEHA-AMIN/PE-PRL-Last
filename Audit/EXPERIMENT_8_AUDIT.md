# EXPERIMENT 8 AUDIT — Exp 3b: Label + Temporal Controlled

**Folder:** `experiments/E-96-3b-Label-Temporal-Controlled/`  
**Audit Date:** 2025  
**Status:** No results logged. Experiment design is sound. Critical model.py bug found.

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Exp 3b: Label + Temporal Controlled |
| Folder name | `E-96-3b-Label-Temporal-Controlled` |
| Code identifier | `DataEmbedding` (modified) |
| pe_mode flag | None — no `pe_mode` dispatch in this experiment |
| README name | "Experiment 3b: Label + Temporal Controlled" |
| Notebook | `exp3b.ipynb` |

---

## 2. Objective

**Exp 3b is a controlled confound-isolation experiment.** Exp 3 (Label Only) removed BOTH sinusoidal PE AND temporal embedding simultaneously, making it impossible to determine whether the degraded MSE (~1.124 vs vanilla ~0.519) was caused by Legendre being a poor label OR by missing temporal context.

Exp 3b fixes this by restoring temporal embedding, creating a **single-variable comparison** between Exp 3b and vanilla:
- **Vanilla:** sinusoidal PE + temporal embedding
- **Exp 3b:** Legendre label + temporal embedding (no sinusoidal PE)

This cleanly isolates the PE type as the only variable.

---

## 3. Mathematical Formulation

**Implemented formula** (verified from [`embed.py`](experiments/E-96-3b-Label-Temporal-Controlled/models/embed.py) lines 134–139):

```
X'_i = X_i + T_i + P_i
```

| Symbol | Definition | Implementation | Location |
|--------|-----------|---------------|----------|
| X_i | TokenEmbedding (Conv1D circular) | `self.value_embedding(x)` | embed.py:134 |
| T_i | TemporalEmbedding (hour/weekday/day/month) | `self.temporal_embedding(x_mark)` | embed.py:135 |
| P_i | LegendrePositionEmbedding(x) | `self.legendre_embedding(x)` | embed.py:136 |

**Formula verification against README:**
- README states: `X'_i = X_i + T_i + P_i` ✅ MATCHES CODE

**Key difference from Exp 3:**
- Exp 3: `X'_i = X_i + P_i` (no temporal)
- Exp 3b: `X'_i = X_i + T_i + P_i` (temporal restored)

---

## 4. Algorithm Flow

```
Input x [B, L, c_in]
↓
TokenEmbedding (Conv1D, circular) → [B, L, D]    ← X_i
↓
LegendrePositionEmbedding (fixed buffer) → [B, L, D]  ← P_i
↓
TemporalEmbedding (hour/weekday/day/month) → [B, L, D]  ← T_i
↓
X'_i = X_i + T_i + P_i
↓
Dropout
↓
Standard FullAttention encoder (no distance decay, no ordering)
↓
Standard decoder
↓
Linear projection → prediction
```

---

## 5. Code Walkthrough

### `embed.py` — `DataEmbedding` class

The class is named `DataEmbedding` (same as vanilla) but its implementation is **modified**. The `__init__` method:
- Initialises `self.value_embedding` (unchanged from vanilla)
- Initialises `self.temporal_embedding` (RESTORED — key change vs Exp 3)
- Initialises `self.legendre_embedding` (LegendrePositionEmbedding, fixed buffer)
- Does **NOT** initialise `self.position_embedding` (sinusoidal PE removed)

The `forward` method combines: `value_emb + temporal_emb + legendre_pos`

### `attn.py` — Standard unmodified FullAttention/ProbAttention

Verified: no distance decay, no delta-V, no ordering. Identical to Exp 3 and Exp3b.

### `model.py` — **CRITICAL BUG**

[`model.py`](experiments/E-96-3b-Label-Temporal-Controlled/models/model.py) imports `DataEmbedding` from `embed.py` and uses it directly with NO `pe_mode` dispatch. This means:
- The **modified** `DataEmbedding` (with Legendre + Temporal) is ALWAYS used
- There is no way to run vanilla mode from this model.py
- The `pe_mode` parameter is **not present** in the constructor signature
- `InformerStack.forward()` treats the embedding as returning a single tensor — this is correct for this experiment (unlike Exp5b/Exp6 which return tuples)

**No pe_mode support in this experiment.** The shell script (referenced in README as `run_exp3b.sh`) does not exist in the folder — this is a missing artifact.

---

## 6. Configuration Audit

No shell script exists in this folder. The README specifies:

| Parameter | README value | Verified |
|-----------|-------------|---------|
| seq_len | 96 | No script to verify |
| label_len | 48 | No script to verify |
| d_model | 512 | No script to verify |
| n_heads | 8 | No script to verify |
| d_ff | 2048 | No script to verify |
| attn | full | No script to verify |
| embed | timeF | No script to verify |
| pred_len | 48/96/192/336/720 | No script to verify |
| seeds | 2021/2022/2023 | No script to verify |

**Finding: No launch script exists.** The README refers to `bash experiments/E-96-3b-Label-Temporal-Controlled/run_exp3b.sh` but this file is absent.

---

## 7. Notebook Audit

`exp3b.ipynb` is present but was not run. No results are logged anywhere.

---

## 8. Result Verification

| Source | MSE | MAE |
|--------|-----|-----|
| README | TBD (marked as unknown) | TBD |
| Notebook | Not run | Not run |
| mse_mae_scores_sorted.txt | Not present | Not present |

**No results exist. This experiment has not been executed.**

---

## 9. Consistency Checks

### Naming consistency
- Folder: `E-96-3b-Label-Temporal-Controlled` — the `E-96` prefix is unusual; not consistent with `exp3`, `exp5b` naming convention. **MODERATE INCONSISTENCY**

### Formula consistency
- README formula `X'_i = X_i + T_i + P_i` matches implementation exactly. ✅

### Architecture consistency
- The `model.py` hardcodes `DataEmbedding` (the modified version). No conditional dispatch. This prevents running vanilla from this model, but since there is no run script, this is a moot issue in practice.

### Missing artifact
- `run_exp3b.sh` referenced in README does not exist. **MODERATE ISSUE**

---

## 10. Inconsistency Report

### Critical Issues
None that would invalidate results (no results exist to invalidate).

### Moderate Issues
1. **Missing run script** — `run_exp3b.sh` referenced in README does not exist in folder
2. **Folder naming inconsistency** — `E-96-3b` prefix does not follow any established convention in this repository
3. **No pe_mode dispatch** — the model.py cannot run vanilla mode; but this is by design for a single-experiment model

### Minor Issues
1. **Notebook not run** — `exp3b.ipynb` has no outputs

---

## 11. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation | 9/10 | Code is clean, formula matches README, Legendre correctly restored temporal |
| Documentation | 6/10 | README is detailed and scientifically sound; missing run script |
| Results | 0/10 | No results exist — experiment never executed |
