# EXPERIMENT_1_AUDIT.md — Exp1-Pre
## Distance Decay Applied Pre-Softmax
**Audit Status:** COMPLETE  
**Evidence Source:** Direct code reading of `exp1_distance_pre_softmax_decay/models/`, `README-E1.md`, `mse_mae_scores_sorted.txt`

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Exp 1 — Distance Decay (Pre-Softmax) |
| Folder | `experiments/exp1_distance_pre_softmax_decay/` |
| pe_mode | N/A — uses experiment-local model.py (no pe_mode dispatch) |
| README identifier | "Exp 1 (D pre)", "Exp1-Pre" |
| Notebook files | `exp1_phase1_pre.ipynb` |
| Shell scripts | No local phase script found in directory listing; notebook used for execution |
| Result directory | `results/exp1_distance_decay/` (NAME MISMATCH — folder says "decay", not "pre_softmax_decay") |
| Results in central file | `mse_mae_scores_sorted.txt` under "Exp1-Pre" header |

---

## 2. Objective

Remove sinusoidal positional encoding from the Informer embedding stack. Replace it with a proximity bias applied **inside the attention mechanism before softmax**. Tests whether index-distance alone is sufficient positional information.

**Hypothesis:** `scores ← scores ⊙ α` pre-softmax biases attention toward local context, providing positional signal without explicit embedding.

---

## 3. Mathematical Formulation (Derived from Code)

### Embedding Formula
**Source:** [`experiments/exp1_distance_pre_softmax_decay/models/embed.py:107-110`](experiments/exp1_distance_pre_softmax_decay/models/embed.py:107)
```python
# EXPERIMENT 1: REMOVE POSITIONAL EMBEDDING
x = self.value_embedding(x) + self.temporal_embedding(x_mark)
```

```
X'_i = X_i + T_i
  (sinusoidal PE removed; temporal embedding RETAINED)
```

### Attention Decay Formula
**Source:** [`experiments/exp1_distance_pre_softmax_decay/models/attn.py:28-35`](experiments/exp1_distance_pre_softmax_decay/models/attn.py:28)
```python
q_idx = torch.arange(L).unsqueeze(1)
k_idx = torch.arange(S).unsqueeze(0)
dist_matrix = torch.abs(q_idx - k_idx).float()
alpha = 1.0 / (1.0 + dist_matrix ** self.decay_a)
scores = scores * alpha.unsqueeze(0).unsqueeze(0)
```

**Formula:**
```
α(i,j) = 1 / (1 + |i−j|^a)
scores_ij = α(i,j) · (Q_i · K_j)  [element-wise, pre-softmax]
A = softmax(scale · scores)
```

### Symbol Table

| Symbol | Definition | Location | Lines |
|--------|-----------|----------|-------|
| α(i,j) | Distance decay weight between query position i and key position j | `attn.py` | 28-32 |
| a | Decay exponent (`decay_a` parameter, default 1.0) | `model.py:16`, `attn.py:17` | — |
| `|i−j|` | Absolute index distance | `attn.py` | 30 |
| scores | Raw QKᵀ attention logits | `attn.py` | 24 |
| scale | 1/√E, E = d_keys per head | `attn.py` | 21 |

**Verified against README-E1.md:** ✅ README states `α(i,j) = 1/(1+|i-j|^a)` and "Multiply attention scores by decay". Both confirmed from code.

**Verified against TECHNICAL_OBSERVATIONS.md:** ✅ Technical doc correctly states "Pre-Softmax: `scores = scores * alpha` (line 35)". Actual line is 35. Confirmed.

---

## 4. Algorithm Flow

```
Raw ETTh1 [B, 96, 7] + time marks
         ↓
TokenEmbedding + TemporalEmbedding (NO sinusoidal PE)
         ↓ X'_i = X_i + T_i
FullAttention Encoder:
   scores = Q·Kᵀ                            [B, H, L, S]
   dist_matrix = |i-j|                       [L, S]  (computed every forward pass)
   alpha = 1/(1+dist_matrix^decay_a)         [L, S]
   scores = scores ⊙ alpha                   [B, H, L, S]  ← PRE-softmax
   A = softmax(scale · scores)               [B, H, L, S]
   V_out = A · V                             [B, L, H, D]
         ↓
Decoder (same embedding formula, FullAttention masked self-attn)
         ↓
Linear projection → MSE/MAE
```

---

## 5. Code Walkthrough

### 5.1 `embed.py` — `DataEmbedding.forward()` (Lines 107-111)
- Identical class name to baseline (`DataEmbedding`) but different `forward()` — no `self.position_embedding` call
- `position_embedding` is still instantiated in `__init__` (line 101) but never called in `forward()` — **dead code / dead weight**

### 5.2 `attn.py` — `FullAttention.forward()` (Lines 24-45)
- `__init__` signature: `decay_a=1.0` (line 11)
- Distance matrix computed **every forward pass** (not cached) — minor efficiency issue for training
- `alpha.unsqueeze(0).unsqueeze(0)` broadcasts from `[L, S]` to `[1, 1, L, S]` then to `[B, H, L, S]`
- Applied BEFORE `masked_fill_` with `-inf` (causal mask applied after decay)
- **ORDER MATTERS:** The causal mask is applied after decay, meaning masked positions get `alpha · score = alpha · (-inf) = -inf` — still correctly excluded from softmax ✅

### 5.3 `model.py` — `Informer.__init__()` (Lines 16, 31, 50-52)
- `decay_a` passed to encoder's `FullAttention` AND decoder's self-attention AND decoder's cross-attention `FullAttention`
- **Note:** Distance decay is thus active in ALL three attention sublayers (encoder self, decoder self, decoder cross)
- `ProbAttention.__init__` does NOT accept `decay_a` — but model.py passes `decay_a=decay_a` to `Attn(...)` where `Attn = ProbAttention if attn=='prob' else FullAttention`. Since experiments use `--attn full`, only FullAttention is instantiated. ✅

### 5.4 Decoder Distance Decay
- Both decoder self-attention and cross-attention get `decay_a=decay_a` passed (model.py lines 50-52)
- This means distance decay is active in the **decoder cross-attention** between decoder positions and encoder positions
- This is unusual: the query positions are decoder steps (0..pred_len+label_len), keys are encoder steps (0..seq_len). The index distance `|i-j|` crosses two different sequence lengths — semantically, position 0 in the decoder is not the same as position 0 in the encoder.

---

## 6. Configuration Audit

| Parameter | README-E1.md | Shell Script | mse_mae_scores_sorted.txt | Consistent? |
|-----------|-------------|-------------|--------------------------|-------------|
| seq_len | 96 | (no explicit script) | — | ⚠️ |
| label_len | 48 | (no explicit script) | — | ⚠️ |
| pred_len | 24 (!) | — | 96, 192 (Phase 1); 48–336 (Phase 2) | 🔴 MISMATCH |
| attn | full | — | — | ✅ (from attn.py) |
| decay_a | 1.0 | — | α=0.5, 1.0, 2.0 tested | ⚠️ README hardcodes a=1.0 |
| d_ff | — | — | — | ⚠️ not stated |

**🔴 Critical mismatch:** `README-E1.md` states `Prediction Length: 24`, but `mse_mae_scores_sorted.txt` shows results for pred_len=48, 96, 192, 336 (no pred_len=24). The README appears to contain a stale or incorrect configuration value — likely copied from an earlier run.

---

## 7. Notebook Audit

**Notebook:** `experiments/exp1_distance_pre_softmax_decay/exp1_phase1_pre.ipynb`  
**Status:** Not fully parsed (large Colab notebook)  
**Known from mse_mae_scores_sorted.txt notation:** Results reference `exp1_phase1_pre.ipynb`

---

## 8. Result Verification

### Phase 1 Results (from `mse_mae_scores_sorted.txt`):
| α | pred_len | MSE | MAE | Source |
|---|---------|-----|-----|--------|
| 0.5 | 96 | 0.9024 | 0.7336 | mse_mae_scores_sorted.txt |
| 1.0 | 96 | 0.8683 | 0.7251 | mse_mae_scores_sorted.txt |
| 2.0 | 96 | 0.8784 | 0.7152 | mse_mae_scores_sorted.txt |
| 0.5 | 192 | 0.8948 | 0.7418 | mse_mae_scores_sorted.txt |
| 1.0 | 192 | 0.8463 | 0.7036 | mse_mae_scores_sorted.txt |
| 2.0 | 192 | 0.9573 | 0.7506 | mse_mae_scores_sorted.txt |

### README.md Summary Table (pred_len=96, seed=2021):
| Source | MSE | MAE |
|--------|-----|-----|
| `README.md` results table | 0.725 | 0.652 |
| `mse_mae_scores_sorted.txt` Phase 1 (α=1.0, pred=96) | 0.8683 | 0.7251 |
| `mse_mae_scores_sorted.txt` Phase 2 (α=1.0, pred=96) | 0.8796 | 0.7264 |

**🔴 CRITICAL INCONSISTENCY:** The README reports Exp1-Pre MSE=0.725, MAE=0.652 for pred_len=96. The central results file shows Phase 1 best as 0.8683 (α=1.0) and Phase 2 as 0.8796. **The README MSE value (0.725) does not appear in any phase of the experiment in the central results file.** There is a discrepancy of ~0.14 MSE.

**Possible explanations:**
1. README uses results from a different run configuration (different seed, different alpha, different hyperparameters)
2. Results were manually entered and are incorrect
3. An earlier run achieved 0.725 before current experiments were expanded

---

## 9. Consistency Checks

### 9.2 Formula
- ✅ `α(i,j) = 1/(1+|i-j|^a)` matches code exactly
- ✅ Pre-softmax application confirmed from code
- ✅ TECHNICAL_OBSERVATIONS.md formula matches code

### 9.3 Hyperparameters
- 🔴 `README-E1.md` states `Prediction Length: 24` — mismatch with actual runs at 48/96/192/336

### 9.4 Results
- 🔴 README MSE=0.725 does not match any logged result in mse_mae_scores_sorted.txt
- ✅ Phase 1 results in mse_mae_scores_sorted.txt are internally consistent (averages match raw values)
- ✅ Phase 2 alpha=1.0 runs at pred=96 match across seeds (0.8796, 0.8706, 0.8509)

### 9.5 Architecture
- ✅ Pre-softmax application confirmed
- ⚠️ Decoder cross-attention also gets distance decay — not clearly documented in README
- ⚠️ `PositionalEmbedding` instantiated but never called — dead weight

### 9.6 README Classification Mismatch
- README.md claims Exp1 MSE=0.725 is 🥉 rank 3 (after Baseline 0.519 and Exp5 0.719)
- Actual measured MSE (from central results file) at pred=96 is 0.8683 (Phase 1, α=1.0)
- **The ranking in README cannot be verified from actual logged results**

---

## 10. Inconsistency Report

### Critical Issues
1. **README MSE (0.725) ≠ any logged result**: README reports pred_len=96 MSE=0.725 for Exp1-Pre, but the central results file shows 0.8683 as the best Phase 1 result. Source of 0.725 is unknown and unverifiable.
2. **README pred_len wrong**: README-E1.md states `Prediction Length: 24` as the configuration; actual runs use 48/96/192/336.

### Moderate Issues
4. **Decoder distance decay undocumented**: Distance decay is active in decoder cross-attention (model.py lines 50-52); this is not mentioned in any documentation.
5. **Dead code**: `PositionalEmbedding` object instantiated but never used in `forward()`.

### Minor Issues
6. **Phase 1 explores 3 alpha values** (0.5, 1.0, 2.0) but `README-E1.md` only documents `a=1.0` as if it's the only configuration.

---

## 11. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation confidence | 9/10 | Code is clear, formula verified, pre-softmax application confirmed |
| Documentation confidence | 3/10 | README pred_len is wrong; MSE value 0.725 cannot be traced to any log |
| Result confidence | 5/10 | Phase 2 results internally consistent; Phase 1 best conflicts with README summary |
