# EXPERIMENT_6_AUDIT.md — Exp6-Pre
## LOD with Pre-Softmax Distance Decay + Delta-V
**Audit Status:** COMPLETE  
**Evidence Source:** `exp6_lod_pre/models/`, `README-E6-Pre.md`, `mse_mae_scores_sorted.txt`

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Exp 6 Pre — LOD with Pre-Softmax Decay |
| Folder | `experiments/exp6_lod_pre/` |
| Notebook | `Exp6_lod_pre-Gcolab.ipynb` |
| Shell scripts | `exp6_lod_pre_phase1.sh` (NOT found in directory listing!), `exp6_lod_pre_phase2.sh` (NOT found) |
| Result directory | **NONE** |
| Central results | `mse_mae_scores_sorted.txt` under "Exp6-Pre" |

**⚠️ CRITICAL:** The `experiments/exp6_lod_pre/` directory listing shows ONLY: `Exp6_lod_pre-Gcolab.ipynb`, `README-E6-Pre.md`, and `models/`. There are **no shell scripts** (no `exp6_lod_pre_phase1.sh` or `exp6_lod_pre_phase2.sh`), despite the README referencing them in the "File Structure" section.

---

## 2. Objective

Combine all three PRL components (Label+Order+Distance) using the **delta-V architectural split** (from Exp5b) but with distance decay applied **before** softmax in attention. Tests whether adding distance bias to the L+O combination with pre-softmax placement improves or degrades performance relative to Exp5 (no decay).

---

## 3. Mathematical Formulation (Derived from Code)

### Embedding (from `exp6_lod_pre/models/embed.py:106-128`):
```python
value_emb = self.value_embedding(x)
delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)
delta_x[:, 0, :] = 0.0
temporal_emb = self.temporal_embedding(x_mark)
legendre_pos = self.legendre_embedding(x)
combined_emb = value_emb + temporal_emb + legendre_pos
return self.dropout(combined_emb), self.dropout(delta_x)   # BOTH dropout'd
```

### Attention (from `exp6_lod_pre/models/attn.py:25-46`):
```python
scores = einsum(Q, K)
# PRE-SOFTMAX DECAY:
alpha = 1/(1 + dist_matrix^decay_a)
scores = scores * alpha
# THEN masking and softmax
A = softmax(scale * scores)
# V from delta_x projection
V = einsum(A, values)
```

**Combined formula:**
```
Embedding: Q/K ← projection(x_i + T_i + p_i)
           V   ← projection(delta_x_i)   where delta_x_i = x_i − x_{i-1}

Attention: scores_ij = (Q_i · K_j) / √d
           scores_ij = scores_ij · α(i,j)          PRE-softmax
           A_ij = softmax(scale · scores)_ij
           Output_i = Σ_j A_ij · V_j
```

---

## 4. Differences from Exp6-Post (Critical)

| Aspect | Exp6-Pre | Exp6-Post |
|--------|----------|-----------|
| Distance decay position | Pre-softmax (shapes distribution) | Post-softmax (rescales probabilities) |
| delta_x dropout | `self.dropout(delta_x)` applied in embed.py | `delta_x` returned without separate dropout |
| Conceptual effect | Hard suppression of distant logits before normalization | Soft re-weighting of already-normalized probabilities |

**Verified from code:** Pre: line 128 `return self.dropout(combined_emb), self.dropout(delta_x)`. Post: line 127 `return self.dropout(combined_emb), delta_x`. The dropout asymmetry is a real difference.

---

## 5. Legendre Implementation

Uses `experiments/exp6_lod_pre/models/legendre_embedding.py` — pure PyTorch recurrence (verified):
```python
for n in range(2, self.d_model):
    P[:, n] = ((2*n - 1) * positions * P[:, n-1] - (n-1) * P[:, n-2]) / n
```
This is mathematically correct Legendre recurrence. Scaling: `P / math.sqrt(self.d_model)`.

---

## 6. Result Verification

### Phase 1 (α sweep, from `mse_mae_scores_sorted.txt`):
| α | pred_len | MSE | MAE |
|---|---------|-----|-----|
| 0.5 | 96 | 0.9234 | 0.7668 |
| 1.0 | 96 | 0.8694 | 0.7332 |
| 2.0 | 96 | 0.8947 | 0.7430 |
| 0.5 | 192 | 0.9915 | 0.7759 |
| 1.0 | 192 | 0.9571 | 0.7634 |
| 2.0 | 192 | 0.9287 | 0.7312 |

### Phase 2 (α=1.0, from `mse_mae_scores_sorted.txt`):
| pred_len | seed | MSE | MAE |
|---------|------|-----|-----|
| 48 | 2021 | 0.8638 | 0.7415 |
| 48 | 2022 | 0.7529 | 0.6613 |
| 48 | 2023 | 0.8664 | 0.7009 |
| 96 | 2021 | 0.7908 | 0.6579 |
| 96 | 2022 | 0.9980 | 0.7910 |
| 96 | 2023 | 0.9743 | 0.7865 |
| 192 | 2021 | 1.0472 | 0.7877 |
| 192 | 2022 | 0.8828 | 0.7267 |
| 192 | 2023 | 0.9093 | 0.7192 |

**README-E6-Pre.md Phase 2 results table:**
| pred_len | seed 2021 | seed 2022 | seed 2023 | Average MSE |
|---------|-----------|-----------|-----------|-------------|
| 48 | 0.8638 | 0.7529 | 0.8664 | 0.8277 |
| 96 | 0.7908 | 0.9980 | 0.9743 | 0.9210 |
| 192 | 1.0472 | 0.8828 | 0.9093 | 0.9464 |

**✅ Phase 2 values match between README-E6-Pre.md and mse_mae_scores_sorted.txt exactly** — this is consistent.

---

## 7. 🔴 CRITICAL FINDING: Identical Results to Exp6-Post

**From `mse_mae_scores_sorted.txt`:**

**Exp6-Pre Phase 1:**
```
0.5|96|2021|0.9234|0.7668
1.0|96|2021|0.8694|0.7332
2.0|96|2021|0.8947|0.7430
0.5|192|2021|0.9915|0.7759
1.0|192|2021|0.9571|0.7634
2.0|192|2021|0.9287|0.7312
```

**Exp6-Post Phase 1:**
```
0.5|96|2021|0.9234|0.7668
1.0|96|2021|0.8694|0.7332
2.0|96|2021|0.8947|0.7430
0.5|192|2021|0.9915|0.7759
1.0|192|2021|0.9571|0.7634
2.0|192|2021|0.9287|0.7312
```

**IDENTICAL. TO. EVERY. DECIMAL.**

And Phase 2:
**Exp6-Pre Phase 2:** 0.8638/0.7529/0.8664 (48), 0.7908/0.9980/0.9743 (96), 1.0472/0.8828/0.9093 (192)  
**Exp6-Post Phase 2:** 0.8638/0.7529/0.8664 (48), 0.7908/0.9980/0.9743 (96), 1.0472/0.8828/0.9093 (192)

**ALL Phase 1 and Phase 2 results for Exp6-Pre and Exp6-Post are byte-for-byte identical in the central results file.**

---

## 8. Consistency Checks

### 8.1 Architecture Differences Confirmed
- ✅ Pre vs Post confirmed from code — the two experiments have genuinely different code
- ✅ embed.py differs between Pre and Post (dropout on delta_x)

### 8.2 Shell Scripts Missing
- 🔴 `exp6_lod_pre_phase1.sh` and `exp6_lod_pre_phase2.sh` referenced in README but NOT present in directory

### 8.3 Results Validity
- 🔴 **All results for Exp6-Pre are identical to Exp6-Post in central file** — scientifically impossible for two different implementations with random initialization and different attention computation
- The README-E6-Pre.md's Phase 2 table perfectly matches central file — suggesting the README was generated FROM the central file
- Two explanations: (a) one experiment's results were copy-pasted into the other's entry, OR (b) both experiments ran the same code (one set of files was not properly copied before running)

---

## 9. Inconsistency Report

### Critical Issues
1. **🔴 DEFINITIVE: Exp6-Pre and Exp6-Post results are identical** — 18 identical values (6 Phase 1 + 12 Phase 2) is statistically impossible by chance. Results were copy-pasted or both ran the same code.
2. **Shell scripts missing**: `exp6_lod_pre_phase1.sh` and `exp6_lod_pre_phase2.sh` referenced in README do not exist in directory.
3. **No results directory**: Cannot verify from log files.

### Moderate Issues
4. **README-E6-Pre.md analysis is built on copied results**: The analysis claiming "Exp6-Pre shows mixed results vs Exp5" is based on numbers that may be from a different experiment.
5. **Decoder cross-attention uses Attn(True,...) WITHOUT decay_a** in model.py (line 52: `Attn(True, factor,...)` — note no decay_a). Pre version's decoder self-attention uses `Attn(True,...,decay_a=decay_a)` but cross-attention: `FullAttention(False,...,decay_a=decay_a)`. Wait — actually reading model.py again: exp6_lod_pre model.py line 51-55 shows decoder self-attention as `Attn(True, ..., decay_a=decay_a)` and cross-attention as `FullAttention(False, ..., decay_a=decay_a)`. So decay is in both — this is consistent but different from other experiments.

---

## 10. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation confidence | 8/10 | Code correctly implements pre-softmax decay + delta-V |
| Documentation confidence | 4/10 | README analysis is consistent with its own data (which may be copied) |
| Result confidence | 1/10 | Results are identical to Exp6-Post — either copy-pasted or wrong experiment ran |
