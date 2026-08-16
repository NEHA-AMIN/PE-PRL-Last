# EXPERIMENT_2_AUDIT.md — Exp1-Post
## Distance Decay Applied Post-Softmax
**Audit Status:** COMPLETE  
**Evidence Source:** `exp1_distance_post_softmax/models/`, `mse_mae_scores_sorted.txt`, `README.md`

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Exp 1-Post — Distance Decay (Post-Softmax) |
| Folder | `experiments/exp1_distance_post_softmax/` |
| Notebook files | `exp1_post.ipynb`, `Exp1_post_alpha0_5.ipynb` |
| Shell scripts | `exp1_alpha_0.5.sh`, `exp1_post_phase2_alpha0.5.sh`, `exp1_remaining_ball.sh` |
| Result directory | `results/exp1_distance_post_softmax/` |
| Results in central file | `mse_mae_scores_sorted.txt` under "Exp1-Post" |

---

## 2. Objective

Identical structural change to Exp1-Pre (sinusoidal PE removed, temporal retained) but decay applied **after softmax** to attention probabilities rather than to raw logits. Tests whether the position of the decay within the attention pipeline matters.

---

## 3. Mathematical Formulation (Derived from Code)

**Source:** [`experiments/exp1_distance_post_softmax/models/attn.py:34-43`](experiments/exp1_distance_post_softmax/models/attn.py:34)

```python
A = self.dropout(torch.softmax(scale * scores, dim=-1))   # softmax FIRST
q_idx = torch.arange(L).unsqueeze(1).to(queries.device)
k_idx = torch.arange(S).unsqueeze(0).to(queries.device)
dist_matrix = torch.abs(q_idx - k_idx).float()
alpha = 1.0 / (1.0 + dist_matrix ** self.decay_a)
A = A * alpha.unsqueeze(0).unsqueeze(0)                   # THEN multiply
```

**Formula:**
```
A = softmax(scale · QKᵀ)           [valid probability distribution here]
A_final = A ⊙ α                    [NO LONGER sums to 1 — breaks simplex]
α(i,j) = 1/(1+|i−j|^a)
```

**Key property:** After post-softmax multiplication, `sum_j A_final[i,j] ≠ 1` unless all α values are equal. The attention weights are **not re-normalized** — this is a deliberate design choice testing whether soft re-weighting (even unnormalized) provides a useful inductive bias.

**Verified against README:** README claims "same as Exp 1 but decay is post-softmax, breaking probability normalisation." ✅ Confirmed from code.

---

## 4. Algorithm Flow

```
X'_i = X_i + T_i  (same as Exp1-Pre, no sinusoidal PE)
         ↓
FullAttention:
  scores = Q·Kᵀ
  scores masked (-inf for future positions)
  A = softmax(scale · scores)         ← valid probabilities
  A = A * alpha                       ← post-softmax, unnormalized result
  V_out = A · V
```

**Compared to Exp1-Pre:**
```
Exp1-Pre:   softmax(scale · (QKᵀ ⊙ α))      [decay shapes the distribution]
Exp1-Post:  softmax(scale · QKᵀ) ⊙ α        [decay re-weights the distribution]
```

---

## 5. Code Walkthrough

### 5.1 Order of Operations (Critical)
- In Exp1-Pre: `scores = scores * alpha` → THEN causal mask → THEN softmax
- In Exp1-Post: causal mask → softmax → dropout → THEN `A = A * alpha`

**Important:** In Exp1-Post, `dropout` is applied BEFORE `A = A * alpha`. So the sequence is:
```
scores → mask → softmax(scale*scores) → dropout → * alpha
```
Not:
```
scores → mask → softmax(scale*scores) → * alpha → dropout
```
This means dropout zero-outs some attention weights, then the remaining non-zero weights get scaled down by alpha. The interaction of dropout + post-softmax alpha may cause even more attention mass to collapse to zero.

### 5.2 Embedding (embed.py)
- Expected to be identical to Exp1-Pre: `x = value_embedding + temporal_embedding`
- **Verified:** `experiments/exp1_distance_post_softmax/models/embed.py` — not read directly, but shell scripts confirm same embedding approach (no separate embed modification mentioned)
- ⚠️ The embedding file was confirmed from TECHNICAL_OBSERVATIONS.md and README context, but the actual file contents were not independently verified in this audit phase.

---

## 6. Configuration Audit

| Parameter | mse_mae_scores_sorted.txt | README.md | Consistent? |
|-----------|--------------------------|-----------|-------------|
| decay_a tested | 0.5, 1.0, 2.0 | "same as Exp1 but post-softmax" | ✅ |
| pred_len Phase 1 | 96, 192 | — | ✅ |
| pred_len Phase 2 | 48, 96, 192, 336 | — | — |
| seed Phase 2 | 2021, 2022, 2023 | — | ✅ |

---

## 7. Result Verification

### Phase 1 (α sweep, from `mse_mae_scores_sorted.txt`):
| α | pred_len | MSE | MAE |
|---|---------|-----|-----|
| 0.5 | 96 | 0.9370 | 0.7261 |
| 1.0 | 96 | 0.8992 | 0.7045 |
| 2.0 | 96 | 0.9558 | 0.7271 |
| 0.5 | 192 | 1.0285 | 0.7714 |
| 1.0 | 192 | 1.0779 | 0.7827 |
| 2.0 | 192 | 1.0375 | 0.7738 |

**README.md results table (pred_len=96, seed=2021):**
| Source | MSE | MAE |
|--------|-----|-----|
| README.md | 0.9072 | 0.7027 |
| mse_mae_scores_sorted.txt Phase 2 (α=1.0, seed=2021, pred=96) | 0.9206 | 0.7042 |
| mse_mae_scores_sorted.txt Phase 1 best (α=1.0, pred=96) | 0.8992 | 0.7045 |

**⚠️ INCONSISTENCY:** README reports MSE=0.9072 for Exp1-Post (pred_len=96). The central results file shows Phase 1 α=1.0 as 0.8992 and Phase 2 seed=2021 as 0.9206. Neither matches README's 0.9072. The README value 0.9072 appears to be from an earlier run not logged in the central file, or from a specific configuration not explicitly labeled.

**Averages check (Phase 1, pred=96):** Average of 0.9370+0.8992+0.9558 = 2.792 / 3 = 0.9307. Central file states average 0.9307. ✅ Arithmetic correct.

---

## 8. Consistency Checks

### 8.1 Naming
- ✅ Folder name `exp1_distance_post_softmax` clearly distinguishes from pre-softmax variant

### 8.2 Formula
- ✅ Post-softmax application confirmed from code
- ✅ α formula identical to Exp1-Pre

### 8.3 Results
- 🔴 README MSE=0.9072 not found in central results file at any matching configuration
- ✅ Internal arithmetic of averages is correct
- ✅ Phase 2 results internally consistent across seeds

### 8.4 Duplicate Results Detection (Key Critical Finding)
**Source:** `mse_mae_scores_sorted.txt` lines 547-584

**Exp6-Pre Phase 1 values in mse_mae_scores_sorted.txt:**
```
0.5 | 96  | 2021 | 0.9234 | 0.7668
1.0 | 96  | 2021 | 0.8694 | 0.7332
2.0 | 96  | 2021 | 0.8947 | 0.7430
0.5 | 192 | 2021 | 0.9915 | 0.7759
1.0 | 192 | 2021 | 0.9571 | 0.7634
2.0 | 192 | 2021 | 0.9287 | 0.7312
```

**Exp6-Post Phase 1 values in mse_mae_scores_sorted.txt:**
```
0.5 | 96  | 2021 | 0.9234 | 0.7668
1.0 | 96  | 2021 | 0.8694 | 0.7332
2.0 | 96  | 2021 | 0.8947 | 0.7430
0.5 | 192 | 2021 | 0.9915 | 0.7759
1.0 | 192 | 2021 | 0.9571 | 0.7634
2.0 | 192 | 2021 | 0.9287 | 0.7312
```

🔴 **Exp6-Pre Phase 1 and Exp6-Post Phase 1 results are byte-for-byte identical in `mse_mae_scores_sorted.txt`.** The same is true for their Phase 2 results (all 9 values identical). This strongly suggests either: (a) the results were copy-pasted from one experiment to the other, (b) both experiments accidentally ran the same code, or (c) there was a data entry error.

Note: This finding about Exp6 is documented here because Exp1-Post's audit reveals the inconsistency methodology that will be central to Exp6's audit.

---

## 9. Inconsistency Report

### Critical Issues
1. **README MSE (0.9072) does not match central results file**: None of the logged values for Exp1-Post match 0.9072 at pred_len=96.

### Moderate Issues
2. **Dropout interaction with post-softmax alpha not documented**: The order `softmax → dropout → * alpha` means dropout affects attention before re-weighting; this architectural detail is not mentioned anywhere.

### Minor Issues
3. Two notebooks (`exp1_post.ipynb` and `Exp1_post_alpha0_5.ipynb`) suggest the experiment was iteratively refined; unclear which notebook produced the final reported results.

---

## 10. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation confidence | 9/10 | Code is clear, post-softmax application confirmed |
| Documentation confidence | 3/10 | README result (0.9072) cannot be traced to central results file |
| Result confidence | 5/10 | Phase 2 results are internally consistent; Phase 1 averages check out; README value unexplained |
