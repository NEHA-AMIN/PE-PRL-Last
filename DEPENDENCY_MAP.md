# DEPENDENCY_MAP.md
## Complete Dependency Graph — All Active Experiments
### Evidence-Based: Derived Entirely from Source Code

---

## Execution Model

Every experiment follows this pattern:
1. Shell script copies experiment-local `models/*.py` files into `Informer2020-original/models/`
2. `cd Informer2020-original` and runs `python main_informer.py` with specific flags
3. `Informer2020-original` acts as **shared mutable execution environment** — overwritten by each experiment

**⚠️ Critical implication**: The current state of `Informer2020-original/models/` reflects the **last experiment that was copied in**, NOT the original Informer. Verified: current state contains `DataEmbedding_ordering_sem` in `embed.py` and only `vanilla`/`ordering_sem` dispatch in `model.py`.

---

## Baseline

```
experiments/Baseline/baseline_phase1.sh
    │
    ├── uses: Informer2020-original/models/ (vanilla state)
    ├── --attn prob  (ProbSparse attention)
    ├── --embed timeF  (TimeFeatureEmbedding, NOT TemporalEmbedding)
    ├── --dropout 0.05, --d_ff 2048
    └── NO pe_mode flag (default: vanilla in embed.py)
         ↓
    embed.py:DataEmbedding.forward()
        x = value_embedding(x) + position_embedding(x) + temporal_embedding(x_mark)
         ↓
    attn.py: Standard ProbAttention (no distance)
         ↓
    Results: results/baseline_ph1_ETTh1_pred{96,192}_seed2021/  [EMPTY DIRECTORIES]
```

**Dependency files:**
- `Informer2020-original/models/embed.py` → `DataEmbedding`
- `Informer2020-original/models/attn.py` → `ProbAttention`
- `Informer2020-original/models/model.py` → `Informer`

---

## Exp1-Pre (Distance Decay, Pre-Softmax)

```
experiments/exp1_distance_pre_softmax_decay/models/attn.py
    │── FullAttention.forward():
    │       scores = einsum(Q, K)
    │       dist_matrix = |q_idx - k_idx|     [computed fresh each forward()]
    │       alpha = 1/(1 + dist_matrix^decay_a)
    │       scores = scores * alpha            ← PRE-softmax
    │       A = softmax(scale * scores)
    │       V = einsum(A, values)
    │
experiments/exp1_distance_pre_softmax_decay/models/embed.py
    │── DataEmbedding.forward():
    │       x = value_embedding(x) + temporal_embedding(x_mark)
    │       ← NO sinusoidal positional embedding
    │
experiments/exp1_distance_pre_softmax_decay/models/model.py
    │── Informer.__init__(decay_a=1.0):
    │       decay_a passed to BOTH ProbAttention and FullAttention
    │       ALSO passed to FullAttention in decoder cross-attention
    │
Shell: no local shell script with clear path; uses Colab notebook
Results: results/exp1_distance_decay/training_log.txt  [NAME MISMATCH]
```

**Key math (verified from code):**
- Embedding: `X'_i = X_i + T_i` (no sinusoidal PE)
- Attention: `A = softmax(scale · QKᵀ ⊙ α)` where `α(i,j) = 1/(1+|i-j|^a)`

---

## Exp1-Post (Distance Decay, Post-Softmax)

```
experiments/exp1_distance_post_softmax/models/attn.py
    │── FullAttention.forward():
    │       scores = einsum(Q, K)
    │       A = softmax(scale * scores)          ← softmax FIRST
    │       dist_matrix = |q_idx - k_idx|
    │       alpha = 1/(1 + dist_matrix^decay_a)
    │       A = A * alpha                        ← THEN multiply by decay
    │       V = einsum(A, values)
    │
embed.py: Same as Exp1-Pre — no sinusoidal PE, temporal only
Results: results/exp1_distance_post_softmax/training_log.txt
```

**Key math (verified from code):**
- Embedding: `X'_i = X_i + T_i`
- Attention: `A_final = softmax(scale · QKᵀ) ⊙ α` — NOT a valid probability simplex after this

---

## Exp2 — Full LOD (Embedding-Level, No Attention Decay)

```
experiments/exp2_full_paper/models/embed.py
    │── DataEmbedding.__init__():
    │       LegendrePositionEmbedding(d_model, scaling=True)  ← dynamic import
    │       DistancePositionOperator(decay_a=1.0, distance_type='l1')  ← dynamic import
    │       distance_scale = 1/√d_model
    │
    │── DataEmbedding.forward():
    │       value_emb = TokenEmbedding(x)
    │       temporal_emb = TemporalEmbedding(x_mark)
    │       legendre_pos = LegendrePositionEmbedding(x)
    │       distance_pos = DistancePositionOperator(legendre_pos) * distance_scale
    │       x = value_emb + temporal_emb + legendre_pos + distance_pos
    │
    │── CRITICAL: distance_operator RECEIVES legendre_pos, NOT value_emb
    │       (positional-space distance, not semantic)
    │
experiments/exp2_full_paper/models/attn.py
    │── UNMODIFIED baseline FullAttention (no distance decay in attention)
    │
experiments/exp2_full_paper/models/distance_operator.py
    │── DistancePositionOperator.forward(X):  [X = legendre_pos here]
    │       alpha = 1/(1+|i-j|^a)             [index-based decay]
    │       d_ij = torch.cdist(X, X, p=1.0)   [L1 feature-space distance on legendre]
    │       w_ij = 1/(1+d_ij)                 [feature-space weighting]
    │       W = alpha * w_ij                  [combined weights]
    │       O = X*(W.sum) - bmm(W, X)         [algebraic rearrangement — avoids [B,L,L,D]]
    │
experiments/exp2_full_paper/models/legendre_embedding.py
    │── [Imported dynamically; NOT the same implementation as exp6]
    │   Uses scipy (confirmed by TECHNICAL_OBSERVATIONS.md reference)
    │   [NOT verified directly — import is inside __init__ via sys.path.insert]
    │
Results: results/exp2_full_paper/training_log.txt
```

---

## Exp3 — Label Only (No Temporal)

```
experiments/exp3_label_only/models/embed.py
    │── DataEmbedding.forward():
    │       value_emb = TokenEmbedding(x)
    │       legendre_pos = LegendrePositionEmbedding(x)
    │       x = value_emb + legendre_pos     ← NO temporal, NO sinusoidal
    │
    │── README claims: "X'_i = X_i + P_i"
    │── Code confirms: IDENTICAL to README claim
    │
Results: results/exp3_label_only/training_log.txt
```

---

## Exp5 — Label + Order (Positional Pairwise)

```
experiments/exp5_label_order/models/embed.py
    │── DataEmbedding.forward():
    │       value_emb = TokenEmbedding(x)
    │       temporal_emb = TemporalEmbedding(x_mark)
    │       legendre_pos = LegendrePositionEmbedding(x)    ← positional
    │       ordering_pos = OrderingOperator(legendre_pos)  ← O applied to P, not X
    │       ordering_pos = ordering_pos / √d_model         ← scaling
    │       x = value_emb + temporal_emb + legendre_pos + ordering_pos
    │
experiments/exp5_label_order/models/ordering_operator.py
    │── OrderingOperator.forward(X):   [X = legendre_pos]
    │       X_i = X.unsqueeze(2)      [B,L,1,D]
    │       X_j = X.unsqueeze(1)      [B,1,L,D]
    │       delta_x = X_i - X_j      [B,L,L,D] — ALL PAIRS
    │       mask diagonal, sum / (L-1)
    │       O = delta_x.sum(dim=2) / (L-1)   [B,L,D]
    │   ← This is O(L²) memory and compute
    │
experiments/exp5_label_order/models/attn.py
    │── UNMODIFIED baseline FullAttention (no distance decay)
    │
Results: results/exp5_label_order/training_log.txt
```

---

## Exp5b — Label + Order (Clean Delta, V-Split)

```
experiments/exp5b_label_order_clean_delta_MV/models/embed.py
    │── DataEmbedding.forward() RETURNS TUPLE:
    │       value_emb = TokenEmbedding(x)
    │       delta_x = value_emb - torch.roll(value_emb, 1, dims=1)
    │       delta_x[:, 0, :] = 0.0              ← zero boundary
    │       temporal_emb = TemporalEmbedding(x_mark)
    │       legendre_pos = LegendrePositionEmbedding(x)
    │       combined_emb = value_emb + temporal_emb + legendre_pos
    │       return dropout(combined_emb), delta_x  ← TUPLE
    │
experiments/exp5b_label_order_clean_delta_MV/models/attn.py
    │── AttentionLayer.forward(q, k, v, mask, delta_values=None):
    │       Q = query_projection(queries)  ← from combined_emb
    │       K = key_projection(keys)       ← from combined_emb
    │       if delta_values:
    │           V = value_projection(delta_values)  ← from delta_x
    │       else:
    │           V = value_projection(values)
    │
experiments/exp5b_label_order_clean_delta_MV/models/encoder.py
    │── EncoderLayer.forward(x, attn_mask, delta_x):
    │       self.attention(x, x, x, attn_mask,
    │           delta_queries=delta_x, delta_keys=delta_x, delta_values=delta_x)
    │       ← Passes delta_x as ALL three (but only delta_values is used in attn.py)
    │
    │── Encoder: delta_x passed through; ConvLayer ALSO applied to delta_x
    │       delta_x = conv_layer(delta_x)   ← downsampling consistent
    │
Results: NO results directory — Exp5b results only in mse_mae_scores_sorted.txt (Phase 1 only)
```

---

## Exp6-Pre — LOD with Pre-Softmax Decay + Delta-V

```
experiments/exp6_lod_pre/models/embed.py
    │── DataEmbedding.forward() RETURNS TUPLE:
    │       value_emb, delta_x (torch.roll, zero boundary)
    │       combined_emb = value_emb + temporal_emb + legendre_pos
    │       return dropout(combined_emb), dropout(delta_x)   ← dropout on delta_x too
    │
    │── DIFFERENCE from exp6_lod_post: both get dropout (post does too)
    │
experiments/exp6_lod_pre/models/attn.py
    │── FullAttention.forward():
    │       scores = einsum(Q, K)
    │       alpha = 1/(1+|i-j|^decay_a)
    │       scores = scores * alpha     ← PRE-softmax
    │       A = softmax(scale * scores)
    │       V = einsum(A, values)       ← values = projected delta_x
    │
Shell: No exp6_lod_pre_phase1.sh found in listing — missing
Results: NO results directory
```

---

## Exp6-Post — LOD with Post-Softmax Decay + Delta-V

```
experiments/exp6_lod_post/models/embed.py
    │── IDENTICAL to exp6_lod_pre embed.py EXCEPT:
    │       return self.dropout(combined_emb), delta_x   ← delta_x NOT dropout'd separately
    │       [exp6_lod_pre: return self.dropout(combined_emb), self.dropout(delta_x)]
    │
experiments/exp6_lod_post/models/attn.py
    │── FullAttention.forward():
    │       scores = einsum(Q, K)
    │       A = softmax(scale * scores)   ← softmax FIRST
    │       alpha = 1/(1+|i-j|^decay_a)
    │       A = A * alpha                 ← POST-softmax
    │       V = einsum(A, values)
    │
experiments/exp6_lod_post/models/encoder.py
    │── Encoder: delta_x passed through but ConvLayer NOT applied to delta_x
    │       x = conv_layer(x)             ← x downsampled
    │       [delta_x NOT downsampled]     ← potential shape mismatch if distil=True
    │
Shell: exp6_lod_post_phase1.sh — uses --attn full, --decay_a, no --embed flag (default 'fixed')
Results: NO results directory
```

---

## Legendre Implementations — Two Distinct Versions

| Experiment | File | Implementation | Difference |
|------------|------|---------------|------------|
| Exp2, Exp3, Exp5, Exp5b | `legendre_embedding.py` via dynamic `sys.path.insert` | Uses `scipy.special.legendre` OR custom | Dynamically imported — exact implementation not directly verified for all |
| Exp6-Pre | `experiments/exp6_lod_pre/models/legendre_embedding.py` | Pure PyTorch recurrence: `P_n = ((2n-1)xP_{n-1}-(n-1)P_{n-2})/n` | No scipy dependency |
| Exp6-Post | `experiments/exp6_lod_post/models/legendre_embedding.py` | Same pure PyTorch recurrence | Scaling: `P / (self.d_model ** 0.5)` vs exp6_lod_pre: `P / math.sqrt(self.d_model)` — functionally identical |

---

## Hyperparameter Consistency Matrix

| Parameter | Baseline | Exp1 | Exp2 | Exp3 | Exp5 | Exp5b | Exp6 |
|-----------|----------|------|------|------|------|-------|------|
| seq_len | 96 | 96 | 96 | 96 | 96 | 96 | 96 |
| label_len | 48 | 48 | 48 | 48 | 48 | 48 | 48 |
| d_model | 512 | 512 | 512 | 512 | 512 | 512 | 512 |
| n_heads | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| e_layers | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| d_layers | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| attn type | prob | full | full | full | full | full | full |
| d_ff | 2048 | 512* | 512* | 512* | 2048 | — | — |
| embed type | timeF | fixed* | fixed* | fixed* | timeF | — | fixed* |
| dropout | 0.05 | 0.0* | 0.0* | 0.0* | 0.05 | — | 0.0* |

*`*` = value inferred from model.py defaults (`d_ff=512`, `dropout=0.0`, `embed='fixed'`) — shell scripts do not override these

**⚠️ Configuration inconsistency**: Baseline uses `--attn prob`, `--embed timeF`, `--dropout 0.05`, `--d_ff 2048`. Ablation experiments use `--attn full`, default `embed='fixed'`, default `dropout=0.0`, default `d_ff=512`. This means the baseline is NOT a clean control for the ablation experiments.
