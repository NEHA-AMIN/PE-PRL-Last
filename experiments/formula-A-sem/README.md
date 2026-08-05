# Experiment 4c — Delta Ordering in Positional Space

## 1. One-Line Summary

Replace the sinusoidal positional embedding with a consecutive-delta ordering
signal derived from the token embeddings.  Every other component is unchanged.

---

## 2. Research Position

This experiment belongs to the **pure ordering** track of the ablation study.
It isolates a single question:

> **Can first-order differences between consecutive token embeddings, injected
> into the positional pathway, carry enough sequential structure to replace
> absolute sinusoidal positional encoding?**

It is the first of the four intended ordering experiments:

| # | Formula | Space | Folder |
|---|---------|-------|--------|
| **4c** | **Delta Ordering** | **Positional** | `exp4c_delta_ordering_pos_space/` ← **this** |
| 4d | Delta Ordering | Semantic | `exp4_ordering_new_sem_space/` |
| 5c | Normalized Delta | Positional | *(planned)* |
| 5d | Normalized Delta | Semantic | `exp4_ordering_new_sem_space/` (normalized variant) |

---

## 3. The Baseline Embedding

The vanilla Informer (Zhou et al., 2021) combines three signals at the embedding
layer before the first attention layer:

$$
\tilde{X}_i = X_i + \mathrm{PE}_i + T_i
$$

| Symbol | Class | What it encodes |
|--------|-------|-----------------|
| $X_i \in \mathbb{R}^d$ | `TokenEmbedding` (Conv1D, 3-kernel, circular) | Semantic content of the 7 input features |
| $\mathrm{PE}_i \in \mathbb{R}^d$ | `PositionalEmbedding` (sinusoidal, frozen) | **Absolute position** of token $i$ in the sequence |
| $T_i \in \mathbb{R}^d$ | `TimeFeatureEmbedding` (linear) | Calendar context (hour, weekday, day, month) |

The sinusoidal PE (Vaswani et al., 2017) is:

$$
\mathrm{PE}_{(i,2k)}   = \sin\!\left(\frac{i}{10000^{2k/d}}\right), \qquad
\mathrm{PE}_{(i,2k+1)} = \cos\!\left(\frac{i}{10000^{2k/d}}\right)
$$

It tells the model **where** a token sits in the sequence using fixed irrational
frequency ratios.  It carries no information about **how the sequence is
changing** at that point.

---

## 4. The Exp 4c Embedding: Delta Ordering

### 4.1 Formula

$$
\boxed{\tilde{X}_i = X_i + \Delta(X_i) + T_i}
$$

where the consecutive delta is:

$$
\Delta(X_i) =
\begin{cases}
X_1              & i = 1 \quad \text{(boundary: no predecessor, treat as } X_1 - 0\text{)} \\
X_i - X_{i-1}   & i \geq 2
\end{cases}
$$

and $X_i = \mathrm{TokenEmbedding}(x_i) \in \mathbb{R}^d$ is the Conv1D output,
not the raw input $x_i$.

### 4.2 What $\Delta(X_i)$ Encodes

$\Delta(X_i)$ is the **signed vector of change** between consecutive token
representations in the $d$-dimensional embedding space.

- **Direction**: if feature $k$ of $X_i$ is larger than feature $k$ of
  $X_{i-1}$, then $\Delta(X_i)_k > 0$; if smaller, $\Delta(X_i)_k < 0$.
- **Magnitude**: large $\|\Delta(X_i)\|$ means the embedding changed a lot
  between step $i-1$ and step $i$ (fast transition); small norm means the
  embedding is stable (slow transition).
- **Position**: the model implicitly knows "where it is" because tokens at
  turning points or trend shifts will have large, signed deltas, while tokens
  in flat regions will have near-zero deltas.

### 4.3 Boundary Condition

At position $i = 0$ (0-indexed), there is no $X_{-1}$, so the formula sets:

$$
\Delta(X_0) = X_0 - 0 = X_0
$$

In code this is:

```python
delta[:, 0, :] = val[:, 0, :]          # first token: Δ = X_0
delta[:, 1:, :] = val[:, 1:, :] - val[:, :-1, :]   # all others: Δ = X_i - X_{i-1}
```

This means the first token's positional signal equals its own content — it is
"positioned by its identity," which is a natural boundary choice when no prior
step exists.

### 4.4 What the Model Receives

For each token $i$, the transformer's first layer receives:

$$
\tilde{X}_i = \underbrace{X_i}_{\text{semantic content}} + \underbrace{\Delta(X_i)}_{\text{ordering signal}} + \underbrace{T_i}_{\text{calendar}}
$$

The attention mechanism then operates on $\tilde{X}_i$ using standard scaled
dot-product attention — no modifications anywhere else in the model.

---

## 5. Relationship to the Baseline: Exactly What Changed

This is the minimal surgical change from the vanilla Informer:

| | Baseline | Exp 4c |
|-|----------|--------|
| `TokenEmbedding` | ✅ unchanged | ✅ unchanged |
| `TemporalEmbedding` | ✅ unchanged | ✅ unchanged |
| `PositionalEmbedding` (sinusoidal) | ✅ active | ❌ **removed** |
| Delta ordering $\Delta(X_i)$ | ❌ not present | ✅ **added** |
| `FullAttention` / `ProbAttention` | ✅ unchanged | ✅ unchanged |
| `Encoder` / `Decoder` | ✅ unchanged | ✅ unchanged |
| Projection head | ✅ unchanged | ✅ unchanged |
| New trainable parameters | — | **0** (delta is computed from existing `TokenEmbedding` output) |

**Single-line diff in `DataEmbedding.forward()`:**

```python
# Baseline:
x = value_embedding(x) + position_embedding(x) + temporal_embedding(x_mark)

# Exp 4c:
x = val + delta + temp
#         ^^^^^
#         replaces position_embedding(x)
```

---

## 6. Mathematical Properties

### 6.1 Order Sensitivity

Permuting the sequence changes $\Delta(X_i)$ at every permuted position.  This
is the core property — the signal encodes *relative order of change*, not just
*absolute position*.

### 6.2 Translation Invariance

Adding a constant vector $c$ to every raw input shifts $X_i \to X_i + c'$
(where $c'$ depends on the Conv1D kernel, but is uniform across positions).
Then for $i \geq 1$:

$$
\Delta(X_i + c') = (X_i + c') - (X_{i-1} + c') = X_i - X_{i-1} = \Delta(X_i)
$$

The ordering signal is **translation-invariant** in the representation space for
all positions except the boundary ($i = 0$, where the full shift $c'$ appears
in the delta).

### 6.3 Scale Sensitivity

Unlike the normalized variant (Exp 4d/5d), this experiment does **not**
normalize the delta.  Multiplying all $X_i$ by a scalar $\alpha$ multiplies
$\Delta(X_i)$ by $\alpha$.  This means the magnitude of the ordering signal
scales with the magnitude of the embeddings, which may dominate or be dominated
by $X_i$ depending on the input scale.  The diagnostic printout in the code
monitors this ratio during training.

### 6.4 Zero Parameters Added

The delta is a deterministic function of the token embeddings.  No weight
matrices, no buffers, no additional memory beyond the $[B, L, D]$ delta tensor
itself.

### 6.5 Comparison to Sinusoidal PE

| Property | Sinusoidal PE | Delta Ordering $\Delta(X_i)$ |
|----------|---------------|------------------------------|
| Input-dependent | No (fixed) | Yes (changes with each batch) |
| Encodes absolute position | Yes | No — encodes *relative change* |
| Encodes sequence dynamics | No | Yes |
| Trainable | No | No |
| Extra parameters | 0 | 0 |
| Same for every input | Yes | No |

---

## 7. Implementation Details

### 7.1 Embedding Class

**File:** [`models/embed.py`](models/embed.py)
**Class:** `DataEmbedding_delta_pos`

```python
class DataEmbedding_delta_pos(nn.Module):
    def forward(self, x, x_mark):
        val  = self.value_embedding(x)            # [B, L, D]   X_i
        temp = self.temporal_embedding(x_mark)    # [B, L, D]   T_i

        delta = torch.zeros_like(val)
        delta[:, 0, :] = val[:, 0, :]            # Δ(X_0) = X_0
        delta[:, 1:, :] = val[:, 1:, :] - val[:, :-1, :]  # Δ(X_i) = X_i - X_{i-1}

        return self.dropout(val + delta + temp)   # [B, L, D]
```

### 7.2 Model Entry Point

**File:** [`models/model.py`](models/model.py)

```python
if pe_mode == 'vanilla':
    self.enc_embedding = DataEmbedding(...)           # sinusoidal PE
elif pe_mode == 'delta_pos':
    self.enc_embedding = DataEmbedding_delta_pos(...) # delta ordering
```

Activated via:

```bash
--pe_mode delta_pos
```

### 7.3 Files Modified vs. Baseline

| File | Status | Change |
|------|--------|--------|
| `models/embed.py` | **Modified** | Added `DataEmbedding_delta_pos`; baseline `DataEmbedding` kept for `pe_mode='vanilla'` |
| `models/model.py` | **Modified** | Added `pe_mode='delta_pos'` branch; everything else unchanged |
| `models/attn.py` | Unchanged | Identical to baseline |
| `models/encoder.py` | Unchanged | Identical to baseline |
| `models/decoder.py` | Unchanged | Identical to baseline |

---

## 8. Training Configuration

All hyperparameters match the baseline and other experiments in the ablation
series for strict comparability.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--model` | `informer` | |
| `--data` | `ETTh1` | ETT Hourly, dataset 1 |
| `--features` | `M` | Multivariate → multivariate |
| `--seq_len` | 96 | 96-step encoder context |
| `--label_len` | 48 | 48-step decoder warm-up |
| `--pred_len` | 24/48/96/192/336/720 | Phase 1 sweep |
| `--e_layers` | 2 | Encoder layers |
| `--d_layers` | 1 | Decoder layers |
| `--d_model` | 512 | Embedding dimension |
| `--n_heads` | 8 | Attention heads |
| `--d_ff` | 2048 | Feed-forward width |
| `--attn` | `full` | Full attention (not ProbSparse) |
| `--embed` | `timeF` | Continuous time features via linear layer |
| `--freq` | `h` | Hourly frequency → 4 time features |
| `--dropout` | 0.05 | |
| `--train_epochs` | 6 | |
| `--patience` | 3 | Early stopping |
| `--learning_rate` | 0.0001 | |
| `--batch_size` | 32 | |
| `--seed` | 2021 | |
| `--pe_mode` | `delta_pos` | **This experiment's switch** |

---

## 9. Code-to-Math Mapping

| Mathematical Expression | Code Location | Code Expression |
|------------------------|---------------|-----------------|
| $X_i = \mathrm{TokenEmbedding}(x_i)$ | `embed.py:DataEmbedding_delta_pos.forward` | `val = self.value_embedding(x)` |
| $\Delta(X_0) = X_0$ | `embed.py` line `delta[:, 0, :] = val[:, 0, :]` | boundary condition |
| $\Delta(X_i) = X_i - X_{i-1}$ | `embed.py` line `delta[:, 1:, :] = val[:, 1:, :] - val[:, :-1, :]` | vectorized consecutive diff |
| $T_i$ | `embed.py` | `temp = self.temporal_embedding(x_mark)` |
| $\tilde{X}_i = X_i + \Delta(X_i) + T_i$ | `embed.py` return | `val + delta + temp` |
| $\mathrm{Attn}(Q,K,V)$ | `attn.py:FullAttention.forward` | `softmax(QK^T/\sqrt{d}) \cdot V` — **unchanged** |

---

## 10. What This Experiment Does NOT Include

This experiment is **strictly pure delta ordering**:

- ❌ No Legendre (label) positional embedding
- ❌ No distance decay / α(i,j) weighting
- ❌ No normalization by sequence mean
- ❌ No pairwise all-to-all displacement (that was the archived exp4 variant)
- ❌ No modification to the attention mechanism
- ❌ No extra parameters or buffers

Any experiment that includes any of the above is a different, hybrid experiment
and cannot be compared directly to this one as a "pure delta ordering" baseline.

---

## 11. Expected Behaviour and Hypotheses

**Hypothesis H1 (ordering carries structure):** The consecutive delta encodes
local trend direction. Timesteps at peaks/troughs produce large-magnitude
deltas; flat regions produce near-zero deltas. If the transformer can exploit
this, it should learn forecasting patterns that are at least comparable to
absolute sinusoidal PE.

**Hypothesis H2 (scale instability):** Without normalization, the delta magnitude
scales with the embedding magnitude. This may cause the ordering signal to
dominate the semantic content $X_i$ at some input scales, degrading performance.
If H2 occurs, the normalized variant (Exp 4d / Exp 5c) is expected to perform
better.

**Hypothesis H3 (boundary artefact):** The first token's positional signal equals
its own content embedding ($\Delta(X_0) = X_0$). This may cause the model to
double-count the first token's content, potentially introducing a systematic
bias for the first position. This is an expected, bounded artefact of the
zero-pad boundary condition.

---

## 12. Relationship to Other Experiments in This Repository

| Experiment | Formula | Positional component | Semantic component |
|------------|---------|---------------------|-------------------|
| Baseline | $X_i + \mathrm{PE}_i + T_i$ | Sinusoidal PE | TokenEmbedding |
| **Exp 4c (this)** | $X_i + \Delta(X_i) + T_i$ | **Delta** | TokenEmbedding |
| Exp 4d | $X_i + \Delta(X_i)/\bar{x} + T_i$ | Normalized delta | TokenEmbedding |
| Exp 5c (positional normalized) | $X_i + T_i + \Delta(P_i)/\bar{p}$ | Normalized Legendre delta | TokenEmbedding |
| Exp 3 (label only) | $X_i + P_i + T_i$ | Legendre label | TokenEmbedding |
| Exp 2 (LOD full paper) | $X_i + T_i + P_i + O_i$ | Legendre + distance order | TokenEmbedding |

---

## 13. How to Run

### On Google Colab / remote GPU

```bash
bash experiments/exp4c_delta_ordering_pos_space/exp4c_delta_ordering_pos_space_ph1.sh
```

The script:
1. Copies all 6 model files to `Informer2020-original/models/`
2. Runs 6 prediction-horizon sweeps (`pred_len ∈ {24, 48, 96, 192, 336, 720}`)
3. Logs each run to `logs/exp4c_delta_ordering_pos_space_phase1/`
4. Prints an MSE/MAE summary table at the end
5. Is idempotent: already-completed runs are skipped automatically

### Manual single run (for debugging)

```bash
cd Informer2020-original
cp ../experiments/exp4c_delta_ordering_pos_space/models/*.py models/
python -u main_informer.py \
    --model informer --data ETTh1 \
    --root_path ./data/ETT/ --data_path ETTh1.csv \
    --features M --attn full \
    --seq_len 96 --label_len 48 --pred_len 96 \
    --e_layers 2 --d_layers 1 --factor 5 \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --d_model 512 --n_heads 8 --d_ff 2048 \
    --dropout 0.05 --embed timeF --freq h --activation gelu \
    --train_epochs 6 --patience 3 --learning_rate 0.0001 \
    --batch_size 32 --itr 1 \
    --pe_mode delta_pos \
    --des debug_exp4c_pred96
```
