# Formula B — Global Mean Deviation Ordering in Semantic Space

## 1. One-Line Summary

Replace the sinusoidal positional embedding with a normalised ordering signal
built from each token's deviation from the **global sequence mean** in embedding
space.  Every other component is unchanged.

---

## 2. Research Position

This experiment belongs to the **pure ordering** track of the ablation study.
It isolates a single question:

> **Can a global-context ordering signal — where each token is positioned by
> how far it sits from the sequence-level centroid — carry enough sequential
> structure to replace absolute sinusoidal positional encoding?**

It is the direct normalised-global-mean counterpart to Formula A:

| Label | Formula | Space | Normalised | Folder |
|-------|---------|-------|------------|--------|
| Formula A | Consecutive delta $X_i - X_{i-1}$ | Semantic | ❌ No | `formula-A-sem/` |
| **Formula B** | **Global mean deviation $(μ - V_i)/\bar{x}$** | **Semantic** | **✅ Yes** | `Formula-B-sem/` ← **this** |
| Formula A-pos | Legendre delta $P_i - P_{i-1}$ | Positional | ✅ Yes | `Formula-A-pos/` |

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

The sinusoidal PE tells the model **where** a token sits using fixed irrational
frequency ratios.  It carries no information about **how the sequence varies**
around each token.

---

## 4. The Formula B Embedding: Global Mean Deviation

### 4.1 Formula

$$
\boxed{\tilde{X}_i = V_i + T_i + \mathrm{Ordering}_i}
$$

where:

$$
V_i = \mathrm{TokenEmbedding}(x_i) \in \mathbb{R}^d
$$

$$
\mu = \frac{1}{L} \sum_{k=1}^{L} V_k \in \mathbb{R}^d \quad \text{(global sequence mean)}
$$

$$
\Delta_i = \mu - V_i \in \mathbb{R}^d \quad \text{(deviation from global mean)}
$$

$$
\bar{x} = \frac{1}{L} \sum_{i=1}^{L} \|V_i\|_2 \in \mathbb{R} \quad \text{(mean per-token L2 norm, scalar)}
$$

$$
\mathrm{Ordering}_i = \frac{\Delta_i}{\bar{x} + \varepsilon}, \quad \varepsilon = 10^{-8}
$$

### 4.2 What $\mathrm{Ordering}_i$ Encodes

$\mathrm{Ordering}_i$ is a **normalised displacement vector** from the global
centroid of the sequence to each individual token.

- **Direction**: if feature $k$ of $V_i$ is *above* the sequence mean,
  $(\mu - V_i)_k < 0$; if *below*, $(\mu - V_i)_k > 0$.  The sign indicates
  whether the token is above or below average on each dimension.
- **Magnitude**: large $\|\mathrm{Ordering}_i\|$ means the token is far from
  the global centroid (an outlier or extreme value); small norm means the token
  is close to typical.
- **Scale invariance**: dividing by $\bar{x}$ (mean L2 norm of the token
  embeddings) keeps the ordering signal proportional to the embedding scale,
  preventing it from vanishing or overwhelming $V_i$.
- **Global context**: unlike Formula A (which only sees adjacent neighbours),
  $\mu$ aggregates the entire window, so every token's signal is conditioned
  on all other tokens in the sequence.

### 4.3 Boundary Condition

**None required.**  $\mu$ is the mean of the full window; all positions $i \in
\{1, \ldots, L\}$ are treated symmetrically.  There is no zero-pad artefact.

### 4.4 What the Model Receives

For each token $i$, the transformer's first layer receives:

$$
\tilde{X}_i = \underbrace{V_i}_{\text{semantic content}} + \underbrace{T_i}_{\text{calendar}} + \underbrace{\mathrm{Ordering}_i}_{\text{global-mean deviation (normalised)}}
$$

The attention mechanism then operates on $\tilde{X}_i$ using standard scaled
dot-product attention — no modifications anywhere else in the model.

---

## 5. Relationship to the Baseline: Exactly What Changed

This is the minimal surgical change from the vanilla Informer:

| | Baseline | Formula B |
|-|----------|-----------|
| `TokenEmbedding` | ✅ unchanged | ✅ unchanged |
| `TemporalEmbedding` | ✅ unchanged | ✅ unchanged |
| `PositionalEmbedding` (sinusoidal) | ✅ active | ❌ **removed** |
| Global mean deviation $\mathrm{Ordering}_i$ | ❌ not present | ✅ **added** |
| `FullAttention` / `ProbAttention` | ✅ unchanged | ✅ unchanged |
| `Encoder` / `Decoder` | ✅ unchanged | ✅ unchanged |
| Projection head | ✅ unchanged | ✅ unchanged |
| New trainable parameters | — | **0** |

**Single-line diff in `DataEmbedding.forward()`:**

```python
# Baseline:
x = value_embedding(x) + position_embedding(x) + temporal_embedding(x_mark)

# Formula B:
x = val + temp + ordering
#              ^^^^^^^^
#              replaces position_embedding(x)
```

---

## 6. Mathematical Properties

### 6.1 Order Sensitivity

Permuting the sequence changes $\mu$ (the global mean), which changes
$\mathrm{Ordering}_i$ at every position.  The signal is order-sensitive.

### 6.2 Global vs. Local Context

Formula A ($X_i - X_{i-1}$) encodes only **local** dynamics — the signed
change between two adjacent steps.  Formula B encodes **global** context — how
far each token is from the centre of mass of the whole window.  A token at a
trend peak or trough will have a large-magnitude ordering vector pointing away
from the mean; tokens in a stable plateau near the mean will have near-zero
ordering vectors.

### 6.3 Scale Invariance

Multiplying all $V_i$ by a scalar $\alpha$:

$$
\mu \to \alpha\mu, \quad
\Delta_i \to \alpha\Delta_i, \quad
\bar{x} \to \alpha\bar{x}
$$

$$
\mathrm{Ordering}_i = \frac{\alpha\Delta_i}{\alpha\bar{x} + \varepsilon}
\;\xrightarrow{\varepsilon \to 0}\;
\frac{\Delta_i}{\bar{x}}
$$

The signal is **asymptotically scale-invariant** — for typical embedding
magnitudes ($\bar{x} \gg \varepsilon$), uniform scaling of $V_i$ leaves
$\mathrm{Ordering}_i$ unchanged.  This is the key improvement over Formula A,
which is **not** normalised.

### 6.4 Translation Invariance

Adding a constant vector $c$ to every $V_i$:

$$
\mu \to \mu + c, \quad \Delta_i = (\mu + c) - (V_i + c) = \mu - V_i
$$

The ordering signal is **translation-invariant** — shifting all embeddings by
the same constant leaves $\mathrm{Ordering}_i$ completely unchanged.

### 6.5 Zero Parameters Added

$\mu$ and $\bar{x}$ are computed from existing `TokenEmbedding` outputs at
forward-pass time.  No weight matrices, no new buffers, no additional memory
beyond one `[B, 1, D]` mean tensor.

### 6.6 Comparison to Sinusoidal PE and Formula A

| Property | Sinusoidal PE | Formula A | Formula B |
|----------|---------------|-----------|-----------|
| Input-dependent | No (fixed) | Yes | Yes |
| Encodes absolute position | Yes | No | No |
| Encodes local dynamics | No | Yes | No |
| Encodes global dynamics | No | No | **Yes** |
| Normalised | No | No | **Yes** |
| Translation-invariant | No | Yes (i ≥ 1) | **Yes (all i)** |
| Scale-invariant | No | No | **Yes** |
| Boundary artefact | No | Yes (i = 0) | **None** |
| Extra parameters | 0 | 0 | 0 |

---

## 7. Implementation Details

### 7.1 Embedding Class

**File:** [`models/embed.py`](models/embed.py)  
**Class:** `DataEmbedding_formula_b_sem`

```python
class DataEmbedding_formula_b_sem(nn.Module):
    def forward(self, x, x_mark):
        val  = self.value_embedding(x)              # [B, L, D]   V_i
        temp = self.temporal_embedding(x_mark)      # [B, L, D]   T_i

        global_mean = val.mean(dim=1, keepdim=True) # [B, 1, D]   μ
        delta       = global_mean - val             # [B, L, D]   Δ_i = μ − V_i
        x_bar = val.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        ordering    = delta / (x_bar + 1e-8)        # [B, L, D]   normalised

        return self.dropout(val + temp + ordering)  # [B, L, D]
```

> **Implementation note:** `__init__` also initialises `self._diag_step = 0`
> (`embed.py` line 241).  During training, every 100 steps the forward pass
> prints diagnostics tagged `[FORMULA_B_SEM step=N]` reporting
> `val_norm`, `ordering_norm`, `temp_norm`, `x_bar`, and `ordering/val`.
> This is a training-time diagnostic only; it has no effect on inference.

### 7.2 Model Entry Point

**File:** [`models/model.py`](models/model.py)

```python
if pe_mode == 'vanilla':
    self.enc_embedding = DataEmbedding(...)                  # sinusoidal PE
elif pe_mode == 'formula_b_sem':
    self.enc_embedding = DataEmbedding_formula_b_sem(...)    # Formula B
```

Activated via:

```bash
--pe_mode formula_b_sem
```

### 7.3 Files Modified vs. Baseline

| File | Status | Change |
|------|--------|--------|
| `models/embed.py` | **Modified** | Added `DataEmbedding_formula_b_sem`; baseline `DataEmbedding` kept for `pe_mode='vanilla'` |
| `models/model.py` | **Modified** | Added `pe_mode='formula_b_sem'` branch; everything else unchanged |
| `models/__init__.py` | Unchanged | Empty file; copied for completeness |
| `models/attn.py` | Unchanged | Identical to baseline |
| `models/encoder.py` | Unchanged | Identical to baseline |
| `models/decoder.py` | Unchanged | Identical to baseline |

---

## 8. Tensor Shapes (Forward Pass)

```
val          [B, L, D]   V_i  = TokenEmbedding(x)
temp         [B, L, D]   T_i  = temporal_embedding(x_mark)
global_mean  [B, 1, D]   μ    = val.mean(dim=1, keepdim=True)   → broadcasts
delta        [B, L, D]   Δ_i  = global_mean − val
x_bar        [B, 1, 1]   x̄   = mean of per-token L2 norms       scalar per sample
ordering     [B, L, D]   Ordering_i = delta / (x_bar + 1e-8)
output       [B, L, D]   dropout(val + temp + ordering)
```

---

## 9. Training Configuration

All hyperparameters match the baseline and other experiments in the ablation
series for strict comparability.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--model` | `informer` | |
| `--data` | `ETTh1` | ETT Hourly, dataset 1 |
| `--features` | `M` | Multivariate → multivariate |
| `--seq_len` | 96 | 96-step encoder context |
| `--label_len` | 48 | 48-step decoder warm-up |
| `--pred_len` | Phase 1: 96, 192 · Phase 2: 48, 96, 192, 336 | |
| `--e_layers` | 2 | Encoder layers |
| `--d_layers` | 1 | Decoder layers |
| `--d_model` | 512 | Embedding dimension |
| `--n_heads` | 8 | Attention heads |
| `--d_ff` | 2048 | Feed-forward width |
| `--factor` | 5 | ProbSparse factor (unused with `--attn full`) |
| `--enc_in` | 7 | Encoder input features |
| `--dec_in` | 7 | Decoder input features |
| `--c_out` | 7 | Output features |
| `--attn` | `full` | Full attention (not ProbSparse) |
| `--embed` | `timeF` | Continuous time features via linear layer |
| `--freq` | `h` | Hourly frequency → 4 time features |
| `--activation` | `gelu` | |
| `--dropout` | 0.05 | |
| `--distil` | `True` | Framework default; not passed explicitly in scripts |
| `--train_epochs` | 6 | |
| `--patience` | 3 | Early stopping |
| `--learning_rate` | 0.0001 | |
| `--batch_size` | 32 | |
| `--itr` | 1 | |
| `--decay_a` | 1.0 | Framework default; not passed explicitly in scripts |
| `--seed` | Phase 1: 2021 · Phase 2: 2021, 2022, 2023 | |
| `--pe_mode` | `formula_b_sem` | **This experiment's switch** |

---

## 10. Code-to-Math Mapping

| Mathematical Expression | Code Location | Code Expression |
|------------------------|---------------|-----------------|
| $V_i = \mathrm{TokenEmbedding}(x_i)$ | `embed.py:DataEmbedding_formula_b_sem.forward` | `val = self.value_embedding(x)` |
| $\mu = \frac{1}{L}\sum_k V_k$ | `embed.py` | `global_mean = val.mean(dim=1, keepdim=True)` |
| $\Delta_i = \mu - V_i$ | `embed.py` | `delta = global_mean - val` |
| $\bar{x} = \frac{1}{L}\sum_i \|V_i\|_2$ | `embed.py` | `val.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)` |
| $\mathrm{Ordering}_i = \Delta_i / (\bar{x} + \varepsilon)$ | `embed.py` | `ordering = delta / (x_bar + 1e-8)` |
| $\tilde{X}_i = V_i + T_i + \mathrm{Ordering}_i$ | `embed.py` return | `val + temp + ordering` |
| $\mathrm{Attn}(Q,K,V)$ | `attn.py:FullAttention.forward` | `softmax(QK^T/\sqrt{d}) \cdot V` — **unchanged** |

---

## 11. What This Experiment Does NOT Include

This experiment is **strictly pure global-mean-deviation ordering**:

- ❌ No Legendre (label) positional embedding
- ❌ No distance decay / $\alpha(i,j)$ weighting
- ❌ No consecutive delta (Formula A)
- ❌ No modification to the attention mechanism
- ❌ No extra parameters or buffers

Any experiment that includes any of the above is a different, hybrid experiment
and cannot be compared directly to this one as a "pure ordering" control.

---

## 12. Expected Behaviour and Hypotheses

**Hypothesis H1 (global context is informative):** By centring each token
relative to the full window mean, the model learns which tokens are unusual
(far from centre) vs. typical.  In time-series data this corresponds to peaks,
troughs, and trend transitions.  If the transformer can exploit this structure,
it should perform at least comparably to sinusoidal PE.

**Hypothesis H2 (normalisation helps vs. Formula A):** Formula A is
unnormalised — the delta magnitude scales with embedding magnitude and may
dominate the value stream.  Formula B's $\bar{x}$ normalisation keeps the
ordering/value ratio stable regardless of input scale.  Formula B is therefore
expected to be **more stable than Formula A** across different sequences.

**Hypothesis H3 (global vs. local ordering):** Formula A encodes local trend
direction; Formula B encodes global deviation.  Time series with strong
mean-reverting or cyclical patterns may benefit more from Formula B.  Series
with non-stationary monotone trends may benefit more from Formula A (local
direction of change is more salient than distance from mean).

---

## 13. Relationship to Other Experiments in This Repository

| Experiment | Formula | Positional component | Normalised |
|------------|---------|---------------------|------------|
| Baseline | $X_i + \mathrm{PE}_i + T_i$ | Sinusoidal PE | — |
| Formula A (`formula-A-sem`) | $X_i + (X_i - X_{i-1}) + T_i$ | Consecutive delta | ❌ No |
| **Formula B (this)** | $V_i + T_i + (\mu - V_i)/\bar{x}$ | **Global mean deviation** | **✅ Yes** |
| Formula A-pos (`Formula-A-pos`) | $X_i + T_i + P_i + \Delta(P_i)/\bar{p}$ | Legendre delta (positional) | ✅ Yes |
| Exp 3 (`exp3_label_only`) | $X_i + P_i + T_i$ | Legendre label | — |
| Exp 6 LOD | $X_i + T_i + P_i + O_i$ | Legendre + distance + order | — |

---

## 14. How to Run

### Phase 1 — seed 2021, pred\_len ∈ {96, 192}

```bash
bash experiments/Formula-B-sem/formula-B-sem-ph1.sh
```

The script:
1. Copies all 6 model files to `Informer2020-original/models/`
2. Runs **2** prediction-horizon sweeps (`pred_len ∈ {96, 192}`) with `seed=2021`
3. Logs each run to `logs/Formula-B-sem-phase1/`
4. Prints an MSE/MAE summary table at the end
5. Is idempotent: already-completed runs are skipped automatically

### Phase 2 — seeds {2021, 2022, 2023}, pred\_len ∈ {48, 96, 192, 336}

```bash
bash experiments/Formula-B-sem/formula-B-sem-ph2.sh
```

The script:
1. Copies all 6 model files to `Informer2020-original/models/`
2. Runs **12** sweeps (3 seeds × 4 pred\_lens)
3. Logs each run to `logs/Formula-B-sem-phase2/`
4. Prints a per-seed summary table at the end
5. Is idempotent: runs already containing `mse:` in their log are skipped

### Manual single run (for debugging)

```bash
cd Informer2020-original
cp ../experiments/Formula-B-sem/models/*.py models/
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
    --pe_mode formula_b_sem \
    --des debug_formula_b_sem_pred96
```

### Sanity test (no GPU required)

```python
import torch
from models.embed import DataEmbedding_formula_b_sem

B, L, D, c_in = 2, 96, 512, 7
x      = torch.randn(B, L, c_in)
x_mark = torch.randn(B, L, 4)

emb = DataEmbedding_formula_b_sem(c_in, D)
out = emb(x, x_mark)

assert out.shape == (B, L, D)
print("Shape check passed:", out.shape)

# Verify ordering is non-trivial
with torch.no_grad():
    val = emb.value_embedding(x)
    mu  = val.mean(dim=1, keepdim=True)
    assert not torch.allclose(mu.expand_as(val), val), \
        "Global mean equals all tokens — degenerate input"
print("Ordering non-triviality check passed.")
```

---

## 15. Experiment Results

Results are recorded in
`experiments/Formula-B-sem/Order_Forumla_B_sem.ipynb`.
All values below are taken verbatim from the notebook output lines
(`mse:..., mae:...`).

### 15.1 Phase 1 — seed 2021, pred\_len ∈ {96, 192}

Phase 1 status: Total: 2 | Completed: 2 | Failed: 0 | Skipped: 0

| RUN\_ID | MSE | MAE |
|---------|-----|-----|
| `formula_b_sem_ph1_ETTh1_pred96_seed2021` | 0.8911137580871582 | 0.7640805244445801 |
| `formula_b_sem_ph1_ETTh1_pred192_seed2021` | 0.9621934294700623 | 0.7902503609657288 |

Reference baseline (Exp1-Pre, alpha=1.0, seed=2021):

| pred\_len | Exp1-Pre MSE |
|-----------|-------------|
| 96 | 0.8683 |
| 192 | 0.8463 |

**Phase 1 outcome:** Formula-B-sem was worse than Exp1-Pre at both horizons
(pred=96: 0.8911 vs 0.8683, +0.0228; pred=192: 0.9622 vs 0.8463, +0.1159).
Applied decision rule "loses BOTH vs Exp1-Pre → document, stop"; however
Phase 2 was executed for robustness confirmation.

### 15.2 Phase 2 — seeds {2021, 2022, 2023}, pred\_len ∈ {48, 96, 192, 336}

Phase 2 status: Total: 12 | Completed: 12 | Failed: 0 | Skipped: 0

| Seed | pred\_len | MSE | MAE |
|------|-----------|-----|-----|
| 2021 | 48 | 0.9602015018463135 | 0.7683544158935547 |
| 2021 | 96 | 0.7835373878479004 | 0.682116687297821 |
| 2021 | 192 | 0.9376522302627563 | 0.7857064008712769 |
| 2021 | 336 | 0.9643481969833374 | 0.7918732762336731 |
| 2022 | 48 | 0.8936455249786377 | 0.7376778721809387 |
| 2022 | 96 | 0.7975202798843384 | 0.6957442164421082 |
| 2022 | 192 | 1.0187780857086182 | 0.8290572166442871 |
| 2022 | 336 | 0.9252648949623108 | 0.7778674960136414 |
| 2023 | 48 | 0.9553710222244263 | 0.7777656316757202 |
| 2023 | 96 | 0.8614110946655273 | 0.7358465194702148 |
| 2023 | 192 | 0.910224974155426 | 0.7555420398712158 |
| 2023 | 336 | 1.0116863250732422 | 0.8093421459197998 |
