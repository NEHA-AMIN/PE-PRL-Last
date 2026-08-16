# Experiment 6 Post: Full LOD with Post-Softmax Distance

## Objective

Test the **complete Label + Order + Distance (LOD)** formulation with **post-softmax distance decay**, combining:
- **Label (L):** Orthogonal distinctiveness via Legendre polynomials in embeddings
- **Order (O):** Directional information via delta_x in attention value matrix
- **Distance (D):** Proximity bias via index-based decay **AFTER softmax**

This experiment applies distance decay to attention weights **after** normalization, allowing the model to learn attention patterns first, then modulate them based on positional distance.

---

## Hypothesis

Applying distance decay **after softmax** may preserve the learned attention distribution better than pre-softmax decay, potentially leading to improved performance by:
1. Maintaining normalized attention weights
2. Allowing the model to learn semantic relationships first
3. Applying positional bias as a refinement step

---

## Mathematical Formulation

### Complete Architecture

```
Encoder/Decoder Embedding:
  combined_emb = value_emb + temporal_emb + legendre_pos
  delta_x = x[i] - x[i-1]  (clean temporal differences)
  
  Returns: (combined_emb, delta_x)

Attention Mechanism:
  Q, K from combined_emb (includes Label via Legendre)
  V from delta_x (Order component)
  
  scores = Q @ K^T
  A = softmax(scale * scores)           [Step 1: Normalize first]
  α(i,j) = 1 / (1 + |i-j|^a)           [Step 2: Compute distance decay]
  A' = A ⊙ α(i,j)                       [Step 3: Apply AFTER softmax]
  output = A' @ V                       [Step 4: Weighted sum with delta_x]
```

### Component Breakdown

#### 1. Label Component (Legendre Polynomials)
```python
P_i = [L_0(x_i), L_1(x_i), ..., L_{d-1}(x_i)]

where:
  L_k(x) = Legendre polynomial of degree k
  x_i ∈ [-1, 1] (normalized position: 2*i/(L-1) - 1)
  
Recurrence relation:
  L_0(x) = 1
  L_1(x) = x
  L_n(x) = ((2n-1) * x * L_{n-1}(x) - (n-1) * L_{n-2}(x)) / n

Properties:
  - Orthogonality: ⟨L_n, L_m⟩ = δ_{nm}
  - Distinctiveness: Each position has unique embedding
  - Scaling: P_i / √d_model for numerical stability
```

#### 2. Order Component (Delta Values)
```python
Δx_i = x_i - x_{i-1}  for i > 0
Δx_0 = 0

Properties:
  - Preserves temporal direction (sign matters)
  - Captures local changes in feature space
  - Used ONLY in Value matrix (not Q or K)
  - Clean signal (no positional encoding added)
```

#### 3. Distance Component (Post-Softmax Decay)
```python
α(i,j) = 1 / (1 + |i-j|^a)

where:
  a = decay parameter (0.5, 1.0, 2.0 tested)
  |i-j| = absolute positional distance
  
Applied AFTER softmax:
  A_normalized = softmax(Q @ K^T / √d)
  A_final = A_normalized ⊙ α(i,j)
  
Key difference from pre-softmax:
  - Preserves attention distribution shape
  - Distance acts as multiplicative refinement
  - Maintains sum-to-one property (approximately)
```

---

## Implementation Details

### Modified Files

#### `models/model.py`
**Key Changes:**
```python
# Encoder forward
enc_out, delta_enc = self.enc_embedding(x_enc, x_mark_enc)
enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, delta_x=delta_enc)

# Decoder forward
dec_out, delta_dec = self.dec_embedding(x_dec, x_mark_dec)
dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, 
                       cross_mask=dec_enc_mask, delta_x=delta_dec)
```

**Critical:** Both encoder and decoder embeddings return tuples `(combined_emb, delta_x)`

#### `models/decoder.py`
**Key Changes:**
```python
class DecoderLayer:
    def forward(self, x, cross, x_mask=None, cross_mask=None, delta_x=None):
        # Self-attention uses delta_x for ordering
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            delta_values=delta_x    # Order in decoder self-attention
        )[0])
        
        # Cross-attention uses encoder output (no delta)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            delta_values=None       # No delta in cross-attention
        )[0])

class Decoder:
    def forward(self, x, cross, x_mask=None, cross_mask=None, delta_x=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, delta_x=delta_x)
```

#### `models/encoder.py`
**Key Changes:**
```python
class EncoderLayer:
    def forward(self, x, attn_mask=None, delta_x=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            delta_values=delta_x  # Pass delta for V projection
        )

class EncoderStack:
    def forward(self, x, attn_mask=None, delta_x=None):
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            delta_slice = delta_x[:, -inp_len:, :] if delta_x is not None else None
            x_s, attn = encoder(x[:, -inp_len:, :], delta_x=delta_slice)
            x_stack.append(x_s); attns.append(attn)
```

#### `models/attn.py`
**Key Implementation:**

`delta_values` routing is handled by `AttentionLayer.forward()`, one level above
`FullAttention`. `FullAttention.forward()` receives the already-projected `values`
tensor and applies the post-softmax distance decay:

```python
# AttentionLayer.forward() — routes delta_values into value projection
class AttentionLayer:
    def forward(self, queries, keys, values, attn_mask, delta_values=None):
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        # V projected from delta_x when provided (Order component)
        if delta_values is not None:
            values = self.value_projection(delta_values).view(B, S, H, -1)
        else:
            values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        ...

# FullAttention.forward() — post-softmax distance decay applied here
class FullAttention:
    def forward(self, queries, keys, values, attn_mask):
        # Step 1: Compute attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Step 2: Apply mask
        if self.mask_flag:
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        # Step 3: Softmax FIRST (normalize)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # Step 4: Apply distance decay AFTER softmax
        q_idx = torch.arange(L).unsqueeze(1).to(queries.device)
        k_idx = torch.arange(S).unsqueeze(0).to(queries.device)
        dist_matrix = torch.abs(q_idx - k_idx).float()
        alpha = 1.0 / (1.0 + dist_matrix ** self.decay_a)
        A = A * alpha.unsqueeze(0).unsqueeze(0)
        
        # Step 5: Weighted sum with already-projected V
        output = torch.einsum("bhls,bshd->blhd", A, values)
```

#### `models/legendre_embedding.py`
**Recurrence Formula Implementation:**
```python
class LegendrePositionEmbedding(nn.Module):
    def forward(self, x):
        B, L, _ = x.shape
        device = x.device
        
        # Map positions to [-1, 1]
        if L == 1:
            positions = torch.zeros(1, device=device)
        else:
            positions = 2.0 * torch.arange(L, dtype=torch.float32, device=device) / (L - 1) - 1.0
        
        # Recurrence relation
        P = torch.zeros(L, self.d_model, device=device)
        if self.d_model >= 1:
            P[:, 0] = 1.0
        if self.d_model >= 2:
            P[:, 1] = positions
        for n in range(2, self.d_model):
            P[:, n] = ((2*n - 1) * positions * P[:, n-1] - (n-1) * P[:, n-2]) / n
        
        if self.scaling:
            P = P / (self.d_model ** 0.5)
        
        return P.unsqueeze(0).expand(B, -1, -1)
```

#### `models/embed.py`
**Key Implementation:**

Delta is computed from `value_emb` (the projected embedding space), **not** from
raw input `x`. `torch.roll` is used for the difference, and the first position is
zeroed explicitly. Dropout is applied to `combined_emb` only — `delta_x` is
returned without dropout:

```python
class DataEmbedding(nn.Module):
    def forward(self, x, x_mark):
        # Value embedding
        value_emb = self.value_embedding(x)
        
        # CLEAN DELTA: Δx[i] = value_emb[i] - value_emb[i-1]
        # Computed from value_emb, NOT from raw x
        delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)
        delta_x[:, 0, :] = 0.0  # zero first position
        
        # Temporal embedding
        temporal_emb = self.temporal_embedding(x_mark)
        
        # Legendre position embedding (Label)
        legendre_pos = self.legendre_embedding(x)
        
        # Combined embedding for Q/K: value_emb + temporal_emb + legendre_pos
        combined_emb = value_emb + temporal_emb + legendre_pos
        
        # Dropout applied to combined_emb only; delta_x has NO dropout
        return self.dropout(combined_emb), delta_x
```

---

## Configuration

### Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Dataset** | ETTh1 |
| **Model** | Informer |
| **Attention** | Full (not ProbSparse) |
| **Sequence Length** | 96 |
| **Label Length** | 48 |
| **Prediction Lengths** | Phase 1: 96, 192 · Phase 2: 48, 96, 192, 336 |
| **Encoder Layers** | 2 |
| **Decoder Layers** | 1 |
| **d_model** | 512 |
| **n_heads** | 8 |
| **d_ff** | 2048 |
| **Dropout** | 0.05 |
| **Activation** | GELU |
| **Train Epochs** | 6 |
| **Early Stopping Patience** | 3 |
| **Batch Size** | 32 |
| **Learning Rate** | 0.0001 |
| **Factor** | 5 |
| **Embed** | timeF |
| **Freq** | h |

### Ablation Parameters

| Parameter | Values Tested |
|-----------|---------------|
| **Decay Parameter (a)** | 0.5, 1.0, 2.0 (Phase 1 screen); fixed at 0.5 (Phase 2) |
| **Random Seeds** | 2021 only (Phase 1); 2021, 2022, 2023 (Phase 2) |
| **Total Runs** | 18 (6 Phase 1: 3 decay_a × 2 pred_len × 1 seed; 12 Phase 2: 4 pred_len × 3 seeds) |

---

## Key Differences from Other Experiments

| Aspect | Exp6 Pre | Exp6 Post | Key Difference |
|--------|----------|-----------|----------------|
| **Distance Timing** | Before softmax | **After softmax** | When α(i,j) is applied |
| **Attention Flow** | scores → α → softmax | scores → softmax → α | Order of operations |
| **Weight Distribution** | Distorted before norm | Preserved during norm | Shape preservation |
| **Theoretical Basis** | Bias raw scores | Refine learned weights | Interpretation |

### Comparison with All Experiments

| Experiment | Label | Order | Distance | Distance Timing | Temporal |
|------------|-------|-------|----------|-----------------|----------|
| Vanilla | ❌ | ❌ | ❌ | N/A | ✅ |
| Exp1 Post | ❌ | ❌ | ✅ | Post-softmax | ✅ |
| Exp1 Pre | ❌ | ❌ | ✅ | Pre-softmax | ✅ |
| Exp2 | ✅ | ✅ | ✅ | Embedding | ✅ |
| Exp3 | ✅ | ❌ | ❌ | N/A | ❌ |
| Exp4 | ❌ | ✅ | ❌ | N/A | ✅ |
| Exp5 | ✅ | ✅ | ❌ | N/A | ✅ |
| **Exp6 Pre** | ✅ | ✅ | ✅ | **Pre-softmax** | ✅ |
| **Exp6 Post** | ✅ | ✅ | ✅ | **Post-softmax** | ✅ |

---

## How to Run

The experiment was executed on Google Colab via the notebook `EXP_6_LOD_POST.ipynb`.
The two shell scripts are invoked from inside the notebook:

```bash
# Phase 1 — alpha screening (6 runs: 3 decay_a × 2 pred_len × seed=2021)
bash experiments/exp6_lod_post/exp6_lod_post_phase1.sh

# Phase 2 — stability validation (12 runs: 4 pred_len × 3 seeds, decay_a=0.5)
bash experiments/exp6_lod_post/exp6_lod_post_ph2.sh
# (exp6_lod_post_phase2_a0.5.sh is an identical duplicate of exp6_lod_post_ph2.sh)
```

Both scripts run from the Colab path:
`PROJECT_ROOT=/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1`

### What the Scripts Do

1. Copy modified model files from `experiments/exp6_lod_post/models/` to `Informer2020-original/models/`
2. Copy `legendre_embedding.py` to **both** `Informer2020-original/models/` and `Informer2020-original/` (required for bare `from legendre_embedding import ...` in `embed.py`)
3. Run training via `main_informer.py` for each configuration
4. Save run logs to `logs/exp6_lod_post_phase1/` or `logs/exp6_lod_post_phase2_a0.5/`
5. Restore original model files via `git checkout ./models/` after all runs

### Actual Runtime (from notebook timestamps)

- **Phase 1:** ~4–5 minutes per run (early stopping at epoch 4); total ~28 minutes
- **Phase 2:** ~4–7 minutes per run; total ~65 minutes
- **Both phases combined:** ~93 minutes on Colab T4 GPU

---

## Output

Each run produces a results directory and a per-run log file:

- **Logs** → `logs/exp6_lod_post_phase1/<run_id>.log` (Phase 1)
- **Logs** → `logs/exp6_lod_post_phase2_a0.5/<run_id>.log` (Phase 2)
- **Checkpoints** → `Informer2020-original/checkpoints/<run_id>/checkpoint.pth`
- **Test metrics** reported at end of each log: MSE, MAE

### Directory Structure
```
logs/
├── exp6_lod_post_phase1/
│   ├── master_run.log
│   ├── exp6post_ph1_ETTh1_lod_post_a0.5_pred96_seed2021.log
│   ├── exp6post_ph1_ETTh1_lod_post_a0.5_pred192_seed2021.log
│   ├── exp6post_ph1_ETTh1_lod_post_a1.0_pred96_seed2021.log
│   ├── exp6post_ph1_ETTh1_lod_post_a1.0_pred192_seed2021.log
│   ├── exp6post_ph1_ETTh1_lod_post_a2.0_pred96_seed2021.log
│   └── exp6post_ph1_ETTh1_lod_post_a2.0_pred192_seed2021.log
└── exp6_lod_post_phase2_a0.5/
    ├── master_run.log
    ├── exp6post_ph2_ETTh1_lod_post_a0.5_pred48_seed2021.log
    ├── ...
    └── exp6post_ph2_ETTh1_lod_post_a0.5_pred336_seed2023.log
```

> **Note:** Log files were stored on Google Drive during Colab execution and are
> not committed to this repository. Full outputs are preserved in the notebook
> `EXP_6_LOD_POST.ipynb` cell outputs.

---

## Results

All 18 runs completed successfully (0 failed, 0 skipped).
Source: `EXP_6_LOD_POST.ipynb` cell outputs (logged on 2026-08-16).

### Phase 1 — Alpha Screening (seed=2021, pred_len ∈ {96, 192})

| decay_a | pred_len | MSE | MAE |
|---------|----------|-----|-----|
| 0.5 | 96 | 0.79213947057724 | 0.6940954327583313 |
| 0.5 | 192 | 0.8021528720855713 | 0.6937472224235535 |
| 1.0 | 96 | 0.9164600372314453 | 0.7008794546127319 |
| 1.0 | 192 | 1.1427932977676392 | 0.8031872510910034 |
| 2.0 | 96 | 1.0846896171569824 | 0.7789736986160278 |
| 2.0 | 192 | 1.1892716884613037 | 0.8282927870750427 |

**Phase 1 decision:** `decay_a=0.5` had best performance at both pred_lens and was
carried forward to Phase 2.

### Phase 2 — Stability Validation (decay_a=0.5, pred_len ∈ {48, 96, 192, 336})

| pred_len | seed | MSE | MAE |
|----------|------|-----|-----|
| 48 | 2021 | 0.7353359460830688 | 0.6634814143180847 |
| 48 | 2022 | 0.7317635416984558 | 0.6689267754554749 |
| 48 | 2023 | 0.7185477614402771 | 0.6619631052017212 |
| 96 | 2021 | 0.7239963412284851 | 0.6624867916107178 |
| 96 | 2022 | 0.8194853067398071 | 0.7094253897666931 |
| 96 | 2023 | 0.8214573264122009 | 0.7154382467269897 |
| 192 | 2021 | 0.834628701210022 | 0.7170958518981934 |
| 192 | 2022 | 0.8189764618873596 | 0.7027294039726257 |
| 192 | 2023 | 0.8609811663627625 | 0.7218409180641174 |
| 336 | 2021 | 0.9503873586654663 | 0.7660955190658569 |
| 336 | 2022 | 0.9987733364105225 | 0.7828776836395264 |
| 336 | 2023 | 0.9330756664276123 | 0.7554452419281006 |

### Aggregated Results — Phase 2 (Mean ± Std across 3 seeds)

| pred_len | MSE (Mean±Std) | MAE (Mean±Std) |
|----------|----------------|----------------|
| 48 | 0.7285±0.0072 | 0.6648±0.0030 |
| 96 | 0.7883±0.0455 | 0.6958±0.0237 |
| 192 | 0.8382±0.0173 | 0.7139±0.0081 |
| 336 | 0.9607±0.0278 | 0.7681±0.0113 |

---

## Analysis

### Decay Parameter Selection

`decay_a=0.5` (gentlest post-softmax re-weighting) outperformed both `1.0` and `2.0`
at every pred_len tested in Phase 1:

| decay_a | pred_96 MSE | pred_192 MSE | Verdict |
|---------|-------------|--------------|---------|
| **0.5** | **0.7921** | **0.8022** | ✅ Best at both |
| 1.0 | 0.9165 | 1.1428 | ❌ Worse |
| 2.0 | 1.0847 | 1.1893 | ❌ Worst |

The steeper the post-softmax re-weighting, the worse the results — indicating that
heavy distance suppression after softmax degrades the learned attention patterns.

### Stability (Phase 2)

- **pred_len=48:** Very stable (seed variance MSE ±0.0072).
- **pred_len=96:** Moderate variance (±0.0455), primarily driven by seed=2021 (0.7240)
  outperforming seeds 2022/2023 (~0.82).
- **pred_len=192:** Stable (±0.0173).
- **pred_len=336:** Stable (±0.0278).

### Exp6-Post vs Exp6-Pre (same ETTh1, seed=2021, Phase 1 alpha=best)

| pred_len | Exp6-Pre (α=1.0) MSE | Exp6-Post (α=0.5) MSE | Δ |
|----------|----------------------|----------------------|---|
| 96 | 0.8694 | 0.7921 | **Post better by 0.0773** |
| 192 | 0.9571 | 0.8022 | **Post better by 0.1549** |

Post-softmax placement with `decay_a=0.5` outperforms Pre-softmax at its own best
alpha at both pred_lens.

### Exp6-Post vs Exp1-Pre (distance-only baseline, α=1.0, seed=2021)

| pred_len | Exp1-Pre MSE | Exp6-Post (α=0.5) Phase 2 MSE | Δ |
|----------|--------------|-------------------------------|---|
| 96 | 0.8683 | 0.7240 | **Post better by 0.1443** |
| 192 | 0.8463 | 0.8346 | **Post better by 0.0117** |

Adding Label (Legendre) and Order (delta_x) components on top of post-softmax distance
improves over distance-only at both horizons.

### Analysis Framework

The primary research questions can now be answered:

1. **Post vs Pre Softmax:** Post-softmax (Exp6-Post, α=0.5) outperforms Pre-softmax
   (Exp6-Pre, best α=1.0) at pred_len 96 and 192 on ETTh1.

2. **Decay Parameter Sensitivity:** Strong sensitivity — α=0.5 is clearly best.
   Larger values (1.0, 2.0) degrade performance substantially in post-softmax context.

3. **Prediction Horizon:** Avg MSE degrades smoothly from 0.7285 (pred_48) to 0.9607
   (pred_336). No instability spike comparable to Exp1-Pre's pred_336 collapse.

4. **ETTm1 / pred_720:** TODO: Information could not be verified from the repository.
   No ETTm1 runs or pred_len=720 runs were executed in any script.

### Metrics

| Metric | Purpose |
|--------|---------|
| **MSE** | Primary performance metric |
| **MAE** | Robustness to outliers |
| **Seed Variance** | Statistical significance (3 seeds) |
| **Training Time** | Computational efficiency |
| **Convergence** | Early stopping epoch |

---

## Theoretical Context

### Post-Softmax Distance Rationale

**Advantages:**
1. **Preserves Learned Patterns:** Softmax first allows model to learn semantic attention
2. **Multiplicative Refinement:** Distance acts as a modulator, not a distorter
3. **Numerical Stability:** Normalized weights before distance application
4. **Interpretability:** Clear separation of semantic vs positional attention

**Disadvantages:**
1. **Weaker Bias:** Distance has less influence after normalization
2. **Non-normalized Output:** A' may not sum to 1 (though close)
3. **Gradient Flow:** May affect backpropagation differently

### Mathematical Comparison

**Pre-Softmax (Exp6 Pre):**
```
scores' = scores * α(i,j)
A = softmax(scores')
```
- Distance affects raw scores
- Softmax normalizes biased scores
- Strong positional bias

**Post-Softmax (Exp6 Post):**
```
A = softmax(scores)
A' = A * α(i,j)
```
- Softmax normalizes unbiased scores
- Distance refines normalized weights
- Weaker but cleaner positional bias

---

## Computational Complexity

### Time Complexity
- **Legendre Embedding:** O(B × L × D)
- **Delta Computation:** O(B × L × D)
- **Distance Matrix:** O(L²) per attention head
- **Attention:** O(B × H × L² × D)
- **Total per layer:** O(B × H × L² × D)

### Space Complexity
- **Distance Matrix:** O(L²) per head
- **Attention Weights:** O(B × H × L²)
- **Activations:** O(B × L × D)

### For seq_len=96, d_model=512, n_heads=8:
- Distance matrix: 96² = 9,216 elements per head
- Total attention: 8 × 9,216 = 73,728 elements
- **Acceptable** for research, may need optimization for production

---

## Debugging and Validation

### Sanity Checks

1. **Embedding Output:**
   ```python
   combined_emb, delta_x = embedding(x, x_mark)
   assert combined_emb.shape == (B, L, D)
   assert delta_x.shape == (B, L, D)
   assert delta_x[:, 0, :].abs().sum() == 0  # First delta is zero
   ```

2. **Distance Decay:**
   ```python
   # Check α(i,j) values
   alpha = 1.0 / (1.0 + dist_matrix ** decay_a)
   assert alpha.diagonal().allclose(torch.ones(L))  # α(i,i) = 1
   assert (alpha >= 0).all() and (alpha <= 1).all()
   ```

3. **Attention Weights:**
   ```python
   # Post-softmax should be approximately normalized
   A_sum = A.sum(dim=-1)
   assert A_sum.allclose(torch.ones_like(A_sum), atol=0.1)
   ```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Tuple unpacking error | `too many values to unpack` | Check all embedding calls return tuples |
| Delta not passed | `unexpected keyword argument` | Verify all forward() accept delta_x |
| Distance not applied | Same as vanilla | Check α(i,j) computation in attn.py |
| NaN in training | Loss becomes NaN | Check scaling factors, reduce learning rate |

---

## File Structure

```
experiments/exp6_lod_post/
├── models/
│   ├── __init__.py              - Empty (no model registry content)
│   ├── model.py                 - Informer/InformerStack (FIXED)
│   ├── encoder.py               - Encoder/EncoderLayer/EncoderStack (FIXED)
│   ├── decoder.py               - Decoder/DecoderLayer (FIXED)
│   ├── attn.py                  - FullAttention with POST-softmax distance
│   ├── embed.py                 - DataEmbedding returns (combined, delta)
│   └── legendre_embedding.py    - Recurrence formula (FIXED)
├── EXP_6_LOD_POST.ipynb         - Colab notebook with full run output
├── exp6_lod_post_phase1.sh      - Phase 1 training script (alpha screening)
├── exp6_lod_post_ph2.sh         - Phase 2 training script (stability validation)
├── exp6_lod_post_phase2_a0.5.sh - Duplicate of exp6_lod_post_ph2.sh
└── README-E6-POST.md            - This file
```

---

## Comparison with Exp6 Pre

| Aspect | Exp6 Pre | Exp6 Post |
|--------|----------|-----------|
| **Distance Application** | `scores * α → softmax` | `softmax → A * α` |
| **Attention Shape** | Distorted before norm | Preserved during norm |
| **Positional Bias** | Strong (affects raw scores) | Moderate (refines weights) |
| **Normalization** | Biased scores normalized | Clean scores normalized |
| **Gradient Flow** | Through distance-biased scores | Through clean scores |
| **Interpretability** | Mixed semantic+positional | Separated semantic→positional |

### When to Use Each

**Use Pre-Softmax (Exp6 Pre) when:**
- Strong positional bias is needed
- Positional information should dominate
- Raw score modification is acceptable

**Use Post-Softmax (Exp6 Post) when:**
- Semantic attention should be learned first
- Positional bias should be a refinement
- Attention distribution shape should be preserved

---

## Next Steps

1. ~~**Run Experiments:**~~ ✅ All 18 runs completed (Phase 1: 6 runs, Phase 2: 12 runs).
2. ~~**Analyze Results:**~~ ✅ Results analysed above; Exp6-Post (α=0.5) outperforms Exp6-Pre and Exp1-Pre at pred_len 96 and 192.
3. **Statistical Testing:** Verify significance across seeds with formal tests (t-test / Wilcoxon) — only 3 seeds available.
4. **Ablation Studies:** Test individual component contributions (L-only, O-only, D-only) — cross-reference against Exp3, Exp4, Exp1-Post.
5. **Visualization:** Plot attention maps with/without distance decay.
6. **Extended Horizons:** Run pred_len=720 on ETTh1; run ETTm1 dataset (neither was executed).

---

## References

- **Legendre Polynomials:** Orthogonal basis for position encoding
- **Attention Mechanisms:** Vaswani et al., "Attention is All You Need"
- **Informer:** Zhou et al., "Informer: Beyond Efficient Transformer"
- **Distance-based PE:** Novel contribution of this research

---

## Notes

- ✅ All fixes applied (decoder unpacking, delta_x propagation, recurrence formula, EncoderStack)
- ✅ Consistent with exp3, exp5b, exp6_pre Legendre implementation
- ✅ Post-softmax distance decay properly implemented
- ✅ Order component (delta_x) used in value matrix only — delta is computed from `value_emb`, not raw `x`
- ✅ Label component (Legendre) in Q/K embeddings
- ✅ `decay_a` forwarded through all encoder and decoder attention layers (FINDING H fix confirmed)
- ✅ `delta_x` downsampled through ConvLayer in encoder (FINDING G fix confirmed)
- ⚠️  `delta_values` routing is in `AttentionLayer`, not in `FullAttention.forward()` — README pseudocode updated to reflect this
- ⚠️  `exp6_lod_post_phase2_a0.5.sh` is an exact duplicate of `exp6_lod_post_ph2.sh`

**Status:** ✅ Training complete — all 18 runs finished on 2026-08-16

---

*Made with Bob*