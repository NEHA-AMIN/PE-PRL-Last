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
```python
class FullAttention:
    def forward(self, queries, keys, values, attn_mask, delta_values=None):
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
        
        # Step 5: Use delta_values if provided (Order component)
        V = values if delta_values is None else delta_values
        output = torch.einsum("bhls,bshd->blhd", A, V)
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
```python
class DataEmbedding(nn.Module):
    def forward(self, x, x_mark):
        # Value embedding
        value_emb = self.value_embedding(x)
        
        # Temporal embedding
        temporal_emb = self.temporal_embedding(x_mark)
        
        # Legendre position embedding (Label)
        legendre_pos = self.legendre_embedding(x)
        
        # Combined embedding for Q/K
        combined_emb = value_emb + temporal_emb + legendre_pos
        combined_emb = self.dropout(combined_emb)
        
        # Clean delta for V (Order)
        delta_x = torch.zeros_like(x)
        delta_x[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        delta_x = self.value_embedding(delta_x)
        
        return combined_emb, delta_x
```

---

## Configuration

### Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Datasets** | ETTh1, ETTm1 |
| **Model** | Informer |
| **Attention** | Full (not ProbSparse) |
| **Sequence Length** | 96 |
| **Label Length** | 48 |
| **Prediction Lengths** | 48, 96, 192, 336, 720 |
| **Encoder Layers** | 2 |
| **Decoder Layers** | 1 |
| **d_model** | 512 |
| **n_heads** | 8 |
| **d_ff** | 2048 |
| **Dropout** | 0.05 |
| **Activation** | GELU |

### Ablation Parameters

| Parameter | Values Tested |
|-----------|---------------|
| **Decay Parameter (a)** | 0.5, 1.0, 2.0 |
| **Random Seeds** | 2021, 2022, 2023 |
| **Total Runs** | 90 (2 datasets × 3 decay_a × 3 seeds × 5 pred_len) |

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

```bash
cd /Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1
bash experiments/exp6_lod_post/run_exp6_post.sh
```

### What the Script Does

1. Copies modified model files to `Informer2020-main/models/`
2. Runs 90 training configurations:
   - 2 datasets (ETTh1, ETTm1)
   - 3 decay parameters (0.5, 1.0, 2.0)
   - 3 random seeds (2021, 2022, 2023)
   - 5 prediction lengths (48, 96, 192, 336, 720)
3. Saves results to `results/lod_post_a{decay_a}_{dataset}_{pred_len}_seed{seed}/`

### Expected Runtime

- **Per run:** ~10-15 minutes (depends on early stopping)
- **Total:** ~15-22 hours for all 90 runs
- **Recommendation:** Run overnight or use parallel execution

---

## Expected Output

Each run produces:
- `training_log.txt` - Full training output
- `checkpoint.pth` - Best model weights
- Test metrics: MSE, MAE, RMSE, MAPE, MSPE

### Directory Structure
```
results/
├── lod_post_a0.5_ETTh1_48_seed2021/
│   ├── training_log.txt
│   └── checkpoint.pth
├── lod_post_a0.5_ETTh1_48_seed2022/
├── ...
└── lod_post_a2.0_ETTm1_720_seed2023/
```

---

## Analysis Framework

### Primary Questions

1. **Post vs Pre Softmax:**
   - Does post-softmax distance decay outperform pre-softmax?
   - Which timing preserves attention patterns better?

2. **Decay Parameter Sensitivity:**
   - How does performance vary with a ∈ {0.5, 1.0, 2.0}?
   - Is there an optimal decay rate?

3. **Prediction Horizon:**
   - Does LOD help more for longer predictions (720) vs shorter (48)?
   - Where is the performance crossover point?

4. **Dataset Dependency:**
   - Does ETTh1 (hourly) vs ETTm1 (15-min) affect LOD effectiveness?
   - Are results consistent across temporal granularities?

### Metrics to Track

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

## Expected Results Format

### Per-Run Metrics
```
Dataset: ETTh1
Decay_a: 1.0
Seed: 2021
Pred_len: 96

Test Results:
  MSE:  0.XXXX
  MAE:  0.XXXX
  RMSE: 0.XXXX
  MAPE: 0.XXXX
  MSPE: 0.XXXX
```

### Aggregated Analysis
```
Average across 3 seeds:
  MSE:  0.XXXX ± 0.XXXX
  MAE:  0.XXXX ± 0.XXXX

Best decay_a: X.X
Best pred_len: XXX
```

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
│   ├── __init__.py              - Model registry
│   ├── model.py                 - Informer/InformerStack (FIXED)
│   ├── encoder.py               - Encoder/EncoderLayer/EncoderStack (FIXED)
│   ├── decoder.py               - Decoder/DecoderLayer (FIXED)
│   ├── attn.py                  - FullAttention with POST-softmax distance
│   ├── embed.py                 - DataEmbedding returns (combined, delta)
│   └── legendre_embedding.py    - Recurrence formula (FIXED)
├── run_exp6_post.sh             - Training script
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

1. **Run Experiments:** Execute all 90 configurations
2. **Analyze Results:** Compare with Exp6 Pre and other baselines
3. **Statistical Testing:** Verify significance across seeds
4. **Ablation Studies:** Test individual component contributions
5. **Visualization:** Plot attention maps with/without distance decay
6. **Optimization:** Tune decay_a based on results

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
- ✅ Order component (delta_x) used in value matrix only
- ✅ Label component (Legendre) in Q/K embeddings

**Status:** Ready for training

---

*Made with Bob*