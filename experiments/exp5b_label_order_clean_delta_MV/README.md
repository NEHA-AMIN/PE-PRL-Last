# Experiment 5b: Label + Order with Clean Delta (L+O Clean)

## Objective
Test the **Label + Order** combination with a **clean delta signal** that contains ONLY value embedding differences, without any positional or temporal components.

## Research Question
**Does a pure temporal ordering signal (x_i - x_{i-1}) combined with Legendre labels provide better performance than the original Exp 5 implementation?**

---

## Mathematical Formulation

### Core Architecture

```
Input to Attention:
  Q, K ← project(x_i + T_i + p_i)    [Combined embedding]
  V ← project(Δx)                     [Clean delta only]

where:
  x_i = TokenEmbedding(input)         [Value embedding]
  T_i = TemporalEmbedding(time_mark)  [Temporal features]
  p_i = Legendre(i)                   [Label - positional distinctiveness]
  Δx[i] = x_i - x_{i-1}              [Order - CLEAN temporal delta]
```

### Key Components

**1. LABEL (p_i) - Positional Distinctiveness:**
```
Legendre Polynomials: p_i = [L_0(x_i), ..., L_{d-1}(x_i)]
Domain: x_i ∈ [-1, 1]
Orthogonality: ⟨p_n, p_m⟩ = δ_{nm}
Scaling: 1/√d_model
```
- Provides unique positional identity
- Added to input before Q/K projections

**2. ORDER (Δx) - Clean Temporal Signal:**
```
Delta Computation: Δx[i] = x_i - x_{i-1}
First Position: Δx[0] = 0
```
- **CRITICAL:** Computed from value_emb ONLY
- **NO p_i terms:** Avoids p_i - p_{i-1}
- **NO T_i terms:** Avoids T_i - T_{i-1}
- Pure temporal ordering in semantic space

**3. Attention Mechanism:**
```
Similarity: scores = (Q · K^T) / √d_k
  where Q, K from (x_i + T_i + p_i)

Aggregation: output = softmax(scores) · V
  where V from Δx (clean delta)
```

---

## Critical Differences from Experiment 5

| Aspect | **Exp 5** | **Exp 5b (This)** |
|--------|-----------|-------------------|
| **Delta Input** | `legendre_pos` (p_i) | `value_emb` (x_i) |
| **Delta Formula** | O_i = (1/(L-1))·Σ(p_i - p_j) | Δx[i] = x_i - x_{i-1} |
| **Delta Type** | Pairwise mean (all positions) | Sequential shift (temporal) |
| **Delta Contains** | Positional differences | Semantic differences |
| **Q/K Source** | x_i + T_i + p_i + O_i | x_i + T_i + p_i |
| **V Source** | x_i + T_i + p_i + O_i | Δx (clean) |
| **Architecture** | Additive (all in embedding) | Separated (delta in V only) |

### Why This Matters

**Exp 5 Approach:**
- Delta computed from Legendre: `O_i = (1/(L-1)) · Σ_{j≠i}(p_i - p_j)`
- Pairwise mean across all positions (not sequential)
- Creates **positional** ordering signal
- Both Label and Order work in same space → potential redundancy

**Exp 5b Solution:**
- Delta computed from values: `Δx[i] = x_i - x_{i-1}`
- This creates **semantic** ordering signal
- Label (positional) and Order (semantic) are orthogonal → complementary

---

## Implementation Details

### File Structure
```
experiments/exp5b_label_order_clean_delta/models/
├── legendre_embedding.py   - Legendre polynomial embeddings (p_i)
├── embed.py                - Returns (combined_emb, delta_x)
├── attn.py                 - Modified to use delta_x for V projection
├── encoder.py              - Passes delta_x through layers
├── model.py                - Handles dual return from embedding
├── decoder.py              - Standard (no delta)
└── __init__.py             - Module initialization
```

### Key Code Changes

**1. embed.py - DataEmbedding:**
```python
def forward(self, x, x_mark):
    # 1. Value embedding
    value_emb = self.value_embedding(x)  # x_i
    
    # 2. CLEAN DELTA: x_i - x_{i-1}
    delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)
    delta_x[:, 0, :] = 0.0  # Zero first position
    
    # 3. Temporal and Legendre
    temporal_emb = self.temporal_embedding(x_mark)  # T_i
    legendre_pos = self.legendre_embedding(x)       # p_i
    
    # 4. Combined for Q/K
    combined_emb = value_emb + temporal_emb + legendre_pos
    
    return combined_emb, delta_x  # Return both!
```

**2. attn.py - AttentionLayer:**
```python
def forward(self, queries, keys, values, attn_mask, 
            delta_queries=None, delta_keys=None, delta_values=None):
    # Q, K from combined embedding
    queries = self.query_projection(queries)
    keys = self.key_projection(keys)
    
    # V from delta_x (if provided)
    if delta_values is not None:
        values = self.value_projection(delta_values)  # Use clean delta!
    else:
        values = self.value_projection(values)
    
    # Standard attention computation
    out, attn = self.inner_attention(queries, keys, values, attn_mask)
    return self.out_projection(out), attn
```

**3. encoder.py - EncoderLayer:**
```python
def forward(self, x, attn_mask=None, delta_x=None):
    # Pass delta_x to attention for V projection
    new_x, attn = self.attention(
        x, x, x,
        attn_mask=attn_mask,
        delta_queries=delta_x,
        delta_keys=delta_x,
        delta_values=delta_x  # This is what matters!
    )
    # Rest is standard
    x = x + self.dropout(new_x)
    # ... feed-forward ...
    return x, attn
```

**4. model.py - Informer:**
```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, ...):
    # Encoder: get both combined and delta
    enc_out, delta_enc = self.enc_embedding(x_enc, x_mark_enc)
    enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, 
                                   delta_x=delta_enc)
    
    # Decoder: standard (no delta)
    dec_out = self.dec_embedding(x_dec, x_mark_dec)
    dec_out = self.decoder(dec_out, enc_out, ...)
    return self.projection(dec_out)
```

---

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset | ETTh1, ETTm1 | Time series forecasting |
| Model | Informer | Transformer-based |
| Attention | Full | --attn full |
| Sequence Length | 96 | Input window |
| Label Length | 48 | Decoder start |
| Prediction Lengths | 48, 96, 192, 336, 720 | Multi-horizon |
| d_model | 512 | Embedding dimension |
| n_heads | 8 | Attention heads |
| e_layers | 2 | Encoder layers |
| d_layers | 1 | Decoder layers |
| d_ff | 2048 | Feed-forward dimension |
| Batch Size | 32 | Training batch |
| Learning Rate | 1e-4 | Adam optimizer |
| Early Stopping | 3 epochs patience | Validation-based |
| Runs per Config | 3 | For statistical significance |

---

## Hypotheses

### Hypothesis A: Clean Delta Improves Performance
```
Exp5b < Exp5
```
**Reasoning:** Separating positional (Label) and semantic (Order) signals should reduce redundancy and improve learning.

### Hypothesis B: Matches or Exceeds Distance-Only
```
Exp5b ≤ Exp1 (Distance)
```
**Reasoning:** Clean delta provides temporal ordering without distance decay, potentially matching distance-based approaches.

### Hypothesis C: Best L+O Combination
```
Exp5b < min(Exp5, Exp4, Exp3)
```
**Reasoning:** Proper separation of Label and Order should outperform either component alone or their naive combination.

---

## Expected Results

### Performance Prediction

| Experiment | Components | Expected MSE | Reasoning |
|------------|-----------|--------------|-----------|
| Vanilla | Standard PE | 0.519 | Baseline |
| **Exp 5b** | **L+O Clean** | **0.65-0.75** | **Clean separation** |
| Exp 5 | L+O (p_i delta) | 0.719* | Positional redundancy |
| Exp 1 | Distance | 0.725 | Distance decay |
| Exp 2 | L+O+D | 0.804 | Over-parameterized |
| Exp 4 | Order only | 0.835 | Missing distinctiveness |
| Exp 3 | Label only | 1.124 | Missing directionality |

*Note: Exp 5 result (0.719) is from original implementation with potential architectural issues. Subject to re-evaluation.

**Key Prediction:** Exp 5b should outperform Exp 5 by 5-10% due to cleaner signal separation.

---

## Analysis Questions

1. **Does clean delta improve over Exp 5?**
   - Compare MSE: Exp5b vs Exp5
   - Expected: Exp5b < Exp5 (better)

2. **How does it compare to Distance-only?**
   - Compare MSE: Exp5b vs Exp1
   - Test if L+O can match/exceed D alone

3. **Is this the best L+O combination?**
   - Compare: Exp5b vs Exp5 vs (Exp3 + Exp4)
   - Validate architectural choice

4. **What's the contribution of each component?**
   - Ablation: Remove Label → Exp4 performance
   - Ablation: Remove Order → Exp3 performance

---

## Theoretical Context

### From the Paper
> "The combination of label + ordering (PoPE + ΔV) fails to match the results of PE"

**Our Investigation:**
- **Exp 5:** Tested PoPE + ΔV with delta from positional space
- **Exp 5b:** Tests PoPE + ΔV with delta from semantic space
- **Goal:** Determine if signal separation improves the L+O combination

### Key Insight
The paper's claim may be due to **implementation details** rather than fundamental limitations:
- If delta contains positional terms (p_i - p_{i-1}), it overlaps with Label
- If delta contains semantic terms (x_i - x_{i-1}), it complements Label
- **Exp 5b tests the second approach**

---

## Running the Experiment

### Quick Start
```bash
cd /Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1
bash experiments/exp5b_label_order_clean_delta/run_exp5b.sh
```

### What It Does
1. Copies model files to `Informer2020-main/models/`
2. Runs training for prediction lengths: 48, 96, 192, 336, 720
3. Saves results to `results/exp5b_label_order_{pred_len}/`
4. Logs training progress to `training_log.txt`

### Monitoring
```bash
# Watch training progress
tail -f results/exp5b_label_order_96/training_log.txt

# Check for completion
grep "Test" results/exp5b_label_order_96/training_log.txt
```

---

## Post-Experiment Analysis

### Metrics to Extract
```bash
# Extract MSE and MAE
grep "mse:" results/exp5b_label_order_96/training_log.txt | tail -1
grep "mae:" results/exp5b_label_order_96/training_log.txt | tail -1
```

### Comparison Table
After running, create comparison:

| Experiment | MSE ↓ | MAE ↓ | Δ from Exp5 | Δ from Exp1 |
|------------|-------|-------|-------------|-------------|
| Exp 5b | ? | ? | ? | ? |
| Exp 5 | 0.719 | 0.635 | - | - |
| Exp 1 | 0.725 | 0.652 | - | - |

### Success Criteria
- ✅ **Success:** Exp5b MSE < 0.719 (better than Exp5)
- ✅ **Strong Success:** Exp5b MSE < 0.725 (better than Distance)
- ⚠️ **Partial:** 0.719 < Exp5b MSE < 0.804 (between Exp5 and Exp2)
- ❌ **Failure:** Exp5b MSE > 0.804 (worse than all L+O variants)

---

## Next Steps

1. **Run Experiment:** Execute `run_exp5b.sh`
2. **Extract Results:** Parse training logs for MSE/MAE
3. **Compare:** Create comparison table with Exp 1-5
4. **Analyze:** Determine if clean delta improves performance
5. **Document:** Update this README with actual results
6. **Publish:** Add findings to paper/report

---

## Technical Notes

### Delta Computation Details
```python
# Using torch.roll for efficient shifting
delta_x = value_emb - torch.roll(value_emb, shifts=1, dims=1)
# Equivalent to:
# delta_x[i] = value_emb[i] - value_emb[i-1] for i > 0
# delta_x[0] = 0

# Why zero first position?
# - No previous value to subtract
# - Maintains sequence length
# - Prevents boundary artifacts
```

### Memory Considerations
- Stores both `combined_emb` and `delta_x` in memory
- Increases memory usage by ~2x for embeddings
- Still manageable for typical batch sizes (32)

### Gradient Flow
- Gradients flow through both paths:
  - Q/K path: combined_emb → value_emb, temporal_emb, legendre_pos
  - V path: delta_x → value_emb
- Value embedding receives gradients from both Q/K and V
- Legendre and temporal only from Q/K (as intended)

---

## References

- **Exp 1:** Distance-only (baseline for positional encoding)
- **Exp 2:** Full L+O+D (paper's proposed method)
- **Exp 3:** Label-only (Legendre polynomials)
- **Exp 4:** Order-only (signed displacements)
- **Exp 5:** Label + Order (positional delta)
- **Exp 5b:** Label + Order (semantic delta) ← **This experiment**

---

## Contact

For questions or issues with this experiment, refer to the main project documentation or check the training logs for error messages.

---

**Status:** ✅ Implementation Complete - Ready to Run
**Last Updated:** 2026-03-24