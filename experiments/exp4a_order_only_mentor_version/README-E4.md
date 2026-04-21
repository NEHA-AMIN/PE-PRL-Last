# Experiment 4a: Order Only - Mentor's Correct Version (Consecutive Delta in Attention Values)

## Objective
Test whether **consecutive differences in the value matrix** provide meaningful positional information through the attention mechanism.

## Research Question
**Can consecutive deltas (Δx = x_i - x_{i-1}) in attention values encode sufficient positional structure?**

## Mathematical Formulation

### Core Equation
```
Embedding: X'_i = X_i + T_i

where:
  X_i = value_embedding(x)           [Semantic content]
  T_i = temporal_embedding(x_mark)   [Time features]
```

### Ordering in Attention (Inside attn.py)
```
Standard Attention:
  scores = softmax(Q·K^T / √d)
  output = scores · V

Modified Attention with Ordering:
  scores = softmax(Q·K^T / √d)  [computed over original x]
  Δx_i = x_i - x_{i-1}          [consecutive differences]
  Δx_0 = 0                      [first position has no predecessor]
  output = scores · ΔV          [where ΔV uses delta values]
```

**Key Properties:**
- ✅ Ordering lives in the **value side of attention**
- ✅ Consecutive differences only: Δx_i = x_i - x_{i-1}
- ✅ Attention scores computed over original embeddings
- ✅ First position delta is zero (no predecessor)
- ❌ NO global mean-field repulsion
- ❌ NO distance decay (no α(i,j))
- ❌ NO labels (no Legendre polynomials)

**Interpretation:** The attention mechanism aggregates consecutive directional changes rather than absolute positions.

---

## Implementation

### Modified: `attn.py` (FullAttention class)
```python
def forward(self, queries, keys, values, attn_mask):
    # Attention scores computed normally over x
    scores = torch.einsum("blhe,bshe->bhls", queries, keys)
    # ... masking and softmax ...
    A = self.dropout(torch.softmax(scale * scores, dim=-1))
    
    # === ORDERING: replace values with delta_x = x_i - x_{i-1} ===
    delta_values = torch.zeros_like(values)
    delta_values[:, 1:, :, :] = values[:, 1:, :, :] - values[:, :-1, :, :]
    # delta_values[:, 0, :, :] remains zero (no previous token)
    
    V = torch.einsum("bhls,bshd->blhd", A, delta_values)
    # === END ORDERING ===
    return (V.contiguous(), None)
```

### Modified: `embed.py`
```python
def forward(self, x, x_mark):
    # Vanilla embedding - ordering happens in attention
    x = self.value_embedding(x) + self.temporal_embedding(x_mark)
    return self.dropout(x)
```

---

## Key Differences from Other Experiments

| Aspect | Exp 1 (D) | Exp 2 (LOD) | Exp 3 (L) | **Exp 4a (O)** |
|--------|-----------|-------------|-----------|----------------|
| **Label (L)** | ❌ | ✅ | ✅ | ❌ |
| **Order (O)** | ❌ | ✅ | ❌ | ✅ **ONLY** |
| **Distance (D)** | ✅ | ✅ | ❌ | ❌ |
| **Temporal** | ✅ | ✅ | ❌ | ✅ |
| **Location** | Attention bias | Embedding | Embedding | **Attention Values** |
| **Signal** | Distance decay | Full aggregation | Labels | **Consecutive Δx** |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ETTh1 |
| Model | Informer |
| Attention | Full |
| Sequence Length | 96 |
| Prediction Length | 24 |
| Encoder Layers | 2 |
| d_model | 512 |
| Temporal Embedding | ✅ Included |

---

## File Structure
```
experiments/exp4a_order_only_mentor_version/
├── models/
│   ├── attn.py                [MODIFIED] - Consecutive delta in values
│   ├── embed.py               [MODIFIED] - Vanilla (no ordering operator)
│   └── ... (other vanilla files)
├── README-E4.md               - This file
└── run_exp4.sh                - Training script
```

**Note:** `ordering_operator.py` has been removed - ordering now happens inside attention mechanism.

---

## How to Run
```bash
bash experiments/exp4_order_only/run_exp4.sh
```

---

## Expected Behavior

### Hypothesis
Ordering provides more structure than Label (Exp3) but less than Distance (Exp1).

**Predicted ranking:**
```
Vanilla (0.519) < Exp1-Distance (0.725) < Exp4-Order (?) < Exp2-LOD (0.753) < Exp3-Label (1.124)
```

**Reasoning:**
- Directional signal > Static labels
- But uniform weighting < Distance-based decay

---

## Results

**Training completed with early stopping after epoch 5.**

| Metric | Value |
|--------|-------|
| MSE    | **0.8348** |
| MAE    | **0.7200** |

**Best Validation Loss:** 0.8350 (Epoch 2)

**Full precision values:**
- MSE: 0.8347994089126587
- MAE: 0.7200072407722473

---

## Complete Comparison

| Experiment | Components | Temporal | MSE ↓ | MAE ↓ | Rank |
|------------|-----------|----------|-------|-------|------|
| Vanilla | Standard PE | ✅ | **0.519** | **0.513** | 🥇 1st |
| Exp 1 (D) | Distance | ✅ | **0.725** | **0.652** | 🥈 2nd |
| Exp 2 (LOD) | L+O+D | ✅ | **0.804** | **0.710** | 🥉 3rd |
| **Exp 4 (O)** | **Order** | ✅ | **0.835** | **0.720** | **4th** |
| Exp 3 (L) | Label | ❌ | **1.124** | **0.855** | 5th |

---

## Analysis Questions

1. **Does consecutive delta in attention values help?**
   - Compare Exp4a (consecutive Δx in values) vs Exp4 (global mean-field)
   
2. **Is the value-side implementation correct?**
   - Verify ordering lives in attention mechanism, not embedding
   
3. **What about the first position?**
   - First token has delta=0 (no predecessor) - is this the right choice?

---

## Analysis

### Key Findings

1. **Order Performs Between LOD and Label**
   - Exp2 (LOD): MSE = 0.804
   - **Exp4 (Order):** MSE = 0.835
   - Exp3 (Label): MSE = 1.124
   - **Difference from LOD:** +0.031 MSE (4% worse)

2. **Directional Signals Provide Moderate Benefit**
   - Order (signed displacements) performs better than Label alone
   - But worse than combining all components (LOD)
   - Suggests directional information is useful but not sufficient alone

3. **Ranking Confirmation**
   ```
   Vanilla (0.519) < Exp1-D (0.725) < Exp2-LOD (0.804) < Exp4-O (0.835) < Exp3-L (1.124)
   ```

4. **Component Effectiveness Summary**
   - **Distance (D):** Most effective (MSE = 0.725)
   - **Order (O):** Moderate effectiveness (MSE = 0.835)
   - **Label (L):** Least effective (MSE = 1.124)
   - **L+O+D:** Combined worse than D alone (MSE = 0.804)

### Interpretation

The ordering operator captures **directional relationships** between positions through signed feature-space displacements. While this provides some positional structure, it's:
- **Less effective** than distance-based decay (Exp1)
- **More effective** than static orthogonal labels (Exp3)
- **Comparable** to the full LOD formulation (Exp2)

This suggests that **uniform weighting** (no distance bias) limits the effectiveness of directional signals. The distance decay in Exp1 likely provides better positional bias than uniform aggregation of signed displacements.

---

## Theoretical Context

**Mentor's Specification:**
> "Attention is computed over x, and then the value matrix is replaced with Δx = x_i - x_{i-1} — consecutive differences only, fed into the value projection. The ordering lives in the value side of attention, not in the input embedding."

**This experiment implements:**
- ✅ Consecutive differences: Δx_i = x_i - x_{i-1}
- ✅ Applied to value matrix inside attention mechanism
- ✅ Attention scores computed over original embeddings
- ✅ First position delta = 0 (no predecessor)

**Key Distinction from Exp4 (original):**
- **Exp4 (wrong):** Global mean-field O_i = (1/N-1)·Σ(X_i - X_j) added to embedding
- **Exp4a (correct):** Consecutive delta Δx_i = x_i - x_{i-1} in attention values

**This is the mathematically correct implementation per mentor's design.**
