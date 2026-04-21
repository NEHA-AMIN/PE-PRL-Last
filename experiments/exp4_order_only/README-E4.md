# Experiment 4: Order Only (Pure Directional Signal)

## Objective
Test whether **signed feature-space displacements** (ordering/directionality) alone provide meaningful positional information, without distance-based weighting or label-based distinctiveness.

## Research Question
**Can directional relationships (X_i relative to all X_j) encode sufficient positional structure?**

## Mathematical Formulation

### Core Equation
```
X'_i = X_i + T_i + O_i

where:
  X_i = value_embedding(x)           [Semantic content]
  T_i = temporal_embedding(x_mark)   [Time features]
  O_i = (1/N-1) · Σ_{j≠i} (X_i - X_j) [ORDER - pure directional signal]
```

### Order Component (Based on Equation 3)
```
Signed Displacement: Δx_ij = X_i - X_j ∈ ℝ^d

Ordering Operator: O_i = (1/N-1) · Σ_{j≠i} Δx_ij
```

**Key Properties:**
- ✅ Preserves directionality (signed differences)
- ✅ Uniform weighting (all positions equal importance)
- ❌ NO distance decay (no α(i,j))
- ❌ NO feature-space weighting (no w_ij)
- ❌ NO labels (no Legendre polynomials)

**Interpretation:** Each position encodes how it differs from the average of all other positions.

---

## Implementation

### New Module: `ordering_operator.py`
```python
class OrderingOperator(nn.Module):
    """Pure ordering via signed displacements."""
    
    def forward(self, X):
        # 1. Compute Δx_ij = X_i - X_j (signed)
        # 2. Mask diagonal (j ≠ i)
        # 3. Sum and normalize: O_i = (1/N-1) · Σ_j Δx_ij
        return O
```

### Modified: `embed.py`
```python
def forward(self, x, x_mark):
    value_emb = self.value_embedding(x)
    ordering_pos = self.ordering_operator(value_emb)
    
    # Value + Temporal + Ordering (no distance, no label)
    x = value_emb + self.temporal_embedding(x_mark) + ordering_pos
    return self.dropout(x)
```

---

## Key Differences from Other Experiments

| Aspect | Exp 1 (D) | Exp 2 (LOD) | Exp 3 (L) | **Exp 4 (O)** |
|--------|-----------|-------------|-----------|---------------|
| **Label (L)** | ❌ | ✅ | ✅ | ❌ |
| **Order (O)** | ❌ | ✅ | ❌ | ✅ **ONLY** |
| **Distance (D)** | ✅ | ✅ | ❌ | ❌ |
| **Temporal** | ✅ | ✅ | ❌ | ✅ |
| **Weighting** | α(i,j) | α·w | None | **Uniform** |
| **Signal** | Attention bias | Full aggregation | Labels | **Signed Δx** |

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
experiments/exp4_order_only/
├── models/
│   ├── ordering_operator.py   [NEW] - O_i computation
│   ├── embed.py               [MODIFIED] - Ordering only
│   └── ... (vanilla files)
├── README.md                   - This file
└── run_exp4.sh                 - Training script
```

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

1. **Does signed displacement help?**
   - Compare Exp4 (O) vs Exp3 (L)
   
2. **Is uniform weighting enough?**
   - Compare Exp4 (O) vs Exp1 (D with decay)
   
3. **Do Order and Distance synergize?**
   - Compare Exp4 (O) vs Exp2 (L+O+D)

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

From paper:
> "Ordering should keep direction, so we aggregate signed differences Δxij"

**This experiment isolates:** Does directionality alone (without distance weighting) provide useful positional structure?
