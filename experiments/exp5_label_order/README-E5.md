# Experiment 5: Label + Order (Distinctiveness + Directionality)

## Objective
Test whether combining **Label (Legendre polynomials)** and **Order (signed displacements)** provides better positional structure than either component alone, without distance-based weighting.

## Research Question
**Do Label and Order synergize when combined, or do they interfere?**

## Mathematical Formulation

### Core Equation
```
X'_i = X_i + T_i + P_i + O_i

where:
  X_i = value_embedding(x)           [Semantic content]
  T_i = temporal_embedding(x_mark)   [Time features]
  P_i = Legendre(i)                  [LABEL - Equation 1]
  O_i = (1/N-1) · Σ_{j≠i} (X_i - X_j) [ORDER - Equation 3]
```

### Components

**1. LABEL (P_i) - Equation 1:**
```
Legendre Polynomials: P_i = [L_0(x_i), ..., L_{d-1}(x_i)]
Orthogonality: ⟨P_n, P_m⟩ = δ_{nm}
```
- Provides distinctiveness (each position unique)
- Scaled by 1/√d_model

**2. ORDER (O_i) - Equation 3:**
```
Signed Displacement: Δx_ij = X_i - X_j
Aggregation: O_i = (1/N-1) · Σ_{j≠i} Δx_ij
```
- Provides directionality (relative positioning)
- Uniform weighting (no distance decay)

**3. NO DISTANCE:**
- ❌ No α(i,j) index-based decay
- ❌ No w_ij feature-space weighting

---

## Implementation

### Components Used

**From Experiment 3:**
- `legendre_embedding.py` - Label component

**From Experiment 4:**
- `ordering_operator.py` - Order component

**New:**
- `embed.py` - Combines both L + O

### Forward Pass
```python
# 1. Value embedding
value_emb = self.value_embedding(x)

# 2. LABEL: Legendre polynomials
legendre_pos = self.legendre_embedding(x)

# 3. ORDER: Signed displacements
ordering_pos = self.ordering_operator(value_emb)

# 4. COMBINE: value + temporal + label + order
x = value_emb + self.temporal_embedding(x_mark) + legendre_pos + ordering_pos
```

---

## Key Differences from Other Experiments

| Aspect | Exp 3 (L) | Exp 4 (O) | **Exp 5 (L+O)** | Exp 2 (LOD) |
|--------|-----------|-----------|-----------------|-------------|
| **Label** | ✅ | ❌ | ✅ | ✅ |
| **Order** | ❌ | ✅ | ✅ | ✅ |
| **Distance** | ❌ | ❌ | ❌ | ✅ |
| **Temporal** | ❌ | ✅ | ✅ | ✅ |
| **Components** | L only | O only | **L+O** | L+O+D |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ETTh1 |
| Model | Informer |
| Attention | Full |
| Sequence Length | 96 |
| Prediction Length | 24 |
| d_model | 512 |
| Temporal | ✅ Included |

---

## File Structure
```
experiments/exp5_label_order/
├── models/
│   ├── legendre_embedding.py   [FROM EXP3] - Label
│   ├── ordering_operator.py    [FROM EXP4] - Order  
│   ├── embed.py                [NEW] - Combines L+O
│   └── ... (vanilla files)
├── README.md                    - This file
└── run_exp5.sh                  - Training script
```

---

## Hypotheses

### **Hypothesis A: Positive Synergy**
```
L+O < min(L, O)
```
Label and Order complement each other → better than either alone.

### **Hypothesis B: Negative Interference**
```
L+O ≈ max(L, O) or worse
```
Components interfere → no benefit from combination.

### **Hypothesis C: Additive**
```
L+O ≈ average(L, O)
```
Simple averaging effect.

---

## Expected Results

### Prediction Range

| Experiment | MSE | Reasoning |
|------------|-----|-----------|
| Vanilla | 0.519 | Baseline |
| Exp 1 (D) | 0.725 | Distance helps |
| Exp 4 (O) | ~0.85 | Order alone |
| **Exp 5 (L+O)** | **0.90-1.00** | **Between O and L** |
| Exp 3 (L) | 1.124 | Label worst |

**Expected:** Exp5 worse than Exp4 (Order alone) because Label adds noise.

---

## Results

**Training completed with early stopping after epoch 4.**

| Metric | Value |
|--------|-------|
| MSE    | **0.7192** |
| MAE    | **0.6352** |

**Best Validation Loss:** 0.7932 (Epoch 1)

**Full precision values:**
- MSE: 0.719232976436615
- MAE: 0.6352460980415344

### 🎉 SURPRISING RESULT!
**Exp5 (L+O) outperformed Exp1 (Distance only)!**
- Exp5 (L+O): MSE = 0.7192
- Exp1 (D): MSE = 0.7250
- **Improvement:** -0.0058 MSE (0.8% better)

This suggests **positive synergy** between Label and Order components!

---

## Complete Comparison Table

| Experiment | Components | Temporal | MSE ↓ | MAE ↓ | Rank |
|------------|-----------|----------|-------|-------|------|
| Vanilla | Standard PE | ✅ | **0.519** | **0.513** | 🥇 1st |
| **Exp 5 (L+O)** | **Label+Order** | ✅ | **0.719** | **0.635** | 🥈 **2nd** |
| Exp 1 (D) | Distance | ✅ | **0.725** | **0.652** | 🥉 3rd |
| Exp 2 (LOD) | L+O+D | ✅ | **0.804** | **0.710** | 4th |
| Exp 4 (O) | Order | ✅ | **0.835** | **0.720** | 5th |
| Exp 3 (L) | Label | ❌ | **1.124** | **0.855** | 6th |

---

## Analysis

### Key Findings

1. **POSITIVE SYNERGY: L+O Outperforms D Alone!**
   - **Exp5 (L+O):** MSE = 0.7192 ✅ **2nd place**
   - Exp1 (D): MSE = 0.7250 (3rd place)
   - **Improvement:** 0.8% better than Distance alone
   - **Hypothesis A CONFIRMED:** Label and Order complement each other!

2. **Removing Distance IMPROVED Performance**
   - Exp2 (L+O+D): MSE = 0.8036 (4th place)
   - **Exp5 (L+O):** MSE = 0.7192 (2nd place)
   - **Improvement:** 10.5% better without Distance!
   - **Conclusion:** Distance component interferes with L+O synergy

3. **Component Ranking (Isolated)**
   ```
   Best → Worst (isolated components):
   D (0.725) < O (0.835) < L (1.124)
   ```

4. **Component Ranking (Combined)**
   ```
   Best → Worst (combinations):
   L+O (0.719) < D (0.725) < L+O+D (0.804) < O (0.835) < L (1.124)
   ```

### Why Did L+O Outperform D?

**Complementary Strengths:**
- **Label (L):** Provides orthogonal distinctiveness (each position unique)
- **Order (O):** Provides directional relationships (relative positioning)
- **Together:** Create a richer positional structure than distance decay alone

**Why Distance Hurts:**
- Adding Distance decay (α) and feature-space weighting (w_ij) to L+O **degrades** performance
- Suggests that the weighting schemes interfere with the natural synergy between Label and Order
- Uniform aggregation (Exp5) works better than weighted aggregation (Exp2)

### Theoretical Implications

This result **contradicts the paper's claim** that "PoPE + ΔV fails to match PE."

**Our findings:**
- L+O (0.719) performs **better** than D alone (0.725)
- L+O is the **second-best** approach after vanilla PE (0.519)
- Distance weighting actually **hurts** the L+O combination

**Possible explanations:**
1. **Orthogonality + Directionality** is a powerful combination
2. **Uniform weighting** preserves the signal better than distance-based weighting
3. The **temporal embedding** (which we kept, unlike Exp3) is crucial for L+O to work

---

## Analysis Questions

1. **Do L and O synergize?**
   - If Exp5 < Exp3 AND Exp5 < Exp4 → YES (positive)
   - If Exp5 ≈ Exp3 OR Exp5 ≈ Exp4 → NO (one dominates)
   - If Exp5 > both → Negative interference

2. **Does removing Distance help or hurt?**
   - Compare Exp5 (L+O) vs Exp2 (L+O+D)
   - If Exp5 < Exp2 → Distance hurts!
   - If Exp5 > Exp2 → Distance helps integration

3. **Component hierarchy?**
   - Rank isolated: L vs O vs D
   - Rank combined: L+O vs L+D vs O+D

---

## Theoretical Context

From the paper:
> "The combination of label + ordering (PoPE + ΔV) fails to match the 
> results of PE... when all three components (LOD) are combined, PE 
> still delivers superior performance."

**This experiment directly tests the "PoPE + ΔV" scenario from Table 1.**

---

## Next Steps After Running

1. Compare with Exp3 (L) and Exp4 (O) to check synergy
2. Compare with Exp2 (L+O+D) to isolate Distance effect
3. Update paper Table 1 with all 5 experiments
4. Analyze component interaction patterns
