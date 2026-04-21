# Experiment 2: Full LOD Formulation (Label + Order + Distance)

## Objective

Test the **complete distance-based positional encoding** approach by combining all three components:
- **Label (L):** Orthogonal distinctiveness via Legendre polynomials
- **Order (O):** Directional information via signed feature-space displacements
- **Distance (D):** Proximity bias via index-based decay and feature-space weighting

## Hypothesis

The full LOD formulation synergistically combines all three components to provide effective positional information for time-series forecasting, potentially approaching or matching the performance of standard sinusoidal positional encoding.

---

## Mathematical Formulation

### Complete Equation
```
X'_i = X_i + T_i + P_i + O_i

where:
  X_i = value_embedding(x)           [Semantic content]
  T_i = temporal_embedding(x_mark)   [Time features: hour, day, month]
  P_i = Legendre(i)                  [LABEL - Equation 1]
  O_i = Σ_{j≠i} α(i,j) · (w_ij ⊙ Δx_ij)  [ORDER + DISTANCE - Equation 3]
```

### Component Breakdown

#### 1. Label Component (Equation 1)
```
P_i = [L_0(x_i), L_1(x_i), ..., L_{d-1}(x_i)]

where:
  L_k = k-th Legendre polynomial
  x_i ∈ [-1, 1] (normalized position)
  Scaled by 1/√d_model
  
Properties:
  - Orthogonality: ⟨P_n, P_m⟩ = δ_{nm}
  - Distinctiveness: Each position has unique embedding
```

#### 2. Order + Distance Component (Equation 3)
```
O_i = (1/√d_model) · Σ_{j≠i} α(i,j) · (w_ij ⊙ Δx_ij)

where:
  Δx_ij = X_i - X_j                 [Signed displacement - preserves direction]
  α(i,j) = 1 / (1 + |i-j|^a)        [Index-based decay, a=1.0]
  w_ij = 1 / (1 + d_ij)             [Feature-space weighting]
  d_ij = ||X_i - X_j||_1            [L1 distance]
  ⊙ = element-wise multiplication
  
Scaling: 1/√d_model applied for numerical stability
```

---

## Implementation Details

### Modified Files

#### `models/embed.py`
**Changes:**
- ✅ Removed standard positional embedding
- ✅ Kept temporal embedding (unlike Exp3)
- ✅ Added Legendre embedding initialization
- ✅ Added Distance operator initialization
- ✅ Added scaling factor: `1/√d_model`
- ✅ Modified forward pass to combine all components

**Key Code:**
```python
# Initialize components
self.legendre_embedding = LegendrePositionEmbedding(d_model, scaling=True)
self.distance_operator = DistancePositionOperator(decay_a=1.0, distance_type='l1')
self.distance_scale = 1.0 / math.sqrt(d_model)

# Forward pass
value_emb = self.value_embedding(x)
temporal_emb = self.temporal_embedding(x_mark)
legendre_pos = self.legendre_embedding(x)

# CRITICAL: Distance operator applied to Legendre embeddings (positional space)
# NOT to value embeddings (feature space)
distance_pos = self.distance_operator(legendre_pos) * self.distance_scale

x = value_emb + temporal_emb + legendre_pos + distance_pos
```

#### `models/legendre_embedding.py`
- Pre-computed Legendre polynomials for positions 0 to max_len
- Orthogonality verified
- Cached as buffer (non-trainable)

#### `models/distance_operator.py`
- Computes pairwise signed displacements: Δx_ij = P_i - P_j (where P is Legendre embedding)
- **CRITICAL:** Applied to Legendre embeddings (positional space), NOT value embeddings
- Applies index-based decay: α(i,j) = 1/(1 + |i-j|^a)
- Applies feature-space weighting: w_ij = 1/(1 + d_ij) where d_ij is computed from Legendre embeddings
- Aggregates with masking (j ≠ i)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ETTh1 |
| Model | Informer |
| Attention | Full (not ProbSparse) |
| Sequence Length | 96 |
| Label Length | 48 |
| Prediction Length | 24 |
| Encoder Layers | 2 |
| Decoder Layers | 1 |
| d_model | 512 |
| **Decay Parameter (a)** | 1.0 |
| **Distance Type** | L1 |
| **Scaling** | 1/√d_model |
| **Temporal Embedding** | ✅ Included |

---

## Key Differences from Other Experiments

| Aspect | Vanilla | Exp1 (D) | Exp2 (LOD) | Exp3 (L) | Exp4 (O) |
|--------|---------|----------|------------|----------|----------|
| **Label (L)** | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Order (O)** | ❌ | ❌ | ✅ | ❌ | ✅ |
| **Distance (D)** | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Temporal** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Standard PE** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Location** | Embedding | Attention | Embedding | Embedding | Embedding |
| **Complexity** | O(L) | O(L²) | O(L²) | O(L) | O(L²) |

---

## How to Run

```bash
cd /Users/neha/Desktop/PRL-SHIVANSH/distance-abl-PRL
bash experiments/exp2_full_paper/run_exp2.sh
```

**Note:** Training will be slower than Exp1 due to O(L²) distance operator.

---

## Expected Output

The script will:
1. Copy modified model files to `Informer2020/models/`
2. Train the model for 6 epochs (with early stopping)
3. Test on the test set
4. Print final metrics: MSE, MAE, RMSE, MAPE, MSPE
5. Save results to `results/exp2_full_paper/training_log.txt`

---

## Results

**Training completed with early stopping after epoch 4.**

| Metric | Value |
|--------|-------|
| MSE    | **0.8036** |
| MAE    | **0.7102** |
| RMSE   | 0.8964 |
| MAPE   | - |
| MSPE   | - |

**Best Validation Loss:** 0.7742 (Epoch 1)

---

## Comparison with Other Experiments

| Experiment | Components | Temporal | MSE ↓ | MAE ↓ | Rank |
|------------|-----------|----------|-------|-------|------|
| Vanilla | Standard PE | ✅ | **0.519** | **0.513** | 🥇 1st |
| Exp 1 (D) | Distance | ✅ | **0.725** | **0.652** | 🥈 2nd |
| **Exp 2 (LOD)** | **L+O+D** | ✅ | **0.804** | **0.710** | 🥉 **3rd** |
| Exp 3 (L) | Label | ❌ | **1.124** | **0.855** | 4th |
| Exp 4 (O) | Order | ✅ | ? | ? | ? |

---

## Analysis

### Key Findings

1. **LOD Underperforms Distance-Only (Exp1)**
   - Exp2 (LOD): MSE = 0.804
   - Exp1 (D): MSE = 0.725
   - **Difference:** +0.079 MSE (11% worse)

2. **Component Interaction**
   - Adding Label (L) and Order (O) to Distance (D) **degraded** performance
   - This suggests **negative synergy** between components
   - Distance alone is more effective than the full LOD formulation

3. **Ranking Summary**
   ```
   Vanilla (0.519) < Exp1-D (0.725) < Exp2-LOD (0.804) < Exp3-L (1.124)
   ```

4. **Possible Explanations**
   - **Overfitting:** Too many positional signals may confuse the model
   - **Interference:** Label and Order components may interfere with Distance
   - **Computational:** O(L²) complexity may require more training epochs
   - **Scaling:** Distance operator scaling (1/√d_model) may need tuning

---

## Analysis Questions

1. **Synergy Test:** Does combining L+O+D perform better than D alone (Exp1)?
2. **Baseline Comparison:** Does LOD approach vanilla performance (0.519)?
3. **Component Contribution:** Which component contributes most to performance?
4. **Computational Trade-off:** Is the O(L²) complexity justified by performance gains?

---

## Theoretical Context

From the paper's formulation:
- **Label:** Provides orthogonal distinctiveness (each position is unique)
- **Order:** Captures directional relationships (X_i relative to all X_j)
- **Distance:** Encodes proximity bias (nearby positions matter more)

**This experiment tests:** Do these three components work together synergistically to encode positional information effectively?

---

## Computational Complexity

- **Space:** O(B × L² × D) for distance matrix
- **Time:** O(B × L² × D) for pairwise computations
- **For seq_len=96, d_model=512:** ~4.7M operations per batch

**Acceptable** for research purposes, but may need optimization for production.
