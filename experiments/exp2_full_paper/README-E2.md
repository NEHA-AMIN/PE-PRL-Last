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
  α(i,j) = 1 / (1 + |i-j|^a)        [Index-based decay, a=1.0 default in embed.py]
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
- ✅ Removed standard positional embedding (instantiated but never called in `forward()`)
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

**Note:** `self.position_embedding` (standard sinusoidal PE) is still instantiated in `__init__` as dead weight but is never called in `forward()`. It adds unused parameters to the checkpoint.

**Note on `decay_a`:** The default value in `embed.py` is `decay_a=1.0`. During Phase 1 runs, the shell script patches this value in-place via `sed` for each alpha sweep. There is no CLI argument for `decay_a` in this experiment — it is hardcoded and overwritten by the script.

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

| Parameter | Value | Source |
|-----------|-------|--------|
| Dataset | ETTh1 | `exp2_phase1.sh` |
| Model | Informer | `exp2_phase1.sh` |
| Attention | Full (not ProbSparse) | `--attn full` |
| Distillation | Enabled (default, `--distil` not overridden) | Informer default |
| Sequence Length | 96 | `--seq_len 96` |
| Label Length | 48 | `--label_len 48` |
| Prediction Length | 96, 192 (Phase 1); 192, 336 (Phase 2 script) | shell scripts |
| Encoder Layers | 2 | `--e_layers 2` |
| Decoder Layers | 1 | `--d_layers 1` |
| d_model | 512 | `--d_model 512` |
| n_heads | 8 | `--n_heads 8` |
| d_ff | 2048 | `--d_ff 2048` |
| dropout | 0.05 | `--dropout 0.05` |
| Embed type | timeF | `--embed timeF` |
| Frequency | h | `--freq h` |
| Activation | gelu | `--activation gelu` |
| Learning rate | 0.0001 | `--learning_rate 0.0001` |
| Batch size | 32 | `--batch_size 32` |
| Factor | 5 | `--factor 5` |
| enc_in / dec_in / c_out | 7 | ETTh1 multivariate (7 features) |
| Train epochs | 6 | `--train_epochs 6` |
| Patience (early stop) | 3 | `--patience 3` |
| **Decay Parameter (a)** | 1.0 default in `embed.py`; swept {0.5, 1.0, 2.0} in Phase 1; fixed 0.5 in Phase 2 | `embed.py`, shell scripts |
| **Distance Type** | L1 | `distance_type='l1'` |
| **Scaling** | 1/√d_model | `embed.py` |
| **Temporal Embedding** | ✅ Included | `embed.py` |

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

## Execution Protocol

This experiment uses a **two-phase protocol**:

### Phase 1 — Alpha Exploration (`exp2_phase1.sh`)

- **Goal:** Identify the best decay parameter `a` before committing to full multi-seed runs.
- **Alpha values swept:** {0.5, 1.0, 2.0}
- **pred_len:** {96, 192}
- **Seed:** 2021 (single — exploration only)
- **Total runs:** 6 (3 alpha × 1 seed × 2 pred_len)
- **Decision rule:** If same alpha wins at both pred_lens → proceed to Phase 2 with that alpha. If different alpha wins at each → run full grid.

### Phase 2 — Results Stability (`exp2_phase2_alpha0.5.sh`)

- **Goal:** Confirm stability of best alpha across multiple seeds and longer horizons.
- **Alpha fixed:** 0.5
- **pred_len (script):** {192, 336}
- **Seeds:** {2021, 2022, 2023}
- **Total runs (script):** 6 (2 pred_len × 3 seeds)

**Note:** `mse_mae_scores_sorted.txt` records Phase 2 results for pred_len ∈ {48, 96, 192} with 3 seeds (9 runs). Results for pred_len=336 are not present in the central log file, which is inconsistent with the Phase 2 script's declared pred_len loop. The pred_len=336 Phase 2 runs may not have completed or were not recorded.

### How to Run

```bash
# Phase 1 — alpha exploration
bash experiments/exp2_full_paper/exp2_phase1.sh

# Phase 2 — stability confirmation (alpha=0.5)
bash experiments/exp2_full_paper/exp2_phase2_alpha0.5.sh
```

**Note:** Training will be slower than Exp1 due to O(L²) distance operator.

---

## Expected Output

Each phase script will:
1. Copy modified model files to `Informer2020-original/models/`
2. Patch `decay_a` in `embed.py` via `sed` (Phase 1 only)
3. Train the model for up to 6 epochs (with early stopping, patience=3)
4. Test on the test set
5. Print final metrics: MSE, MAE
6. Save per-run logs to `logs/exp2_phase1/<RUN_ID>.log` or `logs/exp2_phase2/<RUN_ID>.log`
7. Save results to `results/<RUN_ID>/`

---

## Results

Source: `mse_mae_scores_sorted.txt`

### Phase 1 — Alpha Exploration (pred_len ∈ {96, 192}, seed=2021)

**Note:** Only α=1.0 runs are recorded in `mse_mae_scores_sorted.txt`. Results for α=0.5 and α=2.0 are absent from the central log file.

| Alpha | pred_len | MSE | MAE |
|-------|---------|-----|-----|
| 1.0 | 96 | 0.8242 | 0.7279 |
| 1.0 | 192 | 0.9002 | 0.7511 |

### Phase 2 — Results Stability (alpha=0.5, seeds 2021/2022/2023)

**Note:** The Phase 2 script targets pred_len ∈ {192, 336}, but the central results file records results for pred_len ∈ {48, 96, 192}. pred_len=336 results are absent.

#### Per-run results

| pred_len | Seed | MSE | MAE |
|---------|------|-----|-----|
| 48 | 2021 | 0.9983 | 0.7945 |
| 48 | 2022 | 0.7760 | 0.6871 |
| 48 | 2023 | 0.9125 | 0.7590 |
| 96 | 2021 | 0.8476 | 0.7159 |
| 96 | 2022 | 0.8439 | 0.7219 |
| 96 | 2023 | 0.8688 | 0.7373 |
| 192 | 2021 | 1.1408 | 0.8648 |
| 192 | 2022 | 0.8643 | 0.7345 |
| 192 | 2023 | 0.8823 | 0.7523 |

#### Average across seeds

| pred_len | Avg MSE | Avg MAE | #Runs |
|---------|---------|---------|-------|
| 48 | 0.8956 | 0.7469 | 3 |
| 96 | 0.8534 | 0.7250 | 3 |
| 192 | 0.9625 | 0.7839 | 3 |
| 336 | TODO: pred_len=336 Phase 2 results not found in mse_mae_scores_sorted.txt | — | 0 |

---

## Comparison with Other Experiments

Source: `mse_mae_scores_sorted.txt`. Values shown are **Phase 2 averages** (3 seeds) where available; Phase 1 single-seed otherwise. All pred_len values shown separately — no single-point aggregate across pred_lens is used.

### Exp2 (LOD) vs Exp1-Pre (D only) — Phase 2 averages

| Experiment | Components | pred_len=48 Avg MSE | pred_len=96 Avg MSE | pred_len=192 Avg MSE | pred_len=336 Avg MSE |
|------------|-----------|---------------------|---------------------|----------------------|----------------------|
| Exp1-Pre (D) | Distance pre-softmax | 0.7980 | 0.8670 | 0.9373 | 1.0368 |
| **Exp2 (LOD)** | **L+O+D** | **0.8956** | **0.8534** | **0.9625** | — |
| Exp3b (L) | Label only | 0.9186 | 0.8858 | 0.8913 | 0.9668 |
| Exp4 (O) | Order only | 0.8658 | 1.0035 | 0.9507 | — |
| Vanilla | Standard PE | TODO: Information could not be verified from the repository. | | | |

**Note:** Vanilla baseline values are not present in `mse_mae_scores_sorted.txt`. The `results/baseline_ph1_ETTh1_pred96_seed2021/` and `results/baseline_ph1_ETTh1_pred192_seed2021/` directories exist but are empty.

---

## Analysis

### Key Findings

1. **LOD vs Distance-Only (Exp1-Pre) — pred_len=96**
   - Exp2 (LOD) avg MSE at pred_96: 0.8534
   - Exp1-Pre (D) avg MSE at pred_96: 0.8670
   - At pred_len=96, LOD is marginally better than Distance-only by 0.0136 MSE.

2. **LOD vs Distance-Only (Exp1-Pre) — pred_len=48**
   - Exp2 (LOD) avg MSE at pred_48: 0.8956
   - Exp1-Pre (D) avg MSE at pred_48: 0.7980
   - At pred_len=48, Distance-only outperforms LOD by 0.0976 MSE.

3. **Seed Instability at pred_len=192**
   - Seed 2021 gives MSE=1.1408 while seeds 2022 and 2023 give 0.8643 and 0.8823.
   - The average (0.9625) is dominated by the outlier at seed=2021. Median would be 0.8823.

4. **Component Interaction**
   - Results are mixed across pred_len: LOD does not consistently outperform or underperform single-component variants.
   - No clear synergistic benefit of combining all three components is evident from the data.

5. **Possible Explanations**
   - **Overfitting:** Too many positional signals may confuse the model
   - **Interference:** Label and Order components may interfere with Distance
   - **Computational:** O(L²) complexity may require more training epochs
   - **Scaling:** Distance operator scaling (1/√d_model) may need tuning
   - **Double-counting:** As documented in `NOTE.md`, `w_ij` is highly correlated with `α(i,j)` when both are applied to Legendre embeddings (both are monotone functions of |i-j|), causing the combined weight to decay faster than either factor alone

---

## Analysis Questions

1. **Synergy Test:** Does combining L+O+D perform better than D alone (Exp1) consistently across all pred_lens?
2. **Baseline Comparison:** Does LOD approach vanilla performance? (Baseline not verifiable — see Results section.)
3. **Component Contribution:** Which component contributes most to performance?
4. **Computational Trade-off:** Is the O(L²) complexity justified by performance gains?

---

## Theoretical Context

From the paper's formulation:
- **Label:** Provides orthogonal distinctiveness (each position is unique)
- **Order:** Captures directional relationships (X_i relative to all X_j)
- **Distance:** Encodes proximity bias (nearby positions matter more)

**This experiment tests:** Do these three components work together synergistically to encode positional information effectively?

**Mathematical note (from `NOTE.md`):** When `distance_operator` is applied to Legendre embeddings (not value embeddings), the feature-space weighting `w_ij = 1/(1+||P_i−P_j||_1)` becomes a deterministic fixed function of index distance (because Legendre embeddings are non-trainable buffers). This makes `w_ij` redundant with the index-based decay `α(i,j)` — both suppress long-range pairs and amplify short-range pairs. The effective attention window shrinks significantly due to this double-decay. For middle positions of the sequence, left/right symmetric contributions nearly cancel, yielding a near-zero ordering signal for most positions.

---

## Computational Complexity

- **Space:** O(B × L² × D) for distance matrix
- **Time:** O(B × L² × D) for pairwise computations
- **For seq_len=96, d_model=512:** ~4.7M operations per batch

**Acceptable** for research purposes, but may need optimization for production.
