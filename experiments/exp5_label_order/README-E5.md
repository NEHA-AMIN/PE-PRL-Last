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
  X_i = value_embedding(x)                              [Semantic content]
  T_i = temporal_embedding(x_mark)                      [Time features]
  P_i = Legendre(i)                                     [LABEL - Equation 1]
  O_i = (1/(L-1)) · Σ_{j≠i} (P_i - P_j) / √d_model   [ORDER - Equation 3, applied in positional space]
```

### Components

**1. LABEL (P_i) - Equation 1:**
```
Legendre Polynomials: P_i = [L_0(x_i), ..., L_{d-1}(x_i)]
Orthogonality: ⟨P_n, P_m⟩ = δ_{nm}
```
- Provides distinctiveness (each position unique)
- Scaled by 1/√d_model
- Pre-computed and cached as non-trainable buffer (`register_buffer`)

**2. ORDER (O_i) - Equation 3:**
```
Input:        P_i (Legendre positional embeddings — NOT value embeddings)
Displacement: Δp_ij = P_i - P_j
Aggregation:  O_i = (1/(L-1)) · Σ_{j≠i} Δp_ij
Scaling:      O_i = O_i / √d_model   (applied in embed.py after OrderingOperator)
```
- Applied to **Legendre positional embeddings** (positional space), NOT to value embeddings (semantic space)
- Provides directionality (relative positioning)
- Uniform weighting (no distance decay)
- Allocates `[B, L, L, D]` tensor internally — O(L²) in both time and space

**3. NO DISTANCE:**
- ❌ No α(i,j) index-based decay
- ❌ No w_ij feature-space weighting

**Note on dead weight:** `self.position_embedding` (standard sinusoidal PE) is instantiated in `DataEmbedding.__init__()` for interface compatibility but is never called in `forward()`.

---

## Implementation

### Components Used

**From Experiment 3:**
- `legendre_embedding.py` - Label component

**From Experiment 4:**
- `ordering_operator.py` - Order component

**New:**
- `embed.py` - Combines both L + O

### Forward Pass (actual code in `models/embed.py`)
```python
# 1. Value embedding
value_emb = self.value_embedding(x)

# 2. Temporal embedding
temporal_emb = self.temporal_embedding(x_mark)

# 3. LABEL: Legendre polynomials (positional distinctiveness)
legendre_pos = self.legendre_embedding(x)

# 4. ORDER: Signed displacements in POSITIONAL space
# CRITICAL: Applied to legendre_pos (positional), NOT value_emb (semantic)
ordering_pos = self.ordering_operator(legendre_pos)
ordering_pos = ordering_pos / math.sqrt(value_emb.size(-1))  # scale for stability

# 5. COMBINE: value + temporal + label + order
x = value_emb + temporal_emb + legendre_pos + ordering_pos
```

---

## Key Differences from Other Experiments

| Aspect | Exp 3 (L) | Exp 4 (O) | **Exp 5 (L+O)** | Exp 2 (LOD) |
|--------|-----------|-----------|-----------------|-------------|
| **Label** | ✅ | ❌ | ✅ | ✅ |
| **Order** | ❌ | ✅ | ✅ | ✅ |
| **Distance** | ❌ | ❌ | ❌ | ✅ |
| **Temporal** | ❌ | ✅ | ✅ | ✅ |
| **Order input** | — | value_emb (semantic) | legendre_pos (positional) | legendre_pos (positional) |
| **Components** | L only | O only | **L+O** | L+O+D |

---

## Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Dataset | ETTh1 | `--data ETTh1` |
| Model | Informer | `--model informer` |
| Attention | Full (not ProbSparse) | `--attn full` |
| Distillation | Enabled (default, `--distil` not overridden) | Informer default |
| Sequence Length | 96 | `--seq_len 96` |
| Label Length | 48 | `--label_len 48` |
| Prediction Length | 96, 192 (Phase 1); 48, 96, 192 (Phase 2) | shell scripts |
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
| Train epochs | 6 | `--train_epochs 6` |
| Patience (early stop) | 3 | `--patience 3` |
| Factor | 5 | `--factor 5` |
| enc_in / dec_in / c_out | 7 | ETTh1 multivariate (7 features) |
| Temporal | ✅ Included | `embed.py` |

---

## File Structure
```
experiments/exp5_label_order/
├── models/
│   ├── legendre_embedding.py   [FROM EXP3] - Label
│   ├── ordering_operator.py    [FROM EXP4] - Order
│   ├── embed.py                [NEW] - Combines L+O
│   └── ... (vanilla files)
├── README-E5.md                 - This file
├── theory.md                    - Detailed theoretical analysis and phase notes
├── Exp5_label_order.ipynb       - Experiment notebook
├── exp5_label_order_ph1.sh      - Phase 1 training script
└── exp5_label_order_ph2.sh      - Phase 2 training script
```

---

## Execution Protocol

This experiment uses a **two-phase protocol**:

### Phase 1 — Feasibility Check (`exp5_label_order_ph1.sh`)

- **Goal:** Test the L+O combination at two pred_lens with a single seed before committing to multi-seed runs.
- **pred_len:** {96, 192}
- **Seed:** 2021 (single — exploration only)
- **Total runs:** 2

### Phase 2 — Validation (`exp5_label_order_ph2.sh`)

- **Goal:** Confirm stability across multiple seeds and prediction lengths.
- **pred_len:** {48, 96, 192}
- **Seeds:** {2021, 2022, 2023}
- **Total runs:** 9 (3 seeds × 3 pred_lens)

**Note:** The Phase 2 script header comment says "12 (4 pred_len × 3 seeds)" but the actual loop iterates only `for pred_len in 48 96 192` (3 values). pred_len=336 is not in the loop. The actual completed run count is 9.

### How to Run

```bash
# Phase 1 — feasibility check
bash experiments/exp5_label_order/exp5_label_order_ph1.sh

# Phase 2 — multi-seed validation
bash experiments/exp5_label_order/exp5_label_order_ph2.sh
```

**Warning:** O(L²) operator — expect ~40-100s per epoch.

## Expected Output

Each script will:
1. Copy modified model files to `Informer2020-original/models/`
2. Train the model for up to 6 epochs (with early stopping, patience=3)
3. Test on the test set
4. Print final metrics: MSE, MAE
5. Save per-run logs to `logs/exp5_label_order_phase1/<RUN_ID>.log` or `logs/exp5_label_order_phase2/<RUN_ID>.log`
6. Save per-run results to `results/<RUN_ID>/`

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
| Vanilla | TODO: Information could not be verified from the repository. | Baseline |
| Exp 1-Pre (D) | Phase 2 avg pred_96: 0.8670 | Distance alone |
| Exp 4 (O) | Phase 2 avg pred_96: 1.0035 | Order alone |
| **Exp 5 (L+O)** | **Phase 2 avg pred_96: 0.8655** | **Label + Order** |
| Exp 3b (L) | Phase 2 avg pred_96: 0.8858 | Label alone (with temporal) |

---

## Results

Source: `mse_mae_scores_sorted.txt`

### Phase 1 (pred_len ∈ {96, 192}, seed=2021)

| pred_len | Seed | MSE | MAE |
|---------|------|-----|-----|
| 96 | 2021 | 0.8519 | 0.7383 |
| 192 | 2021 | 0.9788 | 0.7980 |

Phase 1 interpretation (from `exp5_label_order_ph2.sh` header):
- pred=96: MSE=0.8519 beats Exp1-Pre (0.8683) ✅
- pred=192: MSE=0.9788 worse than Exp1-Pre (0.8463) ❌
- Adding Order to Label helps at short horizon, less clear at long horizon

### Phase 2 (pred_len ∈ {48, 96, 192}, seeds 2021/2022/2023)

#### Per-run results

| pred_len | Seed | MSE | MAE |
|---------|------|-----|-----|
| 48 | 2021 | 0.8328 | 0.7210 |
| 48 | 2022 | 0.7529 | 0.6773 |
| 48 | 2023 | 0.8448 | 0.7343 |
| 96 | 2021 | 0.8972 | 0.7471 |
| 96 | 2022 | 0.8450 | 0.7387 |
| 96 | 2023 | 0.8545 | 0.7253 |
| 192 | 2021 | 1.1154 | 0.8443 |
| 192 | 2022 | 0.8733 | 0.7359 |
| 192 | 2023 | 0.9154 | 0.7534 |

#### Average across seeds

| pred_len | Avg MSE | Avg MAE | #Runs |
|---------|---------|---------|-------|
| 48 | 0.8102 | 0.7109 | 3 |
| 96 | 0.8655 | 0.7370 | 3 |
| 192 | 0.9680 | 0.7779 | 3 |

---

## Complete Comparison Table

Source: `mse_mae_scores_sorted.txt`. Values shown are Phase 2 averages (3 seeds) where available; Phase 1 single-seed otherwise.

| Experiment | Components | pred_len=48 Avg MSE | pred_len=96 Avg MSE | pred_len=192 Avg MSE | Notes |
|------------|-----------|---------------------|---------------------|----------------------|-------|
| Vanilla | Standard PE | TODO: Information could not be verified from the repository. | | | Baseline dirs empty |
| Exp 1-Pre (D) | Distance pre-softmax | 0.7980 | 0.8670 | 0.9373 | Phase 2 avg, α=1.0 |
| Exp 2 (LOD) | L+O+D | 0.8956 | 0.8534 | 0.9625 | Phase 2 avg |
| **Exp 5 (L+O)** | **Label+Order** | **0.8102** | **0.8655** | **0.9680** | **Phase 2 avg** |
| Exp 3b (L) | Label only | 0.9186 | 0.8858 | 0.8913 | Phase 2 avg |
| Exp 4 (O) | Order only (semantic) | 0.8658 | 1.0035 | 0.9507 | Phase 2 avg |

**Note:** Vanilla baseline values are not present in `mse_mae_scores_sorted.txt`. The `results/baseline_ph1_ETTh1_pred96_seed2021/` and `results/baseline_ph1_ETTh1_pred192_seed2021/` directories exist but are empty.

---

## Analysis

### Key Findings

1. **L+O vs Label-only (Exp3b) — pred_len=96**
   - Exp5 (L+O) avg MSE at pred_96: 0.8655
   - Exp3b (L) avg MSE at pred_96: 0.8858
   - At pred_len=96, adding Order to Label improves MSE by 0.0203.

2. **L+O vs Distance-only (Exp1-Pre) — pred_len=96**
   - Exp5 (L+O) avg MSE at pred_96: 0.8655
   - Exp1-Pre (D) avg MSE at pred_96: 0.8670
   - At pred_len=96, L+O is marginally better than Distance-only by 0.0015 MSE.

3. **L+O vs Order-only (Exp4) — pred_len=96**
   - Exp5 (L+O) avg MSE at pred_96: 0.8655
   - Exp4 (O) avg MSE at pred_96: 1.0035
   - Adding Label to Order substantially improves performance (+0.1380 MSE reduction).

4. **L+O vs L+O+D (Exp2) — pred_len=96**
   - Exp5 (L+O) avg MSE at pred_96: 0.8655
   - Exp2 (L+O+D) avg MSE at pred_96: 0.8534
   - At pred_len=96, adding Distance to L+O slightly improves MSE by 0.0121.

5. **Seed Instability at pred_len=192**
   - Seed 2021 gives MSE=1.1154 while seeds 2022 and 2023 give 0.8733 and 0.9154.
   - The average (0.9680) is elevated by the seed=2021 outlier. Median would be 0.9154.

6. **Possible Explanations**
   - **Positional Space is Key:** Applying Order in positional (Legendre) space rather than semantic space may create a purer positional signal
   - **Uniform weighting** preserves the signal better than distance-based weighting for short horizons
   - **Temporal embedding** (kept in Exp5, absent in Exp3) may be crucial for L+O to work

### Why Did L+O Outperform D at pred_96?

**Complementary Strengths:**
- **Label (L):** Provides orthogonal distinctiveness (each position unique)
- **Order (O):** Provides directional relationships (relative positioning in Legendre space)
- **Together:** Create a richer positional structure than distance decay alone at short horizons

**Why Distance Hurts at pred_96:**
- Adding Distance decay (α) and feature-space weighting (w_ij) to L+O slightly degrades pred_96 performance (Exp2 0.8534 vs Exp5 0.8655 — Exp2 is actually slightly better here)
- The relationship is pred_len-dependent: benefits of adding Distance vary by horizon

### Theoretical Implications

This experiment directly tests the "PoPE + ΔV" scenario.

**Our findings (from Phase 2 averages):**
- L+O (pred_96 avg: 0.8655) performs comparably to D alone (pred_96 avg: 0.8670)
- L+O shows better stability than O alone (Exp4 pred_96 avg: 1.0035)
- Results are pred_len-dependent — no single experiment dominates across all horizons

---

## Analysis Questions

1. **Do L and O synergize?**
   - If Exp5 < Exp3 AND Exp5 < Exp4 → YES (positive)
   - If Exp5 ≈ Exp3 OR Exp5 ≈ Exp4 → NO (one dominates)
   - If Exp5 > both → Negative interference

2. **Does removing Distance help or hurt?**
   - Compare Exp5 (L+O) vs Exp2 (L+O+D)
   - If Exp5 < Exp2 → Distance hurts
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
