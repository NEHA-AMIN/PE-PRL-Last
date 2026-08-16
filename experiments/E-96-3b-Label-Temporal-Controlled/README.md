# Experiment 3b: Label + Temporal Controlled

## 1. Experiment Purpose

### The Confound Problem in Exp 3

Experiment 3 (Label Only) implemented the following architecture:
```
X'_i = X_i + P_i
```

Where:
- `X_i` = Token embedding (semantic content)
- `P_i` = Legendre polynomial position embedding (label)

**The problem:** Exp 3 removed BOTH sinusoidal positional encoding AND temporal embedding simultaneously. When it achieved MSE = 1.0989 (pred=96, seed=2021, from `mse_mae_scores_sorted.txt`), compared to vanilla's MSE — TODO: vanilla baseline MSE could not be verified from the repository — we cannot determine if the poor performance was caused by:

1. **Legendre polynomials being an inadequate label**, OR
2. **Missing temporal context** (hour/day/month/weekday information)

This is a **confound** - two variables changed at once, making it impossible to isolate the cause of failure.

### Why Exp 3b Exists

**Experiment 3b exists solely to isolate the Label component by restoring temporal embedding.** This creates a controlled comparison where the ONLY difference between vanilla Informer and Exp 3b is the type of positional encoding:

- **Vanilla Informer:** Sinusoidal PE + Temporal Embedding
- **Exp 3b:** Legendre PE + Temporal Embedding

Everything else (attention mechanism, model architecture, temporal context) remains identical.

---

## 2. Side-by-Side Comparison Table

| Component | Vanilla Informer | Exp 3 (Label Only) | Exp 3b (Label+Temporal) |
|-----------|------------------|-------------------|------------------------|
| **Token Embedding** | ✅ | ✅ | ✅ |
| **Sinusoidal PE** | ✅ | ❌ | ❌ |
| **Temporal Embedding** | ✅ | ❌ | ✅ |
| **Legendre Label** | ❌ | ✅ | ✅ |
| **Ordering Operator** | ❌ | ❌ | ❌ |
| **Distance Decay** | ❌ | ❌ | ❌ |
| **Attention Type** | Full/Prob | Full | Full |
| **MSE (ETTh1, pred=96)** | TODO: unverifiable from repository | 1.0989 (seed=2021, Phase 1) | **0.8858 (avg 3 seeds, Phase 2)** |

---

## 3. Mathematical Formulation

### Vanilla Informer:
```
X'_i = X_i + PE_i + T_i
```
Where:
- `X_i` = Token embedding (Conv1D)
- `PE_i` = Sinusoidal positional encoding
- `T_i` = Temporal embedding (hour/day/month/weekday)

### Exp 3 (Label Only):
```
X'_i = X_i + P_i
```
Where:
- `X_i` = Token embedding (Conv1D)
- `P_i` = Legendre polynomial label
- **Missing:** Both PE_i and T_i

### Exp 3b (Label + Temporal Controlled):
```
X'_i = X_i + T_i + P_i
```
Where:
- `X_i` = Token embedding (Conv1D)
- `T_i` = Temporal embedding (hour/day/month/weekday) - **RESTORED**
- `P_i` = Legendre polynomial label
- **Missing:** Only PE_i (sinusoidal)

---

## 4. What This Experiment Answers

### Research Questions:

1. **Is Exp 3's poor performance (MSE = 1.0989 at pred=96, seed=2021) due to Legendre being a bad label, or due to missing temporal context?**

2. **When temporal context is restored, how does Legendre-only labelling perform compared to vanilla sinusoidal PE?**

### Controlled Comparison:

By restoring temporal embedding, we create a fair comparison:

- **Exp 3b vs Vanilla:** Isolates the effect of PE type (Legendre vs Sinusoidal)
- **Exp 3b vs Exp 3:** Isolates the effect of temporal embedding
- **Exp 3b vs Exp 5b:** Shows what ordering adds on top of label+temporal

---

## 5. Results and Interpretation

### Actual Results (from `mse_mae_scores_sorted.txt`)

#### Phase 1 — Screen (seed=2021 only)

| pred_len | MSE    | MAE    |
|----------|--------|--------|
| 96       | 0.9265 | 0.7660 |
| 192      | 0.8888 | 0.7347 |

#### Phase 2 — Full Sweep (3 seeds × 4 pred_lens = 12 runs)

| pred_len | seed | MSE    | MAE    |
|----------|------|--------|--------|
| 48       | 2021 | 0.8706 | 0.7344 |
| 48       | 2022 | 0.8937 | 0.7399 |
| 48       | 2023 | 0.9917 | 0.7959 |
| 96       | 2021 | 0.8823 | 0.7573 |
| 96       | 2022 | 0.8450 | 0.7373 |
| 96       | 2023 | 0.9302 | 0.7362 |
| 192      | 2021 | 0.9006 | 0.7564 |
| 192      | 2022 | 0.8847 | 0.7430 |
| 192      | 2023 | 0.8886 | 0.7564 |
| 336      | 2021 | 0.9989 | 0.8081 |
| 336      | 2022 | 0.9770 | 0.7949 |
| 336      | 2023 | 0.9244 | 0.7550 |

#### Phase 2 — Averages across 3 seeds

| pred_len | Avg MSE | Avg MAE | #Runs |
|----------|---------|---------|-------|
| 48       | 0.9186  | 0.7567  | 3     |
| 96       | 0.8858  | 0.7436  | 3     |
| 192      | 0.8913  | 0.7519  | 3     |
| 336      | 0.9668  | 0.7860  | 3     |

---

### Three Possible Outcomes (Pre-registered)

#### Scenario A: Exp 3b MSE ≈ Exp 3 MSE (~1.1)
**Interpretation:** Temporal embedding is NOT the primary cause of failure. Legendre labelling genuinely fails without additional structure (ordering/distance). The label alone is insufficient for the model to learn temporal patterns.

**Scientific Conclusion:** Pure orthogonal distinctiveness (Legendre) cannot replace sinusoidal PE without additional inductive biases.

---

#### Scenario B: Exp 3b MSE << Exp 3 MSE (e.g., ~0.7)
**Interpretation:** Temporal embedding was the primary cause of Exp 3's failure. Legendre labelling is adequate when combined with temporal context. The poor performance in Exp 3 was due to missing hour/day/month information, not the Legendre polynomials themselves.

**Scientific Conclusion:** Legendre labels can work, but require temporal context to be effective. The label provides distinctiveness, while temporal embedding provides semantic time information.

---

#### Scenario C: Exp 3b MSE ≈ Vanilla MSE (~0.52)
**Interpretation:** Legendre polynomials are a valid replacement for sinusoidal PE when combined with temporal context. The orthogonal distinctiveness property is sufficient for position encoding in time series forecasting.

**Scientific Conclusion:** Sinusoidal PE is not strictly necessary - orthogonal polynomial bases can serve as effective positional encodings when paired with temporal embeddings.

---

### Observed Outcome

The actual Phase 2 averages place Exp 3b between Scenario A and Scenario B. Exp 3b avg MSE (pred=96: 0.8858) is substantially below Exp 3 Phase 1 (1.0989), confirming that missing temporal embedding was a major cause of Exp 3's failure. However, Exp 3b remains above Exp1-Pre avg MSE (pred=96: 0.8670) — see reference table in `exp3b_phase2.sh`. Scenario B (partial recovery) is the best characterisation: restoring temporal context significantly helps, but Legendre-only labelling does not fully close the gap to Exp1-Pre at pred=96 or pred=336.

TODO: Vanilla baseline MSE could not be verified from the repository; a direct Exp 3b vs. Vanilla comparison cannot be confirmed numerically.

---

## 6. Implementation Notes

### Code Changes from Vanilla Informer:

**ONLY ONE FILE DIFFERS:** `embed.py`

#### Key Modifications:

1. **Import added at top:**
   ```python
   from legendre_embedding import LegendrePositionEmbedding
   ```

2. **In `DataEmbedding.__init__`:**
   - ✅ Keeps `self.temporal_embedding` (RESTORED from vanilla)
   - ✅ Adds `self.legendre_embedding` (Legendre label)
   - ❌ Does NOT initialize `self.position_embedding` (sinusoidal PE removed)

3. **In `DataEmbedding.forward`:**
   ```python
   value_emb = self.value_embedding(x)           # Token embedding
   temporal_emb = self.temporal_embedding(x_mark) # Temporal context (RESTORED)
   legendre_pos = self.legendre_embedding(x)      # Legendre label
   
   x = value_emb + temporal_emb + legendre_pos    # Three components
   return self.dropout(x)
   ```

### Files Identical to Vanilla:

- `attn.py` - Standard FullAttention (no distance decay, no ordering)
- `encoder.py` - Standard Informer encoder
- `decoder.py` - Standard Informer decoder
- `model.py` - Standard Informer model
- `__init__.py` - Empty file

### Files Identical to Exp 3:

- `legendre_embedding.py` - Legendre polynomial implementation (unchanged)

---

## 7. How to Run

### Execute the experiment:

```bash
# Phase 1 — quick screen (pred_len 96 & 192, seed 2021, 2 runs)
bash experiments/E-96-3b-Label-Temporal-Controlled/exp3b_phase1.sh

# Phase 2 — full sweep (3 seeds × 4 pred_lens = 12 runs)
bash experiments/E-96-3b-Label-Temporal-Controlled/exp3b_phase2.sh
```

### Experiment Configuration:

- **Dataset:** ETTh1 (hourly)
- **Seeds:** 2021, 2022, 2023
- **Prediction Lengths:** 48, 96, 192, 336
- **Total Runs:** 12 (1 dataset × 3 seeds × 4 pred_lens, Phase 2)

### Hyperparameters:

```bash
--model informer
--data ETTh1
--root_path ./data/ETT/
--data_path ETTh1.csv
--features M
--target OT
--freq h
--seq_len 96
--label_len 48
--enc_in 7
--dec_in 7
--c_out 7
--d_model 512
--n_heads 8
--e_layers 2
--d_layers 1
--d_ff 2048
--factor 5
--padding 0
--distil
--dropout 0.05
--attn full
--embed timeF
--activation gelu
--mix
--train_epochs 6
--batch_size 32
--patience 3
--learning_rate 0.0001
--lradj type1
--itr 1
--num_workers 0
```

### Import Resolution Note

`embed.py` uses a bare `from legendre_embedding import LegendrePositionEmbedding`. The shell scripts resolve this by:
1. Setting `PYTHONPATH="$INFORMER_DIR/models:$INFORMER_DIR:$PYTHONPATH"` at run time.
2. Copying `legendre_embedding.py` to **both** `$INFORMER_DIR/models/` and `$INFORMER_DIR/` (Informer root).

After each run, the scripts restore the original Informer models via `git checkout -- ./models/` and remove the root-level copy.

### Results Location:

Phase 2 result directories follow the naming pattern:

```
results/exp3b_ph2_ETTh1_leg_temporal_pred{pred_len}_seed{seed}/
```

Examples:
```
results/exp3b_ph2_ETTh1_leg_temporal_pred48_seed2021/
results/exp3b_ph2_ETTh1_leg_temporal_pred96_seed2021/
results/exp3b_ph2_ETTh1_leg_temporal_pred192_seed2021/
results/exp3b_ph2_ETTh1_leg_temporal_pred336_seed2023/
```

Phase 1 result directories follow:
```
results/exp3b_ph1_ETTh1_leg_temporal_pred{pred_len}_seed2021/
```

---

## 8. Result Interpretation Guide

#### Compare Exp 3b vs Exp 3 (Isolate Temporal Effect):

```bash
# Exp 3 (Label Only, pred=96, seed=2021): MSE = 1.0989
# Exp 3b (Label + Temporal, pred=96, avg 3 seeds): MSE = 0.8858

# Exp 3b << Exp 3:
#   → Temporal embedding was critical
#   → Legendre labels work substantially better with temporal context
```

#### Compare Exp 3b vs Vanilla (Isolate PE Type):

```bash
# Vanilla (Sinusoidal + Temporal): MSE = TODO: unverifiable from repository
# Exp 3b (Legendre + Temporal, pred=96, avg 3 seeds): MSE = 0.8858

# If Exp 3b ≈ Vanilla:
#   → Legendre is a valid PE replacement
#   → Orthogonal distinctiveness sufficient

# If Exp 3b > Vanilla:
#   → Sinusoidal PE has advantages
#   → Smooth periodic structure matters
```

#### Compare Exp 3b vs Exp 5b (Isolate Ordering Effect):

```bash
# Exp 3b (Label + Temporal, pred=96, avg 3 seeds): MSE = 0.8858
# Exp 5b (Label + Temporal + Ordering): MSE = TODO: value not available in this experiment's scope

# Difference shows the value of ordering operator
# on top of label + temporal baseline
```

### Key Metrics to Track:

1. **MSE** - Primary performance metric
2. **MAE** - Robustness to outliers
3. **Training Stability** - Loss curves
4. **Convergence Speed** - Epochs to best validation

### Statistical Significance:

With 3 seeds per configuration, compute:
- Mean MSE across seeds
- Standard deviation
- 95% confidence intervals

---

## Summary

**Exp 3b is a controlled experiment** designed to answer: "Is Legendre labelling inadequate, or was Exp 3's failure due to missing temporal context?"

By restoring temporal embedding while keeping Legendre labels, we isolate the effect of positional encoding type (Sinusoidal vs Legendre) in a fair comparison with vanilla Informer.

Both phases completed. Phase 2 (12 runs: 3 seeds × 4 pred_lens) shows Exp 3b avg MSE of 0.8858 at pred=96, substantially below Exp 3's 1.0989 — confirming temporal embedding was a major cause of Exp 3's failure. Legendre labels combined with temporal context provide meaningful partial recovery. Exp 3b does not fully reach Exp1-Pre performance (pred=96 avg 0.8670), indicating that sinusoidal PE carries additional information beyond what Legendre labels alone provide.