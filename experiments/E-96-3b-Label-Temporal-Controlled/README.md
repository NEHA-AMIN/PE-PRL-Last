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

**The problem:** Exp 3 removed BOTH sinusoidal positional encoding AND temporal embedding simultaneously. When it achieved MSE = 1.124 (compared to vanilla's 0.519), we cannot determine if the poor performance was caused by:

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
| **MSE (ETTh1, pred=96)** | 0.519 | 1.124 | **TBD** |

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

1. **Is Exp 3's poor performance (MSE = 1.124) due to Legendre being a bad label, or due to missing temporal context?**

2. **When temporal context is restored, how does Legendre-only labelling perform compared to vanilla sinusoidal PE?**

### Controlled Comparison:

By restoring temporal embedding, we create a fair comparison:

- **Exp 3b vs Vanilla:** Isolates the effect of PE type (Legendre vs Sinusoidal)
- **Exp 3b vs Exp 3:** Isolates the effect of temporal embedding
- **Exp 3b vs Exp 5b:** Shows what ordering adds on top of label+temporal

---

## 5. Expected Results and Interpretation

### Three Possible Outcomes:

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
bash experiments/E-96-3b-Label-Temporal-Controlled/run_exp3b.sh
```

### Experiment Configuration:

- **Datasets:** ETTh1 (hourly), ETTm1 (15-minute)
- **Seeds:** 2021, 2022, 2023
- **Prediction Lengths:** 48, 96, 192, 336, 720
- **Total Runs:** 30 (2 datasets × 3 seeds × 5 pred_lens)

### Hyperparameters:

```bash
--seq_len 96
--label_len 48
--d_model 512
--n_heads 8
--e_layers 2
--d_layers 1
--d_ff 2048
--batch_size 32
--learning_rate 0.0001
--dropout 0.05
--attn full
--embed timeF
```

### Results Location:

```
results/exp3b_label_temporal_ETTh1_48_seed2021/
results/exp3b_label_temporal_ETTh1_96_seed2021/
results/exp3b_label_temporal_ETTh1_192_seed2021/
...
results/exp3b_label_temporal_ETTm1_720_seed2023/
```

---

## 8. Result Interpretation Guide

### After Results Are Available:

#### Compare Exp 3b vs Exp 3 (Isolate Temporal Effect):

```bash
# Exp 3 (Label Only): MSE = 1.124
# Exp 3b (Label + Temporal): MSE = ?

# If Exp 3b << Exp 3:
#   → Temporal embedding was critical
#   → Legendre labels work with temporal context

# If Exp 3b ≈ Exp 3:
#   → Temporal embedding not the issue
#   → Legendre labels fundamentally insufficient
```

#### Compare Exp 3b vs Vanilla (Isolate PE Type):

```bash
# Vanilla (Sinusoidal + Temporal): MSE = 0.519
# Exp 3b (Legendre + Temporal): MSE = ?

# If Exp 3b ≈ Vanilla:
#   → Legendre is a valid PE replacement
#   → Orthogonal distinctiveness sufficient

# If Exp 3b > Vanilla:
#   → Sinusoidal PE has advantages
#   → Smooth periodic structure matters
```

#### Compare Exp 3b vs Exp 5b (Isolate Ordering Effect):

```bash
# Exp 3b (Label + Temporal): MSE = ?
# Exp 5b (Label + Temporal + Ordering): MSE = ?

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

The results will definitively show whether orthogonal polynomial labels can serve as effective positional encodings when combined with temporal information.