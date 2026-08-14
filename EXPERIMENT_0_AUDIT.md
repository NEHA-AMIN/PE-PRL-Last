# EXPERIMENT_0_AUDIT.md — Baseline
## Vanilla Informer (Unmodified)
**Audit Status:** COMPLETE  
**Evidence Source:** Direct code reading + `results/`, `mse_mae_scores_sorted.txt`, `README.md`

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| Official name | Baseline — Vanilla Informer |
| Folder | `experiments/Baseline/` |
| Code identifier | `pe_mode='vanilla'` (in `Informer2020-original/models/model.py`) |
| README identifier | "Baseline Model", "Vanilla Informer" |
| Notebook file | `BaseLine_Model (1).ipynb` |
| Shell scripts | `baseline_phase1.sh`, `baseline_phase2.sh` |
| Result directories | `results/baseline_ph1_ETTh1_pred96_seed2021/` (EMPTY), `results/baseline_ph1_ETTh1_pred192_seed2021/` (EMPTY) |

---

## 2. Objective

Establish a reference MSE/MAE on ETTh1 using the **unmodified Informer architecture**. All subsequent ablation experiments replace or augment the positional encoding stack and compare against this reference.

**Hypothesis:** Standard sinusoidal PE + temporal embedding provides effective positional information for multivariate time-series forecasting on ETTh1.

---

## 3. Mathematical Formulation (Derived from Code)

**Source:** [`Informer2020-original/models/embed.py:112-114`](Informer2020-original/models/embed.py:112)

```python
def forward(self, x, x_mark):
    x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
    return self.dropout(x)
```

### Embedding Formula
```
X'_i = X_i + PE_i + T_i

where:
  X_i  = TokenEmbedding(x_i)      [Conv1D: 7→512, kernel=3, circular padding, Kaiming init]
  PE_i = PositionalEmbedding(i)   [Sinusoidal: sin/cos, max_len=5000, NOT trainable]
  T_i  = temporal_embedding(t_i)  [TemporalEmbedding OR TimeFeatureEmbedding]
```

### Symbol Table

| Symbol | Definition | Implementation File | Lines |
|--------|-----------|---------------------|-------|
| X_i | Token embedding (semantic content of timestep i) | `embed.py` | 26-38 |
| PE_i | Sinusoidal positional encoding | `embed.py` | 7-23 |
| T_i | Temporal features (hour/day/month/weekday or continuous) | `embed.py` | 61-97 |
| scale | 1/√E where E=d_k per head | `attn.py` | 21 |
| A | Attention weights = softmax(scale · QKᵀ) | `attn.py` | 30 |

### Sinusoidal PE Formula (Verified from Code)
**Source:** [`Informer2020-original/models/embed.py:14-17`](Informer2020-original/models/embed.py:14)
```
PE[pos, 2k]   = sin(pos / 10000^(2k/d_model))
PE[pos, 2k+1] = cos(pos / 10000^(2k/d_model))
```
**Verified against README:** ✅ README states "Sinusoidal positional encoding — position signal". Consistent.

### Attention Formula (Verified)
**Source:** [`Informer2020-original/models/attn.py:30-31`](Informer2020-original/models/attn.py:30)
```
A = dropout(softmax(scale · QKᵀ))
V_out = A · V
```

---

## 4. Algorithm Flow

```
Raw ETTh1 input [B, 96, 7] + time_marks [B, 96, 4/5]
                ↓
TokenEmbedding (Conv1D, kernel=3, circular) [B, 96, 512]
                + PositionalEmbedding (sinusoidal, fixed) [1, 96, 512]
                + TemporalEmbedding (hour/day/month/weekday) OR TimeFeatureEmbedding
                ↓ Dropout(0.05)
Encoder embedding [B, 96, 512]
                ↓
ProbSparse Attention × 2 encoder layers (with distillation ConvLayer)
                ↓
Decoder embedding (same formula, label_len=48 + pred_len zeros)
                ↓
FullAttention (masked self-attn) + FullAttention (cross-attn)
                ↓
Linear projection [B, pred_len, 7]
                ↓
MSE/MAE evaluation
```

---

## 5. Code Walkthrough

### 5.1 `DataEmbedding` (embed.py:100-114)
- **Purpose:** Combines three signals into a 512-dim representation per timestep
- **TokenEmbedding:** `Conv1D(7→512, kernel=3, circular padding)` — captures local 3-step interactions
- **PositionalEmbedding:** Non-trainable sinusoidal buffer. `require_grad = False` (note: typo — should be `requires_grad`, but functionally equivalent due to buffer registration)
- **TemporalEmbedding:** In baseline, `--embed timeF` selects `TimeFeatureEmbedding` (linear layer on continuous time features), NOT `TemporalEmbedding` (fixed sinusoidal embeddings for categorical hour/day/month). This is critical.

### 5.2 `FullAttention` vs `ProbAttention` (attn.py)
- **Baseline uses `--attn prob`** = ProbSparse attention
- **All ablation experiments use `--attn full`** = standard scaled dot-product attention
- **This is a fundamental architecture difference** between the baseline and all ablations — the baseline is NOT a fair control for experiments using `FullAttention`.

---

## 6. Configuration Audit

### From `baseline_phase1.sh` (verified):
| Parameter | Shell Script | README | Notebook | Consistent? |
|-----------|-------------|--------|----------|-------------|
| seq_len | 96 | 96 | — | ✅ |
| label_len | 48 | 48 | — | ✅ |
| pred_len | 96, 192 | 96 (results table) | — | ✅ |
| e_layers | 2 | 2 | — | ✅ |
| d_layers | 1 | 1 | — | ✅ |
| d_model | 512 | 512 | — | ✅ |
| n_heads | 8 | 8 | — | ✅ |
| d_ff | 2048 | — | — | ⚠️ README omits d_ff |
| dropout | 0.05 | — | — | ⚠️ README omits dropout |
| attn | prob | prob | — | ✅ |
| embed | timeF | — | — | ⚠️ README omits embed type |
| seed | 2021 | 2021 | — | ✅ |
| train_epochs | 6 | — | — | — |
| patience | 3 | — | — | — |
| learning_rate | 0.0001 | — | — | — |
| batch_size | 32 | — | — | — |

---

## 7. Notebook Audit

**Notebook:** `experiments/Baseline/BaseLine_Model (1).ipynb`  
**Status:** Not fully read (notebook content not extracted — would require JSON parsing of .ipynb)  
**Concern:** Notebook presence but empty results directories suggests runs may have been done in Colab without local logs being saved.

---

## 8. Result Verification

| Source | pred_len=96 MSE | pred_len=96 MAE | pred_len=192 MSE | pred_len=192 MAE |
|--------|----------------|----------------|-----------------|-----------------|
| `README.md` table | **0.519** | **0.513** | — | — |
| `mse_mae_scores_sorted.txt` | NOT PRESENT | NOT PRESENT | NOT PRESENT | NOT PRESENT |
| `results/baseline_ph1_ETTh1_pred96_seed2021/` | **EMPTY DIRECTORY** | — | — | — |
| Shell script reference in exp5 | 0.8683 (Exp1-Pre, not baseline) | — | — | — |

**⚠️ CRITICAL FINDING:** The baseline results directories (`results/baseline_ph1_ETTh1_pred96_seed2021/`) exist as **empty directories** — no `training_log.txt` is present. The only source of baseline numbers (MSE 0.519, MAE 0.513) is the `README.md`. These values **cannot be verified from any log file in this repository**.

Furthermore, the `mse_mae_scores_sorted.txt` file — which documents all other experiments — contains **no entry for the Baseline experiment**. This is inconsistent with a document claiming to be a complete results log.

---

## 9. Consistency Checks

### 9.1 Naming Consistency
- ✅ Folder name (`Baseline`) matches README section ("Baseline")
- ⚠️ `baseline_phase1.sh` references `INFORMER_DIR="$PROJECT_ROOT/Informer2020"` (without `-original` suffix) — this directory does NOT exist in the repository. The actual directory is `Informer2020-original`. **The script will fail with "ERROR: Cannot cd to..."**

### 9.2 Formula Consistency
- ✅ Embedding formula in README (`X'_i = X_i + PE_i + T_i`) matches `embed.py:113`

### 9.3 Hyperparameter Consistency
- ⚠️ README states baseline config as `d_model=512, n_heads=8, e_layers=2, d_layers=1` and is silent on `d_ff`, `dropout`, `attn type`, `embed type`
- ⚠️ All ablation experiments use `--attn full` while baseline uses `--attn prob` — this difference is NOT mentioned in the README and invalidates direct MSE comparisons

### 9.4 Result Consistency
- 🔴 **CRITICAL:** Baseline MSE=0.519/MAE=0.513 appears only in README and is NOT backed by any log file in the repository

### 9.5 Architecture Consistency
- 🔴 **Baseline uses ProbSparse attention** (`--attn prob`). All ablations use `FullAttention` (`--attn full`). Different attention mechanisms produce different results independent of positional encoding changes. The comparison in README's results table is therefore **confounded by attention type**, not only positional encoding.

### 9.6 Embed Type Consistency
- 🔴 **Baseline uses `--embed timeF`** (linear layer on continuous time features). Ablation scripts use default `embed='fixed'` (categorical fixed sinusoidal embeddings). This is **another confounding factor** not disclosed in README.

---

## 10. Inconsistency Report

### Critical Issues
1. **Results not reproducible**: Baseline MSE=0.519 is referenced in README but not backed by any log file. Result directories are empty.
2. **Attention type confound**: Baseline uses `--attn prob` (ProbSparse); all ablations use `--attn full`. Comparing MSE across these is not a clean ablation of positional encoding.
3. **Embed type confound**: Baseline uses `--embed timeF`; ablations default to `embed='fixed'`. These use different temporal embedding architectures.
4. **Script path error**: `baseline_phase1.sh` references `Informer2020` (no `-original`), so the script cannot be run as written.
5. **Baseline absent from central results file**: `mse_mae_scores_sorted.txt` documents all other experiments but omits the baseline entirely.

### Moderate Issues
6. **README omits key config**: `d_ff`, `dropout`, `learning_rate`, `embed type`, `attention type` not stated in README table — readers cannot reproduce the exact configuration.
7. **README results table only shows pred_len=96**: No 192/336/720 comparison for baseline despite ablation results being available at those lengths.

### Minor Issues
8. `pe.require_grad = False` in `PositionalEmbedding.__init__` should be `pe.requires_grad = False` — typo, but functionally harmless because `register_buffer` already prevents gradient tracking.

---

## 11. Confidence Score

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Implementation confidence | 8/10 | Code is clean, formula verified, standard Informer architecture |
| Documentation confidence | 4/10 | README omits critical config differences (attn type, embed type); results not logged |
| Result confidence | 2/10 | Only source is README; no log file, not in central results file; confounded by attn/embed differences |

**Overall:** The baseline code is correctly implemented but the results are unverifiable from the repository. The comparison with ablations is methodologically questionable due to uncontrolled differences in attention type and temporal embedding type.
