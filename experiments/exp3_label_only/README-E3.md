# Experiment 3: Label Only (Legendre Polynomials)

## Objective
Test whether Legendre polynomial position embeddings (Label component) alone can provide meaningful positional information for time series forecasting.

## Hypothesis
Pure distinctiveness (orthogonal labeling) without ordering or distance decay.

## Implementation

### Mathematical Formulation
```
X'_i = X_i + P_i

where:
  X_i = value_embedding(x)      [Semantic content]
  P_i = Legendre(i)             [LABEL ONLY - Equation 1]
  
  NO temporal embedding
  NO distance operator
  NO ordering signal
```

**Note on `x_mark`:** The Informer model passes `x_mark_enc` to `DataEmbedding.forward()` as required by its interface, but `forward()` silently ignores it — `x_mark` is not used in any computation. The `self.temporal_embedding` and `self.position_embedding` objects are still instantiated in `__init__` (as dead weight, for interface compatibility) but are never called in `forward()`. They add unused parameters to the checkpoint.

### Label Component (Equation 1)
```
Legendre Polynomials: P_i = [L_0(x_i), L_1(x_i), ..., L_{d-1}(x_i)]

Orthogonality: ⟨P_n, P_m⟩ = { 1 if n=m, 0 if n≠m }
```

- Generated using scipy.special.legendre
- Positions normalized to [-1, 1]
- Scaled by 1/√d_model
- Pre-computed and cached as non-trainable buffer (`register_buffer`)

## Key Differences from Other Experiments

| Aspect | Exp 1 (D) | Exp 2 (L+O+D) | Exp 3 (L) |
|--------|-----------|---------------|-----------|
| **Label (L)** | ❌ | ✅ | ✅ ONLY |
| **Order (O)** | ❌ | ✅ | ❌ |
| **Distance (D)** | ✅ | ✅ | ❌ |
| **Temporal** | ✅ | ✅ | ❌ REMOVED |
| **Components** | α(i,j) bias | Full LOD | Pure Label |

## Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Dataset | ETTh1 | `--data ETTh1` |
| Model | Informer | `--model informer` |
| Attention | Full (not ProbSparse) | `--attn full` |
| Distillation | Enabled | `--distil` |
| Sequence Length | 96 | `--seq_len 96` |
| Label Length | 48 | `--label_len 48` |
| Prediction Length | 96, 192 (Phase 1) | `exp3_phase1.sh` loop |
| Encoder Layers | 2 | `--e_layers 2` |
| Decoder Layers | 1 | `--d_layers 1` |
| d_model | 512 | `--d_model 512` |
| n_heads | 8 | `--n_heads 8` |
| d_ff | 2048 | `--d_ff 2048` |
| dropout | 0.05 | `--dropout 0.05` |
| Embed type | timeF | `--embed timeF` (passed to model; temporal embedding suppressed in `forward()`) |
| Frequency | h | `--freq h` |
| Activation | gelu | `--activation gelu` |
| Learning rate | 0.0001 | `--learning_rate 0.0001` |
| Batch size | 32 | `--batch_size 32` |
| Train epochs | 6 | `--train_epochs 6` |
| Patience (early stop) | 3 | `--patience 3` |
| Factor | 5 | `--factor 5` |
| enc_in / dec_in / c_out | 7 | ETTh1 multivariate (7 features) |
| Loss | mse | `--loss mse` |
| LR adjustment | type1 | `--lradj type1` |
| Mix | Enabled | `--mix` |
| Padding | 0 | `--padding 0` |
| num_workers | 0 | `--num_workers 0` |
| Target | OT | `--target OT` |
| Legendre Scaling | 1/√d_model | `scaling=True` hardcoded in `embed.py` |

## File Structure
```
experiments/exp3_label_only/
├── models/
│   ├── legendre_embedding.py   - Label component (Legendre polynomial embedding)
│   ├── embed.py                - MODIFIED: Label only (value + Legendre, no temporal)
│   ├── attn.py                 - Vanilla FullAttention (unchanged)
│   └── ... (other vanilla files)
├── README-E3.md                 - This file
└── exp3_phase1.sh               - Phase 1 training script
```

## Execution Protocol

This experiment uses a **single-phase protocol** (Phase 1 only):

### Phase 1 — Feasibility Check (`exp3_phase1.sh`)

- **Goal:** Gauge whether Legendre-only embedding provides a useful positional signal before committing to multi-seed runs.
- **pred_len:** {96, 192}
- **Seed:** 2021 (single — exploration only)
- **Total runs:** 2 (1 config × 1 seed × 2 pred_len)
- **Decision rule:**
  - Exp3 wins at BOTH pred_lens vs best previous experiment → Proceed to Phase 2
  - Exp3 wins at ONE pred_len (mixed) → Still proceed to Phase 2, note instability
  - Exp3 loses at BOTH pred_lens clearly → Document as negative result, skip Phase 2

**Phase 2 status:** Phase 2 was **not run** for this experiment. `mse_mae_scores_sorted.txt` explicitly marks "⚠ Phase 2 — NOT FOUND in notebook outputs." Based on the Phase 1 results (see Results section), Exp3 lost at both pred_lens vs the best previous experiment (Exp1-Pre, α=1.0), so Phase 2 was skipped. The multi-seed Label-only follow-on was carried out as a separate experiment (**Exp3b**).

### How to Run

```bash
# Phase 1 only
bash experiments/exp3_label_only/exp3_phase1.sh
```

## Expected Output

The script will:
1. Copy modified model files to `Informer2020-original/models/`
2. Train the model for up to 6 epochs (with early stopping, patience=3)
3. Test on the test set
4. Print final metrics: MSE, MAE
5. Save per-run logs to `logs/exp3_phase1/<RUN_ID>.log`
6. Save per-run results to `results/exp3_ph1_ETTh1_legendre_pred<L>_seed2021/`
7. Save master log to `logs/exp3_phase1/master_run.log`

## Results

Source: `mse_mae_scores_sorted.txt`

### Phase 1 (pred_len ∈ {96, 192}, seed=2021)

| pred_len | Seed | MSE | MAE |
|---------|------|-----|-----|
| 96 | 2021 | 1.0989 | 0.8541 |
| 192 | 2021 | 1.4844 | 0.9973 |

**Phase 2:** Not run. See Exp3b for the multi-seed Label-only results.

## Comparison with Baselines

Source: `mse_mae_scores_sorted.txt`. Values shown are Phase 2 averages (3 seeds) where available; Phase 1 single-seed where Phase 2 was not run.

| Experiment | Components | pred_len=96 MSE | pred_len=192 MSE | Notes |
|------------|-----------|-----------------|------------------|-------|
| Vanilla | Standard PE | TODO: Information could not be verified from the repository. | | Baseline directories empty |
| Exp 1-Pre | D (pre-softmax) | 0.8670 | 0.9373 | Phase 2 avg, α=1.0 |
| Exp 2 (LOD) | L+O+D | 0.8534 | 0.9625 | Phase 2 avg |
| **Exp 3 (L)** | **L only** | **1.0989** | **1.4844** | Phase 1 only, seed=2021 |
| Exp 3b (L variant) | L only (clean run) | 0.8858 | 0.8913 | Phase 2 avg, 3 seeds |

**Note:** Vanilla baseline values are not present in `mse_mae_scores_sorted.txt`. The `results/baseline_ph1_ETTh1_pred96_seed2021/` and `results/baseline_ph1_ETTh1_pred192_seed2021/` directories exist but are empty.

## Analysis Questions

1. Does orthogonal distinctiveness provide any benefit?
2. Is Label component meaningful without Order/Distance?
3. How does pure Label compare to pure Distance (Exp1)?
4. Does removing temporal embedding hurt more than adding Label helps?

## Theoretical Justification

**Legendre Polynomials Provide:**
- ✅ Distinctiveness (orthogonality)
- ✅ Complete basis (spans function space)
- ❌ NO ordering information
- ❌ NO distance decay
- ❌ NO temporal semantics

**This tests:** Can position be encoded through distinctiveness alone?
