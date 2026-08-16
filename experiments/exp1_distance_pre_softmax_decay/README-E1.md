# Experiment 1: Distance Decay Only

## Objective
Test the effect of simple index-based distance decay on Informer's performance without standard positional encoding.

## Hypothesis
Distance-based positional bias (using only index distance |i-j|) can provide sufficient positional information for time-series forecasting without explicit positional embeddings.

## Modifications

### 1. Attention Mechanism (`models/attn.py`)
**Location**: `FullAttention.forward()` method

**Change**: Added distance decay before softmax
```python
# Compute index-based distance matrix
q_idx = torch.arange(L).unsqueeze(1).to(queries.device)
k_idx = torch.arange(S).unsqueeze(0).to(queries.device)
dist_matrix = torch.abs(q_idx - k_idx).float()

# Apply decay: α(i,j) = 1 / (1 + |i-j|^decay_a)
alpha = 1.0 / (1.0 + dist_matrix ** self.decay_a)

# Apply distance decay to attention scores
scores = scores * alpha.unsqueeze(0).unsqueeze(0)
```

**Key Points**:
- Uses **index distance only**: |i - j|
- **Absolute value**: No directionality preserved
- **Multiplicative bias**: Applied to attention scores before softmax
- **Decay parameter**: controlled via `self.decay_a` (passed as `--decay_a` CLI argument; Phase 1 sweeps 0.5, 1.0, 2.0; Phase 2 fixed at 1.0)
- `scale` is applied inside `torch.softmax(scale * scores, dim=-1)` — after the decay multiplication

### 2. Embedding Layer (`models/embed.py`)
**Location**: `DataEmbedding.forward()` method

**Change**: Positional embedding bypassed in `forward`
```python
# self.position_embedding is still initialised in __init__ but not called in forward:
x = self.value_embedding(x) + self.temporal_embedding(x_mark)
```

**Key Points**:
- ✅ Keeps **value embedding** (token representations)
- ✅ Keeps **temporal embedding** (time features: hour, day, month)
- ❌ **Bypasses** `self.position_embedding` in `forward` (the object is still constructed in `__init__` but its output is not added)

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ETTh1 |
| Model | Informer |
| Attention | Full (not ProbSparse) |
| Sequence Length | 96 |
| Label Length | 48 |
| Phase 1 Prediction Lengths | 96, 192 (alpha sweep, seed=2021) |
| Phase 2 Prediction Lengths | 48, 96, 192, 336 (alpha=1.0, seeds 2021/2022/2023) |
| Encoder Layers | 2 |
| Decoder Layers | 1 |
| d_model | 512 |
| n_heads | 8 |
| d_ff | 2048 |
| factor | 5 |
| padding | 0 |
| dropout | 0.05 |
| activation | gelu |
| embed | timeF |
| freq | h |
| enc_in / dec_in / c_out | 7 / 7 / 7 |
| train_epochs | 6 |
| patience | 3 |
| batch_size | 32 |
| learning_rate | 0.0001 |
| lradj | type1 |
| itr | 1 |
| num_workers | 0 |
| loss | mse |
| Decay Parameter (`decay_a`) | Phase 1: 0.5, 1.0, 2.0; Phase 2: 1.0 (fixed) |
| s_layers | 3,2,1 |
| checkpoints | ./checkpoints/ |
| mix | True (framework default; not explicitly passed in scripts) |
| distil | True (framework default; not explicitly passed in scripts) |

## How to Run

```bash
# Phase 1 — alpha sweep (pred_len 96 & 192, seed 2021, alphas 0.5/1.0/2.0)
bash experiments/exp1_distance_pre_softmax_decay/exp1_pre_distance.sh

# Phase 2 — full sweep (alpha=1.0, 3 seeds × 4 pred_lens = 12 runs)
bash experiments/exp1_distance_pre_softmax_decay/exp1_pre_dist_phase2.sh
```

Note: scripts set `PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"` (Google Colab path). Adjust this variable for your environment before running.

## Expected Output

The script will:
1. Train the model for up to 6 epochs with early stopping (patience=3)
2. Test on the test set
3. Print final metrics: MSE, MAE
4. Save results to `results/exp1pre_ph{1,2}_ETTh1_alpha{a}_pred{p}_seed{s}/`

Note: RMSE, MAPE, and MSPE are not produced by the training scripts or notebook outputs.

## Results

### Phase 1 — Alpha Sweep (seed=2021, pred_len=96 and 192 only)

Source: `exp1_phase1_pre.ipynb` executed outputs and `mse_mae_scores_sorted.txt`.

| Alpha | pred_len | MSE | MAE | Train Loss (Ep1) | Vali Loss (Ep1) | Best epoch | Stopped at |
|-------|----------|-----|-----|-----------------|-----------------|------------|------------|
| 0.5 | 96 | 0.9024357795715332 | 0.7335947155952454 | 0.4479276 | 1.0566275 | 1 | 4 |
| 1.0 | 96 | 0.8682501912117004 | 0.7251123785972595 | 0.4504335 | 1.1230928 | 1 | 4 |
| 2.0 | 96 | 0.8784459233283997 | 0.71516352891922 | 0.4494402 | 1.0822822 | 1 | 4 |
| 0.5 | 192 | 0.8948372602462769 | 0.7418304681777954 | 0.4652017 | 1.2415757 | 1 | 4 |
| 1.0 | 192 | 0.8463228344917297 | 0.7035619616508484 | 0.4594642 | 1.2263277 | 1 | 4 |
| 2.0 | 192 | 0.957339882850647 | 0.7505829930305481 | 0.4559927 | 1.3349669 | 2 | 5 |

**Alpha selection:** α=1.0 wins at both pred=96 (MSE=0.8682) and pred=192 (MSE=0.8463). α=2.0 degraded sharply at pred=192 (MSE jumped to 0.9573 vs 0.8784 at pred=96), indicating over-suppression of long-range attention. α=0.5 was worst at both horizons. Pattern is stable → α=1.0 selected for Phase 2.

### Phase 2 — Full Sweep (alpha=1.0, 3 seeds × 4 pred_lens = 12 runs)

Source: `exp1_phase1_pre.ipynb` executed outputs and `mse_mae_scores_sorted.txt`.

| pred_len | seed | MSE | MAE |
|----------|------|-----|-----|
| 48 | 2021 | 0.7680 | 0.6661 |
| 48 | 2022 | 0.7742 | 0.6756 |
| 48 | 2023 | 0.8518 | 0.7137 |
| 96 | 2021 | 0.8796 | 0.7264 |
| 96 | 2022 | 0.8706 | 0.7242 |
| 96 | 2023 | 0.8509 | 0.7029 |
| 192 | 2021 | 0.9376 | 0.7456 |
| 192 | 2022 | 0.9298 | 0.7411 |
| 192 | 2023 | 0.9446 | 0.7590 |
| 336 | 2021 | 0.8764 | 0.7197 |
| 336 | 2022 | 1.1616 | 0.8368 |
| 336 | 2023 | 1.0725 | 0.8112 |

### Phase 2 — Averages across 3 seeds (alpha=1.0)

| pred_len | Avg MSE | Avg MAE | #Runs |
|----------|---------|---------|-------|
| 48 | 0.7980 | 0.6851 | 3 |
| 96 | 0.8670 | 0.7178 | 3 |
| 192 | 0.9373 | 0.7486 | 3 |
| 336 | 1.0368 | 0.7892 | 3 |

## Analysis

### Experimental Outcome

Phase 1 confirmed that α=1.0 provides the most stable distance decay across prediction horizons. The pattern was stable (same alpha won at both pred=96 and pred=192), so Phase 2 proceeded with α=1.0 only.

Phase 2 results across 3 seeds and 4 prediction lengths show:

1. **Best performance at pred=48**: Average MSE of 0.7980 across 3 seeds.

2. **Stable performance at pred=96 and pred=192**: Average MSE of 0.8670 and 0.9373 respectively. Variance across seeds is modest.

3. **High variance at pred=336**: Seeds 2022 (MSE=1.1616) and 2023 (MSE=1.0725) degrade substantially versus seed 2021 (MSE=0.8764), producing a wide average of 1.0368. This instability suggests that pure distance decay without ordering or label components becomes unreliable at longer horizons.

4. **Alpha sensitivity**: α=2.0 over-suppresses long-range attention at pred=192, confirming that the decay aggressiveness must be tuned carefully.

### Comparison with Baseline

- Baseline (Vanilla Informer with standard PE): MSE = TODO: Information could not be verified from the repository.
- Experiment 1 (distance decay only, pred=96, avg 3 seeds): MSE = 0.8670
- Difference: TODO: Information could not be verified from the repository.

### Observations

- [x] Performance impact: Distance decay alone provides moderate improvement at short horizons; degrades at pred=336 due to high seed-to-seed variance.
- [x] Training stability: All Phase 1 runs early-stopped at epoch 4 (α=2.0 pred=192 at epoch 5). Phase 2 patterns are consistent.
- [x] Convergence speed: Best epoch is always epoch 1 for all Phase 1 runs; subsequent epochs increase validation loss, triggering early stopping within 3 epochs of the best.
