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

# Apply decay: α(i,j) = 1 / (1 + |i-j|^a)
a = 1.0
alpha = 1.0 / (1.0 + dist_matrix ** a)

# Multiply attention scores by decay
scores = scores * alpha.unsqueeze(0).unsqueeze(0)
```

**Key Points**:
- Uses **index distance only**: |i - j|
- **Absolute value**: No directionality preserved
- **Multiplicative bias**: Applied to attention scores before softmax
- **Decay parameter**: a = 1.0

### 2. Embedding Layer (`models/embed.py`)
**Location**: `DataEmbedding.forward()` method

**Change**: Removed positional embedding
```python
# Before: x = value_embedding + position_embedding + temporal_embedding
# After:  x = value_embedding + temporal_embedding
x = self.value_embedding(x) + self.temporal_embedding(x_mark)
```

**Key Points**:
- ✅ Keeps **value embedding** (token representations)
- ✅ Keeps **temporal embedding** (time features: hour, day, month)
- ❌ Removes **positional embedding** (sinusoidal position encoding)

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
| Decay Parameter (a) | 1.0 |

## How to Run

```bash
cd /Users/neha/Desktop/PRL-SHIVANSH/distance-abl-PRL
bash experiments/exp1_distance_decay/run_exp1.sh
```

## Expected Output

The script will:
1. Train the model for 6 epochs
2. Test on the test set
3. Print final metrics: MSE, MAE, RMSE, MAPE, MSPE
4. Save results to `results/exp1_distance_decay/`

## Results

*To be filled after running the experiment*

| Metric | Value |
|--------|-------|
| MSE    | - |
| MAE    | - |
| RMSE   | - |
| MAPE   | - |
| MSPE   | - |

## Analysis

*To be filled after running the experiment*

### Comparison with Baseline
- Baseline (with standard PE): MSE = ?
- Experiment 1 (distance decay only): MSE = ?
- Difference: ?

### Observations
- [ ] Performance impact
- [ ] Training stability
- [ ] Convergence speed
