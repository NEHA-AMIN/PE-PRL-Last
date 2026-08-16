# PLANNED_RUN_CONFIGURATION.md
# Exact Planned Configuration for Exp6-Pre and Exp6-Post Phase 1
# ============================================================

## Overview

Both Phase 1 scripts follow the same configuration protocol:
- **Phase**: 1 (exploration / screening)
- **Goal**: Detect whether the LOD (distance-weighted) architecture improves over Exp5b
- **Total runs per script**: 2 (1 seed × 2 pred_lens)

---

## Exp6-Pre Phase 1 Configuration

### Experiment Identity
| Property | Value | Source |
|----------|-------|--------|
| Name | Exp6 LOD Pre-Softmax | README-E6-Pre.md |
| Variant | LOD Pre (distance weighting applied BEFORE softmax) | attn.py lines 20-40 |
| Phase | 1 | Protocol |
| Script path | `experiments/exp6_lod_pre/exp6_lod_pre_phase1.sh` | To create |

### Architecture
```
Input:  x [B, L, 7]
        ↓
embed.py → DataEmbedding
        → combined_emb = value_emb + temporal_emb + legendre_pos
        → delta_x = x_i - x_{i-1}  (via LegendrePositionEmbedding)
        → returns TUPLE: (combined_emb, delta_x)
        ↓
model.py Informer.forward() unpacks tuple
        → Q, K ← combined_emb  (via encoder projection)
        → V     ← delta_x       (via encoder projection)
        ↓
attn.py FullAttention.forward()
        → scores = Q·K^T / sqrt(d)
        → dist_matrix[i,j] = |i - j|
        → alpha[i,j] = 1 / (1 + dist_matrix[i,j] ^ decay_a)   ← PRE-SOFTMAX
        → scores = scores * alpha      ← multiplication BEFORE softmax
        → weights = softmax(scores)
        → output = weights · V
```

### Hyperparameters (Phase 1)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| `pred_len` | 96, 192 | Phase 1 universal pattern |
| `seed` | 2021 | Phase 1 universal pattern |
| `decay_a` (`alpha`) | 1.0 | README-E6-Pre.md: "α=1.0 was best in Phase 1"; attn.py default=1.0 |
| `seq_len` | 96 | All scripts |
| `label_len` | 48 | All scripts |
| `e_layers` | 2 | All scripts |
| `d_layers` | 1 | All scripts |
| `d_model` | 512 | All scripts |
| `n_heads` | 8 | All scripts |
| `d_ff` | 2048 | All scripts |
| `dropout` | 0.05 | All scripts |
| `batch_size` | 32 | All scripts |
| `learning_rate` | 0.0001 | All scripts |
| `train_epochs` | 6 | All scripts |
| `patience` | 3 | All scripts |
| `attn` | full | All ablation scripts |
| `embed` | timeF | All ablation scripts |
| `itr` | 1 | All scripts |

### Files Copied
```
__init__.py
attn.py          ← pre-softmax distance weighting
embed.py         ← returns (combined_emb, delta_x) tuple
encoder.py       ← fixed: ConvLayer applied to delta_x too
decoder.py
model.py         ← fixed: InformerStack now correct
legendre_embedding.py   ← CRITICAL: imported by embed.py at runtime
```

### Run Matrix
| Run | pred_len | seed | decay_a | RUN_ID |
|-----|----------|------|---------|--------|
| 1 | 96 | 2021 | 1.0 | `exp6pre_ph1_ETTh1_lod_pre_pred96_seed2021` |
| 2 | 192 | 2021 | 1.0 | `exp6pre_ph1_ETTh1_lod_pre_pred192_seed2021` |

### Log Paths
```
$PROJECT_ROOT/logs/exp6_lod_pre_phase1/master_run.log
$PROJECT_ROOT/logs/exp6_lod_pre_phase1/exp6pre_ph1_ETTh1_lod_pre_pred96_seed2021.log
$PROJECT_ROOT/logs/exp6_lod_pre_phase1/exp6pre_ph1_ETTh1_lod_pre_pred192_seed2021.log
```

---

## Exp6-Post Phase 1 Configuration

### Experiment Identity
| Property | Value | Source |
|----------|-------|--------|
| Name | Exp6 LOD Post-Softmax | README-E6-POST.md |
| Variant | LOD Post (distance weighting applied AFTER softmax) | attn.py lines 20-50 |
| Phase | 1 | Protocol |
| Script path | `experiments/exp6_lod_post/exp6_lod_post_phase1.sh` | To create |

### Architecture
```
Input:  x [B, L, 7]
        ↓
embed.py → DataEmbedding (same as Pre)
        → returns TUPLE: (combined_emb, delta_x)
        ↓
attn.py FullAttention.forward()
        → scores = Q·K^T / sqrt(d)
        → weights = softmax(scores)        ← softmax FIRST
        → dist_matrix[i,j] = |i - j|
        → alpha[i,j] = 1 / (1 + dist_matrix[i,j] ^ decay_a)
        → weights = weights * alpha        ← multiplication AFTER softmax
        → output = weights · V
```

### Hyperparameters (Phase 1)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| `pred_len` | 96, 192 | Phase 1 universal pattern |
| `seed` | 2021 | Phase 1 universal pattern |
| `decay_a` (`alpha`) | 1.0 | README-E6-POST.md states best α; attn.py default=1.0 |
| (all others) | Same as Exp6-Pre | Phase 1 universal parameters |

### Files Copied
```
__init__.py
attn.py          ← post-softmax distance weighting
embed.py         ← returns (combined_emb, delta_x) tuple; no dropout on delta_x
encoder.py       ← fixed: ConvLayer applied to delta_x too
decoder.py
model.py         ← fixed: InformerStack now correct
legendre_embedding.py   ← CRITICAL: imported by embed.py at runtime
```

### Run Matrix
| Run | pred_len | seed | decay_a | RUN_ID |
|-----|----------|------|---------|--------|
| 1 | 96 | 2021 | 1.0 | `exp6post_ph1_ETTh1_lod_post_pred96_seed2021` |
| 2 | 192 | 2021 | 1.0 | `exp6post_ph1_ETTh1_lod_post_pred192_seed2021` |

### Log Paths
```
$PROJECT_ROOT/logs/exp6_lod_post_phase1/master_run.log
$PROJECT_ROOT/logs/exp6_lod_post_phase1/exp6post_ph1_ETTh1_lod_post_pred96_seed2021.log
$PROJECT_ROOT/logs/exp6_lod_post_phase1/exp6post_ph1_ETTh1_lod_post_pred192_seed2021.log
```

---

## Pre vs Post Softmax Comparison

| Property | Exp6-Pre | Exp6-Post |
|----------|----------|-----------|
| Distance application | Before softmax | After softmax |
| Formula | `scores = scores * alpha; weights = softmax(scores)` | `weights = softmax(scores); weights = weights * alpha` |
| embed.py delta dropout | Yes (`dropout(delta_x)`) | No (raw `delta_x`) |
| attn.py verified | Lines 20-40 | Lines 20-50 |
| Expected behavior | Stronger distance suppression (exponential effect through softmax) | Softer suppression (linear rescaling post-softmax) |

---

*All configuration values derived from source code reading. No assumptions made.*
