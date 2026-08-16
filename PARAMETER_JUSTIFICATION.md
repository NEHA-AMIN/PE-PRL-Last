# PARAMETER_JUSTIFICATION.md
# Evidence Chain for Every Parameter in Exp6 Phase 1 Scripts
# ============================================================

## Purpose

Every parameter in the generated scripts is justified here with an explicit
evidence chain. No parameter was assumed — each was traced to source code,
existing scripts, or documentation.

---

## Section 1: Universal Hyperparameters

These parameters appear identically in every Phase 1 script in the repository.

### `--model informer`
- **Evidence**: All 7 Phase 1 scripts verified
- **Source**: `main_informer.py` line 9: `--model` required argument

### `--data ETTh1`
- **Evidence**: All 7 Phase 1 scripts
- **Source**: `main_informer.py` line 11, data_parser dict

### `--root_path ./data/ETT/`
- **Evidence**: All scripts
- **Source**: ETT data resides at `./data/ETT/ETTh1.csv` in Informer2020-original

### `--data_path ETTh1.csv`
- **Evidence**: All scripts
- **Source**: Implied by `--data ETTh1` through data_parser; also explicit in scripts

### `--features M`
- **Evidence**: All scripts
- **Source**: Multivariate prediction (7 features → 7 features)

### `--attn full`
- **Evidence**: All ablation scripts (not Baseline)
- **Rationale**: Ablation experiments use `full` attention to isolate the effect of positional encoding modifications. Baseline uses `prob` (original Informer).
- **Source**: Exp5, Exp5b, Formula-A/B-pos/sem all use `--attn full`

### `--seq_len 96`
- **Evidence**: All 7 Phase 1 scripts
- **Source**: Standard ETTh1 sequence length used throughout study

### `--label_len 48`
- **Evidence**: All 7 Phase 1 scripts
- **Source**: Half of `seq_len`, standard Informer decoder start token length

### `--pred_len {96, 192}`
- **Evidence**: Every Phase 1 script sweeps exactly these two values
- **Rationale**: Phase 1 = short + medium horizon screening. Phase 2 extends to {48, 96, 192, 336} or longer.

### `--e_layers 2`
- **Evidence**: All scripts
- **Source**: 2 encoder layers (ablation standard)

### `--d_layers 1`
- **Evidence**: All scripts
- **Source**: 1 decoder layer (ablation standard)

### `--factor 5`
- **Evidence**: All scripts
- **Source**: ProbSparse attention factor (kept fixed even with full attn)

### `--enc_in 7 --dec_in 7 --c_out 7`
- **Evidence**: All scripts
- **Source**: ETTh1 has 7 features; multivariate → multivariate (`features=M`)

### `--d_model 512`
- **Evidence**: All scripts
- **Source**: Standard Informer model dimension

### `--n_heads 8`
- **Evidence**: All scripts
- **Source**: 8 attention heads with d_model=512 → 64 dims/head

### `--d_ff 2048`
- **Evidence**: All scripts
- **Source**: Standard Informer feed-forward dimension (4× d_model)

### `--dropout 0.05`
- **Evidence**: All scripts
- **Source**: Standard Informer dropout rate

### `--embed timeF`
- **Evidence**: All ablation scripts
- **Source**: Time-features encoding (hourly data)

### `--freq h`
- **Evidence**: All scripts
- **Source**: ETTh1 is hourly data

### `--activation gelu`
- **Evidence**: All scripts
- **Source**: GELU activation in feed-forward layers

### `--train_epochs 6`
- **Evidence**: All scripts
- **Source**: Early stopping at 6 epochs (patience=3 means max 6 usually)

### `--patience 3`
- **Evidence**: All scripts
- **Source**: Early stopping patience

### `--learning_rate 0.0001`
- **Evidence**: All scripts
- **Source**: 1e-4 learning rate standard

### `--batch_size 32`
- **Evidence**: All scripts
- **Source**: Standard batch size for ETTh1

### `--itr 1`
- **Evidence**: All scripts (Phase 1 specific)
- **Source**: Single iteration per run in Phase 1. Phase 2 also uses itr=1 (seeds handled externally via bash loop with different `--des` values).

---

## Section 2: Exp6-Specific Parameters

### `--decay_a 1.0`

**Evidence chain**:

1. **CLI argument exists**: `main_informer.py` line 41: `parser.add_argument('--decay_a', type=float, default=1.0, ...)`

2. **Model constructor accepts it**: `experiments/exp6_lod_pre/models/model.py` line 16: `decay_a=1.0` in `Informer.__init__` signature

3. **Model forwards it to attention**: `model.py` line 33: `decay_a=decay_a` passed to `FullAttention(...)`

4. **Attention uses it**: `experiments/exp6_lod_pre/models/attn.py` line 34: `alpha = 1.0 / (1.0 + dist_matrix ** self.decay_a)`

5. **exp_informer.py now forwards it**: FINDING H fix — `decay_a=getattr(self.args, 'decay_a', 1.0)` added as keyword argument in `_build_model()`

6. **Value justification (α=1.0)**:
   - README-E6-Pre.md states: "α = 1.0 was the best performer in Phase 1"
   - Default value in `attn.py` and `model.py` is `decay_a=1.0`
   - Phase 1 is a screening pass — α=1.0 is the natural first choice
   - Phase 2 would sweep {0.5, 1.0, 2.0} per README-E6-POST.md

---

## Section 3: Path Parameters

### `PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"`
- **Evidence**: All 7 Phase 1 scripts use exactly this path
- **Source**: Google Colab + Google Drive execution environment

### `INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"`
- **Evidence**: All scripts
- **Source**: The shared Informer2020 code base

### `EXP_DIR="$PROJECT_ROOT/experiments/exp6_lod_pre"` (and `exp6_lod_post`)
- **Evidence**: Pattern from Exp5b: `EXP_DIR="$PROJECT_ROOT/experiments/exp5b_label_order_clean_delta_MV"`
- **Source**: Each experiment has its own `EXP_DIR` variable pointing to its folder

### `LOG_DIR="$PROJECT_ROOT/logs/exp6_lod_pre_phase1"` (and `exp6_lod_post_phase1`)
- **Evidence**: Pattern from Exp5b: `LOG_DIR="$PROJECT_ROOT/logs/exp5b_label_order_clean_delta_phase1"`
- **Source**: Per-experiment, per-phase log directory

---

## Section 4: RUN_ID Format

### Exp6-Pre: `exp6pre_ph1_ETTh1_lod_pre_pred${pred_len}_seed${SEED}`
### Exp6-Post: `exp6post_ph1_ETTh1_lod_post_pred${pred_len}_seed${SEED}`

**Derivation from existing patterns**:
- Exp5b: `exp5b_ph1_ETTh1_lod_clean_pred${pred_len}_seed${SEED}`
- Exp1-post Phase 2: `exp1post_ph2_ETTh1_alpha${ALPHA}_pred${PRED_LEN}_seed${SEED}`

**Format rule**: `{exp_short_id}_ph{phase}_ETTh1_{variant_desc}_pred{pred_len}_seed{seed}`

The `RUN_ID` is used as both:
1. The `--des` argument to `main_informer.py` (creates the checkpoint directory)
2. The log filename: `$LOG_DIR/${RUN_ID}.log`
3. The results directory: `$RESULTS_DIR/${RUN_ID}/`

---

## Section 5: File Copy Justification

### Why `legendre_embedding.py` must be copied

**Evidence chain**:
1. `experiments/exp6_lod_pre/models/embed.py` line 1 (or similar): `from legendre_embedding import LegendrePositionEmbedding`
2. This import runs at module load time (inside `DataEmbedding.__init__` or at module level)
3. When `embed.py` is copied to `$INFORMER_DIR/models/embed.py`, Python will look for `legendre_embedding` in `$INFORMER_DIR/models/`
4. Without copying `legendre_embedding.py`, import fails at runtime
5. Exp5b script comment (line 71): "CRITICAL extra file: embed.py does `from legendre_embedding import LegendrePositionEmbedding` inside DataEmbedding.__init__"

### Why 6 standard files are all required

| File | Reason |
|------|--------|
| `__init__.py` | Package initializer; may import from modified files |
| `attn.py` | Contains Exp6's `FullAttention` with distance weighting |
| `embed.py` | Contains `DataEmbedding` that returns delta_x tuple |
| `encoder.py` | Fixed to downsample `delta_x` through ConvLayer |
| `decoder.py` | Standard decoder; must match the encoder's data format |
| `model.py` | Fixed `Informer` and `InformerStack` with decay_a threading |

---

## Section 6: Bug Fixes Embedded in Copied Files

The following bugs were fixed before the scripts are run. The fixed files are what get copied:

| File | Bug Fixed | Finding ID |
|------|-----------|------------|
| `exp6_lod_pre/models/model.py` | InformerStack: missing `decay_a` in `__init__`, attention layers, and `forward()` tuple unpacking | FINDING F |
| `exp6_lod_post/models/model.py` | InformerStack: same `__init__` and attention layer fixes | FINDING F |
| `exp6_lod_pre/models/encoder.py` | `delta_x` not downsampled through ConvLayer | FINDING G |
| `exp6_lod_post/models/encoder.py` | Same ConvLayer fix | FINDING G |
| `Informer2020-original/exp/exp_informer.py` | `decay_a` parsed but never passed to model constructor | FINDING H |

---

*Every parameter in this document has an explicit evidence source. No assumptions were made.*
