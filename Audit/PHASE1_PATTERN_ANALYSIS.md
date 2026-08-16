# PHASE1_PATTERN_ANALYSIS.md
# Forensic Reverse-Engineering of Phase 1 Script Pattern
# Evidence source: all Phase 1 scripts read directly
# ============================================================

## 1. Scripts Analysed

The following Phase 1 scripts were read in full and used as evidence:

| File | Read | Lines |
|------|------|-------|
| `experiments/Baseline/baseline_phase1.sh` | YES | ~190 |
| `experiments/exp5_label_order/exp5_label_order_ph1.sh` | YES | ~200 |
| `experiments/exp5b_label_order_clean_delta_MV/e5b_lab_ord_clean_delta_mv_ph1.sh` | YES | 206 |
| `experiments/Formula-A-pos/formula-A-pos-ph1.sh` | YES | ~200 |
| `experiments/Formula-B-pos/formula-B-pos-ph1.sh` | YES | ~200 |
| `experiments/Formula-A-sem/formula-A-sem-ph1.sh` | YES | ~200 |
| `experiments/Formula-B-sem/formula-B-sem-ph1.sh` | YES | ~200 |

Phase 2 scripts read for contrast and alpha handling:

| File | Read | Purpose |
|------|------|---------|
| `experiments/exp1_distance_post_softmax/exp1_post_phase2_alpha0.5.sh` | YES | Alpha sed pattern |
| `experiments/exp1_distance_pre_softmax/exp1_pre_phase1.sh` | YES (partial) | Phase 1 pre reference |

---

## 2. Universal Fixed Parameters (Phase 1 — verified across ALL scripts)

| Parameter | Value | Verified In |
|-----------|-------|-------------|
| `--model` | `informer` | All 7 scripts |
| `--data` | `ETTh1` | All 7 scripts |
| `--root_path` | `./data/ETT/` | All 7 scripts |
| `--data_path` | `ETTh1.csv` | All 7 scripts |
| `--features` | `M` | All 7 scripts |
| `--attn` | `full` | All ablation scripts (Baseline uses `prob`) |
| `--seq_len` | `96` | All 7 scripts |
| `--label_len` | `48` | All 7 scripts |
| `--e_layers` | `2` | All 7 scripts |
| `--d_layers` | `1` | All 7 scripts |
| `--factor` | `5` | All 7 scripts |
| `--enc_in` | `7` | All 7 scripts |
| `--dec_in` | `7` | All 7 scripts |
| `--c_out` | `7` | All 7 scripts |
| `--d_model` | `512` | All 7 scripts |
| `--n_heads` | `8` | All 7 scripts |
| `--d_ff` | `2048` | All 7 scripts |
| `--dropout` | `0.05` | All 7 scripts |
| `--embed` | `timeF` | All ablation scripts |
| `--freq` | `h` | All 7 scripts |
| `--activation` | `gelu` | All 7 scripts |
| `--train_epochs` | `6` | All 7 scripts |
| `--patience` | `3` | All 7 scripts |
| `--learning_rate` | `0.0001` | All 7 scripts |
| `--batch_size` | `32` | All 7 scripts |
| `--itr` | `1` | All 7 scripts |
| `PROJECT_ROOT` | `/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1` | All 7 scripts |
| `INFORMER_DIR` | `$PROJECT_ROOT/Informer2020-original` | All 7 scripts |

---

## 3. Phase 1 Sweep Parameters

### pred_len
- **Value set**: `{96, 192}`
- **Total pred_len values**: 2
- **Evidence**: Every Phase 1 script sweeps exactly these two values
- **Contrast**: Phase 2 sweeps {48, 96, 192, 336} or {48, 96, 192, 336, 720}

### seed
- **Value**: `2021` (single seed)
- **Evidence**: Every Phase 1 script uses only `SEED=2021`
- **Contrast**: Phase 2 scripts use `for SEED in 2021 2022 2023`

### Total runs per Phase 1 script
- **Formula**: 1 seed × 2 pred_lens = **2 runs**
- **Evidence**: All scripts print "Total runs: 2" in header

---

## 4. File Copy Pattern

### Standard 6 files (all experiments):
```
cp "$EXP_DIR/models/__init__.py"  "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"      "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"     "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"   "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"   "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"     "$INFORMER_DIR/models/model.py"
```

### Critical 7th file (all experiments using LegendrePositionEmbedding):
```
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"
```

**Evidence for 7th file requirement:**
- Exp5b script explicitly copies it with comment: "CRITICAL extra file: embed.py does `from legendre_embedding import LegendrePositionEmbedding` inside DataEmbedding.__init__"
- Formula-A/B-pos/sem scripts all copy it
- Exp6 `embed.py` verified to contain the same `from legendre_embedding import LegendrePositionEmbedding` import

### Restore after run:
All scripts call `git checkout ./models/` at the end to restore original files.

---

## 5. Resume Logic Pattern

**Pattern A** (used by Exp5, Exp5b, all Formula scripts — preferred):
```bash
if grep -q "STATUS: COMPLETED" "$RUN_LOG" 2>/dev/null; then
    echo "SKIPPING — already completed. Remove log to re-run."
    SKIPPED=$((SKIPPED + 1))
    COMPLETED=$((COMPLETED + 1))
    continue
fi
```

**Pattern B** (Baseline, Exp1-post scripts):
```bash
if grep -q "^mse:" "$RUN_LOG" 2>/dev/null; then
    echo "SKIPPING — already completed with valid results"
    ...
fi
```

Exp6 Phase 1 scripts will use Pattern A (matches Exp5b — the closest architecture template).

---

## 6. Completion Detection Pattern

After successful Python run, ALL Phase 1 scripts do:
```bash
COMPLETED=$((COMPLETED + 1))
echo "STATUS: COMPLETED" | tee -a "$RUN_LOG"
echo "STATUS: COMPLETED" | tee -a "$MASTER_LOG"
```

Failure path:
```bash
FAILED=$((FAILED + 1))
echo "STATUS: FAILED — check $RUN_LOG" | tee -a "$MASTER_LOG"
```

---

## 7. Summary Table Format (end of all Phase 1 scripts)

All Phase 1 scripts end with:
1. A summary counter section (Total / Completed / Failed / Skipped)
2. A results table parsed from logs using `grep -oP 'mse:\s*\K[0-9]+\.[0-9]+'`
3. A DECISION GUIDE section comparing results to reference benchmarks

Reference benchmark values used in decision guides:
- Exp1-Pre, alpha=1.0, seed=2021: `pred=96 MSE≈0.8683`, `pred=192 MSE≈0.8463`
- These are the standard "best previous result" comparators

---

## 8. Directory Structure Created by Scripts

```
$PROJECT_ROOT/
├── logs/
│   └── <exp_name>_phase1/
│       ├── master_run.log
│       └── <RUN_ID>.log   (one per run)
└── results/
    └── <RUN_ID>/          (one per run, created by mkdir -p)
```

---

## 9. Python Command Structure

```bash
if python -u main_informer.py \
    --model informer \
    ... (all params) ...
    --des "$RUN_ID" \
    2>&1 | tee "$RUN_LOG"; then
    COMPLETED=$((COMPLETED + 1))
    echo "STATUS: COMPLETED" | tee -a "$RUN_LOG"
    ...
fi
```

Note: some scripts use `python -u`, others `python3 -u`. The Exp5b/Formula pattern uses `python -u` (Colab environment).

---

*All data in this document derived from direct reading of source scripts. No assumptions made.*
