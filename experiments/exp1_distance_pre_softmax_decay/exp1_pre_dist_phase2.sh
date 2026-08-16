#!/bin/bash
# =============================================================================
# Exp1-Pre — Phase 2: Distance-Based Attention (Pre-Softmax Decay)
#
# Experiment name  : exp1_distance_pre_softmax_decay
# Mechanism        : D only — distance decay applied BEFORE softmax
#                    score(i,j) = QK^T / sqrt(d) * 1/(1 + |i-j|^alpha)
#
# Phase 2 goal
# ------------
#   Validate best alpha from Phase 1 (alpha=1.0) across all standard
#   ETTh1 pred_lens and 3 seeds for statistical robustness.
#
#   Phase 1 outcome:
#     alpha=1.0  won at both pred_len=96 (MSE=0.8683) and pred_len=192 (MSE=0.8463)
#     alpha=2.0  degraded at pred_len=192 (MSE jumped 0.878 → 0.957)
#     alpha=0.5  worst at both horizons
#
#   Alpha:        1.0 (fixed — best from Phase 1)
#   Dataset:      ETTh1 (multivariate, 7 features)
#   Seeds:        2021, 2022, 2023
#   Pred lengths: 48, 96, 192, 336
#   Total runs:   12  (1 alpha × 3 seeds × 4 pred_len)
#
# Final deliverable
# -----------------
#   Per-run MSE/MAE table + average MSE/MAE per pred_len across the 3 seeds.
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp1_distance_pre_softmax_decay"
LOG_DIR="$PROJECT_ROOT/logs/exp1_pre_phase2"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================="
echo "Exp1-Pre  |  PHASE 2 — VALIDATION"
echo "Components:   D only (alpha before softmax)"
echo "Alpha:        1.0 (fixed — best from Phase 1)"
echo "Dataset:      ETTh1"
echo "Seeds:        2021, 2022, 2023"
echo "Pred lengths: 48, 96, 192, 336"
echo "Total runs:   12 (1 alpha × 3 seeds × 4 pred)"
echo "Phase 1 note: alpha=2.0 degraded at pred=192"
echo "              alpha=0.5 worst at both horizons"
echo "Start time:   $(date)"
echo "============================================="
} | tee "$MASTER_LOG"

# --- FILE COPY ---
cp "$EXP_DIR/models/__init__.py" "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"    "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"   "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py" "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py" "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"   "$INFORMER_DIR/models/model.py"
echo "Files copied." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# --- FIXED ALPHA ---
ALPHA="1.0"

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

for seed in 2021 2022 2023; do
    for pred_len in 48 96 192 336; do

        TOTAL=$((TOTAL + 1))
        RUN_ID="exp1pre_ph2_ETTh1_alpha${ALPHA}_pred${pred_len}_seed${seed}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "" | tee -a "$MASTER_LOG"
        echo "---------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN:        $RUN_ID"      | tee -a "$MASTER_LOG"
        echo "Alpha:      $ALPHA (fixed)" | tee -a "$MASTER_LOG"
        echo "Seed:       $seed"        | tee -a "$MASTER_LOG"
        echo "Pred len:   $pred_len"    | tee -a "$MASTER_LOG"
        echo "Start:      $(date)"      | tee -a "$MASTER_LOG"
        echo "---------------------------------------------" | tee -a "$MASTER_LOG"

        if grep -q "STATUS: COMPLETED" "$RUN_LOG" 2>/dev/null; then
            echo "SKIPPING — already completed." | tee -a "$MASTER_LOG"
            SKIPPED=$((SKIPPED + 1))
            COMPLETED=$((COMPLETED + 1))
            continue
        fi

        cd "$INFORMER_DIR" || { echo "ERROR: cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

        if python -u main_informer.py \
            --model      informer \
            --data       ETTh1 \
            --root_path  ./data/ETT/ \
            --data_path  ETTh1.csv \
            --features   M \
            --target     OT \
            --freq       h \
            --checkpoints ./checkpoints/ \
            --seq_len    96 \
            --label_len  48 \
            --pred_len   "$pred_len" \
            --enc_in     7 \
            --dec_in     7 \
            --c_out      7 \
            --d_model    512 \
            --n_heads    8 \
            --e_layers   2 \
            --d_layers   1 \
            --s_layers   "3,2,1" \
            --d_ff       2048 \
            --factor     5 \
            --padding    0 \
            --dropout    0.05 \
            --attn       full \
            --embed      timeF \
            --activation gelu \
            --decay_a    "$ALPHA" \
            --itr        1 \
            --train_epochs 6 \
            --batch_size 32 \
            --patience   3 \
            --learning_rate 0.0001 \
            --des        "$RUN_ID" \
            --loss       mse \
            --lradj      type1 \
            --num_workers 0 \
            2>&1 | tee "$RUN_LOG"; then

            COMPLETED=$((COMPLETED + 1))
            echo "STATUS: COMPLETED" | tee -a "$RUN_LOG"
            echo "STATUS: COMPLETED" | tee -a "$MASTER_LOG"
        else
            FAILED=$((FAILED + 1))
            echo "STATUS: FAILED — check $RUN_LOG" | tee -a "$MASTER_LOG"
        fi

        echo "End: $(date)" | tee -a "$MASTER_LOG"

    done
done

# --- SUMMARY ---
echo "" | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"
echo "PHASE 2 COMPLETE — RESULTS SUMMARY (alpha=${ALPHA})" | tee -a "$MASTER_LOG"
echo "Total: $TOTAL | Done: $COMPLETED | Skipped: $SKIPPED | Failed: $FAILED" | tee -a "$MASTER_LOG"
echo "End time: $(date)"                            | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"
echo ""                                             | tee -a "$MASTER_LOG"

# --- PER-RUN TABLE ---
echo "Per-run MSE/MAE (alpha=${ALPHA}):"            | tee -a "$MASTER_LOG"
echo "Seed  | PredLen | MSE      | MAE"             | tee -a "$MASTER_LOG"
echo "------|---------|----------|----------"       | tee -a "$MASTER_LOG"

for seed in 2021 2022 2023; do
    for pred_len in 48 96 192 336; do
        RUN_ID="exp1pre_ph2_ETTh1_alpha${ALPHA}_pred${pred_len}_seed${seed}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"

        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep -i "mse:" "$RUN_LOG" | tail -1 | grep -oP 'mse:\K[0-9]+\.[0-9]+')
            MAE=$(grep -i "mae:" "$RUN_LOG" | tail -1 | grep -oP 'mae:\K[0-9]+\.[0-9]+')
            MSE=${MSE:-"N/A"}
            MAE=${MAE:-"N/A"}
        else
            MSE="NO LOG"; MAE="NO LOG"
        fi

        printf "%-5s | %-7s | %-24s | %s\n" "$seed" "$pred_len" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
    done
done

# --- AVERAGED TABLE (python inline) ---
echo "" | tee -a "$MASTER_LOG"
echo "Averaged MSE/MAE across 3 seeds per pred_len:" | tee -a "$MASTER_LOG"

python3 - "$LOG_DIR" "$ALPHA" <<'PYEOF' 2>/dev/null | tee -a "$MASTER_LOG"
import sys, re, os

log_dir = sys.argv[1]
alpha   = sys.argv[2]
seeds   = [2021, 2022, 2023]
pred_lens = [48, 96, 192, 336]

print(f"{'PredLen':<8} | {'Avg MSE':<12} | {'Avg MAE':<12} | Notes")
print(f"{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-------")

for pl in pred_lens:
    mses, maes = [], []
    for s in seeds:
        run_id  = f"exp1pre_ph2_ETTh1_alpha{alpha}_pred{pl}_seed{s}"
        logfile = os.path.join(log_dir, f"{run_id}.log")
        if not os.path.isfile(logfile):
            continue
        with open(logfile) as f:
            lines = f.read()
        m = re.findall(r'mse:([0-9]+\.[0-9]+)', lines, re.IGNORECASE)
        a = re.findall(r'mae:([0-9]+\.[0-9]+)', lines, re.IGNORECASE)
        if m: mses.append(float(m[-1]))
        if a: maes.append(float(a[-1]))
    if mses:
        avg_mse = sum(mses) / len(mses)
        avg_mae = sum(maes) / len(maes) if maes else float('nan')
        note = "⚠️  Unstable" if (max(mses) - min(mses)) > 0.15 else "✅ Stable"
        print(f"{pl:<8} | {avg_mse:<12.4f} | {avg_mae:<12.4f} | {note}")
    else:
        print(f"{pl:<8} | {'N/A':<12} | {'N/A':<12} | no logs found")
PYEOF

echo "" | tee -a "$MASTER_LOG"
echo "NEXT STEP:"                                                                              | tee -a "$MASTER_LOG"
echo "  Average MSE across 3 seeds per pred_len to get your final result table"               | tee -a "$MASTER_LOG"
echo "  Compare against baseline Informer (no distance decay)"                                | tee -a "$MASTER_LOG"
echo "  This is your Exp1-Pre final result for alpha=1.0"                                     | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"
