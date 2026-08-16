#!/bin/bash
# =============================================================================
# Exp1-Pre — Phase 1: Distance-Based Attention (Pre-Softmax Decay)
#
# Experiment name  : exp1_distance_pre_softmax_decay
# Mechanism        : D only — distance decay applied BEFORE softmax
#                    score(i,j) = QK^T / sqrt(d) * 1/(1 + |i-j|^alpha)
# Controlled flag  : --decay_a  (alpha value for decay strength)
#
# Phase 1 goal
# ------------
#   Explore three alpha values {0.5, 1.0, 2.0} at pred_len ∈ {96, 192}
#   with a single seed (2021) to identify which alpha is most promising
#   before committing to full multi-seed runs.
#
#   Alpha values: 0.5, 1.0, 2.0
#   Dataset:      ETTh1 (multivariate, 7 features)
#   Seed:         2021 (single — exploration only)
#   Pred lengths: 96, 192 (trend detection only)
#   Total runs:   6  (3 alpha × 1 seed × 2 pred_len)
#
# Decision rule after Phase 1
# ---------------------------
#   Same alpha wins at BOTH pred_lens → pattern stable → Phase 2
#     (best alpha only, all pred_lens, 3 seeds)
#   Different alpha wins at each → pattern unstable → run full 45 runs
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp1_distance_pre_softmax_decay"
LOG_DIR="$PROJECT_ROOT/logs/exp1_pre_phase1"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================="
echo "Exp1-Pre  |  PHASE 1 — EXPLORATION"
echo "Components:   D only (alpha before softmax)"
echo "Alpha values: 0.5, 1.0, 2.0"
echo "Dataset:      ETTh1"
echo "Seed:         2021 (single — exploration only)"
echo "Pred lengths: 96, 192 (trend detection only)"
echo "Total runs:   6 (3 alpha × 1 seed × 2 pred)"
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

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

SEED=2021

for alpha in 0.5 1.0 2.0; do
    for pred_len in 96 192; do

        TOTAL=$((TOTAL + 1))
        RUN_ID="exp1pre_ph1_ETTh1_alpha${alpha}_pred${pred_len}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "" | tee -a "$MASTER_LOG"
        echo "---------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN:        $RUN_ID"    | tee -a "$MASTER_LOG"
        echo "Alpha:      $alpha"     | tee -a "$MASTER_LOG"
        echo "Pred len:   $pred_len"  | tee -a "$MASTER_LOG"
        echo "Start:      $(date)"    | tee -a "$MASTER_LOG"
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
            --decay_a    "$alpha" \
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
echo "PHASE 1 COMPLETE — PATTERN SUMMARY"         | tee -a "$MASTER_LOG"
echo "Total: $TOTAL | Done: $COMPLETED | Skipped: $SKIPPED | Failed: $FAILED" | tee -a "$MASTER_LOG"
echo "End time: $(date)"                           | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"
echo ""                                            | tee -a "$MASTER_LOG"

echo "MSE results extracted from logs:"           | tee -a "$MASTER_LOG"
echo "Alpha  | PredLen | MSE      | MAE"           | tee -a "$MASTER_LOG"
echo "-------|---------|----------|----------"     | tee -a "$MASTER_LOG"

for alpha in 0.5 1.0 2.0; do
    for pred_len in 96 192; do
        RUN_ID="exp1pre_ph1_ETTh1_alpha${alpha}_pred${pred_len}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"

        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep -i "mse:" "$RUN_LOG" | tail -1 | grep -oP 'mse:\K[0-9]+\.[0-9]+')
            MAE=$(grep -i "mae:" "$RUN_LOG" | tail -1 | grep -oP 'mae:\K[0-9]+\.[0-9]+')
            MSE=${MSE:-"N/A"}
            MAE=${MAE:-"N/A"}
        else
            MSE="NO LOG"; MAE="NO LOG"
        fi

        printf "%-6s | %-7s | %-24s | %s\n" "$alpha" "$pred_len" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
    done
done

echo "" | tee -a "$MASTER_LOG"
echo "NEXT STEP:"                                                                     | tee -a "$MASTER_LOG"
echo "  1. Check if same alpha wins at both pred_lens 96+192"                        | tee -a "$MASTER_LOG"
echo "  2. If YES  → pattern stable → run Phase 2 (best alpha only, all pred_lens, 3 seeds)" | tee -a "$MASTER_LOG"
echo "  3. If NO   → pattern unstable → must run full 45 runs"                       | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"
