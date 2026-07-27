#!/bin/bash
# =============================================================================
# Experiment 4 revised: Phase 1 — Ordering in Semantic Space
#
# Name: ordering_new_sem_space
# Class: DataEmbedding_ordering_sem
#
# Formula:
#   X'_i = X_i + T_i + O_i^sem
#   O_i^sem = delta_i^val / (x_bar^val + 1e-8)
#   delta_i^val = X_i - X_{i-1}  (delta_0 = 0)
#   x_bar^val   = (1/N) sum_i ||X_i||_2
#
# Components: value YES, temporal YES, sinusoidal PE NO, Legendre NO
#
# pe_mode flag passed to model: --pe_mode ordering_sem
#
# Phase 1 goal:
#   Does a locally order-sensitive signal built from content transitions
#   provide sequential structure? Sweep pred_len {96, 192} at seed 2021.
#
# Total runs: 2
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp4_ordering_new_sem_space"
LOG_DIR="$PROJECT_ROOT/logs/exp4_ordering_new_sem_space_phase1"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo " Experiment 4 revised — Phase 1: Ordering in Semantic Space"
echo " Date: $(date)"
echo "------------------------------------------------------------"
echo " Formula:  X'_i = X_i + T_i + O_i^sem"
echo " Signal:   O_i^sem = (X_i - X_{i-1}) / x_bar^val"
echo " pe_mode:  ordering_sem"
echo "------------------------------------------------------------"
echo " Sweep:    pred_len in {96, 192}   seed=2021"
echo " Total runs: 2"
echo "============================================================"
} | tee "$MASTER_LOG"

# --- FILE COPY ---
echo "" | tee -a "$MASTER_LOG"
echo "Copying model files to $INFORMER_DIR/models/ ..." | tee -a "$MASTER_LOG"

cp "$EXP_DIR/models/__init__.py"  "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"      "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"     "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"   "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"   "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"     "$INFORMER_DIR/models/model.py"

echo "File copy complete." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

SEED=2021

for pred_len in 96 192; do

    TOTAL=$((TOTAL + 1))
    RUN_ID="exp4_ord_sem_ph1_ETTh1_pred${pred_len}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"
    mkdir -p "$RESULTS_DIR/${RUN_ID}"

    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "RUN $TOTAL/2: $RUN_ID" | tee -a "$MASTER_LOG"
    echo "Start: $(date)" | tee -a "$MASTER_LOG"

    if grep -q "STATUS: COMPLETED" "$RUN_LOG" 2>/dev/null; then
        echo "SKIPPING — already completed." | tee -a "$MASTER_LOG"
        SKIPPED=$((SKIPPED + 1))
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    cd "$INFORMER_DIR" || { echo "ERROR: cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

    if python -u main_informer.py \
        --model informer \
        --data ETTh1 \
        --root_path ./data/ETT/ \
        --data_path ETTh1.csv \
        --features M \
        --attn full \
        --seq_len 96 \
        --label_len 48 \
        --pred_len "$pred_len" \
        --e_layers 2 \
        --d_layers 1 \
        --factor 5 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 512 \
        --n_heads 8 \
        --d_ff 2048 \
        --dropout 0.05 \
        --embed timeF \
        --freq h \
        --activation gelu \
        --train_epochs 6 \
        --patience 3 \
        --learning_rate 0.0001 \
        --batch_size 32 \
        --itr 1 \
        --pe_mode ordering_sem \
        --des "$RUN_ID" \
        2>&1 | tee "$RUN_LOG"; then

        COMPLETED=$((COMPLETED + 1))
        echo "STATUS: COMPLETED" | tee -a "$RUN_LOG"
        echo "STATUS: COMPLETED" | tee -a "$MASTER_LOG"
    else
        FAILED=$((FAILED + 1))
        echo "STATUS: FAILED — check $RUN_LOG" | tee -a "$MASTER_LOG"
    fi

    echo "End: $(date)" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

done

# --- SUMMARY ---
echo "============================================================" | tee -a "$MASTER_LOG"
echo " PHASE 1 SUMMARY — Exp4 Ordering Semantic Space"            | tee -a "$MASTER_LOG"
echo " Total: $TOTAL | Completed: $COMPLETED | Failed: $FAILED | Skipped: $SKIPPED" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"

printf " %-55s %10s %10s\n" "RUN_ID" "MSE" "MAE" | tee -a "$MASTER_LOG"
printf " %-55s %10s %10s\n" "-------------------------------------------------------" "----------" "----------" | tee -a "$MASTER_LOG"

for pred_len in 96 192; do
    RUN_ID="exp4_ord_sem_ph1_ETTh1_pred${pred_len}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"

    if [ -f "$RUN_LOG" ]; then
        MSE=$(grep -i "mse:" "$RUN_LOG" | tail -1 | grep -oP 'mse:\s*\K[0-9]+\.[0-9]+')
        MAE=$(grep -i "mae:" "$RUN_LOG" | tail -1 | grep -oP 'mae:\s*\K[0-9]+\.[0-9]+')
        MSE=${MSE:-"N/A"}
        MAE=${MAE:-"N/A"}
    else
        MSE="NO LOG"; MAE="NO LOG"
    fi

    printf " %-55s %10s %10s\n" "$RUN_ID" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
done

echo "============================================================" | tee -a "$MASTER_LOG"
