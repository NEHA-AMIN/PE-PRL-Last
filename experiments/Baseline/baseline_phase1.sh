#!/bin/bash
# =============================================================================
# Phase 1: Baseline Informer Model on ETTh1
# Runs:     2 (pred_len 96, 192 × seed 2021)
# =============================================================================

# Adjust paths for Colab environment
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020"
LOG_DIR="$PROJECT_ROOT/logs/baseline_phase1"
RESULTS_DIR="$PROJECT_ROOT/results"

mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

echo "=============================================" | tee "$MASTER_LOG"
echo "Baseline |  PHASE 1 — EXPLORATION"              | tee -a "$MASTER_LOG"
echo "Components:   Standard Informer (No Ablation)"  | tee -a "$MASTER_LOG"
echo "Dataset:      ETTh1"                            | tee -a "$MASTER_LOG"
echo "Seed:         2021 (single — exploration only)" | tee -a "$MASTER_LOG"
echo "Pred lengths: 96, 192 (trend detection only)"   | tee -a "$MASTER_LOG"
echo "Total runs:   2"                                | tee -a "$MASTER_LOG"
echo "Start time:   $(date)"                          | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"

cd "$INFORMER_DIR" || { echo "ERROR: Cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

echo "Running from directory: $(pwd)" | tee -a "$MASTER_LOG"

TOTAL=0; COMPLETED=0; FAILED=0; SKIPPED=0
SEED=2021

for PRED_LEN in 96 192; do
    TOTAL=$((TOTAL + 1))
    RUN_ID="baseline_ph1_ETTh1_pred${PRED_LEN}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"
    mkdir -p "$RESULTS_DIR/${RUN_ID}"

    echo "" | tee -a "$MASTER_LOG"
    echo "---------------------------------------------" | tee -a "$MASTER_LOG"
    echo "RUN:        $RUN_ID"                             | tee -a "$MASTER_LOG"
    echo "Pred len:   $PRED_LEN"                           | tee -a "$MASTER_LOG"
    echo "Start:      $(date)"                             | tee -a "$MASTER_LOG"
    echo "---------------------------------------------" | tee -a "$MASTER_LOG"

    # Resume logic
    if grep -q "^mse:" "$RUN_LOG" 2>/dev/null; then
        echo "SKIPPING — already completed" | tee -a "$MASTER_LOG"
        SKIPPED=$((SKIPPED + 1))
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    # Standard Informer Baseline uses --attn prob
    if python -u main_informer.py \
        --model informer \
        --data ETTh1 \
        --root_path ./data/ETT/ \
        --data_path ETTh1.csv \
        --features M \
        --attn prob \
        --seq_len 96 \
        --label_len 48 \
        --pred_len "$PRED_LEN" \
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
        --des "$RUN_ID" \
        2>&1 | tee "$RUN_LOG"; then

        if grep -q "^mse:" "$RUN_LOG"; then
            COMPLETED=$((COMPLETED + 1))
            echo "STATUS: COMPLETED" | tee -a "$RUN_LOG"
            echo "STATUS: COMPLETED" | tee -a "$MASTER_LOG"
        else
            FAILED=$((FAILED + 1))
            echo "STATUS: FAILED — exited 0 but no MSE found" | tee -a "$MASTER_LOG"
        fi
    else
        FAILED=$((FAILED + 1))
        echo "STATUS: FAILED — check $RUN_LOG" | tee -a "$MASTER_LOG"
    fi

    echo "End: $(date)" | tee -a "$MASTER_LOG"
done

# =============================================================================
# SUMMARY
# =============================================================================
echo "" | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"
echo "PHASE 1 COMPLETE — BASELINE SUMMARY"           | tee -a "$MASTER_LOG"
echo "Total: $TOTAL | Done: $COMPLETED | Skipped: $SKIPPED | Failed: $FAILED" | tee -a "$MASTER_LOG"
echo "End time: $(date)" | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "MSE results extracted from logs:"             | tee -a "$MASTER_LOG"
echo "PredLen | MSE      | MAE      | Status"       | tee -a "$MASTER_LOG"
echo "--------|----------|----------|-------"       | tee -a "$MASTER_LOG"

for PRED_LEN in 96 192; do
    RUN_ID="baseline_ph1_ETTh1_pred${PRED_LEN}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"
    if [ -f "$RUN_LOG" ]; then
        MSE=$(grep -oP 'mse:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
        MAE=$(grep -oP 'mae:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
        STATUS=$(grep -oP 'STATUS: \K\w+' "$RUN_LOG" | tail -1)
        printf "%-7s | %-8s | %-8s | %s\n" "$PRED_LEN" "$MSE" "$MAE" "$STATUS" | tee -a "$MASTER_LOG"
    else
        echo "$PRED_LEN  | LOG NOT FOUND" | tee -a "$MASTER_LOG"
    fi
done

# Made with Bob
