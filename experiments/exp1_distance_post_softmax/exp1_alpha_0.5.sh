#!/bin/bash
# =============================================================================
# Phase 2: Experiment 1 POST SOFTMAX on ETTh1 (Alpha = 0.5)
# Runs:     4 (pred_len 48, 96, 192, 336) × 3 (seed 2021, 2022, 2023)
# =============================================================================

PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
LOG_DIR="$PROJECT_ROOT/logs/exp1post_phase2"
RESULTS_DIR="$PROJECT_ROOT/results"

mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run_alpha0.5.log"

echo "============================================================" | tee "$MASTER_LOG"
echo "Exp1-Post  |  PHASE 2 — RESULTS STABILITY (alpha=0.5)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "Seed      : 2021, 2022, 2023 | Pred lengths: 48, 96, 192, 336"    | tee -a "$MASTER_LOG"
echo "Start: $(date)"                                               | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

cd "$INFORMER_DIR" || { echo "ERROR: Cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

# Copy the custom models to the Informer directory
cp -r ../experiments/exp1_distance_post_softmax/models/* ./models/
echo "Files copied." | tee -a "$MASTER_LOG"

TOTAL=0; COMPLETED=0; FAILED=0; SKIPPED=0

ALPHA="0.5"

# Modify the decay_a parameter in the copied attn.py
sed -i.bak -e "s/decay_a=[0-9.]*/decay_a=${ALPHA}/g" ./models/attn.py
rm -f ./models/attn.py.bak

for SEED in 2021 2022 2023; do
    for PRED_LEN in 48 96 192 336; do
        TOTAL=$((TOTAL + 1))
        RUN_ID="exp1post_ph2_ETTh1_alpha${ALPHA}_pred${PRED_LEN}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "" | tee -a "$MASTER_LOG"
        echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN $TOTAL/12: $RUN_ID"                                        | tee -a "$MASTER_LOG"
        echo "pred_len=$PRED_LEN | seed=$SEED | alpha=$ALPHA"             | tee -a "$MASTER_LOG"
        echo "Start: $(date)"                                                | tee -a "$MASTER_LOG"

        # Resume logic — check for actual MSE output
        if grep -q "^mse:" "$RUN_LOG" 2>/dev/null; then
            echo "SKIPPING — already completed with valid results" | tee -a "$MASTER_LOG"
            SKIPPED=$((SKIPPED + 1))
            COMPLETED=$((COMPLETED + 1))
            continue
        fi

        if python3 -u main_informer.py \
            --model informer \
            --data ETTh1 \
            --root_path ./data/ETT/ \
            --data_path ETTh1.csv \
            --features M \
            --attn full \
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
                echo "STATUS: FAILED — exited 0 but no MSE found — check $RUN_LOG" | tee -a "$MASTER_LOG"
            fi
        else
            FAILED=$((FAILED + 1))
            echo "STATUS: FAILED — check $RUN_LOG" | tee -a "$MASTER_LOG"
        fi

        echo "End: $(date)" | tee -a "$MASTER_LOG"
    done
done

# Restore original files
git checkout ./models/

# =============================================================================
# SUMMARY
# =============================================================================
echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "PHASE 2 COMPLETE — RESULTS SUMMARY (alpha=$ALPHA)"                          | tee -a "$MASTER_LOG"
echo "Total: $TOTAL | Done: $COMPLETED | Skipped: $SKIPPED | Failed: $FAILED" | tee -a "$MASTER_LOG"
echo "End time: $(date)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Per-run MSE/MAE (alpha=$ALPHA, post-softmax):" | tee -a "$MASTER_LOG"
echo "Seed  | PredLen | MSE      | MAE"                      | tee -a "$MASTER_LOG"
echo "------|---------|----------|----------"                       | tee -a "$MASTER_LOG"

for SEED in 2021 2022 2023; do
    for PRED_LEN in 48 96 192 336; do
        RUN_ID="exp1post_ph2_ETTh1_alpha${ALPHA}_pred${PRED_LEN}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep "mse:" "$RUN_LOG" | tail -1 | sed -E 's/.*mse:([0-9]+\.[0-9]+).*/\1/')
            MAE=$(grep "mae:" "$RUN_LOG" | tail -1 | sed -E 's/.*mae:([0-9]+\.[0-9]+).*/\1/')
            printf "%-5s | %-7s | %-8s | %-8s\n" "$SEED" "$PRED_LEN" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
        else
            echo "$SEED  | $PRED_LEN  | LOG NOT FOUND" | tee -a "$MASTER_LOG"
        fi
    done
done
