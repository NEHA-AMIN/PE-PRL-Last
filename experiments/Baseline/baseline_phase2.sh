#!/bin/bash
# =============================================================================
# Phase 2: Comprehensive Baseline Informer Model on ETTh1
# Runs:     18 (pred_len [24,48,96,192,336,720] × seed [2021,2022,2023])
# =============================================================================

PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020"
LOG_DIR="$PROJECT_ROOT/logs/baseline_phase2"
RESULTS_DIR="$PROJECT_ROOT/results"

mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

echo "=============================================" | tee "$MASTER_LOG"
echo "Baseline |  PHASE 2 — COMPREHENSIVE EVALUATION" | tee -a "$MASTER_LOG"
echo "Components:   Standard Informer (No Ablation)"  | tee -a "$MASTER_LOG"
echo "Dataset:      ETTh1"                            | tee -a "$MASTER_LOG"
echo "Seeds:        2021, 2022, 2023 (stability)"     | tee -a "$MASTER_LOG"
echo "Pred lengths: 24, 48, 96, 192, 336, 720"        | tee -a "$MASTER_LOG"
echo "Total runs:   18 (6 pred_len × 3 seeds)"        | tee -a "$MASTER_LOG"
echo "Start time:   $(date)"                          | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"

cd "$INFORMER_DIR" || { echo "ERROR: Cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

echo "Running from directory: $(pwd)" | tee -a "$MASTER_LOG"

TOTAL=0; COMPLETED=0; FAILED=0; SKIPPED=0

# Loop through all prediction lengths and seeds
for PRED_LEN in 24 48 96 192 336 720; do
    for SEED in 2021 2022 2023; do
        TOTAL=$((TOTAL + 1))
        RUN_ID="baseline_ph2_ETTh1_pred${PRED_LEN}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "" | tee -a "$MASTER_LOG"
        echo "---------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN:        $RUN_ID"                             | tee -a "$MASTER_LOG"
        echo "Pred len:   $PRED_LEN"                           | tee -a "$MASTER_LOG"
        echo "Seed:       $SEED"                               | tee -a "$MASTER_LOG"
        echo "Progress:   $COMPLETED/$TOTAL completed"         | tee -a "$MASTER_LOG"
        echo "Start:      $(date)"                             | tee -a "$MASTER_LOG"
        echo "---------------------------------------------" | tee -a "$MASTER_LOG"

        # Resume logic - skip if already completed
        if grep -q "^mse:" "$RUN_LOG" 2>/dev/null; then
            echo "SKIPPING — already completed" | tee -a "$MASTER_LOG"
            SKIPPED=$((SKIPPED + 1))
            COMPLETED=$((COMPLETED + 1))
            continue
        fi

        # Adjust seq_len and label_len based on pred_len for better performance
        if [ "$PRED_LEN" -le 48 ]; then
            SEQ_LEN=96
            LABEL_LEN=48
        elif [ "$PRED_LEN" -le 192 ]; then
            SEQ_LEN=168
            LABEL_LEN=168
        else
            SEQ_LEN=336
            LABEL_LEN=336
        fi

        # Standard Informer Baseline with --attn prob
        if python -u main_informer.py \
            --model informer \
            --data ETTh1 \
            --root_path ./data/ETT/ \
            --data_path ETTh1.csv \
            --features M \
            --attn prob \
            --seq_len "$SEQ_LEN" \
            --label_len "$LABEL_LEN" \
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
        
        # Progress update
        REMAINING=$((TOTAL - COMPLETED))
        echo "Progress: $COMPLETED/$TOTAL completed, $REMAINING remaining" | tee -a "$MASTER_LOG"
    done
done

# =============================================================================
# SUMMARY
# =============================================================================
echo "" | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"
echo "PHASE 2 COMPLETE — COMPREHENSIVE BASELINE"    | tee -a "$MASTER_LOG"
echo "Total: $TOTAL | Done: $COMPLETED | Skipped: $SKIPPED | Failed: $FAILED" | tee -a "$MASTER_LOG"
echo "End time: $(date)" | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# =============================================================================
# DETAILED RESULTS TABLE
# =============================================================================
echo "MSE/MAE results extracted from logs:" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "PredLen | Seed | MSE      | MAE      | Status" | tee -a "$MASTER_LOG"
echo "--------|------|----------|----------|----------" | tee -a "$MASTER_LOG"

for PRED_LEN in 24 48 96 192 336 720; do
    for SEED in 2021 2022 2023; do
        RUN_ID="baseline_ph2_ETTh1_pred${PRED_LEN}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep -oP 'mse:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
            MAE=$(grep -oP 'mae:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
            STATUS=$(grep -oP 'STATUS: \K\w+' "$RUN_LOG" | tail -1)
            printf "%-7s | %-4s | %-8s | %-8s | %s\n" "$PRED_LEN" "$SEED" "$MSE" "$MAE" "$STATUS" | tee -a "$MASTER_LOG"
        else
            printf "%-7s | %-4s | LOG NOT FOUND\n" "$PRED_LEN" "$SEED" | tee -a "$MASTER_LOG"
        fi
    done
    echo "--------|------|----------|----------|----------" | tee -a "$MASTER_LOG"
done

# =============================================================================
# AGGREGATED STATISTICS (Mean ± Std across seeds)
# =============================================================================
echo "" | tee -a "$MASTER_LOG"
echo "Aggregated Results (Mean ± Std across 3 seeds):" | tee -a "$MASTER_LOG"
echo "PredLen | MSE (Mean±Std)      | MAE (Mean±Std)" | tee -a "$MASTER_LOG"
echo "--------|---------------------|-------------------" | tee -a "$MASTER_LOG"

for PRED_LEN in 24 48 96 192 336 720; do
    MSE_VALUES=""
    MAE_VALUES=""
    
    for SEED in 2021 2022 2023; do
        RUN_ID="baseline_ph2_ETTh1_pred${PRED_LEN}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep -oP 'mse:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
            MAE=$(grep -oP 'mae:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
            if [ -n "$MSE" ]; then
                MSE_VALUES="$MSE_VALUES $MSE"
                MAE_VALUES="$MAE_VALUES $MAE"
            fi
        fi
    done
    
    if [ -n "$MSE_VALUES" ]; then
        # Calculate mean and std using awk
        MSE_STATS=$(echo "$MSE_VALUES" | awk '{
            sum=0; sumsq=0; n=0;
            for(i=1; i<=NF; i++) {
                sum+=$i; sumsq+=$i*$i; n++;
            }
            mean=sum/n;
            std=sqrt(sumsq/n - mean*mean);
            printf "%.4f±%.4f", mean, std;
        }')
        
        MAE_STATS=$(echo "$MAE_VALUES" | awk '{
            sum=0; sumsq=0; n=0;
            for(i=1; i<=NF; i++) {
                sum+=$i; sumsq+=$i*$i; n++;
            }
            mean=sum/n;
            std=sqrt(sumsq/n - mean*mean);
            printf "%.4f±%.4f", mean, std;
        }')
        
        printf "%-7s | %-19s | %s\n" "$PRED_LEN" "$MSE_STATS" "$MAE_STATS" | tee -a "$MASTER_LOG"
    else
        printf "%-7s | NO DATA\n" "$PRED_LEN" | tee -a "$MASTER_LOG"
    fi
done

echo "" | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"
echo "All logs saved to: $LOG_DIR" | tee -a "$MASTER_LOG"
echo "All results saved to: $RESULTS_DIR" | tee -a "$MASTER_LOG"
echo "=============================================" | tee -a "$MASTER_LOG"

# Made with Bob
