#!/bin/bash
# =============================================================================
# Experiment 4: Phase 2 VALIDATION — Ordering Only
# Formula:  X'_i = X_i + T_i + O_i
# O_i = (1/N-1) · Σ_{j≠i} (X_i - X_j) — uniform signed displacements
# Decision: Phase 1 showed mixed results:
#           pred=96  MSE=0.8220 ← best seen so far ✅
#           pred=192 MSE=0.9997 ← worse than all except Exp3 ❌
#           → Validate across seeds to check if 96 win is real
# Runs:     9 (3 pred_len × 3 seeds × 1 config)
# NOTE:     Skipping pred_len=720 — O(L²) operator is too slow on Colab
# WARNING:  Each epoch is ~60s at pred=96, ~110s at pred=192
# =============================================================================

PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp4_order_only"
LOG_DIR="$PROJECT_ROOT/logs/exp4_order_only_phase2"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

echo "============================================================" | tee "$MASTER_LOG"
echo "Experiment 4 — Phase 2: Ordering Only"                       | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "Formula   : X'_i = X_i + T_i + O_i"                         | tee -a "$MASTER_LOG"
echo "O_i       : (1/N-1) * sum_j(X_i - X_j)"                     | tee -a "$MASTER_LOG"
echo "  Uniform signed displacements — no decay, no Legendre"      | tee -a "$MASTER_LOG"
echo ""                                                             | tee -a "$MASTER_LOG"
echo "Phase 1 note: pred=96  MSE=0.8220 ← best seen so far"        | tee -a "$MASTER_LOG"
echo "              pred=192 MSE=0.9997 ← worse than Exp1-Pre"     | tee -a "$MASTER_LOG"
echo "              Epoch time ~60s at pred=96, ~110s at pred=192"  | tee -a "$MASTER_LOG"
echo ""                                                             | tee -a "$MASTER_LOG"
echo "Seeds     : 2021, 2022, 2023"                                | tee -a "$MASTER_LOG"
echo "Pred lens : 48, 96, 192 (720 skipped — O(L²) too slow)"      | tee -a "$MASTER_LOG"
echo "Total runs: 9 (1 config × 3 seeds × 3 pred_lens)"           | tee -a "$MASTER_LOG"
echo "Start: $(date)"                                               | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "Copying experiment files..." | tee -a "$MASTER_LOG"
cp "$EXP_DIR/models/__init__.py"          "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"              "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"             "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"           "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"           "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"             "$INFORMER_DIR/models/model.py"
cp "$EXP_DIR/models/ordering_operator.py" "$INFORMER_DIR/models/ordering_operator.py"
echo "Files copied successfully." | tee -a "$MASTER_LOG"

cd "$INFORMER_DIR" || { echo "ERROR: Cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

TOTAL=0; COMPLETED=0; FAILED=0; SKIPPED=0

for seed in 2021 2022 2023; do
    for pred_len in 48 96 192; do
        TOTAL=$((TOTAL + 1))
        RUN_ID="exp4_ph2_ETTh1_ordering_pred${pred_len}_seed${seed}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "" | tee -a "$MASTER_LOG"
        echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN $TOTAL/9: $RUN_ID"                                       | tee -a "$MASTER_LOG"
        echo "pred_len=$pred_len | seed=$seed | config=ordering_only"       | tee -a "$MASTER_LOG"
        echo "Start: $(date)"                                                | tee -a "$MASTER_LOG"

        # Resume logic — check for actual MSE output
        if grep -q "^mse:" "$RUN_LOG" 2>/dev/null; then
            echo "SKIPPING — already completed with valid results" | tee -a "$MASTER_LOG"
            SKIPPED=$((SKIPPED + 1))
            COMPLETED=$((COMPLETED + 1))
            continue
        fi

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

# =============================================================================
# SUMMARY
# =============================================================================
echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "PHASE 2 COMPLETE — Experiment 4 (Ordering Only)"            | tee -a "$MASTER_LOG"
echo "TOTAL=$TOTAL | COMPLETED=$COMPLETED | FAILED=$FAILED | SKIPPED=$SKIPPED" | tee -a "$MASTER_LOG"
echo "End time: $(date)"                                           | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Per-run MSE/MAE:"                                            | tee -a "$MASTER_LOG"
echo "Seed  | PredLen | MSE      | MAE"                            | tee -a "$MASTER_LOG"
echo "------|---------|----------|----------"                       | tee -a "$MASTER_LOG"

for seed in 2021 2022 2023; do
    for pred_len in 48 96 192; do
        RUN_ID="exp4_ph2_ETTh1_ordering_pred${pred_len}_seed${seed}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep -oP 'mse:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
            MAE=$(grep -oP 'mae:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
            printf "%-5s | %-7s | %-8s | %s\n" "$seed" "$pred_len" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
        fi
    done
done

echo "" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo "EXP1-PRE REFERENCE (alpha=1.0, best baseline):"              | tee -a "$MASTER_LOG"
echo "Seed  | PredLen | MSE"                                        | tee -a "$MASTER_LOG"
echo "2021  | 48      | 0.7680"                                     | tee -a "$MASTER_LOG"
echo "2021  | 96      | 0.8796"                                     | tee -a "$MASTER_LOG"
echo "2021  | 192     | 0.9376"                                     | tee -a "$MASTER_LOG"
echo "2022  | 48      | 0.7742"                                     | tee -a "$MASTER_LOG"
echo "2022  | 96      | 0.8706"                                     | tee -a "$MASTER_LOG"
echo "2022  | 192     | 0.9298"                                     | tee -a "$MASTER_LOG"
echo "2023  | 48      | 0.8518"                                     | tee -a "$MASTER_LOG"
echo "2023  | 96      | 0.8509"                                     | tee -a "$MASTER_LOG"
echo "2023  | 192     | 0.9446"                                     | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "NEXT STEP:"                                                   | tee -a "$MASTER_LOG"
echo "  Average MSE across 3 seeds per pred_len"                   | tee -a "$MASTER_LOG"
echo "  Key question: Does pred=96 win hold across all 3 seeds?"   | tee -a "$MASTER_LOG"
echo "  If yes → ordering helps at short horizons → document"      | tee -a "$MASTER_LOG"
echo "  If no  → Phase 1 win was lucky seed → negative result"     | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
