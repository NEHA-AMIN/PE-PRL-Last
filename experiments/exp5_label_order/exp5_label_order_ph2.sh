# =============================================================================
# Experiment 5: Phase 2 VALIDATION — Label + Order (L + O)
# Formula:  X'_i = X_i + T_i + P_i + O_i
#           O_i = OrderingOperator(P_i) — order in POSITIONAL/Legendre space
#
# Phase 1 findings (seed=2021):
#   pred=96  MSE=0.8519 ← beats Exp1-Pre (0.8683) ✅
#   pred=192 MSE=0.9788 ← worse than Exp1-Pre (0.8463) ❌
#   Adding Order to Label helps at short horizon, hurts at long
#   Less stable than Exp4b (consecutive local ordering)
#
# Runs:     12 (4 pred_len × 3 seeds × 1 config)
# Pred:     48, 96, 192
# Seeds:    2021, 2022, 2023
# WARNING:  O(L²) operator — expect ~40-70s per epoch
# =============================================================================

PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp5_label_order"
LOG_DIR="$PROJECT_ROOT/logs/exp5_label_order_phase2"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

echo "============================================================" | tee "$MASTER_LOG"
echo "Experiment 5 — Phase 2: Label + Order (L + O)"              | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "Formula   : X'_i = X_i + T_i + P_i + O_i"                  | tee -a "$MASTER_LOG"
echo "  P_i = Legendre position label"                             | tee -a "$MASTER_LOG"
echo "  O_i = OrderingOperator(P_i) in POSITIONAL space"          | tee -a "$MASTER_LOG"
echo ""                                                             | tee -a "$MASTER_LOG"
echo "Phase 1 note: pred=96  MSE=0.8519 ← beats Exp1-Pre"         | tee -a "$MASTER_LOG"
echo "              pred=192 MSE=0.9788 ← worse than Exp1-Pre"    | tee -a "$MASTER_LOG"
echo "              Order helps short horizon, hurts long"         | tee -a "$MASTER_LOG"
echo ""                                                             | tee -a "$MASTER_LOG"
echo "Seeds     : 2021, 2022, 2023"                                | tee -a "$MASTER_LOG"
echo "Pred lens : 48, 96, 192"                                     | tee -a "$MASTER_LOG"
echo "Total runs: 12 (1 config × 3 seeds × 4 pred_lens)"          | tee -a "$MASTER_LOG"
echo "WARNING   : O(L²) — expect ~40-70s per epoch"               | tee -a "$MASTER_LOG"
echo "Start: $(date)"                                               | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

echo "" | tee -a "$MASTER_LOG"
echo "Copying experiment files..." | tee -a "$MASTER_LOG"
cp "$EXP_DIR/models/__init__.py"           "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"               "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"              "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"            "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"            "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"              "$INFORMER_DIR/models/model.py"
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"
cp "$EXP_DIR/models/ordering_operator.py"  "$INFORMER_DIR/models/ordering_operator.py"
echo "Files copied successfully." | tee -a "$MASTER_LOG"

cd "$INFORMER_DIR" || { echo "ERROR: Cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

TOTAL=0; COMPLETED=0; FAILED=0; SKIPPED=0

for seed in 2021 2022 2023; do
    for pred_len in 48 96 192; do
        TOTAL=$((TOTAL + 1))
        RUN_ID="exp5_ph2_ETTh1_label_order_pred${pred_len}_seed${seed}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "" | tee -a "$MASTER_LOG"
        echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN $TOTAL/12: $RUN_ID"                                       | tee -a "$MASTER_LOG"
        echo "pred_len=$pred_len | seed=$seed | config=label+order_positional" | tee -a "$MASTER_LOG"
        echo "Start: $(date)"                                                | tee -a "$MASTER_LOG"

        # Resume logic
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
                echo "STATUS: FAILED — exited 0 but no MSE — check $RUN_LOG" | tee -a "$MASTER_LOG"
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
echo "PHASE 2 COMPLETE — Experiment 5 (Label + Order)"            | tee -a "$MASTER_LOG"
echo "TOTAL=$TOTAL | COMPLETED=$COMPLETED | FAILED=$FAILED | SKIPPED=$SKIPPED" | tee -a "$MASTER_LOG"
echo "End time: $(date)"                                           | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Per-run MSE/MAE:"                                            | tee -a "$MASTER_LOG"
echo "Seed  | PredLen | MSE      | MAE"                            | tee -a "$MASTER_LOG"
echo "------|---------|----------|----------"                       | tee -a "$MASTER_LOG"

for seed in 2021 2022 2023; do
    for pred_len in 48 96 192; do
        RUN_ID="exp5_ph2_ETTh1_label_order_pred${pred_len}_seed${seed}"
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
echo "EXP1-PRE REFERENCE (alpha=1.0):"                             | tee -a "$MASTER_LOG"
echo "Seed  | PredLen | MSE"                                        | tee -a "$MASTER_LOG"
echo "2021  | 48      | 0.7680"                                     | tee -a "$MASTER_LOG"
echo "2021  | 96      | 0.8796"                                     | tee -a "$MASTER_LOG"
echo "2021  | 192     | 0.9376"                                     | tee -a "$MASTER_LOG"
echo "2021  | 336     | 0.8764"                                     | tee -a "$MASTER_LOG"
echo "2022  | 48      | 0.7742"                                     | tee -a "$MASTER_LOG"
echo "2022  | 96      | 0.8706"                                     | tee -a "$MASTER_LOG"
echo "2022  | 192     | 0.9298"                                     | tee -a "$MASTER_LOG"
echo "2022  | 336     | 1.1616"                                     | tee -a "$MASTER_LOG"
echo "2023  | 48      | 0.8518"                                     | tee -a "$MASTER_LOG"
echo "2023  | 96      | 0.8509"                                     | tee -a "$MASTER_LOG"
echo "2023  | 192     | 0.9446"                                     | tee -a "$MASTER_LOG"
echo "2023  | 336     | 1.0725"                                     | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "EXP3b REFERENCE (Label+Temporal):"                           | tee -a "$MASTER_LOG"
echo "Seed  | PredLen | MSE"                                        | tee -a "$MASTER_LOG"
echo "2021  | 48      | 0.8706"                                     | tee -a "$MASTER_LOG"
echo "2021  | 96      | 0.8823"                                     | tee -a "$MASTER_LOG"
echo "2021  | 192     | 0.9006"                                     | tee -a "$MASTER_LOG"
echo "2021  | 336     | 0.9989"                                     | tee -a "$MASTER_LOG"
echo "2022  | 48      | 0.8937"                                     | tee -a "$MASTER_LOG"
echo "2022  | 96      | 0.8450"                                     | tee -a "$MASTER_LOG"
echo "2022  | 192     | 0.8847"                                     | tee -a "$MASTER_LOG"
echo "2022  | 336     | 0.9770"                                     | tee -a "$MASTER_LOG"
echo "2023  | 48      | 0.9917"                                     | tee -a "$MASTER_LOG"
echo "2023  | 96      | 0.9302"                                     | tee -a "$MASTER_LOG"
echo "2023  | 192     | 0.8886"                                     | tee -a "$MASTER_LOG"
echo "2023  | 336     | 0.9244"                                     | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "NEXT STEP:"                                                   | tee -a "$MASTER_LOG"
echo "  Average MSE across 3 seeds per pred_len"                   | tee -a "$MASTER_LOG"
echo "  Key Q1: Does pred=96 win hold across seeds?"               | tee -a "$MASTER_LOG"
echo "  Key Q2: Is Exp5 better OR worse than Exp3b overall?"       | tee -a "$MASTER_LOG"
echo "  Key Q3: Is Exp5 better OR worse than Exp4b overall?"       | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
