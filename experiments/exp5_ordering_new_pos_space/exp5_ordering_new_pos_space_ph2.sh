#!/bin/bash
# =============================================================================
# Experiment 5 revised: Phase 2 VALIDATION — Ordering in Positional Space
#
# Name:  ordering_new_pos_space
# Class: DataEmbedding_ordering_pos
#
# Formula:
#   X'_i = X_i + T_i + P_i + O_i^pos
#   O_i^pos = delta_i^leg / (p_bar^leg + 1e-8)
#   delta_i^leg = P_i - P_{i-1}  (delta_0 = 0)
#   p_bar^leg   = (1/N) sum_i ||P_i||_2
#
# Components: value YES, temporal YES, Legendre YES, sinusoidal PE NO
#
# pe_mode flag: --pe_mode ordering_pos
#
# CRITICAL: embed.py imports LegendrePositionEmbedding from
#   legendre_embedding.py at __init__ time. Both files MUST be copied.
#
# Phase 1 findings (seed=2021):
#   pred=96  MSE=0.8118 ← best seen across all experiments ✅
#   pred=192 MSE=0.7896 ← best seen across all experiments ✅
#   Won at BOTH horizons — Phase 2 is mandatory.
#
# Runs:  9 (3 seeds × 3 pred_lens × 1 config)
# Seeds: 2021, 2022, 2023
# Pred:  48, 96, 192
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp5_ordering_new_pos_space"
LOG_DIR="$PROJECT_ROOT/logs/exp5_ordering_new_pos_space_phase2"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo " Experiment 5 revised — Phase 2: Ordering in Positional Space"
echo " Date: $(date)"
echo "------------------------------------------------------------"
echo " Formula:  X'_i = X_i + T_i + P_i + O_i^pos"
echo " Signal:   O_i^pos = (P_i - P_{i-1}) / p_bar^leg"
echo " pe_mode:  ordering_pos"
echo "------------------------------------------------------------"
echo " Phase 1 note: pred=96  MSE=0.8118 <- best seen ✅"
echo "               pred=192 MSE=0.7896 <- best seen ✅"
echo "               Won at BOTH horizons — Phase 2 mandatory"
echo "------------------------------------------------------------"
echo " Seeds:    2021, 2022, 2023"
echo " Pred lens: 48, 96, 192"
echo " Total runs: 9 (1 config × 3 seeds × 3 pred_lens)"
echo "============================================================"
} | tee "$MASTER_LOG"

# --- FILE COPY ---
echo "" | tee -a "$MASTER_LOG"
echo "Copying model files to $INFORMER_DIR/models/ ..." | tee -a "$MASTER_LOG"

cp "$EXP_DIR/models/__init__.py"            "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"               "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"              "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"            "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"            "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"              "$INFORMER_DIR/models/model.py"

# CRITICAL: embed.py imports LegendrePositionEmbedding at __init__ time.
# Without this file the import will fail immediately.
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"

echo "File copy complete." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

cd "$INFORMER_DIR" || { echo "ERROR: cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

for seed in 2021 2022 2023; do
    for pred_len in 48 96 192; do

        TOTAL=$((TOTAL + 1))
        RUN_ID="exp5_ord_pos_ph2_ETTh1_pred${pred_len}_seed${seed}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "" | tee -a "$MASTER_LOG"
        echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN $TOTAL/9: $RUN_ID"                                        | tee -a "$MASTER_LOG"
        echo "pred_len=$pred_len | seed=$seed | config=ordering_pos"        | tee -a "$MASTER_LOG"
        echo "Start: $(date)"                                                | tee -a "$MASTER_LOG"

        # Resume logic — skip if MSE line already present in log
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
            --pe_mode ordering_pos \
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
echo "PHASE 2 COMPLETE — Exp5 Ordering Positional Space"          | tee -a "$MASTER_LOG"
echo "TOTAL=$TOTAL | COMPLETED=$COMPLETED | FAILED=$FAILED | SKIPPED=$SKIPPED" | tee -a "$MASTER_LOG"
echo "End time: $(date)"                                           | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Per-run MSE/MAE:"                                            | tee -a "$MASTER_LOG"
echo "Seed  | PredLen | MSE      | MAE"                            | tee -a "$MASTER_LOG"
echo "------|---------|----------|----------"                       | tee -a "$MASTER_LOG"

for seed in 2021 2022 2023; do
    for pred_len in 48 96 192; do
        RUN_ID="exp5_ord_pos_ph2_ETTh1_pred${pred_len}_seed${seed}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep -oP 'mse:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
            MAE=$(grep -oP 'mae:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
            MSE=${MSE:-"N/A"}
            MAE=${MAE:-"N/A"}
            printf "%-5s | %-7s | %-8s | %s\n" "$seed" "$pred_len" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
        fi
    done
done

echo "" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo "EXP1-PRE REFERENCE (alpha=1.0, best baseline):"              | tee -a "$MASTER_LOG"
echo "Seed  | PredLen | MSE"                                        | tee -a "$MASTER_LOG"
echo "2021  | 48      | 0.7680"                                     | tee -a "$MASTER_LOG"
echo "2021  | 96      | 0.8683"                                     | tee -a "$MASTER_LOG"
echo "2021  | 192     | 0.8463"                                     | tee -a "$MASTER_LOG"
echo "2022  | 48      | 0.7742"                                     | tee -a "$MASTER_LOG"
echo "2022  | 96      | 0.8706"                                     | tee -a "$MASTER_LOG"
echo "2022  | 192     | 0.9298"                                     | tee -a "$MASTER_LOG"
echo "2023  | 48      | 0.8518"                                     | tee -a "$MASTER_LOG"
echo "2023  | 96      | 0.8509"                                     | tee -a "$MASTER_LOG"
echo "2023  | 192     | 0.9446"                                     | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "EXP5 PHASE 1 REFERENCE (seed=2021 only):"                    | tee -a "$MASTER_LOG"
echo "2021  | 96      | 0.8118  <- best seen at Phase 1"           | tee -a "$MASTER_LOG"
echo "2021  | 192     | 0.7896  <- best seen at Phase 1"           | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "NEXT STEP:"                                                   | tee -a "$MASTER_LOG"
echo "  Average MSE across 3 seeds per pred_len."                  | tee -a "$MASTER_LOG"
echo "  Key Q1: Do pred=96 and pred=192 wins hold across seeds?"   | tee -a "$MASTER_LOG"
echo "  Key Q2: Is Exp5-pos consistently better than Exp1-Pre?"    | tee -a "$MASTER_LOG"
echo "  Key Q3: How does Exp5-pos compare to Exp4b-pos (if run)?"  | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
