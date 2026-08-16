#!/bin/bash
# =============================================================================
# Experiment 3b — Phase 2: Label + Temporal (Controlled)
#
# Formula   : X'_i = X_i + T_i + P_i
# Value     : TokenEmbedding (semantic content)
# Temporal  : TimeFeatureEmbedding (RESTORED)
# Position  : Legendre polynomials (replaces sinusoidal PE)
# No distance operator, no alpha, no ordering signal
#
# Phase 1 note: pred=96  MSE=0.9265 vs Exp1-Pre 0.8683
#               pred=192 MSE=0.8888 vs Exp1-Pre 0.8463
#               Partial recovery vs Exp3 but below Exp1-Pre at pred=96,
#               narrowing at pred=192 → full sweep warranted.
#
# Phase 2 goal:
#   Full statistical sweep — 3 seeds × 4 pred_lens = 12 runs.
#   Produces mean ± std per pred_len for rigorous comparison vs Exp1-Pre.
#
# Runs: 1 config × 3 seeds × 4 pred_lens = 12 runs
#   Seeds     : 2021, 2022, 2023
#   Pred lens : 48, 96, 192, 336
#
# Key fix: PYTHONPATH includes models/ for legendre import
#          (embed.py uses bare `from legendre_embedding import ...`)
#          legendre_embedding.py also copied to Informer root for safety
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/E-96-3b-Label-Temporal-Controlled"
LOG_DIR="$PROJECT_ROOT/logs/exp3b_phase2"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo "Experiment 3b — Phase 2: Label + Temporal (Controlled)"
echo "============================================================"
echo "Formula   : X'_i = X_i + T_i + P_i"
echo "Value     : TokenEmbedding (semantic content)"
echo "Temporal  : TimeFeatureEmbedding (RESTORED)"
echo "Position  : Legendre polynomials (replaces sinusoidal PE)"
echo "No distance operator, no alpha, no ordering signal"
echo ""
echo "Phase 1 note: pred=96  MSE=0.9265 vs Exp1-Pre 0.8683"
echo "              pred=192 MSE=0.8888 vs Exp1-Pre 0.8463"
echo "              Partial recovery vs Exp3 but below Exp1-Pre"
echo ""
echo "Seeds     : 2021, 2022, 2023"
echo "Pred lens : 48, 96, 192, 336"
echo "Total runs: 12 (1 config × 3 seeds × 4 pred_lens)"
echo "Start: $(date)"
echo "============================================================"
} | tee "$MASTER_LOG"

# --- FILE COPY ---
echo "" | tee -a "$MASTER_LOG"
echo "Copying experiment files..." | tee -a "$MASTER_LOG"

cp "$EXP_DIR/models/__init__.py"           "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"               "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"              "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"            "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"            "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"              "$INFORMER_DIR/models/model.py"
# CRITICAL: embed.py uses bare `from legendre_embedding import LegendrePositionEmbedding`
# Python resolves bare imports from CWD and from the models/ package directory.
# Copy to BOTH locations to ensure the import resolves regardless of working directory.
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/legendre_embedding.py"

echo "Files copied successfully." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

# --- RUNS ---
for SEED in 2021 2022 2023; do
    for pred_len in 48 96 192 336; do

        TOTAL=$((TOTAL + 1))
        RUN_ID="exp3b_ph2_ETTh1_leg_temporal_pred${pred_len}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN $TOTAL/12: $RUN_ID" | tee -a "$MASTER_LOG"
        echo "pred_len=$pred_len | seed=$SEED | config=legendre+temporal" | tee -a "$MASTER_LOG"
        echo "Start: $(date)" | tee -a "$MASTER_LOG"

        # RESUME LOGIC — skip if already completed (safe for Colab disconnects)
        if grep -q "STATUS: COMPLETED" "$RUN_LOG" 2>/dev/null; then
            echo "SKIPPING — already completed. Remove log to re-run." | tee -a "$MASTER_LOG"
            SKIPPED=$((SKIPPED + 1))
            COMPLETED=$((COMPLETED + 1))
            echo "End: $(date)" | tee -a "$MASTER_LOG"
            echo "" | tee -a "$MASTER_LOG"
            continue
        fi

        # PYTHON RUN
        cd "$INFORMER_DIR" || { echo "ERROR: cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

        # PYTHONPATH includes models/ so that `from legendre_embedding import ...` resolves
        PYTHONPATH="$INFORMER_DIR/models:$INFORMER_DIR:$PYTHONPATH" \
        python -u main_informer.py \
            --model informer \
            --data ETTh1 \
            --root_path ./data/ETT/ \
            --data_path ETTh1.csv \
            --features M \
            --target OT \
            --freq h \
            --attn full \
            --embed timeF \
            --seq_len 96 \
            --label_len 48 \
            --pred_len "$pred_len" \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --d_model 512 \
            --n_heads 8 \
            --e_layers 2 \
            --d_layers 1 \
            --d_ff 2048 \
            --factor 5 \
            --padding 0 \
            --distil \
            --dropout 0.05 \
            --activation gelu \
            --mix \
            --train_epochs 6 \
            --batch_size 32 \
            --patience 3 \
            --learning_rate 0.0001 \
            --lradj type1 \
            --itr 1 \
            --num_workers 0 \
            --des "$RUN_ID" \
            2>&1 | tee "$RUN_LOG"
        PYTHON_EXIT=${PIPESTATUS[0]}

        if [ "$PYTHON_EXIT" -eq 0 ]; then
            COMPLETED=$((COMPLETED + 1))
            echo "STATUS: COMPLETED" | tee -a "$RUN_LOG"
            echo "STATUS: COMPLETED" | tee -a "$MASTER_LOG"
        else
            FAILED=$((FAILED + 1))
            echo "STATUS: FAILED (exit $PYTHON_EXIT) — check $RUN_LOG" | tee -a "$MASTER_LOG"
        fi

        echo "End: $(date)" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"

    done
done

# --- RESTORE ORIGINAL MODELS ---
cd "$INFORMER_DIR" || exit 1
git checkout -- ./models/
# Remove root-level legendre_embedding.py that was added for the import fix
rm -f "$INFORMER_DIR/legendre_embedding.py"

# --- SUMMARY TABLE ---
{
echo "============================================================"
echo "PHASE 2 COMPLETE — Experiment 3b (Label + Temporal)"
echo "TOTAL=$TOTAL | COMPLETED=$COMPLETED | FAILED=$FAILED | SKIPPED=$SKIPPED"
echo "End time: $(date)"
echo ""
echo "Per-run MSE/MAE:"
echo "Seed  | PredLen | MSE      | MAE"
echo "------|---------|----------|----------"
} | tee -a "$MASTER_LOG"

for SEED in 2021 2022 2023; do
    for pred_len in 48 96 192 336; do
        RUN_ID="exp3b_ph2_ETTh1_leg_temporal_pred${pred_len}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"

        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep -i "mse:" "$RUN_LOG" | tail -1 | grep -oP 'mse:\K[0-9]+\.[0-9]+')
            MAE=$(grep -i "mae:" "$RUN_LOG" | tail -1 | grep -oP 'mae:\K[0-9]+\.[0-9]+')
            MSE=${MSE:-"N/A"}
            MAE=${MAE:-"N/A"}
        else
            MSE="NO LOG"
            MAE="NO LOG"
        fi

        printf "%-5s | %-7s | %-8s | %s\n" "$SEED" "$pred_len" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
    done
done

{
echo ""
echo "------------------------------------------------------------"
echo "EXP1-PRE REFERENCE (alpha=1.0, best baseline):"
echo "Seed  | PredLen | MSE"
echo "2021  | 48      | 0.7680"
echo "2021  | 96      | 0.8796"
echo "2021  | 192     | 0.9376"
echo "2021  | 336     | 0.8764"
echo "2022  | 48      | 0.7742"
echo "2022  | 96      | 0.8706"
echo "2022  | 192     | 0.9298"
echo "2022  | 336     | 1.1616"
echo "2023  | 48      | 0.8518"
echo "2023  | 96      | 0.8509"
echo "2023  | 192     | 0.9446"
echo "2023  | 336     | 1.0725"
echo ""
echo "NEXT STEP:"
echo "  Average MSE across 3 seeds per pred_len"
echo "  If Exp3b avg < Exp1-Pre avg at most pred_lens → improvement confirmed"
echo "  If Exp3b avg > Exp1-Pre avg consistently → negative result, document"
echo "============================================================"
} | tee -a "$MASTER_LOG"
