#!/bin/bash
# =============================================================================
# Formula A (Semantic Space) — Phase 2: Consecutive Delta Ordering
#
# Experiment name : formula-A-sem
# Embedding class : DataEmbedding_delta_pos
# pe_mode flag    : --pe_mode delta_pos
#
# Formula
# -------
#   Baseline:  X'_i = X_i + PE_i + T_i
#   Formula A: X'_i = X_i + Δ(X_i) + T_i
#
#   where:
#     Δ(X_i) = X_i − X_{i−1}   for i ≥ 1
#     Δ(X_0) = X_0 − 0         for i = 0  (zero-pad boundary)
#     X_i    = TokenEmbedding(x)  [B, L, d_model]
#
# What changed from baseline
# --------------------------
#   REMOVED : sinusoidal PositionalEmbedding (PE_i)
#   ADDED   : consecutive delta Δ(X_i) in the positional slot
#   UNCHANGED: TokenEmbedding, TemporalEmbedding, Encoder, Decoder,
#              Attention, projection head, all hyperparameters
#
# Extra files needed: none — embed.py is self-contained (no legendre helper)
#
# Phase 2 goal
# ------------
#   Multi-seed robustness check across seeds {2021, 2022, 2023}.
#   pred_len ∈ {48, 96, 192, 336}.
#   Confirm that Phase 1 results hold across seeds and extended horizons.
#
#   NOTE: main_informer.py does not accept a --seed flag. Each run uses a
#   distinct --des identifier so results are logged separately per seed.
#
# Total runs: 12  (3 seeds × 4 pred_lens)
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/formula-A-sem"
LOG_DIR="$PROJECT_ROOT/logs/formula-A-sem-phase2"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo " Formula A (Semantic Space) — Phase 2: Consecutive Delta Ordering"
echo " Date: $(date)"
echo "------------------------------------------------------------"
echo " Formula:   X'_i = X_i + Delta(X_i) + T_i"
echo " Signal:    Delta(X_i) = X_i - X_{i-1}  (zero-pad at i=0)"
echo " pe_mode:   delta_pos"
echo " Replaces:  sinusoidal PE"
echo "------------------------------------------------------------"
echo " Seeds:      2021, 2022, 2023"
echo " Pred lens:  48, 96, 192, 336"
echo " Total runs: 12"
echo "============================================================"
} | tee "$MASTER_LOG"

# --- FILE COPY ---
echo "" | tee -a "$MASTER_LOG"
echo "Copying model files to $INFORMER_DIR/models/ ..." | tee -a "$MASTER_LOG"

cp "$EXP_DIR/models/__init__.py" "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"    "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"   "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py" "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py" "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"   "$INFORMER_DIR/models/model.py"

echo "File copy complete." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

cd "$INFORMER_DIR" || { echo "ERROR: cannot cd to $INFORMER_DIR" | tee -a "$MASTER_LOG"; exit 1; }

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

for seed in 2021 2022 2023; do
for pred_len in 48 96 192 336; do

    TOTAL=$((TOTAL + 1))
    RUN_ID="formula_a_sem_ph2_ETTh1_pred${pred_len}_seed${seed}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"
    mkdir -p "$RESULTS_DIR/${RUN_ID}"

    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "RUN $TOTAL/12: $RUN_ID" | tee -a "$MASTER_LOG"
    echo "pred_len=$pred_len | seed=$seed" | tee -a "$MASTER_LOG"
    echo "Start: $(date)" | tee -a "$MASTER_LOG"

    # Resume guard — check for actual MSE output in the log
    if grep -q "^mse:" "$RUN_LOG" 2>/dev/null; then
        echo "SKIPPING — already completed with valid results." | tee -a "$MASTER_LOG"
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
        --pe_mode delta_pos \
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
    echo "" | tee -a "$MASTER_LOG"

done
done

# --- SUMMARY ---
echo "============================================================" | tee -a "$MASTER_LOG"
echo " PHASE 2 SUMMARY — Formula A (Semantic Space)"              | tee -a "$MASTER_LOG"
echo " Total: $TOTAL | Completed: $COMPLETED | Failed: $FAILED | Skipped: $SKIPPED" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo "Seed  | PredLen | MSE      | MAE"                            | tee -a "$MASTER_LOG"
echo "------|---------|----------|----------"                       | tee -a "$MASTER_LOG"

for seed in 2021 2022 2023; do
for pred_len in 48 96 192 336; do
    RUN_ID="formula_a_sem_ph2_ETTh1_pred${pred_len}_seed${seed}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"

    if [ -f "$RUN_LOG" ]; then
        MSE=$(grep -oP 'mse:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
        MAE=$(grep -oP 'mae:\K[0-9]+\.[0-9]+' "$RUN_LOG" | tail -1)
        MSE=${MSE:-"N/A"}
        MAE=${MAE:-"N/A"}
    else
        MSE="NO LOG"; MAE="NO LOG"
    fi

    printf "%-5s | %-7s | %-8s | %s\n" "$seed" "$pred_len" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
done
    echo "" | tee -a "$MASTER_LOG"
done

echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo " REFERENCE — Exp1-Pre alpha=1.0:"                            | tee -a "$MASTER_LOG"
echo " seed=2021 | pred=96  MSE=0.8683"                            | tee -a "$MASTER_LOG"
echo " seed=2021 | pred=192 MSE=0.8463  ← Phase 1 target"         | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo " DECISION GUIDE:"                                             | tee -a "$MASTER_LOG"
echo "   Consistent improvement across all 3 seeds → result is robust"           | tee -a "$MASTER_LOG"
echo "   Improvement only at seed=2021             → Phase 1 was a fluke, investigate" | tee -a "$MASTER_LOG"
echo "   Mixed results across pred_lens            → horizon-dependent, report range"  | tee -a "$MASTER_LOG"
echo "   Regression vs Phase 1 at all seeds        → implementation issue, debug"      | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "End: $(date)" | tee -a "$MASTER_LOG"
