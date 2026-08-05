#!/bin/bash
# =============================================================================
# Formula A (Positional Space) — Phase 1: Consecutive Legendre Delta Ordering
#
# Experiment name : Formula-A-pos
# Embedding class : DataEmbedding_ordering_pos
# pe_mode flag    : --pe_mode ordering_pos
#
# Formula
# -------
#   Output:   X'_i = X_i + T_i + P_i + O_i^pos
#
#   where:
#     P_i       = LegendrePositionEmbedding(x)    [B, L, D]
#     delta_i   = 0               if i = 0        (zero-pad boundary)
#                 P_i - P_{i-1}   if i >= 1
#     p_bar     = (1/N) * sum_i ||P_i||_2          scalar normaliser
#     O_i^pos   = delta_i / (p_bar + 1e-8)        [B, L, D]
#
# Contrast with Formula-B-pos
# ---------------------------
#   Formula A (positional): delta_i = P_i - P_{i-1}      consecutive diff
#   Formula B (positional): delta_i = mu_p - P_i          global mean deviation
#
# What changed from baseline
# --------------------------
#   REMOVED : sinusoidal PositionalEmbedding (PE_i)
#   ADDED   : Legendre label P_i + consecutive-delta ordering O_i^pos
#   UNCHANGED: TokenEmbedding, TemporalEmbedding, Encoder, Decoder,
#              Attention, projection head, all hyperparameters
#
# CRITICAL extra file: embed.py imports LegendrePositionEmbedding from
# legendre_embedding.py at __init__ time. legendre_embedding.py MUST be copied.
#
# Phase 1 goal
# ------------
#   seed=2021, pred_len ∈ {96, 192}.
#   Establish whether consecutive-delta ordering in positional (Legendre) space
#   can match or exceed Exp1-Pre (alpha=1.0) on ETTh1 multi-variate.
#
# Total runs: 2
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/Formula-A-pos"
LOG_DIR="$PROJECT_ROOT/logs/Formula-A-pos-phase1"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo " Formula A (Positional Space) — Phase 1: Legendre Consecutive Delta"
echo " Date: $(date)"
echo "------------------------------------------------------------"
echo " Formula:  X'_i = X_i + T_i + P_i + O_i^pos"
echo " Signal:   O_i^pos = (P_i - P_{i-1}) / p_bar  (zero-pad at i=0)"
echo " pe_mode:  ordering_pos"
echo "------------------------------------------------------------"
echo " Seed:       2021"
echo " Pred lens:  96, 192"
echo " Total runs: 2"
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
# Without this file the import will fail immediately at runtime.
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"

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
    RUN_ID="formula_a_pos_ph1_ETTh1_pred${pred_len}_seed${SEED}"
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
        --pe_mode ordering_pos \
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
echo " PHASE 1 SUMMARY — Formula A (Positional Space)"            | tee -a "$MASTER_LOG"
echo " Total: $TOTAL | Completed: $COMPLETED | Failed: $FAILED | Skipped: $SKIPPED" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"

printf " %-55s %10s %10s\n" "RUN_ID" "MSE" "MAE" | tee -a "$MASTER_LOG"
printf " %-55s %10s %10s\n" "-------------------------------------------------------" "----------" "----------" | tee -a "$MASTER_LOG"

for pred_len in 96 192; do
    RUN_ID="formula_a_pos_ph1_ETTh1_pred${pred_len}_seed${SEED}"
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

echo "" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo " REFERENCE — Exp1-Pre alpha=1.0 (seed=2021):"               | tee -a "$MASTER_LOG"
echo "   pred=96  MSE=0.8683"                                      | tee -a "$MASTER_LOG"
echo "   pred=192 MSE=0.8463  ← target to beat"                   | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo " DECISION GUIDE:"                                             | tee -a "$MASTER_LOG"
echo "   Formula-A-pos wins BOTH vs Exp1-Pre → proceed to Phase 2" | tee -a "$MASTER_LOG"
echo "   Formula-A-pos wins ONE  vs Exp1-Pre → Phase 2 to confirm" | tee -a "$MASTER_LOG"
echo "   Formula-A-pos loses BOTH            → document, stop"     | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "End: $(date)" | tee -a "$MASTER_LOG"
