#!/bin/bash
# =============================================================================
# Experiment 3b — Phase 1: Label + Temporal (Controlled)
#
# Formula   : X'_i = X_i + T_i + P_i
# Value     : TokenEmbedding (semantic content)
# Temporal  : TimeFeatureEmbedding (RESTORED vs Exp3)
# Position  : Legendre polynomials (replaces sinusoidal PE)
# No distance operator, no alpha, no ordering signal
#
# Key fix   : PYTHONPATH includes models/ for legendre import
#             (embed.py uses bare `from legendre_embedding import ...`)
#             legendre_embedding.py is also copied to Informer root for safety
#
# Architecture (only embed.py differs from vanilla):
#   value_emb   = TokenEmbedding(x)
#   temporal_emb = TimeFeatureEmbedding(x_mark)    ← RESTORED
#   legendre_pos = LegendrePositionEmbedding(x)
#   output      = dropout(value_emb + temporal_emb + legendre_pos)
#
# Phase 1 goal:
#   Quick 2-run screen (pred_len 96 & 192, seed 2021) to verify that
#   restoring temporal embedding recovers performance vs Exp3 (label-only).
#
# Runs: 2 pred_lens × 1 seed = 2 runs
#   pred_lens : 96, 192
#   seed      : 2021
#
# Reference (seed=2021):
#   Exp1-Pre  alpha=1.0  pred=96  → MSE ~0.8683
#   Exp1-Pre  alpha=1.0  pred=192 → MSE ~0.8463  ← best so far
#   Exp2-LOD             pred=96  → MSE ~0.8242
#   Exp2-LOD             pred=192 → MSE ~0.9002
#   Exp3-LabelOnly       pred=96  → MSE ~1.0989  ← no temporal
#   Exp3-LabelOnly       pred=192 → MSE ~1.4844  ← no temporal
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/E-96-3b-Label-Temporal-Controlled"
LOG_DIR="$PROJECT_ROOT/logs/exp3b_phase1"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo "Experiment 3b — Phase 1: Label + Temporal (Controlled)"
echo "============================================================"
echo "Formula   : X'_i = X_i + T_i + P_i"
echo "Value     : TokenEmbedding (semantic content)"
echo "Temporal  : TimeFeatureEmbedding (RESTORED vs Exp3)"
echo "Position  : Legendre polynomials (replaces sinusoidal PE)"
echo "No distance operator, no alpha, no ordering signal"
echo ""
echo "Key fix   : PYTHONPATH includes models/ for legendre import"
echo "Seed      : 2021 | Pred lengths: 96, 192 | Total runs: 2"
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

# --- FIXED PARAMETERS ---
SEED=2021

# --- RUNS ---
for pred_len in 96 192; do

    TOTAL=$((TOTAL + 1))
    RUN_ID="exp3b_ph1_ETTh1_leg_temporal_pred${pred_len}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"
    mkdir -p "$RESULTS_DIR/${RUN_ID}"

    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "RUN $TOTAL/2: $RUN_ID" | tee -a "$MASTER_LOG"
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

# --- RESTORE ORIGINAL MODELS ---
cd "$INFORMER_DIR" || exit 1
git checkout -- ./models/
# Remove root-level legendre_embedding.py that was added for the import fix
rm -f "$INFORMER_DIR/legendre_embedding.py"

# --- SUMMARY TABLE ---
{
echo "============================================================"
echo "PHASE 1 RESULTS — Experiment 3b (Label + Temporal)"
echo "TOTAL=$TOTAL | COMPLETED=$COMPLETED | FAILED=$FAILED | SKIPPED=$SKIPPED"
echo ""
echo "PredLen | MSE      | MAE      | Status"
echo "--------|----------|----------|-------"
} | tee -a "$MASTER_LOG"

for pred_len in 96 192; do
    RUN_ID="exp3b_ph1_ETTh1_leg_temporal_pred${pred_len}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"

    if [ -f "$RUN_LOG" ]; then
        MSE=$(grep -i "mse:" "$RUN_LOG" | tail -1 | grep -oP 'mse:\K[0-9]+\.[0-9]+')
        MAE=$(grep -i "mae:" "$RUN_LOG" | tail -1 | grep -oP 'mae:\K[0-9]+\.[0-9]+')
        STATUS=$(grep "STATUS:" "$RUN_LOG" | tail -1 || echo "UNKNOWN")
        MSE=${MSE:-"N/A"}
        MAE=${MAE:-"N/A"}
    else
        MSE="NO LOG"
        MAE="NO LOG"
        STATUS="NO LOG"
    fi

    printf "%-7s | %-8s | %-8s | %s\n" "$pred_len" "$MSE" "$MAE" "$STATUS" | tee -a "$MASTER_LOG"
done

{
echo ""
echo "------------------------------------------------------------"
echo "REFERENCE — Previous experiments (seed=2021):"
echo "Experiment         | pred=96 MSE | pred=192 MSE"
echo "-------------------|-------------|-------------"
echo "Exp1-Pre a=1.0     | 0.8683      | 0.8463  ← best so far"
echo "Exp2-LOD           | 0.8242      | 0.9002"
echo "Exp3-LabelOnly     | 1.0989      | 1.4844  ← no temporal"
echo ""
echo "DECISION GUIDE:"
echo "  Exp3b wins at BOTH vs Exp1-Pre → Legendre+Temporal better → Phase 2"
echo "  Exp3b between Exp3 and Exp1-Pre → partial recovery → Phase 2"
echo "  Exp3b similar to Exp3 despite temporal → Legendre is problem → stop"
echo "============================================================"
echo "End: $(date)"
} | tee -a "$MASTER_LOG"
