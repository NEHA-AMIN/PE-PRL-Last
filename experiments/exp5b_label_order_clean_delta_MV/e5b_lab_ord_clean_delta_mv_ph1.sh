#!/bin/bash
# =============================================================================
# Experiment 5b: Phase 1 — Label + Order with Clean Delta (L+O Clean)
#
# Architecture summary:
#   Q, K ← project(value_emb + temporal_emb + legendre_pos)   [combined]
#   V    ← project(delta_x)   where delta_x[i] = x_i - x_{i-1} (clean)
#
# Key facts from code inspection:
#   - No alpha / decay_a anywhere in attn.py, embed.py, or model.py
#   - No sweep needed: single fixed configuration
#   - embed.py returns a TUPLE (combined_emb, delta_x) — non-standard
#   - legendre_embedding.py is imported inside embed.py → MUST be copied
#   - decoder.py uses DataEmbeddingDecoder (standard, no delta) 
#   - delta_x flows: embed → encoder → EncoderLayer → AttentionLayer (V only)
#
# Phase 1 goal:
#   Detect whether L+O Clean beats best previous result at short + mid horizon.
#   Best reference so far: Exp1-Pre alpha=1.0
#
# Runs: 1 config × 1 seed × 2 pred_lens = 2 runs
#   pred_lens : 96, 192
#   seed      : 2021
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp5b_label_order_clean_delta_MV"
LOG_DIR="$PROJECT_ROOT/logs/exp5b_label_order_clean_delta_phase1"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo " Experiment 5b — Phase 1: Label + Order Clean Delta"
echo " Date: $(date)"
echo "------------------------------------------------------------"
echo " Architecture:"
echo "   Q, K  ← combined_emb  (value + temporal + legendre)"
echo "   V     ← delta_x       (x_i - x_{i-1}, value_emb ONLY)"
echo "------------------------------------------------------------"
echo " What is fixed:   single config, no alpha sweep"
echo " What is swept:   pred_len in {96, 192}"
echo " Seed:            2021 (Phase 1 only)"
echo " Total runs:      2"
echo "------------------------------------------------------------"
echo " Reference results (Exp1-Pre, alpha=1.0, seed=2021):"
echo "   pred_len=96  → MSE ~0.725, MAE ~0.652  [CHECK YOUR LOGS]"
echo "   pred_len=192 → MSE ~0.725, MAE ~0.652  [CHECK YOUR LOGS]"
echo " Reference results (Exp5, seed=2021):"
echo "   pred_len=96  → MSE ~0.719, MAE ~0.635  [CHECK YOUR LOGS]"
echo "   pred_len=192 → MSE ~0.719, MAE ~0.635  [CHECK YOUR LOGS]"
echo "============================================================"
} | tee "$MASTER_LOG"

# --- FILE COPY ---
# Standard 6 model files
echo "" | tee -a "$MASTER_LOG"
echo "Copying model files to $INFORMER_DIR/models/ ..." | tee -a "$MASTER_LOG"

cp "$EXP_DIR/models/__init__.py"  "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"      "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"     "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"   "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"   "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"     "$INFORMER_DIR/models/model.py"

# CRITICAL extra file: embed.py does `from legendre_embedding import LegendrePositionEmbedding`
# inside DataEmbedding.__init__ AND DataEmbeddingDecoder.__init__
# Without this file the import will fail silently at runtime
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"

echo "File copy complete." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

# --- LOOPS ---
# No alpha loop: this experiment has a single fixed configuration.
# Phase 1: seed=2021 only, pred_len in {96, 192}

SEED=2021

for pred_len in 96 192; do

    TOTAL=$((TOTAL + 1))
    RUN_ID="exp5b_ph1_ETTh1_lod_clean_pred${pred_len}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"
    mkdir -p "$RESULTS_DIR/${RUN_ID}"

    echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "RUN $TOTAL/2: $RUN_ID" | tee -a "$MASTER_LOG"
    echo "Start: $(date)" | tee -a "$MASTER_LOG"

    # RESUME LOGIC — skip if already completed (safe for Colab disconnects)
    if grep -q "STATUS: COMPLETED" "$RUN_LOG" 2>/dev/null; then
        echo "SKIPPING — already completed. Remove log to re-run." | tee -a "$MASTER_LOG"
        SKIPPED=$((SKIPPED + 1))
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    # PYTHON RUN
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

# --- SUMMARY TABLE ---
echo "============================================================" | tee -a "$MASTER_LOG"
echo " PHASE 1 SUMMARY — Experiment 5b (L+O Clean Delta)"        | tee -a "$MASTER_LOG"
echo " Total: $TOTAL | Completed: $COMPLETED | Failed: $FAILED | Skipped: $SKIPPED" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo " Results (parsed from logs):"                                | tee -a "$MASTER_LOG"
echo ""                                                            | tee -a "$MASTER_LOG"

printf " %-55s %10s %10s\n" "RUN_ID" "MSE" "MAE" | tee -a "$MASTER_LOG"
printf " %-55s %10s %10s\n" "-------------------------------------------------------" "----------" "----------" | tee -a "$MASTER_LOG"

for pred_len in 96 192; do
    RUN_ID="exp5b_ph1_ETTh1_lod_clean_pred${pred_len}_seed${SEED}"
    RUN_LOG="$LOG_DIR/${RUN_ID}.log"

    if [ -f "$RUN_LOG" ]; then
        # Extract last occurrence of mse and mae from test output
        MSE=$(grep -i "mse:" "$RUN_LOG" | tail -1 | grep -oP 'mse:\s*\K[0-9]+\.[0-9]+')
        MAE=$(grep -i "mae:" "$RUN_LOG" | tail -1 | grep -oP 'mae:\s*\K[0-9]+\.[0-9]+')
        MSE=${MSE:-"N/A"}
        MAE=${MAE:-"N/A"}
    else
        MSE="NO LOG"
        MAE="NO LOG"
    fi

    printf " %-55s %10s %10s\n" "$RUN_ID" "$MSE" "$MAE" | tee -a "$MASTER_LOG"
done

echo "" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo " Reference (Exp5, positional delta, seed=2021):"            | tee -a "$MASTER_LOG"
echo "   pred_96  → MSE ~0.719  MAE ~0.635"                       | tee -a "$MASTER_LOG"
echo "   pred_192 → MSE ~0.719  MAE ~0.635  [update with actuals]"| tee -a "$MASTER_LOG"
echo " Reference (Exp1-Pre, alpha=1.0, seed=2021):"               | tee -a "$MASTER_LOG"
echo "   pred_96  → MSE ~0.725  MAE ~0.652  [update with actuals]"| tee -a "$MASTER_LOG"
echo "   pred_192 → MSE ~0.725  MAE ~0.652  [update with actuals]"| tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo ""                                                            | tee -a "$MASTER_LOG"
echo " DECISION GUIDE:"                                            | tee -a "$MASTER_LOG"
echo ""                                                            | tee -a "$MASTER_LOG"
echo " Does Exp5b beat Exp5 (MSE < 0.719) at BOTH pred_lens?"     | tee -a "$MASTER_LOG"
echo "   YES → Clean delta improves L+O. Proceed to Phase 2."     | tee -a "$MASTER_LOG"
echo "   NO  → Does it beat Exp5 at even one pred_len?"           | tee -a "$MASTER_LOG"
echo "         YES (mixed) → Still worth Phase 2 validation."     | tee -a "$MASTER_LOG"
echo "         NO (worse everywhere) → Document as negative."     | tee -a "$MASTER_LOG"
echo ""                                                            | tee -a "$MASTER_LOG"
echo " Also check: does Exp5b beat Exp1-Pre (MSE < 0.725)?"       | tee -a "$MASTER_LOG"
echo "   This answers: does L+O Clean match Distance-only?"        | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"