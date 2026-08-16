#!/bin/bash
# =============================================================================
# Experiment 6 LOD Post-Softmax — Phase 2: Stability Validation (decay_a=0.5)
#
# Architecture summary:
#   Q, K ← project(value_emb + temporal_emb + legendre_pos)   [combined]
#   V    ← project(delta_x)   where delta_x[i] = x_i - x_{i-1}
#
#   Distance weighting (POST-softmax):
#     dist_matrix[i,j] = |i - j|
#     alpha[i,j]       = 1 / (1 + dist_matrix[i,j] ^ decay_a)
#     weights          = softmax(scores)               ← softmax FIRST
#     weights          = weights * alpha               ← applied AFTER softmax
#     output           = weights · V
#
# Key facts from code inspection:
#   - embed.py returns a TUPLE (combined_emb, delta_x) — non-standard
#   - embed.py applies dropout to combined_emb but NOT to delta_x (differs from Pre)
#   - legendre_embedding.py is imported inside embed.py → MUST be copied
#   - encoder.py: ConvLayer applied to both x and delta_x (FINDING G fix)
#   - model.py: InformerStack fully corrected (FINDING F fix)
#   - decay_a forwarded through exp_informer.py (FINDING H fix)
#   - model.py accepts **kwargs so pe_mode from exp_informer.py is silently ignored
#
# Phase 2 goal:
#   Stability validation — fix decay_a=0.5 and sweep all pred_lens across
#   3 seeds to confirm reproducibility of Phase 1 result.
#
# decay_a fixed at 0.5
# pred_lens: 48, 96, 192, 336
# seeds:     2021, 2022, 2023
# Total runs: 12  (4 pred_lens × 3 seeds)
# =============================================================================

# --- PATHS ---
PROJECT_ROOT="/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP_DIR="$PROJECT_ROOT/experiments/exp6_lod_post"
LOG_DIR="$PROJECT_ROOT/logs/exp6_lod_post_phase2_a0.5"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_run.log"

# --- HEADER ---
{
echo "============================================================"
echo " Experiment 6 LOD Post-Softmax — Phase 2: Stability Validation (decay_a=0.5)"
echo " Date: $(date)"
echo "------------------------------------------------------------"
echo " Architecture:"
echo "   Q, K  ← combined_emb  (value + temporal + legendre)"
echo "   V     ← delta_x       (x_i - x_{i-1}, no dropout)"
echo "   weights = softmax(Q·K^T / sqrt(d))   [softmax first]"
echo "   alpha   = 1/(1 + |i-j|^decay_a)       [post-softmax]"
echo "   output  = (weights * alpha) · V"
echo "------------------------------------------------------------"
echo " decay_a fixed at 0.5"
echo " Pred lens:     48, 96, 192, 336"
echo " Seeds:         2021, 2022, 2023"
echo " Total runs:    12  (4 pred_lens × 3 seeds)"
echo "------------------------------------------------------------"
echo " Difference vs Exp6-Pre:"
echo "   Pre:  scores = scores * alpha → softmax → weights  [stronger suppression]"
echo "   Post: softmax → weights → weights * alpha           [softer, linear]"
echo "============================================================"
} | tee "$MASTER_LOG"

# --- FILE COPY (once, before loops) ---
echo "" | tee -a "$MASTER_LOG"
echo "Copying model files to $INFORMER_DIR/models/ ..." | tee -a "$MASTER_LOG"

cp "$EXP_DIR/models/__init__.py"           "$INFORMER_DIR/models/__init__.py"
cp "$EXP_DIR/models/attn.py"               "$INFORMER_DIR/models/attn.py"
cp "$EXP_DIR/models/embed.py"              "$INFORMER_DIR/models/embed.py"
cp "$EXP_DIR/models/encoder.py"            "$INFORMER_DIR/models/encoder.py"
cp "$EXP_DIR/models/decoder.py"            "$INFORMER_DIR/models/decoder.py"
cp "$EXP_DIR/models/model.py"              "$INFORMER_DIR/models/model.py"
# CRITICAL: embed.py uses `from legendre_embedding import LegendrePositionEmbedding`
# This is a bare (non-package) import. Python resolves it from the CWD ($INFORMER_DIR)
# and from inside the models/ package directory. Copy to BOTH locations to be safe.
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"
cp "$EXP_DIR/models/legendre_embedding.py" "$INFORMER_DIR/legendre_embedding.py"

echo "File copy complete (7 files + legendre_embedding.py in models/ and root)." | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# --- COUNTERS ---
TOTAL=0
COMPLETED=0
FAILED=0
SKIPPED=0

# --- LOOPS ---
# Phase 2: decay_a=0.5 fixed; outer=pred_len, inner=seed

for pred_len in 48 96 192 336; do
    for SEED in 2021 2022 2023; do

        TOTAL=$((TOTAL + 1))
        RUN_ID="exp6post_ph2_ETTh1_lod_post_a0.5_pred${pred_len}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        mkdir -p "$RESULTS_DIR/${RUN_ID}"

        echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
        echo "RUN $TOTAL/12: $RUN_ID" | tee -a "$MASTER_LOG"
        echo "pred_len=$pred_len | seed=$SEED | decay_a=0.5" | tee -a "$MASTER_LOG"
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

        python -u main_informer.py \
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
            --decay_a 0.5 \
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
git checkout ./models/
# Also remove the root-level legendre_embedding.py that was added for the import fix
rm -f "$INFORMER_DIR/legendre_embedding.py"

# --- SUMMARY TABLE ---
echo "============================================================" | tee -a "$MASTER_LOG"
echo " PHASE 2 SUMMARY — Experiment 6 LOD Post-Softmax (decay_a=0.5)" | tee -a "$MASTER_LOG"
echo " Total: $TOTAL | Completed: $COMPLETED | Failed: $FAILED | Skipped: $SKIPPED" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------" | tee -a "$MASTER_LOG"
echo " Results by pred_len (all 3 seeds side by side):"           | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

printf " %-7s | %-4s | %-8s | %-8s | %s\n" "PredLen" "Seed" "MSE" "MAE" "Status" | tee -a "$MASTER_LOG"
echo " --------|------|----------|----------|----------" | tee -a "$MASTER_LOG"

for pred_len in 48 96 192 336; do
    for SEED in 2021 2022 2023; do
        RUN_ID="exp6post_ph2_ETTh1_lod_post_a0.5_pred${pred_len}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"

        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep -i "mse:" "$RUN_LOG" | tail -1 | grep -oP 'mse:\s*\K[0-9]+\.[0-9]+')
            MAE=$(grep -i "mae:" "$RUN_LOG" | tail -1 | grep -oP 'mae:\s*\K[0-9]+\.[0-9]+')
            STATUS=$(grep -oP 'STATUS: \K\w+' "$RUN_LOG" | tail -1)
            MSE=${MSE:-"N/A"}
            MAE=${MAE:-"N/A"}
            STATUS=${STATUS:-"N/A"}
            printf " %-7s | %-4s | %-8s | %-8s | %s\n" "$pred_len" "$SEED" "$MSE" "$MAE" "$STATUS" | tee -a "$MASTER_LOG"
        else
            printf " %-7s | %-4s | LOG NOT FOUND\n" "$pred_len" "$SEED" | tee -a "$MASTER_LOG"
        fi
    done
    echo " --------|------|----------|----------|----------" | tee -a "$MASTER_LOG"
done

# =============================================================================
# AGGREGATED STATISTICS (Mean ± Std across seeds)
# =============================================================================
echo "" | tee -a "$MASTER_LOG"
echo " Aggregated Results (Mean ± Std across 3 seeds):" | tee -a "$MASTER_LOG"
echo " PredLen | MSE (Mean±Std)      | MAE (Mean±Std)" | tee -a "$MASTER_LOG"
echo " --------|---------------------|-------------------" | tee -a "$MASTER_LOG"

for pred_len in 48 96 192 336; do
    MSE_VALUES=""
    MAE_VALUES=""

    for SEED in 2021 2022 2023; do
        RUN_ID="exp6post_ph2_ETTh1_lod_post_a0.5_pred${pred_len}_seed${SEED}"
        RUN_LOG="$LOG_DIR/${RUN_ID}.log"
        if [ -f "$RUN_LOG" ]; then
            MSE=$(grep -i "mse:" "$RUN_LOG" | tail -1 | grep -oP 'mse:\s*\K[0-9]+\.[0-9]+')
            MAE=$(grep -i "mae:" "$RUN_LOG" | tail -1 | grep -oP 'mae:\s*\K[0-9]+\.[0-9]+')
            if [ -n "$MSE" ]; then
                MSE_VALUES="$MSE_VALUES $MSE"
                MAE_VALUES="$MAE_VALUES $MAE"
            fi
        fi
    done

    if [ -n "$MSE_VALUES" ]; then
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

        printf " %-7s | %-19s | %s\n" "$pred_len" "$MSE_STATS" "$MAE_STATS" | tee -a "$MASTER_LOG"
    else
        printf " %-7s | NO DATA\n" "$pred_len" | tee -a "$MASTER_LOG"
    fi
done

echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo " All logs saved to:    $LOG_DIR" | tee -a "$MASTER_LOG"
echo " All results saved to: $RESULTS_DIR" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"