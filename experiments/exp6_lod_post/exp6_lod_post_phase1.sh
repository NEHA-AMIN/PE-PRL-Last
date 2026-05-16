#!/bin/bash

# exp6_lod_post_phase1.sh
# Search for optimal decay_a (alpha) across pred_len 96 and 192 (Seed 2021)

ALPHAS=(0.5 1.0 2.0)
PRED_LENS=(96 192)

for alpha in "${ALPHAS[@]}"; do
  # Convert 0.5 to 0p5 for the descriptor string
  alpha_str="${alpha//./p}"
  
  for pred_len in "${PRED_LENS[@]}"; do
    echo "Running Phase 1: decay_a=${alpha}, pred_len=${pred_len}, seed=2021"
    
    python -u main_informer.py \
      --model informer --data ETTh1 --features M \
      --seq_len 96 --label_len 48 --pred_len $pred_len \
      --enc_in 7 --dec_in 7 --c_out 7 \
      --e_layers 2 --d_layers 1 --attn full --factor 5 \
      --des "exp6_lod_post_ph1_ETTh1_a${alpha_str}_pred${pred_len}_seed2021" \
      --itr 1 --train_epochs 6 \
      --decay_a $alpha
  done
done
