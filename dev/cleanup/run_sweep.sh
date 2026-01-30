#!/bin/bash
cd "$(dirname "$0")/.."

  # Run scale=10 with seeds 6-10 (you already have 1-5)
for seed in 1 2 3 4 5 6 7 8 9 10; do
   echo "Model1: seed=$seed..."
   uv run dev/qtable_feats_potential_best.py --reward-shaping --gamma 0.9 --alpha 0.01 --potential-scale 50 --episodes 300 --seed $seed --label "bestmodel1_s50_g0.9_a0.01_ep300_seed${seed}"
done

# Generate summary
# uv run dev/summarize_results.py
