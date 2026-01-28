#!/bin/bash
cd "$(dirname "$0")/.."

for scale in 40.0 50.0 60.0 70.0 80.0 90.0 100.0; do
  echo "running scale= $scale.."
  # Run scale=10 with seeds 6-10 (you already have 1-5)
  for seed in 1 2 3 4 5 6 7 8 9 10; do
      echo "Running scale=10, seed=$seed..."
      uv run dev/qtable_feats_potential.py --reward-shaping --gamma 0.9 --potential-scale $scale --episodes 160 --seed $seed --label "s${scale}_seed${seed}"
      done
done

# Generate summary
uv run dev/summarize_results.py
