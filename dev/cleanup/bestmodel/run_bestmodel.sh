#!/bin/bash
# Run best model configuration with 7 different seeds for robustness evaluation
# Your groupmate should fill in the best parameters from the sweep below

cd "$(dirname "$0")/../.."

# ============================================================
# FILL IN BEST PARAMETERS FROM SWEEP HERE
# ============================================================
GAMMA=0.9                  # Discount factor (e.g., 0.9, 0.95, 0.99)
ALPHA=0.1                  # Learning rate (e.g., 0.1, 0.05, 0.01)
EPISODES=160               # Number of training episodes
REWARD_SHAPING=true        # Set to true or false
EPSILON_DECAY=0.995        # Epsilon decay rate (e.g., 0.995, 0.99)
# ============================================================

# Fixed parameters
POTENTIAL_SCALE=50.0

echo "========================================"
echo "Running Best Model Configuration"
echo "========================================"
echo "gamma=$GAMMA, alpha=$ALPHA, episodes=$EPISODES"
echo "reward_shaping=$REWARD_SHAPING, potential_scale=$POTENTIAL_SCALE"
echo "epsilon_decay=$EPSILON_DECAY"
echo "========================================"
echo ""

# Run 7 seeds (seeds 4-10, assuming sweep used seeds 1-3)
for seed in 4 5 6 7 8 9 10; do
    echo "Running seed=$seed..."

    if [ "$REWARD_SHAPING" = true ]; then
        python3 dev/qtable_feats_potential.py \
            --reward-shaping \
            --gamma $GAMMA \
            --alpha $ALPHA \
            --episodes $EPISODES \
            --epsilon-decay $EPSILON_DECAY \
            --potential-scale $POTENTIAL_SCALE \
            --seed $seed \
            --label "bestmodel_seed${seed}"
    else
        python3 dev/qtable_feats_potential.py \
            --gamma $GAMMA \
            --alpha $ALPHA \
            --episodes $EPISODES \
            --epsilon-decay $EPSILON_DECAY \
            --seed $seed \
            --label "bestmodel_seed${seed}"
    fi
done

echo ""
echo "========================================"
echo "All seeds complete. Run summarize_results_bestmodel.py to analyze."
echo "========================================"
