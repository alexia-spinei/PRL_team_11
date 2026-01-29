#!/bin/bash
# Train all 10 seeds with best-checkpoint saving
# Re-trains to capture the BEST validation Q-table (not just final)

cd "$(dirname "$0")/../.."

# ============================================================
# FILL IN THE 3 SWEEP RUN DIRECTORIES HERE (from results/qlearning/)
# These are the directory names, e.g., "20250128_143022_s50.0_seed1"
# ============================================================
SWEEP_RUN_1="20260129_002515_bestmodel_seed4"
SWEEP_RUN_2="20260129_002515_bestmodel_seed7"
SWEEP_RUN_3="20260129_002515_bestmodel_seed9"


QLEARNING_DIR="results/qlearning"
config_path="$QLEARNING_DIR/$run_dir/config.json"

OUTPUT_DIR="results/best_checkpoints"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Training Best Checkpoints (10 seeds)"
echo "========================================"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Process the 3 sweep runs first
echo ""
echo "Processing 3 sweep runs..."
for run_dir in "$SWEEP_RUN_1" "$SWEEP_RUN_2" "$SWEEP_RUN_3"; do
    config_path="$QLEARNING_DIR/$run_dir/config.json"
    if [ ! -f "$config_path" ]; then
        echo "ERROR: Config not found: $config_path"
        echo "Please fill in the correct directory names at the top of this script"
        exit 1
    fi
    echo ""
    echo "========================================"
    echo "Training from: $run_dir"
    echo "========================================"
    python3 dev/bestmodel/train_best_checkpoint.py --from-config "$config_path" --output-dir "$OUTPUT_DIR"
done

# Process the 7 bestmodel runs (seeds 4-10)
echo ""
echo "Processing 7 bestmodel runs (seeds 4-10)..."
for seed in 4 5 6 7 8 9 10; do
    echo ""
    echo "========================================"
    echo "Training seed=$seed (from bestmodel config)..."
    echo "========================================"
    python3 dev/bestmodel/train_best_checkpoint.py --auto-find --seed $seed --output-dir "$OUTPUT_DIR"
done

echo ""
echo "========================================"
echo "All seeds complete!"
echo "========================================"
echo ""
echo "Output files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"
