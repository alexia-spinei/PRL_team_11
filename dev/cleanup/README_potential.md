# Q-Learning with Potential-Based Reward Shaping

Script: `qtable_feats_potential.py`

## Reward Shaping

Adjusts rewards based on storage value changes:
- Buying: storage increases → positive credit (reduces cost)
- Selling: storage decreases → negative credit (reduces profit)

```python
potential_reward = storage_change * price_factor * scale_factor
shaped_reward = base_reward + potential_reward
```

## CLI Arguments

### Hyperparameters
- `--alpha FLOAT` - Learning rate (default: 0.1)
- `--gamma FLOAT` - Discount factor (default: 0.9)
- `--epsilon FLOAT` - Initial exploration rate (default: 1.0)
- `--epsilon-decay FLOAT` - Decay per episode (default: 0.995)
- `--epsilon-min FLOAT` - Minimum epsilon (default: 0.01)
- `--episodes INT` - Training episodes (default: 500)

### Discretization
- `--n-storage INT` - Storage bins (default: 6)
- `--n-price INT` - Price bins (default: 6)
- `--price-window INT` - Price normalization window (default: 168)

### Reward Shaping
- `--reward-shaping` - Enable potential-based shaping
- `--potential-scale FLOAT` - Scale factor (default: 1.0), try 5-10
- `--use-avg-price` - Use 0.5 instead of current price

### Other
- `--seed INT` - Random seed
- `--label STR` - Experiment label
- `--config PATH` - Load from JSON
- `--output-dir PATH` - Results directory
- `--no-save` - Don't save results

## Examples

```bash
# Best config found
uv run qtable_feats_potential.py --reward-shaping --gamma 0.9 --potential-scale 10.0 --episodes 160 --seed 2

# Basic run
uv run qtable_feats_potential.py --reward-shaping --potential-scale 5.0

# Baseline (no shaping)
uv run qtable_feats_potential.py --label "baseline"
```

## Outputs

Results saved to `../results/qlearning/{timestamp}_{label}/`:
- `config.json`, `eval_metrics.json`, `training_metrics.json`
- `qtable.npz`, `visit_counts.npz`
- `*.png` diagnostic plots
