# Hydroelectric Dam Trading Agent

Q-learning agent for energy arbitrage in a hydroelectric dam environment.

## Usage

```bash
pip install -r requirements.txt
```

```bash
python3 main.py --excel_file data/validate.xlsx
```

## Files

- `main.py` - Agent implementation
- `qtable_best.npz` - Trained Q-table
- `TestEnv.py` - Provided test environment

## Approach

Tabular Q-learning with discretized state space:
- Storage level (6 bins)
- Price (6 bins, rolling percentile normalization)
- Hour period (5 categories)
- Weekend/weekday
- Season

Actions: sell (-1), idle (0), buy (+1)

Trained with potential-based reward shaping for faster convergence.
