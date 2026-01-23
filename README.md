# PRL Team 11

## Setup (one time)

1. **Install uv** (copy-paste this):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Close and reopen your terminal after this.

2. **Remove old venv** (if you have one):
   ```bash
   cd PRL_team_11
   rm -rf venv .venv
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```
   That's it. This creates a `.venv` folder with everything installed.

## Running code

Use `uv run` before any python command:

```bash
uv run python dev/qtable.py
uv run jupyter notebook
```

## Using shared code in notebooks

In any notebook, you can import the shared environment:

```python
from src.environment import DamEnvGym, DamConfig
```

## Adding new packages

```bash
uv add package-name
```

This updates `pyproject.toml` and `uv.lock` automatically.

---

## Training the Q-Learning Agent

The main training script is `dev/qtable_feats.py`. Run it with:

```bash
uv run python dev/qtable_feats.py
```

This trains a Q-learning agent and saves results to `results/qlearning/`.

### Options

Customize training with command-line flags:

```bash
uv run python dev/qtable_feats.py --episodes 500 --seed 42 --label my_experiment
```

Common options:
- `--episodes N` — Number of training episodes (default: 200)
- `--seed N` — Random seed for reproducibility
- `--label NAME` — Name your experiment (shows up in folder name)
- `--alpha N` — Learning rate (default: 0.1)
- `--no-save` — Run without saving results

See all options with `--help`:
```bash
uv run python dev/qtable_feats.py --help
```

### Using a Config File

For reproducible experiments, use a JSON config:

```bash
uv run python dev/qtable_feats.py --config dev/configs/default.json
```

CLI flags override config file values, so you can do:
```bash
uv run python dev/qtable_feats.py --config dev/configs/default.json --episodes 1000
```

### Reward Shaping (optional)

Penalizes entering peak price periods with low storage (encourages saving for high-price selling).

```bash
uv run python dev/qtable_feats.py --reward-shaping
```

See `--help` for tuning options (`--peak-penalty`, `--low-storage-threshold`, `--peak-periods`)

---

## Visualizing Results

After training, visualize the Q-table with:

```bash
uv run python dev/visualize_qtable.py results/qlearning/<your_experiment_folder>/
```

This generates plots in a `plots/` subfolder showing:
- Policy heatmap (what action the agent takes in each state)
- Visit counts (how often each state was visited)
- Q-value heatmaps (learned values for each action)
- Training curves (rewards and epsilon over episodes)

### Options

View a specific slice (season, time of day, day type):
```bash
uv run python dev/visualize_qtable.py results/qlearning/<folder>/ --slice summer midday weekday
```

Generate all 40 slices:
```bash
uv run python dev/visualize_qtable.py results/qlearning/<folder>/ --all-slices
```

Show plots interactively instead of saving:
```bash
uv run python dev/visualize_qtable.py results/qlearning/<folder>/ --show
```
