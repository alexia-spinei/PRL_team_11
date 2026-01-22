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
