# MyoSuite MJX (In development)

This directory contains MJX (MuJoCo XLA) implementations of MyoSuite environments for accelerated training.

## Installation
To install the MyoSuite MJX environments, you can use either `pip` or [`uv`](https://docs.astral.sh/uv/). The MJX environments are designed to work with GPU acceleration.

### For MJX environments (GPU-accelerated):
```bash
# With pip:
pip install -e ".[mjx]"

# With uv:
uv sync --extra mjx
```

### For standard MyoSuite (CPU):
```bash
# With pip:
pip install -e ".[standard]"
# With uv:
uv sync --extra standard
```

## Examples
Train jax ppo
```bash
uv run train_jax_ppo.py
```
Remember to initialize the submodules before running the examples:
```bash
git submodule update --init --recursive
```
