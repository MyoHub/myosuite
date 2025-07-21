# MyoSuite MJX (In development)

This directory contains MJX (MuJoCo XLA) implementations of MyoSuite environments for accelerated training.

## Installation

### Standard Installation (Python ≥3.9, MuJoCo 3.3.0):
```bash
# With uv:
uv sync -p 3.9

# With pip:
pip install -e .
```

### MJX Installation (Python ≥3.10, MuJoCo 3.3.4):

1. Switch to python 3.10 and install MJX dependencies:
   ```bash
   # With uv:
   uv sync --extra mjx -p 3.10
   uv remove mujoco
   uv add "mujoco==3.3.4"
   uv add "mujoco-mjx==3.3.4"

   # With pip:
   pip install -e ".[mjx]"
   pip uninstall mujoco -y
   pip install "mujoco==3.3.4"
   pip install "mujoco-mjx==3.3.4"
   ```

2. **Verify installation**:
   ```bash
   # remove uv run if you installed with pypi

   uv run python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
   uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"
   ```

## Examples
Train JAX PPO with:
```bash
uv run train_jax_ppo.py
```
Remember to initialize the submodules before running the examples:
```bash
git submodule update --init --recursive
```
