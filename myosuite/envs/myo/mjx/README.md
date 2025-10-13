# MyoSuite MJX (In development)

This directory contains MJX (MuJoCo XLA) implementations of MyoSuite environments for accelerated training.

## Installation

### Standard Installation
The default installation requires Python ≥3.9 and MuJoCo 3.3.0. See the [main README](../../../../README.md) for detailed installation instructions using uv, conda, or pip.

### MJX Installation (Python ≥3.10, MuJoCo 3.3.4):

1. Switch to python 3.10 and install MJX dependencies:
   ```bash
   # With uv:
   uv sync --extra mjx -p 3.10
   uv remove mujoco
   uv add "mujoco==3.3.6"
   uv add "mujoco-mjx==3.3.6" # use "mujoco-mjx[warp]" for warp support

   # With pip:
   pip install -e ".[mjx]"
   pip uninstall mujoco -y
   pip install "mujoco==3.3.6"
   pip install "mujoco-mjx==3.3.6" # use "mujoco-mjx[warp]" for warp support
   ```

NOTE: 
   - For [warp](https://github.com/google-deepmind/mujoco_warp) support, until it is integrated into the main mujoco release, you should depend on the warp tag: `mujoco-mjx[warp]`
   - For jax with cuda support, you could install `jax[cuda]` instead of `jax`, with `uv add "jax[cuda]>=0.4.20"` or `pip install "jax[cuda]>=0.4.20"`.


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
Remember to initialize the submodules with `uv run myoapi_init` before running the examples (see the [main README](../../../../README.md) for more details).
