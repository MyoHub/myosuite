# MyoSuite MJX and MJWarp

This directory contains MJX (MuJoCo XLA) and MJWarp implementations of MyoSuite environments for accelerated training.

## Installation

### Standard Installation
The default installation requires Python ≥3.9 and MuJoCo 3.3.6. See the [main README](../../../../README.md) for detailed installation instructions using uv, conda, or pip.

### Installation (Python ≥3.10, MuJoCo 3.5):

1. Switch to python 3.10 and install dependencies:
   ```bash
   # With uv:
   uv sync --extra mjx -p 3.10 # replace "mjx" with "mjx-cuda" for jax with cuda support

   # With pip:
   pip install -e ".[mjx]" # replace "mjx" with "mjx-cuda" for jax with cuda support
   ```

2. Apply patch to use MJWarp via the MJX API

To enable `naccdmax` support in MJX (see [MuJoCo PR #3096](https://www.google.com/search?q=https://github.com/google-deepmind/mujoco/pull/3096)), run the following command to overwrite the local `io.py` with the fixed version:

```bash
# Apply the surgical patch to the active environment
curl -fsSL https://raw.githubusercontent.com/google-deepmind/mujoco/refs/pull/3096/head/mjx/mujoco/mjx/_src/io.py \
-o $(python -c "import mujoco.mjx._src.io as io; print(io.__file__)")

```

3. **Verify installation**:
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

