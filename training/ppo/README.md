# MyoSuite PPO Reference Trainers (`myosuite-ppo`)

Standalone PPO training package (`training/ppo/`) providing MyoSuite's reference on-policy trainers outside the installable `myosuite` runtime wheel (see [`docs/wiki/training-guidelines.md`](../../docs/wiki/training-guidelines.md)):

- **CPU trainer (`myosuite_ppo.sb3_train`)** — Stable-Baselines3 (`PPO`, `SAC`, `TD3`) over `myosuite.make_env`, `VecNormalize`, `OnnxCheckpointCallback`, and all-env sweep utilities (`sb3_sweep`).
- **GPU trainer (`myosuite_ppo.mjlab_train`)** — RSL-RL (`MjlabOnPolicyRunner`) over `mjlab` parallel environments with muscle-command `--feature` flags, W&B/local checkpoint resume, and deterministic early-stop (`--stop-on-success`).
- **Presets (`myosuite_ppo.presets`)** — Canonical 4-task benchmark list (`FOUR_BENCHMARK_TASKS`) and CPU `WinnerPpoConfig` (`mjlab` tasks register their own `myo_ppo_runner_cfg` defaults via `mjlab.tasks.registry.load_rl_cfg`).

Top-level CLI wrappers [`scripts/train_sb3.py`](../../scripts/train_sb3.py) and [`scripts/train_mjlab.py`](../../scripts/train_mjlab.py) remain in place and resolve `training/ppo/` automatically from a source checkout (or delegate to an installed `myosuite-ppo` package).

## Install

From a local MyoSuite checkout:

```bash
# CPU PPO (Stable-Baselines3; --no-deps preserves the local editable myosuite checkout)
pip install -e ".[rl]"
pip install --no-deps -e training/ppo

# GPU PPO (mjlab + RSL-RL)
pip install -e ".[mjlab]"
pip install --no-deps -e training/ppo
```

Or from Git without a local checkout (until the next PyPI release shipping `myosuite.policies`, install `myosuite` from the matching Git revision):

```bash
# CPU PPO (Stable-Baselines3)
pip install "myosuite[rl] @ git+https://github.com/MyoHub/myosuite.git@dev"
pip install "myosuite-ppo[cpu] @ git+https://github.com/MyoHub/myosuite.git@dev#subdirectory=training/ppo"

# GPU PPO (mjlab + RSL-RL)
pip install "myosuite[mjlab] @ git+https://github.com/MyoHub/myosuite.git@dev"
pip install "myosuite-ppo[mjlab] @ git+https://github.com/MyoHub/myosuite.git@dev#subdirectory=training/ppo"
```

## Usage

```bash
# Unified CLI: CPU Stable-Baselines3 PPO
myosuite-ppo sb3 myoElbowPose1D6MRandom-v0 --timesteps 500000 --normalize

# Unified CLI: GPU mjlab RSL-RL PPO
myosuite-ppo mjlab myoElbowPose1D6MRandom-v0 --env.scene.num-envs 1024

# Source-checkout script entry points (equivalent)
python scripts/train_sb3.py myoElbowPose1D6MRandom-v0 --timesteps 500000
python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0 --env.scene.num-envs 1024
```

## Tests

```bash
pytest training/ppo/tests
```
