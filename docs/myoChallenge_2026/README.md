# Saber Challenge — backend structure and usage guide

`myoChallengeSaberP0-v0` has two working backend paths:

- **CPU / Gymnasium** via `myosuite/envs/modular_env.py`
- **mjlab (split-scene, GPU-parallelised)** via `myosuite/envs/myo/backends/mjlab/`

There is **no MJX saber backend today** — the shared `TaskSpec` registers only `{"cpu"}` and the generic MJX path does not implement the saber target-pool runtime.

---

## Quick-start

### CPU env

```python
import gymnasium as gym
from myosuite import register_all_envs

register_all_envs()
env = gym.make("myoChallengeSaberP0-v0")
obs, _ = env.reset(seed=0)
print("obs shape:", obs.shape)   # (632,)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
```

### mjlab env (single-env play / evaluation)

```python
import torch
from myosuite.core.registry import make_env
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import bootstrap_myosuite_mjlab_registry

bootstrap_myosuite_mjlab_registry()  # registers all mjlab envs
env = make_env("myoChallengeSaberP0-v0", backend="mjlab", device="cuda", num_envs=128)  # or "cpu"
obs, _ = env.reset(seed=0)
policy_obs = obs["policy"]  # shape (128, 632)

for _ in range(10):
    action = torch.zeros(1, env.action_space.shape[-1])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated.any() or truncated.any():
        obs, _ = env.reset()

env.close()
```

### Training with mjlab (RSL-RL)

```python
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

task_id = "myoChallengeSaberP0-v0"
env_cfg = load_env_cfg(task_id, play=False)
env_cfg.scene.num_envs = 256           # parallelism for training
agent_cfg = load_rl_cfg(task_id)

train_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")
train_env = RslRlVecEnvWrapper(train_env, clip_actions=agent_cfg.clip_actions)
# pass train_env to your RSL-RL Runner
```

### Training with mjlab + MuscleMimic clip augmentation

```python
from mjlab.tasks.registry import register_mjlab_task
from myosuite.core.trajectory_io import MotionClip
from myosuite.envs.myo.backends.mjlab.register_mjlab_saber import (
    SaberMimicMixCfg,
    register_saber_p0_mjlab_task_with_mimic_mix,
)
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
    default_mimic_clip_on_policy_runner_cfg,
)

clip: MotionClip = ...   # load with load_motion_clip()
mimic_cfg = SaberMimicMixCfg(
    clips=(clip,),
    reward_mode="augmented",     # "mimic", "env", or "augmented"
    mimic_reward_weight=5.0,
    env_reward_weight=1.0,
    use_reference_state_initialization=True,
)
register_saber_p0_mjlab_task_with_mimic_mix(
    register_mjlab_task,
    default_mimic_clip_on_policy_runner_cfg,
    mimic_mix=mimic_cfg,
    # task_id defaults to "myoChallengeSaberP0Mimic-v0"
)
```

Or via the convenience wrapper that also registers bimanual/fullbody mimic envs:

```python
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
    register_mimic_mjlab_tasks_with_clip,
)

register_mimic_mjlab_tasks_with_clip(
    register_mjlab_task,
    default_mimic_clip_on_policy_runner_cfg,
    clip=clip,
    reward_mode="augmented",
)
# registers myoMimicBimanual-v0, myoMimicFullbody-v0, myoChallengeSaberP0Mimic-v0
```

### Play / render a trained policy (mjlab)

```python
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from myosuite.utils.html import show_video
from myosuite.utils.video_io import write_video
import torch

task_id = "myoChallengeSaberP0-v0"
env_cfg = load_env_cfg(task_id, play=True)
env_cfg.scene.num_envs = 1
play_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda", render_mode="rgb_array")
play_env = RslRlVecEnvWrapper(play_env)

policy = ...   # load your checkpoint
frames = []
for _ in range(1000):
    with torch.no_grad():
        obs = play_env.get_observations()
        actions = policy(obs)
        obs, reward, done, info = play_env.step(actions)
        frame = play_env.unwrapped.render()
        frames.append(frame)
    if done.any():
        play_env.reset()

write_video("test.mp4", frames, fps=int(1/play_env.unwrapped._saber_logic.task_cfg.backend.ctrl_dt))
show_video("test.mp4")
```

---

## Training scripts

Three ready-to-run training scripts live in `tutorials/mc26/`.

### `mc26_train_saber_cpu.py` — CPU/SB3 baseline

Trains `myoChallengeSaberP0-v0` on CPU with Stable-Baselines3 PPO.
Checkpoints are saved as ONNX bundles (`.onnx` files with an embedded `.zip` SB3 payload).

```bash
# Quick smoke-test (200 k steps, 4 envs)
python tutorials/mc26/mc26_train_saber_cpu.py

# Longer run with explicit options
python tutorials/mc26/mc26_train_saber_cpu.py \
    --total-timesteps 5_000_000 \
    --n-envs 16 \
    --save-freq 100_000 \
    --log-root logs/sb3_ppo/saber_cpu \
    --run-name my_run

# Resume from a previous ONNX checkpoint
python tutorials/mc26/mc26_train_saber_cpu.py \
    --resume-from logs/sb3_ppo/saber_cpu/my_run/model_200000.onnx
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--total-timesteps` | `200_000` | Total environment steps |
| `--n-envs` | `4` | Parallel CPU envs |
| `--n-steps` | `2048` | Rollout length per env |
| `--batch-size` | `256` | PPO mini-batch size |
| `--learning-rate` | `3e-4` | Adam learning rate |
| `--save-freq` | `50_000` | ONNX checkpoint interval (steps) |
| `--resume-from` | — | Path to an existing `.onnx` bundle |
| `--log-root` | `logs/sb3_ppo/saber_cpu` | Output directory |

---

### `mc26_train_saber_mjlab.py` — mjlab/RSL-RL GPU training

Trains the mjlab saber env with RSL-RL PPO across many parallel envs on GPU.
Checkpoints are saved as ONNX bundles under `logs/rsl_rl/saber_mjlab/<run>/`.

```bash
# Default run (256 envs, 1000 iterations)
python tutorials/mc26/mc26_train_saber_mjlab.py

# Scale down
python tutorials/mc26/mc26_train_saber_mjlab.py \
    --num-envs 128 \
    --max-iterations 3000 \
    --save-interval 150 \
    --device cuda:0 \
    --run-name my_mjlab_run

# Resume from checkpoint
python tutorials/mc26/mc26_train_saber_mjlab.py \
    --resume-from logs/rsl_rl/saber_mjlab/my_mjlab_run/model_450.onnx

# Add MuscleMimic clip augmentation (augmented reward by default)
python tutorials/mc26/mc26_train_saber_mjlab.py \
    --mimic-clip path/to/motion_clip.npz \
    --mimic-reward-mode augmented \
    --mimic-reward-weight 5.0 \
    --env-reward-weight 1.0
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--num-envs` | `256` | Parallel mjlab envs |
| `--num-steps-per-env` | `24` | Rollout horizon per env |
| `--max-iterations` | `1000` | PPO iterations |
| `--save-interval` | `150` | ONNX checkpoint interval (iterations) |
| `--device` | `auto` | `cuda:0` if available, else `cpu` |
| `--resume-from` | — | Path to an existing `.onnx` bundle |
| `--mimic-clip` | — | Optional motion clip; switches to `myoChallengeSaberP0Mimic-v0` |
| `--mimic-reward-mode` | `augmented` | `mimic` / `env` / `augmented` |
| `--mimic-reward-weight` | `5.0` | Scale for clip-mimic reward |
| `--env-reward-weight` | `1.0` | Scale for native saber reward |
| `--log-root` | `logs/rsl_rl/saber_mjlab` | Output directory |

---

### `mc26_train_saber_mjlab_mimic_init.py` — mjlab PPO warm-started from the full-body mimic checkpoint

Initialises the RSL-RL actor **and** critic from the full-body MuscleMimic Orbax
checkpoint before fine-tuning with PPO on the native saber task.
The policy uses a `LayerNormSiLUMLP` architecture that matches the checkpoint layout
exactly, so weights load without conversion.

```bash
# Default: loads checkpoint from Boxing/checkpoint_13114
python tutorials/mc26/mc26_train_saber_mjlab_mimic_init.py --skip-sanity

# Full run with sanity rollouts enabled (3 episodes ≥ 950 steps each)
python tutorials/mc26/mc26_train_saber_mjlab_mimic_init.py \
    --max-iterations 3000 \
    --num-envs 256 \
    --device cuda:0 \
    --run-name mimic_warmstart

# Point at a different Orbax checkpoint and motion clip
python tutorials/mc26/mc26_train_saber_mjlab_mimic_init.py \
    --mimic-checkpoint path/to/orbax_checkpoint_root \
    --reference-motion path/to/motion_clip.npz \
    --skip-sanity

# Resume from a previous native saber checkpoint (bypasses mimic import)
python tutorials/mc26/mc26_train_saber_mjlab_mimic_init.py \
    --resume-from logs/rsl_rl/saber_mjlab_mimic_init/my_run/model_150.onnx \
    --skip-sanity

# Tune PPO hyperparameters for fine-tuning
python tutorials/mc26/mc26_train_saber_mjlab_mimic_init.py \
    --actor-init-std 0.02 \
    --learning-rate 1e-4 \
    --num-learning-epochs 2 \
    --clip-param 0.1 \
    --desired-kl 0.005 \
    --skip-sanity
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--num-envs` | `256` | Parallel mjlab envs |
| `--max-iterations` | `1000` | PPO iterations |
| `--save-interval` | `150` | ONNX checkpoint interval (iterations) |
| `--device` | `auto` | `cuda:0` if available, else `cpu` |
| `--mimic-checkpoint` | `Boxing/checkpoint_13114` | Orbax checkpoint root |
| `--reference-motion` | `Boxing/motions/…/punchboxing_push_poses.npz` | Reference motion clip |
| `--actor-init-std` | `0.05` | Initial Gaussian std (low = near-deterministic start) |
| `--learn-actor-std` | off | If set, PPO adapts the exploration std |
| `--learning-rate` | `3e-4` | PPO Adam learning rate |
| `--num-learning-epochs` | `1` | PPO update epochs per iteration |
| `--clip-param` | `0.2` | PPO clip ε |
| `--entropy-coef` | `0.01` | Entropy regularisation coefficient |
| `--desired-kl` | `0.01` | Adaptive-KL target |
| `--resume-from` | — | Skip mimic import; resume a native saber `.onnx` bundle |
| `--skip-sanity` | off | Skip the post-init deterministic sanity rollouts |
| `--sanity-episodes` | `3` | Number of sanity episodes to run |
| `--sanity-min-length` | `950` | Minimum acceptable episode length |
| `--log-root` | `logs/rsl_rl/saber_mjlab_mimic_init` | Output directory |

---

## Entry points (current)

| Surface | File | Role |
|---|---|---|
| Env IDs | `myosuite/envs/myo/tasks/challenge/saber_task_spec.py` | `SABER_P0_ENV_ID`, `SABER_P0_MIMIC_ENV_ID`, CPU registration |
| Task config | `myosuite/envs/myo/tasks/challenge/saber_task_config.py` | `SaberP0Task`, `SaberP0Cfg`, `saber_term_params()` |
| Scene assets | `myosuite/envs/myo/tasks/challenge/saber_scene_assets.py` | `build_saber_scene_spec()`, `saber_scene_callable()`, XML builders |
| CPU env runtime | `myosuite/envs/modular_env.py` | Saber target-pool hooks wired from `task_config.saber_cfg` |
| mjlab env runtime | `myosuite/envs/myo/backends/mjlab/saber_mjlab_env.py` | `SaberTaskLogicState`, obs/reward/event wrappers |
| mjlab registration | `myosuite/envs/myo/backends/mjlab/register_mjlab_saber.py` | `register_saber_p0_mjlab_task`, `register_saber_p0_mjlab_task_with_mimic_mix` |
| Target-pool state machine | `myosuite/envs/myo/tasks/challenge/saber_vs/saber_target_pool.py` | CPU NumPy runtime; shared constants |
| Shared obs terms | `myosuite/terms/base_obs.py` | `saber_target_pool_obs`, `health_status_obs`, `joint_pos_obs`, … |
| Shared reward terms | `myosuite/terms/base_reward.py` | `saber_target_pool_reward`, `upright_posture_reward`, … |
| Shared termination | `myosuite/terms/base_termination.py` | `saber_pool_done`, `upright_posture_failure` |
| Dynamics parity notes | `docs/saber_backend_alignment/KNOWN_DIFF.md` | Known CPU ↔ mjlab divergence, tolerances, ContactSensor assessment |

---

## Shared vs backend-specific code

### Shared across CPU and mjlab (as of latest PRs)

- **Scene geometry** — both backends build from the same XML asset files via `build_saber_scene_spec()` / `saber_scene_callable()`.
- **Task config and saber cfg** — `SaberP0Task` / `SaberP0Cfg` are authoritative; both `modular_env.py` and `register_mjlab_saber.py` read directly from `task_config.saber_cfg`.
- **Term functions** — all obs/reward/termination term functions live in `base_obs.py`, `base_reward.py`, `base_termination.py`. Both backends call the same functions (the mjlab path uses `_saber_obs_from_shared` / `_saber_reward_dense_from_shared` adapter wrappers).
- **Term parameter mapping** — `saber_term_params(cfg)` in `saber_task_config.py` is the single source of truth for per-term kwargs; used by both `modular_env.py` and `register_mjlab_saber.py`.
- **Target-pool constants / config dataclass** — `saber_target_pool.py`.
- **Keyframe reset / saber snap-to-grasp** — `_saber_keyframe_reset_event()` in `saber_mjlab_env.py` mirrors `ModularTaskEnv._snap_free_sabers_to_grasp_sites()` using the same mount quaternion and grasp-site logic.

### Backend-specific

| Concern | CPU (`modular_env.py`) | mjlab (`saber_mjlab_env.py`) |
|---|---|---|
| Target-pool step | `pool_pre_physics()` / `pool_post_physics()` (NumPy loops) | `SaberTaskLogicState.prepare_control_step()` / `ensure_post_step()` (vectorised Torch) |
| Contact detection | `pool_post_physics()` contact iteration | `_find_body_contact_slots()` via `wp_data.contact` tensors |
| Scene structure | Single `MjModel` via `saber_scene_callable` | Split-scene entities (robot + left saber + right saber + 100 target entities) |
| Action term | `muscle_normalize_action` | `SaberMuscleActionCfg` (also calls `prepare_control_step`) |
| MuscleMimic augmentation | N/A | `register_saber_p0_mjlab_task_with_mimic_mix` + `SaberMimicMixCfg` |

---

## Open items

- **GPU rebaseline** — `test_myo_challenge_saber_mjlab_matches_cpu_control_contract` passes on CPU device. The tier-2 parity suite (`test_myo_saber_mjlab_parity.py`) needs a CUDA+mjlab run to fill in the observed divergence table in `KNOWN_DIFF.md`.
- **MJX backend** — not planned; would require a full saber target-pool runtime for MJX.
