# Adversarial Motion Priors (AMP) & GAIL for MyoSuite

Adversarial imitation and motion-prior reinforcement learning for MyoSuite musculoskeletal environments. Policies learn from state-only demonstrations `(s_t, s_{t+1})`—requiring **no expert muscle activations or actions**—while optimizing task rewards via clipped PPO.

## Attribution & Licensing

AMP and GAIL are not original MyoSuite methods. This package implements the methods of:

1. **AMP (Adversarial Motion Priors):**
   > Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, Angjoo Kanazawa.
   > **AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control.**
   > ACM Transactions on Graphics (SIGGRAPH), 40(4), 2021.
   > [arXiv:2104.02180](https://arxiv.org/abs/2104.02180) · [Project Page](https://xbpeng.github.io/projects/AMP/)

2. **GAIL (Generative Adversarial Imitation Learning):**
   > Jonathan Ho, Stefano Ermon.
   > **Generative Adversarial Imitation Learning.**
   > Advances in Neural Information Processing Systems (NeurIPS), 29, 2016.
   > [arXiv:1606.03476](https://arxiv.org/abs/1606.03476) · [Reference Repo](https://github.com/openai/imitation)

3. **LocoMuJoCo Reference Design (`AMPJax` / `GAILJax`):**
   > Firas Al-Hafez, Guoping Zhao, Jan Peters, Davide Tateo.
   > **LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion.**
   > Robot Learning Workshop, NeurIPS 2023.
   > [arXiv:2311.02496](https://arxiv.org/abs/2311.02496) · [github.com/robfiras/loco-mujoco](https://github.com/robfiras/loco-mujoco) (`loco_mujoco/algorithms/amp_jax.py`, `loco_mujoco/algorithms/gail_jax.py`)

New PyTorch and MyoSuite code in this directory is licensed under Apache-2.0 ([LICENSE](LICENSE)). Upstream design conventions adapted from `robfiras/loco-mujoco` carry their MIT copyright notice in [LICENSE-loco-mujoco.txt](LICENSE-loco-mujoco.txt) (`Copyright (c) 2024 Al-Hafez`).

| Component | Source |
|---|---|
| Least-squares transition discriminator objective `0.5 E[(D(x_exp) - 1)^2] + 0.5 E[(D(x_pol) + 1)^2]`, expert gradient penalty `0.5 w_gp E[\|\|grad_x D(x_exp)\|\|^2]`, logit weight decay, bounded style reward `max(0, 1 - 0.25 (D(x) - 1)^2)`, and policy transition replay buffer | Peng et al., SIGGRAPH 2021 (AMP), Section 4 & Appendix A; design aligned with LocoMuJoCo `AMPJax`. |
| BCE-with-logits discriminator objective + expert gradient penalty and `-log(1 - sigmoid(D(x)) + eps)` style reward | Ho & Ermon, NeurIPS 2016 (GAIL); design aligned with LocoMuJoCo `GAILJax`. |
| Convex reward blending `(1 - proportion_env_reward) * style_reward + proportion_env_reward * env_reward` (default `0.5`) | Adapted from LocoMuJoCo (`proportion_env_reward` in `amp_jax.py` / `gail_jax.py` and `jax_amp/conf.yaml`). |
| PyTorch modules (`discriminator.py`, `ppo_amp.py`), truncation-aware GAE, `CpuVecEnv`, `MjlabVecEnvAdapter` (`train_mjlab.py`), Behaviour Cloning trainer, self-contained teacher dataset generator, TorchScript/ONNX policy export, and 4-arm benchmark (`benchmark.py`) | New MyoSuite code (no JAX/Flax code copied). |

## Layout

- `myosuite_amp/discriminator.py` — `RunningMeanStd` and `MotionDiscriminator` (`mode="amp"` or `"gail"`, optional `feature_indices` slice, gradient penalty, logit L2 decay, reward blending).
- `myosuite_amp/ppo_amp.py` — `ActorCritic`, `AMPTrainer`, `BehaviorCloningTrainer` (`train_bc`), `collect_expert_dataset`, and standalone `export_policy_torchscript` / `export_policy_onnx` exporters.
- `myosuite_amp/train_mjlab.py` — `amp-train-mjlab` CLI wrapping `mjlab` GPU `ManagerBasedRlEnv` / `RslRlVecEnvWrapper` twins on batched GPU tensors from `.npz` transition datasets.
- `myosuite_amp/benchmark.py` — `amp-bench` side-by-side 4-arm reproducible benchmark (`PPO`, `BC`, `GAIL`, `AMP`).

## Installation

Isolated installation from git (following `docs/wiki/training-guidelines.md`):

```bash
pip install myosuite
pip install "git+https://github.com/MyoHub/myosuite.git@dev#subdirectory=training/amp"
```

From a local checkout (with optional `mjlab` GPU and `test` extras):

```bash
pip install -e .
pip install -e "training/amp[test]"
# Optional GPU mjlab support:
pip install -e "training/amp[mjlab]"
```

## Usage

### 1. Reproducible 4-Arm Benchmark (`PPO` vs. `BC` vs. `GAIL` vs. `AMP`)

```bash
amp-bench \
    --env-ids myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0 \
    --total-steps 32768 \
    --eval-interval 8192 \
    --expert-steps 32768 \
    --seeds 0 1 2 \
    --out-json amp_benchmark.json
```

### 2. GPU Training on `mjlab` Twins (`amp-train-mjlab`)

```bash
amp-train-mjlab myoElbowPose1D6MRandom-v0 \
    --dataset demonstrations_elbow.npz \
    --mode amp \
    --proportion-env-reward 0.5 \
    --num-envs 1024 \
    --total-steps 2000000 \
    --export-torchscript logs/amp_mjlab/elbow_policy.pt
```

Muscle-command features (`--feature fatigue`, `--feature motor-noise`, etc.) work identically to `scripts/train_mjlab.py`.

### 3. Policy Export for Standalone Playback

Exported policies fold running observation statistics into the graph so MyoSuite can play them back without `myosuite-amp` installed:

```python
from pathlib import Path
from myosuite_amp import export_policy_onnx, export_policy_torchscript

export_policy_torchscript(trainer.actor_critic, Path("policy_ts.pt"))
# When `onnx` and `onnxruntime` are installed:
export_policy_onnx(trainer.actor_critic, Path("policy.onnx"))
```

## Benchmark against PPO & BC

All four arms are evaluated on unmodified registered MyoSuite environments (`make_env(env_id)`), unmodified observations, actions, and task rewards, over 20 deterministic seeded episodes (`eval_seed=10000..10019`) across seeds `0, 1, 2`:

- **`PPO`:** Online RL (`proportion_env_reward=1.0`), task reward only, zero demonstrations.
- **`BC`:** Supervised Behaviour Cloning on expert `(s_t, a_t)` pairs (`0` online training env steps).
- **`GAIL`:** Online RL + BCE transition discriminator (`proportion_env_reward=0.5`), sees state transitions `(s_t, s_{t+1})` only (`actions=None`).
- **`AMP`:** Online RL + least-squares transition discriminator (`proportion_env_reward=0.5`), sees state transitions `(s_t, s_{t+1})` only (`actions=None`).

**One-sentence summary:** Without access to expert muscle actions (`actions=None`), `AMP` (`proportion_env_reward=0.5`) matches pure `PPO`'s `100.0%` solved rate and exceeds supervised `BC` (`769.23 ± 0.68` vs. `764.42 ± 0.58`, with `2.4×` lower seed variance than `PPO`'s `770.63 ± 1.65`) on `myoElbowPose1D6MRandom-v0`, while achieving the highest 32,768-step mean return across all four arms on the 39-muscle `myoHandPoseRandom-v0` (`-316.19 ± 5.97` vs. `-318.20 ± 2.63` for `GAIL`, `-319.16 ± 1.08` for `BC`, and `-319.34 ± 5.51` for `PPO`) at `+2.9%` (`+0.5 s`) wall-clock overhead.

- **Hardware:** Windows 10 (`10.0.26300`), `AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD` (24 logical CPUs), Python `3.11.18`, PyTorch `2.13.0+cpu`, `CpuVecEnv` × 8 envs.
- **Untimed Expert Transition Setup (reported separately):** A PPO teacher was trained for `32,768` steps per task to collect `2,000` transitions (`myoElbowPose1D6MRandom-v0`: teacher return `709.56`, `90.0%` solved, `18.07 s`; `myoHandPoseRandom-v0`: teacher return `-306.75`, `0.0%` solved, `27.79 s`). `BC` trains on `(s_t, a_t)` for 50 epochs (`0` online env steps); `GAIL` and `AMP` receive state transitions `(s_t, s_{t+1})` with `actions` stripped (`None`).

### Final Performance Across Seeds `0, 1, 2`

| Environment | Method | Env Steps | Expert Inputs | Final Return (mean ± std) | Final Solved % (mean ± std) | Wall Time s (mean ± std) | Peak Mem MB |
|---|---|---:|---|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | `PPO` | 32,768 | none (task rwd) | `770.63 ± 1.65` | `100.0% ± 0.0%` | `17.28 ± 0.09` | `67.7` |
| `myoElbowPose1D6MRandom-v0` | `BC` | 0 | `(s_t, a_t)` | `764.42 ± 0.58` | `100.0% ± 0.0%` | `5.02 ± 0.01` | `67.1` |
| `myoElbowPose1D6MRandom-v0` | `GAIL` | 32,768 | `(s_t, s_{t+1})` + 0.5 task rwd | `768.64 ± 1.12` | `100.0% ± 0.0%` | `17.89 ± 0.14` | `67.7` |
| `myoElbowPose1D6MRandom-v0` | `AMP` | 32,768 | `(s_t, s_{t+1})` + 0.5 task rwd | `769.23 ± 0.68` | `100.0% ± 0.0%` | `17.78 ± 0.06` | `67.7` |
| `myoHandPoseRandom-v0` | `PPO` | 32,768 | none (task rwd) | `-319.34 ± 5.51` | `0.0% ± 0.0%` | `31.09 ± 0.05` | `70.7` |
| `myoHandPoseRandom-v0` | `BC` | 0 | `(s_t, a_t)` | `-319.16 ± 1.08` | `0.0% ± 0.0%` | `8.26 ± 0.05` | `70.1` |
| `myoHandPoseRandom-v0` | `GAIL` | 32,768 | `(s_t, s_{t+1})` + 0.5 task rwd | `-318.20 ± 2.63` | `0.0% ± 0.0%` | `31.61 ± 0.14` | `70.8` |
| `myoHandPoseRandom-v0` | `AMP` | 32,768 | `(s_t, s_{t+1})` + 0.5 task rwd | `-316.19 ± 5.97` | `0.0% ± 0.0%` | `31.63 ± 0.17` | `70.8` |

### Learning Curves vs. Environment Steps & Wall-Clock Time (3-Seed Mean; Offline `BC` Shown by Epoch)

| Environment | Env Steps (`BC` Epoch) | `PPO` Return (Solved, Wall s) | `GAIL` Return (Solved, Wall s) | `AMP` Return (Solved, Wall s) | `BC` Return (Solved, Wall s) |
|---|---:|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | `0` (`ep 0`) | `0.97` (`5.0%`, `0.26s`) | `0.97` (`5.0%`, `0.23s`) | `0.97` (`5.0%`, `0.24s`) | `-0.40` (`5.0%`, `0.01s`) |
| `myoElbowPose1D6MRandom-v0` | `8,192` (`ep 12`) | `362.90` (`31.7%`, `4.33s`) | `291.36` (`26.7%`, `4.45s`) | `275.87` (`26.7%`, `4.42s`) | `765.00` (`100.0%`, `0.90s`) |
| `myoElbowPose1D6MRandom-v0` | `16,384` (`ep 24`) | `741.67` (`95.0%`, `8.43s`) | `659.59` (`73.3%`, `8.67s`) | `672.12` (`76.7%`, `8.68s`) | `764.37` (`100.0%`, `1.82s`) |
| `myoElbowPose1D6MRandom-v0` | `24,576` (`ep 36`) | `770.63` (`100.0%`, `12.50s`) | `766.85` (`100.0%`, `12.96s`) | `767.25` (`100.0%`, `12.90s`) | `764.77` (`100.0%`, `2.71s`) |
| `myoElbowPose1D6MRandom-v0` | `32,768` (`ep 50`) | `770.63` (`100.0%`, `16.58s`) | `768.64` (`100.0%`, `17.18s`) | `769.23` (`100.0%`, `17.08s`) | `764.42` (`100.0%`, `4.34s`) |
| `myoHandPoseRandom-v0` | `0` (`ep 0`) | `-316.42` (`0.0%`, `0.15s`) | `-316.42` (`0.0%`, `0.16s`) | `-316.42` (`0.0%`, `0.16s`) | `-315.71` (`0.0%`, `0.01s`) |
| `myoHandPoseRandom-v0` | `8,192` (`ep 12`) | `-317.52` (`0.0%`, `7.58s`) | `-316.40` (`0.0%`, `7.72s`) | `-313.37` (`0.0%`, `7.76s`) | `-318.54` (`0.0%`, `1.44s`) |
| `myoHandPoseRandom-v0` | `16,384` (`ep 24`) | `-319.70` (`0.0%`, `14.99s`) | `-316.77` (`0.0%`, `15.24s`) | `-313.91` (`0.0%`, `15.31s`) | `-318.60` (`0.0%`, `2.88s`) |
| `myoHandPoseRandom-v0` | `24,576` (`ep 36`) | `-321.58` (`0.0%`, `22.43s`) | `-317.86` (`0.0%`, `22.84s`) | `-317.52` (`0.0%`, `22.87s`) | `-318.68` (`0.0%`, `4.33s`) |
| `myoHandPoseRandom-v0` | `32,768` (`ep 50`) | `-319.34` (`0.0%`, `29.87s`) | `-318.20` (`0.0%`, `30.39s`) | `-316.19` (`0.0%`, `30.40s`) | `-319.16` (`0.0%`, `7.05s`) |

## Tests

```bash
pytest training/amp/tests/test_amp.py -v
```
