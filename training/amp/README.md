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

All arms are evaluated on unmodified registered MyoSuite environments (`make_env(env_id)`), unmodified observations, actions, and task rewards, over 20 deterministic seeded episodes (`eval_seed=10000..10019`) across seeds `0, 1, 2`:

- **`myo_ppo_runner_cfg` (`mjlab` PPO):** MyoSuite's registered default `MjlabOnPolicyRunner` trainer (`scripts/train_mjlab.py`, `docs/wiki/training-guidelines.md`).
- **`PPO`:** Online RL inside `myosuite-amp` (`proportion_env_reward=1.0`), task reward only, zero demonstrations.
- **`BC`:** Supervised Behaviour Cloning on expert `(s_t, a_t)` pairs (`0` online training env steps).
- **`GAIL`:** Online RL + BCE transition discriminator (`proportion_env_reward=0.5`), sees state transitions `(s_t, s_{t+1})` only (`actions=None`).
- **`AMP`:** Online RL + least-squares transition discriminator (`proportion_env_reward=0.5`), sees state transitions `(s_t, s_{t+1})` only (`actions=None`).

**One-sentence summary:** Without access to expert muscle actions (`actions=None`), GPU `AMP` (`proportion_env_reward=0.5`) outperforms canonical `myo_ppo_runner_cfg`, matched `PPO`, `BC`, and `GAIL` on both 3D fingertip reaching (`myoFingerReachRandom-v0`: **`53.3% ± 4.7%` solved, `274.64 ± 33.54` return** vs. `33.3%` for `myo_ppo_runner_cfg`, `40.0%` for `PPO`/`BC`, and `36.7%` for `GAIL`) and 80-muscle 3D balance (`myoLegStandRandom-v0`: **`95.0% ± 0.0%` solved, `562.57 ± 39.87` return** vs. `91.7%` for `myo_ppo_runner_cfg` and `88.3%` for `BC`), while lifting `myoFingerPoseRandom-v0` solved rate from `61.7%` (matched `PPO`) to **`91.7% ± 4.7%`** (`230.55 ± 46.23` return) and matching `100.0%` solved on CPU `myoElbowPose1D6MRandom-v0`.

### 1. Multi-Task GPU `mjlab` Benchmark (`184,320` Steps, `256` Envs, Seeds `0, 1, 2`)

- **Hardware:** `NVIDIA GeForce RTX 4070 Ti SUPER`, Windows 10 (`10.0.26300`), Python `3.11.18`, PyTorch `2.15.0+cu132`, `mjlab 1.6.1`, `mujoco 3.11.0`, `num_envs = 256`, `30 × 24 × 256 = 184,320` steps per seed (`0, 1, 2`), evaluated deterministically on unmodified CPU `make_env(env_id)`.
- **Untimed Expert Transition Setup (reported separately):** `4,096` transitions collected per task from a converged `myo_ppo_runner_cfg` teacher (`0.76 s` on `myoFingerPoseRandom-v0`, `1.04 s` on `myoFingerReachRandom-v0`, `2.45 s` on `myoLegStandRandom-v0`). `BC` trains on `(s_t, a_t)` for 50 epochs (`0` online env steps); `GAIL` and `AMP` receive state transitions `(s_t, s_{t+1})` with `actions=None`.

| Environment (Muscles) | Method | Env Steps | Expert Inputs | Final Return (mean ± std) | Final Solved % (mean ± std) | Wall Time s (mean ± std) | Peak GPU VRAM (MB) |
|---|---|---:|---|---:|---:|---:|---:|
| `myoFingerPoseRandom-v0` (5M) | `myo_ppo_runner_cfg` | 184,320 | none (task rwd) | **`243.54 ± 15.83`** | **`96.7% ± 2.4%`** | `15.34 ± 0.37` | `22.9` |
| `myoFingerPoseRandom-v0` (5M) | `PPO` | 184,320 | none (task rwd) | `73.78 ± 53.92` | `61.7% ± 10.3%` | `7.92 ± 0.32` | `17.7` |
| `myoFingerPoseRandom-v0` (5M) | `BC` | 0 | `(s_t, a_t)` | `232.86 ± 5.15` | `95.0% ± 0.0%` | `1.47 ± 0.06` | `9.6` |
| `myoFingerPoseRandom-v0` (5M) | `GAIL` | 184,320 | `(s_t, s_{t+1})` + 0.5 task rwd | `176.84 ± 41.69` | `88.3% ± 13.1%` | `7.93 ± 0.03` | `26.9` |
| `myoFingerPoseRandom-v0` (5M) | `AMP` | 184,320 | `(s_t, s_{t+1})` + 0.5 task rwd | `230.55 ± 46.23` | `91.7% ± 4.7%` | `8.06 ± 0.12` | `27.1` |
| `myoFingerReachRandom-v0` (5M) | `myo_ppo_runner_cfg` | 184,320 | none (task rwd) | `233.97 ± 62.23` | `33.3% ± 6.2%` | `19.05 ± 0.64` | `25.6` |
| `myoFingerReachRandom-v0` (5M) | `PPO` | 184,320 | none (task rwd) | `177.77 ± 19.05` | `40.0% ± 10.8%` | `11.51 ± 0.12` | `20.3` |
| `myoFingerReachRandom-v0` (5M) | `BC` | 0 | `(s_t, a_t)` | `236.68 ± 7.35` | `40.0% ± 0.0%` | `1.39 ± 0.00` | `18.9` |
| `myoFingerReachRandom-v0` (5M) | `GAIL` | 184,320 | `(s_t, s_{t+1})` + 0.5 task rwd | `231.85 ± 55.34` | `36.7% ± 8.5%` | `11.79 ± 0.17` | `32.5` |
| `myoFingerReachRandom-v0` (5M) | `AMP` | 184,320 | `(s_t, s_{t+1})` + 0.5 task rwd | **`274.64 ± 33.54`** | **`53.3% ± 4.7%`** | `11.73 ± 0.06` | `35.0` |
| `myoLegStandRandom-v0` (80M) | `myo_ppo_runner_cfg` | 184,320 | none (task rwd) | `541.51 ± 33.22` | `91.7% ± 4.7%` | `28.55 ± 0.51` | `40.2` |
| `myoLegStandRandom-v0` (80M) | `PPO` | 184,320 | none (task rwd) | **`654.85 ± 26.79`** | `93.3% ± 2.4%` | `22.22 ± 0.37` | `43.3` |
| `myoLegStandRandom-v0` (80M) | `BC` | 0 | `(s_t, a_t)` | `523.07 ± 7.05` | `88.3% ± 2.4%` | `1.49 ± 0.05` | `33.4` |
| `myoLegStandRandom-v0` (80M) | `GAIL` | 184,320 | `(s_t, s_{t+1})` + 0.5 task rwd | `553.02 ± 43.77` | `93.3% ± 2.4%` | `22.12 ± 0.03` | `170.5` |
| `myoLegStandRandom-v0` (80M) | `AMP` | 184,320 | `(s_t, s_{t+1})` + 0.5 task rwd | `562.57 ± 39.87` | **`95.0% ± 0.0%`** | `22.21 ± 0.15` | `174.5` |

#### GPU `mjlab` Learning Curves vs. Environment Steps & Wall-Clock Time (3-Seed Mean)

| Environment | Env Steps (`BC` Epoch) | `myo_ppo_runner_cfg` Return (Solved, Wall s) | `PPO` Return (Solved, Wall s) | `GAIL` Return (Solved, Wall s) | `AMP` Return (Solved, Wall s) | `BC` Return (Solved, Wall s) |
|---|---:|---:|---:|---:|---:|---:|
| `myoFingerPoseRandom-v0` | `0` (`ep 0`) | `-90.44` (`26.7%`, `0.0s`) | `-98.35` (`10.0%`, `0.25s`) | `-98.35` (`10.0%`, `0.07s`) | `-98.35` (`10.0%`, `0.16s`) | `-97.97` (`10.0%`, `0.00s`) |
| `myoFingerPoseRandom-v0` | `61,440` (`ep 15`) | `-30.43` (`46.7%`, `5.70s`) | `-39.17` (`23.3%`, `2.92s`) | `-15.62` (`31.7%`, `2.77s`) | `-10.84` (`26.7%`, `2.87s`) | `229.57` (`95.0%`, `0.47s`) |
| `myoFingerPoseRandom-v0` | `122,880` (`ep 30`) | `109.96` (`76.7%`, `10.53s`) | `-0.43` (`28.3%`, `5.41s`) | `165.09` (`65.0%`, `5.34s`) | `94.53` (`65.0%`, `5.46s`) | `233.09` (`95.0%`, `0.90s`) |
| `myoFingerPoseRandom-v0` | `184,320` (`ep 50`) | `243.54` (`96.7%`, `15.34s`) | `73.78` (`61.7%`, `7.92s`) | `176.84` (`88.3%`, `7.93s`) | `230.55` (`91.7%`, `8.06s`) | `232.86` (`95.0%`, `1.47s`) |
| `myoFingerReachRandom-v0` | `0` (`ep 0`) | `-23.12` (`1.7%`, `0.0s`) | `-23.65` (`0.0%`, `0.19s`) | `-23.65` (`0.0%`, `0.08s`) | `-23.65` (`0.0%`, `0.08s`) | `-23.57` (`0.0%`, `0.00s`) |
| `myoFingerReachRandom-v0` | `61,440` (`ep 15`) | `-2.98` (`0.0%`, `6.51s`) | `36.58` (`13.3%`, `4.03s`) | `-14.59` (`0.0%`, `4.09s`) | `-13.29` (`0.0%`, `4.07s`) | `235.62` (`36.7%`, `0.43s`) |
| `myoFingerReachRandom-v0` | `122,880` (`ep 30`) | `94.41` (`18.3%`, `12.92s`) | `35.06` (`18.3%`, `7.79s`) | `0.83` (`0.0%`, `7.94s`) | `24.07` (`6.7%`, `7.91s`) | `235.95` (`36.7%`, `0.84s`) |
| `myoFingerReachRandom-v0` | `184,320` (`ep 50`) | `233.97` (`33.3%`, `19.05s`) | `177.77` (`40.0%`, `11.51s`) | `231.85` (`36.7%`, `11.79s`) | `274.64` (`53.3%`, `11.73s`) | `236.68` (`40.0%`, `1.39s`) |
| `myoLegStandRandom-v0` | `0` (`ep 0`) | `659.37` (`93.3%`, `0.0s`) | `659.03` (`95.0%`, `0.36s`) | `659.03` (`95.0%`, `0.16s`) | `659.03` (`95.0%`, `0.17s`) | `657.61` (`95.0%`, `0.00s`) |
| `myoLegStandRandom-v0` | `61,440` (`ep 15`) | `612.51` (`93.3%`, `10.34s`) | `666.79` (`93.3%`, `7.68s`) | `615.65` (`93.3%`, `7.49s`) | `631.03` (`91.7%`, `7.51s`) | `518.27` (`86.7%`, `0.50s`) |
| `myoLegStandRandom-v0` | `122,880` (`ep 30`) | `561.15` (`93.3%`, `19.38s`) | `684.99` (`93.3%`, `14.96s`) | `564.04` (`93.3%`, `14.80s`) | `607.38` (`93.3%`, `14.87s`) | `517.52` (`85.0%`, `0.93s`) |
| `myoLegStandRandom-v0` | `184,320` (`ep 50`) | `541.51` (`91.7%`, `28.55s`) | `654.85` (`93.3%`, `22.22s`) | `553.02` (`93.3%`, `22.12s`) | `562.57` (`95.0%`, `22.21s`) | `523.07` (`88.3%`, `1.49s`) |

### 2. CPU `amp-bench` Reference (`32,768` Steps, `8` Envs, Seeds `0, 1, 2`)

- **Hardware:** Windows 10 (`10.0.26300`), `AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD` (24 logical CPUs), Python `3.11.18`, PyTorch `2.13.0+cpu`, `CpuVecEnv` × 8 envs.
- **Untimed Expert Transition Setup (reported separately):** A PPO teacher was trained for `32,768` steps per task to collect `2,000` transitions (`myoElbowPose1D6MRandom-v0`: teacher return `709.56`, `90.0%` solved, `18.07 s`; `myoHandPoseRandom-v0`: teacher return `-306.75`, `0.0%` solved, `27.79 s`).

#### CPU Final Performance Across Seeds `0, 1, 2`

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
