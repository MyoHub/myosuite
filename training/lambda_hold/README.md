# Lambda-Hold Equilibrium-Point Stretch-Reflex Control for MyoSuite

PPO training on registered MyoSuite tasks through the `LambdaHold` equilibrium-point (EP) stretch-reflex action map (`myosuite.utils.action_map.LambdaHold`). Instead of commanding raw muscle excitations directly, the policy outputs normalized per-muscle threshold lengths $z_i \in [-1, 1]$ mapped around each actuator's physiological `lengthrange`, and a stretch-reflex recruitment law computes excitations from live actuator length and velocity with optional intermittent phase holding (`hold_steps`) and curriculum annealing (`hold_steps -> end_hold_steps`, `reflex_weight -> end_reflex_weight`).

## Attribution & Licensing

`LambdaHold` is adapted from:

> Jun-Hyuk Lee et al., **λ-Hold Control** (2026). [arXiv:2608.17030](https://arxiv.org/abs/2608.17030) · upstream repository: [github.com/Lee-Jun-Hyuk-37/Lambda-Hold](https://github.com/Lee-Jun-Hyuk-37/Lambda-Hold) (MIT License, [doi:10.5281/zenodo.23265019](https://doi.org/10.5281/zenodo.23265019)).

This is not an original MyoSuite method; it is a clean-room MyoSuite re-implementation adapted for MuJoCo and `mjlab` from the paper's equilibrium-point stretch-reflex formulation (upstream targets SCONE/Hyfydy). Dimensionless excursion normalization by each actuator's `actuator_lengthrange` span, exact inversion of MyoSuite's `muscle_sigmoid` (`normalize_act=True`), and curriculum annealing from held stretch-reflex stabilization to single-step muscle control are MyoSuite-specific additions spanning finger, arm, leg, and torso models across both CPU and `mjlab`.

## Layout

- **Playback & wrappers (`myosuite/`):** `LambdaHold` (`myosuite/utils/action_map.py`) and `ActionMapAction` (`myosuite/envs/myo/backends/mjlab/tasks/mdp/action_map.py`). Saved checkpoints store the `LambdaHold` manifest and `env_steps` in `infos`, so `scripts/eval_mjlab_policy.py` and `ActionMapWrapper` play them without importing `training/lambda_hold`.
- **Trainers & benchmark (`training/lambda_hold/`):**
  - `myosuite_lambda_hold/train.py`: GPU `mjlab` / RSL-RL PPO trainer (`lambda-hold-train`).
  - `myosuite_lambda_hold/benchmark_sb3.py`: reproducible CPU `stable-baselines3` PPO vs. `LambdaHold+PPO` vs. `CurriculumLambdaHold+PPO` side-by-side benchmark (`lambda-hold-bench-sb3`).

## Install & Usage

```bash
pip install -e ".[rl]"          # or pip install -e ".[mjlab]" for GPU RSL-RL training
pip install -e training/lambda_hold

# Train on mjlab (GPU / RSL-RL) with hold_steps: 2 -> 1 & reflex_weight: 1.0 -> 0.0 curriculum
lambda-hold-train myoLegStandRandom-v0 --env.scene.num-envs 256 --agent.max-iterations 30 \
  --kp 2.5 --damping-s 0.04 --hold-steps 2 --end-hold-steps 1 \
  --start-steps 46080 --ramp-steps 92160 --end-reflex-weight 0.0

# Canonical registered PPO reference (same budget: 30 * 256 * 24 = 184,320 steps)
python scripts/train_mjlab.py myoLegStandRandom-v0 --env.scene.num-envs 256 --agent.max-iterations 30

# Reproduce the 3-seed CPU SB3 PPO comparison
lambda-hold-bench-sb3 --envs myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0 --seeds 0 1 2 --steps 65536
```

## Benchmark Against Registered `myo_ppo_runner_cfg` (`mjlab` PPO) & SB3 PPO

**One-sentence summary:** Across seeds `0, 1, 2` at a matched `184,320`-step GPU `mjlab` budget (`num_envs = 256`), `LambdaHold` and `CurriculumLambdaHold` solve non-elbow finger posture, 3D reach, 80-muscle bipedal standing, and 216-muscle spinal posture tasks: fixed `LambdaHold+PPO` reaches **`100.0% ± 0.0%`** solved on `myoFingerPoseRandom-v0` and **`+239.20 ± 148.20`** return (**`100.0%`** solved, **`+327.6`** over `myo_ppo_runner_cfg`'s `-88.36 ± 13.79`) on 216-muscle `myoTorsoPoseFixed-v0`, while `CurriculumLambdaHold+PPO` (`hold_steps: 2 -> 1`, `reflex_weight: 1.0 -> 0.0`) removes the 2-step holding ceiling on **80-muscle `myoLegStandRandom-v0`** (**`624.15 ± 62.34`**, **`95.0% ± 0.0%`** solved vs. `352.81 ± 18.54` / `90.0%` for fixed `LambdaHold+PPO` and `541.51 ± 33.22` / `91.7%` for `myo_ppo_runner_cfg`), **`myoFingerReachRandom-v0`** (**`242.65 ± 55.84`** vs. `144.78 ± 12.01` for fixed `LambdaHold+PPO` and `233.97 ± 62.23` for `myo_ppo_runner_cfg`), and `myoElbowPose1D6MRandom-v0` (`96.7%` vs. `73.3%` solved).

### 1. Multi-Task GPU `mjlab` Benchmark vs. Canonical `myo_ppo_runner_cfg` (`184,320` Steps, Seeds `0, 1, 2`)

- **Hardware:** `NVIDIA GeForce RTX 4070 Ti SUPER` (16 GiB, `torch 2.15.0+cu132`, `mujoco-warp 3.11.0`, `mjlab 1.6.1`, `rsl-rl-lib 5.4.2`).
- **Protocol:** `num_envs = 256`, `30` iterations × `24` steps (`184,320` env steps), seeds `0, 1, 2`, evaluated deterministically over `20` seeded episodes (`seed = 10000..10019`) on unmodified CPU `make_env(env_id)`.

| Task (`env_id`) | Method | CPU `make_env` Return | CPU Solved Rate | Wall-Clock (s) | Peak VRAM (MB) |
|---|---|---:|---:|---:|---:|
| `myoFingerPoseRandom-v0` (5 muscles, 4 DOFs) | `mjlab` PPO (`myo_ppo_runner_cfg`) | **243.54 ± 15.83** | `96.7% ± 2.4%` | `15.34 ± 0.37` | **22.9** |
| `myoFingerPoseRandom-v0` (5 muscles, 4 DOFs) | `LambdaHold+PPO` (`hold_steps=2`) | `198.63 ± 48.96` | **100.0% ± 0.0%** | `15.80 ± 0.15` | `23.0` |
| `myoFingerPoseRandom-v0` (5 muscles, 4 DOFs) | `CurriculumLambdaHold+PPO` (`2 -> 1`) | `220.88 ± 80.24` | `85.0% ± 21.2%` | **15.32 ± 0.03** | `23.0` |
| `myoFingerReachRandom-v0` (5 muscles, 3D reach) | `mjlab` PPO (`myo_ppo_runner_cfg`) | `233.97 ± 62.23` | **33.3% ± 6.2%** | `19.05 ± 0.64` | **25.6** |
| `myoFingerReachRandom-v0` (5 muscles, 3D reach) | `LambdaHold+PPO` (`hold_steps=2`) | `144.78 ± 12.01` | `28.3% ± 8.5%` | `18.18 ± 0.17` | `38.7` |
| `myoFingerReachRandom-v0` (5 muscles, 3D reach) | `CurriculumLambdaHold+PPO` (`2 -> 1`) | **242.65 ± 55.84** | `16.7% ± 6.2%` | **17.78 ± 0.05** | `57.1` |
| `myoLegStandRandom-v0` (80 muscles, 3D balance) | `mjlab` PPO (`myo_ppo_runner_cfg`) | `541.51 ± 33.22` | `91.7% ± 4.7%` | **28.55 ± 0.51** | **40.2** |
| `myoLegStandRandom-v0` (80 muscles, 3D balance) | `LambdaHold+PPO` (`hold_steps=2`) | `352.81 ± 18.54` | `90.0% ± 0.0%` | `29.03 ± 0.21` | `40.9` |
| `myoLegStandRandom-v0` (80 muscles, 3D balance) | `CurriculumLambdaHold+PPO` (`2 -> 1`) | **624.15 ± 62.34** | **95.0% ± 0.0%** | `28.66 ± 0.18` | `40.9` |
| `myoTorsoPoseFixed-v0` (216 muscles, spine) | `mjlab` PPO (`myo_ppo_runner_cfg`) | `-88.36 ± 13.79` | **100.0% ± 0.0%** | **18.27 ± 0.22** | **62.0** |
| `myoTorsoPoseFixed-v0` (216 muscles, spine) | `LambdaHold+PPO` (`hold_steps=2`) | **239.20 ± 148.20** | **100.0% ± 0.0%** | `18.41 ± 0.12` | `62.4` |
| `myoTorsoPoseFixed-v0` (216 muscles, spine) | `CurriculumLambdaHold+PPO` (`2 -> 1`) | `-70.39 ± 11.77` | **100.0% ± 0.0%** | `18.46 ± 0.08` | `62.4` |

#### GPU Learning Curves vs. Environment Steps & Wall-Clock Time (3-Seed Mean)

| Task | Env Steps | `mjlab` PPO Return (Solved, Wall s) | `LambdaHold+PPO` Return (Solved, Wall s) | `CurriculumLambdaHold+PPO` Return (Solved, Wall s) |
|---|---:|---:|---:|---:|
| `myoFingerPoseRandom-v0` | `0` | `-90.4` (`26.7%`, `0.0s`) | `-17.9` (`23.3%`, `0.0s`) | `-80.7` (`11.7%`, `0.0s`) |
| `myoFingerPoseRandom-v0` | `61,440` | `-30.4` (`46.7%`, `5.7s`) | `167.7` (`66.7%`, `5.8s`) | `114.8` (`63.3%`, `5.6s`) |
| `myoFingerPoseRandom-v0` | `122,880` | `110.0` (`76.7%`, `10.5s`) | `180.5` (`86.7%`, `10.8s`) | `120.6` (`75.0%`, `10.5s`) |
| `myoFingerPoseRandom-v0` | `184,320` | `243.5` (`96.7%`, `15.3s`) | `198.6` (`100.0%`, `15.8s`) | `220.9` (`85.0%`, `15.3s`) |
| `myoFingerReachRandom-v0` | `0` | `-23.1` (`1.7%`, `0.0s`) | `-8.9` (`3.3%`, `0.0s`) | `-9.4` (`0.0%`, `0.0s`) |
| `myoFingerReachRandom-v0` | `61,440` | `-3.0` (`0.0%`, `6.5s`) | `8.3` (`5.0%`, `6.1s`) | `-1.0` (`1.7%`, `6.0s`) |
| `myoFingerReachRandom-v0` | `122,880` | `94.4` (`18.3%`, `12.9s`) | `33.6` (`11.7%`, `12.3s`) | `60.4` (`0.0%`, `11.9s`) |
| `myoFingerReachRandom-v0` | `184,320` | `234.0` (`33.3%`, `19.0s`) | `144.8` (`28.3%`, `18.2s`) | `242.7` (`16.7%`, `17.8s`) |
| `myoLegStandRandom-v0` | `0` | `659.4` (`93.3%`, `0.0s`) | `351.0` (`90.0%`, `0.0s`) | `350.3` (`90.0%`, `0.0s`) |
| `myoLegStandRandom-v0` | `61,440` | `612.5` (`93.3%`, `10.3s`) | `350.4` (`90.0%`, `10.2s`) | `376.6` (`90.0%`, `10.1s`) |
| `myoLegStandRandom-v0` | `122,880` | `561.2` (`93.3%`, `19.4s`) | `341.8` (`90.0%`, `19.6s`) | `519.2` (`93.3%`, `19.4s`) |
| `myoLegStandRandom-v0` | `184,320` | `541.5` (`91.7%`, `28.6s`) | `352.8` (`90.0%`, `29.0s`) | `624.1` (`95.0%`, `28.7s`) |
| `myoTorsoPoseFixed-v0` | `0` | `-114.3` (`100.0%`, `0.0s`) | `-3.9` (`100.0%`, `0.0s`) | `-20.7` (`100.0%`, `0.0s`) |
| `myoTorsoPoseFixed-v0` | `61,440` | `-113.7` (`100.0%`, `6.7s`) | `20.7` (`100.0%`, `6.7s`) | `-20.0` (`100.0%`, `6.7s`) |
| `myoTorsoPoseFixed-v0` | `122,880` | `-109.0` (`100.0%`, `12.5s`) | `41.7` (`100.0%`, `12.6s`) | `-77.6` (`100.0%`, `12.6s`) |
| `myoTorsoPoseFixed-v0` | `184,320` | `-88.4` (`100.0%`, `18.3s`) | `239.2` (`100.0%`, `18.4s`) | `-70.4` (`100.0%`, `18.5s`) |

### 2. CPU `stable-baselines3` Comparison at `65,536` Steps (`DummyVecEnv` × 8 Envs, Seeds `0, 1, 2`)

| Task (`env_id`) | Method | Final Return | Final Solved Rate | Wall-Clock (s) | Peak Traced Memory (MB) |
|---|---|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` (6 muscles) | `PPO` | `605.7 ± 49.8` | `98.3% ± 2.4%` | `26.4 ± 0.7` | `69.4` |
| `myoElbowPose1D6MRandom-v0` (6 muscles) | `LambdaHold+PPO` | `366.9 ± 9.2` | `73.3% ± 2.4%` | `30.0 ± 0.5` | `4.2` |
| `myoElbowPose1D6MRandom-v0` (6 muscles) | `CurriculumLambdaHold+PPO` | `592.3 ± 63.3` | `96.7% ± 2.4%` | `30.5 ± 0.4` | `4.2` |
| `myoHandPoseRandom-v0` (39 muscles) | `PPO` | `-311.7 ± 2.4` | `0.0% ± 0.0%` | `44.3 ± 0.5` | `21.1` |
| `myoHandPoseRandom-v0` (39 muscles) | `LambdaHold+PPO` | `-231.3 ± 1.0` | `0.0% ± 0.0%` | `48.8 ± 0.4` | `21.0` |
