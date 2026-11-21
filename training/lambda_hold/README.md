# Lambda-Hold Equilibrium-Point Stretch-Reflex Control for MyoSuite

PPO training on registered MyoSuite tasks through the `LambdaHold` equilibrium-point (EP) stretch-reflex action map (`myosuite.utils.action_map.LambdaHold`). Instead of commanding raw muscle excitations directly, the policy outputs normalized per-muscle threshold lengths $z_i \in [-1, 1]$ mapped around each actuator's physiological `lengthrange`, and a stretch-reflex recruitment law computes excitations from live actuator length and velocity with optional intermittent phase holding (`hold_steps`).

## Attribution & Licensing

`LambdaHold` is adapted from:

> Jun-Hyuk Lee et al., **λ-Hold Control** (2026). [arXiv:2608.17030](https://arxiv.org/abs/2608.17030) · upstream repository: [github.com/Lee-Jun-Hyuk-37/Lambda-Hold](https://github.com/Lee-Jun-Hyuk-37/Lambda-Hold) (MIT License, [doi:10.5281/zenodo.23265019](https://doi.org/10.5281/zenodo.23265019)).

This is not an original MyoSuite method; it is a clean-room MyoSuite re-implementation adapted for MuJoCo and `mjlab` from the paper's equilibrium-point stretch-reflex formulation (upstream targets SCONE/Hyfydy). Dimensionless excursion normalization by each actuator's `actuator_lengthrange` span and exact inversion of MyoSuite's `muscle_sigmoid` (`normalize_act=True`) are MyoSuite-specific additions so one gain configuration spans finger, arm, and leg models across both CPU and `mjlab`.

## Layout

- **Playback & wrappers (`myosuite/`):** `LambdaHold` (`myosuite/utils/action_map.py`) and `ActionMapAction` (`myosuite/envs/myo/backends/mjlab/tasks/mdp/action_map.py`). Saved checkpoints store the `LambdaHold` manifest in `infos`, so `scripts/eval_mjlab_policy.py` and `ActionMapWrapper` play them without importing `training/lambda_hold`.
- **Trainers & benchmark (`training/lambda_hold/`):**
  - `myosuite_lambda_hold/train.py`: GPU `mjlab` / RSL-RL PPO trainer (`lambda-hold-train`).
  - `myosuite_lambda_hold/benchmark_sb3.py`: reproducible CPU `stable-baselines3` PPO vs. `LambdaHold+PPO` side-by-side benchmark (`lambda-hold-bench-sb3`).

## Install & Usage

```bash
pip install -e ".[rl]"          # or pip install -e ".[mjlab]" for GPU RSL-RL training
pip install -e training/lambda_hold

# Train on mjlab (GPU / RSL-RL)
lambda-hold-train myoLegWalk-v0 --env.scene.num-envs 1024 --kp 2.5 --damping-s 0.04 --hold-steps 2

# Reproduce the 3-seed CPU PPO comparison below
lambda-hold-bench-sb3 --envs myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0 --seeds 0 1 2 --steps 65536
```

## Benchmark Against PPO

**One-sentence summary:** Compared side by side at an equal 65,536-step budget across seeds `0, 1, 2`, `LambdaHold+PPO` (`kp=2.5, damping_s=0.04, hold_steps=2`) yields substantially higher early-training return (`293.9` vs. `126.5` at 32k steps on `myoElbowPose1D6MRandom-v0`) and `+80.4` higher final return on the 39-muscle `myoHandPoseRandom-v0` (`-231.3 ± 1.0` vs. `-311.7 ± 2.4`), while direct per-step `PPO` reaches higher asymptotic accuracy on the low-dimensional 6-muscle `myoElbowPose1D6MRandom-v0` at 65k steps (`605.7 ± 49.8`, `98.3%` solved vs. `366.9 ± 9.2`, `73.3%` solved) where holding thresholds over 2 steps limits fine single-joint settling.

- **Hardware:** Windows 10, AMD64 (`Family 26 Model 68 Stepping 0, AuthenticAMD`), CPU (`DummyVecEnv` × 8 envs), Python 3.11.18.
- **Protocol:** Unmodified registered observations and rewards on `myoElbowPose1D6MRandom-v0` (6 muscles) and `myoHandPoseRandom-v0` (39 muscles), identical PPO hyperparameters (`lr=3e-4`, `batch_size=256`, `n_epochs=5`), seeds `0, 1, 2`, 20 deterministic seeded evaluation episodes per checkpoint.

### Final Performance at 65,536 Environment Steps (Mean ± Std over Seeds `0, 1, 2`)

| Task (`env_id`) | Method | Final Return | Final Solved Rate | Wall-Clock (s) | Peak Traced Memory (MB) |
|---|---|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` (6 muscles) | `PPO` | `605.7 ± 49.8` | `98.3% ± 2.4%` | `25.4 ± 0.1` | `69.4` |
| `myoElbowPose1D6MRandom-v0` (6 muscles) | `LambdaHold+PPO` | `366.9 ± 9.2` | `73.3% ± 2.4%` | `29.0 ± 0.2` | `4.2` |
| `myoHandPoseRandom-v0` (39 muscles) | `PPO` | `-311.7 ± 2.4` | `0.0% ± 0.0%` | `44.3 ± 0.5` | `21.1` |
| `myoHandPoseRandom-v0` (39 muscles) | `LambdaHold+PPO` | `-231.3 ± 1.0` | `0.0% ± 0.0%` | `48.8 ± 0.4` | `21.0` |

### Learning Curves vs. Environment Steps & Wall-Clock Time (3-Seed Mean)

| Task | Env Steps | `PPO` Wall (s) | `PPO` Return (Solved) | `LambdaHold+PPO` Wall (s) | `LambdaHold+PPO` Return (Solved) |
|---|---:|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | `0` | `0.0` | `0.5` (`40.0%`) | `0.0` | `245.4` (`55.0%`) |
| `myoElbowPose1D6MRandom-v0` | `16,384` | `6.3` | `49.9` (`43.3%`) | `7.1` | `258.1` (`55.0%`) |
| `myoElbowPose1D6MRandom-v0` | `32,768` | `12.7` | `126.5` (`45.0%`) | `14.4` | `293.9` (`55.0%`) |
| `myoElbowPose1D6MRandom-v0` | `49,152` | `19.1` | `278.7` (`70.0%`) | `21.8` | `329.8` (`61.7%`) |
| `myoElbowPose1D6MRandom-v0` | `65,536` | `25.4` | `605.7` (`98.3%`) | `29.0` | `366.9` (`73.3%`) |
| `myoHandPoseRandom-v0` | `0` | `0.0` | `-316.5` (`0.0%`) | `0.0` | `-230.4` (`0.0%`) |
| `myoHandPoseRandom-v0` | `16,384` | `11.2` | `-314.3` (`0.0%`) | `12.0` | `-229.3` (`0.0%`) |
| `myoHandPoseRandom-v0` | `32,768` | `22.2` | `-313.0` (`0.0%`) | `24.5` | `-229.5` (`0.0%`) |
| `myoHandPoseRandom-v0` | `49,152` | `33.3` | `-312.1` (`0.0%`) | `36.8` | `-231.5` (`0.0%`) |
| `myoHandPoseRandom-v0` | `65,536` | `44.3` | `-311.7` (`0.0%`) | `48.8` | `-231.3` (`0.0%`) |
