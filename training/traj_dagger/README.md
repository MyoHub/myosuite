# Multi-Teacher Trajectory-Swap DAgger for MyoSuite

Self-contained PyTorch distillation package (`myosuite-traj-dagger`, import name
`myosuite_traj_dagger`) implementing **Multi-Teacher Trajectory-Swap DAgger** for
consolidating specialist policies across multiple teachers, target sub-regions, or
motion clips into a single deployable student policy on both CPU (`MyoGymnasiumEnv`)
and GPU (`mjlab` `ManagerBasedRlEnv`).

Unlike `training/chasetag_teacher`—which implements a task-specific steered
SuperMimic teacher, yaw canonicalisation, and 1v1 ChaseTag bout distillation—this
package provides a **task-agnostic multi-teacher trajectory-swap distillation
trainer** that consumes any registered MyoSuite task contract without modifying
`myosuite/` (see `docs/wiki/training-guidelines.md`).

---

## Attribution and Licensing

This is not an original MyoSuite algorithm. All Python source files in this package
are newly written PyTorch/NumPy implementations for MyoSuite licensed under
Apache-2.0 (`LICENSE`), adapting the algorithmic design of **DAgger** and
**LocoMuJoCo's JAX distillation suite**:

1. **DAgger (Dataset Aggregation):**
   > Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell. *"A Reduction of
   > Imitation Learning and Structured Prediction to No-Regret Online Learning."*
   > AISTATS 2011 ([arXiv:1011.0686](https://arxiv.org/abs/1011.0686)).
2. **LocoMuJoCo Multi-Teacher / Trajectory-Swap Distillation:**
   > Firas Al-Hafez, Guoping Zhao, Jan Peters, Davide Tateo. *"LocoMuJoCo: A
   > Comprehensive Imitation Learning Benchmark for Locomotion."*
   > Source repository: [github.com/robfiras/loco-mujoco](https://github.com/robfiras/loco-mujoco)
   > (`loco_mujoco/algorithms/vanilla_dagger_jax.py` and
   > `examples/training_examples/jax_distill/experiment_traj_swap.py`),
   > MIT License (`Copyright (c) 2024 Al-Hafez`; preserved in
   > `LICENSE-loco-mujoco.txt`).

| Component | Upstream / Paper Origin | Implementation in `myosuite_traj_dagger` |
|---|---|---|
| Interactive aggregation loop (`beta` annealing) | Ross, Gordon & Bagnell (AISTATS 2011) | Re-implemented in PyTorch (`TrajSwapDagger`, `distill_on_vec_env`). |
| Sticky per-env mixture horizon `DaggerRolloutState` (`using_teacher ~ Bernoulli(beta)`, `steps_remaining ~ UniformInt(h_min, h_max)`) | Adapted from `DaggerRolloutState` in `loco-mujoco` (`vanilla_dagger_jax.py`, MIT) | Re-implemented for both NumPy and PyTorch GPU tensors in `buffer.py`. |
| Combined short-term ring buffer + long-term `ReservoirBuffer` (`min_include_prob`) | Adapted from `ReservoirBuffer` + rollout replay in `loco-mujoco` (`vanilla_dagger_jax.py`, MIT) | Re-implemented in PyTorch (`PersistentDaggerBuffer` in `buffer.py`) with cross-chunk persistence and source-pair tracking. |
| Weighted `(teacher, reset_options)` chunk swapping & deduplicated teacher loading via pair repetition | Adapted from `experiment_traj_swap.py` in `loco-mujoco` (MIT) | New `SwapPair`, `repeat_pair`, and `TeacherRegistry` supporting callable, PyTorch, TorchScript, RSL-RL (`model_*.pt`), SB3 (`.zip`), and ONNX teachers (`dagger.py`). |
| Gaussian policy NLL `-log pi_student(a_teacher \| s)` + optional critic MSE `\|\|V_student(s) - V_teacher(s)\|\|^2` | Adapted from `vanilla_dagger_jax.py` loss formulation (`loco-mujoco`, MIT) | New `StudentActorCritic` with embedded `RunningMeanStd`, independent student layer sizes, and standalone TorchScript / ONNX export (`dagger.py`). |
| Offline Behavior Cloning (`OfflineBC`, `beta=1.0`), `mjlab` GPU CLI (`train_mjlab.py`), and 3-arm benchmark (`benchmark.py`) | Follow `docs/wiki/training-guidelines.md` | New MyoSuite code (`Apache-2.0`). |

---

## Trajectory-Swap Semantics

1. **Weighted `(teacher, reset_options)` Schedule & Single-Load Registry:**
   A schedule is a list of `SwapPair(teacher=TeacherSpec(...), reset_options=...)`.
   Repeating a pair `K` times (`repeat_pair(pair, K)`) gives it `K` times the chunk
   selection probability while `TeacherRegistry` loads and caches each unique
   `TeacherSpec.key` once.
2. **Clean Environment Carry Reset, Persistent Memory & Student State:**
   At every chunk boundary, all parallel environments reset cleanly with the new
   pair's `reset_options` and `DaggerRolloutState` resamples fresh sticky horizons
   so no stale physics carry crosses specialist boundaries. Meanwhile,
   `PersistentDaggerBuffer` (ring + reservoir) and the `StudentActorCritic` weights,
   `RunningMeanStd` normalizer, and Adam moments persist uninterrupted across swaps.
3. **Sticky Mixture Rollouts (`DaggerRolloutState`):**
   Instead of switching between teacher and student independently at every single
   control step (which induces high-frequency chattering in muscle-actuated
   dynamics), each environment commits to either the teacher or the student for
   `h ~ UniformInt(h_min, h_max)` consecutive steps before resampling
   `using_teacher ~ Bernoulli(beta)` (or immediately upon episode reset). Every
   visited state is labeled with the active specialist teacher's deterministic
   action (and optional `V_teacher(s)`).
4. **Student Objective & Export:**
   The student minimizes Gaussian negative log-likelihood
   `-log pi_student(a_teacher | s)` plus optional value distillation MSE
   `value_loss_coef * ||V_student(s) - V_teacher(s)||^2`, and exports directly to a
   self-contained `.onnx` (`obs -> action`, matching `myosuite.utils.export_onnx`)
   or TorchScript `.ts` archive.

---

## Installation

```bash
# Base CPU install (from a MyoSuite checkout):
pip install -e .
pip install -e "training/traj_dagger[test]"

# With GPU mjlab distillation CLI:
pip install -e ".[mjlab]" -e "training/traj_dagger[mjlab]"

# Or standalone from git:
pip install myosuite
pip install "git+https://github.com/MyoHub/myosuite.git@dev#subdirectory=training/traj_dagger"
```

---

## CLI Usage

### 1. GPU `mjlab` Multi-Teacher Distillation (`traj-dagger-mjlab`)

```bash
# Distill two RSL-RL specialist checkpoints (weighting run_a 2x) into a smaller [128, 128] student:
traj-dagger-mjlab myoElbowPose1D6MRandom-v0 \
    --teachers "2*logs/rsl_rl/elbow_low/model_500.pt" "logs/rsl_rl/elbow_high/model_500.pt" \
    --env.scene.num-envs 1024 \
    --actor-hidden-dims 128 128 \
    --total-steps 1000000
```

Outputs `student_final.pt`, `student_final.ts` (TorchScript), and `student_final.onnx`
under `logs/traj_dagger/<env_id>/<timestamp>/`.

### 2. Reproducible 3-Arm Benchmark (`traj-dagger-bench`)

```bash
traj-dagger-bench \
    --envs myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0 \
    --total-steps 32768 \
    --teacher-steps 32768 \
    --eval-interval 8192 \
    --seeds 0 1 2 \
    --n-envs 8 \
    --n-eval-episodes 20 \
    --out-json benchmark_results.json
```

---

## Benchmark: PPO vs. Offline BC vs. Multi-Teacher TrajSwapDAgger

Follows `docs/wiki/training-guidelines.md`:

- **Environments (unmodified observations, actions, rewards, and evaluation):**
  - **GPU `mjlab` (`184,320` steps, `256` parallel envs, seeds `0, 1, 2`):** `myoFingerPoseRandom-v0` (4-DOF, 5 muscles), `myoFingerReachRandom-v0` (3D fingertip reach, 5 muscles), and `myoLegStandRandom-v0` (3D pelvis balance, 80 muscles), compared against MyoSuite's registered `myo_ppo_runner_cfg` (`scripts/train_mjlab.py`) and teacher-only `BC`.
  - **CPU `traj-dagger-bench` (`32,768` steps, `8` parallel envs, seeds `0, 1, 2`):** `myoElbowPose1D6MRandom-v0` (1-DOF, 6 muscles) and `myoHandPoseRandom-v0` (23-DOF, 39 muscles).
- **Specialist Teacher Setup (reported transparently):**
  Two disjoint workspace-half specialist teachers (`teacher_low` and `teacher_high`) are pre-trained on `[low, mid]` and `[mid, high]` along the widest target coordinate, leaving all other coordinates intact so a 50/50 chunk schedule covers the exact target distribution of the unmodified `Random` task (`TargetHalfSpaceWrapper` supports both `_target_jnt_range` pose tasks and `target_reach_range` / workspace-sampled 3D reach tasks).

**One-sentence summary:** Consolidating two disjoint half-workspace specialists via `TrajSwapDAgger` outperforms MyoSuite's canonical `myo_ppo_runner_cfg` from scratch on 3D fingertip reaching (`myoFingerReachRandom-v0`: **`307.23 ± 1.47` return, `45.0% ± 0.0%` solved** vs. `233.97 ± 62.23`, `33.3% ± 6.2%` for `myo_ppo_runner_cfg`) and 80-muscle 3D balance (`myoLegStandRandom-v0`: **`620.75 ± 1.94` return, `95.0% ± 0.0%` solved** vs. `541.51 ± 33.22`, `91.7%` for `myo_ppo_runner_cfg` and `614.56 ± 3.07` for `BC`), reaches **`96.7%` solved on `myoFingerPoseRandom-v0` in `61,440` steps** (`3×` fewer steps than `myo_ppo_runner_cfg`, where individual half-specialists score only `20.0%` and `60.0%` on the full range), and beats both scratch `PPO` and offline `BC` on CPU `myoElbowPose1D6MRandom-v0` (`411.56 ± 110.40`, `50.0%` vs. `362.43` `BC` / `260.98` `PPO`) and 39-muscle `myoHandPoseRandom-v0` (`-312.88 ± 2.63` vs. `-319.37` `BC` / `-325.99` `PPO`).

### 1. Multi-Task GPU `mjlab` Benchmark (`184,320` Steps, `256` Envs, Seeds `0, 1, 2`)

- **Hardware:** `NVIDIA GeForce RTX 4070 Ti SUPER`, Windows 10 (`10.0.26300`), Python `3.11.18`, PyTorch `2.15.0+cu132`, `mjlab 1.6.1`, `mujoco 3.11.0`, evaluated deterministically over 20 seeded episodes (`eval_seed=10000..10019`) on unmodified CPU `make_env(env_id)`.
- **Untimed Specialist Full-Range Baselines (showing single-specialist degradation outside its half-space):**
  - `myoFingerPoseRandom-v0`: `teacher_low` = `29.21` (`20.0%` solved), `teacher_high` = `94.83` (`60.0%` solved).
  - `myoFingerReachRandom-v0`: `teacher_low` = `197.75` (`50.0%` solved), `teacher_high` = `302.15` (`30.0%` solved).
  - `myoLegStandRandom-v0`: `teacher_low` = `740.87` (`95.0%` solved), `teacher_high` = `501.36` (`90.0%` solved).

| Environment (Muscles) | Arm | Budget (`env_steps`) | Final Return (`mean ± std`) | Terminal Solved (`mean ± std`) | Wall-Clock (`s`) | Peak GPU VRAM (`MB`) |
|---|---|---:|---:|---:|---:|---:|
| `myoFingerPoseRandom-v0` (5M) | `myo_ppo_runner_cfg` | 184,320 | **`243.54 ± 15.83`** | **`96.7% ± 2.4%`** | `15.34 ± 0.37` | `22.9` |
| `myoFingerPoseRandom-v0` (5M) | `BC` (`beta=1.0`) | 184,320 | `224.80 ± 29.25` | `80.0% ± 4.1%` | `19.10 ± 1.22` | `20.8` |
| `myoFingerPoseRandom-v0` (5M) | `TrajSwapDAgger` | 184,320 | `178.51 ± 12.74` *(`223.96` @ `61k`)* | `66.7% ± 2.4%` *(`96.7%` @ `61k`)* | `19.68 ± 0.87` | `21.1` |
| `myoFingerReachRandom-v0` (5M) | `myo_ppo_runner_cfg` | 184,320 | `233.97 ± 62.23` | `33.3% ± 6.2%` | `19.05 ± 0.64` | `25.6` |
| `myoFingerReachRandom-v0` (5M) | `BC` (`beta=1.0`) | 184,320 | `305.29 ± 0.53` | **`46.7% ± 2.4%`** | `21.44 ± 0.45` | `25.4` |
| `myoFingerReachRandom-v0` (5M) | `TrajSwapDAgger` | 184,320 | **`307.23 ± 1.47`** | `45.0% ± 0.0%` *(`48.3%` @ `61k`)* | `21.21 ± 0.05` | `28.4` |
| `myoLegStandRandom-v0` (80M) | `myo_ppo_runner_cfg` | 184,320 | `541.51 ± 33.22` | `91.7% ± 4.7%` | `28.55 ± 0.51` | `40.2` |
| `myoLegStandRandom-v0` (80M) | `BC` (`beta=1.0`) | 184,320 | `614.56 ± 3.07` | **`95.0% ± 0.0%`** | `30.43 ± 0.10` | `117.8` |
| `myoLegStandRandom-v0` (80M) | `TrajSwapDAgger` | 184,320 | **`620.75 ± 1.94`** | **`95.0% ± 0.0%`** | `30.52 ± 0.09` | `125.8` |

#### GPU `mjlab` Learning Curves vs. Environment Steps & Wall-Clock Time (3-Seed Mean)

| Environment | Env Steps | `myo_ppo_runner_cfg` Return (Solved, Wall s) | `BC` Return (Solved, Wall s) | `TrajSwapDAgger` Return (Solved, Wall s) |
|---|---:|---:|---:|---:|
| `myoFingerPoseRandom-v0` | `0` | `-90.44` (`26.7%`, `0.0s`) | `-106.75` (`11.7%`, `0.0s`) | `-106.75` (`11.7%`, `0.0s`) |
| `myoFingerPoseRandom-v0` | `61,440` | `-30.43` (`46.7%`, `5.70s`) | `215.81` (`93.3%`, `7.10s`) | **`223.96` (`96.7%`, `7.28s`)** |
| `myoFingerPoseRandom-v0` | `122,880` | `109.96` (`76.7%`, `10.53s`) | `189.12` (`90.0%`, `13.24s`) | `172.55` (`73.3%`, `13.65s`) |
| `myoFingerPoseRandom-v0` | `184,320` | `243.54` (`96.7%`, `15.34s`) | `224.80` (`80.0%`, `19.10s`) | `178.51` (`66.7%`, `19.68s`) |
| `myoFingerReachRandom-v0` | `0` | `-23.12` (`1.7%`, `0.0s`) | `-17.07` (`3.3%`, `0.0s`) | `-17.07` (`3.3%`, `0.0s`) |
| `myoFingerReachRandom-v0` | `61,440` | `-2.98` (`0.0%`, `6.51s`) | `312.79` (`50.0%`, `7.91s`) | **`318.11` (`48.3%`, `7.68s`)** |
| `myoFingerReachRandom-v0` | `122,880` | `94.41` (`18.3%`, `12.92s`) | `308.44` (`48.3%`, `14.79s`) | `303.16` (`48.3%`, `14.57s`) |
| `myoFingerReachRandom-v0` | `184,320` | `233.97` (`33.3%`, `19.05s`) | `305.29` (`46.7%`, `21.44s`) | **`307.23` (`45.0%`, `21.21s`)** |
| `myoLegStandRandom-v0` | `0` | `659.37` (`93.3%`, `0.0s`) | `726.83` (`95.0%`, `0.0s`) | `726.83` (`95.0%`, `0.0s`) |
| `myoLegStandRandom-v0` | `61,440` | `612.51` (`93.3%`, `10.34s`) | `620.50` (`95.0%`, `10.88s`) | `621.96` (`95.0%`, `10.92s`) |
| `myoLegStandRandom-v0` | `122,880` | `561.15` (`93.3%`, `19.38s`) | `614.13` (`93.3%`, `20.77s`) | `625.26` (`95.0%`, `20.85s`) |
| `myoLegStandRandom-v0` | `184,320` | `541.51` (`91.7%`, `28.55s`) | `614.56` (`95.0%`, `30.43s`) | **`620.75` (`95.0%`, `30.52s`)** |

### 2. CPU `traj-dagger-bench` Reference (`32,768` Steps, `8` Envs, Seeds `0, 1, 2`)

- **Hardware:** Windows 10 (`10.0.26300`), `AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD` (24 logical CPUs), Python `3.11.18`, PyTorch `2.13.0+cpu`, `SyncEnvPool` × 8 envs.
- **Untimed Specialist Pre-training Diagnostics (3-seed mean):** On `myoElbowPose1D6MRandom-v0` (split on `r_elbow_flex` at `1.135 rad`), specialist pre-training took `17.32 s` per seed; each specialist averaged `426.1` return in its own half-space (`teacher_low` dropped to `172.4` on the full range). On `myoHandPoseRandom-v0` (split on `md5_flexion_r` at `0.793 rad`), specialist pre-training took `38.28 s` per seed.

#### CPU Final Results (`seeds = 0, 1, 2`, 20 deterministic episodes per seed)

| Environment | Arm | Budget (`env_steps`) | Final Return (`mean ± std`) | Terminal Solved (`mean ± std`) | Wall-Clock (`s`) | Peak Memory (`MB`) |
|---|---|---:|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | `PPO` | 32,768 | `260.98 ± 107.11` | `33.3% ± 10.3%` | `17.44 ± 0.16` | `0.94` |
| `myoElbowPose1D6MRandom-v0` | `BC` | 32,768 | `362.43 ± 155.09` | `43.3% ± 24.9%` | `17.48 ± 0.23` | `1.00` |
| `myoElbowPose1D6MRandom-v0` | `TrajSwapDAgger` | 32,768 | `411.56 ± 110.40` | `50.0% ± 18.7%` | `30.74 ± 0.98` | `0.97` |
| `myoHandPoseRandom-v0` | `PPO` | 32,768 | `-325.99 ± 11.19` | `0.0% ± 0.0%` | `28.73 ± 2.18` | `2.08` |
| `myoHandPoseRandom-v0` | `BC` | 32,768 | `-319.37 ± 1.37` | `0.0% ± 0.0%` | `28.68 ± 2.13` | `1.09` |
| `myoHandPoseRandom-v0` | `TrajSwapDAgger` | 32,768 | `-312.88 ± 2.63` | `0.0% ± 0.0%` | `40.78 ± 2.79` | `1.08` |

### Learning Curves vs. Environment Steps & Wall-Clock Time (3-Seed Mean)

| Environment | Env Steps | `PPO` Return (Solved, Wall s) | `BC` Return (Solved, Wall s) | `TrajSwapDAgger` Return (Solved, Wall s) |
|---|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | `0` | `34.87` (`10.0%`, `0.69s`) | `73.00` (`15.0%`, `0.69s`) | `32.10` (`11.7%`, `0.71s`) |
| `myoElbowPose1D6MRandom-v0` | `8,192` | `165.20` (`25.0%`, `4.78s`) | `354.22` (`35.0%`, `4.76s`) | `448.84` (`53.3%`, `7.49s`) |
| `myoElbowPose1D6MRandom-v0` | `16,384` | `228.45` (`30.0%`, `8.89s`) | `350.11` (`40.0%`, `8.90s`) | `416.71` (`50.0%`, `15.19s`) |
| `myoElbowPose1D6MRandom-v0` | `24,576` | `211.87` (`25.0%`, `13.05s`) | `350.54` (`40.0%`, `13.01s`) | `404.67` (`48.3%`, `22.82s`) |
| `myoElbowPose1D6MRandom-v0` | `32,768` | `260.98` (`33.3%`, `17.15s`) | `362.43` (`43.3%`, `17.19s`) | `411.56` (`50.0%`, `30.44s`) |
| `myoHandPoseRandom-v0` | `0` | `-307.00` (`0.0%`, `1.18s`) | `-305.10` (`0.0%`, `1.18s`) | `-307.30` (`0.0%`, `1.19s`) |
| `myoHandPoseRandom-v0` | `8,192` | `-329.75` (`0.0%`, `7.94s`) | `-318.69` (`0.0%`, `8.00s`) | `-313.08` (`0.0%`, `10.60s`) |
| `myoHandPoseRandom-v0` | `16,384` | `-329.87` (`0.0%`, `14.76s`) | `-320.00` (`0.0%`, `14.86s`) | `-312.70` (`0.0%`, `20.76s`) |
| `myoHandPoseRandom-v0` | `24,576` | `-328.33` (`0.0%`, `21.74s`) | `-320.44` (`0.0%`, `21.74s`) | `-312.79` (`0.0%`, `30.73s`) |
| `myoHandPoseRandom-v0` | `32,768` | `-325.99` (`0.0%`, `28.57s`) | `-319.37` (`0.0%`, `28.54s`) | `-312.88` (`0.0%`, `40.63s`) |

---

## Tests

```bash
pytest training/traj_dagger/tests/test_traj_dagger.py -v
```
