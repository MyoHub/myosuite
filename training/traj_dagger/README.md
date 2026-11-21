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
  `myoElbowPose1D6MRandom-v0` (1-DOF, 6 muscles) and `myoHandPoseRandom-v0`
  (23-DOF, 39 muscles), built via `myosuite.make_env`.
- **Specialist Teacher Setup (reported transparently):**
  Per seed, two workspace-half specialist teachers (`teacher_low` and
  `teacher_high`) are pre-trained on disjoint halves of the widest target joint
  range (`[low, mid]` and `[mid, high]`, leaving all other joint ranges intact so a
  50/50 chunk schedule equals the exact uniform target distribution of the
  unmodified `Random` task).
- **Three Matched-Budget Arms (`seeds = 0, 1, 2`, 20 deterministic eval episodes):**
  1. `PPO`: online clipped PPO from scratch on `make_env(env_id)` for `total_steps`.
  2. `BC`: offline Behavior Cloning on `total_steps` transitions collected solely
     from the two specialist teachers (`beta = 1.0`, no student rollouts).
  3. `TrajSwapDAgger`: online sticky-horizon trajectory-swap DAgger over
     `total_steps` interactive environment steps consolidating both specialists.

**One-sentence summary:** At an equal `32,768`-step budget across seeds `0, 1, 2`, `TrajSwapDAgger` consolidates two disjoint workspace-half specialist teachers into one generalist student that outperforms both `PPO` from scratch and offline teacher-only `BC` on both tasks—reaching `411.56 ± 110.40` return and `50.0% ± 18.7%` terminal solved rate (`83.3%` any-step solved) on `myoElbowPose1D6MRandom-v0` vs. `362.43 ± 155.09` (`43.3%`) for `BC` and `260.98 ± 107.11` (`33.3%`) for `PPO`, and `-312.88 ± 2.63` return on the 39-muscle `myoHandPoseRandom-v0` vs. `-319.37 ± 1.37` for `BC` and `-325.99 ± 11.19` for `PPO`.

- **Hardware:** Windows 10 (`10.0.26300`), `AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD` (24 logical CPUs), Python `3.11.18`, PyTorch `2.13.0+cpu`, `SyncEnvPool` × 8 envs.
- **Untimed Specialist Pre-training Diagnostics (3-seed mean):** On `myoElbowPose1D6MRandom-v0` (split on `r_elbow_flex` at `1.135 rad`), specialist pre-training took `17.32 s` per seed; each specialist averaged `426.1` return in its own half-space (`teacher_low` dropped to `172.4` on the full range). On `myoHandPoseRandom-v0` (split on `md5_flexion_r` at `0.793 rad`), specialist pre-training took `38.28 s` per seed.

### Final Results (`seeds = 0, 1, 2`, 20 deterministic episodes per seed)

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
