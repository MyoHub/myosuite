# Differentiable Short-Horizon Actor-Critic (`newton_shac`)

First-order differentiable policy learning for MyoSuite musculoskeletal environments (`PoseEnvV0`) using Short-Horizon Actor-Critic (SHAC) over batched GPU `mujoco_warp` rollouts with analytic Hill-type muscle, tendon, and implicit-constraint vector-Jacobian products (VJPs).

> **Summary vs. PPO:** Under an identical 102,400-step budget across 3 seeds (`[0, 1, 2]`), `newton_shac` improves final unmodified task return over matched PPO from `751.34 ± 3.59` to `769.34 ± 1.52` on `myoElbowPose1D6MRandom-v0` (surpassing PPO's 102.4k-step return by 53,248 steps in equal ~33 s wall-clock time) and from `-322.06 ± 6.60` to `-252.38 ± 1.65` (`+69.68` return; `+64.38` on CPU `make_env` playback) on 39-muscle `myoHandPoseRandom-v0`, at ~2.0× wall-clock time per step due to the analytic backward pass.

---

## Attribution & License

- **SHAC algorithm:** Jie Xu, Viktor Makoviychuk, Yashraj Narang, Fabio Ramos, Wojciech Matusik, Animesh Garg, and Dieter Fox, *"Accelerated Policy Learning with Parallel Differentiable Simulation"*, ICLR 2022 ([arXiv:2204.07137](https://arxiv.org/abs/2204.07137); reference repository [`NVlabs/DiffRL`](https://github.com/NVlabs/DiffRL)).
- **PPO baseline:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov, *"Proximal Policy Optimization Algorithms"*, 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)).
- **Clean-room implementation:** All code in `training/newton_shac/` is a new Apache-2.0 MyoSuite implementation written from the published equations; no code was copied or adapted from `NVlabs/DiffRL`.

---

## Installation

Per `docs/wiki/training-guidelines.md`, `newton_shac` is isolated from the core `myosuite` package dependencies:

```bash
pip install -e .
pip install -e "training/newton_shac[test]"
```

**Analytic muscle-tendon VJP dependency:** Analytic GPU backward passes (`mjw.enable_grad()` + `mujoco_warp._src.adjoint.step_backward`) require `mujoco_warp` with analytic Hill-type muscle and spatial/wrapping tendon adjoints ([`google-deepmind/mujoco_warp#1970`](https://github.com/google-deepmind/mujoco_warp/pull/1970)). When installed against an older `mujoco_warp` release without `_src.adjoint`, `DifferentiablePoseEnv` automatically falls back to CPU central finite differences for small-batch verification and unit tests.

---

## Method Details

1. **Task Contract Preservation:** `DifferentiablePoseEnv` loads the compiled `MjModel`, `frame_skip`, `obs_keys`, `rwd_keys_wt`, `pose_thd`, `_init_qpos`, target joint ranges, and `sigmoid_muscle_activation` (`1 / (1 + exp(-5 (a - 0.5)))`) directly from `make_env(env_id)` without modifying registered environments.
2. **Per-Control-Step Checkpointed Autograd (`_MuscleStepFn`):** Forward execution steps `frame_skip` `mujoco_warp` substeps on GPU; `backward()` replays the `frame_skip` substeps from saved control-step boundary tensors `(qpos, qvel, act, ctrl)` and propagates analytic implicit-differentiation + muscle/tendon adjoints in reverse, keeping memory $O(\text{frame\_skip})$ independent of horizon length $H$.
3. **Smooth Differentiable Actor Surrogate vs. Exact Evaluation:** `PoseEnvV0`'s discrete threshold bonus `(dist < pose_thd) + (dist < 1.5 * pose_thd)` has zero gradient almost everywhere; during SHAC actor/critic updates, those two indicator steps are smoothed with `sigmoid((thd - dist) / 0.05)`. **All reported benchmark metrics and PPO updates use the unmodified discrete `PoseEnvV0` reward and `solved` flag**, verified on both GPU and CPU `make_env(env_id)`.

---

## Benchmark Against Matched PPO

### Experimental Setup

- **Hardware:** `NVIDIA GeForce RTX 4070 Ti SUPER` (16 GiB, CUDA, `torch 2.14.1+cu126`, `warp 1.19.0`)
- **Environments:** `myoElbowPose1D6MRandom-v0` (1 DOF, 6 muscles, 100-step episodes) and `myoHandPoseRandom-v0` (23 DOFs, 39 muscles with spatial & cylinder/sphere-wrapping tendons, 100-step episodes)
- **Seeds:** `3` independent seeds (`0, 1, 2`) per method and environment (`12` training runs total, after JIT warmup)
- **Matched Hyperparameters:** `num_envs = 256`, rollout horizon `H = 16` (`4,096` steps/iter), budget `102,400` env steps (`25` iterations), shared `(128, 128)` LayerNorm-ELU `ActorCritic` + Welford `RunningMeanStd`, `gamma = 0.97`, `lam = 0.95`, `max_grad_norm = 1.0`, deterministic evaluation across `64` GPU episodes per checkpoint plus `16` episodes on unmodified CPU `make_env(env_id)`.

### Reproduce

```bash
python -m newton_shac.benchmark \
  --envs myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0 \
  --seeds 0 1 2 \
  --num-envs 256 \
  --horizon 16 \
  --total-steps 102400 \
  --eval-interval 25600
```

### Final Summary (Mean ± Std Across Seeds `0, 1, 2` at `102,400` Env Steps)

| Environment | Method | GPU Eval Return | GPU Solved Rate | CPU `make_env` Return | CPU Solved Rate | Wall-Clock Time (s) | Peak PyTorch VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `myoElbowPose1D6MRandom-v0` | **Newton-SHAC** | **769.34 ± 1.52** | **100.0% ± 0.0%** | **769.07 ± 1.17** | **100.0%** | 67.97 ± 0.26 | 32.4 |
| `myoElbowPose1D6MRandom-v0` | PPO | 751.34 ± 3.59 | 100.0% ± 0.0% | 754.35 ± 1.18 | 100.0% | **33.00 ± 1.22** | **24.6** |
| `myoHandPoseRandom-v0` | **Newton-SHAC** | **-252.38 ± 1.65** | 0.0% ± 0.0% | **-261.95 ± 0.36** | 0.0% | 72.78 ± 0.90 | 45.7 |
| `myoHandPoseRandom-v0` | PPO | -322.06 ± 6.60 | 0.0% ± 0.0% | -326.33 ± 11.75 | 0.0% | **37.55 ± 1.36** | **30.3** |

*(Note: On 23-DOF, 39-muscle `myoHandPoseRandom-v0`, strict `solved` requires simultaneous `< 0.35 rad` Euclidean error across all 23 joints, which requires longer multi-million-step budgets; over the first 102.4k steps, analytic muscle-tendon gradients drive steady pose-error reduction whereas zeroth-order PPO exploration has not yet escaped the initial random-pose plateau.)*

### Learning Curves (Env Steps & Wall-Clock Time)

#### `myoElbowPose1D6MRandom-v0`

| Env Steps | SHAC Wall Time (s) | SHAC Return | SHAC Solved | PPO Wall Time (s) | PPO Return | PPO Solved |
| ---: | ---: | :---: | :---: | ---: | :---: | :---: |
| 0 | 0.0 | 96.86 ± 1.26 | 45.8% | 0.0 | 96.86 ± 1.26 | 45.8% |
| 28,672 | 17.2 | 611.42 ± 204.26 | 95.3% | 8.4 | 392.82 ± 141.31 | 73.4% |
| 53,248 | 33.4 | 758.63 ± 1.25 | 99.5% | 15.5 | 717.21 ± 17.79 | 97.9% |
| 77,824 | 49.0 | 765.76 ± 1.21 | 100.0% | 22.9 | 744.77 ± 9.96 | 99.5% |
| 102,400 | 65.0 | **769.34 ± 1.52** | **100.0%** | 30.0 | 751.34 ± 3.59 | **100.0%** |

#### `myoHandPoseRandom-v0`

| Env Steps | SHAC Wall Time (s) | SHAC Return | PPO Wall Time (s) | PPO Return |
| ---: | ---: | :---: | ---: | :---: |
| 0 | 0.0 | -317.13 ± 0.42 | 0.0 | -317.13 ± 0.42 |
| 28,672 | 19.5 | -291.16 ± 5.82 | 9.5 | -315.09 ± 4.24 |
| 53,248 | 36.2 | -268.34 ± 1.20 | 17.2 | -320.40 ± 4.24 |
| 77,824 | 53.5 | -259.36 ± 0.37 | 26.2 | -322.28 ± 8.25 |
| 102,400 | 69.7 | **-252.38 ± 1.65** | 33.7 | -322.06 ± 6.60 |

---

## Running Tests

```bash
pytest training/newton_shac/tests/test_newton_shac.py -v
```
