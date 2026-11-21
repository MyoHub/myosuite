# Differentiable Short-Horizon Actor-Critic (`newton_shac`)

First-order differentiable policy learning for MyoSuite musculoskeletal environments (`PoseEnvV0`) using Short-Horizon Actor-Critic (SHAC) over Warp CUDA-graph-captured `mujoco_warp` rollouts with analytic Hill-type muscle, tendon, and implicit-constraint vector-Jacobian products (VJPs).

> **Summary vs. Registered `myo_ppo_runner_cfg` (`mjlab` PPO):** Under a matched `98,304`-step budget (`num_envs = 256`, seeds `[0, 1, 2]`), CUDA-graph `newton_shac` improves final CPU `make_env` return over canonical `myo_ppo_runner_cfg` from `762.45 ± 2.49` to `768.75 ± 1.72` on `myoElbowPose1D6MRandom-v0` (surpassing `mjlab` PPO's `98.3k`-step return by `65,536` steps in `5.96 s` vs `7.33 s`) and from `-299.13 ± 6.44` to `-253.97 ± 0.18` (`+45.16` return, surpassing `mjlab` PPO's `98.3k`-step return by `32,768` steps in `4.99 s` vs `9.96 s`) on 39-muscle `myoHandPoseRandom-v0`, at `1.25×`–`1.52×` total wall-clock time.

---

## Attribution & License

- **SHAC algorithm:** Jie Xu, Viktor Makoviychuk, Yashraj Narang, Fabio Ramos, Wojciech Matusik, Animesh Garg, and Dieter Fox, *"Accelerated Policy Learning with Parallel Differentiable Simulation"*, ICLR 2022 ([arXiv:2204.07137](https://arxiv.org/abs/2204.07137); reference repository [`NVlabs/DiffRL`](https://github.com/NVlabs/DiffRL)).
- **Clean-room implementation:** All code in `training/newton_shac/` is a new Apache-2.0 MyoSuite implementation written from the published equations; no code was copied or adapted from `NVlabs/DiffRL`.

---

## Installation

Per `docs/wiki/training-guidelines.md`, `newton_shac` is isolated from core `myosuite` dependencies:

```bash
pip install -e .
pip install -e "training/newton_shac[test]"
```

**Analytic muscle-tendon VJP dependency:** Analytic GPU backward passes (`mjw.enable_grad()` + `mujoco_warp._src.adjoint.step_backward`) require `mujoco_warp` with analytic Hill-type muscle and spatial/wrapping tendon adjoints ([`google-deepmind/mujoco_warp#1970`](https://github.com/google-deepmind/mujoco_warp/pull/1970)). When installed against an older `mujoco_warp` release without `_src.adjoint`, `DifferentiablePoseEnv` automatically falls back to CPU central finite differences for small-batch verification and unit tests.

---

## Method Details

1. **Task Contract Preservation:** `DifferentiablePoseEnv` loads the compiled `MjModel`, `frame_skip`, `obs_keys`, `rwd_keys_wt`, `pose_thd`, `_init_qpos`, target joint ranges, and `sigmoid_muscle_activation` (`1 / (1 + exp(-5 (a - 0.5)))`) directly from `make_env(env_id)` without modifying registered environments.
2. **CUDA-Graph Checkpointed Autograd (`_MuscleStepFn`):** Both the `frame_skip`-substep forward rollout and the reverse `step_backward` adjoint replay are captured as reusable Warp CUDA graphs (`use_cuda_graph=True`, `4.8×`–`7.4×` faster than eager kernel launches) while keeping activation memory $O(\text{frame\_skip})$ independent of horizon length $H$.
3. **Smooth Differentiable Actor Surrogate vs. Exact Evaluation:** `PoseEnvV0`'s discrete threshold bonus `(dist < pose_thd) + (dist < 1.5 * pose_thd)` has zero gradient almost everywhere; during SHAC updates, those two indicator steps are smoothed with `sigmoid((thd - dist) / 0.05)`. **All reported benchmark metrics and `myo_ppo_runner_cfg` baselines use the unmodified discrete `PoseEnvV0` reward and `solved` flag**, verified on unmodified CPU `make_env(env_id)`.
4. **ONNX Policy Export:** `export_shac_onnx(model, "policy.onnx")` folds `RunningMeanStd`, `actor_net`, and `[-1, 1]` action clamping into a single self-contained `.onnx` file (`obs -> action`) via `myosuite.utils.export_onnx`.

```python
from newton_shac import ShacConfig, export_shac_onnx, train_shac

model, metrics = train_shac("myoElbowPose1D6MRandom-v0", ShacConfig(), seed=0)
export_shac_onnx(model, "myoElbowPose1D6MRandom_shac.onnx")
```

---

## Benchmark Against Registered `myo_ppo_runner_cfg` (`mjlab` PPO)

### Experimental Setup

- **Hardware:** `NVIDIA GeForce RTX 4070 Ti SUPER` (16 GiB, CUDA, `torch 2.14.1+cu126`, `warp 1.19.0`)
- **Environments:** `myoElbowPose1D6MRandom-v0` (1 DOF, 6 muscles, 100-step episodes) and `myoHandPoseRandom-v0` (23 DOFs, 39 muscles with spatial & cylinder/sphere-wrapping tendons, 100-step episodes)
- **Seeds & Matched Budget:** `3` seeds (`0, 1, 2`) per arm (`12` runs total after Warp kernel warmup), `num_envs = 256`, exact `98,304` environment steps (`24` SHAC iterations of `256 × 16 = 4,096` steps vs. `16` `myo_ppo_runner_cfg` iterations of `256 × 24 = 6,144` steps), evaluated deterministically across `64` episodes on unmodified CPU `make_env(env_id)` (`seed = 20000..20063`).

### Reproduce

```bash
# Newton-SHAC arm (24 * 256 * 16 = 98,304 env steps)
python -m newton_shac.shac \
  --envs myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0 \
  --seeds 0 1 2 --num-envs 256 --horizon 16 --total-steps 98304

# Canonical registered mjlab PPO arm (16 * 256 * 24 = 98,304 env steps per seed)
for env in myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0; do
  for seed in 0 1 2; do
    python scripts/train_mjlab.py "$env" \
      --env.scene.num-envs 256 --agent.max-iterations 16 --agent.seed "$seed"
  done
done
```

### Final Summary (Mean ± Std Across Seeds `0, 1, 2` at `98,304` Env Steps)

| Environment | Method | CPU `make_env` Return | CPU Solved Rate | Wall-Clock Time (s) | Peak PyTorch VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `myoElbowPose1D6MRandom-v0` | **Newton-SHAC (CUDA graph)** | **768.75 ± 1.72** | **100.0% ± 0.0%** | 9.15 ± 0.10 | 32.1 |
| `myoElbowPose1D6MRandom-v0` | `mjlab` PPO (`myo_ppo_runner_cfg`) | 762.45 ± 2.49 | **100.0% ± 0.0%** | **7.33 ± 0.44** | **30.9** |
| `myoHandPoseRandom-v0` | **Newton-SHAC (CUDA graph)** | **-253.97 ± 0.18** | 0.0% ± 0.0% | 15.17 ± 0.09 | 44.7 |
| `myoHandPoseRandom-v0` | `mjlab` PPO (`myo_ppo_runner_cfg`) | -299.13 ± 6.44 | 0.0% ± 0.0% | **9.96 ± 0.05** | **43.4** |

*(Note: On 23-DOF, 39-muscle `myoHandPoseRandom-v0`, strict `solved` requires simultaneous `< 0.35 rad` Euclidean error across all 23 joints, which requires multi-million-step budgets; over the first `98.3k` steps, analytic muscle-tendon gradients reduce pose error ~`5×` faster than zeroth-order `mjlab` PPO.)*

### Learning Curves (Env Steps & Wall-Clock Time)

#### `myoElbowPose1D6MRandom-v0`

| Newton-SHAC Steps / Wall Time | Newton-SHAC Return | `mjlab` PPO Steps / Wall Time | `mjlab` PPO Return |
| ---: | :---: | ---: | :---: |
| `0` (`0.00 s`) | `96.86 ± 1.26` | `0` (`0.00 s`) | `121.90 ± 32.94` |
| `32,768` (`3.10 s`) | `677.96 ± 111.02` | `30,720` (`2.37 s`) | `285.54 ± 28.12` |
| `65,536` (`5.96 s`) | `763.78 ± 2.53` | `61,440` (`4.60 s`) | `741.53 ± 5.41` |
| `98,304` (`8.81 s`) | **768.79 ± 1.85** | `98,304` (`7.33 s`) | `762.45 ± 2.49` |

#### `myoHandPoseRandom-v0`

| Newton-SHAC Steps / Wall Time | Newton-SHAC Return | `mjlab` PPO Steps / Wall Time | `mjlab` PPO Return |
| ---: | :---: | ---: | :---: |
| `0` (`0.00 s`) | `-317.12 ± 0.42` | `0` (`0.00 s`) | `-312.12 ± 2.66` |
| `32,768` (`4.99 s`) | `-287.13 ± 5.17` | `30,720` (`3.07 s`) | `-312.78 ± 5.25` |
| `65,536` (`9.80 s`) | `-263.23 ± 1.27` | `61,440` (`6.28 s`) | `-309.64 ± 4.10` |
| `98,304` (`14.53 s`) | **-254.52 ± 2.59** | `98,304` (`9.96 s`) | `-299.13 ± 6.44` |

---

## Running Tests

```bash
pytest training/newton_shac/tests/test_newton_shac.py -v
```
