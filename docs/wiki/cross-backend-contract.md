# Cross-Backend Contract

**Read this before adding mjlab support to a task that will be evaluated on CPU, mjlab, or browser (mjswan).**

The two supported backends are **CPU (`MyoGymnasiumEnv`)** and **GPU (mjlab
`ManagerBasedRlEnvCfg`)**; the invariants below are what keep a policy trained on
mjlab portable to CPU playback/fine-tune. (An **MJX** backend also exists, but it
is **experimental, may not be maintained long-term, and does not meet these
invariants**: its envs differ from the CPU/mjlab envs in obs order, reward weights,
thresholds and reset/target distributions, so MJX-trained policies are not
portable. See the limitations in `myosuite/envs/myo/backends/mjx/README.md`.
Don't build new work on it.)

A policy is only portable across backends if all five invariants below hold. Violating any produces wrong behaviour at eval time without an obvious error.

---

## The Five Invariants

### 1. Observation vector — identical shape, order, and units

- Obs terms registered in the same order in `ObservationGroupCfg` (mjlab) and the browser config.
- Per-term `scale` and `clip` must match exactly — these are applied in TypeScript at inference.
- **No implicit clipping.** CPU observation spaces are float32 `Box(-inf, inf)` and observations are
  only cast to float32; neither backend clips beyond an explicit per-term `clip`.
- **Fresh derived quantities.** Observations, rewards and terminations read positions, `cvel`,
  actuator length/velocity/force, sensors and contacts of the post-step state. CPU `step()` runs
  `mj_forward` after `mj_step` (`MyoGymnasiumEnv._step_physics`). mjlab scores rewards and terminations
  before its own post-step `forward()`, so twins whose rewards or terminations read derived quantities
  register `mdp.sync_forward` as their first termination term.
- **Browser (mjswan) export: no `VecNormalize` or running mean/std.** The browser runtime cannot load normalization statistics. Express normalization as fixed `scale` in `ObservationTermCfg` and train with `obs_normalization=False`.
- **CPU / mjlab / ONNX: running normalization is allowed.** The basic-suite runner (`tasks/rl.py`) enables it by default; the frozen statistics are folded into the policy by `load_rslrl_policy` / `export_rslrl_to_onnx`, so the policy still takes the raw CPU observation vector. SB3 `VecNormalize` statistics are folded the same way by `export_sb3_to_onnx(vec_normalize=...)` and `OnnxCheckpointCallback`.

### 2. Action space — identical dimensionality, scaling, and activation

| Training class | Browser equivalent | Scaling |
|---|---|---|
| `MyoMuscleActivationAction` | `MuscleActionCfg` | `sigmoid(x)` → ctrl ∈ [0, 1] |
| `TendonLengthActionCfg` | `JointPositionActionCfg` | `scale × x + default_pos` |

`encoder_bias` in the exported JSON must equal `model.key_qpos[0]` — ensure the XML has a keyframe at index 0 representing the neutral pose.

### 3. Control timing — identical `ctrl_dt`

```
ctrl_dt = decimation × sim_dt
```

mjswan hardcodes decimation as `round(0.02 / model.opt.timestep)` (50 Hz control). Use only timesteps that are integer divisors of 0.02:

| `timestep` | Browser decimation | ctrl_dt |
|---|---|---|
| 0.002 s | 10 | 0.02 s ✓ |
| 0.001 s | 20 | 0.02 s ✓ |
| 0.003 s | 7 | 0.021 s ✗ |

### 4. Model XML — identical physics

- Use `myo_sim.get_path(...)` in the mjlab `spec_fn` — never hardcode paths.
- The browser model must come from the same source XML used during training.
- Do not strip actuators, sensors, or meshes for deployment.
- If DR is applied during training (mass, friction), the browser uses the nominal model — policies must be robust to this by design.

### 5. Observation normalization — per-term scale/clip only

Fixed `scale` in `ObservationTermCfg` is exported to JSON and replicated in TypeScript. `VecNormalize` stats are not.

---

## Sensorimotor Delay and Observation Noise

`SensorimotorCfg(obs_delay_steps, action_delay_steps, obs_noise_std)`
(`myosuite/core/sensorimotor.py`, default off) is the `sensorimotor` kwarg of a
CPU registration (or of `gym.make`). The mjlab twin reads it from the same
registration through `cpu_reference` (like `muscle_condition`). Counts are
**control steps**: delay in ms = steps × `ctrl_dt` × 1000 (20 ms per step for
`ctrl_dt = 0.02 s`). Both backends implement exactly:

| | Semantics | CPU (`MyoGymnasiumEnv`) | mjlab |
|---|---|---|---|
| Obs delay `k` | Policy obs at step `t` = obs computed at step `max(0, t − k)` since the last reset; a reset fills the history with the reset obs | Base class wraps every subclass `step`/`reset` (once, also for `.unwrapped` and `super()` chains) and delays the flat obs vector | `DelayedObservation` on every `actor` term; partial resets (`reset(env_ids=...)`) fill only the reset rows |
| Action delay `k` | Raw policy action of step `t` is applied at step `t + k`, **before** the action → excitation mapping (clip, sigmoid, fatigue, reafferentation, and later motor noise); the first `k` steps after a reset apply the raw action **0** | Delayed before the task's own `step` (clip + `_apply_action`) | `MyoAction.process_actions`, before the same mapping |
| Obs noise `σ` | i.i.d. `N(0, σ²)` added to every element of the policy obs, **after** the delay (fresh noise every step, also during the reset fill) | `np_random.normal`, drawn only when `σ > 0` | mjlab `GaussianNoiseCfg` on the `actor` terms (`enable_corruption=True`), torch RNG |

- Raw action 0 maps to an excitation of `sigmoid(5 (0 − 0.5)) ≈ 0.076` for
  sigmoid-mapped muscles, 0 for the leg-walk envs (action space `[0, 1]`, no
  sigmoid) and the middle of `ctrlrange` for normalized motors.
- Rewards, terminations, the mjlab `critic` group and CPU `info["obs_dict"]`
  use the current, noise-free state. mjlab `raw_action` is the policy's
  undelayed action; `processed_action` is the ctrl actually applied.
- Delays draw no random numbers on either backend, so enabling a delay does
  not change any reset, target or noise draw. (mjlab's own
  `ObservationTermCfg.delay_*_lag` is not used: its `DelayBuffer` draws
  `torch.randint` every step even for a fixed lag, and mjlab 1.4 and 1.6 handle
  partial resets differently. Actuator `delay_*_lag` counts physics substeps and
  acts after the excitation mapping.)
- Supported by the `cpu_reference` twins (pose, reach, leg stand / walk /
  directional). Sensorimotor is part of the task, so it stays on in the `play`
  config.
- Tests: `test_sensorimotor.py` (CPU), `test_sensorimotor_mjlab.py` (twin and
  CPU ↔ mjlab index-shift equality).

---

## Export Checklist

- [ ] Obs term key order identical between mjlab and browser config (assert in `test_mjlab_task_builder.py`)
- [ ] Obs term `scale` and `clip` values match exactly
- [ ] Browser export only: no running mean/std normalization (`obs_normalization=False`)
- [ ] `ctrl_dt = decimation × timestep = 0.02 s`
- [ ] XML path via `myo_sim.get_path(...)`
- [ ] Keyframe at index 0 for `encoder_bias`
- [ ] Action class exported correctly
- [ ] Smoke-test rollout: 10 steps in browser without NaN

---

## Parity Status

| Task | gym ↔ mjlab obs | gym ↔ mjlab action | mjswan export | Blocker |
|---|---|---|---|---|
| Elbow | ✓ 9D | ✓ 6D sigmoid | ✗ | No TypeScript: `pose_err`, `act`, `qvel×ctrl_dt` |
| Walk | ✓ 403D | ✓ 80D sigmoid | ✗ | No TypeScript: all 12 custom obs terms |
| TableTennis | ✓ 417D | ~ | ✗ | Custom obs term (no TypeScript) |
| ChaseTag FBP2 | ✓ 537D (`test_chasetag_fbp2_parity.py`) | ✓ 354D direct | ✗ | No TypeScript: `chasetag_obs` blocks, scripted opponent; ctrl_dt 0.01 s |

All passing parity tests live in `myosuite/tests/test_mjlab_task_builder.py`.

**To unblock mjswan export:** either contribute TypeScript obs term implementations to the [mjswan repo](https://github.com/ttktjmt/mjswan), or rewrite mjlab obs functions to use only mjswan built-ins (`joint_pos_rel`, `joint_vel_rel`, etc.) where semantically equivalent.
