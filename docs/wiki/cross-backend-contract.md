# Cross-Backend Contract

**Read this before adding mjlab support to a task that will be evaluated on CPU, mjlab, or browser (mjswan).**

The two supported backends are the **CPU env** (`MyoGymnasiumEnv`) and the **GPU env** (mjlab `ManagerBasedRlEnvCfg`).
One `env_id` has one CPU half and one mjlab half (the "twin"). A policy trained on mjlab is portable to CPU playback and
fine-tuning only if the five invariants below hold; violating one gives wrong behaviour at evaluation time without an
obvious error.

The **MJX** (JAX) backend is experimental and may not be maintained. Its envs differ in observation order, reward
weights, thresholds and reset distributions, so MJX-trained policies are not portable. Do not build new work on it
(limitations: [myosuite/envs/myo/backends/mjx/README.md](../../myosuite/envs/myo/backends/mjx/README.md)).

---

## The Five Invariants

### 1. Observation vector — identical shape, order, and units

- Obs terms registered in the same order in `ObservationGroupCfg` (mjlab) and the browser config.
- Per-term `scale` and `clip` must match exactly — these are applied in TypeScript at inference.
- **No implicit clipping.** CPU observation spaces are float32 `Box(-inf, inf)`; observations are only cast to
  float32. Neither backend clips beyond an explicit per-term `clip`.
- **Fresh derived quantities.** Observations, rewards and terminations read positions, `cvel`, actuator
  length/velocity/force, sensors and contacts of the post-step state. CPU `step()` runs `mj_forward` after `mj_step`
  (`MyoGymnasiumEnv._step_physics`). mjlab scores rewards and terminations before its own post-step `forward()`, so a
  twin whose rewards or terminations read derived quantities registers `mdp.sync_forward` as its first termination term.
- **Browser (mjswan): no running normalization.** The browser cannot load `VecNormalize` or running mean/std
  statistics. Use a fixed `scale` in `ObservationTermCfg` and train with `obs_normalization=False`.
- **CPU, mjlab and ONNX: running normalization is allowed.** The basic-suite runner (`tasks/rl.py`) enables it by
  default. The frozen statistics are folded into the policy (`load_rslrl_policy`, `export_rslrl_to_onnx`; for SB3
  `export_sb3_to_onnx(vec_normalize=...)` and `OnnxCheckpointCallback`), so the policy still takes the raw CPU
  observation vector.

### 2. Action space — identical dimensionality, scaling, and activation

| Training class | Browser equivalent | Scaling |
|---|---|---|
| `MyoAction` (twins of the CPU envs; `MyoMuscleActivationAction` on the `TaskConfig` route) | `MuscleActionCfg` | `sigmoid(x)` → ctrl ∈ [0, 1] (the walk envs use the action as-is in [0, 1]) |
| `TendonLengthActionCfg` | `JointPositionActionCfg` | `scale × x + default_pos` |

`encoder_bias` in the exported JSON must equal `model.key_qpos[0]` — ensure the XML has a keyframe at index 0 representing the neutral pose.

### 3. Control timing — identical `ctrl_dt`

```
ctrl_dt = decimation × sim_dt
```

On both backends the control step is set with `EnvConfig(env_id, ctrl_dt=...)` (see [Building an env](#building-an-env-make_env)).

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

For the browser export, only the fixed `scale` and `clip` of `ObservationTermCfg` are exported to JSON and replicated in
TypeScript; `VecNormalize` statistics are not.

---

## Task Parameters

- mjlab twins read every task parameter (target ranges and sampler, thresholds such as `far_th`, reward
  weights, episode length) from the CPU registration through `tasks/cpu_reference.py`. Change the CPU
  registration, never the twin.
- A reset must not start in a terminal state. For example, a reach task's far threshold must exceed the
  farthest target its sampler can draw from the start pose (`test_reach_far_threshold.py`, both backends).

## Muscle-command features (wrappers)

Motor noise, fatigue, reafferentation and sarcopenia are **wrappers** (`myosuite.envs.wrappers`), not env kwargs:

| Wrapper | Effect |
|---|---|
| `MotorNoiseWrapper` | noise on the muscle excitations (see [Motor noise](#motor-noise-optional-off-by-default)) |
| `FatigueWrapper` | 3CC-r muscle fatigue |
| `ReafferentationWrapper` | reroutes one muscle's command to another (hand models: EIP to EPL) |
| `SarcopeniaWrapper` | scales the muscle forces once, by editing the model |

- **Off by default.** A plain env runs only its own action map. The ids `myoFati*`, `myoSarc*` and `myoReaf*` are the base
  env registered with the matching wrapper (`condition_wrapper_specs(...)` builds the spec); noise additionally needs a
  nonzero level.
- **Removed kwargs.** `muscle_condition`, `fatigue_reset_vec`, `fatigue_reset_random` and `motor_noise` no longer exist;
  an env raises a `TypeError` that names the replacement.
- **Once per env.** Applying the same wrapper twice, or a wrapper the id already registers (`FatigueWrapper` on
  `myoFati*`), raises a `ValueError`. Wrap the base id instead, or change the installed wrapper's options
  (`env.set_motor_noise(...)`, `env.set_fatigue_reset_random(...)`).
- **Which envs.** The CPU envs that run these wrappers: pose, key turn, object hold, pen, SAR reorient, reach (arm,
  finger, hand), torso, leg, the MyoChallenge muscle envs and the experimental `ModularTaskEnv`. A wrapper on any other
  env (for example MuscleMimic) raises a `TypeError`.
- **`TaskConfig` holds the task only.** It has no fatigue, condition or noise fields. Its variants register wrappers
  like any other id (`VariantSpec(features=...)`).

### Stage order

Each wrapper installs one **stage** in the env's action pipeline. The built-in stages always run in this order,
whatever the wrapping order:

```
env action map -> noise -> fatigue -> reroute -> custom stages -> write ctrl
```

The env's own map (sigmoid, as-is for the walk envs, or clipped `ctrl` with `normalize_act=False`) stays inside the env:
`normalize_act` also sets the initial joint pose, so the map is not a wrapper. mjlab runs the same stages in `MyoAction`,
built from the registration's wrapper specs.

### Custom stages

A custom stage runs after the built-in ones, in the order it was installed (the wrapping order, or the order of
`EnvConfig.features`). To insert one earlier, pass an explicit `order` between 10 and 100 (the built-ins are noise 20,
fatigue 30, reroute 40; the env map is 10 and the `ctrl` write 100). Two stages with the same explicit order raise a
`StageOrderWarning`.

- **Portable (CPU and mjlab):** subclass `ExcitationStage` (`__call__(u, xp)` on the muscle excitations, written for
  numpy and torch; optional `reset(env_ids)` for state; `LowPassStage` is a shipped example) and add it with
  `ExcitationStageWrapper(env, factory)`. The mjlab twin builds the stage from the same factory when it is registered in
  `additional_wrappers`. The factory must be a module-level callable (a class or `functools.partial`) because some tools
  pickle the env. The CPU env and the twin are tested to agree (`test_motor_noise_mjlab.py`).
- **Env-aware (CPU only):** `CtrlStageWrapper(env, apply, name=..., order=None)` with `apply(env, ctrl) -> ctrl`. It
  receives the env, so it cannot run on mjlab.
- **Acting on the raw `[-1, 1]` action** (a delay, say) is not a stage: use a plain `gym.ActionWrapper` on the outside.

## Building an env: `make_env`

`make_env` is the one way to build an env in code, tests, docs and tutorials. `from myosuite import make_env` also
registers every env id. On the CPU it calls `gym.make`, which still works.

**By name only** (the registered defaults of the env id):

```python
from myosuite import make_env

env_cpu = make_env("myoElbowPose1D6MRandom-v0")                                # CPU env
env_gpu = make_env("myoElbowPose1D6MRandom-v0", backend="mjlab", num_envs=4096)   # same task on the GPU (mjlab twin)
```

**With an `EnvConfig`**, which overrides the registered defaults for every backend in one place:

```python
from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.envs.wrappers import FatigueWrapper, MotorNoiseWrapper

cfg = EnvConfig(
    "myoElbowPose1D6MRandom-v0",
    max_episode_steps=300,
    ctrl_dt=0.01,
    features=((MotorNoiseWrapper, {"motor_noise": {"constant_std": 0.05}}), FatigueWrapper),
)
env_cpu = make_env(cfg)                                # CPU, with noise and fatigue
env_gpu = make_env(cfg, backend="mjlab", num_envs=4096)   # GPU twin, with the same noise and fatigue
```

Each entry of `features` is a wrapper class (`FatigueWrapper`), a `(class, kwargs)` pair, or a `WrapperSpec`
(`wrapper_spec(cls, **kwargs)`; the `additional_wrappers` of a registration take specs).

| `EnvConfig` field | What it sets | On the CPU env | On the mjlab twin |
|---|---|---|---|
| `features` | muscle-command wrappers | wrapped around the env | added to the CPU registration the twin is built from |
| `max_episode_steps` | episode length limit | passed to `gym.make` | sets the twin's episode length |
| `ctrl_dt` | control step in seconds (physics substeps follow) | sets `frame_skip = ctrl_dt / model timestep` | sets the twin's `decimation` to the same value |
| `num_envs` | parallel envs | must be 1 | number of parallel envs |
| `task_kwargs` | env constructor kwargs | passed to the env | waypoint twins accept scene/task overrides; other twins read the registration |
| `backend_options`, extra kwargs | options of one backend only | passed to `gym.make` (for example `render_mode`) | passed to the env (for example `device`) |

- `ctrl_dt` must be a whole multiple of the model timestep (`ValueError` otherwise). The `ModularTaskEnv` ids take their
  timing from `task_config.backend`, so `ctrl_dt` raises a `ValueError` for them on both backends.
- A feature a backend cannot run raises; it is never dropped silently (see the tables below).
- The training and evaluation scripts take the same features as `--feature NAME[=JSON]` flags
  (`scripts/train_mjlab.py`, `scripts/eval_mjlab_policy.py`; names `motor-noise`, `fatigue`, `sarcopenia`,
  `reafferentation`, see `myosuite/utils/feature_cli.py`).
- Harness wrappers (`PerturbationWrapper`, recording, a delay on the raw action) are not features. They are plain
  gymnasium wrappers on the CPU env and not part of the config.

## Which features run where

Table 1: **which backend can run each feature** (a feature is a wrapper in `EnvConfig.features` or in an id's
registration).

| Feature | CPU env | mjlab twin | MJX (experimental) |
|---|---|---|---|
| `MotorNoiseWrapper`, `FatigueWrapper` | yes | yes | no |
| `ReafferentationWrapper` (hand models with EIP/EPL) | yes | yes | no |
| `SarcopeniaWrapper` (model edit) | yes | yes | yes |
| `ExcitationStageWrapper` (portable custom stage) | yes | yes | no |
| `CtrlStageWrapper` (env-aware custom stage) | yes | no | no |

Table 2: **which envs take features, per backend**. The twin is rebuilt from the CPU registration, so it takes the
features of that registration or of the `EnvConfig`.

| Env family | CPU env takes features | Has an mjlab twin | mjlab twin takes features |
|---|---|---|---|
| Pose and reach (hand, finger, arm, elbow, motor finger), torso, leg walk / stand / terrain, leg directional | yes | yes | yes |
| Key turn, object hold, pen, reorient (SAR, ID, OOD, 100, 8) | yes | no | - |
| MyoChallenge Baoding, Bimanual, Relocate, Die Reorient, Soccer, OSL run | yes | no | - |
| ChaseTag, single agent | yes | only `myoChallengeChaseTagFBP2-v0` | no |
| Waypoint (`myoFullBodyWaypoint-v0`, `WaypointEnv`) | no | yes | no |
| Table Tennis P0, P1, P2 | yes | yes | no |
| ChaseTag against a scripted opponent (multi-agent) | no | no | - |
| MuscleMimic (fullbody, bimanual, directional) | no | yes, registered separately (own implementation) | no |

Unsupported combinations raise: `NotImplementedError` from `make_env` on mjlab or MJX, `TypeError` from a wrapper on an
env that does not run stages. The condition ids (`myoFati*`, `myoSarc*`, `myoReaf*`) have mjlab twins only for the
families whose last column is "yes". `make_env` is the authority.

## Motor noise (optional, off by default)

`MotorNoiseCfg(signal_dependent_std, constant_std)` (`myosuite.terms.base_action`) adds human motor noise to the muscle
excitations. `MotorNoiseCfg.van_beers_2004()` gives the levels 0.103 / 0.185 used by Fischer et al. (2021) and
User-in-the-Box.

- **Semantics.** `u' = clip(u + σ_sd·u·n1 + σ_c·n2, 0, 1)` with `n1, n2 ~ N(0, 1)` drawn independently per muscle and per
  control step (and per env on mjlab). The sample is held over the `frame_skip` / decimation substeps. Motors are never
  noised.
- **Position in the pipeline.** After the env's action map and before fatigue (see [Stage order](#stage-order)). Both
  backends use the same pure term `motor_noise`; for the same normals the CPU and mjlab outputs match exactly.
- **Random numbers.**
  - CPU: the env's `np_random` (seeded by `reset(seed=...)`), drawn only while noise is on, so default rollouts are
    unchanged. With noise on, the draws advance `np_random`, so later per-episode samples (targets) differ from the
    noiseless env unless every episode is reseeded.
  - mjlab: `torch.randn` on the sim device from the global torch RNG that mjlab seeds (`seed_rng`).
  - The backends agree in distribution, not sample by sample.
- **Configuration.** CPU: `MotorNoiseWrapper(env, MotorNoiseCfg.van_beers_2004())`, with a cfg or a dict of its fields;
  `env.set_motor_noise(...)` changes the levels. Both backends: put a `MotorNoiseWrapper` spec in `EnvConfig.features` or
  in the id's registration. A single mjlab config: `env_cfg.actions["muscles"].motor_noise`.
- **Clipping.** Near the bounds the clip rectifies the noise: at `u = 0.076` (policy output 0 with the sigmoid) the van
  Beers levels clip 34 % of the samples to 0 and raise the mean excitation to 0.118 (see
  `docs/source/quickstart_neuroscience.rst`).

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
| Walk | ✓ 403D | ✓ 80D, [0, 1] activations (no sigmoid) | ✗ | No TypeScript: all 12 custom obs terms |
| TableTennis | ✓ 417D | ✓ 275D | ✗ | Custom obs term (no TypeScript) |
| ChaseTag FBP2 | ✓ 537D (`test_chasetag_fbp2_parity.py`) | ✓ 354D direct | ✗ | No TypeScript: `chasetag_obs` blocks, scripted opponent; ctrl_dt 0.01 s |

The parity tests are `test_mjlab_cpu_twins.py` (basic suite, per family: see `docs/source/backend_parity.rst`),
`test_mjlab_task_builder.py`, `test_chasetag_fbp2_parity.py`, `test_table_tennis_mjlab_parity.py` and
`test_tabletennis_cpu_mjlab_parity.py`.

**To unblock mjswan export:** either contribute TypeScript obs term implementations to the [mjswan repo](https://github.com/ttktjmt/mjswan), or rewrite mjlab obs functions to use only mjswan built-ins (`joint_pos_rel`, `joint_vel_rel`, etc.) where semantically equivalent.
