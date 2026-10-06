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

## Task Parameters

- mjlab twins read every task parameter (target ranges and sampler, thresholds such as `far_th`, reward
  weights, episode length) from the CPU registration through `tasks/cpu_reference.py`. Change the CPU
  registration, never the twin.
- A reset must not start in a terminal state. For example, a reach task's far threshold must exceed the
  farthest target its sampler can draw from the start pose (`test_reach_far_threshold.py`, both backends).

## Muscle-command stages (wrappers)

Muscle noise, fatigue and reafferentation are **wrappers** (`myosuite.envs.wrappers`), not env
kwargs: `MotorNoiseWrapper`, `FatigueWrapper`, `ReafferentationWrapper` (and `SarcopeniaWrapper`,
which edits the model once). The `myoFati*`, `myoSarc*` and `myoReaf*` ids are registrations of the
base env with the matching wrapper in `additional_wrappers`; `condition_wrapper_specs(...)` builds the
spec. Constructor kwargs `muscle_condition`, `fatigue_reset_vec`, `fatigue_reset_random` and
`motor_noise` no longer exist: an env raises a `TypeError` that names the replacement.

**None of the stages is on by default** (a plain env runs only its own map): the `myoFati*`, `myoReaf*` and `myoSarc*` ids
register the matching wrapper, and noise additionally needs a nonzero level.

Each wrapper installs one **stage** in the env's action pipeline
(`myosuite.envs.muscle_stages.CtrlStageHost.add_ctrl_stage`). The three built-in stages run in a fixed
order, set by the stage and not by the wrapping order; custom stages run after them:

```
clip action -> env map (10: sigmoid, or as-is for the walk envs, or clipped ctrl when
normalize_act=False) -> noise (20) -> fatigue (30) -> reroute (40, reafferentation)
-> custom stages -> ctrl (100)
```

**Custom stages.** Two kinds. Both run after the built-in stages, in the order they were installed (the wrapping
order, or the order of `EnvConfig.features` / the registered wrappers), so there is nothing to configure. To insert one
earlier, pass an explicit `order` strictly between 10 and 100 (for example 25: after noise, before fatigue):

- **Portable:** subclass `ExcitationStage` (`__call__(u, xp)` on the muscle excitations, written for numpy and torch;
  `reset(env_ids)` for state; `LowPassStage` is the shipped example, a first-order filter) and add it with `ExcitationStageWrapper(env, factory)`. A CPU env runs
  it with numpy; the mjlab twin builds the stage from the same factory (registered in `additional_wrappers`, read by
  `cpu_reference.action_cfg` into `MyoActionCfg.excitation_stages`) and runs it with torch on `(n_envs, n_muscles)`.
  The factory must be a module-level callable (class or `functools.partial`) because the wrapped env is pickled by some
  tools. The CPU and the twin agree in `tests/test_motor_noise_mjlab.py`.
- **Env-aware, CPU only:** `CtrlStageWrapper(env, apply, name=..., order=None)` with `apply(env, ctrl) -> ctrl`
  (it gets the host env, so it cannot run on mjlab).

Two stages with the same **explicit** order run in installation order and raise a `StageOrderWarning` on either backend
(also for a clash with a built-in order); stages without an order never clash. To act on the raw `[-1, 1]` action (a delay, say),
use a plain `gym.ActionWrapper` on the outside instead.

mjlab applies the same order in `MyoAction`; `cpu_reference.action_cfg` builds it from the
registration's wrapper specs, so registering an env id with a wrapper configures both halves. The
envs that run stages are the basic pose, key-turn, object-hold, pen, SAR-reorient, arm/finger/hand
reach, torso, leg, the MyoChallenge muscle envs and the experimental `ModularTaskEnv`; wrapping any other env
raises a `TypeError` (for example MuscleMimic). A `TaskConfig` holds the task only: it has no `muscle_fatigue`,
`ActuatorGroupSpec.condition` or `.noise`; its variants (`VariantSpec(features=...)`) register the wrappers like any
other id. A stage is installed **once per env**: a second wrapper of the same kind, or one on an id that
already registers it (`FatigueWrapper` on `myoFati*`, `ReafferentationWrapper` on `myoReaf*`), raises a `ValueError`;
`SarcopeniaWrapper` raises it if sarcopenia is already applied to the model (it would scale the forces twice). Wrap the
base id, or change the installed wrapper's options (`env.set_motor_noise(...)`, `env.set_fatigue_reset_random(...)`).

### One call for every backend: `make_env(EnvConfig(...))`

`make_env` is **the** way to build an env, in the docs, tutorials, scripts and tests: `from myosuite import make_env`
(importing `myosuite` also registers every env id), then `make_env(env_id)` for a plain CPU env or
`make_env(EnvConfig(...), backend=...)` for any backend. On the CPU it calls `gym.make` underneath, so `gym.make(env_id)`
still works, but use `make_env` in new code. The registration of the env id gives the defaults; the config overrides them:

```python
from myosuite.core.config import EnvConfig
from myosuite import make_env
from myosuite.envs.wrappers import FatigueWrapper, MotorNoiseWrapper, wrapper_spec

cfg = EnvConfig(
    "myoElbowPose1D6MRandom-v0",
    max_episode_steps=300,
    ctrl_dt=0.01,
    features=(wrapper_spec(MotorNoiseWrapper, motor_noise=0.05), wrapper_spec(FatigueWrapper)),
)
env = make_env(cfg)                                          # CPU
envs = make_env(cfg, backend="mjlab", num_envs=4096)         # GPU twin, same noise + fatigue
```

| Field | CPU | mjlab |
|---|---|---|
| `features` (wrapper specs) | wrappers around `gym.make` | added to the CPU registration the twin is built from |
| `max_episode_steps` | `make_env(max_episode_steps=...)` | `episode_length_s` of the twin |
| `ctrl_dt` (the one timing knob) | `frame_skip = ctrl_dt / model timestep` | `decimation` (twin rebuilt with that `frame_skip`) |
| `num_envs` | must be 1 | `scene.num_envs` |
| `task_kwargs` | env constructor kwargs | raises (the twin reads the registration) |
| `backend_options` / `**overrides` | `gym.make` kwargs (`render_mode`, ...) | `device`, ... |

`ctrl_dt` must be a whole multiple of the model timestep (`ValueError` otherwise); the `ModularTaskEnv` ids take their
timing from `task_config.backend`, so `ctrl_dt` raises a `ValueError` for them on both backends. Rules: a feature a backend cannot
run raises (`mjx`: only `SarcopeniaWrapper`, a model edit, is supported, the rest raises `NotImplementedError`; a twin that is not built from the CPU
registration, such as MuscleMimic, ChaseTag or Table Tennis: `NotImplementedError`), it is never dropped silently;
adding a wrapper the id already registers (`FatigueWrapper` on `myoFati*`) raises a `ValueError` on both backends.
Harness wrappers (`PerturbationWrapper`, recording, a delay on the raw action) are not features: they are plain
gymnasium wrappers on the CPU env and are not part of the config.

### Which features run where

**Table 1: which backend can run each feature.** A feature is a wrapper in `EnvConfig.features` or in an id's
registration.

| Feature | Runs on the CPU env | Runs on the mjlab twin | Runs on MJX (experimental) |
|---|---|---|---|
| `MotorNoiseWrapper`, `FatigueWrapper` | yes | yes | no |
| `ReafferentationWrapper` (hand models with EIP/EPL) | yes | yes | no |
| `SarcopeniaWrapper` (model edit) | yes | yes | yes |
| `ExcitationStageWrapper` (portable custom stage) | yes | yes | no |
| `CtrlStageWrapper` (env-aware custom stage) | yes | no | no |

**Table 2: which envs accept features, per backend.** "Accepts features" means `make_env` can apply the features of
Table 1 to that env on that backend. The twin is rebuilt from the CPU registration, so it takes the features of the
registration or of `EnvConfig`.

| Env family | CPU env accepts features | mjlab twin exists | mjlab twin accepts features |
|---|---|---|---|
| Pose and reach (hand, finger, arm, elbow, motor finger), torso, leg walk / stand / terrain, `ModularTaskEnv` ids with a twin | yes | yes | yes |
| Hand tasks without a twin: key turn, object hold, pen, reorient (SAR, ID, OOD, 100, 8) | yes | no | n/a |
| MyoChallenge: Baoding, Bimanual, Relocate, Die Reorient, Soccer, OSL run | yes | no | n/a |
| ChaseTag (single agent) | yes | `myoChallengeChaseTagFBP2-v0` only | no |
| Table Tennis P0 / P1 / P2 | yes | yes | no |
| ChaseTag vs. scripted opponent (multi-agent) | no | no | n/a |
| MuscleMimic (fullbody, bimanual, directional) | no (wrapping raises `TypeError`) | no | n/a |

Giving features for an env or backend marked "no" raises (`NotImplementedError` from `make_env(..., backend="mjlab")`
or `"mjx"`, `TypeError` from a wrapper on an env that does not run stages); they are never dropped silently. The
muscle-condition ids (`myoFati*`, `myoSarc*`, `myoReaf*`) exist as mjlab twins only for the families marked "yes" in the
last column. `make_env` is the authority.

The env's own map stays inside the env: `normalize_act` also sets the initial joint pose, so the
sigmoid is not a wrapper.

## Motor noise (optional, off by default)

`MotorNoiseCfg(signal_dependent_std, constant_std)` (`myosuite.terms.base_action`) adds
human motor noise to muscle excitations. `MotorNoiseCfg.van_beers_2004()` gives the levels
0.103 / 0.185 used by Fischer et al. (2021) and User-in-the-Box.

- **Semantics.** `u' = clip(u + σ_sd·u·n1 + σ_c·n2, 0, 1)` with `n1, n2 ~ N(0, 1)` drawn
  independently per muscle and per control step (and per env on mjlab). The sample is held over
  the `frame_skip` / decimation substeps. Motors are never noised.
- **Order.** After the env's action-to-excitation map and before fatigue (see the stage order
  above). Both backends use the pure term `motor_noise`; the CPU and mjlab outputs match exactly
  for the same normals.
- **RNG.** CPU: the env's `np_random` (seeded by `reset(seed=...)`), drawn only when the noise is
  on, so default rollouts and the RNG stream are unchanged. With noise on, the draws advance
  `np_random`, so later per-episode task samples (targets) differ from the noiseless env unless
  every episode is reseeded. mjlab: `torch.randn` on the sim device from the global torch RNG
  that mjlab seeds (`seed_rng`). The two backends agree in distribution, not sample by sample.
- **Configuration.** `MotorNoiseWrapper(env, MotorNoiseCfg.van_beers_2004())` on CPU (a cfg or a
  dict of its fields; `env.set_motor_noise(...)` changes the levels). To configure both backends,
  register an env id with a `MotorNoiseWrapper` spec: the twin reads it through
  `cpu_reference.action_cfg`. For one mjlab config, set `env_cfg.actions["muscles"].motor_noise`.
- **Clipping.** Near the bounds the clip rectifies the noise: at `u = 0.076` (policy output 0
  with the sigmoid) the van Beers levels clip 34 % of the samples to 0 and raise the mean
  excitation to 0.118 (see `docs/source/quickstart_neuroscience.rst`).

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
| TableTennis | ✓ 417D | ~ | ✗ | Custom obs term (no TypeScript) |
| ChaseTag FBP2 | ✓ 537D (`test_chasetag_fbp2_parity.py`) | ✓ 354D direct | ✗ | No TypeScript: `chasetag_obs` blocks, scripted opponent; ctrl_dt 0.01 s |

All passing parity tests live in `myosuite/tests/test_mjlab_task_builder.py`.

**To unblock mjswan export:** either contribute TypeScript obs term implementations to the [mjswan repo](https://github.com/ttktjmt/mjswan), or rewrite mjlab obs functions to use only mjswan built-ins (`joint_pos_rel`, `joint_vel_rel`, etc.) where semantically equivalent.
