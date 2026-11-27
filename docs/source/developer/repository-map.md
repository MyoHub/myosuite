# Repository Map

> **Source of truth priority:** code and tests → [AGENTS.md](https://github.com/MyoHub/myosuite/blob/dev/AGENTS.md) → this guide.
> **Package boundaries:** enforced in CI by `myosuite/tests/test_package_boundaries.py` and `myosuite/tests/test_package_discovery.py`.

## Subpackage Ownership & Allowed-Import Matrix

`myo_sim` owns reusable anatomical MJCF/meshes, muscle strength/sarcopenia, muscle fatigue, joint couplings, and anthropometric scaling. `myosuite` consumes public `myo_sim` APIs and never duplicates library-side model mechanics.

| Layer | Subpackage / Directory | Status | Responsibility | Allowed `myosuite.*` Runtime Imports |
| :--- | :--- | :--- | :--- | :--- |
| **Foundation** | `myosuite/core/` | Active | `EnvConfig`, `registry`, `make_env`, `ModelBuilder`, recipes, scene/spec `builder/`, pure MDP `terms/`, muscle conditions/scaling facades, trajectory I/O | `core`, `physics` |
| **Numerical Math** | `myosuite/physics/` | Active | Shared numerical helpers and compatibility imports (`quat*`, `running_stats`, `colored_noise`, `joint_coupling`, `fatigue_jax`); canonical model mechanics live in `myo_sim` | `physics` |
| **MDP Terms (Shim + Catalog)** | `myosuite/terms/` | Facade *(+ catalog & ChaseTag terms)* | Backward-compatible aliases to `core/terms/`, `myosuite-terms` CLI (`catalog.py`), and ChaseTag task terms (`multiplayer/chase_tag_vs_*`) | `core`, `physics`, `terms` |
| **Scene Assembly (Shim)** | `myosuite/scenes/` | Closed facade | Backward-compatible aliases to `core/builder/{scene_library,scene_builder}.py` | `core`, `scenes` |
| **Motion & Trials** | `myosuite/motion/` | *Planned target* | Flat motion contract subpackage (`MotionClip`, trajectory `io.py`, `transforms.py`, playback/reference contracts, `hf_cache.py`, composable `KinematicTrial` / `EMGTrial`); today use `core/trajectory_io.py` and `core/trajectory_yaw.py` | `core`, `physics`, `motion` |
| **Scientific Analysis** | `myosuite/biomechanics/` | Active | Site IK (`mink`), coupling-aware landmark IK, inverse dynamics, static optimization, CEINMS, CMC, EMG processing, external-load wrench mapping, effort, observability/uncertainty | `core`, `physics`, `biomechanics` |
| **Ecosystem Bridges** | `myosuite/integrations/` | Active | Lightweight source-specific readers/adapters (markers, InstantHMR, GVHMR, SMPL+H, MANO, OpenSim `.mot`/`.sto`) and `musclemimic` runtime bridges | `core`, `terms`, `physics`, `biomechanics`, `utils`, `integrations` *(+ `motion`, `policies` once created)* |
| **Runtime Inference** | `myosuite/policies/` | *Planned target* | Reusable checkpoint/ONNX loading, `export_onnx`, checkpoint manifests and `action_map` runtime wrappers (today in `myosuite/utils/`) | `core`, `terms`, `physics`, `biomechanics`, `policies` *(+ `motion` once created)* |
| **Environments** | `myosuite/envs/` | Active | `gymnasium_env.py`, `modular_env.py`, `wrappers.py`, `muscle_stages.py`, and `envs/myo/{tasks,backends,myoedits,assets}/` | `core`, `scenes`, `terms`, `physics`, `biomechanics`, `integrations`, `utils`, `envs` *(+ `motion`, `policies` once created)* |
| **Visualization** | `myosuite/viz/` | Active | Offscreen/interactive viewer, rollout playback/video I/O, volumetric `muscle_meshes`/`muscle_tubes`, `.skn`/`.glb` skinning, USD/Blender export | `core`, `scenes`, `physics`, `biomechanics`, `envs`, `viz` *(+ `motion`, `policies` once created)* |
| **Evaluation Suites** | `myosuite/benchmarks/` | *Planned target* | Standardized evaluation suites (`paired_replay`, `push_*`, `performance_card`, `examine_env`, `env_check`, rollout logging — today in `utils/` and `logger/`) | `core`, `terms`, `physics`, `biomechanics`, `integrations`, `envs`, `viz`, `benchmarks` *(+ `motion`, `policies` once created)* |
| **Rollout Logging** | `myosuite/logger/` | Active | Rollout trace/dataset logging and reference-motion helpers (scheduled for split into `motion/` and `viz/`) | `core`, `physics`, `logger` |
| **Shared Utilities** | `myosuite/utils/` | Active | Shared runtime/CLI utilities (scheduled for split into `core/`, `policies/`, `envs/`, `viz/`, and `benchmarks/`) | `core`, `terms`, `physics`, `biomechanics`, `envs`, `viz`, `utils` |
| **Out-of-Wheel RL** | `training/<method>/` | Active | Self-contained RL/control projects with independent `pyproject.toml` (`chasetag_teacher`, `dynsyn`, `supermimic`, ...); see [training-guidelines.md](training-guidelines.md) | Consumes `myosuite`; **never** imported by `myosuite/**` |
| **Out-of-Wheel Tools** | `tools/<tool>/` | Active | Heavy CV/perception extractors (`instanthmr`) with independent `pyproject.toml` | Consumes `myosuite`; **never** imported by `myosuite/**` |

## Top-Level

```
myosuite/          # core package (see below; tests/ excluded from built wheels)
docs/source/       # Sphinx documentation (quickstarts, environment reference, API)
docs/source/developer/ # this developer guide
scripts/           # training, evaluation and rendering CLIs (train_mjlab, train_sb3, eval_mjlab_policy, render_blender), parity baselines
training/          # self-contained training projects outside the package, one per method (chasetag_teacher/, dynsyn/, supermimic/); see training-guidelines.md
tutorials/         # notebooks (1.x-5.x) and their companion files in files/X.Y/
tools/             # standalone data/perception CLIs with isolated dependencies (instanthmr/)
.github/           # CI and release workflows
```

## `myosuite/` Structure

| Directory | Contents |
|---|---|
| `core/` | Registry and `make_env`, config dataclasses (`EnvConfig`, `TaskConfig`), `ModelBuilder`/recipes, `core/builder/` (scenes, MJCF spec/geom/joint/asset helpers), `core/terms/` (backend-agnostic MDP terms), muscle conditions/scaling facades |
| `terms/` | Backward-compatible aliases to `core/terms/` plus `catalog.py` (`myosuite-terms`) and ChaseTag multiplayer terms |
| `physics/` | Model-level runtime math — quaternions, running stats, colored noise, fatigue, joint couplings. No backend imports. |
| `biomechanics/` | Musculoskeletal analysis of motions (inverse kinematics and dynamics, static optimization, CMC, OpenSim motions). NumPy / SciPy / MuJoCo only. |
| `envs/` | `gymnasium_env.py` (CPU base), `modular_env.py` and `multi_agent_modular_env.py` (data-driven envs), `wrappers.py` and `muscle_stages.py` (muscle noise / fatigue / reafferentation / sarcopenia as wrappers) |
| `envs/myo/tasks/` | Task definitions organized by collection (`basic/`, `challenge/`, `mimic/`) |
| `envs/myo/backends/` | `mjx/` (JAX) and `mjlab/` (Warp) execution backends |
| `envs/myo/myoedits/` | Model-editing helpers and the registered edit variants |
| `utils/` | Generic helpers — dict, xml, path, tensor, onnx export, plotting, `examine_env.py`, `env_check.py`. No domain math. Checkpoint playback: `rslrl_policy.py`, `checkpoint_manifest.py`, `action_map.py` (action maps such as DynSyn: policy actions -> per-muscle actions, CPU wrapper, checkpoint reading) |
| `viz/` | Rendering and visualization; `blender_render.py` (USD export and Blender studio scene), `muscle_tubes.py` (volumetric muscle geometry) and `skin.py` (`.skn` body skin; bundled skin in `assets/`) |
| `integrations/` | Third-party integrations (musclemimic) |
| `logger/` | Rollout logging / grouped datasets |
| `scenes/` | Backward-compatible aliases to `core/builder/` |
| `tests/` | Test suites (tiers 1–3, see pyproject markers); `tests/benchmarks/` holds the throughput benchmarks |

## Task Layout (as it exists)

```
envs/myo/tasks/          # CPU implementations (MyoGymnasiumEnv)
├── basic/
│   ├── __init__.py         # registers basic CPU envs via registry.register_env
│   ├── muscle_mixin.py     # runs the muscle-command stages (noise, fatigue, ...) for the basic envs
│   ├── arm/ hand/ leg/ torso/ full_body/   # MyoGymnasiumEnv subclasses: pose.py, reach.py, ...
│   └── specs/, leg/specs/  # TaskConfig data-driven references (elbow pose, leg directional → cpu + mjx)
├── challenge/              # chasetag, relocate, reorient, soccer, tabletennis, ...
└── mimic/                  # MuscleMimic CPU tasks; tracking/ = myoMimic-v0 (task cfg, models, motion lib, metrics)

envs/myo/backends/       # GPU implementations (same env_id as the CPU task)
├── mjlab/                # MuJoCo-Warp ManagerBasedRlEnvCfg (supported)
│   ├── tasks/            # twins of the CPU envs, built from the CPU registration:
│   │                     #   cpu_reference.py, registration.py, pose/ reach/ leg/ waypoint/
│   │                     #   mimic_tracking/ (myoMimic-v0), mdp/ (terms, MyoAction)
│   ├── register_mjlab_*.py   # special-purpose tasks (Table Tennis, ChaseTag, Mimic)
│   └── configs/, mjlab_task_builder.py   # shared config and TaskConfig builders
└── mjx/                  # JAX/brax env classes (experimental; not guaranteed long-term)
```

A task's **CPU** env (`tasks/`) and its supported **GPU** implementation
(`backends/mjlab/`) share one `env_id` and the cross-backend contract — see
[engineering-standards.md](engineering-standards.md). Tasks hand-write the CPU `MyoGymnasiumEnv`; the mjlab
twin of the basic envs is built from the CPU registration (`backends/mjlab/tasks/cpu_reference.py`), so
parameters live in one place. (An experimental `TaskConfig` route can instead generate a data-driven CPU env
plus an MJX backend from one dataclass — the elbow pose and leg directional references; MJX is not
guaranteed long-term.)

## Navigation

| Goal | Go to |
|---|---|
| Add/change CPU env registration | `myosuite/core/registry.py` + the suite `__init__.py` |
| Add/change model or scene composition | `myosuite/core/model_builder.py`, `model_recipes.py`, `core/builder/` |
| Add/change a term function | `myosuite/core/terms/base_*.py` |
| Add/change biomechanics math | `myosuite/physics/` |
| Analyse a motion (IK, ID, static optimization, CMC), retarget motion data (SMPL+H, InstantHMR), muscle effort of rollouts | `myosuite/biomechanics/`, `tools/instanthmr/` |
| Add a CPU env (MyoGymnasiumEnv) | `envs/myo/tasks/basic/<effector>/` or `challenge/` |
| Add the matched mjlab GPU task | `envs/myo/backends/mjlab/tasks/<family>/` (twins of CPU envs), `register_mjlab_*.py` (special tasks) |
| Muscle noise, fatigue, reafferentation, sarcopenia | `envs/wrappers.py`, `envs/muscle_stages.py`; on mjlab `backends/mjlab/tasks/mdp/actions.py` |
| Play a checkpoint trained through an action map (DynSyn) | `utils/action_map.py`; on mjlab `backends/mjlab/tasks/mdp/action_map.py`; training in `training/dynsyn/` |
| `make_env` / `EnvConfig` (one entry point for both backends) | `core/registry.py`, `core/config.py`; `utils/feature_cli.py` for the `--feature` flags |
| Data-driven `TaskConfig` route (experimental) | `envs/myo/tasks/basic/specs/` + [adding-a-new-task.md](adding-a-new-task.md) |
| Change the CPU step/reset loop | `myosuite/envs/gymnasium_env.py` |

## Naming Conventions

- No `myo_` prefix inside `myosuite/` — redundant within the package.
- Effector directories: anatomical nouns (`arm/`, `hand/`, `leg/`, `torso/`).
- Env **IDs** keep their public `-v0`/`-v1` suffix (Gymnasium convention, part of
  the public API). Some existing env **classes** also carry a `V0` suffix
  (`ReachEnvV0`) for historical reasons; new CPU env classes need not.
