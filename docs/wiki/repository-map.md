# Repository Map

> **Source of truth priority:** code and tests → `CLAUDE.md` → this wiki.

## Top-Level

```
myosuite/          # core package (see below)
docs/wiki/         # this wiki
scripts/           # developer maintenance scripts (parity baselines, training, export)
tutorials/         # runnable usage examples and pre-trained baselines
benchmarks/        # performance benchmarks
tasks/             # working notes: todo.md, lessons.md, reports (not shipped)
```

## `myosuite/` Structure

| Directory | Contents |
|---|---|
| `core/` | Registry, config dataclasses, model builder/recipes, muscle conditions, citations |
| `terms/` | Backend-agnostic pure term functions (obs, reward, action, event, termination) |
| `physics/` | Biomechanics math — quaternions, fatigue, IK, min-jerk. No backend imports. |
| `envs/` | `gymnasium_env.py` (CPU base), `modular_env.py` (data-driven env), wrappers |
| `envs/myo/tasks/` | Task definitions organized by collection (`basic/`, `challenge/`, `mimic/`) |
| `envs/myo/backends/` | `mjx/` (JAX) and `mjlab/` (Warp) execution backends |
| `utils/` | Generic helpers — dict, xml, path, tensor, onnx export, plotting. No domain math. |
| `viz/` | Rendering and visualization |
| `integrations/` | Third-party integrations (musclemimic) |
| `logger/` | Rollout logging / grouped datasets |
| `scenes/` | Scene assembly helpers |
| `tests/` | Test suites (tiers 1–3, see pyproject markers) |

## Task Layout (as it exists)

```
envs/myo/tasks/          # CPU implementations (MyoGymnasiumEnv)
├── basic/
│   ├── __init__.py         # registers basic CPU envs via registry.register_env
│   ├── arm/ hand/ leg/ torso/   # MyoGymnasiumEnv subclasses: pose.py, reach.py, ...
│   └── specs/              # TaskConfig data-driven reference (elbow_pose_spec.py → cpu + mjx)
├── challenge/              # chasetag, relocate, reorient, soccer, tabletennis, ...
└── mimic/                  # MuscleMimic CPU tasks

envs/myo/backends/       # GPU implementations (same env_id as the CPU task)
├── mjlab/                # register_mjlab_*.py — MuJoCo-Warp ManagerBasedRlEnvCfg (supported)
└── mjx/                  # JAX/playground env classes (experimental; not guaranteed long-term)
```

A task's **CPU** env (`tasks/`) and its supported **GPU** implementation
(`backends/mjlab/`) share one `env_id` and the cross-backend contract — see
`engineering-standards.md`. Tasks hand-write the CPU `MyoGymnasiumEnv` and the
mjlab config separately. (An experimental `TaskConfig` route can instead generate
a data-driven CPU env plus an MJX backend from one dataclass — elbow reference
only; MJX is not guaranteed long-term.)

## Navigation

| Goal | Go to |
|---|---|
| Add/change CPU env registration | `myosuite/core/registry.py` + the suite `__init__.py` |
| Add/change model composition | `myosuite/core/model_builder.py`, `model_recipes.py` |
| Add/change a term function | `myosuite/terms/base_*.py` |
| Add/change biomechanics math | `myosuite/physics/` |
| Add a CPU env (MyoGymnasiumEnv) | `envs/myo/tasks/basic/<effector>/` or `challenge/` |
| Add the matched mjlab GPU task | `envs/myo/backends/mjlab/register_mjlab_*.py` |
| Data-driven `TaskConfig` route (experimental) | `envs/myo/tasks/basic/specs/` + `adding-a-new-task.md` |
| Change the CPU step/reset loop | `myosuite/envs/gymnasium_env.py` |

## Naming Conventions

- No `myo_` prefix inside `myosuite/` — redundant within the package.
- Effector directories: anatomical nouns (`arm/`, `hand/`, `leg/`, `torso/`).
- Env **IDs** keep their public `-v0`/`-v1` suffix (Gymnasium convention, part of
  the public API). Some existing env **classes** also carry a `V0` suffix
  (`ReachEnvV0`) for historical reasons; new CPU env classes need not.
