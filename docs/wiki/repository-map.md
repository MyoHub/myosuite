# Repository Map

> **Source of truth priority:** code and tests → [CLAUDE.md](../../CLAUDE.md) → this wiki.

## Top-Level

```
myosuite/          # core package (see below)
docs/source/       # Sphinx documentation (quickstarts, environment reference, API)
docs/wiki/         # this developer wiki
scripts/           # training, evaluation and rendering CLIs (train_mjlab, train_sb3, eval_mjlab_policy, render_blender), parity baselines
tutorials/         # notebooks (1.x-5.x) and their companion files in files/X.Y/
.github/           # CI and release workflows
```

## `myosuite/` Structure

| Directory | Contents |
|---|---|
| `core/` | Registry and `make_env`, config dataclasses (`EnvConfig`, `TaskConfig`), model builder/recipes, muscle conditions |
| `terms/` | Backend-agnostic pure term functions (obs, reward, action, event, termination) |
| `physics/` | Biomechanics math — quaternions, fatigue, IK, min-jerk. No backend imports. |
| `envs/` | `gymnasium_env.py` (CPU base), `modular_env.py` and `multi_agent_modular_env.py` (data-driven envs), `wrappers.py` and `muscle_stages.py` (muscle noise / fatigue / reafferentation / sarcopenia as wrappers) |
| `envs/myo/tasks/` | Task definitions organized by collection (`basic/`, `challenge/`, `mimic/`) |
| `envs/myo/backends/` | `mjx/` (JAX) and `mjlab/` (Warp) execution backends |
| `envs/myo/myoedits/` | Model-editing helpers and the registered edit variants |
| `utils/` | Generic helpers — dict, xml, path, tensor, onnx export, plotting. No domain math. |
| `viz/` | Rendering and visualization; `blender_render.py` (USD export and Blender studio scene), `muscle_tubes.py` (volumetric muscle geometry) and `skin.py` (`.skn` body skin; bundled skin in `assets/`) |
| `integrations/` | Third-party integrations (musclemimic) |
| `logger/` | Rollout logging / grouped datasets |
| `scenes/` | Scene assembly helpers |
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
└── mimic/                  # MuscleMimic CPU tasks

envs/myo/backends/       # GPU implementations (same env_id as the CPU task)
├── mjlab/                # MuJoCo-Warp ManagerBasedRlEnvCfg (supported)
│   ├── tasks/            # twins of the CPU envs, built from the CPU registration:
│   │                     #   cpu_reference.py, registration.py, pose/ reach/ leg/ mdp/ (terms, MyoAction)
│   ├── register_mjlab_*.py   # special-purpose tasks (Table Tennis, ChaseTag, Mimic)
│   └── configs/, mjlab_task_builder.py   # shared config and TaskConfig builders
└── mjx/                  # JAX/playground env classes (experimental; not guaranteed long-term)
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
| Add/change model composition | `myosuite/core/model_builder.py`, `model_recipes.py` |
| Add/change a term function | `myosuite/terms/base_*.py` |
| Add/change biomechanics math | `myosuite/physics/` |
| Add a CPU env (MyoGymnasiumEnv) | `envs/myo/tasks/basic/<effector>/` or `challenge/` |
| Add the matched mjlab GPU task | `envs/myo/backends/mjlab/tasks/<family>/` (twins of CPU envs), `register_mjlab_*.py` (special tasks) |
| Muscle noise, fatigue, reafferentation, sarcopenia | `envs/wrappers.py`, `envs/muscle_stages.py`; on mjlab `backends/mjlab/tasks/mdp/actions.py` |
| `make_env` / `EnvConfig` (one entry point for both backends) | `core/registry.py`, `core/config.py`; `utils/feature_cli.py` for the `--feature` flags |
| Data-driven `TaskConfig` route (experimental) | `envs/myo/tasks/basic/specs/` + [adding-a-new-task.md](adding-a-new-task.md) |
| Change the CPU step/reset loop | `myosuite/envs/gymnasium_env.py` |

## Naming Conventions

- No `myo_` prefix inside `myosuite/` — redundant within the package.
- Effector directories: anatomical nouns (`arm/`, `hand/`, `leg/`, `torso/`).
- Env **IDs** keep their public `-v0`/`-v1` suffix (Gymnasium convention, part of
  the public API). Some existing env **classes** also carry a `V0` suffix
  (`ReachEnvV0`) for historical reasons; new CPU env classes need not.
