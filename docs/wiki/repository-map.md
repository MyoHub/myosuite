# Repository Map

## Top-Level

```
myosuite/          # core package
docs/wiki/         # this wiki
myosuite/tests/    # test suites
scripts/           # developer maintenance scripts
tutorials/         # usage examples and pre-trained baselines
```

## `myosuite/` Structure

| Directory | Contents |
|---|---|
| `core/` | Registry, config, model builder, I/O adapters |
| `terms/` | Backend-agnostic pure term functions (obs, reward, reset, termination) |
| `physics/` | Biomechanics math — quaternions, fatigue, IK, min-jerk. No backend imports. |
| `envs/myo/tasks/` | Task definitions organized by collection |
| `envs/myo/backends/` | mjlab and MJX execution backends |
| `utils/` | Generic helpers — dict, xml, path, tensor. No domain math. |
| `viz/` | Rendering and visualization |
| `integrations/` | Third-party integrations (musclemimic) |

## Task Layout

```
tasks/
├── basic/challenge/mimic/      # collection type
│   └── <effector or topic>/    # e.g. arm/, hand/, saber/
│       └── task_spec.py        # TaskSpec definition only — no class, no math
```

## The Core Invariant

> `tasks/` — what effector, what task
> `terms/` — how it is computed
> `TaskSpec` joins them
> Backend is declared in `EnvSpec` and is orthogonal to both

## Navigation

| Goal | Go to |
|---|---|
| Add/change env registration | `myosuite/core/registry.py` |
| Add/change model composition | `myosuite/core/model_builder.py`, `model_recipes.py` |
| Add/change task math | `myosuite/terms/<domain>.py` |
| Add/change biomechanics math | `myosuite/physics/` |
| Add a new task | `tasks/basic/<effector>/` or `tasks/challenge/` or `tasks/mimic/` |

## Naming Conventions

- No `_v0`/`_v1` suffixes — versions live in git.
- No `myo_` prefix inside `myosuite/` — redundant within the package.
- Effector directories: anatomical nouns (`arm/`, `hand/`, `leg/`, `torso/`, `full_body/`).
