# Repository Map

This page describes how the repository is organized so agents can quickly find the right place to edit.

## Top-Level Layout

- `myosuite/` - Core Python package with environments, shared abstractions, term logic, and utilities.
- `docs/` - Documentation source, wiki, and media assets (`docs/media/` for video/images).
- `myosuite/tests/` - Test suites for architecture, parity, and regressions.
- `scripts/` - Developer scripts (for maintenance, generation, and utility workflows).
- `benchmarks/`, `tutorials/` - Benchmark materials and usage examples. Pre-trained baseline policies live under `tutorials/mc26/baselines/`.

## `myosuite/` Structure

- `myosuite/core/` - Foundational architecture and contracts (registry, config, model builder, I/O adapters).
- `myosuite/terms/` - Backend-agnostic pure term functions for observations, rewards, reset, and termination.
- `myosuite/physics/` - Biomechanics and motor control math (quaternions, fatigue, IK, min-jerk). No backend imports.
- `myosuite/envs/` - Environment implementations and task-specific definitions.
- `myosuite/utils/` - Generic helpers (dict, xml, path, tensor, import, spec, curriculum). No domain math.
- `myosuite/viz/` - Rendering and visualization utilities.
- `myosuite/logging/` - Data logging, reference motion, and dataset utilities.

## Task Taxonomy

All tasks live under `myosuite/envs/myo/tasks/` organized by **collection type**:

```
tasks/
├── basic/      # fundamental motor control tasks, organized by effector
│   ├── arm/    # reach, pose, key_turn
│   ├── hand/   # pen, obj_hold
│   ├── leg/    # walk, reach
│   ├── torso/  # pose
│   └── full_body/  # walk (whole-body locomotion)
├── mimic/      # trajectory mimicry tasks
└── challenge/  # MyoChallenge competition tasks (boxing_mannequin, boxing_vs, saber_vs)
```

`basic/`, `mimic/`, and `challenge/` are all task collections — differentiated by purpose,
not by organizational level.

Within `basic/`, the two-level structure (`effector/task.py`) resolves naming ambiguity:
`leg/walk.py` vs `full_body/walk.py` are unambiguous because the directory names the effector.

## Backends

Backend-specific environments live under `myosuite/envs/myo/backends/`:

```
backends/
├── mjx/    # JAX/MJX execution
└── mjlab/  # MuJoCo Lab integration
```

Backends are alternate execution paths for the same tasks — not a separate task collection.
A task gains backend support by declaring it in `EnvSpec(backends={...})`.

## Terms Structure

`myosuite/terms/` is organized by **motor control domain**, not by task name.
This is the shared computation layer that multiple tasks reuse.

```
terms/
├── locomotion.py   # walking, stability, ground contact
├── posture.py      # joint pose error, limits, velocity
├── manipulation.py # grasp, contact force, object pose
├── reach.py        # endpoint error, path efficiency
└── multiplayer/
    ├── common.py
    ├── boxing.py
    └── saber.py
```

The `myo_` prefix is dropped inside `myosuite/terms/` — it is redundant.

## The Core Invariant

> `tasks/` answers: *what effector, what task?*
> `terms/` answers: *how is it computed?*
> `TaskSpec` joins them.
> Backend is declared in `EnvSpec` and is orthogonal to both.

## Architectural Boundaries

- Put framework-level abstractions in `myosuite/core/`.
- Put biomechanics / motor control math in `myosuite/physics/`. Never duplicate across backends.
- Put reusable generic helpers in `myosuite/utils/`. No domain math here.
- Put task/domain behavior in `myosuite/terms/` and orchestrate from `TaskSpec`.
- Put rendering and visualization in `myosuite/viz/`.
- Avoid introducing new ad hoc layers when a shared module already exists.

## Naming Conventions

- No `_v0` / `_v1` suffixes on files — versions are tracked in git.
- No `gymnasium` in task filenames — the framework is implicit from the env base class.
- No `myo_` prefix inside `myosuite/` — redundant within the package.
- Effector directories use anatomical nouns: `arm/`, `hand/`, `leg/`, `torso/`, `full_body/`.

## Navigation Heuristics for Agents

- If adding/changing env registration → `myosuite/core/registry.py`.
- If adding/changing model composition → `myosuite/core/model_builder.py` and `model_recipes.py`.
- If adding/changing task logic math → `myosuite/terms/<domain>.py`.
- If adding/changing biomechanics math (quaternions, fatigue, IK) → `myosuite/physics/`.
- If adding a new task → `tasks/basic/<effector>/` or `tasks/challenge/` or `tasks/mimic/`.
- If adding/changing rendering → `myosuite/viz/`.
- If adding/changing public behavior → inspect relevant tests and update/add tests.
