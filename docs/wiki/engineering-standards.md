# Engineering Standards

This page summarizes the development standards agents must apply when making code changes.
It is the authoritative detail layer; `CLAUDE.md` contains the non-negotiable summary.

## Design Principles

- Prefer simple, explicit, modular changes over clever abstractions.
- Reuse existing shared modules before adding new files or parallel logic.
- Keep public interfaces stable unless a breaking change is explicitly requested.
- Separate configuration from runtime logic, and avoid hidden side effects.
- Avoid deeply nested logic — flatten with early returns or helper functions.
- Handle errors with meaningful messages; no silent `except: pass`.

## Search-Before-Write Rule (mandatory)

Before writing any new class, function, helper, or wrapper, you **must**:

1. `grep -r "relevant_keyword" myosuite/` — confirm it does not already exist in the repo.
2. Check `docs/wiki/library-usage.md` — confirm no library already provides it.
3. If a similar implementation exists (≥ 80% overlap), **extend or import it**. Do not write a second copy.

**Why this matters:** Past agent sessions have produced 3 independent copies of the same running mean/std logic, 3 near-identical muscle action term classes, and multiple re-implementations of `pathlib`, sigmoid, and observation normalization — all because each session was scoped to one file and did not search sibling files first. Every duplication increases maintenance cost and drift risk.

The canonical locations for shared logic:

| Kind of code | Where it lives |
|---|---|
| Observation / reward / reset term functions | `myosuite/terms/` |
| Physics math (quat, fatigue, min-jerk) | `myosuite/physics/` |
| Generic utilities (paths, dicts, tensors) | `myosuite/utils/` |
| MuscleMimic inference helpers | `myosuite/integrations/musclemimic/` |
| mjlab action/obs/event wiring | `myosuite/envs/myo/backends/mjlab/` — **one class per abstraction** |

## Python Standards

- Target **Python 3.10+**: `match/case`, `X | Y` unions, walrus operator where clear.
- Type hints on all function signatures and class attributes.
- Follow PEP 8. Use `pathlib.Path` instead of `os.path`.
- Prefer `dataclasses` or `pydantic` models over raw dicts for structured data.
- Use `logging` instead of `print` for non-trivial output.
- No commented-out dead code, unused imports, or mutable default arguments.

## Documentation

- Docstrings on all public functions/classes/modules using **Google style**:

```python
def compute_something(x: float, y: float) -> float:
    """Compute the result of a specific operation.

    Args:
        x: The first input value.
        y: The second input value.

    Returns:
        The computed result.

    Raises:
        ValueError: If inputs are out of valid range.
    """
```

- Inline comments only when the logic is non-obvious.

## Copyright Header (all new files)

```python
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
```

## MyoSuite Architecture Rules

### The Golden Rule

> **A new musculoskeletal task requires: one `TaskSpec`, term functions, and a `ModelBuilder` recipe. No new class. No new registration file. No backend-specific code.**

### Target Architecture

```
TaskSpec (dataclass)
  ├── recipe: str              → ModelBuilder recipe name
  ├── obs_terms: list[fn]      → called in order, dicts merged → obs vector
  ├── reward_terms: list[fn]   → called in order, dicts merged, "dense" summed
  ├── reset_terms: list[fn]    → called on episode reset
  ├── termination_terms: list[fn] → "done" OR'd each step
  ├── config: dataclass        → typed, backend-agnostic
  └── variants: list[VariantSpec]  → config deltas (sarcopenia, fatigue, etc.)

EnvSpec (dataclass)
  ├── task_spec: TaskSpec
  └── backends: set[str]       → {"cpu", "mjx", "mjlab"}

register(EnvSpec)              → registers on all declared backends automatically
```

`ComposableEnv` (CPU) and `MjxComposableEnv` (MJX) are the **only** concrete env classes.

### Layer Rules

**Terms (`myosuite/terms/`)**
- Pure functions — no side effects, no mutable state.
- Backend-agnostic: use `accessor.array_module()`. Never import `numpy`, `jnp`, or `torch` directly.
- `EnvAccessor` is the only interface to physics state. Never access `mujoco.MjData` directly.
- Return contracts: obs terms → `dict[str, Any]`; reward terms → see **Reward Dict Contract** below; termination terms → `{"done": bool}`.

**Configuration**
- All configs are typed `@dataclass` objects — never raw dicts, never `ml_collections.ConfigDict`.
- `ConfigDict` allowed **only** inside `to_config_dict()` adapter, never in task/env code.
- MJX-specific keys (`num_envs`, `mjx_impl`) live in a separate `MjxConfig` dataclass.

**Registration**
- All environments registered via `register(EnvSpec(...))`.
- Variants declared as `VariantSpec` entries on `TaskSpec.variants` — never via string-manipulation helpers.
- `__init__.py` files expose public APIs only; no registration logic.

**Model Building**
- **Prefer `ModelBuilder`** for all new tasks. For challenge tasks requiring unsupported MJCF
  features (sensors, heightfields, mocap bodies), direct `mujoco.MjSpec` loading is permitted —
  add a comment explaining why `ModelBuilder` is insufficient.
- Every model variant is a named recipe registered with `@model_recipe(name)`.
- Recipe names validated at import time.

**Reward Dict Contract**

Every `get_reward_dict()` must return a `dict` with these mandatory keys:

| Key | Type | Meaning |
|---|---|---|
| `dense` | `float` | Scalar reward signal passed to the RL agent |
| `solved` | `bool` | Goal achieved this step (logging / curriculum) |
| `done` | `bool` | Episode should terminate for task-internal reasons |

Additional task-specific keys (`"vel_reward"`, `"total_hits"`, …) are encouraged
for logging and analysis. The mandatory subset is enforced at runtime by
`_validate_reward_dict()` in `gymnasium_env.py`.

```python
# Minimal example
def get_reward_dict(self, obs_dict):
    dist = float(obs_dict["pose_err"].mean())
    return {"dense": -dist, "solved": dist < 0.02, "done": False}
```

**Base Classes**
- CPU envs: `ComposableEnv`. MJX envs: `MjxComposableEnv`. No other subclassing for task logic.
  New infrastructure abstractions (accessors, wrappers, utilities) may be classes.
- `BaseV0`, `MujocoEnv`, `env_base.MujocoEnv` are deleted — do not reference them.
- Always return 5-tuple from `step()`: `(obs, rwd, terminated, truncated, info)`.
  Exception: `ModularMultiAgentTaskEnv` returns 5-tuples of per-agent dicts — intentional.

### Task Organization Rules

- New tasks go in `myosuite/envs/myo/tasks/<collection>/<effector>/` for basic tasks,
  or `tasks/challenge/` or `tasks/mimic/` for their respective collections.
- A task file (`walk.py`) contains only a `TaskSpec` definition — no class, no math.
- Shared computation between similar tasks lives in `myosuite/terms/<domain>.py`.
- Backend support is declared once in `EnvSpec(backends={...})`.

### Terms Organization Rules

- Terms are grouped by **motor control domain**: `locomotion`, `posture`, `manipulation`, `reach`, `multiplayer/`.
- No `myo_` prefix on term files — redundant inside `myosuite/terms/`.

### Physics / Biomechanics Math Rules

- All quaternion math → `myosuite/physics/quat_math.py` (numpy) or `quat_math_jax.py` (JAX).
- All fatigue modeling → `myosuite/physics/fatigue.py` / `fatigue_jax.py`.
- All minimum-jerk trajectories → `myosuite/physics/min_jerk.py`.
- `myosuite/utils/` is for generic non-domain helpers only (dict, xml, path, tensor).

### Dos and Don'ts

| Do | Don't |
|---|---|
| Define new tasks as `TaskSpec` + term functions | Subclass `ComposableEnv` or `MjxComposableEnv` |
| Use `accessor.array_module()` in terms | Import numpy/jax/torch directly in terms |
| Use `register(EnvSpec(...))` for all backends | Edit `mjx/__init__.py`, `mjlab/__init__.py` directly |
| Use `VariantSpec` for sarcopenia/fatigue/etc. | Use `register_env_with_variants()` string hacks |
| Use typed `@dataclass` for all configs | Use `config_dict.create(...)` or raw dicts |
| Use `ModelBuilder` + named recipes | Call `MjSpec.from_file()` in env code |
| Use `myo_sim.get_path(...)` for assets | Use hardcoded `curr_dir/../../simhive/...` paths |
| Return 5-tuple from `step()` | Return 4-tuple (old gym style) |

### Adding a New Environment (checklist)

1. Write term functions in `myosuite/terms/` (obs, reward, reset, termination).
2. Add or reuse a `ModelBuilder` recipe in `myosuite/core/model_recipes.py`.
3. Define a `@dataclass` config in `myosuite/envs/myo/configs/<task>_config.py`.
4. Define a `TaskSpec` in `myosuite/envs/myo/<suite>/<task>_spec.py`.
5. Call `register(EnvSpec(task_spec=..., backends={...}))` — once, in the suite's `__init__.py`.
6. Run parity tests: `pytest myosuite/tests/test_parity.py -v` — CPU `atol ≤ 1e-7`.

### Parity Policy

- Run `pytest myosuite/tests/test_parity.py -v` after every env change.
- CPU `atol ≤ 1e-7`. Any regression blocks the PR.
- Do not delete old env code until parity tests pass.
- If a migrated env changes behavior intentionally, regenerate its baseline:
  ```bash
  python scripts/generate_parity_baselines.py --env-id <env-id>
  ```

## Quality Gates

- Add or update tests for non-trivial behavior changes.
- Run relevant test suites before concluding work (`pytest myosuite/tests/ -v`).
- Ensure lints/formatting remain clean (`pre-commit run --all-files`).

## Code Review Checklist (Agent Self-Review)

- Is this the smallest change that solves the problem?
- **Did I search the repo before writing this?** (`grep` evidence in plan or notes)
- **Does a library already provide this?** (check `docs/wiki/library-usage.md`)
- Is any logic duplicated that should be shared?
- Are error messages actionable and explicit?
- Are tests aligned with behavior changes?
- Is documentation updated where behavior or structure changed?
