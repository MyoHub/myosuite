# Engineering Standards

## Design Principles

- Simple, explicit, modular. Flat over nested. Early returns over deep nesting.
- Reuse before adding. Separate config from runtime logic. No hidden side effects.
- Meaningful error messages. No silent `except: pass`.

## Search-Before-Write (mandatory)

Before any new class, function, or wrapper:
1. `grep -r "keyword" myosuite/` — confirm it does not already exist.
2. Check `library-usage.md` — confirm no library already provides it.
3. If ≥ 80% similar to something existing, extend or import it.

| Kind of code | Canonical location |
|---|---|
| Obs / reward / reset / termination term functions | `myosuite/terms/` |
| Physics math (quat, fatigue, min-jerk) | `myosuite/physics/` |
| Generic utilities | `myosuite/utils/` |
| MuscleMimic helpers | `myosuite/integrations/musclemimic/` |
| mjlab action/obs/event wiring | `myosuite/envs/myo/backends/mjlab/` |

## Python Standards

- Python 3.10+. PEP 8. `pathlib.Path` over `os.path`.
- Type hints on all signatures. Google-style docstrings on all public APIs.
- `@dataclass` for all structured config — never raw dicts or `ConfigDict`.
- No commented-out code, unused imports, or mutable default arguments.

## Architecture Rules

### Golden Rule
> **One `TaskSpec` + term functions + `ModelBuilder` recipe. No task-specific env subclass. No new registration file. No backend-specific code.**

### Target Structure

```
TaskSpec
  ├── recipe: str              → ModelBuilder recipe name
  ├── obs_terms: list[fn]      → dicts merged → obs vector
  ├── reward_terms: list[fn]   → "dense" summed
  ├── reset_terms: list[fn]
  ├── termination_terms: list[fn]
  ├── config: dataclass
  └── variants: list[VariantSpec]   → sarcopenia, fatigue, etc.

register(EnvSpec(task_spec, backends={"cpu","mjx","mjlab"}))
```

### Layer Rules

**Terms** — pure functions, no side effects. Use `accessor.array_module()`. Return: obs terms → `dict[str, Any]`; reward terms → `{"dense": float, "solved": bool, "done": bool, ...}`; termination terms → `{"done": bool}`.

**Config** — typed `@dataclass` only. `ConfigDict` only inside `to_config_dict()` adapters.

**Registration** — `register(EnvSpec(...))` only. Variants as `VariantSpec` entries. No per-backend registration.

**Models** — `ModelBuilder` + named recipes. Direct `mujoco.MjSpec.from_file()` only for challenge tasks requiring unsupported MJCF features (add a comment explaining why).

### Base Classes

- CPU: `ComposableEnv`. MJX: `MjxComposableEnv`. No other subclassing for task logic.
- `BaseV0`, `MujocoEnv`, `env_base.MujocoEnv` are deleted — do not reference them.

### Terms Organization

Grouped by motor control domain: `locomotion.py`, `posture.py`, `manipulation.py`, `reach.py`, `multiplayer/`. No `myo_` prefix.

## Adding a New Environment (checklist)

1. Write term functions in `myosuite/terms/<domain>.py`.
2. Add/reuse a `ModelBuilder` recipe in `myosuite/core/model_recipes.py`.
3. Define a `@dataclass` config.
4. Define a `TaskSpec`.
5. Call `register(EnvSpec(...))` once, in the suite's `__init__.py`.
6. Run: `pytest myosuite/tests/test_parity.py -v` — CPU `atol ≤ 1e-7`.

## Parity Policy

`pytest myosuite/tests/test_parity.py -v` after every env change. Regressions block PRs. To regenerate a baseline after an intentional change:
```bash
python scripts/generate_parity_baselines.py --env-id <env-id>
```

## Quality Gates

- Add/update tests for non-trivial behavior changes.
- `pre-commit run --all-files` must pass.
- Self-review: smallest change that solves the problem? Searched first? No duplication?
