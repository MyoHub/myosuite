# Claude Code – Project Instructions

Read `docs/wiki/index.md` and the relevant wiki pages before making substantial changes.
The wiki is the detail layer; this file contains only the non-negotiable rules.

---

## Non-Negotiables

### MyoSuite: The Golden Rule

> **A new musculoskeletal task requires: one `TaskSpec`, term functions, and a `ModelBuilder` recipe. No task-specific env subclass. No new registration file. No backend-specific code.**

- Term functions must be pure and backend-agnostic (`accessor.array_module()` only).
- All configs are typed `@dataclass` — never raw dicts or `ConfigDict` in task/env code.
- All envs registered via `register(EnvSpec(...))`. No per-backend registration.
- **"No new class" scope:** Do not subclass `ModularTaskEnv`, `MyoGymnasiumEnv`,
  `MyoMjxEnvBase`, or any backend base class for task logic. New infrastructure
  abstractions (accessors, wrappers, utilities) may be classes.
- **Prefer `ModelBuilder`** for all new tasks. For challenge tasks that require MJCF
  features not yet supported by `ModelBuilder` (sensors, heightfields, mocap bodies),
  direct `mujoco.MjSpec` loading is permitted — add a comment explaining why.
- Always return 5-tuple from `step()`: `(obs, rwd, terminated, truncated, info)`.
  **Exception:** `ModularMultiAgentTaskEnv` returns 5-tuples of per-agent dicts
  `(obs_dict, rwd_dict, term_dict, trunc_dict, info_dict)` — this is intentional.
- Use `myo_sim.get_path(...)` for assets — no hardcoded paths.

### Code Quality

- Simple, readable, modular. DRY, SOLID, separation of concerns.
- **Search before writing.** Before adding any new class or function, `grep` the repo for an existing implementation. If one exists at ≥ 80% similarity, extend or import it — never duplicate it.
- **Use the library.** Before writing any helper, check whether `mjlab`, `gymnasium`, `mujoco`, `numpy`, `torch`, or `scipy` already provides it. Custom re-implementations of library features are a defect, not a style choice. See `docs/wiki/library-usage.md` for the approved mapping.
- Reuse existing helpers before adding new code.
- No commented-out dead code, unused imports, or silent `except: pass`.
- Type hints on all signatures. Google-style docstrings on all public APIs.
- Python 3.10+. PEP 8. `pathlib.Path` over `os.path`.

### Workflow

- Plan before implementing non-trivial tasks (write to `tasks/todo.md`).
- **Pre-task search is mandatory for any new class, function, or config.** Run `grep -r "keyword" myosuite/` first. If a similar abstraction exists, use it.
- After any user correction: update `tasks/lessons.md`.
- Never mark a task done without proving it works.

---

## Verification (run before every commit)

```bash
pre-commit run --all-files
pytest myosuite/tests/test_model_builder.py -v
pytest myosuite/tests/test_terms_cpu.py -v
pytest myosuite/tests/test_fragment_compat.py -v
pytest myosuite/tests/test_parity.py -v      # atol ≤ 1e-7 — regressions block PR
pytest myosuite/tests/test_registry.py -v
```

---

## Wiki

- `docs/wiki/engineering-standards.md` — full architecture rules, Python standards, checklist for new envs, dos/don'ts.
- `docs/wiki/agent-workflow.md` — workflow, task management, subagent strategy.
- `docs/wiki/library-usage.md` — **approved library feature map; read before writing any helper.**
- `docs/wiki/mjlab-design-guide.md` — **canonical mjlab patterns + new-task checklist; read before writing any mjlab backend code.**
- `docs/wiki/cross-backend-contract.md` — **obs/action/timing invariants required for CPU↔mjlab↔mjswan policy portability.**
- `docs/wiki/repository-map.md` — where everything lives in the repo.
- `docs/wiki/index.md` — start here for navigation.
