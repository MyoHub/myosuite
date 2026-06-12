# Claude Code – Project Instructions

Read the relevant wiki pages in `docs/wiki/` before making substantial changes.

---

## Non-Negotiables

### The Golden Rule

> **A new task requires: one `TaskSpec`, term functions, and a `ModelBuilder` recipe. No task-specific env subclass. No new registration file. No backend-specific code.**

- Term functions are pure and backend-agnostic — use `accessor.array_module()` only.
- All configs are typed `@dataclass` — never raw dicts or `ConfigDict`.
- All envs registered via `register(EnvSpec(...))` — never `gym.register()` directly.
- Always return 5-tuple from `step()`: `(obs, rwd, terminated, truncated, info)`.
  **Exception:** `ModularMultiAgentTaskEnv` returns 5-tuples of per-agent dicts — intentional.
- Use `myo_sim.get_path(...)` for assets — no hardcoded paths.
- New envs must include a `CitationBundle` — see `myosuite/core/citation.py`.

### mjlab Non-Negotiables

> **Read `docs/wiki/mjlab-design-guide.md` and `docs/wiki/library-usage.md` before touching any mjlab file.**

- State reads: use `entity.data.*` (e.g. `entity.data.joint_pos`) — never `entity.data.data.*`.
  Accepted exceptions (no stable API equivalent): `data.data.act`, `data.data.cvel`, `data.data.actuator_*`.
- State writes: use the Entity write API (`write_joint_state_to_sim`, `write_root_state_to_sim`, etc.) — never write to `wp_data` or `data.qpos` directly.
- No Python loops over environments in obs/reward/termination functions — all must be fully vectorised.
- Env-id normalisation: import `normalize_mjlab_env_ids` from `mjlab_env_base` — never re-implement it.
- `MjSpec` → `MjModel`: always call `.compile()` before passing a spec to functions that expect a `MjModel` (e.g. `_muscle_tendon_names`, `_init_state_from_model`).
- `wp_data` attribute names: verify they exist in mujoco-warp before use — do not assume by analogy with mujoco-py (e.g. `site_xvelp` does not exist; derive from `cvel`).

### Code Quality

- **Search before writing.** `grep` the repo first. If a similar implementation exists (≥ 80%), extend or import it.
- **Use the library.** Check `docs/wiki/library-usage.md` before writing any helper.
- No commented-out dead code, unused imports, or silent `except: pass`.
- Type hints on all signatures. Google-style docstrings on all public APIs.
- Python 3.10+. PEP 8. `pathlib.Path` over `os.path`.
- No `myo_` prefix on new modules or classes inside the package — it's redundant.

### Workflow

- Plan non-trivial tasks in `tasks/todo.md` before implementing.
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
pytest myosuite/tests/test_boxing_registry.py -v
pytest myosuite/tests/test_saber_env.py -v
```

---

## Wiki

- `docs/wiki/engineering-standards.md` — architecture rules, checklist for new envs.
- `docs/wiki/library-usage.md` — approved library feature map; read before writing any helper.
- `docs/wiki/mjlab-design-guide.md` — canonical mjlab patterns; read before writing any mjlab code.
- `docs/wiki/cross-backend-contract.md` — obs/action/timing invariants for policy portability.
- `docs/wiki/repository-map.md` — where everything lives.
