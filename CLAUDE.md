# Claude Code – Project Instructions

Read the relevant wiki pages in `docs/wiki/` before making substantial changes.

---

## Non-Negotiables

### The Golden Rule

> **A new task requires: one `TaskSpec`, term functions, and a `ModelBuilder` recipe. No task-specific env subclass. No new registration file. No backend-specific code.**

- Term functions are pure and backend-agnostic — use `accessor.array_module()` only.
- All configs are typed `@dataclass` — never raw dicts or `ConfigDict`.
- All envs registered via `register(EnvSpec(...))`.
- Always return 5-tuple from `step()`: `(obs, rwd, terminated, truncated, info)`.
- Use `myo_sim.get_path(...)` for assets — no hardcoded paths.

### Code Quality

- **Search before writing.** `grep` the repo first. If a similar implementation exists (≥ 80%), extend or import it.
- **Use the library.** Check `docs/wiki/library-usage.md` before writing any helper.
- No commented-out dead code, unused imports, or silent `except: pass`.
- Type hints on all signatures. Google-style docstrings on all public APIs.
- Python 3.10+. PEP 8. `pathlib.Path` over `os.path`.

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
```

---

## Wiki

- `docs/wiki/engineering-standards.md` — architecture rules, checklist for new envs.
- `docs/wiki/library-usage.md` — approved library feature map; read before writing any helper.
- `docs/wiki/mjlab-design-guide.md` — canonical mjlab patterns; read before writing any mjlab code.
- `docs/wiki/cross-backend-contract.md` — obs/action/timing invariants for policy portability.
- `docs/wiki/repository-map.md` — where everything lives.
