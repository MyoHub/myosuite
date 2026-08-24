# Claude Code – Project Instructions

Read the relevant wiki pages in `docs/wiki/` before making substantial changes.

---
add minimal comments and descriptions in the code / python files
## Non-Negotiables

### The Golden Rule

> **A task has two matched halves under one `env_id`: a CPU `MyoGymnasiumEnv` (playback / fine-tune / debug) and a GPU mjlab `ManagerBasedRlEnvCfg` (parallel training). Reuse term functions and `ModelBuilder` recipes across both. Don't invent a new registration mechanism or backend-specific hack — extend the primitives instead.**

- **Two supported backends only:** CPU = `MyoGymnasiumEnv` subclass, registered via `registry.register_env(...)`; GPU = mjlab `ManagerBasedRlEnvCfg` + runner cfg, registered via `register_mjlab_task(...)`. They share one `env_id` and the cross-backend contract (`docs/wiki/cross-backend-contract.md`).
- **MJX (JAX) and the `TaskConfig` / `ModularTaskEnv` route are experimental** — may not be maintained long-term. Don't build new work on them; prefer CPU + mjlab.
- Term functions are pure and backend-agnostic — use `accessor.array_module()` only.
- Prefer typed `@dataclass` configs over raw dicts / `ConfigDict`.
- Register via `registry.register_env(...)` (CPU) or `register_mjlab_task(...)` (mjlab) — never `gym.register()` directly.
- Always return 5-tuple from `step()`: `(obs, rwd, terminated, truncated, info)`.
  **Exception:** `ModularMultiAgentTaskEnv` returns 5-tuples of per-agent dicts — intentional.
- Use `myo_sim.get_path(...)` / `ModelBuilder` for assets — no hardcoded paths.
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
- **Don't duplicate across task variants.** When writing multiple variants of the same task (e.g. different host models, difficulty tiers, or agent counts), factor shared model-builder/config logic into one parameterized function or base class before duplicating a file — don't write near-identical sibling files side by side in the same change. (Flagged in PR #101 review: `chase_tag_vs_fullbody_model.py` / `chase_tag_vs_model.py` duplicated their model builders and `ModelMeta` classes wholesale instead of parameterizing one implementation.)
- **Use the library.** Check `docs/wiki/library-usage.md` before writing any helper.
- No commented-out dead code, unused imports, or silent `except: pass`.
- Type hints on all signatures. Google-style docstrings on all public APIs.
- Python 3.10+. PEP 8. `pathlib.Path` over `os.path`.
- No `myo_` prefix on new modules or classes inside the package — it's redundant.

### Workflow

- Plan non-trivial tasks in `tasks/todo.md` before implementing.
- After any user correction: update `tasks/lessons.md`.
- Never mark a task done without proving it works.
- **Never add an AI assistant (e.g. Claude, Anthropic, Cursor, Codex, Gemini) as a commit co-author** — via `Co-Authored-By` trailers or otherwise. This repo's CLA check requires every commit author/co-author to have signed the CLA, and AI tools cannot sign it, so AI co-author trailers break the check. Enforced by the `no-ai-coauthor` `commit-msg` pre-commit hook (`scripts/reject_ai_coauthor.py`); run `pre-commit install` so it is active locally.

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

- `docs/wiki/getting-started.md` — developer onboarding; start here if new to the codebase.
- `docs/wiki/engineering-standards.md` — architecture rules, checklist for new envs.
- `docs/wiki/adding-a-new-task.md` — step-by-step worked example for registering a new task; read before adding any new env ID.
- `docs/wiki/library-usage.md` — approved library feature map; read before writing any helper.
- `docs/wiki/mjlab-design-guide.md` — canonical mjlab patterns; read before writing any mjlab code.
- `docs/wiki/cross-backend-contract.md` — obs/action/timing invariants for policy portability.
- `docs/wiki/repository-map.md` — where everything lives.
