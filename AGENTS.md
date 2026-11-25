# Agent Instructions (AGENTS.md; CLAUDE.md imports this file)

## Code Review Rules

The review rules for every reviewer (people, Claude Code Review, Codex, Antigravity/Gemini, local `scripts/review/local_review.sh`) are in [REVIEW.md](REVIEW.md). Read it before reviewing or opening a pull request.

### Core-team PR review and merge policy

| Change | Gate |
|---|---|
| Routine docs, changelog, wording (`docs/**`, `*.md`; not `REVIEW.md`, `AGENTS.md`, `CLAUDE.md`, `.claude/`, `.agents/`, `.github/`, `scripts/review/`) | Auto-merge after the local checks (`scripts/review/local_review.sh`) |
| Bug fix preserving supported contracts | Approval from a person other than the PR opener, after verified reproduction, regression checks and the local checks |
| Correct fix intentionally changing behavior or compatibility | Human approval of the consequences, then merge after checks |
| New feature, architecture, design or agent instructions | Human agreement on scope and implementation approach before coding; human implementation approval before merge |
| Pure refactor or other change outside the docs auto-merge lane | Human implementation approval after relevant checks |
| Unproven bug, uncertain impact, or accidental regression | Block pending evidence or resolution |

- The docs auto-merge lane applies only to core-team pull requests opened by a person (agent-assisted work included). Mixed changes take the stricter gate. Existing branch protections and required checks still apply; this policy does not install automation or authorize bypassing protections. External contributions require human approval.
- Auto-merge requires the local checks recorded on the latest revision (the PR's "Local checks" block), passing applicable required checks, and no unresolved blocking findings or change requests. The authoring agent cannot approve its own work. Missing checks/evidence or uncertain classification require human review; a human approval cannot substitute for fixing an accidental regression.
- **Every bug fix requires a minimal reproduction:** expected behavior and its basis, smallest input, command, dependencies/seed where relevant, and exact target/proposed commits. Run the same reproduction: it must fail for the claimed reason before the fix and pass afterward. Retain a focused regression test where practical.
- Validate the expected result against a documented contract, approved design, reference calculation, or established supported behavior; an LLM-generated assertion alone is not proof. Setup failures do not prove an unrelated bug. GPU/rendering/stochastic defects may use reproducible scripts with measured results. Unexecuted evidence is unverified and cannot qualify for auto-merge.
- Agent reviewers must support blocking bug findings with a reproduction or concrete, verifiable counterexample. Report unsupported concerns as questions.
- Human review is required for changes to observation/action contracts, rewards, termination, reset distributions, timing, CPU/mjlab parity, published checkpoint compatibility, or scientifically meaningful model behavior. Also require it for public API/support, dependency, licensing, CI/release, design/review policy, baseline, and tolerance changes. Do not update baselines or loosen tolerances just to pass checks.
- Link human design agreement in feature/policy PRs. Human implementation approval must come from an eligible maintainer other than the author; an agent acting through a human account does not count. New implementation commits require renewed approval; scope changes require renewed design agreement. Revalidate affected checks when the base changes.
- Keep one updated review summary: category, reproduction results, compatibility impact, checks run/not run, and merge decision. Separate defects from optional suggestions.

Read the relevant wiki pages in `docs/wiki/` before making substantial changes.

---
add minimal comments and descriptions in the code / python files
## Non-Negotiables

### The Golden Rule

> **A task has two matched halves under one `env_id`: a CPU `MyoGymnasiumEnv` (playback / fine-tune / debug) and a GPU mjlab `ManagerBasedRlEnvCfg` (parallel training). Reuse term functions and `ModelBuilder` recipes across both. Don't invent a new registration mechanism or backend-specific hack — extend the primitives instead.**

- **Two supported backends only:** CPU = `MyoGymnasiumEnv` subclass, registered via `registry.register_env(...)`; GPU = mjlab `ManagerBasedRlEnvCfg` + runner cfg, registered via `register_cpu_twins(...)` (twins of CPU envs) or `register_mjlab_task(...)` (special tasks). They share one `env_id` and the cross-backend contract (`docs/wiki/cross-backend-contract.md`).
- **MJX (JAX) and the `TaskConfig` / `ModularTaskEnv` route are experimental** — may not be maintained long-term. Don't build new work on them; prefer CPU + mjlab.
- Term functions are pure and backend-agnostic — use `accessor.array_module()` only.
- Prefer typed `@dataclass` configs over raw dicts / `ConfigDict`.
- Register via `registry.register_env(...)` (CPU) or `register_cpu_twins(...)` / `register_mjlab_task(...)` (mjlab) — never `gym.register()` directly.
- Build envs with `from myosuite import make_env` (`make_env(env_id)`, or `make_env(EnvConfig(...), backend=...)`) — not `gym.make()` — in code, tests, docs and tutorials.
- Always return 5-tuple from `step()`: `(obs, rwd, terminated, truncated, info)`.
  **Exception:** `ModularMultiAgentTaskEnv` returns 5-tuples of per-agent dicts — intentional.
- Use `myo_sim.get_path(...)` / `ModelBuilder` for assets — no hardcoded paths.
- **Training/learning implementations:** read `docs/wiki/training-guidelines.md` before adding trainers, world models, algorithm-specific evaluation, or method-specific dependencies.
  - Method-specific code (network architectures, trainers, buffers, self-play pools) goes in `training/<method>/`, not `myosuite/`, even when it only serves to load that method's checkpoints. Reviewers check placement against the guidelines' *Boundary* section.
  - A new method ships with PPO comparisons on all four [required benchmark environments](docs/wiki/training-guidelines.md#benchmark-against-ppo), with matched seeds and budgets. CPU training and evaluation are mandatory; both CPU and GPU (mjlab) are preferred. Reviewers verify coverage, results and linked human approval for any exception.

### CPU Env Contract

The CLAUDE.md verification list does not cover these. `test_obs_contract.py`, `test_reset_reproducibility.py` and `test_runtime_geometry_consistency.py` check them.

- **Step overrides.**
  - Advance physics with `self._step_physics()`, which runs `mj_step` and then `mj_forward`. Finish with `self._finalize_step(obs_dict, rwd_dict)`.
  - Never refresh with `mj_kinematics` alone. Muscle length/velocity/force, sensors, contacts and `cvel` would then lag `qpos` by one substep.
- **Observations are float32 and never clipped.** Declare spaces with `self._unbounded_obs_space(dim)` (see `docs/wiki/cross-backend-contract.md`).
- **`reset(seed=...)` must fully determine the episode.**
  - Draw every random quantity from `self.np_random`, including in sub-components: the fatigue reset (`FatigueWrapper`'s stage, run by `CtrlStageHost._run_reset_stages()`), opponents and terrain.
  - Never use `np.random.*`, `random.*` or an unseeded `default_rng()`.
  - Clear all per-episode state: reward dicts, buffers, counters.
- **Mocap targets.** After `mj_resetData`, write a mocap target into `data.mocap_pos` / `data.mocap_quat`. Writing only `model.body_pos` takes effect at the *next* reset.
- **Runtime model edits.** After editing a compiled model's `geom_type/size/pos/quat` or `body_mass`, call `myosuite.utils.mujoco_geom_utils.refresh_geom_derived_fields(model, spec, body_ids)`.
  - Otherwise the collision bounds and inertia go stale, and MuJoCo ≥ 3.8 silently culls contacts against the stale bounds.

### mjlab Non-Negotiables

> **Read `docs/wiki/mjlab-design-guide.md` and `docs/wiki/library-usage.md` before touching any mjlab file.**

- **Physics options:** build every `MujocoCfg` from the CPU model with `tasks.cpu_reference.mujoco_cfg_from_model()`. mjlab ignores the spec's `<option>` and defaults to `implicitfast`, while every CPU model uses Euler.
- **Root velocities:** `write_root_state_to_sim` takes a world-frame angular velocity, but a MuJoCo free-joint `qvel[3:6]` is body-frame. Rotate it first.
- **Joint-name keys:** `InitialStateCfg.joint_pos` keys are regexes, so anchor exact names (`^name$`). Otherwise `knee_angle_r` also matches the knee coupling joints.
- **Per-episode state** (e.g. a previous distance or rally state) lives on a `ManagerTermBase` with `reset(env_ids)`, not on the env object.
- **Step order:** mjlab runs termination → reward → reset → `forward()` → observation. Compute per-step derived state once, in the first termination term, before rewards read it. Use `mdp.sync_forward` when rewards or terminations score derived quantities.
- **Contact buffers:** mujoco-warp does not clear contact rows at or beyond `nacon`; mask them.

- State reads: use `entity.data.*` (e.g. `entity.data.joint_pos`) — never `entity.data.data.*`.
  Accepted exceptions (no stable API equivalent): `data.data.act`, `data.data.cvel`, `data.data.actuator_*`.
- State writes: use the Entity write API (`write_joint_state_to_sim`, `write_root_state_to_sim`, etc.) — never write to `wp_data` or `data.qpos` directly.
- No Python loops over environments in obs/reward/termination functions — all must be fully vectorised.
- Env-id normalisation: import `normalize_mjlab_env_ids` from `mjlab_env_base` — never re-implement it.
- `MjSpec` → `MjModel`: always call `.compile()` before passing a spec to functions that expect a `MjModel` (e.g. `_muscle_tendon_names`, `_init_state_from_model`).
- `wp_data` attribute names: verify they exist in mujoco-warp before use — do not assume by analogy with mujoco-py (e.g. `site_xvelp` does not exist; derive from `cvel`).

### Ownership and minimum-change review

The ownership (`myo_sim` vs MyoSuite) and minimum-change rules live in [REVIEW.md](REVIEW.md). Apply them when writing code, not only when reviewing it.

### Code Quality

- **Search before writing.** Follow the ownership and minimum-change rules in REVIEW.md: search both repositories and compare existing behavior and constraints before extending or importing a primitive.
- **Don't duplicate across task variants.** When writing multiple variants of the same task (e.g. different host models, difficulty tiers, or agent counts), factor shared model-builder/config logic into one parameterized function or base class before duplicating a file — don't write near-identical sibling files side by side in the same change. (Flagged in PR #101 review: `chase_tag_vs_fullbody_model.py` / `chase_tag_vs_model.py` duplicated their model builders and `ModelMeta` classes wholesale instead of parameterizing one implementation.)
- **Use the library.** Check `docs/wiki/library-usage.md` before writing any helper.
- No commented-out dead code, unused imports, or silent `except: pass`.
- Type hints on all signatures. Google-style docstrings on all public APIs.
- Python 3.10+. PEP 8. `pathlib.Path` over `os.path`.
- No `myo_` prefix on new modules or classes inside the package — it's redundant.

### Package Boundaries and Motion Analysis

Use these boundaries for new code and small, compatible refactors. This PR defines ownership and dependency rules; migrations remain separate changes.

| Location | Responsibility |
|---|---|
| `myo_sim` | Anatomical assets/composition, reusable model mechanics and physical transforms. MyoSuite consumes its public APIs. |
| `core/` | Environment configuration, registration and task assembly (`make_env`, `ModelBuilder`, recipes), delegating anatomical composition to `myo_sim`. |
| `physics/` | Shared numerical helpers and compatibility imports; delegate canonical model mechanics to `myo_sim`. |
| `motion/` (target) | Small, flat subpackage: `clip.py` (motion contracts), `io.py` (model-coordinate trajectory I/O), `transforms.py` (trajectory transforms). |
| `biomechanics/` | Scientific fitting, analysis and control: IK/retargeting solvers, inverse dynamics, static optimization, CMC, EMG, effort and diagnostics. |
| `integrations/` | Lightweight source-specific readers, adapters and calibration (e.g. markers, InstantHMR, GVHMR, SMPL+H, MANO); call shared fitting APIs. |
| `viz/` | Rendering and visual reconstruction, including muscle meshes, skin and Blender. |
| `scripts/`, `tools/` | Thin workflows, batch execution and reports; isolate heavyweight perception inference in standalone tools with their own dependencies. |

- **Reuse existing motion APIs.** `motion/` is not implemented yet. Reuse `MotionClip` in `core/trajectory_io.py` and transforms in `core/trajectory_yaw.py` until a separate migration lands. Prefer `motion` to `mocap`: these contracts also serve simulation and playback. Keep task-specific sampling in tasks; consolidate reference-motion implementations only after comparing their contracts.
- **Allowed dependencies.** Consumers may import shared motion contracts; motion must not import environments, trainers, visualization or source adapters. Adapters call reusable fitting/analysis APIs; those algorithms must not import adapters, environment implementations or visualization. Workflows compose these layers. MyoSuite imports public `myo_sim` APIs, never the reverse. Define allowed module imports rather than assuming every directory is one dependency tier.
- **Separate algorithms from workflows.** Numerical solvers must not own dataset traversal, caches, reports or rendering. Policy loading/export is reusable runtime functionality consumed by benchmarks, not owned by them.
- **Explicit data contracts.** Document units, timing, coordinate frames, quaternion order, joint ordering and missing-data masks. Keep marker observations and source body-model poses distinct from model-coordinate trajectories. Compose EMG with optional kinematic data; support EMG-only trials. Return structured convergence/residual/quality diagnostics and document mutation/failure policy.
- **Preserve numerical intent.** Site IK and coupling-aware landmark fitting have different objectives. Share contracts and demonstrated common primitives; do not merge solvers without proving equivalent constraint handling and numerical behavior.
- **Small, compatible migrations.** Extract workflows first, then consolidate shared motion contracts. Preserve old public imports, serialization and checkpoint compatibility, update callers/docs, and verify numerical behavior. Keep lightweight readers installable without heavy perception stacks; avoid eager heavyweight and private cross-package imports. Test internal dependency boundaries as well as third-party imports.
- **Separate design decisions.** Broader folder renames, CPU/GPU task co-location, moving PPO presets/entry points, test relocation or wheel exclusions, dependency-extra changes, and MuJoCo/myo-sim upgrades require separate proposals and validation. Preserve current training commands/default runners and the CPU/mjlab contract. Audit all runtime consumers (including reference-motion downloads) before changing extras; measure transitive installs and Python/platform coverage before claiming installation improvements. This guidance does not commit the broader layout to v3.2.0.

### Workflow

- Plan non-trivial tasks in `tasks/todo.md` before implementing.
- After any user correction: update `tasks/lessons.md`.
  (`tasks/` is git-ignored local scratch space, not part of the repository.)
- Never mark a task done without proving it works.
- **Small PRs.** One concern per PR: a fix, a refactor, a task, or a doc change — not several at once.
  - Split unrelated changes into separate PRs, even when found together.
  - **Stack dependent PRs.** Base each PR on the branch of the PR it depends on; a PR's base must be a MyoHub/myosuite branch, so push a fork's prerequisite branch to MyoHub first. When a lower PR is updated or rebased, rebase every PR above it (GitHub's stack rebase); updating only the bottom PR leaves the ones above carrying stale copies of its commits.
  - Never carry another open PR's commits, except when its branch can't be pushed to MyoHub: then open the dependent PR as a draft that names the PR it stacks on, and rebase it onto `dev` once that PR lands.
  - Keep a PR reviewable in one sitting (aim for < ~500 changed lines excluding generated files); split larger work into a sequence.
  - Exception: release PRs into `main` (`dev` → `main`) bundle already-reviewed small PRs and can be as large as the release needs. Don't add new, unreviewed changes in them.
- **Model weights live on Hugging Face, not in git.** Never commit `.pt`, `.pth`, `.onnx`, `.ckpt` or `.safetensors` files — not in `myosuite/`, `tutorials/`, `docs/` or `training/`.
  - Upload them to a MyoHub Hugging Face repo and download at a pinned revision (tag or commit): `myosuite.core.hf_io` for baselines, otherwise `hf_hub_download(..., revision=...)`. Never download from `main` unpinned.
  - Enforced by the `no-model-weights` pre-commit hook. The three legacy tutorial checkpoints it exempts are to be migrated; don't add to that list.
- **Branching and CI.** Branch from `dev` and open PRs against `dev`.
  - Isolated or agent worktrees may start from `main`, so create the branch from `dev` explicitly.
  - CI runs for non-draft pull requests into `main` and `dev` (a draft stays quiet until it is marked ready) that touch code; docs, markdown, `training/` and tooling-config changes skip it. Run the Verification block below locally before pushing.
  - A PR into `dev` runs Ubuntu / Python 3.11 as two parallel jobs, `mjlab` and `cpu` (`conftest.py` marks every module that uses mjlab / MuJoCo Warp with `mjlab`), and the tutorials only when it touches `tutorials/`. Python-only PRs can select affected tests with pytest-testmon, against the record of the last push to `dev`. The five Verification modules below plus `test_term_catalog.py`, `test_import_footprint.py`, `test_mjlab_warning_filters.py` and `test_train_mjlab.py` still run without testmon selection in their applicable group. Registration modules, `__init__.py`, data files under `myosuite/`, `tutorials/`, dependencies and test plumbing run the full job. The SQLite record is checkpointed before caching; a busy checkpoint fails the job. The push to `dev` after a merge, release PRs into `main` and the nightly run cover the full matrix: 3.11 on macOS, Ubuntu and Windows, 3.10, 3.14 and 3.15 on Ubuntu, the tutorials, and (nightly) the `slow` tests.
- **Parity baselines.** Regenerate (`scripts/generate_parity_baselines.py --env-id <id>`) only the envs your change intentionally alters.
  - Generate them only with the package versions CI resolves. CI installs with `uv pip install -e ".[...]"` and sets `UV_NO_SYNC=true` so tests and tutorial subprocesses keep that fresh resolution. Its dependency report records the tested versions; `uv.lock` is not used for this job.
  - Never skip a case as "non-deterministic" without checking that a replay really differs between processes.
  - Make sure every tolerance fails on a known-bad mutation.
- **Registered-env changes.** If a change alters a registered env's observations, actions, dynamics or task distribution, add a CHANGELOG entry and update the compatibility notes in `docs/baseline_checkpoints.md`. Published checkpoints are tied to the old contract.
- **Never author a commit as an AI assistant, and never add one as a co-author.** `Author` and `Committer` must be the human who signed the CLA — not `Cursor Agent`, `cursoragent@cursor.com`, or any other AI identity. Do not add `Co-Authored-By` (or similar) trailers for Claude, Anthropic, Cursor, Codex, Gemini, or other AI tools. AI tools cannot sign the CLA, so an AI author or co-author fails the check. Enforced by the `no-ai-coauthor` `commit-msg` hook (`scripts/reject_ai_coauthor.py`), which rejects both an AI `Author`/`Committer` and AI attribution lines; run `pre-commit install` so it is active locally.


### Maintainer verification of commit attribution

This is a maintainer check, not an automated code-review finding.

- List the submitted commits with `GET /repos/MyoHub/myosuite/pulls/<number>/commits`. Only SHAs in that response are evidence about this PR.
- For those exact SHAs, inspect the API's stored `commit.author`, `commit.committer` and commit message for prohibited AI attribution. GitHub account associations and local configuration are not stored commit identities.
- If a cited SHA is absent from that list or GitHub cannot retrieve it, reject the attribution claim. Do not rewrite valid human-attributed history to match a review sandbox.
- `scripts/reject_ai_coauthor.py` checks pending commit creation using `git var`; running it in a review sandbox does not audit existing PR commits.

---

## Verification (run before every commit)

```bash
pre-commit run --all-files
pytest myosuite/tests/test_model_builder.py -v
pytest myosuite/tests/test_terms_cpu.py -v
pytest myosuite/tests/test_fragment_compat.py -v
pytest myosuite/tests/test_parity.py -v      # atol ≤ 1e-6 — regressions block PR
pytest myosuite/tests/test_registry.py -v
```

Boxing and beat-saber task implementations were removed from this repo (archived separately,
commit `68a77811`) — there is no `test_boxing_registry.py` / `test_saber_env.py` to run anymore.

---

## Wiki

- `docs/wiki/getting-started.md` — developer onboarding; start here if new to the codebase.
- `docs/wiki/engineering-standards.md` — architecture rules, checklist for new envs.
- `docs/wiki/adding-a-new-task.md` — step-by-step worked example for registering a new task; read before adding any new env ID.
- `docs/wiki/library-usage.md` — approved library feature map; read before writing any helper.
- `docs/wiki/writing-term-functions.md` — signatures and the batch rule for obs / reward / termination / action terms.
- `docs/wiki/mjlab-design-guide.md` — canonical mjlab patterns; read before writing any mjlab code.
- `docs/wiki/cross-backend-contract.md` — obs/action/timing invariants for policy portability.
- `docs/wiki/training-guidelines.md` — placement and dependency-isolation rules for training/learning methods.
- `docs/wiki/repository-map.md` — where everything lives.
