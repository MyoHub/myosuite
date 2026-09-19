# Consistency Rename/Move Matrix (Strict PEP8, full-breaking)

Date: 2026-04-16

This document is the execution-ready mapping of **old → new** names/paths, grouped into ordered waves. It also lists the expected blast radius so refactors are systematic and testable.

## Wave A — Internal symbol normalization (low risk first)

### A1) Quaternion helper functions (`camelCase` → `snake_case`)

- **`myosuite/utils/quat_math.py`**
  - `mulQuat` → `mul_quat`
  - `negQuat` → `neg_quat`
  - `diffQuat` → `diff_quat`
  - `quatDiff2Vel` → `quat_diff_to_vel`
  - `rotVecMatT` → `rot_vec_mat_t`
  - `rotVecMat` → `rot_vec_mat`
  - `rotVecQuat` → `rot_vec_quat`

- **`myosuite/utils/quat_math_jax.py`** (same mapping as above)

- **Blast radius**
  - Direct in-repo usage detected in:
    - `myosuite/tests/mjx/test_quat_math.py`
  - Additional risk: downstream user code importing these helpers.

- **Execution notes**
  - Since this is full-breaking, we will update call sites and tests in-repo and remove old aliases once tests pass.

### A2) Non-CapWords class naming (local cleanups)

- `myosuite/logger/roboset_logger.py`
  - `RoboSet_Trace` → `RoboSetTrace`

- `myosuite/utils/examine_env.py`
  - `rand_policy` → `RandPolicy`

- **Blast radius**
  - Expected low (mostly local). Confirm via repo-wide symbol search before applying.

### A3) Replace unittest-style fixtures in pytest suite

- Convert `setUp`/`tearDown`/`setUpClass` usage in:
  - `myosuite/tests/test_editor.py`
  - `myosuite/tests/test_heightfields.py`
  - `myosuite/tests/test_mjx.py`
  - `myosuite/tests/mjx/test_fatigue.py`
  - `myosuite/tests/mjx/test_quat_math.py`
  - `myosuite/tests/mjx/test_reference_motion.py`

- **Blast radius**
  - Tests only.

## Wave B — Module/file/package structure normalization

### B1) Reflex baseline module normalization

- Directory rename:
  - `myosuite/agents/baseline_Reflex/` → `myosuite/agents/baseline_reflex/` (done)

- File renames:
  - `myosuite/agents/baseline_reflex/ReflexCtrInterface.py` → `myosuite/agents/baseline_reflex/reflex_ctr_interface.py` (done)
  - `myosuite/agents/baseline_reflex/reflexCtr.py` → `myosuite/agents/baseline_reflex/reflex_ctr.py` (done)

- Tutorial file renames (for consistency with baseline code):
  - `tutorials/4b_reflex/ReflexCtrInterface.py` → `tutorials/4b_reflex/reflex_ctr_interface.py` (done)
  - `tutorials/4b_reflex/reflexCtr.py` → `tutorials/4b_reflex/reflex_ctr.py` (done)

- **Blast radius**
  - Docs referencing old paths/names:
    - `docs/source/baselines.rst` (imports `ReflexCtrInterface`)
    - `docs/source/quickstart_neuroscience.rst` (mentions `baseline_Reflex`)
  - Any user scripts/tutorial instructions.

### B2) SAR tutorial/module normalization

- Directory rename:
  - `tutorials/SAR/` → `tutorials/sar/` (done)

- File renames:
  - `tutorials/sar/SAR_tutorial_utils.py` → `tutorials/sar/sar_tutorial_utils.py` (done)

- **Blast radius**
  - `tutorials/sar/run_sar_full.py` now imports `sar_tutorial_utils` as a sibling module (done).
  - `scripts/train_sar_jax_ppo.py` references tutorial paths in docstrings and comment text.

### B3) `path_utils` vs `paths_utils` consolidation

- Canonical split (target state):
  - `myosuite/utils/path_utils.py`: pure library utilities (typed, minimal deps)
  - `myosuite/utils/path_cli.py`: CLI tooling formerly embedded in `paths_utils.py`
  - `myosuite/utils/path_plotting.py`: plotting/render helpers (optional deps)

- Planned mapping:
  - `myosuite/utils/paths_utils.py` → split into `path_cli.py` and `path_plotting.py`
  - Update imports:
    - `myosuite/utils/examine_env.py`: `paths_utils.plot` → `path_plotting.plot`
    - `myosuite/logger/examine_logs.py`: `paths_utils.plot` → `path_plotting.plot`

- **Blast radius**
  - Internal call sites above, plus any external usage of `paths_utils.py` as a script.

## Wave C — Public API and registration normalization

### C1) Make registration explicit (reduce `import myosuite` side effects)

- Target state:
  - Keep `myosuite/__init__.py` lightweight and explicit.
  - Provide explicit entrypoints like `myosuite.register_all_envs()` or `myosuite.envs.register_all()` and update tests/scripts accordingly.

- **Blast radius**
  - Many tests/scripts explicitly rely on side-effect import (`import myosuite  # triggers registrations`).
  - Must update:
    - `myosuite/tests/test_entry_points.py`
    - `myosuite/tests/test_parity.py`
    - `scripts/generate_parity_baselines.py`
    - benchmark scripts under `benchmarks/sar_backends/`

### C2) Backend registry consistency (cpu/mjlab/mjx)

- Align naming and discovery APIs between:
  - `myosuite/core/registry.py` (`register_env`, `make_env`)
  - `myosuite/envs/myo/mjx/` (MJX `make`, `ALL_ENVS`)
  - `myosuite/envs/myo/mjlab/` (`REGISTERED_TASKS`, `register_mjlab_tasks`)

- **Blast radius**
  - High: env IDs + backend behavior is user-facing and heavily tested.

## Wave D — Update all dependents

- Update all imports in:
  - Python sources under `myosuite/`, `scripts/`, `benchmarks/`, `tutorials/`
  - Docs under `docs/source/`
- Update any references to moved paths in `.rst` links and docstrings.

## Wave E — Remove old names/paths

- Remove compatibility aliases and deleted modules once tests and minimal smoke checks pass.
