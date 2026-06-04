# Wiki Maintenance Log

Append-only record of wiki updates and why they happened.

## [2026-04-30] bootstrap | repository wiki initialized

- Created initial wiki structure (`index`, repository map, standards, workflow, log).
- Established persistent wiki conventions for LLM-agent navigation and maintenance.
- Linked wiki usage to agent operating instructions in `CLAUDE.md`.

## [2026-04-30] refactor | full repository reorganization executed (branch: refactor/repo-reorganization)

- Phase 1: `myobase/` → `tasks/basic/<effector>/`, `myochallenge/` → `tasks/challenge/`, `myomimic/` → `tasks/mimic/`. Files renamed (dropped `_v0`/`_gymnasium_v0`). Backward-compat shims left at all old paths.
- Phase 2: `terms/` reorganized by motor control domain. `myo_*` → `base_*.py`. Combat terms → `terms/multiplayer/`.
- Phase 3: `myosuite/physics/` package created. `quat_math`, `fatigue`, `min_jerk`, `inverse_kinematics` moved from `utils/` and `envs/myo/`. Duplicate `fatigue_jax.py` merged.
- Phase 4: `renderer/` → `viz/`. `core/site_marker_viz.py` moved to `viz/`.
- Phase 5: `mjx/` + `mjlab/` → `backends/mjx/` and `backends/mjlab/`. Entry-point compat shims preserved.
- Fixed: compat shim at `mjlab/mimic_mjlab_env.py` now imports `ClipTrajectorySource` from `clip_trajectory_source` explicitly (lazy TYPE_CHECKING import was not a module-level attribute).
- Verified: 647 tests pass (71 skipped), mc26 environments functional (BoxingVs, SaberVs, BoxingMannequin, BoxingP0, BoxingVsClone all OK).

## [2026-04-30] architecture | repository reorganization plan + wiki update

- Defined target structure for `myosuite/envs/myo/tasks/` with effector × task taxonomy.
- Established `tasks/basic/<effector>/`, `tasks/mimic/`, `tasks/challenge/` as the three task collections.
- Defined `myosuite/physics/` as the single home for all biomechanics/motor control math.
- Defined `myosuite/terms/` organized by motor control domain (locomotion, posture, manipulation, reach, combat/).
- Defined `myosuite/viz/` consolidating renderer/ and scattered viz helpers.
- Established the core invariant: tasks/ = what, terms/ = how, TaskSpec = join, backend orthogonal.
- Updated `repository-map.md` and `engineering-standards.md` to reflect the new structure.
- Reorganization plan written to `tasks/reorganization_plan.md`.

## [2026-04-30] maintenance | moved wiki under docs

- Relocated wiki files from `wiki/` to `docs/wiki/` to keep the repository root cleaner.
- Updated `CLAUDE.md` mandatory wiki references to point to `docs/wiki/*`.
- Updated internal wiki links and workflow instructions to use `docs/wiki/*`.
