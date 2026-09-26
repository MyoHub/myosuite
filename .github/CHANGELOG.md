# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1.0] - 2026-04-20

### Added
- **MuscleMimic full-body policy tracking test** (`test_notebook_policy_tracking.py`): 10-test end-to-end suite covering checkpoint loading, policy rollout, off-screen rendering, and MP4 video saving.
- **New tutorial**: `MuscleMimic_Fullbody_Policy_Trajectory.ipynb` — load `hf://amathislab/mm-10m-2`, track a KIT walking clip, render and save video.
- **Coverage gap tests** (`test_coverage_gaps.py`): 39 new tests for `CumulativeFatigue`, `apply_sarcopenia`, `apply_fatigue`, `resolve_motion_path`, `load_motion_clip`, `walk_env_reward`, `MuscleActionTerm`, registry variant paths, and `make_env` import-error paths.
- `modular_task_config.ipynb` tutorial for `TaskSpec`-first environment authoring.

### Changed
- **MyoChallenge registration**: All challenge environments (`Bimanual`, `TableTennis`, `Soccer`, `RunTrack`, `ChaseTag`) now register against Gymnasium-native implementations. The transitional `*Native-v0` duplicate IDs have been removed.
- **Legacy code removed**: Deleted `tabletennis_v0.py`, `reorient_v0.py`, `bimanual_v0.py`, `reorient_sar_v0.py`, `myodm_v0.py`; removed Robot delegate pattern from `bimanual_gymnasium_v0.py`; replaced Robot-based step/reset in `tabletennis_gymnasium_v0.py` with direct MuJoCo API calls.
- **DeprecationWarnings removed** from `env_base.py` and `base_v0.py`; all `LEGACY_REMOVE_AFTER` markers cleaned.
- Per-env MyoChallenge parity test files removed (superseded by `test_parity.py`).
- README, `tutorials/ReadMe.md`, and `docs/backend_compatibility_matrix.md` updated to reflect v4 state.

### Fixed
- `bimanual_gymnasium_v0.py`: native mode was shadowed by the legacy delegate; delegate removed so native MuJoCo path is active.
- `tabletennis_gymnasium_v0.py`: removed stale `_sync_state_from_sensors()` call.

## [2.4.0] - 2024-05-13
[FEATURE] Added 3CC-r Fatigue Model (#167). Thanks to @fl0fischer
[FEATURE] Update to MuJoCo 3.1.2 and dm-control 1.0.16 (2bddf8c)
[BUGFIX] Fixed Tutorial `2_Load_Policy.ipynb` (038457a)

## [2.3.0] - 2024-05-01
[FEATURE] Support for both Gym/Gymnasium (#142)
[FEATURE] Add support for TorchRL by @vmoens (5efdf93)
[FEATURE] Improve Inverse Dynamics tutorial (98daff2). Thanks to @andreh1111

## [2.2.0] - 2024-01-20
[FEATURE] Inverse dynamics tutorial. Thanks to @andreh1111 #121
[RELEASE] MyoArm and MyoLeg models (4c01023, cd9a25e)
[RELEASE] MyoChallenge'23 environments release (#128)
[BUGFIX] Fixed heightfield collisions for myoleg scenes #132
[BUGFIX] Fixed names of data keys from _int to _init in myodm by @andreh1111 in (#119)

## [1.3.0] - 2023-01-11
- Rebase and building on RoboHive v0.3

## [1.2.4] - 2022-11-12
- fix Baoding Ball environment for MyoChallenge Phase 1

## [1.2.3] - 2022-10-21
- update horizon for MyoChallenge Die Reorient task - Phase 2

## [1.2.2] - 2022-10-21
- update MyoChallenge Die Reorient task and Baoding Ball to Phase 2

## [1.2.1] - 2022-10-09
- update horizon for MyoChallenge Die Reorient task
- update tutorials

## [1.2.0] - 2022-08-13
- Rebase and building on RoboHive v0.2
- Adding the myochallenge envs
- Fundamental bugfixes on the RoboHive engine
- Bugfixes on myo environments as well
- Closes baselines are on RoboHive-v0.2
- Next planned baseline release will align when Robohive-v0.3dev moves to prerelease.
- Renaming the metrics for clarity and changed sign from `act_mag` to `effort` and `solved` to `score`

## [1.1.0] - 2022-08-12
- Upgrade to mj_env v0.2 experimental
- add Die Rotation and Baoding Ball task for MyoChallenge (https://sites.google.com/view/myochallenge)

## [1.0.1] - 2022-05-23
- First Release of MyoSuite.
- Basic Documentation
