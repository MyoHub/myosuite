# Changelog

All notable changes to this project are documented in this file.

## [2.13.0] - unreleased

Changes since the last official release, **v2.12.2** (2026-05-06). `git log v2.12.2..v2.13.0` has the
full commit list.

### Highlights

* **One task, three execution paths.** The same `env_id` runs on the **CPU** (Gymnasium: playback,
  debugging, Stable-Baselines3), on **mjlab** (MuJoCo Warp + RSL-RL: thousands of parallel
  environments on one GPU) and on an **experimental MJX** (JAX) path. CPU and mjlab share the
  observation, action and timing contract, so a policy trained on the GPU can be replayed on the CPU.
* **MuscleMimic support.** Full-body and bimanual MuscleMimic environments
  (`myoMimicFullbody-v0`, `myoMuscleMimicFullbody-v0`, `myoMimicBimanual-v0`,
  `myoMuscleMimicBimanual-v0`, `myoFullBodyDirectional-v0`), loaders for the MuscleMimic
  checkpoints and retargeted motion datasets from Hugging Face, the `myosuite-musclemimic-fullbody-eval`
  tool, and tutorials 5.1–5.5 (load a policy, train on CPU/JAX and with mjlab, directional
  locomotion, SAR).
* **The full MyoChallenge suite as Gymnasium environments** (21 ids): Baoding, Bimanual, ChaseTag
  (P1, P2, P2eval, full-body P2, 1v1 full-body), Die Reorient, OSL run, Relocate, Soccer and Table
  Tennis, with mjlab versions of the ChaseTag full-body and Table Tennis tasks.
* **Modular task framework.** A new task needs a `TaskSpec`/`EnvSpec`, term functions and a
  `ModelBuilder` recipe, without subclassing an environment class. Term functions (observation, reward,
  action) are backend-agnostic; `ModelBuilder` composes models from fragments, recipes and pre-built
  `MjSpec` fragments; a data-driven `TaskConfig` environment covers simple tasks.
* **mjlab twins of the basic suite (118 env ids).** Pose, reach (hand, finger, arm, elbow, motor finger),
  hand numeral poses, torso (incl. exosuit) and leg tasks (walk, directional, stand, terrain), each
  with sarcopenia, fatigue and reafferentation variants where the CPU env has them. Twins are built from
  the CPU registration and covered by step-parity tests.
* **Training and evaluation tooling.** Shared PPO defaults for muscle tasks, an `Episode_Metrics/success`
  metric on every twin, `train_mjlab.py` that stops early once the deterministic policy succeeds,
  `eval_mjlab_policy.py` (success rate, return, grid videos on both backends), and ready-made
  policies with evaluation videos in `baselines/`.
* **Muscle conditions.** A torch fatigue model with CPU parity, episode-persistent and resumable
  fatigue states, spec-level sarcopenia and reafferentation.
* **`myo_sim` as a pip package** (replacing the git submodule); models are composed from it and the
  hand, arm and leg orientations were re-calibrated.
* **Tutorials and documentation.** 20 notebooks in five numbered tracks (basics, training,
  analysis, musculoskeletal modelling, MuscleMimic) with companion files in `tutorials/files/X.Y/`;
  documentation by audience, an environment reference, backend-parity and baselines pages, and a
  developer wiki.
* **Installation.** `uv` installation and CI, Python 3.10–3.14, MuJoCo 3.6.

### Added

* Modular framework: `TaskSpec`/`EnvSpec` registry, `ModelBuilder` (`attach_fragment`, `attach_spec`,
  recipes, `myo_sim` compose pipeline), shared term functions, data-driven `TaskConfig`
  environments (`d84e35c`, `bbd6d94`, `06ad927`).
* MJX backend and env classes for pose, reach, walk and mimic tasks (`57d50bd`, `f5530ca`).
* mjlab backend and tasks: pose and reach twins (`25e5297`), success metric (`f379909`), shared PPO
  defaults (`528aa0e`, `9962a1f`), torso exosuit, leg stand, terrain and rebuilt walk/directional twins
  (`c6372fc`, `1382ef3`, `7c5d31f`, `9765398`, `917bed2`), BoxingP0 baseline env (`9223454`, later
  removed), vectorized Table Tennis contact detection (`7814b83`).
* Challenge suite: Gymnasium P1 env (`c5fd378`), leg-directional and 1v1 ChaseTag (`fd97274`),
  full-body ChaseTag baselines (`b3067db`).
* Evaluation and training tools: `eval_mjlab_policy.py` with CPU/mjlab rollouts and grid videos
  (`8f6bbdf`, `69d35f7`, `57a2273`, `b651821`), `train_mjlab.py` early stop on success (`79b2158`),
  sampling from the learned action std (`83b7ade`), default-policy checkpoints and videos
  (`eb91414`, `89f23fd`, `e764f74`).
* Fatigue: episode-persistent and resumable states (`a4b979c`), torch 3CC-r parity (`99e8812`).
* Arm-reach model edits (thumb frozen, digits under their metacarpals) (`e3e326d`, `f8f9e06`).
* `GoalSpec(target_type="site_positions")` samples per-episode targets from per-site (x, y, z) ranges on
  CPU and MJX, with an mjlab command helper (`site_position_command_cfg`); `MotionClip` carries optional
  per-frame `weights` (issue #410).
* Tutorials: restructured numbered tracks (`5bb6d3a`, `1e61f7e`, `3d1d28c`), SAR tutorials and pretrained
  pickles (`bb5cf7b`), fatigue tutorial for MyoSuite 3 (`d1eda5c`), trained-policy loader (`5bb6d3a`).
* Documentation: quickstarts, environment reference, backend parity, baselines, MJX env list,
  "What's new" block, `CHANGELOG.md`.

### Changed

* `myo_sim` moves from a git submodule to a pip package; all hard-coded `simhive/myo_sim` paths
  became pip-first resolvers; hand, arm and torso tasks use the composed `myo_sim` models
  (`df2dfce`, `f44e0c2`, `fbb121c`, `42c8802`).
* Basic-suite reward terms generalized to run on numpy, JAX and torch (`8231ce1`).
* Documentation and developer wiki cut down and reorganised (`897744f`, `b7c83a9`, `8079643`).
* Tutorials simplified for newcomers and verified with real training runs (`cf03010`, `89faae4`).
* Python support 3.10–3.14 (`38cf140`, `2546095`, `3b025ab`); MuJoCo 3.6.0 (`21edbfc`).
* **Observations are no longer clipped.** CPU envs declared `Box(-10, 10)` and clipped every
  observation to it, which saturated positions and forces in 43 envs (Soccer ball, goal and keeper
  x ≈ 40–50 m; OslRun forward progress; ground reaction forces of Soccer, OslRun and ChaseTag;
  HandReorient muscle forces; Bimanual velocities). Every env now declares a float32
  `Box(-inf, inf)` and observations are only cast to float32, as in legacy MyoSuite and mjlab; the
  mjlab twins dropped their matching ±10 clip. Policies trained on the clipped observations of these
  envs may need retraining.
* **One step contract for every CPU env.** The env classes that override `step()` now end it with
  the shared `MyoGymnasiumEnv._finalize_step`. 61 envs (the Reach, KeyTurn, ObjHold, PenTwirl, Torso
  pose and TableTennis families) returned float64 observations outside their float32 observation
  space and now return float32; every `step()` validates the reward dict and honours
  `mujoco_render_frames`; Bimanual and TableTennis `info` now carries the reward components (e.g.
  `solved`) like every other env.
* **Observations and rewards read the current state.** `step()` followed `mj_step` with
  `mj_kinematics` only, so actuator length/velocity/force, sensors (ground reaction forces),
  contacts, `cvel` and `subtree_com` were one physics substep old in 53 envs (Soccer, ChaseTag,
  OslRun, HandReorient, leg walk and terrain). CPU `step()` and the legacy `forward()` now run
  `mj_forward` (`MyoGymnasiumEnv._step_physics`); the mjlab twins refresh with a full forward before
  rewards and terminations (`mdp.sync_forward`, now also in the directional twins) and no longer
  emulate the stale values. The simulated trajectories are bit-identical except OslRun, whose
  prosthesis controller now reads its current load sensor. Policies trained on the stale
  observations of these envs may need retraining.
* **The `motorFinger*` envs use motors with four times the stock gear** (80/20/20/40/40 instead of
  20/5/5/10/10, a `motor_finger` model recipe shared by the CPU and mjlab envs): policies trained with the
  stock motors stayed at 0% success and now reach 100% on the pose and fixed-reach tasks.
* **The Random finger-reach tasks sample targets the fingertip can reach.** `myoFingerReachRandom-v0` and
  `motorFingerReachRandom-v0` drew targets uniformly in a box of which only about 55% lies in the fingertip's
  workspace (two opposite corners: near the base but high, and far out but low), which capped any policy near
  55%. `ReachEnvV0(target_sampling="workspace")` (used by these ids and their muscle-condition variants, on CPU
  and mjlab) now draws fingertip positions over the joint ranges that lie inside the box. Policies trained on
  the old targets need retraining; the other reach tasks are unchanged.
* **Joint velocities are observed as `qvel * ctrl_dt` on every backend** (the CPU task envs already
  did): the directional-leg twin and the MJX pose and reach envs observed raw `qvel`. The previous
  directional-leg checkpoints were retrained. `ElbowPoseTask` (tutorial 4.3) now also observes the
  `pose_error`, so its observation grows from 8 to 9 values.
* `myoChallengeChaseTagFBP2-v0` is one task on both backends, so mjlab-trained policies run on
  the CPU env. The CPU env now uses the 537-dim `chasetag_obs` layout, a 0.01 s control step
  (2000 steps = 20 s) and flat ground, all on purpose. The mjlab task takes the CPU rewards
  (unscaled by dt), the out-of-bounds lose, the physics options, the keyframe reset and the
  colored-noise opponent. Its distance reward now restarts every episode.
* The CPU `myoMimicBimanual-v0` and `myoMimicFullbody-v0` (random targets) are the CPU half of their
  mjlab twins. They observe `[qpos, qvel * ctrl_dt, act, site position, target, target - position]`
  (199 / 684 values, was 137 / 532 with raw `qvel` and a scalar tracking error). The reward is
  `exp(-2 * mean site error)` on both backends (mjlab used `exp(-20 * error)`, which gives almost no
  signal at the 0.8 m initial error), and actions go through the muscle sigmoid on both without a clip. CPU policies trained on these ids need retraining. The mjlab Mimic rewards and deviation
  check now score the post-step site positions (`mdp.sync_forward`); they read them one physics substep
  stale.

### Fixed

* **TableTennis from any working directory.** The table, net and paddle meshes and textures were added with
  paths relative to the working directory, so making a TableTennis env (CPU or mjlab) with the working
  directory on another drive raised `ValueError: path is on mount ...` on Windows, and an env's spec no
  longer compiled after a `chdir`. They are absolute now; the compiled models are otherwise unchanged.
* RunTrack keyframe joint values clamped to their ranges (#399, `b0514c0`).
* Walk rotation termination uses the root free-joint quaternion (`aa2dd77`); leg model root/torso
  orientation (`531dba0`); hand composition and reorient hand orientation (`53f17a9`, `3f4bf9d`).
* CPU fatigue activation-rate term (`9e237e3`); fatigue and sarcopenia parity between CPU and mjlab
  (`b6d903e`, `99e8812`).
* Per-muscle fatigue parameters are found for mjlab scene actuators (`robot/BIClong`) and
  side-suffixed muscles (`ECRL_r`, `BIClong_l`); both fell back to `Default`, so the fatigue twins
  and the CPU hand pose/reach envs used one F / R / r for every muscle.
* mjlab command API compatibility and isolated per-task registration failures (`658cfdb`); RSI event
  handles `env_ids=None` (#407, `be917b9`).
* Leg twins: a stale root velocity in shared rewards under mjlab, batch-safe heading terms
  (`917bed2`); ChaseTag fall threshold and opponent policy fall-through (`f681cb4`, `33d89ce`).
* `.gitignore` no longer ignores `myosuite/**/tasks` and `scripts/*.py` (`b4f7338`).
* `register_all_envs()` is idempotent (a second call emptied the suite lists); pickling keeps the
  terrain type of hilly/stairs walk envs and `frame_skip` of the CPU MuscleMimic envs.
* Many CI, packaging and notebook fixes (`fd09a79`, `1315c36`, `3ca1ed3`, `50aff25`, `af2d51d` and others).
* mjlab MuscleMimic resets and terminations: RSI writes the clip's root angular velocity in the world
  frame; early termination measures the root error against the clip's reference root, so bimanual
  clips no longer end every episode after one step; clip frames follow the integer episode step
  counter instead of float32 sim time, which lagged the CPU twin on 69% of steps; the SAR tasks
  start standing, expose the `actor`/`critic` groups rsl_rl needs and are the muscle-space task with
  a synergy action term; unsupported `reward_mode`/`env_reward_weight` values raise instead of being
  ignored.
* Follow-ups to the MuscleMimic fixes: random mimic targets are resampled per env when its own episode
  restarts; an episode that plays past the end of its clip is truncated (it used to wrap and could
  end as a termination); a clip whose `frequency` differs from the control rate warns; the bimanual
  lookahead observation and DeepMimic reward no longer treat hinge angles as a root, so bimanual mimic
  checkpoints trained before need retraining; the ONNX/Orbax bridge keeps its observation history and
  running normalizer per env.
* **Mimic clip end and start**: on both backends, the step that truncates at the clip end is scored against
  the clip's last frame. It read the wrapped frame 0, so a non-looping clip scored about 0 on that step. A
  mid-episode checkpoint-playback reset to a clip frame no longer counts as a clip end.
  `MuscleMimicClipEnvV0` takes `random_start` and rejects unknown keyword arguments, so `render_mimic.py`
  (whose `random_start=False` was silently ignored) renders from frame 0, with the ghost on the env's frame.
* `reset(seed=...)` reproduces the episode in the challenge envs (state no longer leaks between
  episodes, random fatigue states, Relocate goals, Soccer goalkeeper and rough tracks draw from the
  env seed); the Bimanual start and goal pillars move to the sampled positions; TableTennis and SAR
  reorient draw their random fatigue state from the env RNG.
* SAR reorient: the action goes through the muscle model, the muscle conditions take effect, episodes
  start palm up, and the geometry-derived fields are refreshed after the per-reset edits (contacts were
  silently culled with MuJoCo >= 3.8).
* The MJX tests and the MJX leg-walk host model work with the pinned `myo-sim`; `eval_mjlab_policy.py`
  reads the `actor` observation group of the twins (falling back to `policy`).
* **mjlab physics options follow the CPU models.** The TableTennis and MuscleMimic mjlab configs ran with
  mjlab's defaults (implicitfast, other iteration counts, `ccd_iterations` 500) instead of the Euler options of
  the CPU model; they now take them from the CPU model, and a guard rejects a timestep that differs from the
  one the control step was derived from.
* **mjlab TableTennis simulates the CPU scene and scores it once per step**: the athlete starts at the
  calibrated pose with the paddle in the hand, stale contact rows are ignored, the terminal bonus/penalty is
  paid once, rewards are per step (not dt-scaled), the P2 randomization hits the ball and the paddle of each
  env (it used to hit the floor), and the ball keeps its 7.2e-7 inertia. On both backends the paddle target
  orientation used the wrong Euler convention (the `paddle_quat` reward never reached its maximum); it now
  matches the keyframe. `TableTennisMixedCtrlAction` is an `ActionTerm`.
* **MuscleMimic bridge and SAR collector.** The bridge mapped a relaxed policy output to half excitation
  (`0.5 * (a + 1)` instead of the `clip(a, 0, 1)` the policies were trained with) and silently skipped
  unmatched names (elbows, left-arm muscles); it now covers 83/83 joints and 354/354 actuators and raises on
  incomplete bridges unless `allow_partial=True`. The SAR activation collector ranks episodes by mean reward
  and resets its state per episode: recollect SAR datasets and re-extract the synergies, and redo evaluations
  made with the old bridge mapping.
* **Mimic mjlab initial state**: the joint-name keys are anchored (`knee_angle_r` no longer also sets
  `knee_angle_rotation{2,3}_*`) and the keyframe's body-frame root angular velocity is converted to the world
  frame.
* **ONNX and SB3.** rsl_rl and Orbax exports are self-contained (no external `.onnx.data`); SB3 exports and
  `OnnxCheckpointCallback` bundles fold `VecNormalize` in, so they take raw observations; `load_policy` and
  `render_sb3_solutions.py` apply the normalization statistics (and handle SAC/TD3). W&B run paths use `/` on
  Windows too.
* **Asset resolver**: resolved/patched model XML copies are written once under content-addressed names
  instead of one new file per `gym.make` call, and the model directory may be read-only. Files leaked by
  earlier versions can be removed with `find myosuite -name '.myosuite_resolved_*.xml' -delete`.
* **Multi-agent envs** (`myoChallengeChaseTagFBVs-v0`) are registered without gymnasium's `TimeLimit`, which
  replaced the per-agent `truncated` dict by a bare `True` on the last step.
* **Experimental MJX backend**: pose targets are matched to joints by name (they were assigned in
  alphabetical order), hand reach tracks each fingertip, every target coordinate has its own random draw, the
  3CC-r fatigue update uses the old state for all deltas, and `FatigueWrapper` keeps the model options. Creating
  an MJX env warns that the backend is experimental and not observation/reward-compatible with the CPU and
  mjlab envs.
* **Tutorial scripts and CI.** The SAR tutorial scripts seed SAC and checkpoint/resume (`--seed`,
  `--play-only`); the 2.3 results depend strongly on the seed. CI runs for PRs into `ms3` and installs the
  `[rl]` extra; the mimic suite no longer comes out empty (it is registered before the challenge suite).

### Removed

* The 2.4 DEP-RL tutorial: the published 2023 baseline no longer walks on the current envs (#468); MyoReflex Walk is now tutorial 2.4.
* The myouser-specific mjlab task and helpers (they live in the standalone myoInteract repository).
* The MyoDM suite (`MyoHand*-v0` hand–object reference-tracking envs, `myosuite_myodm_suite`).
* Unused Boxing meshes (`PunchingBag.obj`, `fencing_helmet.stl`) and the console scripts
  `myosuite-musclemimic-fullbody-parity` and `myosuite-musclemimic-mjx-train` (their modules were deleted).
* Boxing and Saber tasks with their shared code (`de44aca`, `eab9c0c`, `e27dcd0`); the `composer`
  package, the legacy `simhive` copies and `myosuite_init`; placeholder `*Modular-v0` challenge
  registrations (`a453fe4`); the Walk Backends demo notebook (`8e9d51f`); the stale examine-rollout
  script (`f499e9a`) and Colab helpers (`1e5271c`).

### Dependencies

* MuJoCo 3.6.0; `myo_sim` pinned (0.2.3) and taken from PyPI; `huggingface_hub` is a base dependency;
  `wandb`, `orbax-checkpoint`, `jax`/`brax` pins for the mjlab and MJX extras; security bumps of
  `gitpython` and `urllib3` and in `uv.lock`.

* Packaging: SPDX `license = "Apache-2.0"` with `license-files`; MJX benchmark plots are no longer
  shipped in the wheel.
* The `furniture-sim`, `mpl-sim`, `object-sim` and `ycb-sim` git dependencies are gone: the 40 files
  MyoSuite uses (MPL left arm/hand, YCB gelatin box, table texture; 2.9 MB) are bundled under
  `myosuite/envs/myo/assets/`, so every dependency now installs from PyPI.
* **`pink-noise-rl` is replaced by `colorednoise`.** The Soccer goalkeeper and the ChaseTag opponents
  draw their velocities from `myosuite.utils.colored_noise.ColoredNoiseProcess` (pink's buffered process
  on `colorednoise.powerlaw_psd_gaussian`); seeded episodes are bit-identical. `import pink` loaded
  stable-baselines3, torch and TensorBoard whenever they were installed, and registering the challenge
  envs imported every `musclemimic` submodule (and `scipy.spatial`); that package now imports its
  submodules on first use. In a fresh process (Windows, Python 3.12) `import myosuite` takes 0.8 s
  instead of 1.8 s (475 instead of 785 modules), and `gym.make` + `reset` takes 0.35 s instead of 5.1 s
  for `myoChallengeSoccerP1-v0` and 0.14 s instead of 4.4 s for `myoChallengeChaseTagP1-v0`, once in
  every subprocess or vectorized-env worker.

### Contributors

Vittorio Caggiano, Florian Fischer, Balint Hodossy, Vikash Kumar, Tatsuki Tsujimoto, Hyoungseo Son,
Cheryl Wang and Calder Robbins.

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
