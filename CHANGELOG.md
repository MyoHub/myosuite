# Changelog

All notable changes to this project are documented in this file.

## [3.0.0] - unreleased

Changes since the last official release, **v2.12.2** (2026-05-06). `git log v2.12.2..v3.0.0` has the
full commit list.

### Highlights

* **One task, three execution paths, enabling massive GPU parallelization and training speed-ups.** The same `env_id` runs on the **CPU** (Gymnasium: playback, debugging, Stable-Baselines3), on **mjlab** (MuJoCo Warp + RSL-RL: thousands of parallel environments on one GPU) and on an **experimental MJX** (JAX) path. Training on the GPU is the big speed-up of this release: one RTX 5090 steps 20,000-38,000 muscle-driven environments per second (full-body MuscleMimic with 354 muscles, 1024 envs: 19.8k steps/s; the 2-billion-step MuscleMimic run finished in about 30 hours), where one CPU thread steps the much smaller hand, leg and arm tasks at about 1,000 steps/s (measured: 1.4k hand reorient, 1.3k leg walk, 0.8k arm reach). CPU and mjlab share the observation, action and timing contract, so a policy trained on the GPU can be replayed on the CPU.
* **Trained baseline policies.** Ready-to-use policies for 36 environments (pose, reach, leg walking, torso and the MuscleMimic full body) with evaluation videos on Hugging Face (`myohub/myosuite-3-baselines`), downloaded automatically by the tutorials and `eval_mjlab_policy.py` and loadable on both the CPU and mjlab backends. Success rates are listed in `docs/baseline_checkpoints.md`.
* **MuscleMimic support.** Full-body and bimanual MuscleMimic environments
  (`myoMimicFullbody-v0`, `myoMuscleMimicFullbody-v0`, `myoMimicBimanual-v0`,
  `myoMuscleMimicBimanual-v0`, `myoFullBodyDirectional-v0`), loaders for the MuscleMimic
  checkpoints and retargeted motion datasets from Hugging Face, the `myosuite-musclemimic-fullbody-eval`
  tool, and tutorials 5.1–5.5 (load a policy, train on CPU/JAX and with mjlab, directional
  locomotion, SAR).
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
  policies with evaluation videos on Hugging Face (`myohub/myosuite-3-baselines`).
* **`myo_sim` as a pip package** (replacing the git submodule); models are composed from it and the
  hand, arm and leg orientations were re-calibrated.
* **Muscle conditions.** Muscle-group- and sex-specific fatigue parameters from the literature, a torch
  fatigue model with CPU parity, episode-persistent and resumable fatigue states, spec-level sarcopenia
  and reafferentation.
* **The full MyoChallenge suite as Gymnasium environments** (21 ids): Baoding, Bimanual, ChaseTag
  (P1, P2, P2eval, full-body P2, 1v1 full-body), Die Reorient, OSL run, Relocate, Soccer and Table
  Tennis, with mjlab versions of the ChaseTag full-body and Table Tennis tasks.
* **Tutorials and documentation.** 20 notebooks in five numbered tracks (basics, training,
  analysis, musculoskeletal modelling, MuscleMimic) with companion files in `tutorials/files/X.Y/`;
  documentation by audience, an environment reference, backend-parity and baselines pages, and a
  developer wiki.
* **Installation.** `uv` installation and CI, Python 3.10–3.14, MuJoCo 3.7 or newer (no official floor, but older versions are not maintained).

### Added

* **Modular framework:** `TaskSpec`/`EnvSpec` registry, `ModelBuilder` (`attach_fragment`, `attach_spec`, recipes, `myo_sim` compose pipeline), shared term functions and data-driven `TaskConfig` environments ([#406]).
* **MJX backend** (experimental) with env classes for pose, reach, walk and mimic tasks ([#354]).
* **mjlab backend and tasks:** pose, reach, torso exosuit, leg (stand, terrain, walk, directional) and Table Tennis twins; `Episode_Metrics/success` on every twin; shared PPO defaults; vectorized Table Tennis contact detection ([#406]).
* **MyoChallenge suite** as Gymnasium envs: P1 envs, leg-directional and 1v1 ChaseTag, full-body ChaseTag baselines ([#406], [#101]).
* **MuscleMimic:** full-body and bimanual envs, checkpoint and motion loaders, tutorials 5.1-5.5; multi-clip training (`register_mimic_mjlab_tasks_with_clip` takes several clips); reusable full-body viewer camera `mimic_viewer_cfg` ([#406], [#436], [#475]).
* **Training and evaluation tools:** `eval_mjlab_policy.py` (CPU/mjlab rollouts, grid videos, `--stochastic`), `train_mjlab.py` early stop on success, default-policy checkpoints and videos on Hugging Face (`myohub/myosuite-3-baselines`) ([#406], [#478]).
* **SAR synergies re-extracted on the current envs** and installed in `tutorials/files/2.3/SAR_pretrained/`; SAR-RL and RL-E2E reach the same success on locomotion, and SAR is not consistently ahead on manipulation (2 of 6 seeds).
* **MuscleMimic baseline:** a single-clip walking policy (2B steps; 86% of episodes reach the clip end) on Hugging Face, see `docs/baseline_checkpoints.md`.
* **Fatigue:** muscle-group- and sex-specific fatigue parameters from the literature (Rakshit et al. 2021; Frey-Law et al. 2012), episode-persistent and resumable states, torch 3CC-r parity with the CPU model ([#406], [#421], [#491]).
* **Targets:** `GoalSpec(target_type="site_positions")` samples per-episode targets from per-site ranges on CPU and MJX (mjlab helper `site_position_command_cfg`); `MotionClip` carries optional per-frame `weights` (#410).
* **Muscle-command wrappers:** `MotorNoiseWrapper` (signal-dependent + constant Gaussian noise on muscle excitations, `MotorNoiseCfg.van_beers_2004()`), `FatigueWrapper`, `ReafferentationWrapper` and `SarcopeniaWrapper`; each installs a stage of the env's action pipeline, run by priority (`map -> noise -> fatigue -> reroute`), also on the mjlab twins; `ExcitationStageWrapper` adds portable custom stages (CPU env and mjlab twin) and `CtrlStageWrapper` env-aware CPU ones; they run after the built-in stages in installation order, or at an explicit `order` to insert one earlier (two equal explicit orders raise a `StageOrderWarning`) ([#488], [#502]).
* **`make_env(EnvConfig(...))`:** one call builds an env on the CPU or on the mjlab twin with the same `features` (muscle-command wrappers, built with `wrapper_spec`), `max_episode_steps` and `num_envs`; the old `EnvConfig` fields (`model`, `scene`, `backend` timing) and `TaskConfig.to_env_config` are removed, as was the unused `config=` argument of `register_env`. `ctrl_dt` is the single timing knob (substeps and decimation follow from it). `TaskConfig` no longer carries features: `muscle_fatigue`, `ActuatorGroupSpec.condition` / `.noise` and `TaskConfig.fatigue_enabled` are removed (use the wrappers; `VariantSpec(features=...)` registers them; `ModularTaskEnv` runs the stages, so the Sarc/Fati TaskConfig ids behave as before, but its additive action noise is replaced by `MotorNoiseWrapper`). A plain `make_env(env_id, backend=..., **kwargs)` works as before.
* **Tutorials and docs:** numbered tracks with companion files in `tutorials/files/X.Y/`, SAR tutorials and pretrained pickles, a fatigue tutorial, quickstarts, environment reference, backend-parity and baselines pages ([#406], [#454], [#455]).

### Changed

* `myo_sim` is a pip package (was a git submodule); hand, arm and torso tasks use the composed models; pinned to 0.2.3 ([#406], [#408]).
* **Thumb CMC joints in `myo_sim` >= 0.2.0:** the two thumb joints exchange name and range (the first CMC joint is `cmc_flexion` with range -0.78 to 0.7, it was `cmc_abduction` with -0.5 to 0.78). This affects the composed hand of the hand pose, reach and reorient tasks and Relocate; `myoHandKeyTurn*`, `myoHandObjHold*` and `myoHandPenTwirl*` keep the legacy order. The published hand baselines were trained and re-evaluated on the new order.
* Basic-suite reward terms run on numpy, JAX and torch ([#406]); Python 3.10-3.14, MuJoCo 3.7 or newer ([#406]).
* **Observations are no longer clipped:** every env declares a float32 `Box(-inf, inf)` (43 envs saturated at ±10). One step contract for every CPU env: float32 observations, validated reward dict, `info` carries the reward components ([#444]).
* **Observations and rewards read the current state:** `step()` runs `mj_forward`, and the mjlab twins refresh with a full forward before rewards and terminations; 53 envs had observed one-substep-old values. Trajectories are bit-identical except OslRun ([#444]).
* **Joint velocities are observed as `qvel * ctrl_dt` on every backend**; the directional-leg twin declares the real 5 ms step ([#437], [#449]). `ElbowPoseTask` also observes the pose error (8 to 9 values) ([#433]).
* **Leg walk actions are `[0, 1]` activations.** `myoLegWalk-v0` and its variants (CPU and mjlab) clip the action to `[0, 1]` and use it as the muscle control; in 2.x the action was in `[-1, 1]` and went through a sigmoid (0 gave 7.6%, +1 gave 92.4%). 2.x walking policies therefore have their negative outputs clipped to 0 and need retraining or a mapping.
* **`motorFinger*` envs use motors with four times the stock gear**, so the pose and fixed-reach tasks are learnable ([#451]).
* **Random finger-reach tasks sample targets the fingertip can reach** (`ReachEnvV0(target_sampling="workspace")`), instead of a box of which only about 55% lies in the workspace ([#452]).
* **`myoArmReachRandom-v0` and `myoHandReachRandom-v0` no longer start episodes beyond `far_th`:** 16.5% (arm) and 97% (hand) of the resets used to end at step 2. `far_th` is now 1.3 (arm) and 0.075 (hand), also for the variants, the mjlab twins and MJX. The published arm policy rises from 67.0% to 76.5% deterministic success; the hand policy is unaffected ([#494], [#496], [#497]).
* **Unknown `obs_keys` raise** a `KeyError` listing the available keys (CPU envs and mjlab pose/reach/stand/walk twins; they were silently dropped), and SAR reorient honours `obs_keys`. The Relocate ids observe every hand joint (`hand_qpos` missed `md5_flexion_r`; P1 156 to 157 values), and `hand_qpos_corrected` is available again ([#477]).
* **`myoChallengeChaseTagFBP2-v0` is one task on both backends** (537-d `chasetag_obs`, 0.01 s control step, CPU rewards on mjlab) ([#456]).
* **CPU `myoMimicBimanual-v0` and `myoMimicFullbody-v0` are the CPU half of their mjlab twins:** observation `[qpos, qvel * ctrl_dt, act, site position, target, target - position]` (199 / 684 values), `exp(-2 * mean site error)` reward on both backends. CPU policies trained on these ids need retraining ([#475]).
* **mjlab physics options follow the CPU models** for TableTennis and MuscleMimic ([#457]); **mjlab TableTennis** simulates the CPU scene with once-per-step scoring and per-env randomization ([#458]).
* **MuscleMimic bridge builds its observation on the sim device** (`TorchFullbodyObsAdapter`): 20-80x faster, float32, not bit-identical to the CPU builder (agrees to about 1e-6); `obs_backend="cpu"` keeps the old path ([#486]).
* **Performance and memory:** cached model specs ([#481]), vectorized full-body mimic observation builder ([#482]), faster OslRun step ([#483]), explicit MjData arenas for full-body Mimic and ChaseTag ([#487], [#493]) and the two-agent `myoChallengeChaseTagFBVs-v0` scene ([#498]), `colorednoise` replaces `pink-noise-rl` and `import myosuite` is 2x faster ([#480]).
* **Fatigue dynamics follow the literature:** the rest multiplier `r` acts only at rest (Rakshit et al. 2021; commands up to `FATIGUE_REST_THRESHOLD = 0.01` count as rest, so sigmoid-mapped muscles can rest) and the `Shoulder` row uses the Frey-Law et al. (2012) fit. Retrain policies on the `myoFati*` envs ([#491]).
* **The `muscle_condition`, `fatigue_reset_vec`, `fatigue_reset_random` and `motor_noise` env kwargs are replaced by wrappers.** The `myoFati*`, `myoSarc*` and `myoReaf*` ids are unchanged, and rollouts are identical to the kwarg versions. Build a custom stack with `FatigueWrapper(env, fatigue_reset_random=True)` and the other wrappers, and use `env.set_fatigue_reset_random(...)` instead of `env.unwrapped.set_fatigue_reset_random(...)`; the old kwargs raise a `TypeError` that names the replacement ([#502]).
* **Randomized die, Baoding P2 and weighted elbow resets refresh collision bounds and inertia** after editing geom sizes and masses (MuJoCo 3.8 and newer cull contacts against stale bounds); the dynamics of these envs differ from earlier builds, so re-evaluate policies and recollect datasets ([#499] review).
* **mjlab walk, Mimic and Table Tennis steps make no host syncs** (GPU stream stalls) in MyoSuite terms, with bit-identical results: walk / terrain twins 34 / 41 per step to 0; Mimic clip mode 33, a 2-clip bank 537 (full body) and random targets 17 to 0 (one reset check on a step that resets an env); Table Tennis P1 / P2 2 / 4 per reset step to 0. `mimic_composite_reward` reports `mean_site_dist` and `solved` per env ([#500]).
* Documentation and developer wiki cut down; tutorials simplified for newcomers ([#406]).

### Fixed

* **Reach and arm models:** arm-reach `IFtip` site back at the fingertip (it sat 1.8 cm short at the DIP joint since `7532d62`; the published arm policy reaches with the fingertip in 64% of episodes without retraining) ([#406]); thumb frozen and digits kept under their metacarpals ([#406]).
* **Muscle conditions:** CPU fatigue activation rate and 3CC-r overshoot ([#421]); per-muscle fatigue parameters for prefixed and side-suffixed names ([#430]); automatic peak force under sarcopenia ([#426]); fatigue/sarcopenia parity between CPU and mjlab ([#406]).
* **Fatigue:** the mjlab twins honour `fatigue_reset_vec` / `fatigue_reset_random`, and `myoFatiElbowPoseTask{Fixed,Random}-v0` now fatigue ([#491]).
* **Challenge envs:** `reset(seed=...)` reproduces the episode ([#438]); Bimanual pillars at the sampled positions ([#441]); TableTennis and SAR reorient fatigue draws from the env RNG ([#442]); TableTennis termination, relaunch and conditions ([#432]), policy action in mjlab ([#428]) and mesh paths on any drive ([#485]); OSL controller `is_running` and `set_motor_param` ([#484]); RunTrack keyframe clamping ([#399]); ChaseTag fall threshold and opponent fall-through ([#406]).
* **Terminate on MuJoCo instability** in every CPU env ([#423]).
* **SAR:** reorient actions, muscle conditions and stale geometry ([#434]); mjlab SAR action reaches the muscles ([#424]); PCA whitening undone in `SARTorchTransform` ([#425]); the bridge and activation collector ([#459]); tutorial scripts seed SAC and resume ([#455]).
* **MuscleMimic mjlab:** resets, terminations, frame index and SAR task configs ([#436]); follow-ups: per-env target resampling, clip-rate check ([#437]); bimanual lookahead without root terms ([#439]); per-env bridge history and normalizer ([#440]); clip end scored against the last frame, `random_start` honoured ([#475]); bridge excitation mapping and name coverage ([#459]); relative angular velocity in `TorchFullbodyObsAdapter` ([#459]); multi-clip resets of partial envs ([#406]); initial-state joint-name keys and root angular velocity ([#406]).
* **ONNX and SB3 exports** are self-contained and fold `VecNormalize` in ([#427], [#461]); two ONNX-export defects of the mimic MDP ([#407]); `find_checkpoint` skips runs of other envs ([#420]); `eval_mjlab_policy.py` reads the `actor` group ([#445]); `train_mjlab` resume and stop-on-success ([#478]).
* **TaskConfig control step:** a control step is `n_substeps` steps of `sim_dt` on every backend, `BackendConfig` rejects a different `ctrl_dt`, and the CPU `ModularTaskEnv` sets the timestep to `sim_dt` (it scaled `joint_vel` by a `ctrl_dt` it did not simulate); the reach workspace table uses the scene's site ids ([#476]).
* **Challenge scoring and targets:** RunTrack and ChaseTag `get_metrics` score a lost episode with the full `maxTime` again, and Baoding, Relocate, Reorient, Bimanual and TableTennis have a legacy-style `get_metrics`; `PoseEnvV0.update_target()` moves the rewarded target with the observed one; Bimanual refreshes the box inertia and rescales the visual box ([#477]).
* **Wrapped envs forward public attributes** again (`env.mj_render()` after `gym.make`) (#378); the multi-agent env is registered without `TimeLimit` ([#463]); resolved model XML written once ([#462]; files leaked by older versions can be removed with `find myosuite -name '.myosuite_resolved_*.xml' -delete`); `register_all_envs()` is idempotent ([#406]).
* **ReferenceMotion:** interpolation, ghost-body rendering, `examine_policy` records ([#471]); no error when time moves backwards after the last frame ([#471]).
* **ModelBuilder options and rebuilds, full-width clips mapped by joint name, fatigue API** ([#473]); quaternion velocity wrap and wrappers ([#474]).
* **Experimental MJX:** silent corruption and limits ([#465]); stale tests and leg-walk host model ([#443]); reach far check from control step 2 and finger reach reading `far_th` and workspace sampling from CPU ([#497]).
* mjlab command API compatibility and RSI event with `env_ids=None` ([#407]); leg twins' stale root velocity and heading terms ([#406]); walk rotation termination and leg model orientation ([#406]); `.gitignore` no longer hides `myosuite/**/tasks` ([#406]).
* CI, packaging, release workflow and notebook fixes ([#447], [#467], [#478], [#479]).

### Removed

* `ObservationNormalizeWrapper` (unused; SB3 `VecNormalize` and mjlab's normalizer cover it) ([#502]).
* The 2.4 DEP-RL tutorial (the 2023 baseline no longer walks on the current envs); MyoReflex Walk is now 2.4 ([#460], [#468]).
* The myouser-specific mjlab task and helpers (now in the standalone myoInteract repository).
* The MyoDM suite, Boxing and Saber tasks with their shared code, the `composer` package, the legacy `simhive` copies, `myosuite_init`, placeholder `*Modular-v0` registrations, the Walk Backends demo notebook, the stale examine-rollout script, Colab helpers and the unused console scripts `myosuite-musclemimic-fullbody-parity` and `myosuite-musclemimic-mjx-train` ([#406]).

### Dependencies

* MuJoCo 3.7 or newer (no official floor, but older versions are not maintained); `myo_sim` 0.2.3 from PyPI; `huggingface_hub` is a base dependency; `wandb`, `orbax-checkpoint`, `jax`/`brax` pins for the mjlab and MJX extras; security bumps of `gitpython`, `urllib3` and `uv.lock`; SPDX license metadata.
* The `furniture-sim`, `mpl-sim`, `object-sim` and `ycb-sim` git dependencies are gone: the 40 files MyoSuite uses (2.9 MB) are bundled under `myosuite/envs/myo/assets/`, so every dependency installs from PyPI.
* `pink-noise-rl` is replaced by `colorednoise` (`myosuite.utils.colored_noise.ColoredNoiseProcess`; seeded episodes are bit-identical) ([#480]).

### Contributors

Vittorio Caggiano, Florian Fischer, Balint Hodossy, Vikash Kumar, Tatsuki Tsujimoto, Hyoungseo Son,
Cheryl Wang, Mark Colley and Calder Robbins.
A big thanks to all MyoSuite 1.0 and 2.0 contributors, whose work this release essentially builds on!

[#101]: https://github.com/MyoHub/myosuite/pull/101
[#354]: https://github.com/MyoHub/myosuite/pull/354
[#399]: https://github.com/MyoHub/myosuite/pull/399
[#406]: https://github.com/MyoHub/myosuite/pull/406
[#407]: https://github.com/MyoHub/myosuite/pull/407
[#408]: https://github.com/MyoHub/myosuite/pull/408
[#420]: https://github.com/MyoHub/myosuite/pull/420
[#421]: https://github.com/MyoHub/myosuite/pull/421
[#423]: https://github.com/MyoHub/myosuite/pull/423
[#424]: https://github.com/MyoHub/myosuite/pull/424
[#425]: https://github.com/MyoHub/myosuite/pull/425
[#426]: https://github.com/MyoHub/myosuite/pull/426
[#427]: https://github.com/MyoHub/myosuite/pull/427
[#428]: https://github.com/MyoHub/myosuite/pull/428
[#430]: https://github.com/MyoHub/myosuite/pull/430
[#432]: https://github.com/MyoHub/myosuite/pull/432
[#433]: https://github.com/MyoHub/myosuite/pull/433
[#434]: https://github.com/MyoHub/myosuite/pull/434
[#436]: https://github.com/MyoHub/myosuite/pull/436
[#437]: https://github.com/MyoHub/myosuite/pull/437
[#438]: https://github.com/MyoHub/myosuite/pull/438
[#439]: https://github.com/MyoHub/myosuite/pull/439
[#440]: https://github.com/MyoHub/myosuite/pull/440
[#441]: https://github.com/MyoHub/myosuite/pull/441
[#442]: https://github.com/MyoHub/myosuite/pull/442
[#443]: https://github.com/MyoHub/myosuite/pull/443
[#444]: https://github.com/MyoHub/myosuite/pull/444
[#445]: https://github.com/MyoHub/myosuite/pull/445
[#447]: https://github.com/MyoHub/myosuite/pull/447
[#449]: https://github.com/MyoHub/myosuite/pull/449
[#451]: https://github.com/MyoHub/myosuite/pull/451
[#452]: https://github.com/MyoHub/myosuite/pull/452
[#454]: https://github.com/MyoHub/myosuite/pull/454
[#455]: https://github.com/MyoHub/myosuite/pull/455
[#456]: https://github.com/MyoHub/myosuite/pull/456
[#457]: https://github.com/MyoHub/myosuite/pull/457
[#458]: https://github.com/MyoHub/myosuite/pull/458
[#459]: https://github.com/MyoHub/myosuite/pull/459
[#460]: https://github.com/MyoHub/myosuite/pull/460
[#461]: https://github.com/MyoHub/myosuite/pull/461
[#462]: https://github.com/MyoHub/myosuite/pull/462
[#463]: https://github.com/MyoHub/myosuite/pull/463
[#465]: https://github.com/MyoHub/myosuite/pull/465
[#467]: https://github.com/MyoHub/myosuite/pull/467
[#468]: https://github.com/MyoHub/myosuite/pull/468
[#471]: https://github.com/MyoHub/myosuite/pull/471
[#473]: https://github.com/MyoHub/myosuite/pull/473
[#474]: https://github.com/MyoHub/myosuite/pull/474
[#475]: https://github.com/MyoHub/myosuite/pull/475
[#476]: https://github.com/MyoHub/myosuite/pull/476
[#477]: https://github.com/MyoHub/myosuite/pull/477
[#478]: https://github.com/MyoHub/myosuite/pull/478
[#479]: https://github.com/MyoHub/myosuite/pull/479
[#480]: https://github.com/MyoHub/myosuite/pull/480
[#481]: https://github.com/MyoHub/myosuite/pull/481
[#482]: https://github.com/MyoHub/myosuite/pull/482
[#483]: https://github.com/MyoHub/myosuite/pull/483
[#484]: https://github.com/MyoHub/myosuite/pull/484
[#485]: https://github.com/MyoHub/myosuite/pull/485
[#486]: https://github.com/MyoHub/myosuite/pull/486
[#487]: https://github.com/MyoHub/myosuite/pull/487
[#488]: https://github.com/MyoHub/myosuite/pull/488
[#491]: https://github.com/MyoHub/myosuite/pull/491
[#493]: https://github.com/MyoHub/myosuite/pull/493
[#494]: https://github.com/MyoHub/myosuite/pull/494
[#496]: https://github.com/MyoHub/myosuite/pull/496
[#497]: https://github.com/MyoHub/myosuite/pull/497
[#498]: https://github.com/MyoHub/myosuite/pull/498
[#499]: https://github.com/MyoHub/myosuite/pull/499
[#500]: https://github.com/MyoHub/myosuite/pull/500
[#502]: https://github.com/MyoHub/myosuite/issues/502

## [2.12.2] - 2026-05-06
* Asset credits updated ([#392]).

## [2.12.1] - 2026-04-23
* `make_data(naccdmax=...)` enabled; MuJoCo 3.6.0 ([#390]).

## [2.12.0] - 2026-04-23
* `uv` installation, CI and MJX environment updates ([#387]); PyPI release CI Python version fixed ([#389]).

## [2.11.6] - 2025-11-04
* Soccer P2: goalkeeper position added to the observations ([#360]).

## [2.11.5] - 2025-10-01
* Inverse-dynamics tutorial fixed ([#349]).

## [2.11.4] - 2025-09-26
* Table Tennis P2 hotfix ([#345]).

## [2.11.3] - 2025-09-23
* MyoChallenge 2025: Table Tennis P2 ([#338]), Soccer P2 ([#339]) and the Phase 2 tasks ([#340]); CI fixes ([#341], [#342], [#343]).

## [2.10.0 - 2.10.3] - 2025-08
* 2.10.0 (08-11): license update ([#322]), HTML utils ([#321]), `myoArmReachRandom-v0` ([#232]), MyoChallenge 2025 updates ([#325], [#330]: Soccer randomizations and metrics), `version.py` synced with PyPI ([#331]).
* 2.10.1 - 2.10.3 (08-19): automated CI version bumps only ([#333], [#335], [#336]).

## [2.9.0] - 2025-07-11
* Tutorials: inverse kinematics ([#298]), OpenSim `.mot` playback on the MyoSkeleton ([#302]), computed-muscle-control elbow ([#301]); minimum Python 3.9 ([#303]).
* MyoChallenge 2025: Soccer ([#309]), Table Tennis base env ([#308]), MC25 tasks ([#314], [#316]); loco head-site fixes ([#310], [#312]); `import myosuite` before SB3 so envs are registered ([#270]); PyPI release CI ([#315]).
* v2.8.6 is the same commit as 2.9.0.

## [2.8.0 - 2.8.4] - 2024-09 to 2024-10
* 2.8.0 (09-23): MyoChallenge 2024 Phase 2 ([#229]) and eval phase ([#223]), locomotion metrics ([#214]) and action-space fix ([#230]), Run Track P2 ([#226]), bimanual variations ([#216]), randomization and OSL changes ([#227]), reflex tutorial fix ([#233]).
* 2.8.1 (09-27): locomotion observation fixes ([#247], [#248]). 2.8.2 (10-02): hfield observation fixes ([#251], [#252]). 2.8.3 (10-28): object-target proximity threshold ([#258], [#259]), TensorBoard directory for W&B ([#256]). 2.8.4 (10-31): manipulation env fixes ([#262], [#265]).

## [2.7.0] - 2024-09-01
* MyoSkeleton model ([#217]); MyoChallenge manipulation-track metric ([#213]); `obsvec` matches the observation space ([#205]).

## [2.5.0] - 2024-07-28
* MyoChallenge 2024 tasks ([#192]); documentation update ([#190]).

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

[#190]: https://github.com/MyoHub/myosuite/pull/190
[#192]: https://github.com/MyoHub/myosuite/pull/192
[#205]: https://github.com/MyoHub/myosuite/pull/205
[#213]: https://github.com/MyoHub/myosuite/pull/213
[#214]: https://github.com/MyoHub/myosuite/pull/214
[#216]: https://github.com/MyoHub/myosuite/pull/216
[#217]: https://github.com/MyoHub/myosuite/pull/217
[#223]: https://github.com/MyoHub/myosuite/pull/223
[#226]: https://github.com/MyoHub/myosuite/pull/226
[#227]: https://github.com/MyoHub/myosuite/pull/227
[#229]: https://github.com/MyoHub/myosuite/pull/229
[#230]: https://github.com/MyoHub/myosuite/pull/230
[#232]: https://github.com/MyoHub/myosuite/pull/232
[#233]: https://github.com/MyoHub/myosuite/pull/233
[#247]: https://github.com/MyoHub/myosuite/pull/247
[#248]: https://github.com/MyoHub/myosuite/pull/248
[#251]: https://github.com/MyoHub/myosuite/pull/251
[#252]: https://github.com/MyoHub/myosuite/pull/252
[#256]: https://github.com/MyoHub/myosuite/pull/256
[#258]: https://github.com/MyoHub/myosuite/pull/258
[#259]: https://github.com/MyoHub/myosuite/pull/259
[#262]: https://github.com/MyoHub/myosuite/pull/262
[#265]: https://github.com/MyoHub/myosuite/pull/265
[#270]: https://github.com/MyoHub/myosuite/pull/270
[#298]: https://github.com/MyoHub/myosuite/pull/298
[#301]: https://github.com/MyoHub/myosuite/pull/301
[#302]: https://github.com/MyoHub/myosuite/pull/302
[#303]: https://github.com/MyoHub/myosuite/pull/303
[#308]: https://github.com/MyoHub/myosuite/pull/308
[#309]: https://github.com/MyoHub/myosuite/pull/309
[#310]: https://github.com/MyoHub/myosuite/pull/310
[#312]: https://github.com/MyoHub/myosuite/pull/312
[#314]: https://github.com/MyoHub/myosuite/pull/314
[#315]: https://github.com/MyoHub/myosuite/pull/315
[#316]: https://github.com/MyoHub/myosuite/pull/316
[#321]: https://github.com/MyoHub/myosuite/pull/321
[#322]: https://github.com/MyoHub/myosuite/pull/322
[#325]: https://github.com/MyoHub/myosuite/pull/325
[#330]: https://github.com/MyoHub/myosuite/pull/330
[#331]: https://github.com/MyoHub/myosuite/pull/331
[#333]: https://github.com/MyoHub/myosuite/pull/333
[#335]: https://github.com/MyoHub/myosuite/pull/335
[#336]: https://github.com/MyoHub/myosuite/pull/336
[#338]: https://github.com/MyoHub/myosuite/pull/338
[#339]: https://github.com/MyoHub/myosuite/pull/339
[#340]: https://github.com/MyoHub/myosuite/pull/340
[#341]: https://github.com/MyoHub/myosuite/pull/341
[#342]: https://github.com/MyoHub/myosuite/pull/342
[#343]: https://github.com/MyoHub/myosuite/pull/343
[#345]: https://github.com/MyoHub/myosuite/pull/345
[#349]: https://github.com/MyoHub/myosuite/pull/349
[#360]: https://github.com/MyoHub/myosuite/pull/360
[#387]: https://github.com/MyoHub/myosuite/pull/387
[#389]: https://github.com/MyoHub/myosuite/pull/389
[#390]: https://github.com/MyoHub/myosuite/pull/390
[#392]: https://github.com/MyoHub/myosuite/pull/392
