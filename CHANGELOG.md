# Changelog

## MyoSuite 3 (branch `ms3`, unreleased) — changes since v2.12.2

Last official release: **v2.12.2** (2026-05-06). This branch has **258 commits** on top of it
(239 without merges): about 1,150 files changed, +124k / −161k lines. The list below
groups the changes by theme; `git log v2.12.2..ms3` has the full commit list.

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

### Fixed

* RunTrack keyframe joint values clamped to their ranges (#399, `b0514c0`).
* Walk rotation termination uses the root free-joint quaternion (`aa2dd77`); leg model root/torso
  orientation (`531dba0`); hand composition and reorient hand orientation (`53f17a9`, `3f4bf9d`).
* CPU fatigue activation-rate term (`9e237e3`); fatigue and sarcopenia parity between CPU and mjlab
  (`b6d903e`, `99e8812`).
* mjlab command API compatibility and isolated per-task registration failures (`658cfdb`); RSI event
  handles `env_ids=None` (#407, `be917b9`).
* Leg twins: a stale root velocity in shared rewards under mjlab, batch-safe heading terms
  (`917bed2`); ChaseTag fall threshold and opponent policy fall-through (`f681cb4`, `33d89ce`).
* `.gitignore` no longer ignores `myosuite/**/tasks` and `scripts/*.py` (`b4f7338`).
* Many CI, packaging and notebook fixes (`fd09a79`, `1315c36`, `3ca1ed3`, `50aff25`, `af2d51d` and others).

### Removed

* Boxing and Saber tasks with their shared code (`de44aca`, `eab9c0c`, `e27dcd0`); the `composer`
  package, the legacy `simhive` copies and `myosuite_init`; placeholder `*Modular-v0` challenge
  registrations (`a453fe4`); the Walk Backends demo notebook (`8e9d51f`); the stale examine-rollout
  script (`f499e9a`) and Colab helpers (`1e5271c`).

### Dependencies

* MuJoCo 3.6.0; `myo_sim` pinned (0.2.3) and taken from PyPI; `huggingface_hub` is a base dependency;
  `wandb`, `orbax-checkpoint`, `jax`/`brax` pins for the mjlab and MJX extras; security bumps of
  `gitpython` and `urllib3` and in `uv.lock`.

### Contributors

Vittorio Caggiano, Florian Fischer, Balint Hodossy, Vikash Kumar, Tatsuki Tsujimoto, Hyoungseo Son,
Cheryl Wang and Calder Robbins.
