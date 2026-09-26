# Changelog

## MyoSuite 3 (branch `ms3`, unreleased) — changes since v2.12.2

Last official release: **v2.12.2** (2026-05-06). This branch has **258 commits** on top of it
(239 without merges): about 1,150 files changed, +124k / −161k lines. The list below
groups the changes by theme; the full commit list is at the end.

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
Cheryl Wang, Calder Robbins and dependabot.

### Full commit list (newest first, merges omitted)

* `47addf7` docs: leg locomotion twins are step-parity tested; update the leg baselines note — Florian Fischer (2026-09-26)
* `917bed2` mjlab: rebuild the myoLegDirectional twins from the CPU TaskConfig for step parity — Florian Fischer (2026-09-26)
* `9765398` mjlab: rebuild the leg terrain twins on the walk twin (knee done, baked terrain) — Florian Fischer (2026-09-26)
* `7c5d31f` mjlab: rebuild myoLegWalk (+Sarc/Fati) from the CPU env for step parity — Florian Fischer (2026-09-26)
* `e764f74` baselines: add myoTorsoExoPoseFixed-v0 (100%) and myoLegStandRandom-v0 (36%, plateaued) — Florian Fischer (2026-09-26)
* `7c8a8fe` myosuite 3 logo updated — Florian Fischer (2026-09-26)
* `ab47b58` docs: add a 'What's new in MyoSuite 3' block (README and docs index) — Florian Fischer (2026-09-26)
* `35d4155` docs: list the MJX-supported env ids — Florian Fischer (2026-09-26)
* `33f0798` README: load the logo from the repository (relative path) — Florian Fischer (2026-09-26)
* `11b163b` README: MyoSuite 3 logo (add the image file) — Florian Fischer (2026-09-26)
* `b6eb965` 2.3: fix rollouts on wrapped envs and show the videos; clean stored outputs — Florian Fischer (2026-09-26)
* `3f7598d` 2.3 helpers: pick an existing camera in get_vid; keep plot_results usable for short logs — Florian Fischer (2026-09-26)
* `2e42afe` tests: read the walk twin's 'actor' observation group — Florian Fischer (2026-09-26)
* `b44940b` 2.5: write the video at the rollout rate (fps = 1/dt = 100) — Florian Fischer (2026-09-26)
* `a702488` 2.2 fixed/overridden — Florian Fischer (2026-09-26)
* `bdc3528` baselines: README notes on convergence and the measurement backend — Florian Fischer (2026-09-26)
* `d0cc567` docs: success metric, evaluation, default policies, mjlab coverage and parity status — Florian Fischer (2026-09-26)
* `5e45292` RslRlPolicy: clear error when the observation size does not fit — Florian Fischer (2026-09-26)
* `fe2ad8c` mjlab: torso exo twins give the CPU observation layout (experimental) — Florian Fischer (2026-09-26)
* `89f23fd` baselines: eval videos of the default-run policies — Florian Fischer (2026-09-26)
* `eb91414` baselines: default-run mjlab policies (32 envs) — Florian Fischer (2026-09-26)
* `a6d0d9f` eval_mjlab_policy: video grids, success rates, pose/reach/velocity markers — Florian Fischer (2026-09-26)
* `628c9b3` find_checkpoint: fall back to baselines/checkpoints/<env_id> — Florian Fischer (2026-09-26)
* `79b2158` train_mjlab: stop early once the success criterion is met — Florian Fischer (2026-09-26)
* `83b7ade` RslRlPolicy: sample actions with the learned std — Florian Fischer (2026-09-26)
* `3b641ec` mjlab: legacy leg twins get a success metric, shared PPO defaults and a Fati walk — Florian Fischer (2026-09-26)
* `1382ef3` mjlab: myoLegStandRandom and terrain walk twins (rough, hilly, stairs) — Florian Fischer (2026-09-26)
* `c6372fc` mjlab: torso exosuit twins (myoTorsoExoPoseFixed and Sarc/Fati variants) — Florian Fischer (2026-09-26)
* `43c60f6` Leg tasks: leg-reach reward and one solved criterion for locomotion — Florian Fischer (2026-09-26)
* `9945724` fix tutorial 5.1 (allow it to run headless) — Florian Fischer (2026-09-26)
* `beb8abb` 2.5: describe the reflex rollout accurately + additional misc. wording changes. (#414) — Calder Robbins (2026-09-25)
* `0ab55d5` 4.2: tqdm progress bar, best-checkpoint training, deterministic evaluation — Florian Fischer (2026-09-24)
* `f18cc2f` 3.2: install mink and quadprog on demand so the notebook runs in CI — Florian Fischer (2026-09-24)
* `013df16` Tutorials: 5.5 remove local paths from outputs; 3.4 fix CMC comparison plots — Florian Fischer (2026-09-24)
* `d8eff7b` Fix pre-commit CI (format, F811, EOF newline); 5.5: force JAX on CPU, clear outputs, update intro — Florian Fischer (2026-09-24)
* `3d1d28c` Tutorials: rename 5.x notebooks, unify headers, add HF licence notes, reset kernel names — Florian Fischer (2026-09-24)
* `f7f99c4` 2.1 tutorial fixed — Florian Fischer (2026-09-24)
* `2e39680` Fix arm-reach camera test after default camera framing; skip remaining tutorials in CI — Florian Fischer (2026-09-24)
* `1e61f7e` Move tutorial companion files to tutorials/files/X.Y; renumber notebook headers; fix mc25 model path — Florian Fischer (2026-09-24)
* `5bb6d3a` Restructure tutorials into numbered tracks; add trained-policy loader and arm-reach camera — Florian Fischer (2026-09-24)
* `7532d62` Make arm reach editor fn more flexible and specific — Balint-H (2026-09-24)
* `11cee91` added num-envs to sample calls for mjlab training — Florian Fischer (2026-09-24)
* `0096b71` Tune eval video fog distances — Florian Fischer (2026-09-24)
* `528aa0e` Share myoInteract-derived PPO defaults across pose/reach mjlab tasks; add --reward-scale — Florian Fischer (2026-09-24)
* `986a5ee` fix(eval): fog that works at any model extent; renderer settings baked in — Florian Fischer (2026-09-24)
* `e3e326d` feat(myoedits): freeze the thumb in the arm-reach model — Florian Fischer (2026-09-24)
* `9962a1f` feat(mjlab): default PPO params that avoid a runaway-stochastic policy — Florian Fischer (2026-09-24)
* `b651821` feat(eval): CPU parity for grids, success rate, floor and camera sweep — Florian Fischer (2026-09-24)
* `f379909` feat(mjlab): log Episode_Metrics/success for every pose and reach twin — Florian Fischer (2026-09-24)
* `f8f9e06` fix(myoedits): keep rebuilt arm-reach digits under their own metacarpals — Florian Fischer (2026-09-24)
* `58ba4bc` feat(eval): per-env episode count and skeleton-only rendering — Florian Fischer (2026-09-24)
* `57a2273` feat(eval): set the mjlab video grid with --num-cols/--num-rows — Florian Fischer (2026-09-24)
* `69d35f7` feat(eval): render all parallel mjlab envs side by side in one video — Florian Fischer (2026-09-24)
* `b5bea12` style: apply ruff-format / pyupgrade to the new mjlab and eval code — Florian Fischer (2026-09-24)
* `88f7d71` chore(tutorials): refresh MuscleMimic notebook kernel name and cell outputs — Florian Fischer (2026-09-24)
* `0412c8c` docs(mjlab): warn about default num_envs=1 and document --agent.resume — Florian Fischer (2026-09-24)
* `8f6bbdf` feat(eval): evaluate mjlab-trained policies on the CPU env or mjlab; fix RSL-RL ONNX export — Florian Fischer (2026-09-24)
* `658cfdb` fix(mjlab): support mjlab<1.6 command API; isolate per-task registration failures — Florian Fischer (2026-09-24)
* `25e5297` feat(mjlab): port CPU pose and reach basic-suite tasks to the mjlab GPU backend — Florian Fischer (2026-09-23)
* `8231ce1` refactor(terms): generalize pose/reach/walk reward terms for mjlab reuse — Florian Fischer (2026-09-23)
* `99e8812` fix(fatigue): spec-level sarcopenia and torch 3CC-r ctrl parity with CPU — Florian Fischer (2026-09-23)
* `b4f7338` fix(gitignore): stop accidentally ignoring myosuite/**/tasks and scripts/*.py — Florian Fischer (2026-09-23)
* `9e237e3` fix(fatigue): correct activation rate LD in CPU CumulativeFatigue — Florian Fischer (2026-09-23)
* `761fa34` docs: fix Python-version/test-list drift, add 2026 citation to index.rst — Florian Fischer (2026-09-23)
* `583c764` docs: remove orphan compat/rename-matrix docs, align GPU install instructions — Florian Fischer (2026-09-23)
* `88f7bdb` render flag added consistently throughout docs — Florian Fischer (2026-09-23)
* `af2d51d` pinned wandb due to bug in mjlab/rsl-rl training — Florian Fischer (2026-09-23)
* `0a48a5a` explicit cpu device for CPU sample code — Florian Fischer (2026-09-23)
* `1e747f2` fixed potential render issues in verification command — Florian Fischer (2026-09-23)
* `b9e68cb` Fix Core CI: regenerate parity baselines under CI deps; xfail hand_sar — Vittorio-Caggiano (2026-09-20)
* `b5cf39a` Regenerate parity baselines affected by the hand-composition fix — Vittorio-Caggiano (2026-09-20)
* `779ae1f` Fix manipulation SAR script's activation-episode crash; add precomputed pkls — Vittorio-Caggiano (2026-09-20)
* `531dba0` Fix leg-model root/torso orientation: cancel pelvis's intrinsic yaw, not torso's — Vittorio-Caggiano (2026-09-20)
* `53f17a9` Fix myo_sim hand-composition fallback and reorient hand orientation — Vittorio-Caggiano (2026-09-20)
* `061574c` Add missing trailing newline to tutorials 1 and 9 — Vittorio-Caggiano (2026-09-20)
* `2546095` Support Python 3.14 and add it to the CI matrix — Vittorio-Caggiano (2026-09-20)
* `aa2dd77` Fix walk rot termination regression: use root freejoint quat — Vittorio-Caggiano (2026-09-20)
* `f545802` Replace MYOSUITE_FULL_ID env var with FULL_ID notebook flag — Vittorio-Caggiano (2026-09-20)
* `0b8a39f` Remove CI video-noop cells from tutorials; gate in executor — Vittorio-Caggiano (2026-09-20)
* `8707731` Fix DEP-RL tutorial and document Python ≤3.11.5 install — Vittorio-Caggiano (2026-09-20)
* `8e9d51f` Remove Walk_Backends_Demo notebook (broken MJX/benchmark paths) — Vittorio-Caggiano (2026-09-20)
* `704b848` fix: pin myo_sim to 0.2.3 and regenerate the parity baselines (#408) — Hyoungseo Son (2026-09-19)
* `0620c3c` Make SB tutorial more user friendly — Balint-H (2026-09-18)
* `a8eaa6c` Update 4c_Train_SB_policy.ipynb — Balint Hodossy (2026-09-18)
* `cec008d` Update 4c_Train_SB_policy.ipynb — Balint Hodossy (2026-09-18)
* `ff248d4` Update 4c_Train_SB_policy.ipynb — Balint Hodossy (2026-09-18)
* `bb5cf7b` Move precomputed SAR pickles into tutorials/sar/; add manipulation full-pipeline script — Vittorio-Caggiano (2026-09-18)
* `3b025ab` Uncap requires-python<3.14; fix SAR_tutorial.ipynb's missing imports — Vittorio-Caggiano (2026-09-18)
* `6e83da9` cleaned tutorial #11c — Florian Fischer (2026-09-17)
* `df0e2ad` added CUDA-torch install instructions to 11c notebook — Florian Fischer (2026-09-17)
* `99c8ee3` clarified supported python versions — Florian Fischer (2026-09-17)
* `eb3a020` pyproject mjlab dependencies fixed — Florian Fischer (2026-09-17)
* `53f3e9a` added dependencies for tutorial #6; fixed mujoco version casing in #6 — Florian Fischer (2026-09-17)
* `b4d29dc` Change kernelspec to Python 3 and clean up metadata — Florian Fischer (2026-09-16)
* `c0b083c` Tighten orbax-checkpoint pin; fix 11d tutorial's own reinstall + CPU device — Florian Fischer (2026-09-16)
* `f0212ef` Update installation instructions for musclemimic package — Florian Fischer (2026-09-16)
* `6ee93ca` Delete tutorials/test_fatigue.py (deprecated, replaced by fatigue_demo.py) — Florian Fischer (2026-09-16)
* `9ce01b0` Fix path in ReadMe.md for clarity — Florian Fischer (2026-09-16)
* `be917b9` Fixed cache key on id(env) to id(env.cfg), RSI Event MDP handles env_ids=None properly (#407) — Tatsuki Tsujimoto (2026-09-16)
* `1315c36` Fix Core (ubuntu-latest) tutorial-execution failure: install pandas — Vittorio-Caggiano (2026-09-14)
* `4f926db` Fix Core (ubuntu-latest)'s full-suite failures: real regression + 3 documented xfails/skips — Vittorio-Caggiano (2026-09-14)
* `bc46094` Skip (not fail) envs needing the gated musclemimic-retargeted HF dataset — Vittorio-Caggiano (2026-09-14)
* `e9fc956` xfail two root-caused model_stability recipes with tracked diagnoses — Vittorio-Caggiano (2026-09-14)
* `28278d7` Add huggingface_hub as a base dependency — Vittorio-Caggiano (2026-09-14)
* `50aff25` Fix remaining PR #406 CI failures: docs build, pre-commit, wandb import — Vittorio-Caggiano (2026-09-14)
* `1e5271c` remove colab — Vittorio-Caggiano (2026-09-14)
* `3ca1ed3` Fix docs CI: cap requires-python<3.14, fix mjx extra's internal jax/orbax conflict — Vittorio-Caggiano (2026-09-14)
* `6390d7a` Fix test failures surfaced after PR #406 merge — Vittorio-Caggiano (2026-09-14)
* `17ecf07` add header — Vittorio-Caggiano (2026-09-14)
* `1bf9ed3` cleanup — Vittorio-Caggiano (2026-09-14)
* `65ec6c2` update logging instructions — Vittorio-Caggiano (2026-09-14)
* `e27dcd0` Delete orphaned scripts/saber_pose_utils.py — Vittorio-Caggiano (2026-09-12)
* `eab9c0c` Remove boxing/saber tasks and their dead shared-term code — Vittorio-Caggiano (2026-09-12)
* `f8b78bd` Fix GPU-training env-id in merged README; pre-commit EOF whitespace — Vittorio-Caggiano (2026-09-12)
* `de44aca` Remove boxing/saber tasks; fix README/pyproject doc bugs found by snippet audit — Vittorio-Caggiano (2026-09-12)
* `8079643` Make docs and tutorials student-runnable and fix asset/torso orientation. — Vittorio-Caggiano (2026-09-12)
* `b3067db` Add chase-tag full-body baseline generation; keep large baselines out of git — Vittorio-Caggiano (2026-09-08)
* `6362368` Remove stale boxing/saber test references from verification checklist — Vittorio-Caggiano (2026-09-08)
* `c015fca` Fix directional-env heading test to locate heading_cmd by key, not offset — Vittorio-Caggiano (2026-09-08)
* `5df865c` Switch chase-tag inference to heading-reuse; correct additive-obs docstring — Vittorio-Caggiano (2026-08-29)
* `aca5e88` Add reusable Modal launcher: cross-backend trajectory GPU verification — Vittorio-Caggiano (2026-08-29)
* `33d89ce` Fix silent opponent-policy fallthrough on malformed probabilities — Vittorio-Caggiano (2026-08-29)
* `a6f9921` Document ChaseTagEnv's reward weights against the confirmed original spec — Vittorio-Caggiano (2026-08-29)
* `f681cb4` Fix mismatched fall-threshold default in ChaseTagVsConfig — Vittorio-Caggiano (2026-08-29)
* `89faae4` Fix hardcoded arial.ttf font fallback; re-verify tutorials with real training runs — Vittorio-Caggiano (2026-08-28)
* `cf03010` Simplify all tutorials for newcomers; fix bugs surfaced by fresh-env execution — Vittorio-Caggiano (2026-08-25)
* `112fbf6` point to myo_sim@dev branch — Vittorio Caggiano (2026-08-24)
* `2706d46` Remove redundant hardcoded arena walls from 1v1 chase-tag render cell — Vittorio-Caggiano (2026-08-24)
* `815b715` Remove redundant hardcoded arena walls from 1v1 chase-tag render cell — Vittorio-Caggiano (2026-08-24)
* `794ae0a` Sync ms3 with mjx: leg-directional chase-tag, muscle fatigue dynamics, myo_sim-native migration, dependency fixes, AI co-author guard — Vittorio-Caggiano (2026-08-24)
* `fd97274` Feat/leg directional chase tag vs (#101) — Vittorio Caggiano (2026-08-24)
* `42c8802` Fix myo_sim dev-branch compatibility, retire musclemimic_models as a hard dependency, migrate arm/hand envs to myo_sim-native composition (#105) — Vittorio Caggiano (2026-08-16)
* `df22ff5` fix: bump GHSA/CVE-vulnerable deps in uv.lock (#104) — Vittorio Caggiano (2026-07-14)
* `155c8ae` chore: reject AI Co-Authored-By trailers in commit-msg hooks — Vittorio-Caggiano (2026-07-14)
* `b7c83a9` Cleanup: honest docs, safe simplifications, and GPU-verified mjlab/mjx tests (#103) — Vittorio Caggiano (2026-07-14)
* `25f0038` refactor: simplify saber mjlab cfg builders, fix camera init, flatten boxing task config (#86) — Vittorio Caggiano (2026-07-03)
* `15edf5f` fix trajectory_io imports — Florian Fischer (2026-07-04)
* `1d64c34` removed 'persist_muscle_fatigue' variable — Florian Fischer (2026-07-04)
* `9edb7dd` test: surface RSL-RL episode reward in mjlab PPO smoke tests, add Saber coverage (#96) — Vittorio Caggiano (2026-07-02)
* `2d27055` docs: prohibit AI assistant co-author trailers (CLA compliance) (#100) — Vittorio Caggiano (2026-07-01)
* `2ac2403` style: ruff auto-fixes on train_mjlab.py — Vittorio-Caggiano (2026-06-28)
* `c313ed4` update method names and enhance simulation step logic — Vittorio-Caggiano (2026-06-28)
* `57ad9fe` point to dev myo_sim branch — Vittorio-Caggiano (2026-06-27)
* `e45b851` reove unised code — Vittorio-Caggiano (2026-06-27)
* `f5530ca` fix: pin MJX jax/brax floor, guard macOS MUJOCO_GL, add registry sweep test (#97) — Vittorio Caggiano (2026-06-20)
* `6d15631` refactor: dedupe shared mimic infra and remove unused wrapper files — Vittorio-Caggiano (2026-06-19)
* `a453fe4` remove placeholder *Modular-v0 challenge registrations, document task workflow — Vittorio-Caggiano (2026-06-19)
* `3f4bf9d` fix: calibrate myo_sim-pip hand/arm coordinate frame, fix reorient_sar obs parity — Vittorio-Caggiano (2026-06-18)
* `c4ae047` style: pre-commit auto-fixes (pre-push lint pass) — Vittorio-Caggiano (2026-06-18)
* `a96f719` fix: resolve myo_sim pip-migration FileNotFoundErrors, port reorient_sar ID/OOD, remove dead BoxingVsClone env — Vittorio-Caggiano (2026-06-18)
* `05ef296` Polish and add new RL components to standing mjlab env — Balint-H (2026-06-18)
* `3181d1d` Port pip migration stack (PRs #89/#91/#93) onto mjx (#94) — Vittorio Caggiano (2026-06-18)
* `ecf4426` Polish and add new RL components to standing mjlab env — Balint-H (2026-06-18)
* `5122d26` fix: document the 3-DOF Soccer obs-shape delta instead of leaving the test red — Vittorio-Caggiano (2026-06-17)
* `e69c76a` fix: restore Soccer DEFAULT_OBS_KEYS to match PyPI baseline composition — Vittorio-Caggiano (2026-06-17)
* `8b6635b` feat: switch Soccer/TableTennis torso muscles to myo_sim pip, warn on calibration divergence — Vittorio-Caggiano (2026-06-17)
* `7b16bba` deps: bump mpl-sim/object-sim pins to v0.2.1/v0.1.1 — Vittorio-Caggiano (2026-06-17)
* `6a307c4` fix: replace all hardcoded simhive submodule paths with pip-first resolvers — Vittorio-Caggiano (2026-06-17)
* `af8db31` feat: resolve arm/torso/hand simhive paths missing from myo_sim pip — Vittorio-Caggiano (2026-06-17)
* `3a2e355` test: elbow/finger parity against myo_sim pip legacy models — Vittorio-Caggiano (2026-06-16)
* `f7f047b` refactor: centralize model resolvers and remove bundled elbow/finger/osl XMLs — Vittorio-Caggiano (2026-06-16)
* `ffb24a1` fix: remove human_lowpoly_norighthand.stl dependency and fix mesh paths for pip myo_sim — Vittorio-Caggiano (2026-06-16)
* `2528915` feat: migrate all hand task registrations to composed hand (myo_sim myohand_r.xml) — Vittorio-Caggiano (2026-06-16)
* `0ca5ce7` refactor: remove _MODEL_CACHE and _content_hash from ModelBuilder — Vittorio-Caggiano (2026-06-16)
* `bb8d561` refactor: simplify _try_myo_sim_compose redundant None checks — Vittorio-Caggiano (2026-06-16)
* `75e7bc4` fix: fall back to myo_sim pip for unresolvable mesh paths in include files — Vittorio-Caggiano (2026-06-16)
* `66f0cd2` feat: route attach_fragment('hand') through myo_sim compose pipeline — Vittorio-Caggiano (2026-06-16)
* `1ea283d` test: verify hand_standard recipe is numerically equivalent to myo_sim.load('myohand_r') — Vittorio-Caggiano (2026-06-16)
* `d0b9bc1` feat: add attach_spec() to ModelBuilder for pre-built MjSpec fragments — Vittorio-Caggiano (2026-06-16)
* `e008e8a` chore: bundle elbow/finger/hand/osl assets in myosuite (interim) — Vittorio-Caggiano (2026-06-15)
* `d4383f1` fix: write generated saber/musclemimic XMLs to temp cache, not pip package dir — Vittorio-Caggiano (2026-06-15)
* `eb97578` point to staged myo_sim — Vittorio-Caggiano (2026-06-15)
* `022354e` fix temporary myo_sim python wheel — Vittorio-Caggiano (2026-06-15)
* `6b5f9a9` remove composer — Vittorio-Caggiano (2026-06-15)
* `8a888c8` chore: add myo-sim dependency for pip installation in pyproject.toml — Vittorio-Caggiano (2026-06-15)
* `4327936` chore: replace all hardcoded simhive/myo_sim paths with pip-first resolution — Vittorio-Caggiano (2026-06-15)
* `76fbd21` fix: update stale error messages and save path for pip-only myo_sim setup — Vittorio-Caggiano (2026-06-15)
* `5517822` fix: resolve simhive/myo_sim paths via pip in bimanual and tabletennis models — Vittorio-Caggiano (2026-06-15)
* `6211916` chore: remove legacy myosuite_init, test_myoapi, myo_model simhive, and empty .gitmodules — Vittorio-Caggiano (2026-06-15)
* `84847eb` fix: resolve myo_sim models/ subdirectory in pip package path resolution — Vittorio-Caggiano (2026-06-15)
* `b72f097` feat: migrate myo_sim from git submodule to pip-installable package — Vittorio-Caggiano (2026-06-15)
* `ea2f185` fix: lint and import fixes for boxing baseline mjlab configs — Vittorio-Caggiano (2026-06-17)
* `e0875d8` Baseline training mjlab env draft — Balint-H (2026-06-17)
* `084c7b9` refactor: enhance fragment resolution and streamline hand model builder (#92) — Cheryl Wang (2026-06-17)
* `15869aa` test: elbow/finger parity against myo_sim pip legacy models — Vittorio-Caggiano (2026-06-16)
* `2ff5ebf` refactor: centralize model resolvers and remove bundled elbow/finger/osl XMLs — Vittorio-Caggiano (2026-06-16)
* `c364146` fix: remove human_lowpoly_norighthand.stl dependency and fix mesh paths for pip myo_sim — Vittorio-Caggiano (2026-06-16)
* `fbb121c` feat: migrate all hand task registrations to composed hand (myo_sim myohand_r.xml) — Vittorio-Caggiano (2026-06-16)
* `01f60a2` refactor: remove _MODEL_CACHE and _content_hash from ModelBuilder — Vittorio-Caggiano (2026-06-16)
* `371e325` refactor: simplify _try_myo_sim_compose redundant None checks — Vittorio-Caggiano (2026-06-16)
* `bd5a809` fix: fall back to myo_sim pip for unresolvable mesh paths in include files — Vittorio-Caggiano (2026-06-16)
* `06ad927` feat: route attach_fragment('hand') through myo_sim compose pipeline — Vittorio-Caggiano (2026-06-16)
* `3a9c8f3` test: verify hand_standard recipe is numerically equivalent to myo_sim.load('myohand_r') — Vittorio-Caggiano (2026-06-16)
* `bbd6d94` feat: add attach_spec() to ModelBuilder for pre-built MjSpec fragments — Vittorio-Caggiano (2026-06-16)
* `1020dba` chore: bundle elbow/finger/hand/osl assets in myosuite (interim) — Vittorio-Caggiano (2026-06-15)
* `928e12e` fix: write generated saber/musclemimic XMLs to temp cache, not pip package dir — Vittorio-Caggiano (2026-06-15)
* `994dd50` point to staged myo_sim — Vittorio-Caggiano (2026-06-15)
* `bd02736` fix temporary myo_sim python wheel — Vittorio-Caggiano (2026-06-15)
* `d9e5885` remove composer — Vittorio-Caggiano (2026-06-15)
* `bc0d23d` chore: add myo-sim dependency for pip installation in pyproject.toml — Vittorio-Caggiano (2026-06-15)
* `f44e0c2` chore: replace all hardcoded simhive/myo_sim paths with pip-first resolution — Vittorio-Caggiano (2026-06-15)
* `4e4cd41` fix: update stale error messages and save path for pip-only myo_sim setup — Vittorio-Caggiano (2026-06-15)
* `7703653` fix: resolve simhive/myo_sim paths via pip in bimanual and tabletennis models — Vittorio-Caggiano (2026-06-15)
* `083ff92` chore: remove legacy myosuite_init, test_myoapi, myo_model simhive, and empty .gitmodules — Vittorio-Caggiano (2026-06-15)
* `a9fd817` fix: resolve myo_sim models/ subdirectory in pip package path resolution — Vittorio-Caggiano (2026-06-15)
* `df2dfce` feat: migrate myo_sim from git submodule to pip-installable package — Vittorio-Caggiano (2026-06-15)
* `a4b979c` add episode-persistent and resumable fatigue states — Florian Fischer (2026-06-05)
* `92a4b88` clear outputs of fatigue tutorial notebook — Florian Fischer (2026-06-04)
* `d1eda5c` fixed tutorial fatigue notebook for myosuite 3.0 — Florian Fischer (2026-06-04)
* `b6d903e` repaired torch/mjlab fatigue model; minor fixes in CPU fatigue model; added default fatigue params from myoSuite 2.4 as fallback/for debugging — Florian Fischer (2026-06-04)
* `514ee21` style: pre-commit auto-fixes — Vittorio-Caggiano (2026-06-12)
* `7d5ab2a` fix: CLAUDE.md compliance — gym.register() bypass, entity.data.data.* reads, mjlab rules — Vittorio-Caggiano (2026-06-12)
* `7814b83` perf(tabletennis): vectorize GPU contact detection; clean up __init__ wrapper exports — Vittorio-Caggiano (2026-06-12)
* `9223454` feat: mjlab backend for BoxingP0 task (6-pad static target variant) (#83) — Vittorio Caggiano (2026-06-11)
* `b0514c0` fix(#399): clamp RunTrack keyframe qpos values to declared joint ranges — Vittorio-Caggiano (2026-06-10)
* `d6c5444` temp. remove policy runner changes require for future persistent-fatigue update — Florian Fischer (2026-06-09)
* `c6e2c8d` add --init-std / --freeze-std to saber training — Florian Fischer (2026-06-09)
* `e2292e5` increase default PPO epochs for saber+mimic-init training (WIP, untested) — Florian Fischer (2026-06-09)
* `df0bc2c` tune SaberP0 reward weights and default mimic-clip PPO hyperparameters (WIP) — Florian Fischer (2026-06-09)
* `15d50d4` save and restore obs_key in ONNX checkpoints; use runner-based play — Florian Fischer (2026-06-09)
* `3a714d7` minor fix in README.md — Florian Fischer (2026-06-09)
* `897744f` docs: minimize CLAUDE.md and wiki — remove repetition, cut 60% — Vittorio-Caggiano (2026-06-09)
* `b17446c` Add gymnasium P1 env implementation — Balint-H (2026-06-05)
* `c5fd378` Add gymnasium P1 env implementation — Balint-H (2026-06-05)
* `d84e35c` feat: MyoSuite ms3 — modular task framework, mjlab/MJX backend, and challenge suite — Vittorio-Caggiano (2026-06-04)
* `9430099` Add UV installation, CI updates, and fix compatibility issues (#395) — Vittorio Caggiano (2026-05-13)
* `062e6c7` regenerated — Vittorio-Caggiano (2026-05-11)
* `37673eb` Bump urllib3 from 2.6.3 to 2.7.0 (#394) — dependabot[bot] (2026-05-11)
* `0c78d28` Bump gitpython from 3.1.49 to 3.1.50 (#393) — dependabot[bot] (2026-05-11)
* `38cf140` add py3.14 — Vittorio-Caggiano (2026-04-30)
* `a9ae2e0` fix compatibility issues — Vittorio-Caggiano (2026-04-30)
* `ef27602` update assets credits — Vittorio-Caggiano (2026-04-26)
* `581a1b8` fix ci missing library — Vittorio-Caggiano (2026-04-23)
* `21edbfc` bump to mujoco==3.6.0 (enables naccdmax kwarg, see mujoco PR #3096) — Florian Fischer (2026-04-23)
* `fd09a79` fix python version on pypi release ci — Vittorio-Caggiano (2026-04-23)
* `97a965c` fixes pre-release (#388) — Vittorio Caggiano (2026-04-23)
* `57d50bd` Mjx (#354) — Vittorio Caggiano (2026-04-22)
* `f499e9a` Remove stale examine rollout (#365) — Vikash Kumar (2025-11-30)
* `c609355` add uv installation and CI (#359) — Vittorio Caggiano (2025-11-04)
