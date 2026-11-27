# MyoSuite Training Modules

MyoSuite keeps method-specific trainers, world models, imitation pipelines, and isolated dependency stacks under `training/<method>/` outside the core installable `myosuite` package (see [`docs/source/developer/training-guidelines.md`](../docs/source/developer/training-guidelines.md)). Every method consumes registered MyoSuite tasks via `make_env` and exports checkpoints that play back on CPU or `mjlab` without requiring the training package at inference time wherever supported.

---

## Which Method Should I Pick?

| Task family / bottleneck | Recommended entry point | Why choose it over default PPO? |
|---|---|---|
| **Standard GPU RL** (`Pose`, `Reach`, `Walk`, `TableTennis`, `ChaseTagFBP2`) | [`scripts/train_mjlab.py`](../scripts/train_mjlab.py) (`pip install -e ".[mjlab]"`) | Default vectorised MuJoCo-Warp + RSL-RL PPO (`1024–4096` parallel envs); fastest first baseline on GPU. |
| **Lightweight CPU prototyping or off-policy RL** (`1–6` muscles) | [`scripts/train_sb3.py`](../scripts/train_sb3.py) (`pip install -e ".[rl]"`) | Zero GPU setup; supports `PPO`, `SAC`, and `TD3` with `DummyVecEnv` or `SubprocVecEnv`. |
| **Overactuated locomotion & rough/hilly terrain** (`80+` muscles) | [`training/dynsyn`](dynsyn/README.md) · [`tutorials/2.3_SAR.ipynb`](../tutorials/2.3_SAR.ipynb) | Compresses co-lengthening or co-activating muscles into lower-dimensional synergy commands, stabilizing KL updates and terrain exploration. |
| **Multi-clip full-body motion tracking** (`myoMimic-v0`, `354` muscles, `972` clips) | [`training/supermimic`](supermimic/README.md) | Learned body world model + analytic kinematics/muscle/reward heads backpropagates policy gradients through horizon-16 rollouts in **~30 min on 1 GPU** (vs ~30 GPU-h for single-clip PPO). |
| **1v1 competitive full-body sports & steerable locomotion** (`myoChallengeChaseTagFBVs-v0`) | [`training/chasetag_teacher`](chasetag_teacher/README.md) | Steers a `myoSuperMimic` tracker via motion matching and distils an SE(2) yaw-canonicalised ONNX student (**61–76% win rate** vs shipped self-play PPO/BC baseline). |
| **Spinal stretch-reflex equilibrium-point control** (`5–216` muscles, early stability) | `training/lambda_hold` ([PR #875](https://github.com/MyoHub/myosuite/pull/875), [PR #884](https://github.com/MyoHub/myosuite/pull/884)) | Commands normalized threshold lengths through an intrinsic stretch-reflex law (`100%` solved & `+327.6` return over PPO on 216-muscle `myoTorsoPoseFixed-v0`; `95.0%` solved & `+82.6` return with curriculum on 80-muscle `myoLegStandRandom-v0`; `+80.4` return on 39-muscle `myoHandPoseRandom-v0`). |
| **High-DOF multi-muscle pose plateaus** (`myoHandPoseRandom-v0`, `23` DOFs, `39` muscles) | `training/newton_shac` ([PR #898](https://github.com/MyoHub/myosuite/pull/898)) | Propagates analytic Hill-muscle, tendon, and constraint VJPs through `mujoco-warp` (`+69.7` GPU / `+64.4` CPU `make_env` return over PPO at 102k steps). |
| **State-only motion priors** (`(s_t, s_{t+1})` clips without muscle actions) | `training/amp` ([PR #889](https://github.com/MyoHub/myosuite/pull/889)) | Blends LSGAN (`AMP`) or logistic (`GAIL`) transition style rewards with task rewards (**`53.3%`** vs `33.3%` PPO solved on `myoFingerReachRandom-v0`; **`95.0%`** vs `91.7%` PPO on 80-muscle `myoLegStandRandom-v0`). |
| **Merging multiple specialist checkpoints into one policy** | `training/traj_dagger` ([PR #890](https://github.com/MyoHub/myosuite/pull/890)) | Sticky-horizon trajectory-swap DAgger distils heterogeneous specialists without step-wise muscle chattering (**`45.0%`** vs `33.3%` PPO solved on `myoFingerReachRandom-v0`; **`95.0%`** / `620.8` vs `541.5` PPO on 80-muscle `myoLegStandRandom-v0`). |
| **Partially observable / history-dependent tasks without BPTT overhead** | `training/s2pg` ([PR #891](https://github.com/MyoHub/myosuite/pull/891)) | Stochastic stateful `GRU`/`LSTM` PPO trained on flat shuffled single-step minibatches (`+4–16%` GPU wall time vs feedforward PPO, **`93.3%`** / `667.2` vs `541.5` `myo_ppo_runner_cfg` on `myoLegStandRandom-v0`, and `+5.0%` solved under velocity-masked POMDP). |
| **Ultra-low interaction budgets (`~2k` steps) or RGB visual observations** | `training/dreamerv3` ([PR #903](https://github.com/MyoHub/myosuite/pull/903)) | Latent RSSM world model supporting proprioceptive or `64x64` offscreen RGB control (`80%` vs `50%` SB3 PPO solved on `myoElbowPose1D6MRandom-v0` in 2,048 steps). |

---

## Method Architecture & Portability Matrix

### Shipped on `dev`

| Module / Entry Point | Paradigm | Training Backend | Isolated Install | Runtime Playback in Core `myosuite` |
|---|---|---|---|---|
| [`scripts/train_mjlab.py`](../scripts/train_mjlab.py) | On-policy clipped PPO (`MjlabOnPolicyRunner`) | GPU `mjlab` (`mujoco-warp` + RSL-RL) | `pip install -e ".[mjlab]"` | `scripts/eval_mjlab_policy.py` (CPU & `mjlab`); `myosuite-export-onnx` |
| [`scripts/train_sb3.py`](../scripts/train_sb3.py) | On-policy `PPO`, off-policy `SAC` / `TD3` | CPU Gymnasium (`DummyVecEnv` / `SubprocVecEnv`) | `pip install -e ".[rl]"` | SB3 `.zip` + `VecNormalize`; `myosuite-export-onnx` |
| [`training/dynsyn`](dynsyn/README.md) (`myosuite-dynsyn`) | k-medoids co-lengthening muscle synergies + PPO ([He et al., ICML 2024](https://arxiv.org/abs/2407.11472)) | CPU `MjModel` grouping + GPU `mjlab` RSL-RL | `pip install -e training/dynsyn` | **Zero extra deps:** `TrainedActionMap` (`kind="dynsyn"`) plays via `scripts/eval_mjlab_policy.py` & `ActionMapWrapper` |
| [`training/supermimic`](supermimic/README.md) (`myosuite-supermimic`) | Learned body world model + analytic kinematics/muscle/task heads + horizon-16 BP ([Liao et al., 2026](https://arxiv.org/abs/2610.28480)) | GPU `mjlab` / `mujoco-warp` (CUDA-graphed) | `pip install -e training/supermimic` | Self-contained contract-checked `model_*.pt` (`supermimic/scripts/evaluate.py`; CPU loader in `chasetag_teacher.tracker`) |
| [`training/chasetag_teacher`](chasetag_teacher/README.md) (`chasetag-teacher`) | Motion-matched `myoSuperMimic` teacher + SE(2) yaw-canonicalised DAgger ([Ross et al., 2011](https://arxiv.org/abs/1011.0686)) | CPU `myoChallengeChaseTagFBVs-v0` & directional scene | `pip install -e training/chasetag_teacher` | **Zero extra deps:** standalone `.onnx` (`obs [B, 546] -> action [B, 354]` with yaw canonicalisation in-graph) |
| **SAR** ([`tutorials/2.3_SAR.ipynb`](../tutorials/2.3_SAR.ipynb), [`5.5_MuscleMimic_SAR.ipynb`](../tutorials/5.5_MuscleMimic_SAR.ipynb)) | PCA + FastICA synergy basis fit on task rollouts ([Berg et al., RSS 2023](https://roboticsconference.org/2023/program/papers/025/)) | CPU SB3 or GPU `mjlab` (`myoMimic*-SAR-v0`, `myoFullBodyWalkSAR-v0`) | `pip install -e ".[rl]"` or `".[mjlab]"` | Built into `myosuite/envs/myo/backends/mjlab/mimic_mjlab_env.py` and tutorial wrappers |
| **MyoReflex** ([`tutorials/2.4_MyoReflex_Walk.ipynb`](../tutorials/2.4_MyoReflex_Walk.ipynb)) | Spinal neuromuscular reflex locomotion controller (Song & Geyer) | CPU Gymnasium (`CMA-ES` / derivative-free tuning) | `pip install -e .` | Pure NumPy controller in `tutorials/files/2.4/` |

### Candidate Packages in Open PRs

| Package & PR | Paradigm & Reference | Training Backend | Runtime Playback Format |
|---|---|---|---|
| `training/lambda_hold` ([PR #875](https://github.com/MyoHub/myosuite/pull/875), [PR #884](https://github.com/MyoHub/myosuite/pull/884)) | Equilibrium-point stretch-reflex action map + hold/reflex curriculum + PPO ([Lee et al., 2026](https://arxiv.org/abs/2608.17030)) | CPU SB3 (`[sb3]`) & GPU `mjlab` RSL-RL (`[mjlab]`) | `TrainedActionMap` (`kind="lambda_hold"`) in `myosuite.utils.action_map` (CPU & `mjlab`, zero GPU host syncs) |
| `training/amp` ([PR #889](https://github.com/MyoHub/myosuite/pull/889)) | Adversarial Motion Priors (`LSGAN`), `GAIL` (`BCE`), and `BC` ([Peng et al., 2021](https://arxiv.org/abs/2104.02180); [Al-Hafez et al., 2023](https://arxiv.org/abs/2311.02496)) | CPU `CpuVecEnv` & GPU `mjlab` | Standalone `.onnx` and TorchScript `.pt` with `RunningMeanStd` baked in |
| `training/traj_dagger` ([PR #890](https://github.com/MyoHub/myosuite/pull/890)) | Multi-teacher sticky-horizon trajectory-swap DAgger + persistent ring/reservoir buffer ([Al-Hafez et al., 2023](https://arxiv.org/abs/2311.02496)) | CPU `SyncEnvPool` & GPU `mjlab` | Standalone `.onnx` and TorchScript `.ts` with `RunningMeanStd` baked in |
| `training/s2pg` ([PR #891](https://github.com/MyoHub/myosuite/pull/891)) | Stochastic Stateful Policy Gradient PPO (`gru` / `lstm` / `vanilla`, no truncated BPTT; [Al-Hafez et al., ICLR 2024](https://openreview.net/forum?id=p3DKPQ7uaAi)) | CPU `CpuVectorEnv` & GPU `mjlab` | Stateful `.onnx` and TorchScript `.pt`: `(obs, hidden, prev_action) -> (action, next_hidden)` |
| `training/newton_shac` ([PR #898](https://github.com/MyoHub/myosuite/pull/898)) | Differentiable Short-Horizon Actor-Critic with analytic Hill-muscle & tendon VJPs ([Xu et al., ICLR 2022](https://arxiv.org/abs/2204.07137)) | GPU `mujoco-warp` (`step_backward` adjoints; CPU FD fallback) | PyTorch `ActorCritic` verified directly on unmodified CPU `make_env(env_id)` |
| `training/dreamerv3` ([PR #903](https://github.com/MyoHub/myosuite/pull/903)) | Latent RSSM world model for proprioceptive & `64x64` RGB control ([Hafner et al., *Nature* 2025](https://arxiv.org/abs/2301.04104)) | CPU `DreamerEnv` (`make_env`) & GPU `MJXDriver` (`jax.vmap`) | Native JAX `embodied` checkpoint directory (`dreamerv3-train` / `dreamerv3-bench`) |
| `training/eden_rl` (Draft [PR #829](https://github.com/MyoHub/myosuite/pull/829)) | `SimBaV2` hyperspherical backbone, `MuonPPO`, SwAV `HIMEstimator`, `GenesisMuscleBridge` ([Lee et al., 2025](https://arxiv.org/abs/2502.15280); [Long et al., ICLR 2024](https://arxiv.org/abs/2312.11460)) | GPU `mjlab` RSL-RL & `genesis-world` | Standalone `.onnx` (`export_simba_to_onnx`) + `register_actor_trunk_builder("eden_rl.simba", ...)` |

---

## Benchmarks vs Default PPO

Every method is evaluated on unmodified registered MyoSuite environments (`make_env(env_id)`) against PPO under matched environment-step budgets and random seeds.

### 1. Bipedal Locomotion & Terrain (`myoLeg`, 80 muscles — GPU `mjlab`)

Matched `MjlabOnPolicyRunner` runs (`desired_kl = 0.03` on both arms, `--stop-on-success` at `95%` deterministic success; [`training/dynsyn`](dynsyn/README.md), PR `#572`):

| Environment | Metric | Default PPO (`80` actions) | `DynSyn` (`K = 40` groups, `max_amplitude = 0`) | Key takeaway |
|---|---|---|---|---|
| `myoLegWalk-v0` (`6` seeds `0–5`) | Iterations to `95%` solved (per seed / median) | `582, 662, 737, 839, 956, 1357` (median **`788`**) | `502, 654, 720, 774, 811, 877` (median **`747`**) | Tighter worst-case seed variance (`877` vs `1357` max iters; `6/6` converge). |
| `myoLegRoughTerrainWalk-v0` (`2` seeds `0–1`) | Iterations to `95%` solved | `736, 912` | **`561, 682`** | `~24–25%` fewer iterations on uneven heightfield contacts. |
| `myoLegHillyTerrainWalk-v0` (`2` seeds `0–1`) | Iterations to `95%` solved | `1014, >3000` (`88%` at iter 3000) | **`894, 1612`** | Converges to `95%` on both seeds where plain PPO stalls on seed 1. |

### 2. Full-Body Motion Tracking (`myoMimic-v0`, 354 muscles — GPU `mjlab`)

Multi-clip tracking on MuscleMimic's `972` training / `108` held-out AMASS KIT motions (`1024` parallel envs on 1× RTX 5090; [`training/supermimic`](supermimic/README.md) & [`docs/baseline_checkpoints.md`](../docs/baseline_checkpoints.md)):

| Method | Scope | Compute / Wall Time | Held-Out Return | Failed Clips | Joint / Site Error |
|---|---|---|---|---|---|
| Default PPO (`scripts/train_mjlab.py`) | **Single clip** (`KIT/167/walking_medium06`) | `2.0B` steps (`~30 GPU-h`, 1 GPU, 1024 envs) | `86%` clip completion | — | Single-clip specialist only |
| `myoSuperMimic` (`training/supermimic`) | **All 972 KIT clips** (`108` held-out) | **`30 min`** (`13,993` iters, warm) | `0.922` | `3 / 108` | **`23.0 mm`** MPJPE (`10.8 min` to `>= 0.83` return) |
| `myoSuperMimic` + 5-step test-time refine | **All 972 KIT clips** (`108` held-out) | Same `30 min` checkpoint (`--refine 5`) | **`0.950`** | **`0 / 108`** | **`18.6 mm`** MPJPE |

### 3. Competitive 1v1 Chase Tag (`myoChallengeChaseTagFBVs-v0`, 354 muscles — CPU)

48 deterministic episodes per match and side against an idle opponent and against the shipped self-play PPO/BC baseline (`onnx/chasetag_fbvs_baseline.onnx`, 4,000 self-play iterations × 2,048 envs; [`training/chasetag_teacher`](chasetag_teacher/README.md) & [`docs/baseline_checkpoints.md`](../docs/baseline_checkpoints.md)):

| Policy | Artifact / Source | Tags vs Idle | Win Rate vs Shipped Self-Play PPO Baseline |
|---|---|---|---|
| Shipped self-play PPO/BC baseline | `onnx/chasetag_fbvs_baseline.onnx` | `100%` | `50%` (reference) |
| Steered `myoSuperMimic` teacher (30-min tracker) | `supermimic/tracker_myofullbody_kit_30min.pt` | `92%` | **`71%`** |
| Yaw-canonicalised DAgger student (round 21) | `onnx/chasetag_supermimic_student.onnx` | `98%` | **`61%`** (single feedforward ONNX) |
| Directional walker distilled via DAgger (round 5) | `directional/walker_myofullbody.pt` | `100%` | **`64%`** (`85%` 10-s survival across 8 headings) |
| Self-play PPO refinement of DAgger student | `onnx/chasetag_supermimic_selfplay.onnx` | `100%` | **`76%`** |

### 4. Upper-Limb (`myoElbowPose1D6MRandom-v0`, 6M) & Dexterous Hand (`myoHandPoseRandom-v0`, 39M)

Matched 3-seed (`0, 1, 2`) side-by-side benchmarks from each package's suite (mean ± std across seeds; PPO row in every block uses the identical step budget, vec-env count, and hardware as the compared method):

| Study & Step Budget | Arm | `myoElbowPose1D6MRandom-v0` Return (Solved %) | `myoHandPoseRandom-v0` Return (Solved %) | Wall Time (Elbow / Hand) | Summary vs Matched PPO |
|---|---|---|---|---|---|
| **Differentiable SHAC** ([PR #898](https://github.com/MyoHub/myosuite/pull/898))<br>`102,400` steps, `256` GPU envs (RTX 4070 Ti SUPER) | `PPO`<br>**`Newton-SHAC`** | `751.3 ± 3.6` (`100%`)<br>**`769.3 ± 1.5` (`100%`)** | `-322.1 ± 6.6` (`0%`)<br>**`-252.4 ± 1.7`** (`0%`) | `33.0 s` / `37.6 s`<br>`68.0 s` / `72.8 s` | Analytic muscle/tendon VJPs break the 39-muscle hand plateau (`+69.7` GPU, `+64.4` on CPU `make_env`) and match PPO's final elbow return in half the steps (`33.4 s`). |
| **Stretch-Reflex Action Map** ([PR #875](https://github.com/MyoHub/myosuite/pull/875), [#884](https://github.com/MyoHub/myosuite/pull/884))<br>`65,536` steps, `8` CPU envs | `PPO`<br>`LambdaHold+PPO`<br>**`CurriculumLambdaHold`** | `605.7 ± 49.8` (`98.3%`)<br>`366.9 ± 9.2` (`73.3%`)<br>**`592.3 ± 63.3` (`96.7%`)** | `-311.7 ± 2.4` (`0%`)<br>**`-231.3 ± 1.0`** (`0%`)<br>— | `26.4 s` / `44.3 s`<br>`30.0 s` / `48.8 s`<br>`30.5 s` / — | `55.0%` zero-shot elbow solve at step `0` and `+80.4` hand return at `65k` steps; annealing hold/reflex (`#884`) retains fast startup and `96.7%` asymptotic solve. |
| **Motion Priors (AMP / GAIL)** ([PR #889](https://github.com/MyoHub/myosuite/pull/889))<br>`32,768` steps (`0` online for `BC`), `8` CPU envs | `PPO`<br>`BC` *(uses actions)*<br>`GAIL` *(states only)*<br>**`AMP` *(states only)*** | `770.6 ± 1.7` (`100%`)<br>`764.4 ± 0.6` (`100%`)<br>`768.6 ± 1.1` (`100%`)<br>**`769.2 ± 0.7` (`100%`)** | `-319.3 ± 5.5` (`0%`)<br>`-319.2 ± 1.1` (`0%`)<br>`-318.2 ± 2.6` (`0%`)<br>**`-316.2 ± 6.0`** (`0%`) | `17.3 s` / `31.1 s`<br>`5.0 s` / `8.3 s`<br>`17.9 s` / `31.6 s`<br>`17.8 s` / `31.6 s` | `AMP` matches PPO return with `2.4x` tighter elbow seed variance and surpasses offline `BC` without requiring expert muscle excitations. |
| **Multi-Teacher Distillation** ([PR #890](https://github.com/MyoHub/myosuite/pull/890))<br>`32,768` steps, `2` disjoint-half specialists | `PPO` *(from scratch)*<br>`BC` *(offline)*<br>**`TrajSwapDAgger`** | `261.0 ± 107.1` (`33.3%`)<br>`362.4 ± 155.1` (`43.3%`)<br>**`411.6 ± 110.4` (`50.0%`)** | `-326.0 ± 11.2` (`0%`)<br>`-319.4 ± 1.4` (`0%`)<br>**`-312.9 ± 2.6`** (`0%`) | `17.4 s` / `28.7 s`<br>`17.5 s` / `28.7 s`<br>`30.7 s` / `40.8 s` | Consolidates disjoint workspace specialists with `+150.6` return over scratch PPO and `+49.1` over offline `BC` (`83.3%` any-step elbow solve). |
| **Stateful Recurrent PPO** ([PR #891](https://github.com/MyoHub/myosuite/pull/891))<br>`32,768` steps, `8` CPU envs | `PPO` *(feedforward)*<br>`S2PG-PPO` *(GRU)* | `768.0 ± 3.5` (`100%`)<br>`650.2 ± 125.7` (`75.0%`) | `-316.1 ± 2.5` (`0%`)<br>**`-316.1 ± 2.4`** (`0%`) | `12.3 s` / `22.4 s`<br>`15.6 s` / `25.8 s` | Adds stochastic `GRU` state in `+15–27%` wall time (flat minibatches, no BPTT); also edges feedforward PPO on 39M hand under `FatigueWrapper` (`-318.0` vs `-318.4`). |
| **Latent World Model** ([PR #903](https://github.com/MyoHub/myosuite/pull/903))<br>**`2,048` steps**, CPU sample-efficiency regime | `PPO` *(SB3)*<br>**`DreamerV3`** | `84.2 ± 22.8` (`50.0%`)<br>**`155.1 ± 15.6` (`80.0%`)** | **`-312.2 ± 2.3`** (`0%`)<br>`-349.0 ± 4.3` (`0%`) | `1.7 s` / `1.9 s`<br>`11.3 s` / `19.7 s` | Reaches `80%` solved on 6-muscle elbow in just `2,048` env steps (`+70.9` return over PPO); 39-muscle hand requires larger world-model capacity and GPU `MJXDriver` beyond `2k` steps. |

### 5. Multi-Task Solved GPU `mjlab` Benchmarks (`5M` Finger, `80M` Leg Stand, `216M` Torso — `184,320` Steps)

Matched 3-seed (`0, 1, 2`) GPU `mjlab` runs (`256` parallel envs on `RTX 4070 Ti SUPER`, `30 × 24 × 256 = 184,320` steps per seed, evaluated deterministically over 20 seeded episodes on unmodified CPU `make_env(env_id)` against MyoSuite's registered `myo_ppo_runner_cfg`):

| Method & PR | `myoFingerPoseRandom-v0` (5M) | `myoFingerReachRandom-v0` (5M) | `myoLegStandRandom-v0` (80M) | `myoTorsoPoseFixed-v0` (216M) | Key Advantage vs `myo_ppo_runner_cfg` |
|---|---:|---:|---:|---:|---|
| **`myo_ppo_runner_cfg` (`scripts/train_mjlab.py`)** | **`243.5 ± 15.8`** (`96.7%`) | `234.0 ± 62.2` (`33.3%`) | `541.5 ± 33.2` (`91.7%`) | `-88.4 ± 13.8` (`100.0%`) | Canonical RSL-RL PPO baseline (`desired_kl=0.01`). |
| **`LambdaHold + PPO`** ([#875](https://github.com/MyoHub/myosuite/pull/875)) | `198.6 ± 49.0` (**`100.0%`**) | `144.8 ± 12.0` (`28.3%`) | `352.8 ± 18.5` (`90.0%`) | **`+239.2 ± 148.2` (`100.0%`)** | **`100%` solved** on 5M finger pose & **`+327.6` return** on 216M spine. |
| **`CurriculumLambdaHold + PPO`** ([#884](https://github.com/MyoHub/myosuite/pull/884)) | `220.9 ± 80.2` (`85.0%`) | **`242.7 ± 55.8`** (`16.7%`) | **`624.2 ± 62.3` (`95.0%`)** | `-70.4 ± 11.8` (`100.0%`) | Recovers full dynamic range (`+82.6` return & **`95.0%`** solved on 80M balance). |
| **`AMP`** *(states `(s_t, s_{t+1})` only)* ([#889](https://github.com/MyoHub/myosuite/pull/889)) | `230.6 ± 46.2` (`91.7%`) | **`274.6 ± 33.5` (`53.3%`)** | `562.6 ± 39.9` (**`95.0%`**) | — | **`+20.0%` solved** on 3D finger reach & **`95.0%` solved** on 80M balance without actions. |
| **`TrajSwapDAgger`** *(2 half-specialists)* ([#890](https://github.com/MyoHub/myosuite/pull/890)) | `178.5 ± 12.7` (`66.7%`; **`96.7%` @ `61k`**) | **`307.2 ± 1.5` (`45.0%`)** | **`620.8 ± 1.9` (`95.0%`)** | — | Consolidates disjoint specialists (`+73.3` reach return, `+79.2` 80M balance return). |
| **`S2PG-PPO` (`gru`)** ([#891](https://github.com/MyoHub/myosuite/pull/891)) | `-30.9 ± 5.6` (`28.3%`; **`33.3%` vs `28.3%` FF on `qvel=0` POMDP**) | `-9.9 ± 3.1` (`0.0%`) | **`667.2 ± 26.7` (`93.3%`)** | — | **`+125.7` return** on 80M balance at `+4.2%` GPU wall time & `+5.0%` solved on velocity-masked POMDP. |
| **`SimBaV2 + MuonPPO`** (Draft [#829](https://github.com/MyoHub/myosuite/pull/829)) | `192.2 ± 140.0` (`85.0%`) | `165.7 ± 30.5` (**`35.0%`**) | **`676.8 ± 6.7`** (`91.7%`) | `-111.8 ± 3.0` (`100.0%`) | Highest 80M balance return (**`676.8 ± 6.7`**, `+135.3` over `myo_ppo_runner_cfg`). |

---

## Dependency Isolation & Contributor Rules

1. **Zero root dependency leakage:** Each `training/<method>/` directory is an independent Python project with its own `pyproject.toml`, `README.md`, license notices, and unit tests. Never add method-specific dependencies to the root [`pyproject.toml`](../pyproject.toml).
2. **Portable inference artifacts:** Export standard ONNX (`myosuite.utils.export_onnx`), TorchScript, or registered `ActionMap` / `rslrl_policy` checkpoints so policies play back on `make_env(env_id)` without installing `training/<method>/`.
3. **Mandatory PPO benchmark:** Any new method added under `training/` must ship with a matched, multi-seed (`>= 3` seeds) comparison against PPO on registered MyoSuite `env_id`s and demonstrate a concrete capability or performance gain (see [`docs/source/developer/training-guidelines.md`](../docs/source/developer/training-guidelines.md)).
