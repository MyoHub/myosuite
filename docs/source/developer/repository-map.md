# Repository Map

> **Source of truth priority:** code and tests → [AGENTS.md](https://github.com/MyoHub/myosuite/blob/dev/AGENTS.md) → this guide.
> **Package boundaries:** enforced in CI by `myosuite/tests/test_package_boundaries.py` and `myosuite/tests/test_package_discovery.py`.

## Subpackage Ownership & Allowed-Import Matrix

`myo_sim` owns reusable anatomical MJCF/meshes, muscle strength/sarcopenia, muscle fatigue, joint couplings, and anthropometric scaling. `myosuite` consumes public `myo_sim` APIs and never duplicates library-side model mechanics.

| Layer | Subpackage / Directory | Status | Responsibility | Allowed `myosuite.*` Runtime Imports |
| :--- | :--- | :--- | :--- | :--- |
| **Foundation** | `myosuite/core/` | Active | `EnvConfig`, `registry`, `make_env`, `ModelBuilder`, recipes, `seeding`, `dict_utils`, `import_utils`, `path_utils`, `prompt_utils`, `reach_workspace`, `step_timing`, `target_ranges`, `tensor_utils`, scene/spec `builder/`, pure MDP `terms/`, muscle conditions/scaling facades | `core`, `physics` |
| **Numerical Math** | `myosuite/physics/` | Active | Shared numerical helpers and compatibility imports (`quat*`, `running_stats`, `colored_noise`, `joint_coupling`, `fatigue_jax`); canonical model mechanics live in `myo_sim` | `physics` |
| **MDP Terms (Shim)** | `myosuite/terms/` | Closed facade | Backward-compatible aliases to `core/terms/`, `benchmarks/term_catalog.py` (`myosuite-terms`), and `envs/tasks/games/chasetag/chase_tag_vs_*` | `core`, `physics`, `terms` *(+ compatibility alias forwarding to `benchmarks` / `envs`)* |
| **Scene Assembly (Shim)** | `myosuite/scenes/` | Closed facade | Backward-compatible aliases to `core/builder/{scene_library,scene_builder}.py` | `core`, `scenes` |
| **Motion & Trials** | `myosuite/motion/` | Active | Flat motion contract subpackage (`MotionClip` & trajectory `io.py`, `transforms.py`, `playback.py`, `hf_cache.py`, `reference.py`, `reference_jax.py`, composable `KinematicTrial` / `EMGTrial`) | `core`, `physics`, `motion` |
| **Scientific Analysis** | `myosuite/biomechanics/` | Active | Site IK (`mink`), coupling-aware landmark IK, inverse dynamics, static optimization, CEINMS, CMC, EMG processing, external-load wrench mapping (`external_loads.py`), effort, observability/uncertainty | `core`, `physics`, `motion`, `biomechanics` |
| **Ecosystem Bridges** | `myosuite/integrations/` | Active | Lightweight source-specific readers/adapters (`capture/markers`, `marker_retarget`, `smplh`, `retarget_smplh`, `mano`, `gvhmr`, `opensim_motion`, `opensim_grf`, `chumpy_pickle`) and `musclemimic` runtime bridges *(InstantHMR adapter `biomechanics/retarget_instanthmr.py` migration deferred)* | `core`, `terms`, `physics`, `motion`, `policies`, `biomechanics`, `integrations` |
| **Runtime Inference** | `myosuite/policies/` | Active | Reusable checkpoint/ONNX/Orbax loading (`onnx_checkpoint`, `onnxruntime_utils`, `orbax_checkpoint`, `rslrl_policy`, `checkpoint_utils`), `export_onnx` CLI, checkpoint/deployment manifests and `action_map` runtime wrappers | `core`, `terms`, `physics`, `motion`, `biomechanics`, `policies` |
| **Environments** | `myosuite/envs/` | Active | `gymnasium_env.py`, `modular_env.py`, `wrappers.py`, `muscle_stages.py`, `feature_cli.py`, `envs/tasks/<domain>/<family>/`, `envs/backends/{mjlab,_experimental_mjx}/`, `envs/assets/`, and closed compatibility facade `envs/myo/` | `core`, `scenes`, `terms`, `physics`, `motion`, `policies`, `biomechanics`, `integrations`, `envs` |
| **Visualization** | `myosuite/viz/` | Active | Offscreen/interactive viewer, rollout logging/plotting/video I/O (`roboset_logger`, `grouped_datasets`, `examine_logs`, `examine_reference`, `path_plotting`, `policy_playback`, `video_io`), volumetric `muscle_meshes`/`muscle_tubes`, `.skn`/`.glb` skinning, USD/Blender export | `core`, `scenes`, `physics`, `motion`, `policies`, `biomechanics`, `envs`, `viz` |
| **Evaluation Suites** | `myosuite/benchmarks/` | Active | Standardized evaluation suites (`paired_replay`, `push_*`, `performance_card`, `examine_env`, `examine_sim`, `env_check`, `term_catalog` / `myosuite-terms`, `musclemimic_fullbody_eval_cli`) | `core`, `terms`, `physics`, `motion`, `policies`, `biomechanics`, `integrations`, `envs`, `viz`, `benchmarks` |
| **Rollout Logging (Shim)** | `myosuite/logger/` | Closed facade | Backward-compatible aliases to `viz/` and `motion/` | `core`, `physics`, `motion`, `viz`, `logger` |
| **Shared Utilities (Shim)** | `myosuite/utils/` | Closed facade *(+ legacy `sb3_*` until training isolation)* | Backward-compatible aliases to `core/`, `policies/`, `envs/`, `viz/`, and `benchmarks/` | `core`, `terms`, `physics`, `motion`, `policies`, `biomechanics`, `envs`, `viz`, `benchmarks`, `utils` |
| **Out-of-Wheel RL** | `training/<method>/` | Active | Self-contained RL/control projects with independent `pyproject.toml` (`chasetag_teacher`, `dynsyn`, `supermimic`, ...); see [training-guidelines.md](training-guidelines.md) | Consumes `myosuite`; **never** imported by `myosuite/**` |
| **Out-of-Wheel Tools** | `tools/<tool>/` | Active | Heavy CV/perception extractors (`instanthmr`) with independent `pyproject.toml` | Consumes `myosuite`; **never** imported by `myosuite/**` |

## Top-Level

```
myosuite/          # core package (see below; tests/ excluded from built wheels)
docs/source/       # Sphinx documentation (quickstarts, environment reference, API)
docs/source/developer/ # this developer guide
scripts/           # training, evaluation and rendering CLIs (train_mjlab, train_sb3, eval_mjlab_policy, render_blender), parity baselines
training/          # self-contained training projects outside the package, one per method (chasetag_teacher/, dynsyn/, supermimic/); see training-guidelines.md
tutorials/         # notebooks (1.x-5.x) and their companion files in files/X.Y/
tools/             # standalone data/perception CLIs with isolated dependencies (instanthmr/)
.github/           # CI and release workflows
```

## `myosuite/` Structure

| Directory | Contents |
|---|---|
| `core/` | Registry and `make_env`, config dataclasses (`EnvConfig`, `TaskConfig`), `ModelBuilder`/recipes, `seeding`, `dict_utils`, `import_utils`, `path_utils`, `prompt_utils`, `reach_workspace`, `step_timing`, `target_ranges`, `tensor_utils`, `core/builder/` (scenes, MJCF spec/geom/joint/asset helpers), `core/terms/` (backend-agnostic MDP terms), muscle conditions/scaling facades |
| `terms/` | Closed backward-compatible aliases to `core/terms/`, `benchmarks/term_catalog.py`, and `envs/tasks/games/chasetag/chase_tag_vs_*` |
| `physics/` | Model-level runtime math — quaternions, running stats, colored noise, fatigue, joint couplings. No backend imports. |
| `motion/` | Model-coordinate `MotionClip` & trajectory I/O, yaw transforms, playback contracts, HF dataset cache, reference motions, and composable `KinematicTrial` / `EMGTrial` |
| `biomechanics/` | Musculoskeletal analysis and fitting solvers (IK, ID, static optimization, CMC, EMG, external loads (`external_loads.py`), effort). NumPy / SciPy / MuJoCo only. |
| `integrations/` | Lightweight source readers/adapters (`capture/markers`, `marker_retarget`, `smplh`, `retarget_smplh`, `mano`, `gvhmr`, `opensim_motion`, `opensim_grf`) and third-party runtime bridges (`musclemimic`) |
| `envs/` | `gymnasium_env.py` (CPU base), `modular_env.py` and `multi_agent_modular_env.py` (data-driven envs), `wrappers.py` and `muscle_stages.py` (muscle noise / fatigue / reafferentation / sarcopenia as wrappers) |
| `envs/tasks/` | Co-located task definitions organized by anatomical/functional domain (`arm/`, `hand/`, `leg/`, `torso/`, `fullbody/`, `games/`, `mimic/`, plus registration `suites/{basic,challenge,edits}.py`) |
| `envs/backends/` | Shared `mjlab/` (Warp) and `_experimental_mjx/` (JAX) execution backends |
| `envs/assets/` | Task-scene MJCF wrappers, terrain heightfields, object subsets (`MPL_sim`, `YCB_sim`, `furniture_sim`), `_resolve.py`, plus retained `_MYO_SIM_BUNDLED_FALLBACKS` (`arm/assets/myoarm_body.xml`, `hand/assets/myohand_body.xml`, `torso/assets/myotorso_arm_chain.xml`) pending `myo_sim` packaging |
| `envs/myo/` | Closed backward-compatible `sys.modules` alias facade to `envs/tasks/`, `envs/backends/`, and `envs/assets/` |
| `policies/` | Checkpoint/ONNX/Orbax loading (`rslrl_policy.py`, `onnx_checkpoint.py`, `onnxruntime_utils.py`, `orbax_checkpoint.py`, `checkpoint_utils.py`), `export_onnx.py`, `checkpoint_manifest.py`, `deployment_manifest.py`, `action_map.py` |
| `benchmarks/` | Evaluation suites (`paired_replay.py`, `push_*.py`, `performance_card.py`, `examine_env.py`, `examine_sim.py`, `env_check.py`, `term_catalog.py`) |
| `utils/` | Backward-compatible aliases to `core/`, `policies/`, `envs/`, `viz/`, and `benchmarks/` (plus `sb3_*` helpers) |
| `viz/` | Rendering, rollout logging/plotting (`roboset_logger.py`, `grouped_datasets.py`, `examine_logs.py`, `path_plotting.py`, `policy_playback.py`, `video_io.py`), `blender_render.py`, `muscle_tubes.py`, `muscle_meshes.py`, and `skin.py` |
| `logger/` | Backward-compatible aliases to `viz/` and `motion/` |
| `scenes/` | Backward-compatible aliases to `core/builder/` |
| `tests/` | Test suites (tiers 1–3, see pyproject markers); `tests/benchmarks/` holds the throughput benchmarks |

## Task Layout

```
envs/tasks/              # Co-located task definitions (CPU + mjlab twin configs)
├── suites/
│   ├── basic.py            # registers basic CPU envs via registry.register_env
│   ├── challenge.py        # registers MyoChallenge CPU envs
│   ├── edits.py            # model-editing variants (e.g. myoArmReach*, key_turn)
│   ├── muscle_mixin.py     # runs the muscle-command stages (noise, fatigue, ...) for basic envs
│   └── challenge_common.py # shared MyoChallenge helpers
├── arm/ hand/ leg/ torso/ fullbody/
│   └── <family>/           # env.py (MyoGymnasiumEnv), spec.py (TaskConfig), mjlab_cfg.py, mjlab_register.py, rl_cfg.py
├── games/                  # chasetag/ (incl. chase_tag_vs/ & chase_tag_vs_{obs,reward}), soccer/, tabletennis/
└── mimic/                  # MuscleMimic CPU/mjlab tasks; tracking/ = myoMimic-v0 (task cfg, models, motion lib, metrics)

envs/backends/           # Shared GPU backend infrastructure
├── mjlab/                # MuJoCo-Warp ManagerBasedRlEnvCfg (supported)
│   ├── tasks/            # shared twin builders & generic MDP terms:
│   │                     #   cpu_reference.py, registration.py, pose/, reach/, mdp/, rl.py
│   └── configs/, register_mjlab_tasks.py, mjlab_task_builder.py
└── _experimental_mjx/    # JAX/brax env classes (experimental; not guaranteed long-term)
```

A task's **CPU** env (`envs/tasks/<domain>/<family>/env.py`) and its supported **GPU** twin
(`mjlab_cfg.py` + `mjlab_register.py` co-located in the same task folder, backed by shared helpers in `envs/backends/mjlab/`) share one `env_id` and the cross-backend contract — see
[engineering-standards.md](engineering-standards.md). Tasks hand-write the CPU `MyoGymnasiumEnv`; the mjlab
twin of the basic envs is built from the CPU registration (`envs/backends/mjlab/tasks/cpu_reference.py`), so
parameters live in one place. (An experimental `TaskConfig` route can instead generate a data-driven CPU env
plus an MJX backend from one dataclass — the elbow pose and leg directional references; MJX is not
guaranteed long-term.)

## Navigation

| Goal | Go to |
|---|---|
| Add/change CPU env registration | `myosuite/core/registry.py` + `myosuite/envs/tasks/suites/{basic,challenge,edits}.py` (or `envs/tasks/mimic/__init__.py`) |
| Add/change model or scene composition | `myosuite/core/model_builder.py`, `model_recipes.py`, `core/builder/` |
| Add/change a term function | `myosuite/core/terms/base_*.py` |
| Add/change biomechanics math | `myosuite/physics/` |
| Analyse a motion (IK, ID, static optimization, CMC) or muscle effort of rollouts | `myosuite/biomechanics/` |
| Retarget motion data (markers, SMPL+H, GVHMR, MANO, OpenSim; InstantHMR adapter currently in `biomechanics/retarget_instanthmr.py`) | `myosuite/integrations/`, `tools/instanthmr/` |
| Add a CPU env (`MyoGymnasiumEnv`) | `envs/tasks/<domain>/<family>/env.py` |
| Add the matched mjlab GPU task | `envs/tasks/<domain>/<family>/{mjlab_cfg,mjlab_register}.py` (using shared builders in `envs/backends/mjlab/tasks/`) |
| Muscle noise, fatigue, reafferentation, sarcopenia | `envs/wrappers.py`, `envs/muscle_stages.py`; on mjlab `envs/backends/mjlab/tasks/mdp/actions.py` |
| Play a checkpoint trained through an action map (DynSyn) | `policies/action_map.py`; on mjlab `envs/backends/mjlab/tasks/mdp/action_map.py`; training in `training/dynsyn/` |
| `make_env` / `EnvConfig` (one entry point for both backends) | `core/registry.py`, `core/config.py`; `envs/feature_cli.py` for the `--feature` flags |
| Data-driven `TaskConfig` route (experimental) | `envs/tasks/<domain>/<family>/spec.py` + [adding-a-new-task.md](adding-a-new-task.md) |
| Change the CPU step/reset loop | `myosuite/envs/gymnasium_env.py` |

## Naming Conventions

- No `myo_` prefix inside `myosuite/` — redundant within the package.
- Effector directories: anatomical nouns (`arm/`, `hand/`, `leg/`, `torso/`).
- Env **IDs** keep their public `-v0`/`-v1` suffix (Gymnasium convention, part of
  the public API). Some existing env **classes** also carry a `V0` suffix
  (`ReachEnvV0`) for historical reasons; new CPU env classes need not.
