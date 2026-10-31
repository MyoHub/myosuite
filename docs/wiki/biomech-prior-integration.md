# BiomechPriorVAE integration proposal

Use the learned gait-state prior from [BiomechPriorVAE](https://github.com/gambimar/BiomechPriorVAE) first as an offline evaluator for full-body MuscleMimic rollouts. Validate its coordinate mapping and score distributions before adding an optional training reward. This is a design proposal; it adds no runtime integration, dependency, checkpoint or environment registration. Upstream correctness fixes are reviewed separately in [BiomechPriorVAE PR #5](https://github.com/gambimar/BiomechPriorVAE/pull/5).

## Initial scope

Compare reference motion, successful policies, deliberately perturbed motion and failed rollouts, grouped by speed and contact phase. Report one reconstruction negative log-likelihood (NLL) per state and retain the existing task metrics alongside it. A low state-prior score alone does not establish dynamic feasibility or temporal consistency.

Start with full-body models because the published prior includes arm coordinates. MyoLeg needs a retrained feature subset; filling absent arm features with zeros changes the input distribution. Reaching and seated-driving tasks similarly need motion data appropriate to those tasks.

## Code placement

Follow the training guidelines proposed in [MyoSuite #581](https://github.com/MyoHub/myosuite/pull/581) (`docs/wiki/training-guidelines.md`): MyoSuite defines the task and feature contract, and the prior is a method that consumes it.

- **`training/biomech_prior/`.** A self-contained project with its own `pyproject.toml` (PyTorch and the upstream model code), README, tests and lockfile. It holds:
  - the network, and checkpoint and scaler loading;
  - retraining for other feature subsets, such as MyoLeg;
  - the offline scoring command;
  - the reward-weight experiments.

  Install it beside MyoSuite with `pip install "git+https://github.com/MyoHub/myosuite.git@dev#subdirectory=training/biomech_prior"`.
- **`myosuite/`.** Only the lightweight interface both sides need:
  - the typed feature specification;
  - a function that maps CPU or mjlab state to the 50 features;
  - later, an optional reward term that calls a scorer supplied by the training project.

  The package gains no PyTorch dependency and no checkpoint.
- **Attribution and licensing.** The README credits the BiomechPriorVAE authors and paper. The reviewed upstream revision has no licence file, so the code and checkpoint terms must be settled with the authors before anything is adapted into `training/`.

## Feature contract

The published network takes 50 features: 23 non-root joint positions, their 23 velocities and four ground-force features. Its latent size is 24. Keep the checkpoint paired with its original scaler, and record the source revision and checksums in an artifact manifest.

| Group | Required mapping |
|---|---|
| Positions and velocities | Match the upstream `target_dof_names` order, using radians and radians per second. Omit the free root, subtalar, MTP and wrist coordinates. Validate anatomical axes, neutral offsets and knee signs with matched poses. |
| Right-foot force | Sum world-frame ground forces on the right foot, divide by subject body mass times 9.81, then record vertical force followed by the norm of the two horizontal components. |
| Left-foot force | Apply the same operation to the left foot, preserving the right-then-left order. Exclude ground contacts on other body parts and contacts with carried objects. |
| World axes | AddBiomechanics/BioMAC uses Y vertical; MuJoCo uses Z vertical. Perform this conversion before forming force features. |

The feature specification should be a typed dataclass under a future `myosuite/integrations/biomech_prior/` package, the only part of this proposal inside the installable package. It must define joint order, signs, offsets, units, force axes, body-mass normalization and missing-feature behavior. Reject unsupported layouts explicitly. Matching names alone does not establish equivalent anatomical coordinates.

The artifact manifest should also identify the architecture, scaler, training version and decoder variance floor. Corrected upstream normalization and KL reduction affect newly trained models, so publish new checkpoint/scaler pairs together. Resolve code and checkpoint redistribution terms before bundling either: the reviewed BiomechPriorVAE revision has no license file or separate checkpoint terms.

Match the upstream inference objective: standardize the input, encode it and decode the latent mean deterministically. Unscale the decoded mean with the paired scaler and the decoded variance with its squared scale, floor that physical variance at `1e-6`, and average Gaussian NLL over features for each state. Do not sample the latent variable or add the training KL term during evaluation. Averaging standardized-space NLL would define a different score.

## Existing implementation hooks

These hooks exist on MyoSuite `dev`:

- [MuscleMimic model bridge](../../myosuite/integrations/musclemimic/model_bridge.py) defines `FULLBODY_NAME_ALIASES`, including the elbow naming differences. Reuse its name mapping while validating the additional anatomical transforms against the training convention.
- [CPU MuscleMimic tasks](../../myosuite/envs/myo/tasks/mimic/cpu.py) and [clip playback](../../myosuite/envs/myo/tasks/mimic/clip_env.py) expose reward/evaluation points for full-body motion.
- [MuscleMimic mjlab configuration](../../myosuite/envs/myo/backends/mjlab/mimic_mjlab_env.py) builds the GPU task rewards. A future online scorer must preserve the same feature and score contract as CPU evaluation.
- A separately trained leg prior could later use [CPU walking](../../myosuite/envs/myo/tasks/basic/leg/walk.py), [mjlab walking terms](../../myosuite/envs/myo/backends/mjlab/tasks/leg/walk_mdp.py) and [walking configuration](../../myosuite/envs/myo/backends/mjlab/tasks/leg/walk_env_cfg.py).

## Evaluation and training stages

1. **Calibrate features.** Replay matched poses, velocities and known single-foot loads through both coordinate systems. Check joint signs and offsets, vertical force, horizontal magnitude, right/left ordering and normalization. Save a small versioned fixture set.
2. **Score rollouts offline** with the scoring command in `training/biomech_prior/`. Load and freeze the network once. Compute one NLL per state, reducing feature dimensions without averaging across environments. Compare real and perturbed motion, check out-of-distribution behavior and profile CPU/GPU inference at representative batch sizes.
3. **Add an optional shared reward.** The reward term lives in MyoSuite and calls the scorer that `training/biomech_prior/` loads; keep the coefficient zero by default. Compare task performance and prior scores with and without the reward before selecting a weight. Include gait speed, falls, contacts and energy measures so an improved prior score cannot hide degraded task behavior.

For CPU and mjlab integration, follow the [cross-backend contract](cross-backend-contract.md). Shared term functions remain pure and use `accessor.array_module()`; the frozen learned scorer comes from `training/biomech_prior/` and is passed to the term. Use a CPU adapter and device-resident Torch inference with the same normalization and per-environment reduction. Avoid the upstream MATLAB wrapper in a GPU step: it averages the entire batch into one scalar and transfers results through NumPy.

Score fresh post-step forces. CPU steps already refresh derived fields; on mjlab use the existing [`sync_forward`](../../myosuite/envs/myo/backends/mjlab/tasks/mdp/terminations.py) ordering described in the [mjlab design guide](mjlab-design-guide.md). Keep inference vectorized, with no loops over environments, `.cpu()`, `.numpy()` or scalar synchronization in the GPU reward path. If future terms retain episode state, use `ManagerTermBase.reset(env_ids)`.

## Acceptance checks

An implementation PR should keep `training/biomech_prior/` independently installable and tested, with no new dependency in the root `pyproject.toml`. It should include matched-pose feature fixtures, CPU/GPU score agreement, tests for right/left contact swaps and invalid layouts, model/scaler compatibility checks, and throughput measurements at the intended environment count. Demonstrate that a zero reward coefficient leaves baseline behavior unchanged. Any intentional registered-task reward or distribution change needs a changelog entry and [checkpoint compatibility notes](../baseline_checkpoints.md).

Keep the offline scorer and subsequent online reward reviewable separately from upstream BiomechPriorVAE fixes. No new task registration is needed for offline scoring; any future task variants must use the existing CPU/mjlab registration primitives and shared `env_id` contract.
