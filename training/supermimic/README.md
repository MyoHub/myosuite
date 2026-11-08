# myoSuperMimic (experimental)

SuperMimic ([Liao et al., arXiv:2610.28480](https://arxiv.org/abs/2610.28480)) for MyoSuite's multi-clip
full-body tracking task `myoMimicFullbodyMulticlip-v0` (provisional id): a learned world model for the body
mechanics, with the muscle activation dynamics and the task's observation, reward and termination computed
analytically, and a policy trained by backpropagating through imagined rollouts.

This project is a port of [muscle-supermimic](https://github.com/kaibohex/muscle-supermimic) (Apache-2.0, commit
`8a6244b`), which ports its SuperMimic runner from [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
(MIT, Chong Zhang, ETH Zurich) and its fast recipe from [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) and
[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (MIT). The notices are in
`NOTICE-muscle-supermimic.txt` and the `LICENSE-*.txt` files here; every ported file names its source. The task
follows [MuscleMimic](https://github.com/amathislab/musclemimic) (its MDP and its KIT train/test split).
`LICENSE-SuperMimic-Reproduced.txt` keeps the notice of the reproduction MyoSuite's first SuperMimic adaptation used.

## What lives where

| | |
|---|---|
| MyoSuite (`myosuite/envs/myo/backends/mjlab/tasks/mimic_multiclip/`) | the task: body, motion library, MuscleMimic's observation / reward / termination terms, and the differentiable kinematics, muscle geometry and activation dynamics they are computed from |
| this project (`supermimic/`) | the method: mirror augmentation, world model, replay buffer, optimizers, collection, runner, evaluation, refinement, fine-tuning |

The runner evaluates the task's own term functions (each manager term carries its analytic implementation) on
imagined states, so the simulator and the imagination score transitions with the same code. MyoSuite never imports
this project.

## Install

```bash
pip install "myosuite[mjlab]"          # or: pip install ".[mjlab]" from a MyoSuite checkout
pip install training/supermimic        # this project, into the same environment
```

The task needs the retargeted KIT clips of MuscleMimic, a gated Hugging Face dataset (accept its terms first):

```python
from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip import download_kit_clips, kit_split
root = download_kit_clips()  # motion_root of kit_split("train") / kit_split("test")
```

## The env

`supermimic.env.make_task_env(TaskEnvCfg(...), device, seed)` builds the task with
`make_env(EnvConfig("myoMimicFullbodyMulticlip-v0", ctrl_dt=..., task_kwargs={...}), backend="mjlab")`.
`TaskEnvCfg`'s defaults are the fast recipe's env flags: 50 Hz control, no muscle observations, return-based resets
with the reset-statistics fix and 30 % of the resets before recent failure points. The registered defaults of the
task are MuscleMimic's (100 Hz, muscle observations, uniform resets).

## Planned: DynSyn action groups

The policy acts through an action map from its outputs to the muscle excitations: identity today, so checkpoints hold
one output per muscle. DynSyn muscle groups ([He et al., ICML 2024](https://arxiv.org/abs/2407.11472)) would replace it
with one output per group; the checkpoint layout already records the map, so adding groups needs no new format. Not
implemented yet.

## Tests

```bash
cd training/supermimic && pytest
```
