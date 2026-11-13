# myoSuperMimic (experimental)

SuperMimic ([Liao et al., arXiv:2610.28480](https://arxiv.org/abs/2610.28480)) for MyoSuite's multi-clip full-body
tracking task `myoMimic-v0`: a learned world model for the body mechanics, with the muscle activation dynamics and the
task's observation, reward and failure rule computed analytically, and a policy trained by backpropagating through
imagined rollouts.

This is a self-contained training project (`training/supermimic/`, import name `supermimic`), not part of the
installable `myosuite` package; see `docs/wiki/training-guidelines.md`. MyoSuite never imports it.

## Attribution and licences

myoSuperMimic is a port, not an original MyoSuite method.

| Source | Licence | What this project takes from it |
|---|---|---|
| SuperMimic, [Liao et al., arXiv:2610.28480](https://arxiv.org/abs/2610.28480) | paper | the method: world model, analytic heads, policy learning through imagined rollouts, test-time refinement |
| [muscle-supermimic](https://github.com/kaibohex/muscle-supermimic) @ `8a6244b` | Apache-2.0 | most of `supermimic/` (runner, world model, replay, optimizers, collection, evaluation, refinement, kinematics, muscle geometry, activation dynamics, mirror augmentation) and `scripts/bench.py`; notice in `NOTICE-muscle-supermimic.txt`, licence in `LICENSE` |
| [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (Chong Zhang, ETH Zurich) | MIT | the SuperMimic runner, networks and buffer muscle-supermimic ported; `LICENSE-supermimic_vibed.txt` |
| [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) and [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) | MIT | the fast-recipe pieces muscle-supermimic ported (policy EMA, no-sync update loop, Muon split, policy learning-rate schedule, reset-statistics fixes); `LICENSE-BlinkMimic-InstantTrack.txt` |
| [XMimic](https://github.com/Sentient-X/xmimic) | MIT | ideas, reimplemented here: the systems options `kin_unbind`, `hoist_goal`, `muon_batched`, `gc_freeze` |
| [MuscleMimic](https://github.com/amathislab/musclemimic) | — | the task's MDP and its KIT train/test split (in MyoSuite's task, not here) |

`LICENSE-SuperMimic-Reproduced.txt` keeps the notice of [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), the reproduction MyoSuite's first SuperMimic adaptation used.
Every ported file says so in its header (`Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite`) and
names its upstream file in its docstring.

New MyoSuite code (header `Copyright (c) MyoSuite Authors`): `supermimic/env.py` (the task through `make_env`),
`supermimic/train.py`, `supermimic/finetune.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/finetune.py`, the
tests not marked as ported, and in the ported files the changes for MyoSuite's task: the task's pure term functions
evaluated on imagined states, myo_sim's body and its joint couplings, the action map and the checkpoint layout.

## What lives where

| | |
|---|---|
| MyoSuite task (`myosuite/envs/myo/tasks/mimic/tracking/`, twin in `backends/mjlab/tasks/mimic_tracking/`) | motion library, reset sampling, presets, metrics; its observation / reward / failure terms are pure functions in `myosuite/terms/mimic_track.py` |
| this project (`supermimic/`) | differentiable kinematics, muscle geometry and activation dynamics (`kinematics.py`, `muscle_geometry.py`, `activation.py`), the task's terms on its states (`heads.py`, `task.py`), mirror augmentation, world model, replay buffer, optimizers, collection, runner, evaluation, refinement, fine-tuning |

Each term of the task's env cfg carries its pure function (`func.pure`). The runner builds the task's `TrackState`
from a body state with its own kinematics and calls those functions, so the simulator and the imagination score
transitions with the same code. The env observes MuJoCo's state, the method its projection onto the joint couplings
(they differ by the couplings' constraint violation, about a millimetre).

## Install

```bash
pip install "myosuite[mjlab]"   # or: pip install -e ".[mjlab]" from a MyoSuite checkout
pip install "git+https://github.com/MyoHub/myosuite.git@v3.1.0#subdirectory=training/supermimic"
# or, from a checkout: pip install -e training/supermimic
```

The project needs the `myosuite` it was written against (`myoMimic-v0` and its pure terms); install both from the
same branch. Use a separate virtual environment if another training project needs other torch / mjlab versions.

The task reads the AMASS KIT motions retargeted onto the task model from a local cache, which
`scripts/mimic/retarget_amass.py` (in MyoSuite) fills from an AMASS download; `MYOSUITE_KIT_ROOT` points at another
folder of retargeted clips.

## The env

`supermimic.env.make_task_env(TaskEnvCfg(...), device, seed)` builds the task with
`make_env(EnvConfig("myoMimic-v0", ctrl_dt=..., task_kwargs={"preset": "supermimic", "overrides": {...}}), backend="mjlab")`.
`TaskEnvCfg`'s defaults are the task's `supermimic` preset (the registered default): 50 Hz control, no muscle
observations, return-based resets with the reset-statistics fix and 30 % of the resets before recent failure points.

## Train, evaluate, refine, fine-tune

```bash
# the benchmark: the fast recipe (fast_runner_cfg + TaskEnvCfg defaults) on myoMimic-v0's default clips, MuscleMimic's
# 972 KIT training motions; evaluate.py scores on the 108 held-out ones (supermimic/data/kit_test.txt) by default.
# Any --agent.* / --env.* field can be overridden.
python training/supermimic/scripts/train.py --agent.max-hours 0.5 --agent.seed 1
# held-out evaluation of every checkpoint of a run (eval.csv: normalized return, failed clips, MPJPE, joint errors)
python training/supermimic/scripts/evaluate.py --run-dir logs/rsl_rl/supermimic_fullbody/<run> --batch 8
# one checkpoint, with test-time action refinement through the world model (5 gradient steps per control step)
python training/supermimic/scripts/evaluate.py --checkpoint <run>/model_004000.pt --refine 5
# fine-tune a baseline on a few clips (a clip, a directory or a .txt list), evaluated on them before and after
python training/supermimic/scripts/finetune.py --checkpoint <run>/model_008000.pt --motion-file my_clips.txt \
    --iterations 2000 --out logs/ft_my_clips
# seconds per iteration, piece by piece
python training/supermimic/scripts/bench.py
```

The recipe: CUDA-graphed world-model and policy updates and graphed collection on a second stream, no host syncs,
50 Hz control, horizon 16, a semi-implicit world model with mirror augmentation, a 200-iteration world-model warm-up,
Beta exploration noise, Muon, policy EMA 0.995. The XMimic systems options give the same updates up to floating-point
rounding and take about 11 % less time per iteration on one RTX 5090: `kin_unbind` and `gc_freeze` are on by default,
`hoist_goal` and `muon_batched` in the fast recipe. On a CPU the graph options turn off. Not ported: MS-Human-700, DEP
exploration, evaluation-guided resets, BlinkMimic's probe graph.

Checkpoints (`model_*.pt`) hold the deployed (averaged) policy, the live policy, world model, critic, all normalizers,
optimizer states, the action map and the env settings they were trained with, so evaluation and fine-tuning match the
control rate and observation without flags. They also store the resolved task contract (every observation term in
order with its width, the actuators, the model and its version, the timing), and loading refuses a checkpoint trained
on another contract; set `accept_contract_mismatch` for an intentional transfer.

The graphed collector writes `ctrl`, `qpos`, `qvel` and `act` of the MuJoCo-Warp data and calls MuJoCo-Warp's `step`,
`forward` and masked `reset_data` itself, so that one collection step, resets included, is captured as one CUDA graph
(the manager env's step and reset cannot be). `tests/test_runner.py` checks it against `env.step` + `env.reset(ids)`.

## Option: PODS-style action targets (experimental, off by default)

After NablaMimic ([sailab-ai/NablaMimic](https://github.com/sailab-ai/NablaMimic), Apache-2.0: `--action-target`,
`--act-cot-clip`) and PODS ([Zamora Mora et al., ICML 2021](https://proceedings.mlr.press/v139/mora21a.html)),
re-implemented in PyTorch for excitation actions (`supermimic/pods.py`; no NablaMimic code is copied). With
`--agent.pods True` the policy no longer takes one optimizer step on the window gradient:

1. A zero offset `d_t` is added to every imagined action, after the in-graph clip below. It feeds the muscles, the
   reward and the next observation's history, so `dJ/dd_t = dJ/da_t` holds every path, including the policy's own
   feedback through later steps. The fast runner has no separate action penalty; all action terms are rewards.
2. Rows (step x trajectory) whose gradient is zero (ended trajectories), has a non-finite entry or whose policy
   input is non-finite get weight 0; non-finite entries are zeroed, so they never reach other rows.
3. With `--agent.act-grad-clip K` the used rows' gradients are clipped to K x one median over all of them.
4. Targets: `clamp(a_t + eta dJ/da_t, 0, 1)`, with `eta` so that the RMS step over the used rows is the current step
   (no used row: `eta = 0`, targets = actions).
5. `pods_iters` minibatch steps of the policy optimizer (Muon / Adam, the usual gradient clipping and non-finite
   gate) on the weighted squared error, `pods_batch` samples each drawn from the window's horizon x `pi_batch` rows.

| option | default | |
|---|---:|---|
| `--agent.pods` | `False` | PODS targets instead of the window-gradient step |
| `--agent.pods-step` | `0.03` | RMS target step, excitation units |
| `--agent.pods-step-floor` | `0.3` | the step follows the learning rate's cosine down to this x `pods-step` |
| `--agent.pods-iters` | `8` | regression steps per policy update |
| `--agent.pods-batch` | `16384` | samples per regression step |
| `--agent.pods-lr-mult` | `3.0` | policy learning rate x this while `pods` is on |
| `--agent.act-grad-clip` | `0` (off) | per-step clip of each trajectory's dJ/da_t to K x the median over rows with gradient (NablaMimic: 3); with `pods` also on the targets; works without `pods` too |

`train_log.csv` then has `act_clip_frac` / `act_clip_removed` (share of rows clipped in the graph and share of
their norm removed, per step), `pods_eta`, `pods_used` (share of rows used), `pods_nonfinite`, `pods_clip`,
`pods_reg_loss`, `pods_step` and `pods_dq_rms`, the RMS excitation change of the update on the last minibatch: compare
it to `pods_step` to tune `pods-lr-mult`. NablaMimic's ring buffer of past windows is not ported (one window, its
K = 1). Both options run inside the captured CUDA graph without host syncs. Untuned on muscles: no A/B yet.

## Results

Speed against muscle-supermimic on MuscleMimic's MyoFullBody (`musclemimic_models`), measured on 2026-11-08 with the
first version of this port (MyoSuite PR #634) before it moved to myo_sim's body. One RTX 5090 per stack and the same
package versions (torch 2.14.1, triton 3.8.0, mjlab 1.6.0, MuJoCo / MuJoCo-Warp 3.11, warp 1.18); the recipe without
muscle observations, 1024 environments, a 30-minute training budget, every 250-iteration checkpoint evaluated on the
108 held-out KIT clips. One seed per row (cold: seed 0, warm: seed 1), so differences of a minute or so are within
run-to-run noise.

| | muscle-supermimic @ 8a6244b | this port |
|---|---:|---:|
| seconds per iteration (training, median) | 0.130 | 0.128 |
| iterations in 30 min (warm compile caches) | 13883 | 13993 |
| compilation before the first policy update (cold caches) | 5.1 min | 5.1 min |
| first held-out normalized return >= 0.83, warm caches | 9.8 min | 10.8 min |
| first held-out normalized return >= 0.83, cold caches | 14.8 min | 15.8 min |
| all 108 held-out clips tracked to the end first, warm caches | 12.5 min | 16.7 min |
| after 30 min, warm caches: return / failed clips / MPJPE | 0.923 / 0 / 25.0 mm | 0.922 / 3 / 23.0 mm |

Test-time refinement (`evaluate.py --refine 5`) of the warm run's last checkpoint raised the held-out return from 0.922
to 0.950 (upstream: 0.923 to 0.946) and lowered the MPJPE from 23.0 to 18.6 mm (upstream: 25.0 to 19.9 mm), with no
failed clip. One fine-tuning example (`finetune.py`, 6 minutes including about 5 minutes of compilation) used that
checkpoint on the held-out clip it tracked worst (KIT 348 `turn_right02`): the return on that clip went from 0.47 to
0.93 and the MPJPE from 46 to 23 mm.

## Planned: DynSyn action groups

The policy acts through an action map from its outputs to the muscle excitations: identity today, so checkpoints hold
one output per muscle. DynSyn muscle groups ([He et al., ICML 2024](https://arxiv.org/abs/2407.11472)) would replace it
with one output per group; the checkpoint layout already records the map, so adding groups needs no new format. Not
implemented yet.

## Tests

```bash
cd training/supermimic && pytest    # the CUDA-graph tests skip without a GPU
```

The tests run on synthetic clips (`tests/conftest.py`), so they need no AMASS data. MyoSuite's CI does not run them.
