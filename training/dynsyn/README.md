# DynSyn muscle synergies for MyoSuite

PPO on a registered MyoSuite task (mjlab / RSL-RL) through DynSyn muscle groups. Muscles whose
lengths change together under random joint motion form one group. The policy commands one action
per group, plus an optional small correction per muscle whose bound grows during training.

## Attribution

DynSyn is the method of

> Kaibo He, Chenhui Zuo, Chengtian Ma, Yanan Sui. **DynSyn: Dynamical Synergistic Representation
> for Efficient Learning and Control in Overactuated Embodied Systems.** ICML 2024.
> [arXiv:2407.11472](https://arxiv.org/abs/2407.11472) ·
> reference implementation: [github.com/kaibohex/DynSyn](https://github.com/kaibohex/DynSyn)

This is not an original MyoSuite method. It is a MyoSuite re-implementation, written from the
paper. The authors' reference implementation was read but not copied: it has no licence file, so
none of its code is included here.

| What | Source |
|---|---|
| Grouping: muscle lengths under random joint velocities, segment similarity, k-medoids (`groups.py`) | The paper (Eq. 2). Centring each segment is a choice of this implementation, and so is the segment length (the paper gives none). The defaults of `synergy_groups` (500k steps, velocity bound 10) follow the reference implementation. |
| Correction form `1 + clip(kappa tanh(w), -c, c)` and `kappa = 0.1` (`myosuite.utils.action_map`) | The paper's Eq. 5-7, with the correction taken relative to the group's action as in the reference implementation. |
| Schedule of the bound, `min(max(k (t - a), 0), kappa)` | The paper's Eq. 6, read as `k (t - a)` (the paper prints `k t + a`; see the docstring of `weight_amplitude`). |
| k-medoids implementation, trainer (`train.py`), checkpoint record, mjlab action term, CPU wrapper | New MyoSuite code. |

The docstrings point out each place where the paper leaves something open and say what is used
there.

## Layout

- `myosuite_dynsyn/groups.py` learns the groups of a task on its CPU `MjModel`.
- `myosuite_dynsyn/train.py` is the trainer. It takes the task's env and runner configs from
  mjlab's task registry (where MyoSuite registers its twins). DynSyn adds one action term in front
  of the task's muscle term; the rest of the task is unchanged.
- MyoSuite holds the playback side: `myosuite.utils.action_map` (the `DynSyn` map, the CPU
  wrapper, checkpoint reading) and the mjlab action term
  `myosuite/envs/myo/backends/mjlab/tasks/mdp/action_map.py`. A DynSyn checkpoint plays with
  `scripts/eval_mjlab_policy.py` without this package.

## Install

```bash
pip install "myosuite[mjlab]"   # or, from a checkout: pip install -e ".[mjlab]"
pip install "git+https://github.com/MyoHub/myosuite.git@v3.1.0#subdirectory=training/dynsyn"
```

The project needs a MyoSuite that has `myosuite.utils.action_map`, so install MyoSuite from the
same branch first.

## Use

```bash
# 1. Groups (writes the JSON the trainer reads)
python -m myosuite_dynsyn.groups myoLegWalk-v0 --groups 40 --out myoLegWalk-v0_k40.json

# 2. Train. --max-amplitude 0 gives groups only (40 actions instead of 80).
python -m myosuite_dynsyn.train myoLegWalk-v0 --groups myoLegWalk-v0_k40.json \
    --env.scene.num-envs 1024 --start-steps 5e7 --ramp-steps 1e8 --max-amplitude 0.1

# 3. Continue: the checkpoint brings its own groups and schedule (also a W&B download,
#    also with another --env.scene.num-envs). --agent.resume picks the newest local run.
python -m myosuite_dynsyn.train myoLegWalk-v0 --resume logs/rsl_rl/myo_leg_walk/<run>/model_500.pt

# 4. Play (MyoSuite only, CPU or mjlab)
python scripts/eval_mjlab_policy.py myoLegWalk-v0 --checkpoint logs/rsl_rl/myo_leg_walk/<run>/model_500.pt
```

All `--env.*` / `--agent.*` options of mjlab's configs apply, and so does `--feature NAME[=JSON]`,
as in `scripts/train_mjlab.py`. The console scripts `dynsyn-groups` and `dynsyn-train` are the same
commands. The step counts of the schedule are environment steps summed over all envs.

## Tests

```bash
cd training/dynsyn && pytest
```

These tests are not part of MyoSuite's CI. MyoSuite's own `myosuite/tests/test_action_map.py`
covers the playback side.
