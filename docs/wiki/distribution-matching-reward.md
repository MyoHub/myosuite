# Distribution-Matching Reward

A reward that is largest when the distribution of a movement feature in the simulation equals a given distribution,
for example a human one, so that the policy reproduces the spread of the movement. Imitation by tracking
(`mimic_reward.py`) reproduces one reference movement.

## The reward

For a feature `x` (a joint angle, a head velocity), a target density `p*` and the policy's own density `q`:

```
r = clip(log p*(x) − log q(x), −c, c)
```

Under the policy's distribution the mean of `r` is `−KL(q ‖ p*)`, which is largest at `q = p*` (state marginal
matching; Lee et al. 2019, arXiv:1906.05274). `q` is estimated online from recent frames.

Without `q` the reward is `log p*(x)`. Its return is largest when every frame sits at the mode of `p*`, so the
policy learns the most likely movement and its spread collapses. Use it only to keep a feature plausible.

## Parts

| Part | Where |
|---|---|
| Target `p*`: `GaussianMixtureSpec.from_gaussian`, `.from_components`, `.from_samples` (scikit-learn fit, BIC picks the components), `.stack` (one mixture per context), `.save` / `.load` | `myosuite/terms/distribution_reward.py` |
| `mixture_log_density`, `density_ratio`, `distribution_matching_reward`, `DistributionMatchingCfg` | `myosuite/terms/distribution_reward.py` |
| Policy density `q`: `KernelDensity`, `RunningGaussian` | `myosuite/utils/density_estimator.py` |
| CPU: `DistributionMatchingWrapper(env, cfg, weight)` | `myosuite/envs/wrappers.py` |
| Modular CPU tasks (experimental): `reward.terms=[distribution_matching_reward]`, `reward.extra={"distribution": cfg}` | `myosuite/envs/modular_env.py` |
| mjlab: `mdp.DistributionMatchingReward` in a `RewardTermCfg` | `myosuite/envs/myo/backends/mjlab/tasks/mdp/rewards.py` |

```python
target = GaussianMixtureSpec.from_samples(recorded_elbow_angles)       # or .from_gaussian(mean, cov)
cfg = DistributionMatchingCfg(
    feature=lambda accessor, task_state: accessor.joint_pos()[..., :1],  # (d,) on the CPU, (N, d) on mjlab
    target=target,
    estimator=KernelDensity(1, capacity=4096, bandwidth_scale=0.3),
)
env = DistributionMatchingWrapper(make_env("myoElbowPose1D6MRandom-v0"), cfg, weight=0.5)       # CPU

env_cfg.rewards["distribution"] = RewardTermCfg(                                                # mjlab
    func=mdp.DistributionMatchingReward, weight=0.5, params={"distribution": cfg, "entity_name": "robot"}
)
# a feature of site frames or velocities (derived quantities) needs the sync as the first termination term
env_cfg.terminations = {mdp.SYNC_TERM: TerminationTermCfg(func=mdp.sync_forward), **env_cfg.terminations}
```

The feature reads the state through the accessor (joints, muscle activations, and site positions, orientations
`site_xmat` and angular velocities `site_ang_vel`) and the task state, for a feature relative to a target. The CPU
wrapper passes the env's `_task_state`; on mjlab the term's `task_state` parameter builds it from the env, as the
task's own terms do. A head feature, for example:

```python
def head_motion(accessor, task_state):
    xp = accessor.array_module()
    head = accessor.site_id("head")
    gaze = accessor.site_xmat(head)[..., :, 0]                 # the head site's x axis, world frame
    to_target = task_state["target_pos"] - accessor.site_xpos(head)
    cos = (gaze * to_target).sum(-1) / (to_target * to_target).sum(-1) ** 0.5
    return xp.concatenate([accessor.site_ang_vel(head), cos[..., None]], -1)
```

Write the feature with operations numpy and torch share (`sum(-1)`, `xp.concatenate(..., -1)`), so both backends
compute the same value. `test_distribution_reward_backends.py` replays a CPU rollout of the elbow state by state
into the mjlab twin, with a feature that reads the task's target and a site's orientation and angular velocity:
the two rewards agree within 4.1e-4, the difference of float32 from float64.

## Rules

- **The estimator is training state.** It learns across episodes while not frozen, so the reward depends on earlier
  steps. Call `estimator.freeze()` (on each one, with one per context) for evaluation and tests; the reward is then
  the same after every `reset(seed)`. Save it with `state_dict()` beside the policy; `load_state_dict()` restores
  what was learnt and leaves the estimator's settings and `frozen` as they are.
- **One estimator per context.** A per-context target (`GaussianMixtureSpec.stack`, with `context_key`) takes a
  list of estimators, one per context, each learning from its own context's frames; selecting those frames syncs
  the device once per context per step. One pooled estimator would pay `log p*(x|c) - log q(x)`, whose mean adds
  `I(x; c)` to `-KL` and so pays for separating the contexts further than the target does; the term refuses it.
- **One estimator per process on the CPU.** In an `AsyncVectorEnv` each worker holds its own copy of the wrapper's
  estimator, which learns from that worker's envs only; mjlab's term learns one estimator from all envs. Use a
  `SyncVectorEnv` (one process) when the CPU and mjlab rewards must match.
- **mjlab scales rewards by the step by default.** It multiplies each term by `weight * step_dt` unless the env
  cfg sets `scale_rewards_by_dt=False`, as MyoSuite's twins do; only then do the CPU wrapper and the mjlab term pay
  the same for the same weight. On a cfg with mjlab's default, divide the mjlab weight by `step_dt`.
- **mjlab copies the env cfg.** The estimator that learns is the env's copy:
  `env.reward_manager.get_term_cfg(name).params["distribution"].estimator`.
- **Sync derived quantities on mjlab.** mjlab scores rewards before its `forward()`, so site positions, orientations
  and velocities lag `qpos` by one physics step. A feature that reads them needs `mdp.sync_forward` as the first
  termination term (`mdp.SYNC_TERM`, as in the leg twins); joint positions and velocities need none. The CPU env
  runs `mj_forward` after every step.
- **Prefer `KernelDensity`.** `RunningGaussian` enforces only the mean and covariance, so the shape of the
  distribution is free.
- **Size the memory for the number of envs.** One step adds one sample per env. `KernelDensity` keeps the latest
  `capacity` samples, so `capacity / num_envs` steps (4 with the default 4096 and 1024 envs). With more envs than
  `capacity` it keeps an evenly spaced part of one step, shifted by one env per step so that every env is learnt.
  `RunningGaussian`'s `decay` is per sample: a step of `n` envs keeps `decay ** n` of the earlier weight, so choose
  `decay = 1 - 1 / (steps * n)`.
- **Bandwidth.** Scott's rule uses the overall SD. On a multimodal target it widens `q`, and the policy narrows to
  compensate; set `bandwidth_scale` below 1.
- **Read out an average of policies.** The policy and its estimate chase each other, so the last policy is a
  snapshot of an oscillation. Evaluate the mixture of several snapshots of the last part of training.
- **A pooled target can be met across episodes.** A bimodal target can be matched by episodes that each keep one
  mode. Condition the target on context (`GaussianMixtureSpec.stack`, with the context index in the task state
  under `context_key`) when the distribution within an episode matters.
- **Score the frames the target describes.** A target fitted to some frames only (for example those inside a
  trial) needs `mask_key`: the task state holds a boolean `(N,)` under it, the other frames are paid 0 and `q`
  does not learn from them. Selecting the rows syncs the device once per step.
- **Low dimension.** Mixtures and kernel densities degrade beyond about 5 to 10 features.
