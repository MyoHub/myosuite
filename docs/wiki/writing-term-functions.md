# Writing Term Functions

Term functions are pure functions that read physics state through an `EnvAccessor` and run identically on CPU (numpy) and mjlab (torch), and on the experimental MJX path (jax.numpy).

All term functions live in `myosuite/terms/`.

## Signatures

| Kind | Signature | Returns |
|---|---|---|
| Obs | `(accessor, **kwargs)` | an array, `(n,)` or `(N, n)`; the env stores it under the term's key |
| Reward | `(accessor, task_state, **kwargs)` | a dict with `dense`, `solved` and `done`, plus any named components |
| Termination | plain arguments, e.g. `pelvis_fall_termination(pelvis_pos, threshold)` | `bool` |
| Action | `(accessor, action, **kwargs)` | the processed action |

`task_state` is the per-episode goal state of the env (for example `{"target_angles": ...}`). The `kwargs` are the
`extra` entries of the `ObsSpec` / `RewardSpec`, shared by all terms of that kind, so every term accepts and ignores
the keys it does not use (`**kwargs`).

## The Only Rule: `accessor.array_module()`

Never import `numpy`, `jax.numpy`, or `torch` directly. Get the right array library at runtime:

```python
def distance_reward(accessor, task_state, *, threshold: float = 0.05, **kwargs):
    xp = accessor.array_module()
    dist = float(xp.linalg.norm(task_state["target_angles"] - accessor.joint_pos()))
    return {"dense": -dist, "solved": dist < threshold, "done": False}
```

Access physics state via `accessor.*` only — never `mujoco.MjData` directly:

```python
qpos  = accessor.joint_pos()          # (nq,) or (N, nq)
qvel  = accessor.joint_vel()          # (nv,) or (N, nv)
sites = accessor.site_xpos(site_ids)  # (K, 3)
t     = accessor.time()
```

Full protocol: `myosuite/core/protocols.py`.

## Minimal Example

An observation term and a reward term, used in a data-driven `TaskConfig` (`ModularTaskEnv`):

```python
def joint_pos_scaled_obs(accessor, *, scale: float = 1.0, **kwargs):
    return scale * accessor.joint_pos()

def pose_tracking_reward(
    accessor, task_state, *, weight=1.0, pose_thd=0.35, **kwargs
) -> dict:
    xp = accessor.array_module()
    dist = float(xp.linalg.norm(task_state["target_angles"] - accessor.joint_pos()))
    return {"pose_err": dist, "dense": -weight * dist, "solved": dist < pose_thd, "done": False}
```

Select terms by name or pass the function. A name `"foo"` resolves to `foo_obs` / `foo_reward` in `myosuite/terms/`;
a function is labelled by its `__name__`, which is also the key of `RewardSpec.weights`:

```python
TaskConfig(
    obs=ObsSpec(keys=["joint_pos", joint_pos_scaled_obs]),
    reward=RewardSpec(
        terms=["act_reg", pose_tracking_reward],
        weights={"pose_tracking_reward": 2.0},    # multiplies that term's "dense"
        extra={"pose_thd": 0.2},                  # keyword arguments for every reward term
    ),
)
```

A hand-written `MyoGymnasiumEnv` calls the same helpers from `_get_obs_dict` and `get_reward_dict`.

## Testing

Test a term against a small fake accessor (`_FakeAccessor` in `myosuite/tests/test_terms_cpu.py`):

```python
def test_pose_reward_solved_when_at_target():
    acc = _FakeAccessor(nq=4)
    result = pose_reward(acc, {"target_angles": np.zeros(4)}, pose_thd=0.35)
    assert {"dense", "solved", "done"} <= result.keys()
    assert result["solved"]
```

Add a CPU parity test in `test_parity.py` (default `atol=1e-6`). For a GPU
task, also honour [cross-backend-contract.md](cross-backend-contract.md) so the mjlab half stays in lockstep.
