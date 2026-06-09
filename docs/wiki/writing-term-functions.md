# Writing Term Functions

Term functions are pure functions that read physics state through an `EnvAccessor` and run identically on CPU (numpy), MJX (jax.numpy), and mjlab (torch).

All term functions live in `myosuite/terms/`.

## Signatures

| Kind | Signature | Return must include |
|---|---|---|
| Obs | `(accessor, **kwargs) -> dict[str, Any]` | any named arrays |
| Reward | `(accessor, obs_dict, **kwargs) -> dict` | `dense`, `solved`, `done` |
| Termination | `(accessor, obs_dict, **kwargs) -> dict` | `done` |
| Action | `(accessor, action, **kwargs) -> Any` | processed action |

## The Only Rule: `accessor.array_module()`

Never import `numpy`, `jax.numpy`, or `torch` directly. Get the right array library at runtime:

```python
def distance_reward(accessor, obs_dict, *, threshold: float = 0.05, **kwargs):
    xp = accessor.array_module()
    dist = float(xp.linalg.norm(obs_dict["site_err"]))
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

```python
def pose_tracking_reward(
    accessor, obs_dict, *, target_key="target_qpos", weight=1.0, threshold=0.05, **kwargs
) -> dict:
    xp = accessor.array_module()
    dist = float(xp.linalg.norm(obs_dict[target_key] - accessor.joint_pos()))
    return {"pose_err": dist, "dense": -weight * dist, "solved": dist < threshold, "done": False}
```

Register in a `TaskSpec`:
```python
TaskSpec(reward_terms=[(pose_tracking_reward, {"weight": 2.0})])
```

## Testing

```python
# myosuite/tests/test_terms_cpu.py
def test_pose_tracking_reward(mock_cpu_accessor, mock_obs_dict):
    result = pose_tracking_reward(mock_cpu_accessor, mock_obs_dict)
    assert {"dense", "solved", "done"} <= result.keys()
    assert isinstance(result["dense"], float)
```

Add a parity test in `test_parity.py` to confirm CPU and MJX produce identical outputs (atol ≤ 1e-7).
