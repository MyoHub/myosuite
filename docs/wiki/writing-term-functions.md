# Writing Backend-Agnostic Term Functions

Term functions are the core building block of every MyoSuite task. A single term
function runs identically on all three compute backends — CPU (numpy), MJX
(jax.numpy), and mjlab (torch) — because it never imports an array library directly.

---

## What is a term function?

A term function is a **pure function** that reads physics state through an
`EnvAccessor` and returns a dict. There are four kinds:

| Kind | Signature | Returns |
|---|---|---|
| Obs term | `(accessor, **kwargs) -> dict[str, Any]` | Named observation arrays |
| Reward term | `(accessor, obs_dict, **kwargs) -> dict` | Must include `dense`, `solved`, `done` |
| Termination term | `(accessor, obs_dict, **kwargs) -> dict` | Must include `done` |
| Action term | `(accessor, action, **kwargs) -> Any` | Processed action |

All term functions live in `myosuite/terms/`.

---

## The golden rule: `accessor.array_module()`

Never import `numpy`, `jax.numpy`, or `torch` directly in a term function.
Use `accessor.array_module()` to get the right array library at runtime:

```python
# ✅ Correct — backend-agnostic
def distance_reward(accessor, obs_dict, *, threshold: float = 0.05, **kwargs):
    xp = accessor.array_module()          # numpy, jnp, or torch depending on backend
    dist = xp.linalg.norm(obs_dict["site_err"])
    solved = bool(dist < threshold)
    return {
        "dense": float(-dist),
        "solved": solved,
        "done": False,
    }

# ❌ Wrong — breaks on MJX and mjlab
import numpy as np
def distance_reward(accessor, obs_dict, **kwargs):
    dist = np.linalg.norm(obs_dict["site_err"])   # crashes on JAX arrays
    ...
```

---

## Accessing physics state

Use the `EnvAccessor` protocol methods — never access `mujoco.MjData` directly:

```python
def my_obs_term(accessor, *, site_ids: list[int], **kwargs):
    xp = accessor.array_module()
    qpos   = accessor.joint_pos()          # (nq,) or (N, nq) batched
    qvel   = accessor.joint_vel()          # (nv,) or (N, nv) batched
    sites  = accessor.site_xpos(site_ids)  # (len(site_ids), 3)
    t      = accessor.time()               # scalar
    return {"qpos": qpos, "qvel": qvel, "sites": sites}
```

Full protocol: `myosuite/core/protocols.py` — `EnvAccessor`.

---

## Worked example: a minimal reward term

```python
# myosuite/terms/base_reward.py  (add alongside existing functions)

def pose_tracking_reward(
    accessor,
    obs_dict,
    *,
    target_key: str = "target_qpos",
    weight: float = 1.0,
    bonus_threshold: float = 0.05,
    **kwargs,
) -> dict:
    """Penalise distance from a target joint configuration.

    Args:
        accessor: Environment state accessor.
        obs_dict: Current observation dict from the obs term.
        target_key: Key in obs_dict holding the target configuration.
        weight: Scale applied to the distance penalty.
        bonus_threshold: Distance below which a completion bonus is given.

    Returns:
        Reward dict with keys ``dense``, ``solved``, ``done``,
        plus ``"pose_err"`` for logging.
    """
    xp = accessor.array_module()
    err  = obs_dict[target_key] - accessor.joint_pos()
    dist = float(xp.linalg.norm(err))
    return {
        "pose_err": dist,
        "dense":   -weight * dist + float(dist < bonus_threshold),
        "solved":  dist < bonus_threshold,
        "done":    False,
    }
```

Register it in your `TaskSpec`:

```python
# myosuite/envs/myo/tasks/basic/arm/my_task_spec.py
from myosuite.terms.base_reward import pose_tracking_reward

MY_TASK = TaskSpec(
    model_path=...,
    obs_terms=[...],
    reward_terms=[
        (pose_tracking_reward, {"weight": 2.0, "bonus_threshold": 0.03}),
    ],
)
```

---

## Dos and Don'ts

| ✅ Do | ❌ Don't |
|---|---|
| `xp = accessor.array_module()` | `import numpy as np` in a term |
| Read state via `accessor.*` | Access `mujoco.MjData` / `mjx_data` directly |
| Return scalars via `float(...)` | Return JAX/torch zero-dim tensors as reward |
| Keep functions pure (no side effects) | Store mutable state in a closure |
| Default all hyperparams as keyword args | Hardcode magic numbers in the function body |

---

## Testing a new term function

Add a parametrized test in `myosuite/tests/test_terms_cpu.py`:

```python
def test_pose_tracking_reward_cpu(mock_cpu_accessor, mock_obs_dict):
    result = pose_tracking_reward(mock_cpu_accessor, mock_obs_dict, weight=1.0)
    assert "dense" in result and "solved" in result and "done" in result
    assert isinstance(result["dense"], float)
```

Then add a parity test in `myosuite/tests/test_parity.py` to ensure CPU and MJX
produce identical outputs (atol ≤ 1e-7).

---

## See also

- `myosuite/core/protocols.py` — full `EnvAccessor` protocol definition
- `myosuite/terms/base_reward.py` — existing reward terms to use as patterns
- `docs/wiki/engineering-standards.md` — full architecture rules and checklist
