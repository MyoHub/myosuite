# Library Usage — Approved Feature Map

Use the library version. Writing a custom re-implementation of a library feature is a defect, not a style choice.

---

## mjlab

| What you need | Use | Do NOT write |
|---|---|---|
| Muscle actuator from XML | `XmlActuatorCfg` | Custom `*ActionCfg` + `*Action` pair |
| Observation composition | `ObservationTermCfg` + `ObservationManager` | Manual obs dicts in `__init__` |
| Reward composition | `RewardTermCfg` + `RewardManager` | Inline reward outside term functions |
| Reset / event logic | `EventTermCfg` + `EventManager` | Custom reset methods on env class |
| Body / tendon / actuator ID lookup | `entity.find_bodies()`, `find_tendons()`, `find_actuators()` | `id(env)` caches |
| Scene entity access | `env.scene["name"]` | Manual `mujoco.mj_name2id` per call |
| Movable prop | Separate `EntityCfg` + `write_root_state_to_sim` | Secondary freejoint in shared XML |
| Domain randomization | `EventTermCfg(func=dr.body_mass / dr.geom_friction, ...)` | Direct `data.model.body_mass[...] = x` |
| Contact detection | `ContactSensorCfg` + `sensor.data.net_forces_w` | Python loop over `data.contact.geom` |
| Root state write | `entity.write_root_state_to_sim(state_13)` | `data.qpos[i, :7] = ...` |
| Joint state write | `entity.write_joint_state_to_sim(pos, vel, joint_ids=...)` | `data.qpos[i, adr] = ...` |
| Ctrl write | `entity.write_ctrl_to_sim(ctrl, ctrl_ids=...)` | `data.ctrl[i, :] = ...` |
| Per-env task state | `ManagerTermBase` instance attribute + `reset(env_ids)` | Module-level dict keyed by `id(env)` |
| Resolve env_ids (mjlab≥1.4) | `from mjlab.envs.mdp.events import resolve_env_ids` | Local `_normalize_env_ids` helper |

See `docs/wiki/mjlab-design-guide.md` for the full pattern reference.

---

## mujoco

| What you need | Use | Do NOT write |
|---|---|---|
| Quaternion multiply | `mujoco.mju_mulQuat` | Hand-rolled quat multiply |
| Quat → rotation matrix | `mujoco.mju_quat2Mat` | Custom Rodrigues formula |
| Contact force | `mujoco.mj_contactForce` | Manual contact struct field access |
| Body/site/joint ID | `mujoco.mj_name2id` | Hardcoded integer indices |
| Jacobian | `mujoco.mj_jacBody`, `mj_jacSite` | Manual Jacobian computation |

**Unified dispatcher (preferred for new code):**

```python
from myosuite.physics.quat import get_quat_ops
xp = accessor.array_module()
qm = get_quat_ops(xp)
q  = qm.mul_quat(qa, qb)   # correct backend automatically
```

Direct imports when backend is statically known:
- `from myosuite.physics import quat_math` — NumPy (CPU, init/reset)
- `from myosuite.physics import quat_math_jax` — JAX (MJX step traces)
- `from myosuite.physics import quat_math_torch` — Torch (mjlab step loop)

Never import `quat_math` (numpy) inside a Torch or JAX execution path.

---

## gymnasium

| What you need | Use | Do NOT write |
|---|---|---|
| Obs normalization | `gymnasium.wrappers.NormalizeObservation` | Custom running-mean/std wrapper |
| Episode time limit | `gymnasium.wrappers.TimeLimit` | Manual step counter |
| Action clipping | `gymnasium.wrappers.ClipAction` | Manual `np.clip` in `step()` |
| Episode statistics | `gymnasium.wrappers.RecordEpisodeStatistics` | Manual `info["episode"]` assembly |

---

## numpy / torch / scipy

| What you need | Use | Do NOT write |
|---|---|---|
| Sigmoid (torch) | `torch.sigmoid` | `1 / (1 + exp(-x))` |
| Sigmoid (numpy) | `scipy.special.expit` | `1 / (1 + np.exp(-x))` |
| Layer norm | `torch.nn.functional.layer_norm` | Manual Welford in forward pass |
| Batched distance | `xp.linalg.norm(batch, axis=-1)` | Python loop over rows |

---

## pathlib

Always `pathlib.Path`. Never `os.path`.

| Instead of | Use |
|---|---|
| `os.path.join(a, b)` | `Path(a) / b` |
| `os.path.exists(p)` | `Path(p).exists()` |
| `os.makedirs(p)` | `Path(p).mkdir(parents=True, exist_ok=True)` |
| `os.path.dirname(p)` | `Path(p).parent` |
| `glob.glob(pattern)` | `Path(dir).glob(pattern)` |

`os.environ`, `os.getpid`, `os.cpu_count` are fine — `os.path` is the banned subset.

---

## Justified Exceptions

These custom implementations exist for documented reasons — do not replace without understanding the constraint:

| File | What | Why |
|---|---|---|
| `register_mjlab_tasks.py` | `MyoMuscleActivationAction` | Tendon-name mismatch in `_find_targets`; `XmlMuscleActuatorCfg` insufficient |
| `physics/quat_math.py` | entire module | numpy + JAX dual-backend predating `mujoco.mju_*`; used in MJX reward terms |
| `integrations/musclemimic/running_stats.py` | `RunningMeanStd` | Must match stats embedded in `.pt` checkpoints exactly |
