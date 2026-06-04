# Library Usage — Approved Feature Map

**Read this before writing any helper, wrapper, or utility.**

If something you need appears in the table below, use the library version. Writing a custom
re-implementation of a library feature is a defect, not a style choice.

---

## mjlab

| What you need | Use this | Do NOT write |
|---|---|---|
| Muscle actuator wrapping existing XML | `XmlActuatorCfg` (auto-detects type from spec, mjlab≥1.3) | Custom `*ActionCfg` + `*Action` pair |
| Observation term composition | `ObservationTermCfg` + `ObservationManager` | Manual obs dicts assembled in `__init__` |
| Reward term composition | `RewardTermCfg` + `RewardManager` | Inline reward computation outside term functions |
| Reset / event logic | `EventTermCfg` + `EventManager` | Custom reset methods on the env class |
| Body / tendon / actuator ID lookup | `entity.find_bodies()`, `find_tendons()`, `find_actuators()` | Custom `id(env)` caches |
| Scene entity access | `env.scene["entity_name"]` | Manual `mujoco.mj_name2id` at every call site |
| Movable prop / floating object | Separate `EntityCfg` per object + `write_root_state_to_sim` | Secondary freejoint inside shared XML + direct Warp write |
| Domain randomization (mass, friction) | `EventTermCfg(func=dr.body_mass / dr.geom_friction, ...)` | `entity.data.model.body_mass[env_ids, bid] = x` |
| Contact detection | `ContactSensorCfg` + `sensor.data.net_forces_w` | Python loop over `data.contact.geom` |
| Root body state write | `entity.write_root_state_to_sim(state_13)` | `data.qpos[i, :7] = ...` |
| Joint state write | `entity.write_joint_state_to_sim(pos, vel, joint_ids=...)` | `data.qpos[i, adr] = ...` |
| Actuator control write | `entity.write_ctrl_to_sim(ctrl, ctrl_ids=...)` | `data.ctrl[i, :] = ...` |
| Per-env mutable task state | `ManagerTermBase` instance attribute + `reset(env_ids)` | Module-level dict keyed by `id(env)` |
| Resolve env_ids (mjlab≥1.4) | `from mjlab.envs.mdp.events import resolve_env_ids` | Local `_normalize_env_ids` helper |

**See `docs/wiki/mjlab-design-guide.md` for the full canonical pattern reference and anti-pattern list.**

---

## mujoco

| What you need | Use this | Do NOT write |
|---|---|---|
| Quaternion multiply | `mujoco.mju_mulQuat` | Hand-rolled quat multiply |
| Quaternion negate | `mujoco.mju_negQuat` | `-quat` or manual conjugate |
| Quat → rotation matrix | `mujoco.mju_quat2Mat` | Custom Rodrigues formula |
| Contact force | `mujoco.mj_contactForce` | Manual contact struct field access |
| Body/site/joint ID | `mujoco.mj_name2id` | Hardcoded integer indices |
| Forward kinematics Jacobian | `mujoco.mj_jacBody`, `mj_jacSite` | Manual Jacobian computation |

**Unified dispatcher (preferred for all new code):**

```python
from myosuite.physics.quat import get_quat_ops
xp = accessor.array_module()   # numpy / jax.numpy / torch
qm = get_quat_ops(xp)
q  = qm.mul_quat(qa, qb)       # same call, right backend
```

Direct imports are fine when the backend is statically known:
- `from myosuite.physics import quat_math`        — NumPy (CPU, init/reset code)
- `from myosuite.physics import quat_math_jax`    — JAX (MJX step traces)
- `from myosuite.physics import quat_math_torch`  — Torch (mjlab step loop)

**Never import `quat_math` (numpy) inside a Torch or JAX execution path** — it
forces a device round-trip and will silently produce wrong dtypes.

---

## gymnasium

| What you need | Use this | Do NOT write |
|---|---|---|
| Observation normalization (single env) | `gymnasium.wrappers.NormalizeObservation` | Custom running-mean/std wrapper |
| Episode time limit | `gymnasium.wrappers.TimeLimit` | Manual step counter in env |
| Action clipping | `gymnasium.wrappers.ClipAction` | Manual `np.clip` in `step()` |
| Episode statistics logging | `gymnasium.wrappers.RecordEpisodeStatistics` | Manual `info["episode"]` dict assembly |

---

## stable_baselines3

| What you need | Use this | Do NOT write |
|---|---|---|
| Vectorized obs normalization | `VecNormalize` (already used in `agents/sb3_job_script.py`) | Inline `(obs - mean) / std` in policy |
| Vectorized env | `make_vec_env` | Manual `SubprocVecEnv` construction |
| Callback for checkpointing | `CheckpointCallback` | Custom `on_step` counter |

---

## numpy / torch / scipy

| What you need | Use this | Do NOT write |
|---|---|---|
| Sigmoid (torch path) | `torch.sigmoid` / `torch.nn.functional.sigmoid` | `1 / (1 + exp(-x))` |
| SiLU (torch path) | `torch.nn.functional.silu` | `x / (1 + exp(-x))` |
| Sigmoid (numpy path) | `scipy.special.expit` | `1 / (1 + np.exp(-x))` |
| Layer norm (torch) | `torch.nn.functional.layer_norm` | Manual Welford in forward pass |
| Batched distance | `xp.linalg.norm(batch, axis=-1)` | Python loop over rows |
| Cosine similarity | `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` or `torch.nn.functional.cosine_similarity` | Custom `calculate_cosine()` |

---

## pathlib (CLAUDE.md requirement)

Always use `pathlib.Path`. Never use `os.path`.

| Instead of | Use |
|---|---|
| `os.path.join(a, b)` | `Path(a) / b` |
| `os.path.exists(p)` | `Path(p).exists()` |
| `os.makedirs(p)` | `Path(p).mkdir(parents=True, exist_ok=True)` |
| `os.path.isfile(p)` | `Path(p).is_file()` |
| `os.path.dirname(p)` | `Path(p).parent` |
| `os.path.splitext(p)` | `Path(p).stem`, `Path(p).suffix` |
| `os.path.realpath(p)` | `Path(p).resolve()` |
| `glob.glob(pattern)` | `Path(dir).glob(pattern)` |

`os.environ`, `os.getpid`, `os.cpu_count` are fine — `os.path` is the banned subset.

---

## Known justified exceptions

These custom implementations exist for documented reasons and should **not** be replaced
without understanding the constraint:

| File | Class / function | Why it exists |
|---|---|---|
| `register_mjlab_tasks.py:122` | `MyoMuscleActivationAction` | `XmlMuscleActuatorCfg` missing in installed mjlab; tendon-name mismatch in `_find_targets` |
| `physics/quat_math.py` | entire module | numpy + JAX dual-backend support predating `mujoco.mju_*`; used in MJX reward terms |
| `integrations/musclemimic/running_stats.py` | `RunningMeanStd` | Checkpoint-portable; must match stats embedded in `.pt` files exactly |

When the upstream constraint is resolved (e.g. mjlab version bump), remove the exception
and replace with the library feature.
