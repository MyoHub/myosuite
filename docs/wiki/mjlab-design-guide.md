# mjlab Design Guide

**Read this before writing any mjlab backend code.**

---

## Golden Rules

1. **One floating body = one entity.** Never embed secondary freejoints in a shared XML. Each floating body gets its own `EntityCfg` + `write_root_state_to_sim`.
2. **Never write to `entity.data.data.*` or `env.sim.wp_data.*` directly.** All state writes go through the Entity write API.
3. **No Python loops over environments in step-rate functions.** Obs, reward, and event functions must be fully vectorized.
4. **No module-level mutable state.** Per-env state belongs on `ManagerTermBase` instance attributes, not globals keyed by `id(env)`.
5. **Domain randomization through `dr.*` event functions.** Direct writes to `data.model.body_mass` etc. corrupt multi-env correctness.

---

## State Writes — Entity Write API

| What to write | Use | Never |
|---|---|---|
| Root pose + vel | `entity.write_root_state_to_sim(state_13)` | `data.qpos[i, :7] = ...` |
| Root pose only | `entity.write_root_link_pose_to_sim(pose_7)` | `data.qpos[i, :7] = ...` |
| Hinge / slide joints | `entity.write_joint_state_to_sim(pos, vel, joint_ids=...)` | `data.qpos[i, adr] = ...` |
| Joint position only | `entity.write_joint_position_to_sim(pos, joint_ids=...)` | `data.qpos[i, adr] = ...` |
| Actuator control | `entity.write_ctrl_to_sim(ctrl, ctrl_ids=...)` | `data.ctrl[i, :] = ...` |
| Mocap body pose | `entity.write_mocap_pose_to_sim(pose_7)` | `data.mocap_pos[i, :] = ...` |
| External wrench | `entity.write_external_wrench_to_sim(forces, torques)` | `data.xfrc_applied[i, :] = ...` |

`torch.cuda.synchronize()` guards around direct Warp writes are a workaround, not a fix. Migrate to the Entity API and remove them.

---

## State Reads — Entity Data API

Use `entity.data.*` (stable, sliced, versioned). Never use `entity.data.data.*` (raw Warp struct, internal implementation detail).

| What to read | Use | Avoid |
|---|---|---|
| Joint positions | `entity.data.joint_pos` — `(N, nj)` | `entity.data.data.qpos[:, 2:]` |
| Joint velocities | `entity.data.joint_vel` — `(N, nj)` | `entity.data.data.qvel` |
| Root position | `entity.data.root_link_pos_w` | `entity.data.data.xpos[:, 0, :]` |
| Root orientation | `entity.data.root_link_quat_w` | `entity.data.data.xquat[:, 0, :]` |
| Projected gravity | `entity.data.projected_gravity_b` | manual `rotate(gravity, quat)` |

**Accepted exceptions** (no `entity.data` API equivalent — add a comment):
- `entity.data.data.act` — muscle activation state
- `entity.data.data.actuator_length` / `actuator_velocity` / `actuator_force`
- `entity.data.data.cvel` — body CoM velocity

---

## Domain Randomization

Always use `dr.*` event functions. Direct writes to `data.model.body_mass` etc. write to a shared `(1, nbody)` array — all environments get the same value (silent correctness bug at `num_envs > 1`).

```python
EventTermCfg(
    func=dr.body_mass,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("pingpong"),
            "mass_distribution_params": (0.0024, 0.0004),
            "operation": "add"},
)
```

---

## Term Functions — Vectorization and Device

- Register everything via `ObservationTermCfg`, `RewardTermCfg`, `EventTermCfg`. Never override `_get_observations()` or assemble obs dicts manually.
- All tensor ops must process all `num_envs` simultaneously — no Python loops over environments.
- No `.detach().cpu()` inside obs/reward/event functions. Every tensor stays on `env.device` until the manager returns.

---

## Action Terms

Custom action classes must subclass `ActionTermCfg` + `ActionTerm` so `ActionManager` can type-check, call `reset()`, and access `action_dim`. A plain class is not recognized.

---

## Per-env State and Caches

Instance attributes on `ManagerTermBase` — not module globals keyed by `id(env)` (memory leak; CPython reuses IDs after GC).

```python
class SaberTaskLogic(ManagerTermBase):
    def __init__(self, cfg, env):
        self._active_target = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids):
        self._active_target[env_ids] = 0
```

For static data derived from XML (same across all instances of a model), use `@functools.lru_cache(maxsize=None)` keyed on the XML path, not `id(env)`.

---

## Contact Detection

Use `ContactSensorCfg` + `sensor.data.net_forces_w`. Iterating over `data.contact.geom` with Python loops causes thousands of CUDA synchronizations per step at scale.

---

## Sim Time

Never write to `data.time` directly — it corrupts CUDA graph state. Multi-rally logic must use episode termination + reset via `TerminationTermCfg` + `EventTermCfg(mode="reset")`, with rally count tracked in a `ManagerTermBase`.

---

## Cross-entity Sensors

If a sensor references a site in one entity and a body in another, strip it from both entity specs and re-add it via `SceneCfg.spec_fn`. Resolve addresses at runtime from `env.sim.mj_model.sensor_adr`.

---

## Anti-Pattern Reference

| ID | Anti-pattern | Severity |
|---|---|---|
| AP-1 | Custom action class not subclassing `ActionTermCfg`/`ActionTerm` | Medium |
| AP-2 | Module-level mutable globals keyed by `id(env)` | High |
| AP-3 | `entity.data.data.*` reads (raw Warp layer) | Medium |
| AP-4 | Direct Warp buffer writes bypassing Entity write API | High |
| AP-5 | Python loops over environments in step-rate functions | High |
| AP-6 | `.detach().cpu()` in obs/reward hot paths | Medium |
| AP-7 | Hand-rolled contact detection instead of `ContactSensor` | Medium |
| AP-8 | Direct `data.model.body_mass` / `geom_friction` writes | **Critical** |
| AP-9 | Module-level `SceneEntityCfg` global mutated at config time | High |
| AP-11 | Raw `qpos[:, 2:]` address arithmetic instead of `entity.data.joint_pos` | Medium |
| AP-12 | `data.time[i] = 0.0` direct sim-time mutation | High |
| AP-14 | `id(env)`-keyed cache — leak on GC, stale IDs on reuse | High |
| AP-15 | Silent `except: pass` in task registration | Low |

---

## Justified Exceptions

| Task | Deviation | Reason |
|---|---|---|
| Saber (`saber_mjlab_env.py`) | Sync-guarded Warp writes for mocap target pool | `spec.delete()` corrupts mocap parent chain; XML refactor in #42 |
| TableTennis | Closure-based terms; AP-8 DR for paddle mass / ball friction | DR migration needs `dr.*` support for split-entity scenes |
| Elbow | `MyoMuscleActivationAction` instead of `XmlActuatorCfg` | `XmlMuscleActuatorCfg` removed in mjlab v1.4 |
