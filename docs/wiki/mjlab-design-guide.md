# mjlab Design Guide

**Read this before writing any mjlab backend code.**

This document defines the canonical patterns for every architectural decision in
myosuite4's mjlab backend. Each rule is grounded in a real anti-pattern found in
the codebase and the mjlab framework design.

---

## The Golden Rules

1. **One movable object = one entity.** Never embed secondary freejoints inside a
   shared XML spec. Each floating body gets its own `EntityCfg` and its own
   `write_root_state_to_sim`.

2. **Never write to `entity.data.data.*` or `env.sim.wp_data.*` directly.** All
   state writes go through the Entity write API.

3. **No Python loops over environments in step-rate functions.** Every obs, reward,
   and event function must be fully vectorized over the batch dimension.

4. **No module-level mutable state.** Per-env caches and task state belong on
   `ManagerTermBase` instance attributes, not in module globals keyed by `id(env)`.

5. **Domain randomization goes through `dr.*` event functions.** Direct writes to
   `data.model.body_mass`, `data.model.geom_friction`, etc. silently corrupt
   multi-env correctness.

---

## 1. Scene Entity Structure

### Rule: one floating body → one entity

Every body that can move freely in world space must be its own top-level entity
in `SceneCfg.entities`. This is mjlab's own pattern for manipulation objects
(see `mjlab/tasks/manipulation/`).

```python
# CORRECT
scene_cfg = SceneCfg(entities={
    "body":        EntityCfg(spec_fn=body_spec),
    "left_saber":  EntityCfg(spec_fn=left_saber_spec),
    "right_saber": EntityCfg(spec_fn=right_saber_spec),
    "pingpong":    EntityCfg(spec_fn=ball_spec),
})

# At reset:
env.scene["left_saber"].write_root_state_to_sim(state_13, env_ids=env_ids)

# WRONG — secondary freejoint inside shared spec
# Forces direct Warp writes + torch.cuda.synchronize() guards → CUDA error 700 risk
```

Each separate entity automatically gets `write_root_state_to_sim`,
`write_joint_state_to_sim`, and all other Entity API methods scoped to its own
joints and bodies.

**Authoring requirement:** each entity's spec must be self-contained from the
start — no cross-body dependencies to other entities (sensors attached to
foreign bodies, mocap targets parented relative to another entity's worldbody,
weld constraints referencing bodies in a different spec). Trying to split a
monolithic XML after the fact using `spec.delete()` will corrupt parent-child
relationships and fail MuJoCo compilation.

**Cross-entity sensors — use `SceneCfg.spec_fn`:** if you need a sensor
(velocimeter, rangefinder, etc.) that references a site in one entity and a
body in another, strip the sensor from both entity specs and re-add it via
`SceneCfg.spec_fn`.  The callback receives the fully-attached combined spec
with all entity names prefixed as `{entity_name}/{element_name}`.

```python
def _my_scene_spec_fn(spec: mujoco.MjSpec) -> None:
    # Sites are reachable under "{entity_name}/{site_name}" after attachment.
    s = spec.add_sensor()
    s.name = "ball_vel_sensor"
    s.type = mujoco.mjtSensor.mjSENS_VELOCIMETER
    s.objtype = mujoco.mjtObj.mjOBJ_SITE
    s.objname = "ball_entity/ball_site"          # lives in ball entity

    s2 = spec.add_sensor()
    s2.name = "paddle_vel_sensor"
    s2.type = mujoco.mjtSensor.mjSENS_VELOCIMETER
    s2.objtype = mujoco.mjtObj.mjOBJ_SITE
    s2.objname = "robot_entity/paddle_site"      # lives in robot entity

scene_cfg = SceneCfg(
    entities={"robot_entity": robot_cfg, "ball_entity": ball_cfg},
    spec_fn=_my_scene_spec_fn,
)
```

Sensor addresses in the combined model differ from any single entity's model.
Resolve them at runtime from `env.sim.mj_model`:

```python
mj_m = env.sim.mj_model
sid = mj_m.sensor("ball_vel_sensor").id
adr = int(mj_m.sensor_adr[sid])
dim = int(mj_m.sensor_dim[sid])
ball_vel = entity.data.data.sensordata[:, adr : adr + dim]
```

Similarly, after splitting an entity, scene-level site/body IDs returned by
`entity.indexing.site_ids` / `entity.indexing.body_ids` are the correct indices
for `entity.data.data.site_xpos` / `.xpos`.  `entity.find_sites()` returns
entity-local indices (into `entity.site_names`) — convert them:

```python
local_ids, _ = ent.find_sites(["paddle"])
scene_site_id = int(ent.indexing.site_ids[local_ids[0]])
paddle_pos = entity.data.data.site_xpos[:, scene_site_id, :]  # correct
```

**Justified exception — saber task:** the saber XML has mocap target pool bodies
(`saber_target_*`) that cross-reference within a single monolithic spec.
`spec.delete()` breaks those references. Saber retains sync-guarded direct Warp
writes until the XML assets are re-authored as separate files (tracked in issue
#42). New tasks must not repeat this pattern.

---

## 2. State Writes — Entity Write API

### Rule: never write directly to Warp buffers

Direct writes to `entity.data.data.qpos`, `entity.data.data.qvel`,
`env.sim.wp_data.*` bypass mjlab's CUDA graph invalidation logic. This causes
**CUDA error 700 (illegal memory access)** under certain allocator states.

| What to write | Use this | Never this |
|---|---|---|
| Root floating body pose + vel | `entity.write_root_state_to_sim(state_13)` | `data.qpos[i, :7] = ...` |
| Root pose only | `entity.write_root_link_pose_to_sim(pose_7)` | `data.qpos[i, :7] = ...` |
| Hinge / slide joints | `entity.write_joint_state_to_sim(pos, vel, joint_ids=...)` | `data.qpos[i, adr] = ...` |
| Hinge position only | `entity.write_joint_position_to_sim(pos, joint_ids=...)` | `data.qpos[i, adr] = ...` |
| Actuator control | `entity.write_ctrl_to_sim(ctrl, ctrl_ids=...)` | `data.ctrl[i, :] = ...` |
| Mocap body pose | `entity.write_mocap_pose_to_sim(pose_7)` | `data.mocap_pos[i, :] = ...` |
| External wrench | `entity.write_external_wrench_to_sim(forces, torques)` | `data.xfrc_applied[i, :] = ...` |

`torch.cuda.synchronize()` guards around direct Warp writes are a temporary
workaround, not a solution. Migrate to the Entity API and remove the guards.

---

## 3. State Reads — Entity Data API

### Rule: use `entity.data.*`, not `entity.data.data.*`

`entity.data.data` is the raw Warp `SimData` struct — an internal implementation
detail that changes across mjlab versions and backends. `entity.data` exposes a
stable, sliced, type-annotated view.

| What to read | Use this | Avoid this |
|---|---|---|
| Joint positions | `entity.data.joint_pos` — `(N, nj)` | `entity.data.data.qpos[:, 2:]` |
| Joint velocities | `entity.data.joint_vel` — `(N, nj)` | `entity.data.data.qvel[:, :]` |
| Root position | `entity.data.root_link_pos_w` | `entity.data.data.xpos[:, 0, :]` |
| Root orientation | `entity.data.root_link_quat_w` | `entity.data.data.xquat[:, 0, :]` |
| Projected gravity | `entity.data.projected_gravity_b` | manual `rotate(gravity, quat)` |

**Accepted exceptions** (no Entity API equivalent yet — document with a comment):
- `entity.data.data.act` — muscle activation state
- `entity.data.data.actuator_length` / `actuator_velocity` / `actuator_force`
- `entity.data.data.cvel` — body CoM velocity (used for locomotion rewards)

For these, add `# accepted: no entity.data API — entity.data.data.X` and file a
feature request upstream.

---

## 4. Domain Randomization

### Rule: always use `dr.*` event functions

Writing directly to `data.model.body_mass`, `data.model.geom_friction`, etc. on
an unexpanded model writes to a **shared `(1, nbody)` array** — all environments
get the same value. This is a silent correctness bug.

The `dr.*` functions call `@requires_model_fields`, which triggers
`sim.expand_model_fields()` so Warp allocates per-world `(N, nbody)` copies.

```python
# CORRECT
EventTermCfg(
    func=dr.body_mass,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("pingpong"),
            "mass_distribution_params": (0.0024, 0.0004),
            "operation": "add"},
)

# WRONG — silent: all envs get the same mass at num_envs > 1
env.scene["pingpong"].data.model.body_mass[env_ids, bid] = sampled_mass
```

---

## 5. Observations, Rewards, Events — Term Functions

### Rule: use `ObservationTermCfg`, `RewardTermCfg`, `EventTermCfg`

Every obs, reward, and event computation must be a standalone function (or
`ManagerTermBase` subclass) registered via the appropriate `*TermCfg`. Never
assemble observation dicts manually or override `_get_observations()`.

```python
# CORRECT
obs_cfg = ObservationGroupCfg(terms={
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": robot_cfg}),
    "muscle_act": ObservationTermCfg(func=my_muscle_act_term, params={"asset_cfg": robot_cfg}),
})

# WRONG
def _get_observations(self):
    return {"joint_pos": self.entity.data.data.qpos[:, 2:], ...}
```

### Rule: term functions must be fully vectorized

Term functions receive `env` and must process all `num_envs` simultaneously with
PyTorch batch operations. **No Python loops over environments in step-rate
functions.**

```python
# CORRECT — vectorized
def ball_height_reward(env, asset_cfg) -> torch.Tensor:
    pos = env.scene[asset_cfg.name].data.root_link_pos_w  # (N, 3)
    return pos[:, 2]  # (N,)

# WRONG — Python loop, destroys GPU throughput at scale
def ball_height_reward(env, asset_cfg) -> torch.Tensor:
    results = []
    for i in range(env.num_envs):
        results.append(env.scene[asset_cfg.name].data.root_link_pos_w[i, 2].item())
    return torch.tensor(results)
```

### Rule: stay on-device

No `.detach().cpu()` inside obs/reward/event functions. Every intermediate
tensor stays on `env.device` until the manager returns the batch to the
training loop.

---

## 6. Action Terms

### Rule: subclass `ActionTermCfg` and `ActionTerm`

Custom action classes must subclass mjlab's abstract base classes so the
`ActionManager` can type-check, call `reset()`, and access `action_dim`.

```python
# CORRECT
@dataclass(kw_only=True)
class MyMuscleCfg(ActionTermCfg):
    class_type: type = MyMuscleAction
    target_names_expr: tuple[str, ...] = (".*_tendon",)

class MyMuscleAction(ActionTerm):
    def __init__(self, cfg: MyMuscleCfg, env): ...
    @property
    def action_dim(self) -> int: ...
    def process_actions(self, actions): ...
    def apply_actions(self): ...

# WRONG — plain class, not recognised by ActionManager
class MyMuscleActivationAction:
    def __init__(self, cfg, env): ...
    def apply_actions(self): ...
```

---

## 7. Per-env State and Caches

### Rule: instance attributes on `ManagerTermBase`, not module globals

Module-level dicts keyed by `id(env)` cause memory leaks and stale-cache bugs
when environments are recreated in the same process (CPython reuses object IDs).

```python
# CORRECT — state lives on the term instance
class SaberTaskLogic(ManagerTermBase):
    def __init__(self, cfg, env):
        self._active_target = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._dwell_counter = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env, env_ids=None):
        ...  # update state

    def reset(self, env_ids):
        self._active_target[env_ids] = 0
        self._dwell_counter[env_ids] = 0.0

# WRONG — module-level cache keyed by id(env), leaks on env gc
_task_cache: dict[int, dict] = {}

def get_task_state(env):
    if id(env) not in _task_cache:
        _task_cache[id(env)] = {"target": 0}
    return _task_cache[id(env)]
```

For static data derived from the XML (body mass, joint addresses) that is the
same across all env instances of a given model, use
`@functools.lru_cache(maxsize=None)` keyed on the XML path, not on `id(env)`.

---

## 8. Simulation Time

### Rule: never write to `data.time` directly

`data.time` is owned by the Warp backend. Writing to it directly corrupts CUDA
graph state and breaks the episode-length counter used by `TerminationManager`.

Multi-rally or sub-episode restart logic must be implemented as episode
termination + full environment reset via `TerminationTermCfg` +
`EventTermCfg(mode="reset")`. Use a per-env `rally_count` tensor in a
`ManagerTermBase` to track multi-rally state across resets.

---

## 9. Contact Detection

### Rule: use `ContactSensor`, not raw `data.contact` struct access

mjlab's `ContactSensor` returns batched force tensors on-device.
Iterating over `data.contact.geom` with Python loops causes thousands of CUDA
synchronizations per step at scale.

```python
# CORRECT
SceneCfg(sensors={"contact": ContactSensorCfg(prim_path="{ENV_REGEX_NS}/pingpong")})

# In reward term:
forces = env.scene.sensors["contact"].data.net_forces_w  # (N, nbodies, 3)
hit = forces[:, ball_body_id, 2].abs() > CONTACT_THRESHOLD  # (N,) bool

# WRONG — Python loop, CPU round-trips, fragile struct access
def _contacts_for_env(env, i):
    contacts = set()
    for k in range(env.data.nacon[i].item()):  # .item() = CUDA sync
        geom = env.data.contact.geom[i, k, 0].item()  # CUDA sync
        contacts.add(geom)
    return frozenset(contacts)
```

---

## 10. Registration

### Rule: use `register_mjlab_task`, never per-backend duplication

All tasks are registered once via `register_mjlab_task(TaskSpec(...))`. No
per-backend registration files, no hardcoded env IDs, no silent `except: pass`
swallowing registration errors.

```python
# CORRECT error handling in registration
try:
    register_mjlab_task(make_sarcopenia_cfg())
except Exception:
    logging.getLogger(__name__).warning(
        "sarcopenia task registration failed", exc_info=True
    )

# WRONG — silent failure makes debugging impossible
try:
    register_mjlab_task(make_sarcopenia_cfg())
except Exception:
    pass
```

---

## Anti-Pattern Reference

| ID | Anti-pattern | Severity | Fix |
|---|---|---|---|
| AP-1 | Custom action class not subclassing `ActionTermCfg`/`ActionTerm` | Medium | §6 |
| AP-2 | Module-level mutable globals keyed by `id(env)` | High | §7 |
| AP-3 | `entity.data.data.*` reads (raw Warp layer) | Medium | §3 |
| AP-4 | Direct Warp buffer writes bypassing Entity write API | High | §2 |
| AP-5 | Python loops over environments in step-rate functions | High | §5 |
| AP-6 | `.detach().cpu()` round-trips in obs/reward hot paths | Medium | §5 |
| AP-7 | Hand-rolled contact detection instead of `ContactSensor` | Medium | §9 |
| AP-8 | Direct `data.model.body_mass` / `geom_friction` writes | **Critical** | §4 |
| AP-9 | Module-level `SceneEntityCfg` global mutated at config time | High | §7 |
| AP-10 | Task logic embedded in Entity subclasses | Medium | §7 |
| AP-11 | Raw `qpos[:, 2:]` address arithmetic instead of `entity.data.joint_pos` | Medium | §3 |
| AP-12 | `data.time[i] = 0.0` direct sim-time mutation | High | §8 |
| AP-13 | Task state class not managed by `EventManager` | Medium | §5 |
| AP-14 | `id(env)`-keyed cache — leak on env GC, stale IDs on reuse | High | §7 |
| AP-15 | Silent `except: pass` in task registration | Low | §10 |

---

## How to Add a New mjlab Task

This is the canonical checklist for porting a CPU/MJX task to mjlab or writing a
new task from scratch. Following it prevents every anti-pattern in the table below.

### Pre-flight: assess the task

Before writing a line of code, answer these questions:

| Question | If yes → |
|---|---|
| Does the model XML embed a second free-floating body (ball, die, object)? | That body must be a **separate EntityCfg** with its own `spec_fn`. Do not use `spec.delete()` to split a monolithic XML — it corrupts mocap parent chains. Strip the secondary body from the primary entity spec and re-add any cross-tree sensors via `SceneCfg.spec_fn`. |
| Does reset need to write to a kinematic/mocap body? | Use `write_root_state_to_sim` inside an `EventTermCfg` reset hook. |
| Does the task have an opponent or external controller updated every step? | The controller runs inside a `compute_actions`-style `EventTermCfg` with `mode="interval"`. All state is on a `ManagerTermBase` subclass. |
| Is the action space mixed (muscles + robotic joints)? | Write a custom `ActionTerm` + `ActionTermCfg` pair (see `MyoMuscleActivationAction` as reference). One shared class, not a per-task copy. |
| Does any obs/reward term touch `env.sim` or call MuJoCo directly? | Port it to the `accessor` API or use `entity.data.*` instead. |
| Does reset perform domain randomization (mass, friction, mesh scale)? | Use `dr.*` event functions. Never write to `data.model.*` directly. |

### Step 1 — Author the XML spec(s)

- One XML file per rigid-body entity (robot, each floating prop).
- No cross-file mocap dependencies between entity XMLs.
- If a mocap target body must exist in the same XML as the robot (e.g. table tennis
  target pool), document the exception and leave a `# monolithic-XML: reason` comment.
- Verify the spec loads: `python -c "import mujoco; mujoco.MjSpec().from_file('your.xml')"`.

### Step 2 — Create a `TaskConfig` subclass (CPU/MJX parity)

```python
# myosuite/envs/myo/tasks/your_task/config.py
@dataclass
class YourTaskConfig(TaskConfig):
    model: str = "your_model_name"
    max_episode_steps: int = 500
    obs: ObsSpec = field(default_factory=lambda: ObsSpec(channels=["qpos", "qvel", ...]))
    reward: RewardSpec = field(default_factory=lambda: RewardSpec(
        terms={"dist": RewardTermSpec(func="your_dist_reward", weight=-1.0)}
    ))
```

The `TaskConfig` is backend-agnostic — it drives CPU, MJX, **and** mjlab.

### Step 3 — Write term functions

All obs, reward, and action term functions must be **pure and backend-agnostic**.
See `docs/wiki/writing-term-functions.md` for the full rules. Summary:

- Use `accessor.array_module()` for all array ops — never import numpy/torch directly.
- Read state via `entity.data.*` — never `env.sim`, never `data.qpos[:,i]` arithmetic.
- No side effects, no mutation of module globals.

### Step 4 — Write a `spec_fn`

```python
def _your_spec_fn() -> mujoco.MjSpec:
    xml_path = myo_sim.get_path("your_model.xml")   # never hardcode paths
    return mujoco.MjSpec().from_file(str(xml_path))
```

For floating props (ball, die), write a separate `_prop_spec_fn` for each.

### Step 5 — Build the mjlab config with the factory

```python
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import mjlab_env_cfg_from_task_config

def _make_your_env_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
    cfg = YourTaskConfig()
    tendon_names = _your_tendon_names()

    observations = {
        "policy": ObservationGroupCfg(terms={
            "your_obs": ObservationTermCfg(func=_your_obs_fn),
        })
    }
    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name="your_entity",
            actuator_names=tendon_names,
        ),
    }
    rewards = {
        "dist": RewardTermCfg(func=_your_dist_reward, weight=-1.0),
    }
    events = {
        "reset": EventTermCfg(func=_your_reset_event, mode="reset"),
    }

    return mjlab_env_cfg_from_task_config(
        cfg=cfg,
        spec_fn=_your_spec_fn,
        entity_name="your_entity",
        actuators=(_XmlActuatorCfg(
            target_names_expr=tuple(f"{n}_tendon" for n in tendon_names),
            transmission_type=TransmissionType.TENDON,
        ),),
        observations=observations,
        actions=actions,
        rewards=rewards,
        events=events,
        num_envs=num_envs,
        decimation=10,
        sim_cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    )
```

If the task has **a second floating entity** (ball, die), add it to `SceneCfg.entities`
directly — the factory's `spec_fn` / `entity_name` / `actuators` args cover only the
primary robot entity. Construct `SceneCfg` manually in that case and pass only the
non-scene kwargs to the factory, or build the `ManagerBasedRlEnvCfg` directly.

### Step 6 — Register

Add a `register_mjlab_task(task_id=..., env_cfg_fn=..., rl_runner_cfg_fn=...)` call
inside `register_mjlab_tasks()` in `register_mjlab_tasks.py`. Do **not** create a new
registration file. Wrap it in a try/except that logs on failure (AP-15).

### Step 7 — Write a parity test

Add to `myosuite/tests/test_mjlab_task_builder.py`:

1. A `_build_reference_<task>_cfg()` function that constructs the config inline.
2. A `_build_factory_<task>_cfg()` function that uses the factory.
3. Assertions on: `decimation`, `episode_length_s`, `sim.mujoco.timestep`,
   obs group keys, action keys, reward keys, and reward weights.

Run: `pytest myosuite/tests/test_mjlab_task_builder.py -v`

### Step 8 — Update the README backend matrix

Change the task row from `–` or `wip` to `beta (env_id)` only after:
- [ ] `register_mjlab_task` call present and tested
- [ ] Parity test passes
- [ ] At least one smoke-test rollout (1 env, 10 steps) succeeds

---

## Known justified exceptions to the canonical pattern

These tasks deviate from the rules above for documented reasons. Do not copy their
patterns without understanding the constraint.

| Task | Deviation | Reason |
|---|---|---|
| Saber (`saber_p0_mjlab_env.py`) | Custom `SaberTaskEntityCfg` subclass; sync-guarded Warp writes for saber mocap targets | `spec.delete()` corrupts mocap target-pool parent chain; XML refactor tracked in #42; control/reset wiring to TaskBuilder pipeline tracked in #45 |
| TableTennis (`register_mjlab_tabletennis.py`) | Closure-based obs/reward/event terms; multiple actuator groups; AP-8 paddle mass / ball friction DR | Ball is a separate entity; cross-tree velocimeter sensors added via `SceneCfg.spec_fn`; DR migration needs `dr.*` support for split-entity scenes |
| Elbow (`register_mjlab_tasks.py`) | `MyoMuscleActivationAction` instead of `XmlActuatorCfg` action | `XmlMuscleActuatorCfg` removed in mjlab v1.4; `MyoMuscleActivationAction` is the shared replacement |

---

## Migration Priority

**P0 — correctness blockers (fix before next training run):**
- AP-8: `data.model.*` direct writes → `dr.*` event functions
- AP-12: `data.time` mutation → multi-rally termination redesign
- AP-9: `_OBS_PROPRIO_ENTITY_CFG` global mutation

**P1 — performance (fix before scaling past ~512 envs):**
- AP-5 + AP-6 + AP-7: Python loops + CPU sync + hand-rolled contacts in TableTennis (reward / obs closures still loop over envs)

**P2 — API stability:**
- AP-4: remaining Warp direct writes → Entity write API
- AP-1: custom action classes → `ActionTermCfg`/`ActionTerm`
- AP-2 + AP-14: module globals → `ManagerTermBase` instance state
- AP-3 + AP-11: `.data.data.*` → `entity.data.*`

**P3 — cleanup:**
- AP-10, AP-13: task logic in entities → `ManagerTermBase`
- AP-15: silent registration exceptions → logging
