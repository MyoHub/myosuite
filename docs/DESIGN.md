# MyoSuite Architecture Design

> Version: 1.0 — covers the v3 refactoring (composable terms, three backend paths, scalable model/task management)

## Overview

MyoSuite is a musculoskeletal reinforcement-learning suite built on MuJoCo.
This document describes the internal architecture after the v3 refactoring.

---

## Three Backend Paths

All three paths run the same task logic via shared **term functions**.
The choice of backend affects only physics execution and tensor types.

| Path | Physics | Tensors | Interface | Primary use |
|---|---|---|---|---|
| CPU / Gymnasium | MuJoCo CPU | numpy | `gymnasium.Env` | Single-env RL, testing, sim-to-real |
| MJX + Brax | MuJoCo JAX | jax.Array | `mujoco_playground.MjxEnv` | GPU-vectorised, JAX/Brax training |
| MuJoCo Warp / mjlab | MuJoCo Warp | torch.Tensor | `mjlab.ManagerBasedRlEnv` | GPU-vectorised, PyTorch/mjlab training |

### Entry Points

```python
# CPU
# NOTE: importing myosuite triggers the Gymnasium side-effect registration.
# gym.make() will raise NameNotFound without this import.
import myosuite               # triggers gymnasium.register() for all myo envs
import gymnasium as gym
env = gym.make("myoElbowPose1D6MRandom-v0")

# MJX + Brax
# NOTE: mujoco_playground.registry.load() only knows about its own hardcoded
# suites — there is no plugin API to inject external environments.
# Use myosuite's own make() factory instead, then wrap for Brax training.
from myosuite.envs.myo.backends.mjx import make
from mujoco_playground import wrapper as pg_wrapper

env = make("MjxElbowPoseRandom-v0")                  # -> MyoMjxEnvBase (MjxEnv subclass)
brax_env = pg_wrapper.wrap_for_brax_training(env)     # -> Brax-compatible

# mjlab
import mjlab.envs
env = mjlab.envs.make("myoElbowPose1D6MRandom-v0")        # -> ManagerBasedRlEnv
```

---

## Shared Term Functions

Term functions are pure, path-agnostic functions in `myosuite/terms/`.
Each receives an `EnvAccessor` that exposes physics state in the native tensor type.

```
myosuite/terms/
  myo_obs_terms.py         joint_pos_obs, joint_vel_obs, tip_pos_obs, pose_error_obs, muscle_act_obs
  myo_reward_terms.py      pose_reward, reach_reward, walk_reward, act_reg, joint_penalty
  myo_termination_terms.py joint_limit_violation, fall_termination
  myo_event_terms.py       reset_random_pose, apply_sarcopenia, apply_fatigue
  myo_action_terms.py      MuscleActionTerm, MuscleActionTermCfg  (mjlab ActionTerm)
```

`EnvAccessor` is a `Protocol` that returns `numpy.ndarray` on CPU, `jax.Array` on MJX,
and `torch.Tensor` on mjlab. Term functions use `accessor.array_module` (numpy / jnp / torch)
for any array operations so the same code runs on all three paths.

---

## Model Management

### Fragment-based Composition

Body-part models are **imported from the `myo_sim` package** (a standalone
pip-installable Python package at `https://github.com/MyoHub/myo_sim`).
`myo_sim` holds all canonical musculoskeletal XML fragments and exposes them
through a `FragmentRegistry` Python API — no local copies in myosuite.

Fragment metadata (version, joint names) is declared in `myo_sim._registry`:

```python
# myo_sim/_registry.py
FragmentRegistry.register("elbow", "elbow/myoelbow_v0.xml",
    version=3, joints=["elbow_flexion", "elbow_supination"])
```

`ModelBuilder` composes fragments programmatically using MuJoCo 3's `MjSpec.attach()`:

```python
import myo_sim
from myosuite.core.model_builder import ModelBuilder

model, spec = (ModelBuilder()
    .attach_fragment("shoulder")   # -> myo_sim.FragmentRegistry.get("shoulder").path
    .attach_fragment("elbow",  parent="shoulder_distal")
    .attach_fragment("hand",   parent="elbow_distal")
    .set_timestep(0.002)
    .build())
```

Built models are cached by the content hash of the installed fragment files.
`ModelBuilder.build()` is idempotent within a Python session.

### Named Recipes

Predefined recipes are registered with `@model_recipe(name)`:

| Recipe name | Fragments |
|---|---|
| `elbow_standard` | `elbow` |
| `elbow_sarcopenia` | `elbow` + sarcopenia transform (force_scale=0.5) |
| `full_arm` | `shoulder` + `elbow` + `hand` |
| `hand_standard` | `hand` |
| `walk_standard` | `leg` + `foot` |

Tasks reference a recipe name in their `TaskConfig.model` field.

### Fragment Versioning

When a fragment's XML changes its joint names, geometry, or actuator count,
its version number must be incremented. Tasks declare which fragment version
they were audited against in `_fragment_versions: ClassVar[dict]`.

CI (`scripts/check_fragment_compat.py`) fails if any fragment version exceeds
a task's declared version, forcing an explicit re-audit before the task ships.

---

## Scene Management

Scenes are declared in `myosuite/scenes/library.py` as `SceneSpec` dataclasses:

```python
SCENES = {
    "flat_floor":   SceneSpec(floor="flat",   objects=[]),
    "cube_reach":   SceneSpec(floor="flat",   objects=[ObjectSpec("cube",   size=0.03)]),
    "sphere_reach": SceneSpec(floor="flat",   objects=[ObjectSpec("sphere", r=0.03)]),
    "baoding_pair": SceneSpec(floor="flat",   objects=[ObjectSpec("ball",   r=0.028, count=2)]),
    "uneven_floor": SceneSpec(floor="uneven", objects=[]),
}
```

`SceneSpec` is backend-aware:
- **CPU / MJX** → rendered to MJCF XML, appended to the model before `MjSpec.compile()`
- **mjlab** → rendered to `RigidObjectCfg` / `TerrainCfg` entries in `SceneCfg`
  (mjlab supports dynamic add/remove of scene components at runtime)

Tasks reference a scene by name in `TaskConfig.scene`.

---

## Task Management

### TaskConfig Dataclass

A task is fully described by a `TaskConfig` dataclass. No logic lives here — only
configuration that wires together a model recipe, a scene, and term functions.

```python
@dataclass
class ElbowPoseTask(TaskConfig):
    model:             str        = "elbow_standard"
    scene:             str        = "flat_floor"
    max_episode_steps: int        = 200
    _fragment_versions: ClassVar  = {"elbow": 3}
    obs:     ObsGroup   = field(default_factory=elbow_obs)
    rewards: RwdGroup   = field(default_factory=pose_rewards)
    events:  EventGroup = field(default_factory=random_pose_reset)
```

### Variant Tasks

Task variants are created by inheriting `TaskConfig` and overriding one field.
No logic is duplicated:

```python
@dataclass
class ElbowPoseSarcopeniaTask(ElbowPoseTask):
    model: str = "elbow_sarcopenia"         # different model recipe

@dataclass
class ElbowPoseFullArmTask(ElbowPoseTask):
    model: str = "full_arm"                 # larger combined model

@dataclass
class ElbowReachCubeTask(ElbowPoseTask):
    scene:   str      = "cube_reach"        # different scene
    rewards: RwdGroup = field(default_factory=reach_rewards)
```

### Auto-Registration

Tasks are registered with a decorator — no central list to maintain:

```python
@register_task("myoElbowPose16DFixed-v0", "myoElbowPose1D6MRandom-v0", max_episode_steps=200)
class ElbowPoseTask(TaskConfig): ...

@register_task("myoElbowPoseSarcopenia-v0", max_episode_steps=200)
class ElbowPoseSarcopeniaTask(ElbowPoseTask): ...
```

---

## Class Hierarchy

```
gymnasium.Env
  └── MyoGymnasiumEnv                 (myosuite/envs/gymnasium_env.py)
        └── PoseEnvV0                 (myosuite/envs/myo/myobase/pose_v0.py)
        └── ReachEnvV0
        └── WalkEnvV0
        └── BaodingEnvV1

mujoco_playground.MjxEnv             (separate hierarchy — NOT MyoGymnasiumEnv)
  └── MyoMjxEnvBase                  (myosuite/envs/myo/backends/mjx/mjx_env_base.py)
        └── MyoMjxPoseEnv
        └── MyoMjxReachEnv

mjlab.ManagerBasedRlEnv              (no subclass needed)
  configured by → MyoElbowPoseCfg(ManagerBasedRlEnvCfg)
                  MyoHandReachCfg(ManagerBasedRlEnvCfg)
                  …
```

---

## MJX Interface (mujoco_playground)

MyoSuite MJX environments implement the exact `mujoco_playground.MjxEnv` interface:

- `reset(rng: jax.Array) -> mujoco_playground.State`
- `step(state: State, action: jax.Array) -> mujoco_playground.State`

`State` fields: `data: mjx.Data`, `obs: jax.Array`, `reward: jax.Array`,
`done: jax.Array`, `metrics: dict[str, jax.Array]`, `info: dict[str, Any]`

Registration uses `mujoco_playground.registry.register_environment(name, cls, cfg_fn)`.
Brax PPO training uses `mujoco_playground.wrapper.wrap_for_training(env)` unchanged.

---

## mjlab Integration

MyoSuite tasks run natively in mjlab by providing `ManagerBasedRlEnvCfg` subclasses
(one per task) under `myosuite/envs/myo/backends/mjlab/configs/`.

mjlab auto-discovers tasks via `pyproject.toml` entry-point:

```toml
[project.entry-points."mjlab.tasks"]
myosuite = "myosuite.envs.myo.backends.mjlab"
```

`MuscleActionTerm` bridges mjlab's action API to MuJoCo muscle actuators,
mapping policy output `[-1, 1]` → muscle excitation `[0, 1]`.

---

## Adding New Content

### New body part

1. **In the `myo_sim` repo** (MyoSuite consumes it as the `myosuite/simhive/myo_sim` git submodule): add the XML fragment to the appropriate directory (e.g. `new_part/myonewpart_v0.xml`)
2. **In `myo_sim/_registry.py`**: call `FragmentRegistry.register("new_part", "new_part/...", version=1, joints=[...])`
3. Bump the submodule pointer in MyoSuite to the new `myo_sim` commit (or tag) after upstream merges
4. **In myosuite**: register a recipe in `myosuite/core/model_recipes.py` using `@model_recipe` (no PyPI `myo-sim` dependency)
5. No existing tasks change

### New scene object / floor

1. Add a `SceneSpec` entry to `myosuite/scenes/library.py`
2. Reference by name in any `TaskConfig.scene`

### New task variant

1. Subclass an existing `TaskConfig`, override one field
2. Decorate with `@register_task("myoNewTask-v0")`
3. No other files change

### New mjlab scene component

1. Add a `RigidObjectCfg` or `TerrainCfg` to the appropriate `SceneCfg` in
   `myosuite/envs/myo/backends/mjlab/configs/`
2. mjlab handles dynamic add/remove at runtime; no MJCF edit needed
