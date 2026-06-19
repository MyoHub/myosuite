# Adding a New Task

This is the concrete, working procedure for adding a new environment under
the modular `TaskConfig` architecture (the Golden Rule from `CLAUDE.md`):
**one `TaskConfig` + term functions + a `ModelBuilder` recipe — no
task-specific env subclass, no new registration file, no backend-specific
code.**

Read `engineering-standards.md` and `writing-term-functions.md` first if you
haven't.

## Worked example

`myosuite/envs/myo/tasks/basic/specs/elbow_pose_spec.py` is the canonical,
fully-working reference — copy its shape, not its specifics:

```python
from myosuite.core.config import (
    ActuatorGroupSpec, BackendConfig, GoalSpec, ObsSpec, RewardSpec, TaskConfig,
)
from myosuite.core.registry import register_task


@dataclass
class ElbowPoseFixedTask(TaskConfig):
    model: str = "elbow_standard"          # ModelBuilder recipe name
    max_episode_steps: int = 100
    obs: ObsSpec = field(
        default_factory=lambda: ObsSpec(keys=["joint_pos", "joint_vel", "muscle_act"])
    )
    goal: GoalSpec = field(
        default_factory=lambda: GoalSpec(
            target_type="joint_angles", randomize=False, range={"r_elbow_flex": (2.0, 2.0)},
        )
    )
    reward: RewardSpec = field(
        default_factory=lambda: RewardSpec(terms=["pose", "act_reg"], weights={"pose": 1.0, "act_reg": 1.0})
    )
    actuators: list[ActuatorGroupSpec] = field(
        default_factory=lambda: [ActuatorGroupSpec(name="elbow_muscles", normalize_actions=True)]
    )

register_task(ElbowPoseFixedTask(), env_id="myoElbowPoseTaskFixed-v0")
```

That's the whole task. `register_task()` runs once at import time; the env
runs through the generic `ModularTaskEnv` — there is no per-task `Env`
subclass to write.

## Step by step

1. **Pick or add a `ModelBuilder` recipe** (`myosuite/core/model_recipes.py`).
   Reuse an existing recipe (`elbow_standard`, `hand_standard`, ...) if it
   already has the bodies/sites you need. If a recipe is genuinely
   incomplete for your task (check its docstring for a "Gaps vs ..." note —
   some recipes are explicitly partial stubs), finish it there rather than
   working around the gap in the task config.

2. **Check available term names** before inventing new ones:
   - Obs: `myosuite/terms/base_obs.py` — `ObsSpec.keys` entry `"foo"` resolves
     to function `foo_obs`.
   - Reward: `myosuite/terms/base_reward.py` — `RewardSpec.terms` entry
     `"foo"` resolves to function `foo_reward` (or a bare `foo` function).
   - If you need a new term, write it there as a pure function:
     `(accessor, obs_dict, **kwargs) -> dict` for reward,
     `(accessor, **kwargs) -> dict[str, Any]` for obs. Use
     `accessor.array_module()` only — never import `numpy`/`jax.numpy`/`torch`
     directly. See `writing-term-functions.md`.

3. **Write the `@dataclass class FooTask(TaskConfig)`** with concrete,
   non-default `obs`, `goal`, and `reward` fields. See `myosuite/core/config.py`
   for the full field list (`ObsSpec`, `GoalSpec`, `RewardSpec`,
   `ActuatorGroupSpec`, `BackendConfig`, `VariantSpec`, ...).

4. **Call `register_task(FooTask(), env_id="myoFoo-v0")`** once, at module
   import time, in your task's `__init__.py` (or a `specs/` module imported
   from it — see `basic/specs/__init__.py`).

5. **Verify it actually runs and produces a real signal** — this is the step
   that's easy to skip and easy to get wrong silently:
   ```python
   import gymnasium as gym
   env = gym.make("myoFoo-v0")
   obs, info = env.reset()
   obs, rwd, terminated, truncated, info = env.step(env.action_space.sample())
   assert rwd != 0.0  # or whatever a real reward looks like for this task
   ```
   A `TaskConfig` with no `obs`/`reward` override still imports and registers
   cleanly — `gym.make()` won't tell you it's hollow. Don't leave a task
   half-specified and assume someone will fill it in "later"; either finish
   it before merging or don't register it yet.

6. **Add it to `test_registry.py`** (or the relevant suite's registry test)
   so its env ID is covered by the registration smoke test.

7. **Run parity / quality gates**:
   ```bash
   pytest myosuite/tests/test_model_builder.py -v
   pytest myosuite/tests/test_terms_cpu.py -v
   pytest myosuite/tests/test_parity.py -v
   pre-commit run --all-files
   ```

## What not to do

- Don't add a registration entry "for coverage" before the obs/reward terms
  exist — a previous batch of `*Modular-v0` IDs did exactly this (one
  `register_task()` call per legacy challenge ID, all left at `TaskConfig`
  defaults) and was removed because none of them had a working reward or
  task-relevant observation; they only added 19 dead env IDs with no test
  depending on them. Register an ID when the task is real, not as a
  placeholder for future work.
- Don't write a new `Env` subclass, a new registration file, or
  backend-specific branching to support a new task — if you find yourself
  doing this, the `TaskConfig`/term-function/recipe primitives are probably
  missing something; extend those instead (see `engineering-standards.md`'s
  Search-Before-Write section).
