# Getting Started (for developers)

New to the MyoSuite codebase? This page takes you from a fresh clone to making
your first change with confidence. It assumes you can write Python but know
nothing about this repository. If you only want to *use* MyoSuite (run
environments, train policies), read the top-level [README.md](../../README.md) instead — this
page is about *changing* the code.

---

## 1. Set up and prove it works

```bash
# from the repo root
uv sync -p 3.10 --extra dev     # or: pip install -e ".[dev]"
# GPU (Linux + CUDA): uv pip install -e ".[mjlab]" --torch-backend=auto
# (picks the torch build for your driver; with pip see docs/source/install.rst)

pytest myosuite/tests/test_registry.py -v
pytest myosuite/tests/test_parity.py -v
```

`test_registry.py` smokes every CPU env (`reset` + one `step`).
`test_parity.py` replays frozen actions against stored baselines
(default `atol=1e-6`). MJX is experimental and is not part of `[dev]`.

On **macOS**, the interactive viewer needs `mjpython` instead of `python`.
Tests run under plain `pytest`.

## 2. The mental model

A MyoSuite environment is four things:

| Piece | Question it answers | Where it lives |
|---|---|---|
| **Model** | What body is being simulated? | a MuJoCo XML / `ModelBuilder` recipe |
| **Observation** | What does the agent see each step? | `_get_obs_dict()` → a vector |
| **Reward** | What is the agent rewarded for? | `get_reward_dict()` → `{"dense", "done", ...}` |
| **Step loop** | How does time advance? | `MyoGymnasiumEnv.step()` (shared) |

Every env returns the Gymnasium 5-tuple from `step()`:
`(obs, reward, terminated, truncated, info)`.

Muscle conditions (motor noise, fatigue, sarcopenia, reafferentation) are **not** part of an env class.
They are wrappers, applied by id (`myoFati…`, `myoSarc…`, `myoReaf…`) or with
`make_env(EnvConfig(env_id, features=...))`; see [cross-backend-contract.md](cross-backend-contract.md).

### CPU and GPU: two matched halves, not a choice

A task normally exists in **two matched implementations under one `env_id`**
(see [engineering-standards.md](engineering-standards.md)):

- **CPU** — a `MyoGymnasiumEnv` subclass. One env at a time, easy to read and
  step through. This is where you **play back and fine-tune** a policy, and
  where you start when adding a task.
- **GPU** — the *same* task as a **mjlab** (`ManagerBasedRlEnvCfg`, MuJoCo-Warp)
  config, running thousands of envs in parallel for **fast RL training**.

You don't pick one: you **train on GPU (mjlab) and play back / fine-tune on
CPU**, so the two must agree on observations, action mapping, and control timing
(the "cross-backend contract"). `test_parity.py` and the mjlab parity tests
guard that agreement. For a brand-new task you can start with just the CPU env
and add the mjlab config when you need parallel training.

> A JAX **MJX** backend also exists but is **experimental and may not be
> maintained long-term** — don't build new work on it.

## 3. Read one env end to end

Before changing anything, read this file top to bottom — it is the canonical,
fully-commented CPU (`MyoGymnasiumEnv`) example:

```
myosuite/envs/myo/tasks/basic/arm/reach.py
```

You will see the four pieces above: the constructor builds `self.model`/`self.data`
and the action/observation spaces; `_get_obs_dict` builds the observation;
`get_reward_dict` scores it; `reset_task` samples a new target. `step()` and
`reset()` are thin wrappers over the shared physics loop.

## 4. Make a safe first change

A good first change is tweaking a reward weight and confirming the effect:

```python
from myosuite import make_env

env = make_env("myoElbowPose1D6MRandom-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info["rwd_dict"])     # every reward component, not just the scalar
```

`info["rwd_dict"]` and `info["obs_dict"]` expose the full breakdown every step —
use them to understand what a task actually rewards before you change it.

After any edit to an env, run its parity test. Not every env has a parity
baseline (run `pytest myosuite/tests/test_parity.py --co -q` to see which do);
the elbow env is covered:

```bash
pytest myosuite/tests/test_parity.py -k ElbowPose -q
```

If your change was **intentional** (you meant to change the physics/reward),
regenerate the baseline:

```bash
python scripts/generate_parity_baselines.py --env-id myoElbowPose1D6MRandom-v0
```

If your change was *not* meant to alter behavior and parity fails, you
introduced a regression — investigate before committing.

## 5. Add a new task

Follow [adding-a-new-task.md](adding-a-new-task.md). In short, build the CPU env first:

1. Copy `basic/arm/reach.py` (or the closest existing task) to a new file.
2. Rewrite `_get_obs_dict` / `get_reward_dict` / `reset_task` for your task.
3. Register it in the suite `__init__.py` with `registry.register_env(...)`.
4. Add its env ID to `test_registry.py`.
5. To let the env take muscle-command features, keep the stage hooks of `reach.py`
   (`CtrlStageHost`: `_run_ctrl_stages` before writing `ctrl`, `_run_reset_stages` in `reset`);
   see [cross-backend-contract.md](cross-backend-contract.md).
6. Run the quality gates (below).

Then, when you need parallel training, add the matched mjlab GPU config under
the same `env_id` — see the CPU/GPU section in [engineering-standards.md](engineering-standards.md).

## 6. Quality gates (run before every commit)

```bash
pre-commit run --all-files
pytest myosuite/tests/test_registry.py -v
pytest myosuite/tests/test_parity.py -v
```

Branch from `dev` and open the pull request against `dev`; CI runs for pull requests into `main` and `dev`.
See [CLAUDE.md](../../CLAUDE.md) for the full gate list. A pre-commit hook blocks imports of the
deleted `BaseV0`/`env_base.MujocoEnv` classes — if it fires, you copied from an
old example; use `MyoGymnasiumEnv` instead.

## Where to look next

| You want to... | Read |
|---|---|
| Find where something lives | [repository-map.md](repository-map.md) |
| Understand the two env patterns | [engineering-standards.md](engineering-standards.md) |
| Write an obs/reward term | [writing-term-functions.md](writing-term-functions.md) |
| Add mjlab GPU support | [mjlab-design-guide.md](mjlab-design-guide.md), [cross-backend-contract.md](cross-backend-contract.md) |
| Use noise, fatigue or other muscle conditions, or build an env on either backend | [cross-backend-contract.md](cross-backend-contract.md) (`make_env`, `EnvConfig`, which envs take which features) |
| Use an existing helper | [library-usage.md](library-usage.md) |
