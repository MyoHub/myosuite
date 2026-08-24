# Getting Started (for developers)

New to the MyoSuite codebase? This page takes you from a fresh clone to making
your first change with confidence. It assumes you can write Python but know
nothing about this repository. If you only want to *use* MyoSuite (run
environments, train policies), read the top-level `README.md` instead — this
page is about *changing* the code.

---

## 1. Set up and prove it works

```bash
# from the repo root
uv sync --extra dev            # or: pip install -e ".[dev]"

# run the fast core tests — these should all pass on a clean checkout
pytest myosuite/tests/test_registry.py -q
pytest myosuite/tests/test_parity.py -q
```

`test_registry.py` loads every registered environment and takes one step —
if it passes, your install is sound. `test_parity.py` replays frozen action
sequences and checks the physics still matches a stored baseline to `1e-7`;
it is your safety net for any change that touches an environment.

On **macOS**, rendering needs `mjpython` instead of `python` (see the README
troubleshooting section). Tests run under plain `pytest`.

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

### CPU and GPU: two matched halves, not a choice

A task normally exists in **two matched implementations under one `env_id`**
(see `engineering-standards.md`):

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
import gymnasium as gym
import myosuite  # registers all envs on import

env = gym.make("myoElbowPose1D6MRandom-v0")
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

Follow `adding-a-new-task.md`. In short, build the CPU env first:

1. Copy `basic/arm/reach.py` (or the closest existing task) to a new file.
2. Rewrite `_get_obs_dict` / `get_reward_dict` / `reset_task` for your task.
3. Register it in the suite `__init__.py` with `registry.register_env(...)`.
4. Add its env ID to `test_registry.py`.
5. Run the quality gates (below).

Then, when you need parallel training, add the matched mjlab GPU config under
the same `env_id` — see the CPU/GPU section in `engineering-standards.md`.

## 6. Quality gates (run before every commit)

```bash
pre-commit run --all-files
pytest myosuite/tests/test_registry.py -q
pytest myosuite/tests/test_parity.py -q
```

See `CLAUDE.md` for the full gate list. A pre-commit hook blocks imports of the
deleted `BaseV0`/`env_base.MujocoEnv` classes — if it fires, you copied from an
old example; use `MyoGymnasiumEnv` instead.

## Where to look next

| You want to... | Read |
|---|---|
| Find where something lives | `repository-map.md` |
| Understand the two env patterns | `engineering-standards.md` |
| Write an obs/reward term | `writing-term-functions.md` |
| Add MJX or mjlab support | `mjlab-design-guide.md`, `cross-backend-contract.md` |
| Use an existing helper | `library-usage.md` |
