# Engineering Standards

> **Source of truth priority:** code and tests → [CLAUDE.md](../../CLAUDE.md) → this wiki.
> If this page disagrees with the code, the code wins — please fix this page.

## Design Principles

- Simple, explicit, modular. Flat over nested. Early returns over deep nesting.
- Reuse before adding. Separate config from runtime logic. No hidden side effects.
- Meaningful error messages. No silent `except: pass`.

## Search-Before-Write (mandatory)

Before any new class, function, or wrapper:
1. `grep -r "keyword" myosuite/` — confirm it does not already exist.
2. Check [library-usage.md](library-usage.md) — confirm no library already provides it.
3. If ≥ 80% similar to something existing, extend or import it.

| Kind of code | Canonical location |
|---|---|
| Obs / reward / action / event / termination term functions | `myosuite/terms/` |
| Physics math (quat, fatigue, min-jerk) | `myosuite/physics/` |
| Muscle noise / fatigue / reafferentation / sarcopenia (wrappers and stages) | `myosuite/envs/wrappers.py`, `myosuite/envs/muscle_stages.py` |
| Building an env on either backend (`make_env`, `EnvConfig`) | `myosuite/core/registry.py`, `myosuite/core/config.py` |
| Generic utilities | `myosuite/utils/` |
| MuscleMimic helpers | `myosuite/integrations/musclemimic/` |
| mjlab action/obs/event wiring | `myosuite/envs/myo/backends/mjlab/` |

## Python Standards

- Python 3.10+. PEP 8. `pathlib.Path` over `os.path`.
- Type hints on all signatures. Google-style docstrings on all public APIs.
- `@dataclass` for structured config — never raw `dict` or `ConfigDict` in new code.
- No commented-out code, unused imports, or mutable default arguments.
- No `myo_` prefix on modules/classes inside the package — redundant.

---

## Architecture: CPU and GPU are two matched halves of one task

MyoSuite supports **two backends**, and a fully-supported task has **both**:

- **CPU — `MyoGymnasiumEnv`** — the single-env MuJoCo implementation. This is
  where you inspect, debug, fine-tune, and play back a policy. It is the only
  CPU option and the natural place to start.
- **GPU — mjlab `ManagerBasedRlEnvCfg`** — a matched, massively-parallel
  MuJoCo-Warp implementation of the *same* task (same `env_id`) used for fast RL
  training.

The two share one `env_id` (e.g. `myoLegWalk-v0` is a `MyoGymnasiumEnv` on CPU
**and** a MuJoCo-Warp task on mjlab) and must satisfy the **cross-backend
contract** — identical observation order/scaling, action mapping, and control
timing — so a policy **trained on GPU (mjlab)** runs unchanged when you **play it
back or fine-tune on CPU**. See [cross-backend-contract.md](cross-backend-contract.md); parity is enforced
by `test_parity.py` (CPU) and the mjlab parity tests.

> Rule of thumb: build the CPU env first, then add the matched mjlab GPU config
> when you need parallel training. Keep them in lockstep — changing the
> obs/reward on one side without the other silently breaks policy portability.

### CPU side — `MyoGymnasiumEnv` subclass

Every task in the suite (arm, hand, leg, torso, mimic, challenge) has a CPU
implementation as a `MyoGymnasiumEnv` subclass — explicit and readable
top-to-bottom in one file.

```python
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.muscle_stages import CtrlStageHost

class ReachEnvV0(CtrlStageHost, MyoGymnasiumEnv, EzPickle):   # CtrlStageHost: runs muscle-command wrappers
    def __init__(self, model_path, ..., **kwargs):
        MyoGymnasiumEnv.__init__(self, frame_skip=frame_skip, ...)
        # build self.model / self.data, set action_space + observation_space
    def _get_obs_dict(self, accessor): ...       # -> dict[str, np.ndarray]
    def get_reward_dict(self, obs_dict): ...      # -> {"dense": float, "done": bool, ...}
    def reset_task(self, np_random): ...          # sample per-episode targets
```

Register it in the suite's `__init__.py`:

```python
from myosuite.core import registry
registry.register_env(
    env_id="myoHandReachRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0",
    max_episode_steps=100,
    kwargs={...},
)
```

`MyoGymnasiumEnv` provides the default `step()`/`reset()`/render/close and the
Gymnasium 5-tuple contract. Override `step()`/`reset()` only when needed (e.g.
muscle-activation action mapping, custom init pose). See
`envs/myo/tasks/basic/arm/reach.py` for the canonical, fully-commented example.

### GPU side — mjlab `ManagerBasedRlEnvCfg`

The matched GPU implementation is a mjlab (MuJoCo-Warp) `ManagerBasedRlEnvCfg`
plus a PPO runner config, registered under the **same `env_id`**. Train it with `scripts/train_mjlab.py`.
Read [mjlab-design-guide.md](mjlab-design-guide.md) and
[cross-backend-contract.md](cross-backend-contract.md) before writing one.

- **Twins of the CPU envs** (pose, reach, torso, leg): the config is built from the CPU registration
  (`cpu_reference.cpu_task_spec(env_id)`), so parameters live in one place. A family has an env-config
  function in `envs/myo/backends/mjlab/tasks/<family>/*_env_cfg.py` (for example
  `make_leg_walk_env_cfg` for `myoLegWalk-v0`) and registers its ids, including the Sarc/Fati/Reaf
  variants, with `register_cpu_twins(env_ids, env_cfg_fn, rl_cfg_fn)`. The observation groups are
  `actor` and `critic`.
- **Special tasks** (Table Tennis, ChaseTag, MuscleMimic): hand-written configs registered with
  `register_mjlab_task(...)` in `envs/myo/backends/mjlab/register_mjlab_*.py` and `mimic_mjlab_env.py`.
  Table Tennis and ChaseTag use a single flat `policy` group, which the runner config maps to the
  actor and critic (`obs_groups={"actor": ("policy",), "critic": ("policy",)}`).
- Size the MuJoCo-Warp constraint buffers (`njmax`/`nconmax`) for the task's contact load.

> **Experimental — MJX / `TaskConfig`:** an MJX (JAX/Brax) backend and the
> data-driven `TaskConfig` + `ModularTaskEnv` path also exist (the elbow pose
> and leg directional references and a few challenge tasks), reachable via
> `register_task(..., backends={"cpu","mjx"})`. **MJX is not guaranteed to be
> maintained long-term — do not build new work on it.** Prefer CPU
> (`MyoGymnasiumEnv`) + GPU (mjlab) for anything you need to rely on.

### Non-negotiables (CPU env, GPU backend, or TaskConfig)

- Register via `registry.register_env(...)` or `registry.register_task(...)` —
  never call `gym.register()` directly.
- `step()` returns the 5-tuple `(obs, rwd, terminated, truncated, info)`.
  (`ModularMultiAgentTaskEnv` returns 5-tuples of per-agent dicts — intentional.)
- Term functions are pure and backend-agnostic: use `accessor.array_module()`,
  never import `numpy`/`jax.numpy`/`torch` directly. See [writing-term-functions.md](writing-term-functions.md).
- Assets via `myo_sim.get_path(...)` / `ModelBuilder` — no hardcoded paths.
- Read env attributes (`model`, `data`, `muscle_fatigue`, ...) in library code, scripts and tutorials through
  `env.unwrapped.<attr>`. The short form `env.<attr>` works only on an env straight from `make_env` or behind
  MyoSuite's own wrappers; third-party wrappers (SB3 `Monitor`, gymnasium's `RecordEpisodeStatistics`, ...) do not
  forward attributes.
- The full **CPU env contract** (physics step via `_step_physics`, float32 unclipped observations, `reset(seed=...)`
  fully determining the episode, mocap targets, runtime model edits) and the **mjlab rules** (physics options from
  the CPU model, root velocities, anchored joint-name regexes, per-episode state, step order, vectorised terms) are
  in [CLAUDE.md](../../CLAUDE.md); `test_obs_contract.py`, `test_reset_reproducibility.py` and
  `test_runtime_geometry_consistency.py` check them.

### Base classes (what actually exists)

| Backend | Status | Role | Base | Registered via |
|---|---|---|---|---|
| CPU (Gymnasium) | supported | playback / fine-tune / debug | `myosuite.envs.gymnasium_env.MyoGymnasiumEnv` subclass | `registry.register_env(...)` |
| mjlab (Warp) | supported | parallel GPU training | mjlab `ManagerBasedRlEnvCfg` + runner cfg | `register_cpu_twins(...)` (twins of CPU envs, `backends/mjlab/tasks/`) or `register_mjlab_task(...)` (`register_mjlab_*.py`) |
| MJX (JAX) | experimental | parallel GPU training | `envs/myo/backends/mjx/` env classes (mujoco-playground) | `register_task(..., backends={"mjx"})` from a `TaskConfig` |
| CPU via `ModularTaskEnv` | experimental | data-driven CPU (elbow and leg directional refs) | generic `ModularTaskEnv` from a `TaskConfig` (runs the muscle stages) | `register_task(..., backends={"cpu"})` |

The CPU and mjlab implementations of one task share a single `env_id` and the
cross-backend contract. **MJX is not guaranteed long-term; don't build new work
on it.**

`BaseV0` / `env_base.MujocoEnv` are **deleted** — do not reference them. A
pre-commit hook blocks new imports of them.

---

## Terms Organization (what actually exists)

Term functions live in `myosuite/terms/`, grouped by term *kind* (not motor
domain):

| File | Contents |
|---|---|
| `base_obs.py` | Observation terms (`<key>_obs`) |
| `base_reward.py` | Reward terms (`<term>_reward` or bare `<term>`) |
| `base_action.py` | Action transforms (e.g. `sigmoid_muscle_activation`) |
| `base_event.py` | Reset / event functions |
| `base_termination.py` | Termination terms |
| `mimic_obs.py`, `mimic_reward.py` | MuscleMimic-specific terms |
| `multiplayer/` | Multi-agent (chase-tag) terms |

`ObsSpec.keys` entry `"foo"` resolves to `foo_obs`; `RewardSpec.terms` entry
`"foo"` resolves to `foo_reward` (the `TaskConfig` route). A hand-written
`MyoGymnasiumEnv` calls these helpers directly or computes obs/reward inline.
Because both backends draw from the same term functions, keeping CPU and GPU in
parity is mostly a matter of using the same terms on both sides.

---

## Adding a New Environment (checklist)

**1. CPU implementation (always):**
1. Write the env class in `envs/myo/tasks/<collection>/<effector>/<task>.py`,
   subclassing `MyoGymnasiumEnv`. Implement `_get_obs_dict`, `get_reward_dict`,
   `reset_task`; override `step`/`reset` only if needed.
2. Reuse term helpers from `myosuite/terms/` where they fit.
3. Register with `registry.register_env(...)` in the suite `__init__.py`.
4. Add the env ID to `test_registry.py`; add a CPU parity baseline.
5. Muscle noise, fatigue, reafferentation and sarcopenia are **wrappers**, not env
   kwargs. To let the env take them, run the stages in its action path
   (`MuscleActionMixin` / `CtrlStageHost`: `_run_ctrl_stages` before writing `ctrl`,
   `_run_reset_stages` in `reset`); see [cross-backend-contract.md](cross-backend-contract.md).

**2. Matched mjlab GPU config (when you need parallel training):**
6. Add a `ManagerBasedRlEnvCfg` + PPO runner config under the **same `env_id`**: for a twin of a basic
   CPU env, an env-config function in `backends/mjlab/tasks/<family>/` registered with
   `register_cpu_twins`; for a special task, `register_mjlab_task` in `backends/mjlab/register_mjlab_*.py`
   (set `njmax`/`nconmax` — see the GPU section above).
7. Honour [cross-backend-contract.md](cross-backend-contract.md) (obs order/scale, action mapping,
   `ctrl_dt`) and add a CPU↔mjlab parity test so the two cannot drift. To make the
   twin take features, build its config from the CPU registration
   (`cpu_reference.cpu_task_spec`, `register_cpu_twins`).

The experimental data-driven `TaskConfig` route (CPU + MJX from one spec) is
walked through in [adding-a-new-task.md](adding-a-new-task.md).

## Parity Policy

`pytest myosuite/tests/test_parity.py -v` after every env change. CPU parity is
gated at `atol ≤ 1e-6` (some contact-rich envs are relaxed); regressions block
PRs. To regenerate a baseline after an
*intentional* change (with the package versions CI resolves, see [CLAUDE.md](../../CLAUDE.md)):

```bash
python scripts/generate_parity_baselines.py --env-id <env-id>
```

## Quality Gates

- Add/update tests for non-trivial behavior changes.
- `pre-commit run --all-files` must pass.
- Self-review: smallest change that solves the problem? Searched first? No duplication?
- A change to a registered env's observations, actions, dynamics or task distribution needs a CHANGELOG entry and a
  note in `docs/baseline_checkpoints.md`: published checkpoints are tied to the old contract.

```bash
pre-commit run --all-files
pytest myosuite/tests/test_model_builder.py -v
pytest myosuite/tests/test_terms_cpu.py -v
pytest myosuite/tests/test_fragment_compat.py -v
pytest myosuite/tests/test_parity.py -v
pytest myosuite/tests/test_registry.py -v
```
