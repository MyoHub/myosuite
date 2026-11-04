# BoxingVs reward v1 (capstone 2026)

Config options for `myoChallengeBoxingVs-v0` that add physical contact between the
boxers, contact-based hits, a reworked reward, start-pose randomisation and a held
dummy opponent. **Every default reproduces the old behaviour exactly** (checked:
identical rewards, observations and info on seeded rollouts). Observations are
unchanged (603-D per agent).

## Train right away (Level 1: learner vs a standing dummy)

`myoChallengeBoxingVsLevel1-v0` is a normal single-agent env (603-D obs, 354 muscle
actions in [0, 1]) built from the preset below, so the usual script works:

```bash
uv run python scripts/train_sb3.py myoChallengeBoxingVsLevel1-v0 --timesteps 1000000 --normalize
```

Smoke-tested: SB3 `check_env` passes and a PPO run trains, saves and evaluates
(~125 steps/s on a laptop CPU, i.e. ~2 h per million steps; `train_sb3.py` uses a
`DummyVecEnv`, so `--n-envs` does not run envs in parallel). Override any config field from Python with
`make_env("myoChallengeBoxingVsLevel1-v0", agent_separation_m=0.5)`. To fade the
shaping out, call `vec_env.env_method("set_shaping_alpha", alpha)` from a callback
(`train_sb3.py` keeps it at 1). Expect a policy trained from scratch to mostly learn
to not fall at first.

To watch a trained model (side view, step / return / hits drawn on the frames):

```bash
uv run python scripts/record_sb3_video.py myoChallengeBoxingVsLevel1-v0            # logs/sb3/<env>/model.zip
uv run python scripts/record_sb3_video.py myoChallengeBoxingVsLevel1-v0 --policy zero   # the env alone
```

Videos are written to `logs/videos/<env_id>/` (git-ignored, share them on the drive).
Options: `--model`, `--episodes`, `--speed 0.5` (slow motion), `--azimuth/--elevation/--distance`.

## Quick start in Python (Level 1: learner = `agent_0` vs a dummy)

```python
from myosuite.envs.myo.tasks.challenge.boxing_vs import (
    BoxingVsEnv, BoxingVsSingleAgentWrapper, boxing_vs_level1_config)

base = BoxingVsEnv(boxing_vs_level1_config())          # override any field: (..., agent_separation_m=0.5)
env = BoxingVsSingleAgentWrapper(base, agent_id="agent_0")  # opponent actions are ignored (dummy)
obs, info = env.reset(seed=0)

# anneal shaping during training (1 -> 0), e.g. from a callback:
base.set_shaping_alpha(0.5)
```

`boxing_vs_level1_config()` turns on: `cross_agent_collisions`, `massless_visual_gear`,
`hit_detection="contact"`, `reward_mode="v1"`, `reset_pose="clip"`, small reset noise,
`dummy_opponent="agent_1"`, `agent_separation_m=0.4` (pelvis gap ≈ 0.6 m).
Keep the learner as `agent_0`: the MuscleMimic observation is in the world frame and
the clip heading matches `agent_0`, not the 180°-rotated `agent_1`.

## Reward v1

```
r = r_hit · (damage dealt − damage received)                 # never annealed
  + shaping_alpha · (r_alive + r_reach · Δreach + r_facing · cos(heading, opponent))
  + r_effort · mean(clip(action, 0, 1)²)
  + fall term (attributed) + r_ko_bonus · [opponent KO]
```

| Term | Default | Notes |
|---|---|---|
| `r_hit` | 1.0 | damage per hit = `w_head` (2) or `w_body` (1) × min(v, 4.5)/4.5 |
| `r_alive` | 0.01 | 0 on the step you fall |
| `r_reach` | 1.0 | per metre the closest fist gets closer to a target zone (potential difference, clipped ±0.1 m/step) |
| `r_facing` | 0.01 | per step |
| `r_effort` | −0.1 | on the **clipped** command, mean over muscles |
| fall | ±5 | I fall: `r_self_fall_penalty`; opponent falls within `knockdown_window_s` (1 s) of my hit: `r_fall_bonus`; opponent slips: 0; both fall: `r_double_fall_penalty` each |

`fall_reward_mode="attributed"` also works with the legacy reward.

## Contact-based hits (`hit_detection="contact"`)

* Collision masks (bits: floor 1, agent 0 = 2, agent 1 = 4): boxers collide with each
  other and the floor, never with themselves; fingers are floor-only. One massless
  sphere per glove (`glove_collision_radius_m`, `glove_solref`).
* A hit is the **start** of a glove contact with the opponent (re-armed when the glove
  leaves, plus `hit_cooldown_s` = 0.15 s). Leaning or clinching never scores twice.
* Zone: head (`head`), body (`thorax`, `torso`), block (opponent forearm, upper arm or
  glove). Head > body > block if several are touched.
* Speed = approach speed along the contact normal, from the previous control step.
  Below `hit_min_speed_mps` (1.5) it is a touch, not a hit. Blocks do no damage.
* Glove on glove: only the glove moving faster towards the other counts as a block.

## Reset and opponent

* `reset_pose="clip"`: joint angles + root height/tilt from `reset_clip_frame` of
  `reset_clip_path` (push clip by default); placement and facing are kept.
* `reset_noise_*`: separation, lateral offset, yaw, joint angles (seeded by `reset(seed)`).
* `dummy_opponent="agent_1"`: that agent is reset to its start pose with muscles off at
  every control step (via the new no-op-by-default hook
  `MultiAgentTaskConfig.pre_physics_step`). It cannot fall. You can lean on it.
* `massless_visual_gear=True`: the visual glove/helmet meshes otherwise add ~2.1 kg per
  hand and ~5.8 kg to the head (~10 kg per agent) compared with the MuscleMimic model.

## Logged in `info` (per step, `<agent>` = agent_0 / agent_1)

* `reward/<term>/<agent>`: every reward term (legacy or v1); they sum to the reward.
* `hits_head`, `hits_body`, `blocked`, `hits_r`, `hits_l`, `impact_speed_max`
  (`/<agent>`, as attacker). Zero with proximity hits.

## Pretrained BC policy (`boxing_bc.pt`, PR #397) as agent_0: smoke test

Steps before the pelvis drops below 0.7 m (500 = whole round), 3 seeds, dummy opponent:

| Setup | Fall step | Hits landed |
|---|---|---|
| legacy defaults | 250, 250, 250 | 0 |
| `massless_visual_gear` only | 130, 130, 130 | 0 |
| massless + `reset_pose="clip"` | 227, 227, 227 | 0 |
| `boxing_vs_level1_config()` | 250, 500, 240 | 0, 1, 0 |

The base policy is not stable for a full round in any setting: the motor prior
(frozen / residual / fine-tuned) has to fix that, not the reward.

## Not in v1 (deliberately)

KO/health rebalance (KO is still unreachable), round outcome ±W at time-out (would
need `terminated` at time-out and time-remaining in the observation), ropes, longer
rounds, egocentric observations for self-play.

## Tests

`uv run python -m pytest myosuite/tests/test_boxing_vs_reward_v1.py` (28 tests, ~15 s):
legacy defaults, masks, masses, real glove impacts on the head/body (one event per
punch, slow touch and leaning score nothing), synthetic block / glove-glove / cooldown
cases, falls, annealing, reset seeding, dummy.
