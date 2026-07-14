# MyoChallenge 2026 — Baseline Policies

Pre-trained baselines for all three competition environments. Load and run directly — no retraining required.

```
tutorials/mc26/baselines/
├── boxing/
│   └──  mannequin_exact_clone.npz  # P0 policy + opponent for myoChallengeBoxingVsClone-v0
└── saber/
    └── saber_p0_baseline.zip      # SB3 PPO for myoChallengeSaberP0-v0
```

---

## `myoChallengeBoxingP0-v0`

**Demo:** `docs/baseline_boxing_p0.mp4`

Single MuscleMimic full-body agent (354 muscles) hitting six fixed pads. Observation: 603-dim flat vector (kinematics + target keypoints). Action: muscle activations `(354,)` ∈ `[0, 1]`.

### Baseline policy

```python
from scripts.boxing_bc import StandaloneBCPolicy
import myosuite, gymnasium as gym

myosuite.register_all_envs()
policy, _ = StandaloneBCPolicy.load("tutorials/mc26/baselines/boxing/mannequin_exact_clone.npz")

env = gym.make("myoChallengeBoxingP0-v0")
obs, _ = env.reset(seed=0)
done = False

while not done:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

### Render to MP4

```bash
uv run python scripts/boxing_bc.py \
    --policy-in tutorials/mc26/baselines/boxing/mannequin_exact_clone.npz \
    --env-id myoChallengeBoxingP0-v0 \
    --skip-teacher-compare \
    --video-out boxing_p0_demo.mp4 \
    --horizon 300 \
    --cam-azimuth 0 --cam-elevation -60 --cam-distance 2.5 \
    --cam-lookat-x -0.05 --cam-lookat-y 0.0 --cam-lookat-z 1.0 \
    --width 960 --height 720
```

---

## `myoChallengeBoxingVs-v0`

**Demo:** `docs/baseline_boxing_vs.mp4`

Two MuscleMimic full-body agents (354 muscles each) stepping simultaneously. Observation per agent: 603-dim (kinematics + opponent keypoints + health). Action per agent: `(354,)` ∈ `[0, 1]`. Episode ends on KO (health ≥ 100) or fall (pelvis < 0.7 m).

### Multi-agent baseline (two numpy MLPs)

Pure numpy MLP, no extra dependencies:

```python
import numpy as np
import myosuite, gymnasium as gym

myosuite.register_all_envs()

d = np.load("tutorials/mc26/baselines/boxing/boxing_vs_baseline.npz")
mean, scale = d["mean"], d["scale"]
W0, b0, W1, b1, W2, b2 = d["W0"], d["b0"], d["W1"], d["b1"], d["W2"], d["b2"]

def predict(obs: np.ndarray) -> np.ndarray:
    x = (obs.astype(np.float32) - mean) / np.maximum(scale, 1e-8)
    x = np.tanh(x @ W0 + b0)
    x = np.tanh(x @ W1 + b1)
    return np.clip(x @ W2 + b2, 0.0, 1.0)

env = gym.make("myoChallengeBoxingVs-v0")
obs, _ = env.reset(seed=0)
done = False

while not done:
    actions = {aid: predict(obs[aid]) for aid in obs}
    obs, rewards, terminated, truncated, info = env.step(actions)
    done = any(terminated.values()) or any(truncated.values())

env.close()
```

### Single-agent training against the clone (`myoChallengeBoxingVsClone-v0`)

The mannequin clone policy is also wired as the **built-in opponent** in a single-agent variant. The opponent runs the same JAX MuscleMimic clone from `mannequin_exact_clone.npz`; the learning agent controls `agent_0` via a standard Gymnasium API:

```python
import myosuite, gymnasium as gym

myosuite.register_all_envs()

env = gym.make("myoChallengeBoxingVsClone-v0")
obs, _ = env.reset(seed=0)
done = False

while not done:
    action = env.action_space.sample()          # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

The opponent's action is computed automatically from the raw physics state using `FullbodyObsAdapter` — no extra setup needed.

> **Why can't I use `PPO.load("tutorials/mc26/baselines/boxing/mannequin_exact_clone.npz")`?**
>
> `PPO.load` is from stable-baselines3 and expects a PyTorch state-dict checkpoint (`.zip`).
> `mannequin_exact_clone.npz` uses a completely different format: it stores a pickled JAX
> parameter PyTree inside a NumPy bytes array, together with running-normalisation stats.
> Use `StandaloneBCPolicy.load(...)` from
> `myosuite.integrations.musclemimic.fullbody_local_policy` instead.

### Info dict keys

| Key | Description |
|-----|-------------|
| `health/agent_0`, `health/agent_1` | Cumulative damage received (0–100) |
| `damage_delivered/agent_0`, `damage_delivered/agent_1` | Damage dealt this step |
| `fell/agent_0`, `fell/agent_1` | Whether each agent is down |
| `ko/agent_0`, `ko/agent_1` | Whether each agent reached KO threshold |

---

## `myoChallengeSaberP0-v0`

**Demo:** `docs/baseline_saber_p0.mp4`

Single MuscleMimic bimanual agent (arms + torso, 336 muscles) wielding two lightsabers, striking spherical targets from a pool of 100. Observation: 349-dim (joint kinematics + 10 nearest targets × 7 floats + time). Action: `(336,)` ∈ `[0, 1]`. Episodes run 600 steps (6 s at 10 ms).

### Baseline policy

```python
from stable_baselines3 import PPO
import myosuite, gymnasium as gym

myosuite.register_all_envs()
model = PPO.load("tutorials/mc26/baselines/saber/saber_p0_baseline.zip")

env = gym.make("myoChallengeSaberP0-v0")
obs, _ = env.reset(seed=0)
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
```

### Render to MP4

```bash
uv run python scripts/lightsaber_render.py \
    --ppo-model-path tutorials/mc26/baselines/saber/saber_p0_baseline.zip \
    --output saber_p0_demo.mp4 \
    --steps 600 \
    --width 960 --height 720
```

### Interactive viewer (macOS)

```bash
uv run mjpython -c "
from stable_baselines3 import PPO
import myosuite, gymnasium as gym
myosuite.register_all_envs()
model = PPO.load('tutorials/mc26/baselines/saber/saber_p0_baseline.zip')
env = gym.make('myoChallengeSaberP0-v0', render_mode='human')
obs, _ = env.reset(seed=0)
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
"
```
