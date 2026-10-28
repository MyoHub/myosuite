# MyoHalloween: task, controller, and tutorial

Load the registered environment through MyoSuite's existing factory. The task
owns physical geometry, reset, launch scheduling, collision detection and metrics.
The separate `HalloweenController` selects references and calls the existing
`TerrainController`; it never advances or overwrites the live actor state.

## Run the working TERRA example

Install the feature revision pinned by
[`requirements-student.txt`](../tutorials/files/halloween/requirements-student.txt)
while TERRA is awaiting review. This branch includes the pending TERRA integration.
Use Python 3.12+; CPU inference is sufficient. Rendering is verified with EGL on
Linux/Colab. The environment itself performs no network access.

The student pins also respect Colab’s preinstalled Numba, Datasets, Diffusers,
and Gradio requirements: NumPy 2.2.6, Hugging Face Hub 1.32.0, and Pillow 12.2.0.
After running an older install cell, use **Runtime → Disconnect and delete runtime**
before running the updated notebook so imported binary libraries are not stale.

```python
from myosuite import make_env
from myosuite.integrations.musclemimic import download_terrain_checkpoint
from myosuite.integrations.musclemimic.halloween import (
    HalloweenController,
    download_halloween_assets,
)

env = make_env("myoHalloween-v0", profile="terra", max_episode_steps=3000)
try:
    obs, info = env.reset(seed=2027)
    controller = HalloweenController(
        env.unwrapped, download_terrain_checkpoint(), download_halloween_assets()
    )
    controller.reset(info)
    while True:
        action = controller(env.unwrapped.data, info)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(info["halloween"]["events"])
finally:
    env.close()
```

The [Python lesson](../tutorials/4.5_Halloween_Avoidance.py) and
[Colab notebook](../tutorials/4.5_Halloween_Avoidance.ipynb) add comments, recording,
quantitative assertions, and exercises. The notebook and Python code match.

## Stable muscle-action interface, explicit profiles

| | Default `profile="benchmark"` | Opt-in `profile="terra"` |
|---|---|---|
| Environment ID | `myoHalloween-v0` | `myoHalloween-v0` |
| Backend | CPU + MJLab | CPU showcase; no MJLab counterpart claimed |
| Actions | 354 excitations in [0, 1] | 354 excitations in [0, 1] |
| Observations | 535 float32 values | Same ordering and shape |
| Physics/control | Euler; 5 × 2 ms | Calibrated TERRA actor; 5 × 2 ms |
| Projectile | 0.09 m radius, 0.35 kg; either Y side | 0.065 m radius, 0.25 kg; frontal +Y |
| Reset | Existing benchmark keyframe and seeded schedule | Bundled settled keyframe, seeded wait and policy seed |
| Completion | Hit, fall, or six-second time limit | Hit, fall, or three throws and recovery |
| Policy | Train/load a matching muscle-action checkpoint | Separate frozen TERRA + decoder + selector |

The default benchmark's settings, actions, observations, CPU/MJLab behavior and
six-second limit remain unchanged. The TERRA profile is an explicit, different
physical curriculum; benchmark policies are not automatically compatible with it.
Set `max_episode_steps=3000` for its roughly 24-second sequence. Leaving the
registered 600-step limit in place truncates it early. `EnvConfig.task_kwargs`
may also select the profile for CPU; it is not a GPU training configuration.

The observation order is root height/quaternion/local linear and angular velocity
(11), joint positions (82), velocities (82), activations (354), relative pumpkin
position and world velocity (6). The controller also receives the current MuJoCo
state to construct TERRA's existing site-based observations.

## Customize the physical task

```python
from dataclasses import replace
from myosuite.envs.myo.tasks.halloween.config import AvoidanceConfig

defaults = AvoidanceConfig()
throws = (replace(defaults.throws[0], target_x=-0.13), *defaults.throws[1:])
config = replace(defaults, throws=throws)
env = make_env(
    "myoHalloween-v0", profile="terra", avoidance_config=config,
    max_episode_steps=3000, render_mode="rgb_array",
)
```

Targets use world X for lateral displacement and absolute world Z height. Launch
X/Y is relative to the reset's home position; pumpkins travel toward positive Y.
The environment samples the first wait from its own reset RNG. Later waits and
recovery windows are explicit config fields. No future throw list is passed to
the learned selector. A new launch is written at an observation boundary, so the
external controller sees the projectile before responding on the next step.

`info["halloween"]` exposes the policy seed, home XY, launch ID, control steps
since launch, completed events and completion flag. Each event records contact
rows, sampled minimum surface distance, minimum pelvis height, maximum/final
home distance and upright recovery. `info["rwd_dict"]` retains hit/fall flags and
reward components. Both are actual measurements; ending the episode is not by
itself success. Misses and failed recoveries must be inspected separately.

## Code ownership

- [`halloween_model.py`](../myosuite/envs/myo/tasks/halloween/halloween_model.py):
  shared projectile/contact recipe and profile-specific actor construction.
- [`config.py`](../myosuite/envs/myo/tasks/halloween/config.py) and
  [`avoidance.py`](../myosuite/envs/myo/tasks/halloween/avoidance.py): typed physical
  curriculum, launch scheduling and per-throw scoring.
- [`cpu.py`](../myosuite/envs/myo/tasks/halloween/cpu.py): existing Gymnasium task,
  reset/step/render lifecycle and muscle-action API.
- [`integrations/musclemimic/halloween.py`](../myosuite/integrations/musclemimic/halloween.py):
  verified learned-data downloads, command references and selection. Uses the
  existing TERRA policy loader/runner without another inference implementation.

The environment can be constructed/reset/tested offline without JAX, TERRA weights
or a selector. Downloading and loading weights is an explicit controller step.
Repeat **both** `env.reset(seed=...)` and `controller.reset(info)` for repeatable
policy noise. Reference changes do not reset the live physical actor.

## Validation and limits

The seed-2027 showcase is regression-checked against the previous explicit lesson:
2,390 actor states are identical in qpos, qvel and activations. It avoids three
throws with zero contacts and finishes each recovery within 10 cm of home.
Contacts are checked at every physics substep; surface distance is sampled near
the body at control rate and is not a continuous-distance guarantee.

Fast tests cover unchanged benchmark creation/step, TERRA profile reset, launch
visibility, real-body-hit termination, invalid configs and detached info snapshots.
The MJLab benchmark remains experimental: CPU/Warp smoke execution was validated;
CUDA dynamics and learnability require their own hardware gate. The showcase's
learned curriculum covers frontal upper-body throws; arbitrary-threat robustness
and jump/low-throw avoidance are not established.

Kenney pumpkin assets are CC0; bundled settled TERRA state provenance is recorded
in the asset README. Public decoder/selector files retain their existing terms;
TERRA weights retain their upstream terms.
