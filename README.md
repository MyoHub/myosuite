<!-- =================================================
# Copyright (c) MyoSuite Authors
================================================= -->
<img src="https://github.com/myohub/myosuite/blob/main/docs/source/images/Full%20Color%20Horizontal%20wider.png?raw=true" width=800>

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.facebook.com/support-ukraine)
[![PyPI](https://img.shields.io/pypi/v/myosuite)](https://pypi.org/project/MyoSuite/)
[![Documentation Status](https://readthedocs.org/projects/myosuite/badge/?version=latest)](https://myosuite.readthedocs.io/en/latest/)
![PyPI - License](https://img.shields.io/pypi/l/myosuite)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/myohub/myosuite/blob/main/docs/CONTRIBUTING.md)
[![Downloads](https://static.pepy.tech/badge/myosuite)](https://pepy.tech/project/myosuite)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zFuNLsrmx42vT4oV8RbnEWtkSJ1xajEo)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://join.slack.com/t/myosuite/shared_invite/zt-1zkpw2zzk-NhVhVlSDxhoMHbzROD8gMA)
[![Twitter Follow](https://img.shields.io/twitter/follow/MyoSuite?style=social)](https://twitter.com/MyoSuite)

**MyoSuite** is a collection of musculoskeletal environments and tasks simulated with the [MuJoCo](http://www.mujoco.org/) physics engine. It serves researchers and practitioners across biomechanics, neuroscience, machine learning, sports medicine, and physical rehabilitation.

[Documentation](https://myosuite.readthedocs.io/en/latest/) · [Tutorials](tutorials/) · [Task list](https://myosuite.readthedocs.io/en/latest/environments.html)

<img width="1240" alt="TasksALL" src="./docs/source/images/MyoSuiteHeader.png?raw=true">


<details>
  <summary><h2>What's new in MyoSuite 3</h2></summary>

MyoSuite 3 brings the whole suite to fast, scalable training while keeping the simple interface you know:

- **Much faster learning.** Train with thousands of environments in parallel on a single GPU, then replay the policy on the CPU.
- **One task, several backends.** The same `env_id` runs on your **CPU** through the standard [Gymnasium](https://gymnasium.farama.org/) interface (to explore, debug and replay policies, or to train with libraries such as Stable-Baselines3), on **mjlab** for massively parallel GPU training, and on an **experimental MJX** (JAX) path.
- **Muscle conditions as features.** Motor noise, fatigue, sarcopenia and reafferentation are wrappers, not env arguments. `make_env(EnvConfig(env_id, features=...))` applies them identically on the CPU env and the GPU twin, together with the episode length and control step ([cross-backend contract](docs/wiki/cross-backend-contract.md)).
- **MuscleMimic support.** Run, evaluate and train full-body and bimanual **MuscleMimic** policies, with ready-to-use checkpoints and motion datasets.
- **Updated musculoskeletal models.** `myo-sim` moved from a moving `dev` branch to a pinned PyPI release (0.2.3) with some model updates, including a torso/pelvis frame fix for the leg models ([myo_sim#132](https://github.com/MyoHub/myo_sim/pull/132)).
- **The complete MyoChallenge suite** as Gymnasium environments: Baoding, Bimanual, Chase Tag, Die Reorient, OSL Run, Relocate, Soccer and Table Tennis.
- **New tasks in a few lines.** Describe a task with a compact spec and reuse the shared observation, reward and model-building blocks instead of writing an environment class.
- **Some existing environments changed.** Observations are no longer clipped and are read after a fresh forward step (55 env ids), the `motorFinger*` envs have 4x stronger motors, the Random finger-reach tasks now sample only targets the fingertip can reach, and several reset and seed behaviours were corrected. Policies trained with MyoSuite 2.x or earlier snapshots may need retraining; see [`CHANGELOG.md`](CHANGELOG.md) for the full list.
- **Ready to use.** Default trained policies on [Hugging Face](https://huggingface.co/myohub/myosuite-3-baselines) (see [`docs/baseline_checkpoints.md`](docs/baseline_checkpoints.md) for the full list and success rates) with evaluation videos, plus updated tutorials from the first rollout to GPU training and MuscleMimic. See the [changelog](CHANGELOG.md) for everything that changed since v2.12.
</details>

## Start here

| I am a…                    | Start here                                                   |
| --------------------------- | ------------------------------------------------------------ |
| **ML / RL**           | [ML guide](docs/source/quickstart_ml.rst)                     |
| **Biomechanics**      | [Biomechanics guide](docs/source/quickstart_biomechanics.rst) |
| **Neuroscience**      | [Neuroscience guide](docs/source/quickstart_neuroscience.rst) |
| **Rehab / sports**    | [Rehab guide](docs/source/quickstart_rehabilitation.rst)      |
| **Changing the code** | [Developer getting started](docs/wiki/getting-started.md)     |


## Install

Python 3.10–3.14 is supported (the GPU `[mjlab]` extra needs Python ≤3.13). From source (recommended):

```bash
git clone https://github.com/MyoHub/myosuite.git
cd myosuite
pip install -e ".[rl]"          # CPU training (Stable-Baselines3)
# pip install -e ".[mjlab]"     # GPU (Linux + CUDA) -- see docs/source/install.rst
                                 # for matching the torch build to your driver's CUDA version
```

Or: `uv sync -p 3.10 --extra rl`.

From PyPI: `pip install -U myosuite`. Musculoskeletal models come from the `myo-sim` package; the few MPL/YCB/furniture assets used are bundled — no git submodules.

Verify (replace "onscreen" with "offscreen" when running on a remote, headless machine):

```bash
python -c "import myosuite; print(len(myosuite.myosuite_env_suite), 'envs')"
python -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0 --render onscreen
# macOS viewer: mjpython -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0 --render onscreen
```

Run these from a directory that does not directly contain a folder named `myosuite` (for example not the parent of your clone); otherwise Python imports that folder as a namespace package and fails with `cannot import name ... from 'myosuite' (unknown location)`.


## Quick start

```python
from myosuite import make_env

env = make_env("myoElbowPose1D6MRandom-v0")
obs, info = env.reset(seed=0)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

`info["obs_dict"]` and `info["rwd_dict"]` break down the observation and reward every step.

Train on CPU:

```python
from stable_baselines3 import PPO
from myosuite import make_env

env = make_env("myoElbowPose1D6MRandom-v0")
model = PPO("MlpPolicy", env, device="cpu")
model.learn(total_timesteps=100_000)
```

Train on GPU (same `env_id`, mjlab / RSL-RL):

```bash
python scripts/train_mjlab.py myoElbowPose1D6MFixed-v0 --env.scene.num-envs 1024
```

The same call builds either backend, and an `EnvConfig` overrides the registered defaults (episode length, control step,
muscle-command features such as motor noise and fatigue) for both:

```python
from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.envs.wrappers import FatigueWrapper, MotorNoiseWrapper

env = make_env("myoElbowPose1D6MRandom-v0")                                   # CPU, registered defaults
envs = make_env("myoElbowPose1D6MRandom-v0", backend="mjlab", num_envs=1024)  # GPU twin (needs the mjlab extra)

cfg = EnvConfig(
    "myoElbowPose1D6MRandom-v0",
    max_episode_steps=300,
    features=((MotorNoiseWrapper, {"motor_noise": {"constant_std": 0.05}}), FatigueWrapper),
)
env = make_env(cfg)                                   # CPU with noise and fatigue
envs = make_env(cfg, backend="mjlab", num_envs=1024)  # the same on the GPU
```

Pathological variants use prefixes, not a `Fatigue` infix: `myoSarcElbowPose1D6MRandom-v0`, `myoFatiElbowPose1D6MFixed-v0`, `myoReafHandPoseRandom-v0`.



## Environments

| Body      | Example IDs                                                     |
| --------- | --------------------------------------------------------------- |
| Elbow     | `myoElbowPose1D6MRandom-v0`, `myoFatiElbowPose1D6MFixed-v0` |
| Finger    | `myoFingerPoseRandom-v0`, `myoFingerReachRandom-v0`         |
| Hand      | `myoHandPoseRandom-v0`, `myoChallengeBaodingP2-v1`          |
| Arm       | `myoArmReachRandom-v0`                                        |
| Leg       | `myoLegWalk-v0`, `myoLegDirectionalForward-v0`              |
| Full body | `myoMimicFullbody-v0`                                         |

List every registered CPU ID: `python -c "import myosuite; print('\n'.join(myosuite.myosuite_env_suite))"`. Annotated catalog: [environments](https://myosuite.readthedocs.io/en/latest/environments.html).

### Backends

| Backend | Use | Build an env in Python | Train |
| --- | --- | --- | --- |
| **CPU** (Gymnasium) | playback, debug, SB3 | `make_env(env_id)` | Stable-Baselines3 (see Quick start) |
| **mjlab** (MuJoCo Warp) | parallel GPU training | `make_env(env_id, backend="mjlab", num_envs=...)` | `scripts/train_mjlab.py <env_id>` |

The mjlab backend needs `pip install -e ".[mjlab]"`. `make_env` only builds the env (to step it, evaluate a policy or write your own training loop); `scripts/train_mjlab.py` is the ready-made PPO training script.

A task’s CPU and mjlab halves share one `env_id` (see [cross-backend contract](docs/wiki/cross-backend-contract.md)). `make_env(EnvConfig(env_id, backend=..., features=...))` builds either half with the same episode length, control step and muscle-command features. An MJX (JAX) path also exists; it is **experimental** and not the supported training route.


## Tutorials

See [`tutorials/README.md`](tutorials/README.md). Start with `1.1_Get_Started.ipynb` then `2.1_Train_SB3_Policy.ipynb`.

GPU walk-through: [`tutorials/2.2_Train_MjLab_Policy.ipynb`](tutorials/2.2_Train_MjLab_Policy.ipynb).

Full-body MuscleMimic playback and training: [`myosuite/integrations/musclemimic/README.md`](myosuite/integrations/musclemimic/README.md).


## License

[Apache License](LICENSE).

## Citation

```bibtex
@Misc{MyoSuite2026,
  author =       {Vittorio, Caggiano AND Balint, Hodossy AND Florian, Fischer, AND Cheryl, Wang, AND MyoSuiteTeam},
  title =        {MyoSuite 3.0 -- A multimodal platform for efficient and scalable musculoskeletal motor control},
  publisher =    {arXiv},
  year =         {2026},
  howpublished = {\url{https://github.com/myohub/myosuite}},
  doi =          {...},
  url =          {...},
}
```

```bibtex
@Misc{MyoSuite2022,
  author =       {Vittorio, Caggiano AND Huawei, Wang AND Guillaume, Durandau AND Massimo, Sartori AND Vikash, Kumar},
  title =        {MyoSuite -- A contact-rich simulation suite for musculoskeletal motor control},
  publisher =    {arXiv},
  year =         {2022},
  howpublished = {\url{https://github.com/myohub/myosuite}},
  doi =          {10.48550/ARXIV.2205.13600},
  url =          {https://arxiv.org/abs/2205.13600},
}
```
