<!-- =================================================
# Copyright (c) MyoSuite Authors
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
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

---

## Start here

| I am a…                    | Start here                                                   |
| --------------------------- | ------------------------------------------------------------ |
| **ML / RL**           | [ML guide](docs/source/quickstart_ml.rst)                     |
| **Biomechanics**      | [Biomechanics guide](docs/source/quickstart_biomechanics.rst) |
| **Neuroscience**      | [Neuroscience guide](docs/source/quickstart_neuroscience.rst) |
| **Rehab / sports**    | [Rehab guide](docs/source/quickstart_rehabilitation.rst)      |
| **Changing the code** | [Developer getting started](docs/wiki/getting-started.md)     |

---

## Install

Python 3.10, 3.11, 3.12, and 3.13 are currently supported. From source (recommended):

```bash
git clone https://github.com/MyoHub/myosuite.git
cd myosuite
pip install -e ".[rl]"          # CPU training (Stable-Baselines3)
# pip install -e ".[mjlab]"     # GPU (Linux + CUDA) -- see docs/source/install.rst
                                 # for matching the torch build to your driver's CUDA version
```

Or: `uv sync -p 3.10 --extra rl`.

From PyPI: `pip install -U myosuite`. Sim assets (`myo-sim`, `furniture-sim`, …) are installed as packages — no git submodules.

Verify (replace "onscreen" with "offscreen" when running on a remote, headless machine):

```bash
python -c "import myosuite; print(len(myosuite.myosuite_env_suite), 'envs')"
python -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0 --render onscreen
# macOS viewer: mjpython -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0 --render onscreen
```

---

## Quick start

```python
import gymnasium as gym
import myosuite  # registers environments

env = gym.make("myoElbowPose1D6MRandom-v0")
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
import gymnasium as gym
import myosuite

env = gym.make("myoElbowPose1D6MRandom-v0")
model = PPO("MlpPolicy", env, device="cpu")
model.learn(total_timesteps=100_000)
```

Train on GPU (same `env_id`, mjlab / RSL-RL):

```bash
python scripts/train_mjlab.py myoElbowPose1D6MFixed-v0 --env.scene.num-envs 1024
```

Pathological variants use prefixes, not a `Fatigue` infix: `myoSarcElbowPose1D6MRandom-v0`, `myoFatiElbowPose1D6MFixed-v0`, `myoReafHandPoseRandom-v0`.

---

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

| Backend                       | Use                   | How                                                                    |
| ----------------------------- | --------------------- | ---------------------------------------------------------------------- |
| **CPU** (Gymnasium)     | playback, debug, SB3  | `gym.make(env_id)`                                                   |
| **mjlab** (MuJoCo Warp) | parallel GPU training | `pip install -e ".[mjlab]"` then `scripts/train_mjlab.py <env_id>` |

A task’s CPU and mjlab halves share one `env_id` (see [cross-backend contract](docs/wiki/cross-backend-contract.md)). An MJX (JAX) path also exists; it is **experimental** and not the supported training route.

---

## Tutorials

See [`tutorials/README.md`](tutorials/README.md). Start with `1.1_Get_Started.ipynb` then `2.1_Train_SB3_Policy.ipynb`.

GPU walk-through: [`tutorials/2.2_Train_MjLab_Policy.ipynb`](tutorials/2.2_Train_MjLab_Policy.ipynb).

Full-body MuscleMimic playback and training: [`myosuite/integrations/musclemimic/README.md`](myosuite/integrations/musclemimic/README.md).

---

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
