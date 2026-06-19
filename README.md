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

[Documentation](https://myosuite.readthedocs.io/en/latest/) | [Tutorials](tutorials/) | [Task list](https://github.com/myohub/myosuite/blob/main/docs/source/suite.rst#tasks)

<img width="1240" alt="TasksALL" src="./docs/source/images/MyoSuiteHeader.png?raw=true">


---

## Who is MyoSuite for?

| I am a... | I want to... | Start here |
|-----------|-------------|------------|
| **Biomechanist** | Extract joint kinematics, muscle forces, inverse dynamics | [Biomechanics Guide](docs/source/quickstart_biomechanics.rst) |
| **Neuroscientist** | Model sensory feedback, reflex controllers, neural fatigue | [Neuroscience Guide](docs/source/quickstart_neuroscience.rst) |
| **ML / RL Researcher** | Train agents, benchmark backends, reproduce baselines | [ML Guide](docs/source/quickstart_ml.rst) |
| **Sports / Rehab Clinician** | Simulate pathological conditions, assistive devices, rehabilitation tasks | [Rehab Guide](docs/source/quickstart_rehabilitation.rst) |

---

## Installation

Requires Python 3.10 or later.

### Using uv (fastest)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager:

```bash
uv sync -p 3.10
```

### Using conda / pip

```bash
conda create --name myosuite python=3.10
conda activate myosuite
pip install -U myosuite
```

### Install from source

All sim assets ship as pip packages (`myo-sim`, `furniture-sim`, `mpl-sim`, `object-sim`, `ycb-sim`) — no git submodule checkout required.

```bash
git clone https://github.com/myohub/myosuite.git
cd myosuite
pip install -e .
```

### Optional extras

```bash
pip install -e ".[rl]"       # Stable-Baselines3 for RL training
pip install -e ".[mjx]"      # JAX/MJX GPU-accelerated backend
pip install -e ".[mjlab]"    # MuJoCo Warp / Isaac Lab backend
pip install -e ".[docs]"     # Sphinx documentation build
```

For advanced options, see the [installation guide](https://myosuite.readthedocs.io/en/latest/install.html).

### Furniture / MPL / object / YCB sim assets

These are installed with the base package (`pip install myosuite`). Older model XML that still uses legacy `simhive/<name>/...`-style paths (from the pre-pip git-submodule layout) is automatically resolved to the installed pip package; a local `myosuite/simhive/<name>` checkout can still be used as an override.

### Verify your installation

```bash
python -m myosuite.tests.test_myo   # lists all available environments

python -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0
# On macOS use: mjpython -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0
```

---

## Quick Start by Audience

### Everyone — run a simulation

```python
import gymnasium as gym
import myosuite  # registers all environments

env = gym.make('myoElbowPose1D6MRandom-v0')
obs, info = env.reset()
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

### Biomechanist — extract kinematics and muscle forces

```python
import gymnasium as gym
import myosuite
import numpy as np

env = gym.make('myoElbowPose1D6MRandom-v0')
obs, info = env.reset()

joint_angles, muscle_forces = [], []
for _ in range(500):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    joint_angles.append(env.unwrapped.data.qpos.copy())              # rad
    muscle_forces.append(env.unwrapped.data.actuator_force.copy())   # N
    if terminated or truncated:
        obs, info = env.reset()

joint_angles = np.array(joint_angles)
print(f"Elbow ROM: {np.ptp(joint_angles, axis=0).round(3)} rad")
```

### Neuroscientist — simulate muscle fatigue

```python
import gymnasium as gym
import myosuite

# Muscles fatigue with sustained activation, just as in vivo
env = gym.make('myoElbowPoseFatigue1D6MFixed-v0')
obs, info = env.reset()
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

### ML Researcher — train with Stable-Baselines3

```python
from stable_baselines3 import SAC
import gymnasium as gym
import myosuite

env = gym.make('myoElbowPose1D6MRandom-v0')
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100_000)
```

---

## Available Environments

| Body part | Example environments |
|-----------|---------------------|
| **Elbow** | `myoElbowPose1D6MRandom-v0`, `myoElbowPoseSarcopenia*`, `myoElbowPoseFatigue*` |
| **Finger** | `myoFingerPoseFixed-v0`, `myoFingerPoseRandom-v0`, `myoFingerReachRandom-v0` |
| **Hand** | `myoHandPoseRandom-v0`, `myoChallengeBaodingP2-v1` |
| **Leg / Gait** | `myoLegWalk-v0`, `myoChallengeRunTrackP2-v0` |
| **Full arm** | `myoShoulder*`, `myoRelocateEnvDemoV0` |

Run `python -m myosuite.tests.test_myo` for the full list, or see the [task specifications](https://github.com/myohub/myosuite/blob/main/docs/source/suite.rst#tasks).

### Backend & stability matrix

> **Legend:** `stable` — tested, parity-verified, production-ready · `beta` — functional and tested, still hardening API/contracts · `wip` — work in progress, API may change · `–` — not yet implemented / not applicable

| Environment family | CPU (Gymnasium) | MJX (JAX) | mjlab (Warp/Isaac) |
|---|---|---|---|
| Elbow pose / sarcopenia | stable | stable (`MjxElbowPose*-v0`) | beta (`myoElbowPose1D6MFixed-v0`, `myoSarcElbow*`) |
| Elbow fatigue | stable | stable | – (pending) |
| Finger pose | stable | stable (`MjxFingerPose*-v0`) | – (config exists, registration pending) |
| Finger reach | stable | beta (`MjxFingerReachRandom-v0`) | – (config exists, registration pending) |
| Hand pose | stable | beta (`MjxHandPoseRandom-v0`) | – (config exists, registration pending) |
| Hand reach | stable | stable (`MjxHandReach*-v0`) | – (config exists, registration pending) |
| Leg walk / gait | stable | beta (`MjxLegWalk-v0`) | beta (`myoLegWalk-v0`, `myoSarcLegWalk-v0`) |
| Mimic bimanual | stable | beta (`MjxMimicBimanual-v0`) | beta (`myoMimicBimanual-v0`) |
| Mimic full-body | beta | beta (`MjxMimicFullbody-v0`) | beta (`myoMimicFullbody-v0`) |
| MyoChallenge TableTennis | stable | wip | beta (`myoChallengeTableTennisP*-v0`) |
| MyoChallenge Reorient | stable | wip | – (planned; see [porting guide](docs/wiki/mjlab-design-guide.md)) |
| MyoChallenge Bimanual | stable | wip | – (planned) |
| MyoChallenge Soccer | stable | wip | – (planned) |
| MyoChallenge RunTrack | stable | wip | – (planned; OSL controller required) |
| MyoChallenge ChaseTag | stable | wip | – (planned) |
| Shoulder / full-arm | stable | – | – |

> **Platform note:** MJX requires Linux + CUDA (or CPU-mode JAX). macOS is CPU-only (`gym.make()`).

---

## Tutorials

| Notebook | Topic | Audience |
|----------|-------|----------|
| [`1_Get_Started`](tutorials/1_Get_Started.ipynb) | Basic simulation loop | Everyone |
| [`2_Load_policy`](tutorials/2_Load_policy.ipynb) | Load and run a trained policy | Everyone |
| [`3_Analyse_movements`](tutorials/3_Analyse_movements.ipynb) | Extract and plot kinematics | Biomechanics |
| [`6_Inverse_Dynamics`](tutorials/6_Inverse_Dynamics.ipynb) | Compute joint torques | Biomechanics |
| [`7_Fatigue_Modeling`](tutorials/7_Fatigue_Modeling.ipynb) | Neuromuscular fatigue model | Neuroscience |
| [`8_inverse_kinematics`](tutorials/8_inverse_kinematics.py) | IK solver | Biomechanics |
| [`9_Computed_muscle_control`](tutorials/9_Computed_muscle_control.ipynb) | Feedforward muscle control | Neuroscience |
| [`10_PlaybackMotFile`](tutorials/10_PlaybackMotFile.ipynb) | OpenSim motion file replay | Biomechanics |
| [`4c_Train_SB_policy`](tutorials/4c_Train_SB_policy.ipynb) | Train with Stable-Baselines3 | ML/RL |
| [`Walk_Backends_Demo`](tutorials/Walk_Backends_Demo.ipynb) | CPU / MJX / mjlab walk benchmark | ML/RL |
| [`11a_MuscleMimic_Fullbody_Policy_Trajectory`](tutorials/11a_MuscleMimic_Fullbody_Policy_Trajectory.ipynb) | Load full-body MuscleMimic policy, trajectory reference, inline video | ML/RL |
| [`11b_MuscleMimic_Fullbody_Training`](tutorials/11b_MuscleMimic_Fullbody_Training.ipynb) | CPU MuscleMimic training (PPO), ghost-body viz, checkpoint + video | ML/RL |
| [`11c_MuscleMimic_Fullbody_mjlab`](tutorials/11c_MuscleMimic_Fullbody_mjlab.ipynb) | MuscleMimic full-body on mjlab / Warp, RSL-RL PPO | ML/RL |
| [`fatigue_demo`](tutorials/fatigue_demo.py) | Elbow pose vs fatigue env comparison (matplotlib) | Neuroscience |
| [`modular_task_config`](tutorials/modular_task_config.ipynb) | Define custom tasks with `TaskSpec` (no subclassing) | ML/RL |

---

## Physics Backends

MyoSuite supports three backends for different scale requirements:

| Backend | Hardware | Parallelism | Use case |
|---------|----------|-------------|----------|
| MuJoCo CPU | CPU | 1 env | Analysis, debugging, sim-to-real |
| MJX (JAX) | GPU/TPU | 4096 envs | Fast RL training |
| MuJoCo Warp (mjlab) | GPU | 4096 envs | PyTorch-based RL |

```bash
# Backend benchmark (CPU by default; set JAX_PLATFORMS=cuda for GPU)
python benchmarks/sar_backends/run_benchmark.py --steps 10000
```

See [benchmarks/sar_backends/README.md](benchmarks/sar_backends/README.md) for full options.

---

## MyoSkeleton

Initialize the full-body MyoSkeleton model:

```bash
python -m myosuite_init   # or: uv run myoapi_init
```

Visualize:
```bash
python -m myosuite.utils.examine_sim -s $(python -c "import myo_sim; print(myo_sim.get_path('arm/myoarm.xml'))")
```

---

## MuscleMimic full-body playback

The eval CLI supports native full-body playback via `--path` + `--motion_path`
with MuJoCo viewer or recording flags. Legacy `--playback` is obsolete and no
longer supported. This is distinct from the `myoMimicFullbody-v0` Gymnasium
tracking task:

```bash
uv run myosuite-musclemimic-fullbody-eval \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco --mujoco_viewer --n_steps 1000 --eval_seed 0
```

See [myosuite/integrations/musclemimic/README.md](myosuite/integrations/musclemimic/README.md)
for setup, troubleshooting, MyoSuite-native training/playback modes, and the
MuscleMimic citation.

### mjlab backend (GPU parallel training)

mjlab runs `myoMimicFullbody-v0` with thousands of parallel environments on GPU.
It is a **training** backend — there is no interactive viewer equivalent to
`--use_mujoco --mujoco_viewer`. Use the MuJoCo CPU path for visual inspection
and mjlab for large-scale RL training.

**Smoke test** (verify mjlab stack):

```bash
uv run myosuite-musclemimic-fullbody-eval --backend mjlab --n-steps 32 --seed 0
```

**Python training** (random targets):

```python
import mjlab.envs as mjlab_envs
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import register_mjlab_tasks

register_mjlab_tasks()   # register all MyoSuite tasks on mjlab
env = mjlab_envs.make("myoMimicFullbody-v0")
obs, _ = env.reset()
```

**Trajectory-mode training** (motion clip as target source):

```python
from myosuite.core.trajectory_io import load_motion_clip, resolve_motion_path
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import register_mimic_mjlab_tasks_with_clip
from mjlab.tasks.registry import register_mjlab_task
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import _elbow_ppo_runner_cfg
import mjlab.envs as mjlab_envs

motion_file = resolve_motion_path("KIT/314/walking_medium09_poses")
clip = load_motion_clip(motion_file, expected_nq=89, expected_nv=88)
register_mimic_mjlab_tasks_with_clip(register_mjlab_task, _elbow_ppo_runner_cfg, clip=clip)
env = mjlab_envs.make("myoMimicFullbody-v0")
```

Requires `pip install 'myosuite[mjlab]'` (GPU/CUDA recommended).

---

## License

MyoSuite is licensed under the [Apache License](LICENSE).

---

## Citation
```bibtex
@Misc{MyoSuite2026,
  author =       {Vittorio, Caggiano AND Balint, Hodossy AND Florian, Fischer, AND MyoSuiteTeam},
  title =        {MyoSuite 3.0 -- A multidomodal platform for understanding human movements},
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

# BUGS
## macOS: `dlopen` libpython error

If you see an error like:

```
Library not loaded: @executable_path/../lib/libpython3.10.dylib
```

### Fix
```bash
# Activate your virtual environment first
source .venv/bin/activate

# Navigate to your venv directory
cd .venv

# Ensure lib directory exists
mkdir -p lib

# Symlink libpython
PYTHON_LIB=$(find "$HOME/.local/share/uv/python" -name "libpython3.10.dylib" | head -n 1)
ln -sf "$PYTHON_LIB" .venv/lib/libpython3.10.dylib
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$(dirname "$PYTHON_LIB")"
```
