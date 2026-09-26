# MJX backend (experimental)

JAX / MuJoCo MJX implementations of a few tasks. **Do not start new work here**
— the supported GPU path is mjlab (`pip install -e ".[mjlab]"`, then
`python scripts/train_mjlab.py <env_id>`).

```bash
pip install -e ".[mjx]"          # Linux; not supported on macOS
# NVIDIA CUDA:
pip install -e ".[mjx-cuda]"
```

```python
from myosuite.envs.myo.backends.mjx import make
env = make("MjxElbowPoseRandom-v0")
```

Supported env ids (13; defined in `__init__.py` of this package). They use the `Mjx`
prefix and are not the `myo…` ids of the CPU and mjlab backends:

| Family | Ids |
|---|---|
| Pose | `MjxElbowPoseFixed-v0`, `MjxElbowPoseRandom-v0`, `MjxFingerPoseFixed-v0`, `MjxFingerPoseRandom-v0`, `MjxHandPoseRandom-v0` |
| Reach | `MjxHandReachFixed-v0`, `MjxHandReachRandom-v0`, `MjxFingerReachRandom-v0` |
| Walk | `MjxLegWalk-v0` (flat ground, mirrors `myoLegWalk-v0`) |
| Mimic | `MjxMimicBimanual-v0`, `MjxMimicFullbody-v0` (aliases `MjxMuscleMimicBimanual-v0`, `MjxMuscleMimicFullbody-v0`) |

There is no MJX version of the torso, arm, leg terrain/stand/directional envs or of the
`myoSarc…`/`myoFati…`/`myoReaf…` variants.
