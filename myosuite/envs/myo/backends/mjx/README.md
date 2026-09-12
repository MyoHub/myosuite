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
