# MyoSuite Tutorials

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=myosuite
# video playback in notebooks: conda install conda-forge::ffmpeg  (or brew install ffmpeg)
```

[ICRA Colab](https://colab.research.google.com/drive/1KGqZgSYgKXF-vaYC33GR9llDsIW9Rp-q)

On Colab, opening a notebook from GitHub does **not** install the package. Use the ICRA Colab above (it has an install cell), or install locally:

```bash
# from a clone
pip install -e .
# or, if you only have the notebook
pip install git+https://github.com/MyoHub/myosuite.git
```

Then import with the public API (not the old `from myosuite.utils import gym` one-liner):

```python
import gymnasium as gym
import myosuite  # required — registers env ids
```

## Path

1. [1 — Get Started](./1_Get_Started.ipynb)
2. [4c — Train with SB3](./4c_Train_SB_policy.ipynb) (`pip install -e ".[rl]"`)
3. [3 — Analyse movements](./3_Analyse_movements.ipynb)

Also useful: [2 — Load a policy](./2_Load_policy.ipynb) (pretrained NPG weights are **not** in a clean clone; the notebook uses a random policy if they are missing), [5 — Hand muscles](./5_Move_Hand_Fingers.ipynb), [7 — Fatigue](./7_Fatigue_Modeling.ipynb), [10 — OpenSim `.mot` playback](./10_PlaybackMotFile.ipynb).

Optional: [6 — Inverse dynamics](./6_Inverse_Dynamics.ipynb) (`osqp`), [9 — CMC](./9_Computed_muscle_control.ipynb), [4a — DEP-RL](./4a_deprl.ipynb) (`pip install deprl`, **Python ≤3.11.5 only**), [4b — MyoReflex](./4b_reflex/MyoSuite_MyoReflex_Walk.ipynb).

By default, notebooks 6 and 9 use the first 80 trajectory frames (`MYOSUITE_FULL_ID=1` for the full CSV). Notebook 11b trains 256 PPO steps (`MYOSUITE_FULL_MIMIC=1` for the 50k demo). Notebook 11c skips Warp env creation on macOS / CPU; use [11c Colab GPU](./11c_MuscleMimic_Fullbody_mjlab_colab.ipynb) on a T4. Notebook 11d stays in `QUICK_MODE` unless `MYOSUITE_FULL_BC=1`.

The notebook [modular_task_config.ipynb](./modular_task_config.ipynb) is **experimental**. Prefer subclassing `MyoGymnasiumEnv` (see `docs/wiki/adding-a-new-task.md`).

GPU: `python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0` (`pip install -e ".[mjlab]"`). Walk-through: [directional_leg_gpu_training.py](./directional_leg_gpu_training.py).

MuscleMimic: [integration README](../myosuite/integrations/musclemimic/README.md). mjlab GPU (Colab T4): [11c Colab](./11c_MuscleMimic_Fullbody_mjlab_colab.ipynb) ([Open in Colab](https://colab.research.google.com/drive/144wHsu_UBVofZqXRTOUWY33ZscziA76R)). Directional locomotion (Colab-ready): [11d — Full-body directional locomotion](./11d_MuscleMimic_Fullbody_directional_locomotion.ipynb) ([Open in Colab](https://colab.research.google.com/drive/1lc64D9YS8mmqz00-p161syndTUbffrg2)). Bimanual clips on Hugging Face are **gated** — request access before running that cell in [11c](./11c_MuscleMimic_Fullbody_mjlab.ipynb).

## Rendering

Use `render_mode="rgb_array"` and `env.render()`. Models without a named camera (elbow) use the free camera (`camera_id=-1`). Hand actuators use a `_r` suffix (`FDP2_r`, `EDC2_r`).

Headless video on Linux often needs `MUJOCO_GL=egl`; on macOS use `MUJOCO_GL=glfw` (the default if unset in the SAR helpers).

## Models

Install from this repo so XML assets match the notebooks:

```bash
pip install -e ".[dev]"
```

`pyproject.toml` pins `myo-sim` from GitHub `dev`. A plain `pip install myo-sim` (0.2.1) is missing some arm/walk files; the package falls back to bundled copies under `myosuite/envs/myo/assets/`.

## Inverse kinematics

```bash
python tutorials/8_inverse_kinematics.py --no-viewer
# interactive (macOS): mjpython tutorials/8_inverse_kinematics.py
```

SAR full training is hours of SB3. The notebook skips those cells unless `MYOSUITE_FULL_SAR=1`. Precomputed-SAR cells still run. Check the install with:

```bash
python tutorials/sar/run_sar_full.py --help
python tutorials/sar/run_sar_full.py --dry-run
```
