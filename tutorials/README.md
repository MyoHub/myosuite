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

## Tutorials by track

Notebooks are numbered `<track>.<number>_Name.ipynb`. Within a track they go from
simple to advanced; extra files of notebook `X.Y` live in `X.Y_files/`.

| # | Notebook | Needs |
|---|---|---|
| **1 — Basics** | | |
| 1.1 | [Get Started](./1.1_Get_Started.ipynb) | `pip install -e .` |
| 1.2 | [Load Policy](./1.2_Load_Policy.ipynb) — run a checkpoint from an mjlab or SB3 training run (random policy if none is found) | a checkpoint from 2.1 or 2.2 (optional) |
| **2 — Training** | | |
| 2.1 | [Train SB3 Policy](./2.1_Train_SB3_Policy.ipynb) — PPO on CPU | `pip install -e ".[rl]"` |
| 2.2 | [Train MjLab Policy](./2.2_Train_MjLab_Policy.ipynb) — thousands of parallel envs on GPU, playback on CPU | Linux + CUDA, `pip install -e ".[mjlab]"` |
| 2.3 | [SAR](./2.3_SAR.ipynb) — synergistic action representations | `pip install -e ".[rl]"` (full training is hours; set `MYOSUITE_FULL_SAR=1`) |
| 2.4 | [DEP-RL](./2.4_DEP_RL.ipynb) | `pip install deprl`, **Python ≤ 3.11.5 only** |
| 2.5 | [MyoReflex Walk](./2.5_MyoReflex_Walk.ipynb) — reflex-based walking baseline | — |
| **3 — Analysis** | | |
| 3.1 | [Analyse Movements](./3.1_Analyse_Movements.ipynb) — kinematics and synergies | `pip install stable-baselines3 scikit-learn` |
| 3.2 | [Inverse Kinematics](./3.2_Inverse_Kinematics.ipynb) | `pip install "myosuite[examples]"` (mink) |
| 3.3 | [Inverse Dynamics](./3.3_Inverse_Dynamics.ipynb) | `osqp` |
| 3.4 | [Computed Muscle Control](./3.4_Computed_Muscle_Control.ipynb) | — |
| 3.5 | [Playback Mot File](./3.5_Playback_Mot_File.ipynb) — OpenSim `.mot` playback | — |
| **4 — Modelling and conditions** | | |
| 4.1 | [Move Hand Fingers](./4.1_Move_Hand_Fingers.ipynb) | — |
| 4.2 | [Fatigue Modeling](./4.2_Fatigue_Modeling.ipynb) | — |
| 4.3 | [Modular Task Config](./4.3_Modular_Task_Config.ipynb) — **experimental**; prefer the `TaskSpec` / `ModelBuilder` workflow (`docs/wiki/adding-a-new-task.md`) | — |
| **5 — MuscleMimic** | | |
| 5.1 | [Fullbody Policy Trajectory](./5.1_Fullbody_Policy_Trajectory.ipynb) | [integration README](../myosuite/integrations/musclemimic/README.md) |
| 5.2 | [Fullbody Training](./5.2_Fullbody_Training.ipynb) — CPU PPO and ghost-body rendering | as 5.1 |
| 5.3 | [Fullbody Mjlab](./5.3_Fullbody_Mjlab.ipynb) — GPU backend ([Colab T4](https://colab.research.google.com/drive/144wHsu_UBVofZqXRTOUWY33ZscziA76R)); the bimanual clips on Hugging Face are **gated**, request access first | CUDA |
| 5.4 | [Fullbody Directional Locomotion](./5.4_Fullbody_Directional_Locomotion.ipynb) ([Colab](https://colab.research.google.com/drive/1lc64D9YS8mmqz00-p161syndTUbffrg2)) | — |
| 5.5 | [MuscleMimic SAR](./5.5_MuscleMimic_SAR.ipynb) | as 5.1 |

Defaults that keep the notebooks quick: 3.3 and 3.4 use the first 80 trajectory frames
(set `FULL_ID = True` for the full CSV); 5.2 trains 256 PPO steps (`MYOSUITE_FULL_MIMIC=1`
for the 50k demo); 5.3 skips Warp env creation on macOS / CPU; 5.4 stays in `QUICK_MODE`
unless `MYOSUITE_FULL_BC=1`.

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
python tutorials/3.2_files/inverse_kinematics.py --no-viewer
# interactive (macOS): mjpython tutorials/3.2_files/inverse_kinematics.py
```

SAR full training (notebook 2.3) is hours of SB3. The notebook skips those cells unless `MYOSUITE_FULL_SAR=1`. Precomputed-SAR cells still run. Check the install with:

```bash
python tutorials/2.3_files/run_sar_full.py --help
python tutorials/2.3_files/run_sar_full.py --dry-run
```
