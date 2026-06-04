# MyoSuite Tutorials

## Requirements

Install Jupyter Notebooks:

```bash
pip install jupyter
```

Or jump-start with the **ICRA2023 Colab Tutorial** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KGqZgSYgKXF-vaYC33GR9llDsIW9Rp-q)

If the kernel for your environment is not recognised, register it:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=myosuite
# To remove it later:
jupyter kernelspec uninstall myosuite
```

You may also need `ffmpeg` for video playback in notebooks:

```bash
conda install conda-forge::ffmpeg
# or: brew install ffmpeg
```

---

## Recommended Learning Paths

### First-time user (< 30 min)
1. **[1 — Get Started](./1_Get_Started.ipynb)** — create an env, render a frame
2. **[4c — Train with SB3](./4c_Train_SB_policy.ipynb)** — train your first policy
3. **[3 — Analyse movements](./3_Analyse_movements.ipynb)** — plot kinematics & synergies

### Biomechanics researcher
4. **[5 — Hand muscles](./5_Move_Hand_Fingers.ipynb)** — direct muscle control
5. **[10 — Mot file playback](./10_PlaybackMotFile.ipynb)** — replay OpenSim data
6. **[6 — Inverse Dynamics](./6_Inverse_Dynamics.ipynb)** — joint torque from motion
7. **[9 — CMC](./9_Computed_muscle_control.ipynb)** — full muscle control pipeline

### Neuroscience / physiology
8. **[7 — Fatigue Modeling](./7_Fatigue_Modeling.ipynb)** — 3CC-r fatigue model

### RL / ML engineer
9. **[Walk Backends Demo](./Walk_Backends_Demo.ipynb)** — CPU vs MJX vs mjlab
10. **[Modular Task Config](./modular_task_config.ipynb)** — define custom tasks

### Advanced (MuscleMimic / SAR)
11. **[MuscleMimic Policy Trajectory](./MuscleMimic_Fullbody_Policy_Trajectory.ipynb)**
12. **[SAR Tutorial](./sar/SAR_tutorial.ipynb)**

---

## Optional dependencies

| Feature | Install command |
|---|---|
| RL training | `pip install stable-baselines3` |
| DEP-RL baseline | `pip install deprl` |
| Inverse dynamics / CMC | `pip install osqp scipy` |
| MuscleMimic checkpoints | `pip install musclemimic_models huggingface_hub` |
| Synergy analysis | `pip install scikit-learn` |
| Video playback | `conda install conda-forge::ffmpeg` or `brew install ffmpeg` |

---

## Tutorials

### Getting started

- [**1 — Get Started**](./1_Get_Started.ipynb): Basic simulation loop, environment creation, observation/action spaces.
- [**2 — Load a policy**](./2_Load_policy.ipynb): Load a trained policy file and run a rollout.
- [**3 — Analyse movements**](./3_Analyse_movements.ipynb): Extract and plot joint kinematics from a trained policy.

### Biomechanics

- [**6 — Inverse Dynamics**](./6_Inverse_Dynamics.ipynb): Compute joint torques from motion data.
  Requires: `pip install osqp matplotlib pandas`
- [**8 — Inverse Kinematics**](./8_inverse_kinematics.py): IK solver using the `mink` library.
- [**9 — Computed Muscle Control (CMC)**](./9_Computed_muscle_control.ipynb): Feedforward muscle control via CMC.
- [**10 — Playback OpenSim Mot Files**](./10_PlaybackMotFile.ipynb): Load OpenSim `.mot` files and replay on MyoSkeleton.
  Run `python -m myosuite_init` first to download the skeleton.

### Neuroscience / physiology

- [**7 — Fatigue Modeling**](./7_Fatigue_Modeling.ipynb): Three-compartment cumulative fatigue model for muscle endurance studies.

### RL / ML training

- [**4a — DEPRL baseline**](./4a_deprl.ipynb): DEP-RL controller.
  Requires: `pip install deprl`
- [**4b — MyoReflex baseline**](./4b_reflex/MyoSuite_MyoReflex_Walk.ipynb): Spinal reflex controller for walking.
- [**4c — Train with Stable-Baselines3**](./4c_Train_SB_policy.ipynb): SAC/PPO training loop with SB3.
  Requires: `pip install stable-baselines3`
- [**Walk Backends Demo**](./Walk_Backends_Demo.ipynb): Side-by-side benchmark of CPU, MJX, and mjlab backends.
- [**Modular Task Config**](./modular_task_config.ipynb): Define new tasks with `TaskSpec` + term functions — no subclassing.

### MuscleMimic full-body tracking

- [**MuscleMimic Fullbody Policy Trajectory**](./MuscleMimic_Fullbody_Policy_Trajectory.ipynb):
  Load a full-body MuscleMimic policy from a Hugging Face checkpoint, run it against a KIT motion clip,
  and save the rendered tracking video.
  Requires: `musclemimic_models`, `orbax-checkpoint`, `imageio`/`ffmpeg`, HF cache (`hf://amathislab/mm-10m-2`).

  ```bash
  # One-time setup
  uv run hf auth login
  uv run myosuite-musclemimic-setup-demo-cache
  ```

- [**MuscleMimic mjlab**](./musclemimic_mjlab.ipynb): MuJoCo Warp / Isaac Lab backend for MuscleMimic training.
- [**SAR tutorial**](./sar/SAR_tutorial.ipynb): Train policies with Spatial Action Representations.

### MyoChallenge boxing (environment naming)

Use these **registered** Gymnasium IDs after importing MyoChallenge (e.g.
`import myosuite.envs.myo.myochallenge` or `from myosuite import register_all_envs;
register_all_envs()`):

| ID | Use case |
| --- | --- |
| `myoChallengeBoxingMannequin-v0` | Scripted mannequin opponent |
| `myoChallengeBoxingP0-v0` | Static six-target machine; long episodes; typical choice for MuscleMimic / BC collection |
| `myoChallengeBoxingVs-v0` | Two MuscleMimic agents (competitive) |


See also: [`docs/boxing_bc_usage.md`](../docs/boxing_bc_usage.md) and the Sphinx environment list
[`docs/source/environments.rst`](../docs/source/environments.rst) (section *MyoChallenge boxing*).
