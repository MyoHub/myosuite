# Here a list of tutorials on how to use MyoSuite

## Requirements
You need to install Jupyter Notebooks with:

``` bash
pip install jupyter
```
or you can jumpstart with **ICRA2023 Colab Tutorial** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KGqZgSYgKXF-vaYC33GR9llDsIW9Rp-q)

Note: in case the kernel for the environent is not recognized, you can install it with the following commands:

``` bash
pip install jupyter ipykernel
python -m ipykernel install --user --name= < name of the environment >
```
You can then remove it with:
``` bash
jupyter kernelspec uninstall myosuite
```

## Tutorials

- [Get Started](./1_Get_Started.ipynb)
- [Load a trained policy and play it](./2_Load_policy.ipynb) *
- [Analyse movements from a trained policy](./3_Analyse_movements.ipynb) *
- [Use the DEPRL baseline](./4a_deprl.ipynb). For this tutorial, `deprl` is needed. You can install it with `pip install deprl` (requires Python 3.10).
- [Use the MyoReflex baseline](./4b_reflex/MyoSuite_MyoReflex_Walk.ipynb). For this tutorial, we provided a wrapper to use MyoReflex together with the tutorial file.
- [Train a new policy with stable baselines (SB)](./4c_Train_SB_policy.ipynb). *
- [Move single finger of the Hand](./5_Move_Hand_Fingers.ipynb)
- [Train policies with SAR](./SAR/SAR_tutorial.ipynb). All required installations are included within the SAR tutorial notebook.
- [Replicate hand movements with inverse dynamics](./6_Inverse_Dynamics.ipynb). *
- [Fatigue Modeling](./7_Fatigue_Modeling.ipynb)
- [Inverse Kinematics](./8_inverse_kinematics.py). This script shows how to perform inverse kinematics using the `mink` library.
- [Computed Muscle Control (CMC)](./9_Computed_muscle_control.ipynb)
- [Playback Opensim Mot Files](./10_PlaybackMotFile.ipynb). This tutorial shows how to load OpenSim Mot files and playback them on the MyoSkeleton (follow the instructions prompted by `python -m myosuite_init` before running the notebook).

*For these tutorials, additional packages are needed. You can install them with `uv sync --extra tutorials` or `pip install -e .[tutorials]`.
