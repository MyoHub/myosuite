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
- [Train a new policy with MJRL](./4_Train_policy.ipynb). *
- [Use the DEPRL baseline](./4a_deprl.ipynb). For this tutorial, `deprl` is needed. You can install it with `pip install deprl`
- [Use the MyoReflex baseline](./4b_reflex/MyoSuite_MyoReflex_Walk.ipynb). For this tutorial, we provided a wrapper to use MyoReflex together with the tutorial file
- [Train a new policy with stable baselines (SB)](./4c_Train_SB_policy.ipynb). For this tutorial, `stable-baselines3` is needed. You can install it with `pip install stable-baselines`
- [Move single finger of the Hand](./5_Move_Hand_Fingers.ipynb)
- [Train policies with SAR](./SAR/SAR%20tutorial.ipynb). All required installations are included within the SAR tutorial notebook.

*For those tutorial, `mjrl` and `mujoco_py` are needed. You can install them with `pip install tabulate matplotlib torch free-mujoco-py git+https://github.com/aravindr93/mjrl.git`
