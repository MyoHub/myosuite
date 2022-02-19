# Baselines

## Installation
1. We use [mjrl](https://github.com/aravindr93/mjrl) for our baselines. Please install it with `python -m pip install git+https://github.com/aravindr93/mjrl.git` ( `mujoco-py` is a prerequisite for this type of installation) or by following mjrl's [install instructions](https://github.com/aravindr93/mjrl/tree/master/setup#installation) before procedding.
3. install hydra `pip install hydra-core --upgrade`
4. install [submitit](https://github.com/facebookincubator/submitit) launcher hydra plugin to launch jobs on cluster/ local `pip install hydra-submitit-launcher --upgrade`


## Launch training
1. Get commands to run
```bash
% sh train_myosuite_suits.sh biomechanics         # runs natively
% sh train_myosuite_suits.sh biomechanics local   # use local launcher
% sh train_myosuite_suits.sh biomechanics slurm   # use slurm launcher
```
2. Further customize the prompts from the previous step and execute.

