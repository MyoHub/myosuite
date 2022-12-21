# Baselines

We offer baselines trained with Natural Policy Gradient (NPG) via `MJRL`. Below, instructions on how to reproduce those baselines [training models via `mjrl`](#Installation-MJRL). In addition, we provide instructions on how to [train models via `stable-baselines3`](#Installation-StableBaselines3).

## Installation MJRL
1. We use [mjrl](https://github.com/aravindr93/mjrl) for our baselines ([install instructions](https://github.com/aravindr93/mjrl/tree/master/setup#installation)) and [PyTorch](https://pytorch.org/).
2. [Hydra](https://github.com/facebookresearch/hydra) `pip install hydra-core==1.1.0`
3. [submitit](https://github.com/facebookincubator/submitit) launcher hydra plugin to launch jobs on cluster/ local 

```bash
pip install tabulate matplotlib torch git+https://github.com/aravindr93/mjrl.git hydra-core==1.1.0 hydra-submitit-launcher submitit
```

## Installation StableBaselines3
Install 
1. [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3]) `pip install stable-baselines3`,
2. [Hydra](https://github.com/facebookresearch/hydra) `pip install hydra-core==1.1.0`
3. [submitit](https://github.com/facebookincubator/submitit) launcher hydra plugin to launch jobs on cluster/ local 
4. [Optional -- requires python >3.8] Register and Install [Weights and Biases](https://github.com/wandb/wandb) for logging `pip install wandb` (needs tensorboard `pip install tensorboard`)

```bash
pip install stable-baselines3
pip install gym==0.13
pip install hydra-core==1.1.0 hydra-submitit-launcher submitit
#optional 
pip install tensorboard wandb
```


## Launch training
1. Get commands to run
```bash
% sh train_myosuite.sh myo         # runs natively
% sh train_myosuite.sh myo local mjrl   # use mjrl with local launcher
% sh train_myosuite.sh myo slurm mjrl   # use mjrl with slurm launcher
% sh train_myosuite.sh myo local sb3   # use stable-baselines3 with local launcher
% sh train_myosuite.sh myo slurm sb3   # use stable-baselines3 with slurm launcher

```
2. Further customize the prompts from the previous step and execute.

3. In `mjrl` to resume training from a previous checkpoint add the `+job_name=<absolute_path_of_previous_checkpoint>` to the command line