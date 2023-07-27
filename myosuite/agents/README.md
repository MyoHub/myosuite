# Baselines

We offer baselines trained with Natural Policy Gradient (NPG) via `MJRL`. Below, instructions on how to reproduce those baselines [training models via `mjrl`](#Installation-MJRL). 
We provide instructions on how to [train models via `stable-baselines3`](#Installation-StableBaselines3). 
Specifically for locomotion, we provide also `DEP-RL Locomotion Baseline`

## Installation MJRL
1. We use [mjrl](https://github.com/aravindr93/mjrl) for our baselines ([install instructions](https://github.com/aravindr93/mjrl/tree/master/setup#installation)) and [PyTorch](https://pytorch.org/).
2. [Hydra](https://github.com/facebookresearch/hydra) `pip install hydra-core==1.1.0`
3. [submitit](https://github.com/facebookincubator/submitit) launcher hydra plugin to launch jobs on cluster/ local 

```bash
pip install tabulate matplotlib torch git+https://github.com/aravindr93/mjrl.git hydra-core==1.1.0 hydra-submitit-launcher submitit
```


### Launch training
Get commands to run
```bash
% sh train_myosuite.sh myo         # runs natively
% sh train_myosuite.sh myo local mjrl   # use mjrl with local launcher
% sh train_myosuite.sh myo slurm mjrl   # use mjrl with slurm launcher
```
### Resume training
To resume training from a previous checkpoint add the `+job_name=<absolute_path_of_previous_checkpoint>` to the command line



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


### Launch training
Get commands to run
```bash
% sh train_myosuite.sh myo local sb3   # use stable-baselines3 with local launcher
% sh train_myosuite.sh myo slurm sb3   # use stable-baselines3 with slurm launcher
```

## Pretrained SAR representations
We provide pretrained synergistic representations for training locomotion and manipulation policies in `/SAR_pretrained`. These representations can be used to enhance learning using the `SAR_RL()` function defined in the SAR tutorial. Custom functions are also defined in the tutorial for automatically loading these representations (`load_locomotion_SAR()` and `load_manipulation_SAR()`). 

It is also possible to build your own SAR representations from scratch by following the steps outlined in the tutorial.

## DEP-RL Locomotion Baseline

### Installation
1. We provide [deprl](https://github.com/martius-lab/depRL) as an additional baseline for locomotion policies.
2. Simply run

```bash
python -m pip install deprl
```
after installing the myosuite.

### Train new policy
For training from scratch, use

```bash
python -m deprl.main baselines_DEPRL/myoLegWalk.json
```
and training should start. Inside the json-file, you can set a custom output folder with `working_dir=myfolder`. Be sure to adapt the `sequential` and `parallel` settings. During a training run, `sequential x parallel` environments are spawned, which consumes over 30 GB of RAM with the default settings for the myoLeg. Reduce this number if your workstation has less memory.

### Visualize trained policy
We provide two mechanisms to visualize policies.
1. If you wish to use your own environment, or want to just quickly try our pre-trained DEP-RL baseline, take a look at the code snippet below

```
import gym
import myosuite
import deprl

env = gym.make("myoLegWalk-v0")
policy = deprl.load_baseline(env)

N = 5 # number of episodes
for i in range(N):
    obs = env.reset()
    while True:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        env.mj_render()
        if done:
            break
```
Use `deprl.load('path', env)`, if you have your own policy.
2. For our visualization, simply run

```bash
python -m deprl.play --path folder/
```
where the last folder contains the `checkpoints` and `config.yaml` files. This runs the policy for several episodes and returns scores. 
3. You can also log some settings to [wandb](https://wandb.ai/). Set it up and afterwards run 

```bash
python -m deprl.log --path baselines_DEPRL/myoLegWalk_20230514/myoLeg/log.csv --project myoleg_deprl_baseline
```
which will log all the training metrics to your `wandb` project.
4. If you want to plot your training run, use

```bash
python -m deprl.plot --path baselines_DEPRL/myoLegWalk_20230514/
```

For more instructions on how to use the plot feature, checkout [TonicRL](https://github.com/fabiopardo/tonic), which is the general-purpose RL library deprl was built on.

### Credit
The DEP-RL baseline for the myoLeg was developed by Pierre Schumacher, Daniel Häufle and Georg Martius as members of the Max Planck Institute for Intelligent Systems and the Hertie Institute for Clinical Brain Research. Please cite the following if you are using this work:
```
@inproceedings{
schumacher2023deprl,
title={{DEP}-{RL}: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems},
author={Pierre Schumacher and Daniel Haeufle and Dieter B{\"u}chler and Syn Schmitt and Georg Martius},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=C-xa_D3oTj6}
}
```
