# Baselines
## NPG Baselines
### Installation
1. We use [mjrl](https://github.com/aravindr93/mjrl) for our baselines ([install instructions](https://github.com/aravindr93/mjrl/tree/master/setup#installation)) and [PyTorch](https://pytorch.org/).
2. [Hydra](https://github.com/facebookresearch/hydra) `pip install hydra-core --upgrade`
3. [submitit](https://github.com/facebookincubator/submitit) launcher hydra plugin to launch jobs on cluster/ local 

```bash
pip install tabulate matplotlib torch git+https://github.com/aravindr93/mjrl.git
pip install hydra-core --upgrade
pip install hydra-submitit-launcher --upgrade
pip install submitit
```

### Launch training
1. Get commands to run
```bash
% sh train_myosuite.sh myo         # runs natively
% sh train_myosuite.sh myo local   # use local launcher
% sh train_myosuite.sh myo slurm   # use slurm launcher
```
2. Further customize the prompts from the previous step and execute.

3. To resume training from a previous checkpoint add the `+job_name=<absolute_path_of_previous_checkpoint>` to the command line



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
