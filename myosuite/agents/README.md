# Baselines

## Installation
1. We use [mjrl](https://github.com/aravindr93/mjrl) for our baselines ([install instructions](https://github.com/aravindr93/mjrl/tree/master/setup#installation)) and [PyTorch](https://pytorch.org/).
2. [Hydra](https://github.com/facebookresearch/hydra) `pip install hydra-core --upgrade`
3. [submitit](https://github.com/facebookincubator/submitit) launcher hydra plugin to launch jobs on cluster/ local 

```bash
pip install tabulate matplotlib torch git+https://github.com/aravindr93/mjrl.git
pip install hydra-core --upgrade
pip install hydra-submitit-launcher --upgrade
pip install submitit
```

## Launch training
1. Get commands to run
```bash
% sh train_myosuite.sh myo         # runs natively
% sh train_myosuite.sh myo local   # use local launcher
% sh train_myosuite.sh myo slurm   # use slurm launcher
```
2. Further customize the prompts from the previous step and execute.

3. To resume training from a previous checkpoint add the `+job_name=<absolute_path_of_previous_checkpoint>` to the command line
