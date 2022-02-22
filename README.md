<!-- =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= -->

# MyoSuite
![PyPI](https://img.shields.io/pypi/v/myosuite)
[![Documentation Status](https://readthedocs.org/projects/myosuite/badge/?version=latest)](https://myosuite.readthedocs.io/en/latest/?badge=latest)
![PyPI - License](https://img.shields.io/pypi/l/myosuite)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/myosuite/blob/main/docs/CONTRIBUTING.md)

`MyoSuite` is a collection of musculoskeletal environments and tasks simulated with the [MuJoCo](http://www.mujoco.org/) physics engine and wrapped in the OpenAI ``gym`` API to enable the application of Machine Learning to bio-mechanic control problems


Below is an overview of the tasks in the MyoSuite. Full task details are available [here], and baseline details are availale [here]
<img width="1240" alt="TasksALL" src="./docs/source/images/myoSuite_All.png">


## Getting Started
You will need Python 3.7.1 or later versions. At this moment the library has been tested only on MacOs and Linux.

It is recommended to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) and to create a separate environment with 
``` bash
conda create --name myosuite python=3.7.1
conda activate myosuite
```

It is possible to install MyoSuite with:
``` bash
pip install -U myosuite
```
for advance installation see [here](setup/README.md).

Test your installation using
``` bash
python myosuite/tests/test_myo.py
```

You can also visualize the environments with random controls using the below command
```
$ python myosuite/utils/examine_env.py --env_name myoElbowPose1D6MRandom-v0
```
**NOTE:** If the visualization results in a GLFW error, this is because `mujoco-py` does not see some graphics drivers correctly. This can usually be fixed by explicitly loading the correct drivers before running the python script. See [this page](setup/README.md#known-issues) for details.

## Examples
It is possible to create and interface with MyoSuite environments like any other OpenAI gym environments. For example, to use the `myoElbowPose1D6MRandom-v0` environment it is possible simply to run: [![Getting started with a Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YFqvzspGoKTWmWcFzgWYEAxq7CQpki2G?usp=sharing)


```python
import myosuite
import gym
env = gym.make('myoElbowPose1D6MRandom-v0')
env.reset()
for _ in range(1000):
  env.sim.render(mode='window')
  env.step(env.action_space.sample()) # take a random action
env.close()
```

You can find tutorials [here](https://github.com/facebookresearch/myosuite/tree/main/docs/source/tutorials#tutorials) on how to load myoSuite models, train them and visualize their outcome.

## License

MyoSuite is licensed under the [Apache License](LICENSE)

## Citing MyoSuite

If you use MyoSuite in your publication, please cite the [arXiv paper](www) by using the following BibTeX entries.

```BibTeX
@Misc{MyoSuite2022,
  author =       {Vittorio, Caggiano AND Huawei, Wang AND Guillaume, Durandau AND Massimo, Sartori AND Vikash, Kumar},
  title =        {MyoSuite: A fast and contact-rich simulation suite for musculoskeletal motor control},
  howpublished = {\url{https://github.com/facebookresearch/myosuite}},
  year =         {2022}
}
```

