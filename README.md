# myoSuite
![PyPI](https://img.shields.io/pypi/v/myosuite)
[![Documentation Status](https://readthedocs.org/projects/myosuite/badge/?version=latest)](https://myosuite.readthedocs.io/en/latest/?badge=latest)
![PyPI - License](https://img.shields.io/pypi/l/myosuite)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/myosuite/blob/main/CONTRIBUTING.md)

`myoSuite` is a collection of musculoskeletal environments and tasks simulated with the [MuJoCo](http://www.mujoco.org/) physics engine and wrapped in the OpenAI ``gym`` API to enable the application of Machine Learning to bio-mechanic control problems

Below is an overview of the tasks in the myoSuite. Full task details are available [here], and baseline details are availale [here]
<img width="1240" alt="TasksALL" src="./docs/source/images/myoSuite_All.png">


## Getting Started
It is adviced to create a separate environment with
``` bash
conda create --name myoSuite python=3.7.1
conda activate myoSuite
```
It is possible to install `myoSuite` with:
``` bash
pip install -U myoSuite
```
for advance installation see here.

## Examples
It is possible to create and interface with myoSuite environments like any other OpenAI gym environments. For example, to use the `ElbowPose1D6MRandom-v0` environment it is possible simply to run: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)


```python
import myoSuite
import gym
env = gym.make('ElbowPose1D6MRandom-v0')
env.reset()
for _ in range(1000):
  env.sim.render(mode='window')
  env.step(env.action_space.sample()) # take a random action
env.close()
```

## License

myoSuuite is licensed under the [Apache License](LICENSE)

## Citing myoSuite

If you use myoSuite in your publication, please cite the [arXiv paper](www) by using the following BibTeX entries.

```BibTeX
@Misc{myoSuite2022,
  author =       {Vittorio, Caggiano AND Huawei, Wang AND Guillaume, Durandau AND Massimo, Sartori AND Vikash, Kumar},
  title =        {myoSuite: A fast and contact-rich simulation suite for musculoskeletal motor control},
  howpublished = {\url{https://github.com/facebookresearch/myoSuite}},
  year =         {2022}
}
```

