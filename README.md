# MyoSuite
`MyoSuite` is a collection of environments/tasks simulated with the [Mujoco](http://www.mujoco.org/) physics engine and wrapped in the OpenAI `gym` API.
Below is an overview of the tasks in the MyoSuite. Full task details are available [here](myosuite/envs/myo/README.md), and baseline details are availale [here](myosuite/agents).
<img width="1240" alt="TasksALL" src="https://user-images.githubusercontent.com/23240128/135134038-1abec2a6-ee47-49fb-b886-34b909f9fc8c.png">


## Getting Started
`MyoSuite` uses git submodules to resolve dependencies. Please follow steps exactly as below to install correctly.

1. The main package dependencies are [MuJoCo v2.1.0](https://github.com/deepmind/mujoco/releases/tag/2.1.0), `python=3.7`, `gym>=0.13`, `mujoco-py>=2.0`, and `pytorch>=1.0`. See `setup/README.md` ([link](setup/README.md)) for detailed install instructions.

2. To get started with `MyoSuite`, clone this repo with pre-populated submodule dependencies
```
$ git clone --recursive https://github.com/facebookresearch/myoSuite.git
```
3. Update submodules
```
$ cd myosuite
$ git submodule update --remote
```
4. Install package using `pip`
```
$ pip install -e .
```
**OR**
Add repo to pythonpath by updating `~/.bashrc` or `~/.bash_profile`
```
export PYTHONPATH="<path/to/myosuite>:$PYTHONPATH"
```
5. Test your installation using
```
python myosuite/tests/test_myo.py
```

6. You can visualize the environments with random controls using the below command
```
$ python myosuite/utils/examine_env.py --env_name myoElbowPose1D6MRandom-v0
```
**NOTE:** If the visualization results in a GLFW error, this is because `mujoco-py` does not see some graphics drivers correctly. This can usually be fixed by explicitly loading the correct drivers before running the python script. See [this page](setup/README.md#known-issues) for details.

## Examples
It is possible to create and interface with MyoSuite environments like any other OpenAI gym environments. For example, to use the "myoElbowPose1D6MRandom-v0" environment it is possible simply to run:

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
