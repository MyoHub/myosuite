""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from myosuite.utils import gym
from myosuite.utils.paths_utils import plot as plotnsave_paths
import click
import numpy as np
import pickle
import time
import os

DESC = '''
Helper script to examine an environment and associated policy for behaviors; \n
- either onscreen, or offscreen, or just rollout without rendering.\n
- save resulting paths as pickle or as 2D plots \n
- rollout either learned policies or scripted policies (e.g. see rand_policy class below) \n
USAGE:\n
    $ python examine_env.py --env_name door-v1 \n
    $ python examine_env.py --env_name door-v1 --policy_path myosuite.utils.examine_env.rand_policy \n
    $ python examine_env.py --env_name door-v1 --policy_path my_policy.pickle --mode evaluation --episodes 10 \n
'''

# Random policy
class rand_policy():
    def __init__(self, env, seed):
        self.env = env
        self.env.action_space.seed(seed) # requires explicit seeding

    def get_action(self, obs):
        # return self.env.np_random.uniform(high=self.env.action_space.high, low=self.env.action_space.low)
        return self.env.action_space.sample(), {'mode': 'random samples', 'evaluation':self.env.action_space.sample()}

def load_class_from_str(module_name, class_name):
    try:
        m = __import__(module_name, globals(), locals(), class_name)
        return getattr(m, class_name)
    except (ImportError, AttributeError):
        return None

# MAIN =========================================================
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-p', '--policy_path', type=str, help='absolute path of the policy file', default=None)
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-n', '--num_episodes', type=int, help='number of episodes to visualize', default=10)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='./', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
@click.option('-sp', '--save_paths', type=bool, default=False, help=('Save the rollout paths'))
@click.option('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
@click.option('-rv', '--render_visuals', type=bool, default=False, help=('render the visual keys of the env, if present'))
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))

def main(env_name, policy_path, mode, seed, num_episodes, render, camera_name, output_dir, output_name, save_paths, plot_paths, render_visuals, env_args):

    # seed and load environments
    np.random.seed(seed)
    envw = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env = envw.unwrapped
    env.seed(seed)

    # resolve policy and outputs
    if policy_path is not None:
        policy_tokens = policy_path.split('.')
        pi = load_class_from_str('.'.join(policy_tokens[:-1]), policy_tokens[-1])

        if pi is not None:
            pi = pi(env, seed)
        else:
            pi = pickle.load(open(policy_path, 'rb'))
            if output_dir == './': # overide the default
                output_dir, pol_name = os.path.split(policy_path)
                output_name = os.path.splitext(pol_name)[0]
            if output_name is None:
                pol_name = os.path.split(policy_path)[1]
                output_name = os.path.splitext(pol_name)[0]
    else:
        pi = rand_policy(env, seed)
        mode = 'exploration'
        output_name ='random_policy' if output_name is None else output_name

    # resolve directory
    if (os.path.isdir(output_dir) == False) and (render=='offscreen' or save_paths or plot_paths is not None):
        os.mkdir(output_dir)

    # examine policy's behavior to recover paths
    paths = env.examine_policy_new(
        policy=pi,
        horizon=envw.spec.max_episode_steps,
        num_episodes=num_episodes,
        frame_size=(640,480),
        mode=mode,
        output_dir=output_dir+'/',
        filename=output_name,
        camera_name=camera_name,
        render=render)

    # evaluate paths
    success_percentage = env.evaluate_success(paths)
    print(f'Average success over rollouts: {success_percentage}%')

    # save paths
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    if save_paths:
        file_name = output_dir + '/' + output_name + '{}_trace.h5'.format(time_stamp)
        paths.save(trace_name=file_name, verify_length=True, f_res=np.float64)

    # plot paths
    if plot_paths:
        file_name = output_dir + '/' + output_name + '{}'.format(time_stamp)
        plotnsave_paths(paths, env=env, fileName_prefix=file_name)

    # render visuals keys
    if env.visual_keys and render_visuals:
        paths.close()
        render_keys = ['env_infos/visual_dict/'+ key for key in env.visual_keys]
        paths.render(output_dir=output_dir, output_format="mp4", groups=["Trial0",], datasets=render_keys, input_fps=1/env.dt)

if __name__ == '__main__':
    main()
