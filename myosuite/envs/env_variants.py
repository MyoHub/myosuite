""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
from gym.envs.registration import register
import collections
from copy import deepcopy
from flatten_dict import flatten, unflatten


# Update base_dict using update_dict
def update_dict(base_dict, update_dict):
    base_dict_flat = flatten(base_dict, reducer='dot')
    update_dict_flat = flatten(update_dict, reducer='dot')
    update_keyval_str = ""
    for key, value in update_dict_flat.items():
        base_dict_flat[key] = value
        update_keyval_str = "{}-{}_{}".format(update_keyval_str, key, value)
    merged_dict = unflatten(base_dict_flat, splitter='dot')
    return merged_dict, update_keyval_str


# Register a variant of pre-registered environment
def register_env_variant(env_id, variants, variant_id=None, silent=False):
    # check if the base env is registered
    assert env_id in gym.envs.registry.env_specs.keys(), "ERROR: {} not found in env registry".format(env_id)

    # recover the specs of the existing env
    env_variant_specs = deepcopy(gym.envs.registry.env_specs[env_id])
    env_variant_id = env_variant_specs.id[:-3]

    # update horizon if requested
    if 'max_episode_steps' in variants.keys():
        env_variant_specs.max_episode_steps = variants['max_episode_steps']
        env_variant_id = env_variant_id+"-hor_{}".format(env_variant_specs.max_episode_steps)
        del variants['max_episode_steps']

    # merge specs._kwargs with variants
    env_variant_specs._kwargs, variants_update_keyval_str = update_dict(env_variant_specs._kwargs, variants)
    env_variant_id += variants_update_keyval_str

    # finalize name and register env
    env_variant_specs.id = env_variant_id+env_variant_specs.id[-3:] if variant_id is None else variant_id
    register(
        id=env_variant_specs.id,
        entry_point=env_variant_specs._entry_point,
        max_episode_steps=env_variant_specs.max_episode_steps,
        kwargs=env_variant_specs._kwargs
    )
    if not silent:
        print("Registered a new env-variant:", env_variant_specs.id)
    return env_variant_specs.id


# Example usage
if __name__ == '__main__':
    import mj_envs
    import pprint

    # Register a variant
    base_env_name = "kitchen-v3"
    base_env_variants={
        'max_episode_steps':50, # special key
        "goal": {"lightswitch_joint": -0.7},
        "obj_init": {"lightswitch_joint": -0.0},
    }
    variant_env_name = register_env_variants(env_id=base_env_name, variants=base_env_variants)

    # Test variant
    print("Base-env kwargs: ")
    pprint.pprint(gym.envs.registry.env_specs[base_env_name]._kwargs)
    print("Env-variant kwargs: ")
    pprint.pprint(gym.envs.registry.env_specs[variant_env_name]._kwargs)
    env = gym.make(variant_env_name)
    env.reset()
    env.sim.render(mode='window')
    for _ in range(50):
        env.step(env.action_space.sample()) # take a random action
    env.close()
