""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from myosuite.utils import gym; register=gym.register
import collections
from copy import deepcopy
from flatten_dict import flatten, unflatten

from myosuite.utils.implement_for import implement_for

#TODO: check versions
@implement_for("gym", None, "0.24")
def gym_registry_specs():
    return gym.envs.registry.env_specs

@implement_for("gym", "0.24", None)
def gym_registry_specs():
    return gym.envs.registry

@implement_for("gymnasium")
def gym_registry_specs():
    return gym.envs.registry

# TODO: move to within the function?
@implement_for("gym", None, "0.24")
def _update_env_spec_kwarg(env_variant_specs, variants, override_keys):
    env_variant_specs._kwargs, variants_update_keyval_str = update_dict(env_variant_specs._kwargs, variants, override_keys=override_keys)
    return variants_update_keyval_str

@implement_for("gym", "0.24", None)
def _update_env_spec_kwarg(env_variant_specs, variants, override_keys):
    env_variant_specs.kwargs, variants_update_keyval_str = update_dict(env_variant_specs.kwargs, variants, override_keys=override_keys)
    return variants_update_keyval_str

@implement_for("gymnasium")
def _update_env_spec_kwarg(env_variant_specs, variants, override_keys):
    env_variant_specs.kwargs, variants_update_keyval_str = update_dict(env_variant_specs.kwargs, variants, override_keys=override_keys)
    return variants_update_keyval_str

@implement_for("gym", None, "0.24")
def _entry_point(env_variant_specs):
    return env_variant_specs._entry_point

@implement_for("gym", "0.24", None)
def _entry_point(env_variant_specs):
    return env_variant_specs.entry_point

@implement_for("gymnasium")
def _entry_point(env_variant_specs):
    return env_variant_specs.entry_point

@implement_for("gym", None, "0.24")
def _kwargs(env_variant_specs):
    return env_variant_specs._kwargs

@implement_for("gym", "0.24", None)
def _kwargs(env_variant_specs):
    return env_variant_specs.kwargs

@implement_for("gymnasium")
def _kwargs(env_variant_specs):
    return env_variant_specs.kwargs

# Update base_dict using update_dict
def update_dict(base_dict:dict, update_dict:dict, override_keys:list=None):
    """
    Update a dict using another dict.
    INPUTS:
    base_dict:      dict to update
    update_dict:    dict with updates (merge operation with base_dict)
    override_keys:  base_dict keys to override. Removes the keys from base_dict and relies on update_dict for updates, if any.
    """
    if override_keys:
        base_dict = {key: item for key, item in base_dict.items() if key not in override_keys}

    base_dict_flat = flatten(base_dict, reducer='dot', keep_empty_types=(dict,))
    update_dict_flat = flatten(update_dict, reducer='dot')
    update_keyval_str = ""
    for key, value in update_dict_flat.items():
        base_dict_flat[key] = value
        update_keyval_str = "{}-{}_{}".format(update_keyval_str, key, value)
    merged_dict = unflatten(base_dict_flat, splitter='dot')
    return merged_dict, update_keyval_str


# Register a variant of pre-registered environment
def register_env_variant(env_id:str, variants:dict, variant_id=None, silent=False, override_keys=None):
    """
    Register a variant of pre-registered environment. Very useful for hyper-parameters sweeps when small changes are required on top of an env
    INPUTS:
    env_id:         name of the original env
    variants:       dict with updates we want on the original env (merge operation with base env)
    variant_id:     name of the varient env. Auto populated if None
    silent:         prints the name of the newly registered env, if True.
    override_keys:  base_env keys to override. Removes the keys from base_env and relies on update_dict for updates, if any.
    """

    # check if the base env is registered
    assert env_id in gym_registry_specs().keys(), "ERROR: {} not found in env registry".format(env_id)

    # recover the specs of the existing env
    env_variant_specs = deepcopy(gym_registry_specs()[env_id])
    env_variant_id = env_variant_specs.id[:-3]

    # update horizon if requested
    if 'max_episode_steps' in variants.keys():
        env_variant_specs.max_episode_steps = variants['max_episode_steps']
        env_variant_id = env_variant_id+"-hor_{}".format(env_variant_specs.max_episode_steps)
        del variants['max_episode_steps']

    # merge specs._kwargs with variants
    variants_update_keyval_str = _update_env_spec_kwarg(env_variant_specs, variants, override_keys)
    env_variant_id += variants_update_keyval_str

    # finalize name and register env
    env_variant_specs.id = env_variant_id+env_variant_specs.id[-3:] if variant_id is None else variant_id
    register(
        id=env_variant_specs.id,
        entry_point=_entry_point(env_variant_specs),
        max_episode_steps=env_variant_specs.max_episode_steps,
        kwargs=_kwargs(env_variant_specs)
    )
    if not silent:
        print("Registered a new env-variant:", env_variant_specs.id)
    return env_variant_specs.id


# Example usage
if __name__ == '__main__':
    import robohive
    import pprint

    # Register a variant
    base_env_name = "kitchen-v3"
    base_env_variants={
        'max_episode_steps':50,                     # special key
        'obj_goal': {"lightswitch_joint": -0.7},    # obj_goal keys will be updated
        'obs_keys_wt': {                            # obs_keys_wt will be updated
            'robot_jnt': 5.0,
            'obj_goal': 5.0,
            'objs_jnt': 5.0,}
    }
    variant_env_name = register_env_variant(env_id=base_env_name, variants=base_env_variants)
    variant_overide_env_name = register_env_variant(env_id=base_env_name, variants=base_env_variants, override_keys="obs_keys_wt") # Instead of updating via merge, obs_keys_wt key will be completeley overwritten

    # Test variant
    print("Base-env kwargs: ")
    pprint.pprint(gym_registry_specs()[base_env_name]._kwargs)
    print("Env-variant kwargs: ")
    pprint.pprint(gym_registry_specs()[variant_env_name]._kwargs)
    print("Env-variant (with override) kwargs: ")
    pprint.pprint(gym_registry_specs()[variant_overide_env_name]._kwargs)

    # Test one of the newly minted env
    env = gym.make(variant_env_name)
    env.reset()
    env.mj_render()
    # env.sim.render(mode='window')
    for _ in range(50):
        env.step(env.action_space.sample()) # take a random action
    env.close()
