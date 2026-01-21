# Adapted from mujoco-playground's registry scripts
import copy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import jax
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground import MjxEnv


MJX_VARIANT_PREFIXES = ("MjxSarc", "MjxFati", "MjxReaf")

_envs = {}

_cfgs = {}

_randomizer = {}


def __getattr__(name):
  if name == "ALL_ENVS":
    return tuple(_envs.keys())
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def config_create_variant(env_config: Callable[[], config_dict.ConfigDict], **kwargs) -> Callable[[], config_dict.ConfigDict]:
    _cfg = copy.deepcopy(env_config())
    for k, v in kwargs.items():
       setattr(_cfg, k, v)
    fn = lambda : _cfg
    return fn

def register_environment(
    env_name: str,
    env_class: Type[MjxEnv],
    cfg_class: Callable[[], config_dict.ConfigDict],
) -> None:
  """Register a new environment.

  Args:
      env_name: The name of the environment.
      env_class: The environment class.
      cfg_class: The default configuration.
  """
  _envs[env_name] = env_class
  _cfgs[env_name] = cfg_class

# register env with all muscle conditions
def register_environment_with_variants(
    env_name: str,
    env_class: Type[MjxEnv],
    cfg_class: Callable[[], config_dict.ConfigDict],
) -> None:
    # register_env_with_variants base env
    register_environment(
        env_name=env_name,
        env_class=env_class,
        cfg_class=cfg_class
    )

    # register variants env with sarcopenia
    if env_name[:3] == "Mjx":
        cfg_class_sarc = config_create_variant(cfg_class, muscle_condition="sarcopenia")
        register_environment(
            env_name=env_name[:3] + "Sarc" + env_name[3:],
            env_class=env_class,
            cfg_class=cfg_class_sarc
        )
    
    # register variants with fatigue
    if env_name[:3] == "Mjx":
        cfg_class_fati =  config_create_variant(cfg_class, muscle_condition="fatigue")
        register_environment(
            env_name=env_name[:3] + "Fati" + env_name[3:],
            env_class=env_class,
            cfg_class=cfg_class_fati
        )

    # register variants with tendon transfer
    if env_name[:7] == "MjxHand":
        cfg_class_reaf =  config_create_variant(cfg_class, muscle_condition="reafferentation")
        register_environment(
            env_name=env_name[:3] + "Reaf" + env_name[3:],
            env_class=env_class,
            cfg_class=cfg_class_reaf
        )

def get_base_env_name(env_name: str) -> str:
  """Extract default environment name (without any muscle condition variant) from env_name."""
  if env_name[:7] in MJX_VARIANT_PREFIXES:
    env_name_default = env_name[:3] + env_name[7:]
  else:
    env_name_default = env_name
  return env_name_default

def get_default_config(env_name: str) -> config_dict.ConfigDict:
  """Get the default configuration for an environment."""
  if env_name not in _cfgs:
    raise ValueError(
        f"Env '{env_name}' not found in default configs. Available configs:"
        f" {list(_cfgs.keys())}"
    )
  return _cfgs[env_name]()


def load(
    env_name: str,
    config: Optional[config_dict.ConfigDict] = None,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
) -> MjxEnv:
  """Get an environment instance with the given configuration.

  Args:
      env_name: The name of the environment.
      config: The configuration to use. If not provided, the default
        configuration is used.
      config_overrides: A dictionary of overrides for the configuration.

  Returns:
      An instance of the environment.
  """
  if env_name not in _envs:
    raise ValueError(
        f"Env '{env_name}' not found. Available envs: {_cfgs.keys()}"
    )
  config = config or get_default_config(env_name)
  return _envs[env_name](config=config, config_overrides=config_overrides)


def get_domain_randomizer(
    env_name: str,
) -> Optional[Callable[[mjx.Model, jax.Array], Tuple[mjx.Model, mjx.Model]]]:
  """Get the default domain randomizer for an environment."""
  if env_name not in _randomizer:
    print(
        f"Env '{env_name}' does not have a domain randomizer in the"
        " manipulation registry."
    )
    return None
  return _randomizer[env_name]
