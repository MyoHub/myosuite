# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MyoSuite MJX environments — JAX-accelerated musculoskeletal RL.

Public API::

    from myosuite.envs.myo.backends.mjx import make
    env = make("MjxElbowPoseFixed-v0")
    env = make("MjxLegWalk-v0")

Environment names
-----------------
Pose (elbow, finger):
    ``MjxElbowPoseFixed-v0``, ``MjxElbowPoseRandom-v0``
    ``MjxFingerPoseFixed-v0``, ``MjxFingerPoseRandom-v0``
    ``MjxHandPoseRandom-v0``
Reach (hand):
    ``MjxHandReachFixed-v0``, ``MjxHandReachRandom-v0``
Reach (finger):
    ``MjxFingerReachRandom-v0``
Walk (leg):
    ``MjxLegWalk-v0``  — flat-ground locomotion, mirrors ``myoLegWalk-v0``
Mimic (site tracking):
    ``MjxMimicBimanual-v0``, ``MjxMimicFullbody-v0``
    (aliases: ``MjxMuscleMimicBimanual-v0``, ``MjxMuscleMimicFullbody-v0``)
"""

from __future__ import annotations

import dataclasses
import copy
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jp
from etils import epath
from ml_collections import config_dict

from myosuite.envs.myo.backends.mjx.mjx_env_config import (
    MjxPoseConfig,
    MjxReachConfig,
    MjxWalkConfig,
)
from myosuite.envs.myo.backends.mjx.musclemimic_env import MjxMuscleMimicIOEnv
from myosuite.envs.myo.backends.mjx.musclemimic_fullbody_env import (
    MjxMuscleMimicFullbodyEnv,
)
from myosuite.envs.myo.backends.mjx.pose_env import MjxPoseEnv
from myosuite.envs.myo.backends.mjx.reach_env import MjxReachEnv
from myosuite.envs.myo.backends.mjx.walk_env import MjxWalkEnv
from myosuite.envs.myo.assets._resolve import (
    resolve_elbow_xml as _resolve_elbow_xml,
    resolve_finger_xml as _resolve_finger_xml,
)
from myosuite.integrations.musclemimic.bimanual_model import default_mimic_config
from myosuite.integrations.musclemimic.fullbody_model import (
    default_mimic_fullbody_config,
)
from myosuite.utils.asset_path_resolver import get_sim_asset_root

# ---------------------------------------------------------------------------
# Conversion helper: dataclass → ConfigDict (required by mujoco_playground)
# ---------------------------------------------------------------------------


def _to_config_dict(obj: object) -> object:
    """Recursively convert a dataclass to a ``ml_collections.ConfigDict``.

    Uses field-level iteration (not ``dataclasses.asdict``) so that non-dataclass
    values like JAX arrays are passed through unchanged rather than deepcopied.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return config_dict.create(
            **{
                f.name: _to_config_dict(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
            }
        )
    return obj


# PPO training hyperparameters (shared across envs)
ppo_config = config_dict.create(
    num_timesteps=40_000_000,
    num_evals=16,
    reward_scaling=0.1,
    num_eval_envs=128,
    clipping_epsilon=0.3,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    num_updates_per_batch=8,
    num_resets_per_eval=1,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=0.001,
    batch_size=512,
    max_grad_norm=1.0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(64, 64, 64),
        value_hidden_layer_sizes=(64, 64, 64),
        policy_obs_key="state",
        value_obs_key="state",
    ),
)

# ---------------------------------------------------------------------------
# Per-environment base model paths
# ---------------------------------------------------------------------------

_MYOSUITE = epath.Path(epath.resource_path("myosuite"))

_ELBOW_MODEL = _resolve_elbow_xml("myoelbow_1dof6muscles.xml")
_FINGER_MODEL = _resolve_finger_xml("myofinger_v0.xml")
_HAND_MODEL = Path(str(_MYOSUITE / "envs/myo/assets/hand/myohand_pose.xml"))

# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

ALL_ENVS = [
    "MjxElbowPoseFixed-v0",
    "MjxElbowPoseRandom-v0",
    "MjxFingerPoseFixed-v0",
    "MjxFingerPoseRandom-v0",
    "MjxHandPoseRandom-v0",
    "MjxHandReachFixed-v0",
    "MjxHandReachRandom-v0",
    "MjxFingerReachRandom-v0",
    # Walk (leg) — mirrors myoLegWalk-v0 on the MJX backend
    "MjxLegWalk-v0",
    # Mimic site-tracking (bimanual / full-body)
    "MjxMimicBimanual-v0",
    "MjxMimicFullbody-v0",
    "MjxMuscleMimicBimanual-v0",
    "MjxMuscleMimicFullbody-v0",
]

# ---------------------------------------------------------------------------
# Walk env config
# ---------------------------------------------------------------------------

_MYO_SIM_ROOT = Path(str(get_sim_asset_root("myo_sim")))
# Optional trimmed MJCF (not shipped): fall back to myolegs.xml and rely on
# :func:`~myosuite.envs.myo.backends.mjx.mjx_spec_preprocess.preprocess_mjx_spec`
# to strip JAX/XLA-incompatible cylinder/ellipsoid contacts when ``mjx_impl`` ≠ warp.
_LEG_MJX_MODEL = _MYO_SIM_ROOT / "leg" / "myolegs_mjx.xml"
_LEG_FALLBACK_MODEL = _MYO_SIM_ROOT / "leg" / "myolegs.xml"


_musclemimic_bimanual_config = default_mimic_config()
_musclemimic_fullbody_config = default_mimic_fullbody_config()

# ---------------------------------------------------------------------------
# Per-environment config factories
#
# Each _cfg_* function returns a typed dataclass.  make() converts to
# ConfigDict at the mujoco_playground boundary via _to_config_dict().
# To add a new MJX env: add a factory and an entry in _MJX_SPECS below.
# ---------------------------------------------------------------------------

_LEG_MODEL = Path(
    str(_LEG_MJX_MODEL if _LEG_MJX_MODEL.exists() else _LEG_FALLBACK_MODEL)
)


def _cfg_elbow_fixed() -> MjxPoseConfig:
    return MjxPoseConfig(
        model_path=_ELBOW_MODEL,
        target_jnt_range={"r_elbow_flex": jp.array((2.0, 2.0))},
    )


def _cfg_elbow_random() -> MjxPoseConfig:
    return MjxPoseConfig(
        model_path=_ELBOW_MODEL,
        target_jnt_range={"r_elbow_flex": jp.array((0.0, 2.27))},
    )


def _cfg_finger_fixed() -> MjxPoseConfig:
    return MjxPoseConfig(
        model_path=_FINGER_MODEL,
        target_jnt_range={
            "IFadb": jp.array((0.0, 0.0)),
            "IFmcp": jp.array((0.0, 0.0)),
            "IFpip": jp.array((0.75, 0.75)),
            "IFdip": jp.array((0.75, 0.75)),
        },
    )


def _cfg_finger_random() -> MjxPoseConfig:
    return MjxPoseConfig(
        model_path=_FINGER_MODEL,
        target_jnt_range={
            "IFadb": jp.array((-0.2, 0.2)),
            "IFmcp": jp.array((-0.4, 1.0)),
            "IFpip": jp.array((0.1, 1.0)),
            "IFdip": jp.array((0.1, 1.0)),
        },
    )


def _cfg_hand_pose_random() -> MjxPoseConfig:
    return MjxPoseConfig(
        model_path=_HAND_MODEL,
        target_jnt_range={
            "pro_sup": jp.array((-0.3, 0.7)),
            "deviation": jp.array((-0.2, 0.3)),
            "flexion": jp.array((-0.8, 0.2)),
            "cmc_abduction": jp.array((-0.2, 0.8)),
            "cmc_flexion": jp.array((-0.2, 0.4)),
            "mp_flexion": jp.array((-1.35, 0.0)),
            "ip_flexion": jp.array((-1.35, 0.0)),
            "mcp2_flexion": jp.array((0.0, 1.4)),
            "mcp2_abduction": jp.array((-0.1, 0.1)),
            "pm2_flexion": jp.array((0.0, 1.5)),
            "md2_flexion": jp.array((0.0, 1.5)),
            "mcp3_flexion": jp.array((0.0, 1.5)),
            "mcp3_abduction": jp.array((-0.1, 0.1)),
            "pm3_flexion": jp.array((0.0, 1.5)),
            "md3_flexion": jp.array((0.0, 1.5)),
            "mcp4_flexion": jp.array((0.0, 1.5)),
            "mcp4_abduction": jp.array((-0.2, 0.2)),
            "pm4_flexion": jp.array((0.0, 1.5)),
            "md4_flexion": jp.array((0.0, 1.5)),
            "mcp5_flexion": jp.array((0.0, 1.5)),
            "mcp5_abduction": jp.array((-0.3, 0.3)),
            "pm5_flexion": jp.array((0.0, 1.5)),
            "md5_flexion": jp.array((0.0, 1.5)),
        },
    )


def _cfg_hand_reach_fixed() -> MjxReachConfig:
    return MjxReachConfig(
        model_path=_HAND_MODEL,
        far_th=0.044,
        target_reach_range={
            "THtip": jp.array(((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495))),
            "IFtip": jp.array(((-0.151, -0.547, 1.455), (-0.151, -0.547, 1.455))),
            "MFtip": jp.array(((-0.146, -0.547, 1.447), (-0.146, -0.547, 1.447))),
            "RFtip": jp.array(((-0.148, -0.543, 1.445), (-0.148, -0.543, 1.445))),
            "LFtip": jp.array(((-0.148, -0.528, 1.434), (-0.148, -0.528, 1.434))),
        },
    )


def _cfg_hand_reach_random() -> MjxReachConfig:
    return MjxReachConfig(
        model_path=_HAND_MODEL,
        far_th=0.034,
        target_reach_range={
            "THtip": jp.array(
                (
                    (-0.165 - 0.020, -0.537 - 0.040, 1.495 - 0.040),
                    (-0.165 + 0.040, -0.537 + 0.020, 1.495 + 0.040),
                )
            ),
            "IFtip": jp.array(
                (
                    (-0.151 - 0.040, -0.547 - 0.020, 1.455 - 0.010),
                    (-0.151 + 0.040, -0.547 + 0.020, 1.455 + 0.010),
                )
            ),
            "MFtip": jp.array(
                (
                    (-0.146 - 0.040, -0.547 - 0.020, 1.447 - 0.010),
                    (-0.146 + 0.040, -0.547 + 0.020, 1.447 + 0.010),
                )
            ),
            "RFtip": jp.array(
                (
                    (-0.148 - 0.040, -0.543 - 0.020, 1.445 - 0.010),
                    (-0.148 + 0.040, -0.543 + 0.020, 1.445 + 0.010),
                )
            ),
            "LFtip": jp.array(
                (
                    (-0.148 - 0.040, -0.528 - 0.020, 1.434 - 0.010),
                    (-0.148 + 0.040, -0.528 + 0.020, 1.434 + 0.010),
                )
            ),
        },
    )


def _cfg_finger_reach_random() -> MjxReachConfig:
    return MjxReachConfig(
        model_path=_FINGER_MODEL,
        far_th=0.10,
        target_reach_range={
            "IFtip": jp.array(((0.1, -0.1, 0.1), (0.27, 0.1, 0.3))),
        },
    )


def _cfg_leg_walk() -> MjxWalkConfig:
    return MjxWalkConfig(model_path=_LEG_MODEL)


# ---------------------------------------------------------------------------
# Declarative registry: env_name → (EnvClass, config_factory).
# To add a new MJX env: add one entry here. Do NOT touch make().
# ---------------------------------------------------------------------------

_MJX_SPECS: dict[str, tuple[type, Callable]] = {
    "MjxElbowPoseFixed-v0": (MjxPoseEnv, _cfg_elbow_fixed),
    "MjxElbowPoseRandom-v0": (MjxPoseEnv, _cfg_elbow_random),
    "MjxFingerPoseFixed-v0": (MjxPoseEnv, _cfg_finger_fixed),
    "MjxFingerPoseRandom-v0": (MjxPoseEnv, _cfg_finger_random),
    "MjxHandPoseRandom-v0": (MjxPoseEnv, _cfg_hand_pose_random),
    "MjxHandReachFixed-v0": (MjxReachEnv, _cfg_hand_reach_fixed),
    "MjxHandReachRandom-v0": (MjxReachEnv, _cfg_hand_reach_random),
    "MjxFingerReachRandom-v0": (MjxReachEnv, _cfg_finger_reach_random),
    "MjxLegWalk-v0": (MjxWalkEnv, _cfg_leg_walk),
    "MjxMimicBimanual-v0": (
        MjxMuscleMimicIOEnv,
        lambda: copy.deepcopy(_musclemimic_bimanual_config),
    ),
    "MjxMuscleMimicBimanual-v0": (
        MjxMuscleMimicIOEnv,
        lambda: copy.deepcopy(_musclemimic_bimanual_config),
    ),
    "MjxMimicFullbody-v0": (
        MjxMuscleMimicFullbodyEnv,
        lambda: copy.deepcopy(_musclemimic_fullbody_config),
    ),
    "MjxMuscleMimicFullbody-v0": (
        MjxMuscleMimicFullbodyEnv,
        lambda: copy.deepcopy(_musclemimic_fullbody_config),
    ),
}


def make(
    env_name: str,
    config_overrides: dict | None = None,
):
    """Instantiate a named MJX environment.

    Looks up *env_name* in the declarative ``_MJX_SPECS`` registry and
    instantiates the environment class with its default config.  To add a new
    MJX environment, add an entry to ``_MJX_SPECS`` — do **not** add
    another branch here.

    Typed dataclass configs (pose/reach/walk) are converted to
    ``ml_collections.ConfigDict`` at this boundary because
    ``mujoco_playground.MjxEnv`` requires ``config.lock()``.

    Args:
        env_name: One of the names in ``ALL_ENVS``.
        config_overrides: Optional dict of config keys to override on the
            default config before passing to the env constructor.

    Returns:
        An initialised MJX environment instance.

    Raises:
        ValueError: If *env_name* is not recognised.
    """
    if env_name not in _MJX_SPECS:
        raise ValueError(f"Unknown MJX env '{env_name}'.  Available: {ALL_ENVS}")

    env_cls, cfg_factory = _MJX_SPECS[env_name]
    cfg = cfg_factory()

    if config_overrides:
        if dataclasses.is_dataclass(cfg):
            cfg = dataclasses.replace(cfg, **config_overrides)
        else:
            for k, v in config_overrides.items():
                cfg[k] = v

    if dataclasses.is_dataclass(cfg):
        cfg = _to_config_dict(cfg)

    return env_cls(config=cfg)


def get_default_config(env_name: str) -> object:
    """Return the default config for a registered MJX environment.

    Returns a typed dataclass for pose/reach/walk envs, or a
    ``ml_collections.ConfigDict`` for mimic envs.
    """
    if env_name not in _MJX_SPECS:
        raise ValueError(
            f"Env '{env_name}' not found. Available: {list(_MJX_SPECS.keys())}"
        )
    _, cfg_factory = _MJX_SPECS[env_name]
    return cfg_factory()
