# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Registration of all myobase environments (reach, pose, walk, etc.)."""

# TaskConfig-based spec registrations (new pattern — no subclassing required)
import myosuite.envs.myo.tasks.basic.specs  # noqa: F401

import pathlib

import numpy as np

import myosuite.core.registry as _registry
from myosuite.envs.myo.assets._resolve import (
    resolve_elbow_xml as _resolve_elbow_xml,
    resolve_finger_xml as _resolve_finger_xml,
    resolve_leg_xml as _resolve_leg_xml,
    resolve_torso_xml as _resolve_torso_xml,
)

_ASSETS_ROOT = pathlib.Path(__file__).parents[2] / "assets"


def _resolve_hand_xml(filename: str) -> pathlib.Path:
    """Resolve hand XML: local task assets first (have task-specific sites), pip package fallback."""
    local = _ASSETS_ROOT / "hand" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "hand" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local


def _reg(env_id, entry_point, max_episode_steps, kwargs):
    """Register base env plus myo* muscle-condition variants."""
    _registry.register_env(
        env_id=env_id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs,
    )
    if env_id[:3] == "myo":
        _registry.register_env(
            env_id="myoSarc" + env_id[3:],
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            kwargs={**kwargs, "muscle_condition": "sarcopenia"},
        )
        _registry.register_env(
            env_id="myoFati" + env_id[3:],
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            kwargs={**kwargs, "muscle_condition": "fatigue"},
        )
    if env_id[:7] == "myoHand":
        _registry.register_env(
            env_id="myoReaf" + env_id[3:],
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            kwargs={**kwargs, "muscle_condition": "reafferentation"},
        )


# Finger-tip reaching ==============================
_reg(
    env_id="motorFingerReachFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0",
    max_episode_steps=200,
    kwargs={
        "model_path": str(_resolve_finger_xml("motorfinger_v0.xml")),
        "target_reach_range": {
            "IFtip": ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),
        },
        "normalize_act": True,
        "frame_skip": 5,
    },
)
_reg(
    env_id="motorFingerReachRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0",
    max_episode_steps=200,
    kwargs={
        "model_path": str(_resolve_finger_xml("motorfinger_v0.xml")),
        "target_reach_range": {
            "IFtip": ((0.1, -0.1, 0.1), (0.27, 0.1, 0.3)),
        },
        "normalize_act": True,
        "frame_skip": 5,
    },
)
_reg(
    env_id="myoFingerReachFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": str(_resolve_finger_xml("myofinger_v0.xml")),
        "target_reach_range": {
            "IFtip": ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),
        },
        "normalize_act": True,
    },
)
_reg(
    env_id="myoFingerReachRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": str(_resolve_finger_xml("myofinger_v0.xml")),
        "target_reach_range": {
            "IFtip": ((0.1, -0.1, 0.1), (0.27, 0.1, 0.3)),
        },
        "normalize_act": True,
    },
)

# Elbow posing ==============================
_reg(
    env_id="myoElbowPose1D6MFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": str(_resolve_elbow_xml("myoelbow_1dof6muscles.xml")),
        "target_jnt_range": {
            "r_elbow_flex": (2, 2),
        },
        "viz_site_targets": ("wrist",),
        "normalize_act": True,
        "pose_thd": 0.175,
        "reset_type": "random",
    },
)
_reg(
    env_id="myoElbowPose1D6MRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": str(_resolve_elbow_xml("myoelbow_1dof6muscles.xml")),
        "target_jnt_range": {
            "r_elbow_flex": (0, 2.27),
        },
        "viz_site_targets": ("wrist",),
        "normalize_act": True,
        "pose_thd": 0.175,
        "reset_type": "random",
    },
)


# Elbow Exo posing ==============================
_reg(
    env_id="myoElbowPose1D6MExoFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": str(_resolve_elbow_xml("myoelbow_1dof6muscles_1dofexo.xml")),
        "target_jnt_range": {
            "r_elbow_flex": (2, 2),
        },
        "viz_site_targets": ("wrist",),
        "normalize_act": True,
        "pose_thd": 0.175,
        "reset_type": "random",
        "weighted_reward_keys": {
            "pose": 1.0,
            "bonus": 4.0,
            "act_reg": 5.0,
            "penalty": 50,
        },
    },
)
_reg(
    env_id="myoElbowPose1D6MExoRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": str(_resolve_elbow_xml("myoelbow_1dof6muscles_1dofexo.xml")),
        "target_jnt_range": {
            "r_elbow_flex": (0, 2.27),
        },
        "viz_site_targets": ("wrist",),
        "normalize_act": True,
        "pose_thd": 0.175,
        "reset_type": "random",
        "weight_bodyname": "carry_weight",
        "weight_range": (0.1, 2),
        "weighted_reward_keys": {
            "pose": 1.0,
            "bonus": 4.0,
            "act_reg": 5.0,
            "penalty": 50,
        },
    },
)


# Finger-Joint posing ==============================
_reg(
    env_id="motorFingerPoseFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
    max_episode_steps=200,
    kwargs={
        "model_path": str(_resolve_finger_xml("motorfinger_v0.xml")),
        "target_jnt_range": {
            "IFadb": (0, 0),
            "IFmcp": (0, 0),
            "IFpip": (0.75, 0.75),
            "IFdip": (0.75, 0.75),
        },
        "viz_site_targets": ("IFtip",),
        "normalize_act": True,
        "frame_skip": 5,
    },
)
_reg(
    env_id="motorFingerPoseRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
    max_episode_steps=200,
    kwargs={
        "model_path": str(_resolve_finger_xml("motorfinger_v0.xml")),
        "target_jnt_range": {
            "IFadb": (-0.2, 0.2),
            "IFmcp": (-0.4, 1),
            "IFpip": (0.1, 1),
            "IFdip": (0.1, 1),
        },
        "viz_site_targets": ("IFtip",),
        "normalize_act": True,
        "frame_skip": 5,
    },
)
_reg(
    env_id="myoFingerPoseFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": str(_resolve_finger_xml("myofinger_v0.xml")),
        "target_jnt_range": {
            "IFadb": (0, 0),
            "IFmcp": (0, 0),
            "IFpip": (0.75, 0.75),
            "IFdip": (0.75, 0.75),
        },
        "viz_site_targets": ("IFtip",),
        "normalize_act": True,
    },
)
_reg(
    env_id="myoFingerPoseRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_path": str(_resolve_finger_xml("myofinger_v0.xml")),
        "target_jnt_range": {
            "IFadb": (-0.2, 0.2),
            "IFmcp": (-0.4, 1),
            "IFpip": (0.1, 1),
            "IFdip": (0.1, 1),
        },
        "viz_site_targets": ("IFtip",),
        "normalize_act": True,
    },
)

# Hand-Joint posing ==============================

# Remove this when the ASL envs stablizes
_reg(
    env_id="myoHandPoseFixed-v0",  # revisit
    entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_recipe": "hand_pose",
        "viz_site_targets": ("THtip_r", "IFtip_r", "MFtip_r", "RFtip_r", "LFtip_r"),
        "target_jnt_value": np.array(
            [
                0,
                0,
                0,
                0.0824475,  # cmc_flexion_r (was index 4 in old model)
                -0.0904,  # cmc_abduction_r (was index 3 in old model)
                -0.681555,
                -0.514888,
                0,
                -0.013964,
                -0.0458132,
                0,
                0.67553,
                -0.020944,
                0.76979,
                0.65982,
                0,
                0,
                0,
                0,
                0.479155,
                -0.099484,
                0.95831,
                0,
            ]
        ),
        "normalize_act": True,
        "pose_thd": 0.7,
        "reset_type": "init",  # none, init, random
        "target_type": "fixed",  # generate/ fixed
    },
)

# Create ASL envs ==============================
jnt_namesHand = [
    "pro_sup_r",
    "deviation_r",
    "flexion_r",
    "cmc_flexion_r",  # index 3 in composed hand (was cmc_abduction at index 3 in old)
    "cmc_abduction_r",  # index 4 in composed hand (was cmc_flexion at index 4 in old)
    "mp_flexion_r",
    "ip_flexion_r",
    "mcp2_flexion_r",
    "mcp2_abduction_r",
    "pm2_flexion_r",
    "md2_flexion_r",
    "mcp3_flexion_r",
    "mcp3_abduction_r",
    "pm3_flexion_r",
    "md3_flexion_r",
    "mcp4_flexion_r",
    "mcp4_abduction_r",
    "pm4_flexion_r",
    "md4_flexion_r",
    "mcp5_flexion_r",
    "mcp5_abduction_r",
    "pm5_flexion_r",
    "md5_flexion_r",
]

ASL_qpos = {}
# Positions 3 and 4 are swapped vs. old model: composed hand has cmc_flexion_r at index 3,
# cmc_abduction_r at index 4 (old model had cmc_abduction at 3, cmc_flexion at 4).
ASL_qpos[0] = (
    "0 0 0 0.28272 0.5624 -0.75573 -1.309 1.30045 -0.006982 1.45492 0.998897 1.26466 0 1.40604 0.227795 1.07614 -0.020944 1.46103 0.06284 0.83263 -0.14399 1.571 1.38248".split(
        " "
    )
)
ASL_qpos[1] = (
    "0 0 0 0.04536 0.0248 -0.7854 -1.309 0.366605 0.010473 0.269258 0.111722 1.48459 0 1.45318 1.44532 1.44532 -0.204204 1.46103 1.44532 1.48459 -0.2618 1.47674 1.48459".split(
        " "
    )
)
ASL_qpos[2] = (
    "0 0 0 0.04536 0.0248 -0.7854 -1.13447 0.514973 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.44532 -0.204204 1.46103 1.44532 1.48459 -0.2618 1.47674 1.48459".split(
        " "
    )
)
ASL_qpos[3] = (
    "0 0 0 0.25305 0.3384 0.01569 -0.0262045 0.645885 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.571 -0.036652 1.52387 1.45318 1.40604 -0.068068 1.39033 1.571".split(
        " "
    )
)
ASL_qpos[4] = (
    "0 0 0 -0.147495 0.6392 -0.7854 -1.309 0.637158 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 0.21994 -0.068068 0.274925 0.01571".split(
        " "
    )
)
ASL_qpos[5] = (
    "0 0 0 0.25305 0.3384 0.01569 -0.0262045 0.645885 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 0.21994 -0.068068 0.274925 0.01571".split(
        " "
    )
)
ASL_qpos[6] = (
    "0 0 0 -0.147495 0.6392 -0.7854 -1.309 0.637158 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 1.1861 -0.2618 1.35891 1.48459".split(
        " "
    )
)
ASL_qpos[7] = (
    "0 0 0 0.01569 0.524 -0.7854 -1.309 0.645885 -0.006982 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.28036 -0.115192 1.52387 1.45318 0.432025 -0.068068 0.18852 0.149245".split(
        " "
    )
)
ASL_qpos[8] = (
    "0 0 0 0.22338 0.428 -0.7854 -1.309 0.645885 -0.006982 0.128305 0.194636 1.39033 0 1.08399 0.573415 0.667675 -0.020944 0 0.06284 0.432025 -0.068068 0.18852 0.149245".split(
        " "
    )
)
ASL_qpos[9] = (
    "0 0 0 0.28272 0.5624 -0.75573 -1.309 1.30045 -0.006982 1.45492 0.998897 0.39275 0 0.18852 0.227795 0.667675 -0.020944 0 0.06284 0.432025 -0.068068 0.18852 0.149245".split(
        " "
    )
)

# ASl Eval envs for each numerals
for k in ASL_qpos.keys():
    _reg(
        env_id="myoHandPose" + str(k) + "Fixed-v0",
        entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
        max_episode_steps=100,
        kwargs={
            "model_recipe": "hand_pose",
            "viz_site_targets": ("THtip_r", "IFtip_r", "MFtip_r", "RFtip_r", "LFtip_r"),
            "target_jnt_value": np.array(ASL_qpos[k], "float"),
            "normalize_act": True,
            "pose_thd": 0.7,
            "reset_type": "init",  # none, init, random
            "target_type": "fixed",  # generate/ fixed
        },
    )

# ASL Train Env
m = np.array([ASL_qpos[i] for i in range(10)]).astype(float)
Rpos = {}
for i_n, n in enumerate(jnt_namesHand):
    Rpos[n] = (np.min(m[:, i_n]), np.max(m[:, i_n]))

_reg(
    env_id="myoHandPoseRandom-v0",  # reconsider
    entry_point="myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_recipe": "hand_pose",
        "viz_site_targets": ("THtip_r", "IFtip_r", "MFtip_r", "RFtip_r", "LFtip_r"),
        "target_jnt_range": Rpos,
        "normalize_act": True,
        "pose_thd": 0.7,
        "reset_type": "random",  # none, init, random
        "target_type": "generate",  # generate/ fixed
    },
)


# Gait Torso Reaching ==============================


_reg(
    env_id="myoLegStandRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.leg.reach:LegReachEnvV0",
    max_episode_steps=150,
    kwargs={
        "model_path": str(_resolve_leg_xml("myolegs_with_torso.xml")),
        "joint_random_range": (
            -0.2,
            0.2,
        ),  # range of joint randomization (jnt = init_qpos + random(range)
        "target_reach_range": {
            "pelvis": ((-0.05, -0.05, 0), (0.05, 0.05, 0)),
        },
        "normalize_act": True,
        "far_th": 0.44,
    },
)


# Gait Torso Walking ==============================
_reg(
    env_id="myoLegWalk-v0",
    entry_point="myosuite.envs.myo.tasks.basic.leg.walk:LegWalkEnvV0",
    max_episode_steps=1000,
    kwargs={
        "model_path": str(_resolve_leg_xml("myolegs_with_torso.xml")),
        "normalize_act": True,
        "min_height": 0.8,  # minimum center of mass height before reset
        "max_rot": 0.8,  # maximum rotation before reset
        "hip_period": 100,  # desired periodic hip angle movement
        "reset_type": "init",  # none, init, random
        "target_x_vel": 0.0,  # desired x velocity in m/s
        "target_y_vel": 1.2,  # desired y velocity in m/s
        "target_rot": None,  # if None then the initial root pos will be taken, otherwise provide quat
    },
)

# Rough Terrain Walking  ==============================
_reg(
    env_id="myoLegRoughTerrainWalk-v0",
    entry_point="myosuite.envs.myo.tasks.basic.leg.walk:LegTerrainEnvV0",
    max_episode_steps=1000,
    kwargs={
        "model_path": str(_resolve_leg_xml("myolegs_with_torso.xml")),
        "normalize_act": True,
        "min_height": 0.8,  # minimum center of mass height before reset
        "max_rot": 0.8,  # maximum rotation before reset
        "hip_period": 100,  # desired periodic hip angle movement
        "reset_type": "init",  # none, init, random
        "target_x_vel": 0.0,  # desired x velocity in m/s
        "target_y_vel": 1.2,  # desired y velocity in m/s
        "target_rot": None,  # if None then the initial root pos will be taken, otherwise provide quat
        "terrain": "rough",
        "variant": None,
    },
)

# Hilly Walking  ==============================
_reg(
    env_id="myoLegHillyTerrainWalk-v0",
    entry_point="myosuite.envs.myo.tasks.basic.leg.walk:LegTerrainEnvV0",
    max_episode_steps=1000,
    kwargs={
        "model_path": str(_resolve_leg_xml("myolegs_with_torso.xml")),
        "normalize_act": True,
        "min_height": 0.8,  # minimum center of mass height before reset
        "max_rot": 0.8,  # maximum rotation before reset
        "hip_period": 100,  # desired periodic hip angle movement
        "reset_type": "init",  # none, init, random
        "target_x_vel": 0.0,  # desired x velocity in m/s
        "target_y_vel": 1.2,  # desired y velocity in m/s
        "target_rot": None,  # if None then the initial root pos will be taken, otherwise provide quat
        "terrain": "hilly",
        "variant": "fixed",
    },
)

# Stair Walking  ==============================
_reg(
    env_id="myoLegStairTerrainWalk-v0",
    entry_point="myosuite.envs.myo.tasks.basic.leg.walk:LegTerrainEnvV0",
    max_episode_steps=1000,
    kwargs={
        "model_path": str(_resolve_leg_xml("myolegs_with_torso.xml")),
        "normalize_act": True,
        "min_height": 0.8,  # minimum center of mass height before reset
        "max_rot": 0.8,  # maximum rotation before reset
        "hip_period": 100,  # desired periodic hip angle movement
        "reset_type": "init",  # none, init, random
        "target_x_vel": 0.0,  # desired x velocity in m/s
        "target_y_vel": 1.2,  # desired y velocity in m/s
        "target_rot": None,  # if None then the initial root pos will be taken, otherwise provide quat
        "terrain": "stairs",
        "variant": "fixed",
    },
)


# Hand-Joint Reaching ==============================
_reg(
    env_id="myoHandReachFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_recipe": "hand_pose",
        "target_reach_range": {
            "THtip_r": ((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495)),
            "IFtip_r": ((-0.151, -0.547, 1.455), (-0.151, -0.547, 1.455)),
            "MFtip_r": ((-0.146, -0.547, 1.447), (-0.146, -0.547, 1.447)),
            "RFtip_r": ((-0.148, -0.543, 1.445), (-0.148, -0.543, 1.445)),
            "LFtip_r": ((-0.148, -0.528, 1.434), (-0.148, -0.528, 1.434)),
        },
        "normalize_act": True,
        "far_th": 0.044,
    },
)
_reg(
    env_id="myoHandReachRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0",
    max_episode_steps=100,
    kwargs={
        "model_recipe": "hand_pose",
        "target_reach_range": {
            "THtip_r": (
                (-0.165 - 0.020, -0.537 - 0.040, 1.495 - 0.040),
                (-0.165 + 0.040, -0.537 + 0.020, 1.495 + 0.040),
            ),
            "IFtip_r": (
                (-0.151 - 0.040, -0.547 - 0.020, 1.455 - 0.010),
                (-0.151 + 0.040, -0.547 + 0.020, 1.455 + 0.010),
            ),
            "MFtip_r": (
                (-0.146 - 0.040, -0.547 - 0.020, 1.447 - 0.010),
                (-0.146 + 0.040, -0.547 + 0.020, 1.447 + 0.010),
            ),
            "RFtip_r": (
                (-0.148 - 0.040, -0.543 - 0.020, 1.445 - 0.010),
                (-0.148 + 0.040, -0.543 + 0.020, 1.445 + 0.010),
            ),
            "LFtip_r": (
                (-0.148 - 0.040, -0.528 - 0.020, 1.434 - 0.010),
                (-0.148 + 0.040, -0.528 + 0.020, 1.434 + 0.010),
            ),
        },
        "normalize_act": True,
        "far_th": 0.034,
    },
)

# Hand-Joint contact tasks ==========================
# These recipes seed from the shared release-era hand assets via an in-memory
# MjSpec and add task furniture through transforms. This preserves public-policy
# observation/contact semantics without duplicating task XMLs.
_reg(
    env_id="myoHandKeyTurnFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.key_turn:KeyTurnEnvV0",
    max_episode_steps=200,
    kwargs={
        "model_recipe": "hand_keyturn",
        "normalize_act": True,
    },
)
_reg(
    env_id="myoHandKeyTurnRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.key_turn:KeyTurnEnvV0",
    max_episode_steps=200,
    kwargs={
        "model_recipe": "hand_keyturn",
        "normalize_act": True,
        "key_init_range": (-np.pi / 2, np.pi / 2),
        "goal_th": 2 * np.pi,
    },
)


# Hold objects ==============================
_reg(
    env_id="myoHandObjHoldFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.hand.obj_hold:ObjHoldFixedEnvV0",
    max_episode_steps=75,
    kwargs={
        "model_recipe": "hand_hold",
        "normalize_act": True,
    },
)
_reg(
    env_id="myoHandObjHoldRandom-v0",  # revisit
    entry_point="myosuite.envs.myo.tasks.basic.hand.obj_hold:ObjHoldRandomEnvV0",
    max_episode_steps=75,
    kwargs={
        "model_recipe": "hand_hold",
        "normalize_act": True,
    },
)

# Pen twirl ==============================
_reg(
    env_id="myoHandPenTwirlFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.hand.pen:PenTwirlFixedEnvV0",
    max_episode_steps=50,
    kwargs={
        "model_recipe": "hand_pen",
        "normalize_act": True,
        "frame_skip": 5,
    },
)
_reg(
    env_id="myoHandPenTwirlRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.hand.pen:PenTwirlRandomEnvV0",
    max_episode_steps=50,
    kwargs={
        "model_recipe": "hand_pen",
        "normalize_act": True,
        "frame_skip": 5,
    },
)

##### MyoTorso Env

_reg(
    env_id="myoTorsoPoseFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.torso.pose:TorsoEnvV0",
    max_episode_steps=200,
    kwargs={
        "model_path": str(_resolve_torso_xml("myotorso.xml")),
        "target_jnt_range": {
            "flex_extension": (0.0, 0.0),
            "lat_bending": (-0.1, 0.1),
            "axial_rotation": (0, 0),
            "Abs_t1": (0, 0),
            "Abs_t2": (0, 0),
            "Abs_r3": (0, 0),
            "L4_L5_FE": (0, 0),
            "L4_L5_LB": (0, 0),
            "L4_L5_AR": (0, 0),
            "L3_L4_FE": (0, 0),
            "L3_L4_LB": (0, 0),
            "L3_L4_AR": (0, 0),
            "L2_L3_FE": (0, 0),
            "L2_L3_LB": (0, 0),
            "L2_L3_AR": (0, 0),
            "L1_L2_FE": (0, 0),
            "L1_L2_LB": (0, 0),
            "L1_L2_AR": (0, 0),
        },
        "normalize_act": True,
        "frame_skip": 5,
    },
)


_reg(
    env_id="myoTorsoExoPoseFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.torso.pose:TorsoEnvV0",
    max_episode_steps=200,
    kwargs={
        "model_path": str(_resolve_torso_xml("myotorso_exosuit.xml")),
        "target_jnt_range": {
            "flex_extension": (0, 0),
            "lat_bending": (-0.1, 0.1),
            "axial_rotation": (0, 0),
            "Abs_t1": (0, 0),
            "Abs_t2": (0, 0),
            "Abs_r3": (0, 0),
            "L4_L5_FE": (0, 0),
            "L4_L5_LB": (0, 0),
            "L4_L5_AR": (0, 0),
            "L3_L4_FE": (0, 0),
            "L3_L4_LB": (0, 0),
            "L3_L4_AR": (0, 0),
            "L2_L3_FE": (0, 0),
            "L2_L3_LB": (0, 0),
            "L2_L3_AR": (0, 0),
            "L1_L2_FE": (0, 0),
            "L1_L2_LB": (0, 0),
            "L1_L2_AR": (0, 0),
        },
        "normalize_act": True,
        "frame_skip": 5,
    },
)


# SAR REORIENT: 8-object ==============================
_reg(
    env_id="myoHandReorient8-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reorient_sar:Geometries8EnvV0Gymnasium",
    max_episode_steps=50,
    kwargs={
        "model_recipe": "hand_sar",
        "normalize_act": True,
        "frame_skip": 5,
    },
)

# SAR REORIENT: 100-object
_reg(
    env_id="myoHandReorient100-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reorient_sar:Geometries100EnvV0Gymnasium",
    max_episode_steps=50,
    kwargs={
        "model_recipe": "hand_sar",
        "normalize_act": True,
        "frame_skip": 5,
    },
)

# SAR TEST ENVIRONMENT: in-distribution
_reg(
    env_id="myoHandReorientID-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reorient_sar:InDistribution",
    max_episode_steps=50,
    kwargs={
        "model_recipe": "hand_sar",
        "normalize_act": True,
        "frame_skip": 5,
    },
)

# SAR TEST ENVIRONMENT: out of distribution
_reg(
    env_id="myoHandReorientOOD-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reorient_sar:OutofDistribution",
    max_episode_steps=50,
    kwargs={
        "model_recipe": "hand_sar",
        "normalize_act": True,
        "frame_skip": 5,
    },
)

# Arm Reaching ==============================
_reg(
    env_id="myoArmReachFixed-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0",
    max_episode_steps=150,
    kwargs={
        "model_recipe": "arm_reach",
        "target_reach_range": {
            "forearm_tip": ((-0.2, -0.2, 1.2), (-0.2, -0.2, 1.2)),
        },
        "normalize_act": True,
        "far_th": 1.0,
    },
)

_reg(
    env_id="myoArmReachRandom-v0",
    entry_point="myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0",
    max_episode_steps=150,
    kwargs={
        "model_recipe": "arm_reach",
        "target_reach_range": {
            "forearm_tip": (
                (-0.2 - 0.15, -0.2 - 0.15, 1.2 - 0.15),
                (-0.2 + 0.15, -0.2 + 0.15, 1.2 + 0.15),
            ),
        },
        "normalize_act": True,
        "far_th": 1.0,
    },
)
