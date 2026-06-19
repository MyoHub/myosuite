# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Registration of all myochallenge environments."""

import pathlib

import numpy as np

import myosuite.core.registry as _registry
from myosuite.envs.myo.assets._resolve import resolve_osl_xml as _resolve_osl_xml
from myosuite.envs.myo.tasks.challenge.saber_task_spec import (
    register_saber_p0_env,
)

_ASSETS_ROOT = pathlib.Path(__file__).parents[2] / "assets"


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


# ============================================== MyoChallenge 2025 envs ==============================================
## MyoChallenge Locomotion P1 (Soccer)
_reg(
    env_id="myoChallengeSoccerP1-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.soccer:SoccerEnv",
    max_episode_steps=2000,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "leg_soccer" / "myolegs_soccer.xml"),
        "normalize_act": True,
        "min_agent_spawn_distance": 1,
        "reset_type": "none",
        "goalkeeper_probabilities": (0, 0, 1),
    },
)

_reg(
    env_id="myoChallengeSoccerP2-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.soccer:SoccerEnv",
    max_episode_steps=2000,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "leg_soccer" / "myolegs_soccer.xml"),
        "normalize_act": True,
        "min_agent_spawn_distance": 1,
        "reset_type": "random",
        "goalkeeper_probabilities": (0.1, 0.3, 0.6),
        "max_time_sec": 10,
    },
)

_reg(
    env_id="myoChallengeTableTennisP0-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.tabletennis:TableTennisEnv",
    max_episode_steps=300,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "arm" / "myoarm_tabletennis.xml"),
        "normalize_act": True,
        "frame_skip": 5,
    },
)

_reg(
    env_id="myoChallengeTableTennisP1-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.tabletennis:TableTennisEnv",
    max_episode_steps=300,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "arm" / "myoarm_tabletennis.xml"),
        "normalize_act": True,
        "ball_xyz_range": {"high": [-1.20, -0.45, 1.5], "low": [-1.25, -0.5, 1.4]},
        "frame_skip": 5,
    },
)

_reg(
    env_id="myoChallengeTableTennisP2-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.tabletennis:TableTennisEnv",
    max_episode_steps=300,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "arm" / "myoarm_tabletennis.xml"),
        "normalize_act": True,
        "ball_qvel": True,
        "paddle_mass_range": (0.10, 0.15),
        "qpos_noise_range": None,
        "ball_xyz_range": {"high": [-0.8, 0.5, 1.5], "low": [-1.25, -0.5, 1.4]},
        "ball_friction_range": {
            "high": [1.1, 0.006, 0.00003],
            "low": [0.9, 0.004, 0.00001],
        },
        "frame_skip": 5,
    },
)

_reg(
    env_id="myoChallengeBimanual-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.bimanual:BimanualEnv",
    max_episode_steps=1000,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "arm" / "myoarm_bionic_bimanual.xml"),
        "normalize_act": True,
        "frame_skip": 5,
        "obj_scale_change": [0.1, 0.05, 0.1],
        "obj_mass_change": (-0.050, 0.050),
        "obj_friction_change": (0.1, 0.001, 0.00002),
    },
)

# MyoChallenge 2024 envs ==============================================
_reg(
    env_id="myoChallengeOslRunFixed-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.run_track:RunTrackEnv",
    max_episode_steps=1000,
    kwargs={
        "model_path": str(_resolve_osl_xml("myoosl_runtrack.xml")),
        "normalize_act": True,
        "reset_type": "random",
        "terrain": "flat",
        "hills_difficulties": (0.0, 0.1, 0.0, 0.5, 0.0, 0.8, 0.0, 1.0),
        "rough_difficulties": (0.0, 0.1, 0.0, 0.15, 0.0, 0.2, 0.0, 0.3),
        "stairs_difficulties": (0.0, 0.05, 0.0, 0.1, 0.0, 0.2, 0.0, 0.3),
        "end_pos": -15,
        "frame_skip": 5,
        "start_pos": 14,
        "init_pose_path": str(_ASSETS_ROOT / "leg" / "sample_gait_cycle.csv"),
        "max_episode_steps": 1000,
    },
)


_reg(
    env_id="myoChallengeOslRunRandom-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.run_track:RunTrackEnv",
    max_episode_steps=60000,
    kwargs={
        "model_path": str(_resolve_osl_xml("myoosl_runtrack.xml")),
        "normalize_act": True,
        "reset_type": "random",
        "terrain": "random",
        "hills_difficulties": (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.03,
            0.0,
            0.06,
            0.0,
            0.09,
            0.0,
            0.12,
            0.0,
            0.15,
            0.0,
            0.18,
            0.0,
            0.21,
            0.0,
            0.24,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        "rough_difficulties": (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.03,
            0.0,
            0.06,
            0.0,
            0.09,
            0.0,
            0.12,
            0.0,
            0.15,
            0.0,
            0.18,
            0.0,
            0.21,
            0.0,
            0.24,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        "stairs_difficulties": (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.03,
            0.0,
            0.06,
            0.0,
            0.09,
            0.0,
            0.12,
            0.0,
            0.15,
            0.0,
            0.18,
            0.0,
            0.21,
            0.0,
            0.24,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        "end_pos": -45,
        "frame_skip": 5,
        "start_pos": 58,
        "init_pose_path": str(_ASSETS_ROOT / "leg" / "sample_gait_cycle.csv"),
        "max_episode_steps": 60000,
    },
)

# MyoChallenge 2023 envs ==============================================
# MyoChallenge Manipulation P1
_reg(
    env_id="myoChallengeRelocateP1-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.relocate:RelocateEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "arm" / "myoarm_relocate.xml"),
        "normalize_act": True,
        "frame_skip": 5,
        "pos_th": 0.1,  # cover entire base of the receptacle
        "rot_th": np.inf,  # ignore rotation errors
        "target_xyz_range": {"high": [0.2, -0.1, 0.9], "low": [0.0, -0.35, 0.9]},
        "target_rxryrz_range": {"high": [0.0, 0.0, 0.0], "low": [0.0, 0.0, 0.0]},
    },
)

# MyoChallenge Manipulation P2
_reg(
    env_id="myoChallengeRelocateP2-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.relocate:RelocateEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "arm" / "myoarm_relocate.xml"),
        "normalize_act": True,
        "frame_skip": 5,
        "pos_th": 0.1,  # cover entire base of the receptacle
        "rot_th": np.inf,  # ignore rotation errors
        "qpos_noise_range": 0.01,  # jnt initialization range
        "target_xyz_range": {"high": [0.3, -0.1, 1.05], "low": [0.0, -0.45, 0.9]},
        "target_rxryrz_range": {"high": [0.2, 0.2, 0.2], "low": [-0.2, -0.2, -0.2]},
        "obj_xyz_range": {"high": [0.1, -0.15, 1.0], "low": [-0.1, -0.35, 1.0]},
        "obj_geom_range": {"high": [0.025, 0.025, 0.025], "low": [0.015, 0.015, 0.015]},
        "obj_mass_range": {"high": 0.200, "low": 0.050},  # 50gms to 200 gms
        "obj_friction_range": {
            "high": [1.2, 0.006, 0.00012],
            "low": [0.8, 0.004, 0.00008],
        },
    },
)

# Register MyoChallenge Manipulation P2 Evals
_reg(
    env_id="myoChallengeRelocateP2eval-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.relocate:RelocateEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "arm" / "myoarm_relocate.xml"),
        "normalize_act": True,
        "frame_skip": 5,
        "pos_th": 0.1,  # cover entire base of the receptacle
        "rot_th": np.inf,  # ignore rotation errors
        "qpos_noise_range": 0.015,  # jnt initialization range
        "target_xyz_range": {"high": [0.4, -0.1, 1.1], "low": [-0.5, -0.5, 0.9]},
        "target_rxryrz_range": {"high": [0.3, 0.3, 0.3], "low": [-0.3, -0.3, -0.3]},
        "obj_xyz_range": {"high": [0.15, -0.10, 1.0], "low": [-0.20, -0.40, 1.0]},
        "obj_geom_range": {"high": [0.025, 0.025, 0.035], "low": [0.015, 0.015, 0.015]},
        "obj_mass_range": {"high": 0.300, "low": 0.050},  # 50gms to 250 gms
        "obj_friction_range": {
            "high": [1.2, 0.006, 0.00012],
            "low": [0.8, 0.004, 0.00008],
        },
    },
)


## MyoChallenge Locomotion P1
_reg(
    env_id="myoChallengeChaseTagP1-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.chasetag:ChaseTagEnv",
    max_episode_steps=2000,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "leg" / "myolegs_chasetag.xml"),
        "normalize_act": True,
        "win_distance": 0.5,
        "min_spawn_distance": 2,
        "reset_type": "init",
        "terrain": "FLAT",
        "task_choice": "CHASE",
        "hills_range": (0.0, 0.0),
        "rough_range": (0.0, 0.0),
        "relief_range": (0.0, 0.0),
        "opponent_probabilities": (0.1, 0.45, 0.45),
    },
)

# MyoChallenge Locomotion P2
_reg(
    env_id="myoChallengeChaseTagP2-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.chasetag:ChaseTagEnv",
    max_episode_steps=2000,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "leg" / "myolegs_chasetag.xml"),
        "normalize_act": True,
        "win_distance": 0.5,
        "min_spawn_distance": 2,
        "reset_type": "random",
        "terrain": "random",
        "task_choice": "random",
        "hills_range": (0.03, 0.23),
        "rough_range": (0.05, 0.1),
        "relief_range": (0.1, 0.3),
        "repeller_opponent": False,
        "chase_vel_range": (1.0, 1.0),
        "random_vel_range": (-2, 2),
        "opponent_probabilities": (0.1, 0.45, 0.45),
    },
)

# Register MyoChallenge Locomotion P2 Evals
_reg(
    env_id="myoChallengeChaseTagP2eval-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.chasetag:ChaseTagEnv",
    max_episode_steps=2000,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "leg" / "myolegs_chasetag.xml"),
        "normalize_act": True,
        "win_distance": 0.5,
        "min_spawn_distance": 2,
        "reset_type": "random",
        "terrain": "random",
        "task_choice": "random",
        "hills_range": (0.03, 0.23),
        "rough_range": (0.05, 0.1),
        "relief_range": (0.1, 0.3),
        "repeller_opponent": True,
        "chase_vel_range": (1, 5),
        "random_vel_range": (-2, 2),
        "repeller_vel_range": (0.3, 1),
        "opponent_probabilities": (0.1, 0.35, 0.35, 0.2),
    },
)

# MyoChallenge 2022 envs ==============================================
# MyoChallenge Die: Trial env
_reg(
    env_id="myoChallengeDieReorientDemo-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.reorient:ReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "hand" / "myohand_die.xml"),
        "normalize_act": True,
        "frame_skip": 5,
        "pos_th": np.inf,  # ignore position error threshold
        "goal_pos": (0, 0),  # 0 cm
        "goal_rot": (-0.785, 0.785),  # +-45 degrees
    },
)
# MyoChallenge Die: Phase1 env
_reg(
    env_id="myoChallengeDieReorientP1-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.reorient:ReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "hand" / "myohand_die.xml"),
        "normalize_act": True,
        "frame_skip": 5,
        "goal_pos": (-0.010, 0.010),  # +- 1 cm
        "goal_rot": (-1.57, 1.57),  # +-90 degrees
    },
)
# MyoChallenge Die: Phase2 env
_reg(
    env_id="myoChallengeDieReorientP2-v0",
    entry_point="myosuite.envs.myo.tasks.challenge.reorient:ReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "hand" / "myohand_die.xml"),
        "normalize_act": True,
        "frame_skip": 5,
        # Randomization in goals
        "goal_pos": (-0.020, 0.020),  # +- 2 cm
        "goal_rot": (-3.14, 3.14),  # +-180 degrees
        # Randomization in physical properties of the die
        "obj_size_change": 0.007,  # +-7mm delta change in object size
        "obj_mass_range": (0.050, 0.250),  # 50gms to 250 gms
        "obj_friction_change": (0.2, 0.001, 0.00002),  # nominal: 1.0, 0.005, 0.0001
    },
)

# MyoChallenge Baoding: Phase1 env
_reg(
    env_id="myoChallengeBaodingP1-v1",
    entry_point="myosuite.envs.myo.tasks.challenge.baoding:BaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "hand" / "myohand_baoding.xml"),
        "normalize_act": True,
        "goal_time_period": (5, 5),
        "goal_xrange": (0.025, 0.025),
        "goal_yrange": (0.028, 0.028),
    },
)

# MyoChallenge Baoding: Phase1 env
_reg(
    env_id="myoChallengeBaodingP2-v1",
    entry_point="myosuite.envs.myo.tasks.challenge.baoding:BaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": str(_ASSETS_ROOT / "hand" / "myohand_baoding.xml"),
        "normalize_act": True,
        "goal_time_period": (4, 6),
        "goal_xrange": (0.020, 0.030),
        "goal_yrange": (0.022, 0.032),
        # Randomization in physical properties of the baoding balls
        "obj_size_range": (0.018, 0.024),  # Object size range. Nominal 0.022
        "obj_mass_range": (0.030, 0.300),  # Object weight range. Nominal 43 gms
        "obj_friction_change": (0.2, 0.001, 0.00002),  # nominal: 1.0, 0.005, 0.0001
        "task_choice": "random",
    },
)

# Register Saber P0 via EnvSpec adapter path.
register_saber_p0_env()

# Register two-agent competitive boxing environments.
# These use factory entry-points (not TaskConfig) because they wrap
# ModularMultiAgentTaskEnv — routed through register_env() which is
# idempotent and applies the instability wrapper consistently.
try:
    from myosuite.core.registry import (
        register_env as _register_env,
        register_task as _register_task,
    )
    from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_task_config import (
        BoxingVsTaskConfig as _BoxingVsTaskConfig,
    )

    _register_task(_BoxingVsTaskConfig(), env_id="myoChallengeBoxingVs-v0")
    _register_env(
        "myoChallengeBoxingMannequin-v0",
        entry_point="myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_env:make_boxing_mannequin_env",
        max_episode_steps=500,
        wrap_mj_instability_termination=False,
    )
    # Six-target pad machine used by MyoChallenge boxing P0.
    _register_env(
        "myoChallengeBoxingP0-v0",
        entry_point="myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_env:make_boxing_6targets",
        max_episode_steps=500,
        wrap_mj_instability_termination=False,
    )
    del _register_env, _register_task, _BoxingVsTaskConfig
except Exception as _e:  # noqa: BLE001
    import warnings as _w

    _w.warn(f"Boxing VS envs not registered: {_e}", stacklevel=1)
    del _e, _w
