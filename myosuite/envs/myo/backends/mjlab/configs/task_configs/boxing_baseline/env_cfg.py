import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import TendonEffortActionCfg
from mjlab.managers import CommandTermCfg, MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViewerConfig
from myosuite.envs.myo.backends.mjlab.configs.asset_configs.muscle_mimic_body.full_body_constants import (
    get_full_body_cfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.envs.mdp import dr
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.sensor import ContactMatch, ContactSensorCfg
from .mdp.rewards import activation
from .mdp.observations import raw_activation, mean_activation
from .mdp.terminations import body_height_below_minimum


def stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Torque actuated standing environment configuration.

    Args:
      play: If True, disables corruption and extends episode length for evaluation.
    """

    # ==============================================================================
    # Scene Configuration
    # ==============================================================================

    scene_cfg = SceneCfg(
        num_envs=64 if not play else 16,  # Fewer envs for play mode
        extent=1.0,  # Spacing between environments
        entities={"human": get_full_body_cfg()},
    )

    scene_cfg = add_contact_sensors(scene_cfg)

    viewer_cfg = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="human",
        body_name="thorax",
        distance=3.0,
        elevation=10.0,
        azimuth=90.0,
    )

    sim_cfg = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.004,  # 250 Hz control
            iterations=4,
        ),
        njmax=270,
        nconmax=65,
    )

    # ==============================================================================
    # Actions
    # ==============================================================================

    actions = {
        "joint_pos": TendonEffortActionCfg(
            entity_name="human",
            actuator_names=(".*",),
            scale=1,
            clip={".*": (0.0, 1.0)},
        ),
    }

    # ==============================================================================
    # Observations
    # ==============================================================================

    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "human/lin_vel"},
            noise=Unoise(n_min=-0, n_max=0),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "human/ang_vel"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("human")},
            noise=Unoise(n_min=-0.00, n_max=0.005),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("human")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("human")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "activations": ObservationTermCfg(
            func=raw_activation, params={"asset_cfg": SceneEntityCfg("human")}
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
    }

    critic_terms = {
        **actor_terms,
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # ==============================================================================
    # Rewards
    # ==============================================================================

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("human"),
                "command_name": "twist",
                "std": math.sqrt(1),
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("human"),
                "command_name": "twist",
                "std": math.sqrt(2),
            },
        ),
        "body_orientation_l2": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=-0.0,
            params={
                "asset_cfg": SceneEntityCfg("human", body_names=(".*pelvis",))
            },  # Set per-human.
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.03,  # Override per-human
            params={
                "asset_cfg": SceneEntityCfg("human", body_names=(".*pelvis",))
            },  # Set per-human.
        ),
        "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-100.0),
        "is_surviving": RewardTermCfg(func=mdp.is_alive, weight=4),
        "joint_acc_l2": RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=-2.5e-9,
            params={"asset_cfg": SceneEntityCfg("human", joint_names=(".*",))},
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("human", joint_names=(".*",))},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.0002),
        "activation_l2": RewardTermCfg(func=activation, weight=-0.001),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-3,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.1,
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=0.2,
            params={
                "asset_cfg": SceneEntityCfg("human", joint_names=(".*",)),
                "command_name": "twist",
                "std_standing": {".*": 0.05},
                "std_walking": {".*": 0.35},
                "std_running": {".*": 0.5},
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        ),
    }

    # ==============================================================================
    # Commands
    # ==============================================================================
    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="human",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=1,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 2.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
            ),
        )
    }

    metrics = {
        "mean_activation": MetricsTermCfg(
            func=mean_activation,
            params={
                "asset_cfg": SceneEntityCfg("human"),
            },
        )
    }

    # ==============================================================================
    # Events
    # ==============================================================================

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("human"),
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.94996 - 0.01, 0.94996 + 0.05),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_human_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("human", joint_names=(".*",)),
            },
        ),
        "push_human": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "asset_cfg": SceneEntityCfg("human"),
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "human", geom_names=(".*foot.*", ".*talus.*", ".*floor.*")
                ),  # Set per-human.
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,  # All foot geoms share the same friction.
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg(
                    "human", body_names=(".*thorax.*")
                ),  # Set per-human.
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        ),
    }

    # ==============================================================================
    # Terminations
    # ==============================================================================

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={
                "asset_cfg": SceneEntityCfg("human"),
                "limit_angle": math.radians(70.0),
            },
        ),
        "head_height": TerminationTermCfg(
            func=body_height_below_minimum,
            params={
                "asset_cfg": SceneEntityCfg("human", body_names=(".*head")),
                "minimum_height": 1.35,
            },
        ),
    }

    # ==============================================================================
    # Environment Configuration
    # ==============================================================================

    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        observations=observations,
        actions=actions,
        rewards=rewards,
        metrics=metrics,
        events=events,
        terminations=terminations,
        commands=commands,
        sim=sim_cfg,
        viewer=viewer_cfg,
        decimation=1,  # No action repeat
        episode_length_s=int(1e9)
        if play
        else 10.0,  # Infinite for play, 10s for training
    )


def add_contact_sensors(scene):
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"(calc).*",
            entity="human",
        ),
        secondary=ContactMatch(mode="geom", pattern="human/floor"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="human"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="human"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    scene.sensors = (scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    return scene
