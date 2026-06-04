# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Register MyoSuite tasks with mjlab's task registry so make_env(..., backend="mjlab") works.

When mjlab loads this package via the mjlab.tasks entry point, this module is not
auto-imported; the entry point targets myosuite.envs.myo.backends.mjlab (the parent __init__.py).
We call register_mjlab_tasks() from there so that env ids like myoElbowPose1D6MFixed-v0
appear in list_tasks() and can be created via load_env_cfg + ManagerBasedRlEnv.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg, ManagerBasedRlEnv
from mjlab.envs.mdp import Entity, dataclass
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.envs.mdp import dr, events as event_fns
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.managers import EventTermCfg, ManagerTermBase, MetricsTermCfg, RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.actuator import XmlActuatorCfg as _XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.sensor import CameraSensorCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
import mujoco
import numpy as np
import torch

from myosuite.envs.myo.backends.mjlab.myouser_utils import (
    add_button_to_spec,
)
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
    MyoMuscleActivationActionCfg,
)

if TYPE_CHECKING:
    from myouser.envs.myo.myouser.hydra_cli import (
        BaseEnvConfig,
        Config,
    )  # TODO: copy hydra API from myouser to myosuite4

_UNIVERSAL_ENTITY_NAME = "universal_robot"
_UNIVERSAL_ENTITY_CFG = SceneEntityCfg(_UNIVERSAL_ENTITY_NAME)
_OBS_PROPRIO_ENTITY_CFG = SceneEntityCfg(
    _UNIVERSAL_ENTITY_NAME, site_names=["fingertip"]
)

_MOBL_ARMS_MUSCLE_NAMES = (
    "DELT1",
    "DELT2",
    "DELT3",
    "SUPSP",
    "INFSP",
    "SUBSC",
    "TMIN",
    "TMAJ",
    "PECM1",
    "PECM2",
    "PECM3",
    "LAT1",
    "LAT2",
    "LAT3",
    "CORB",
    "TRIlong",
    "TRIlat",
    "TRImed",
    "ANC",
    "SUP",
    "BIClong",
    "BICshort",
    "BRA",
    "BRD",
    "PT",
    "PQ",
)


def _get_myouser_root() -> Path:
    """Resolve the installed ``myouser`` package root lazily."""
    from etils import epath

    return Path(epath.resource_path("myouser")).parent


def _universal_spec_fn(xml_file: Path, config: BaseEnvConfig) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml_file))
    spec = add_task_relevant_geoms(spec, config)
    return spec


def add_task_relevant_geoms(
    spec: mujoco.MjSpec, config: BaseEnvConfig
) -> mujoco.MjSpec:
    """
    Add task-specific targets to the MuJoCo model.

    Parameters
    ----------
    spec : mujoco.MjSpec
        The model specification to modify.

    Returns
    -------
    mujoco.MjSpec
        The updated specification including task-specific geometry elements.
    """
    task_config = config.env.task_config
    targets = task_config.targets
    num_targets = targets.num_targets
    targets = [getattr(targets, f"target_{i}") for i in range(num_targets)]
    len(targets)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ##TODO: do not hardcode!
    target_coordinates_origin = np.array([-0.0068, -0.1747, 1.0257])

    for i, target_cfg in enumerate(targets):
        if target_cfg.name == "pointing_target":
            ##TODO: unify add_sphere_to_spec and add_button_to_spec
            # target_obj = ButtonTargetClass(
            #     phase_number=i,
            #     total_phases=total_phases,
            #     position=(np.array(target_cfg.position) + target_coordinates_origin).tolist(),
            #     geom_size=target_cfg.size,
            #     site_size=target_cfg.site_size,
            #     site_pos=target_cfg.site_pos,
            #     geom_margin=target_cfg.geom_margin,
            #     euler=target_cfg.euler,
            #     min_touch_force=target_cfg.min_touch_force,
            #     weighted_reward_keys=task_config.weighted_reward_keys,
            #     show_all_targets=task_config.show_all_targets,
            #     enable_extra_dist=task_config.enable_extra_dist,
            #     use_vision=False,
            # )
            spec = add_button_to_spec(
                spec,
                target_cfg,
                target_id=i,
                target_coordinates_origin=target_coordinates_origin,
            )
        else:
            raise ValueError(f"Unsupported target type: {target_cfg['name']}")
    return spec


_independent_joint_names = [
    "elv_angle",
    "shoulder_elv",
    "shoulder_rot",
    "elbow_flexion",
    "pro_sup",
]
ee_pos_name = (
    "fingertip"  # TODO: compute from config (config.task_config.reach_settings.ee_site)
)


def _time(env) -> torch.Tensor:
    data = env.scene[_UNIVERSAL_ENTITY_NAME].data.data
    return data.time


def _qpos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids

    jnt_range = asset.data.joint_pos_limits[:, jnt_ids, :]
    qpos = asset.data.joint_pos[:, jnt_ids]  # asset.data.joint_pos[:, jnt_ids]

    qpos = (qpos - jnt_range[..., 0]) / (jnt_range[..., 1] - jnt_range[..., 0])
    qpos = (qpos - 0.5) * 2
    return qpos


def _qvel(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids

    qvel = asset.data.joint_vel[:, jnt_ids]
    return qvel


def _qacc(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids

    qacc = asset.data.joint_acc[:, jnt_ids]
    return qacc


def _act(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    act = (asset.data.data.act - 0.5) * 2
    return act


def _ctrl(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ctrl = asset.data.tendon_effort_target

    return ctrl


def _sites_pos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    sites_pos = asset.data.site_pos_w[
        :, asset.find_sites(ee_pos_name)[0][0]
    ]  # TODO: directly insert as asset_cfg
    return sites_pos


def _next_target_pos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target_pos = asset.data.body_com_pos_w[
        torch.arange(env.num_envs),
        asset.target_pos_ids[
            asset.current_target_id[:, 0].clip(0, asset.num_targets - 1)
        ],
    ]
    return target_pos


def _next_target_size(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target_size = asset.data.model.geom_size[
        torch.arange(env.num_envs),
        asset.target_size_ids[
            asset.current_target_id[:, 0].clip(0, asset.num_targets - 1)
        ],
    ]
    return target_size


def _phase_progress(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    phase_progress = asset.current_target_id[:, 0] / asset.num_targets
    phase_progress = -1.0 + 2.0 * phase_progress.reshape(-1, 1)
    return phase_progress


def _dwell_fraction(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    dwell_fraction = (
        (asset.target_dwell_steps_batch > 0)
        * asset.steps_inside_target
        / asset.target_dwell_steps_batch.clip(1, np.inf)
    )  # avoid nan for targets with 0 dwell steps
    dwell_fraction = -1.0 + 2.0 * dwell_fraction
    return dwell_fraction


def _sequential_distance_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG,
    exponential_distance_reward=False,
    distance_metric=10.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    distance_to_target = asset.distance_to_target[:, 0]

    ## add remaining distances for future targets
    if asset.num_targets > 1:
        distances_total = (
            distance_to_target
            + asset.between_target_distances.flip(dims=(1,))
            .cumsum(dim=1)
            .flip(dims=(1,))[
                torch.arange(env.num_envs),
                asset.current_target_id[:, 0].clip(0, asset.num_targets - 1),
            ]
        )
    else:
        distances_total = distance_to_target

    if exponential_distance_reward:
        distance_reward = (
            (1.0 - asset.inside_target)
            * (torch.exp(-distances_total * distance_metric) - 1.0)
            / distance_metric
        )
    else:
        distance_reward = -1.0 * distances_total

    return distance_reward


def _neural_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ctrl = asset.data.tendon_effort_target

    ctrl_magnitude = torch.linalg.vector_norm(ctrl, dim=-1)
    neural_effort = -1.0 * (ctrl_magnitude**2)
    return neural_effort


def _jac_effort(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    ctrl = asset.data.tendon_effort_target
    r_effort = 0.00198 * torch.linalg.vector_norm(ctrl, dim=-1) ** 2
    r_jacc = 6.67e-6 * torch.linalg.vector_norm(_qacc(env, asset_cfg), dim=-1) ** 2
    effort_cost = -(r_effort + r_jacc)
    return effort_cost


def _phase_bonus(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    phase_bonus = asset.phase_successfully_completed[:, 0]
    return phase_bonus


def _phase_successfully_completed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _OBS_PROPRIO_ENTITY_CFG,
    phase_id: int = 0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    current_target_id = asset.current_target_id[:, 0]
    phase_successfully_completed = current_target_id > phase_id
    return phase_successfully_completed


@dataclass
class TaskEntityCfg(EntityCfg):
    base_config: BaseEnvConfig | None = None

    def build(self) -> TaskEntity:
        """Build task entity instance from this config."""
        return TaskEntity(self)


class TaskEntity(Entity):
    def __init__(self, cfg: TaskEntityCfg):
        super().__init__(cfg)
        base_config = self.cfg.base_config

        self.task_config = base_config.env.task_config
        self.ctrl_dt = base_config.env.ctrl_dt
        self.num_targets = self.task_config.targets.num_targets

        self.target_pos_ids = torch.tensor(
            [
                self.find_bodies(f"body_target_{i}")[0][0]
                for i in range(self.num_targets)
            ],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.target_size_ids = torch.tensor(
            [
                self.find_geoms(f"geom_target_{i}")[0][0]
                for i in range(self.num_targets)
            ],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


class SequentialTaskLogic(ManagerTermBase):
    permitted_target_types = [
        "pointing",
        "button",
    ]  # TODO: check if this is sufficient for all target types defined in the config

    def __init__(self, cfg, env):
        asset = env.scene[_UNIVERSAL_ENTITY_NAME]
        self.asset = asset

        asset.current_target_id = torch.tensor(
            [[0]] * env.num_envs, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        asset.steps_inside_target = torch.tensor(
            [[0]] * env.num_envs, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        asset.between_target_distances = torch.zeros(
            (env.num_envs, asset.num_targets), device=asset.current_target_id.device
        )

        asset.target_types = [
            getattr(asset.task_config.targets, f"target_{i}").name.split("_")[0]
            for i in range(asset.num_targets)
        ]  # TODO: map to integer indices (enum) and use for success constraints etc., or remove from asset and directly use target_cfg in the relevant methods
        asset.target_dwell_steps = torch.tensor(
            [
                [
                    getattr(
                        getattr(asset.task_config.targets, f"target_{i}"),
                        "dwell_duration",
                        0.0,
                    )
                    / asset.ctrl_dt
                    for i in range(asset.num_targets)
                ]
            ]
            * env.num_envs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        asset.target_dwell_steps_batch = asset.target_dwell_steps[
            torch.arange(env.num_envs),
            asset.current_target_id[:, 0].clip(0, asset.num_targets - 1),
        ].reshape(-1, 1)
        asset.target_min_touch_force = torch.tensor(
            [
                [
                    getattr(
                        getattr(asset.task_config.targets, f"target_{i}"),
                        "min_touch_force",
                        0.0,
                    )
                    for i in range(asset.num_targets)
                ]
            ]
            * env.num_envs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        asset.inside_target, asset.distance_to_target, target_pos, asset.target_size = (
            self._inside_target(env)
        )
        asset.button_clicked, asset.button_touch_force = self._button_clicked(env)

        asset.phase_successfully_completed = torch.zeros(
            (env.num_envs, 1), dtype=torch.bool, device=asset.current_target_id.device
        )

    def _compute_extra_distances(self, asset, env_ids=None):
        if asset.num_targets > 1:
            asset.between_target_distances[env_ids, : asset.num_targets - 1] = (
                torch.cat(
                    tuple(
                        [
                            torch.linalg.vector_norm(
                                asset.data.body_com_pos_w[
                                    env_ids, asset.target_pos_ids[target_id]
                                ]
                                - asset.data.body_com_pos_w[
                                    env_ids, asset.target_pos_ids[target_id - 1]
                                ],
                                dim=1,
                                keepdim=True,
                            )
                            for target_id in range(1, asset.num_targets)
                        ]
                    ),
                    dim=1,
                )
            )
        else:
            pass  # no update needed

    def reset(self, env_ids) -> None:
        asset = self.asset

        asset.current_target_id[env_ids] *= 0
        asset.steps_inside_target[env_ids] *= 0

        self._compute_extra_distances(asset, env_ids)

    def __call__(self, env, env_ids) -> torch.Tensor:
        asset = env.scene[_UNIVERSAL_ENTITY_NAME]

        (
            asset.inside_target[env_ids],
            asset.distance_to_target[env_ids],
            target_pos,
            asset.target_size[env_ids],
        ) = self._inside_target(env, env_ids)
        asset.steps_inside_target[env_ids] += asset.inside_target[env_ids]

        asset.button_clicked[env_ids], asset.button_touch_force[env_ids] = (
            self._button_clicked(env, env_ids)
        )

        asset.phase_successfully_completed[env_ids] = (
            asset.steps_inside_target[env_ids]
            >= asset.target_dwell_steps_batch[env_ids]
        ) & asset.button_clicked[env_ids]
        asset.steps_inside_target[env_ids] *= asset.steps_inside_target[env_ids] * (
            ~(asset.phase_successfully_completed[env_ids])
        )  # reset if target completed

        asset.current_target_id[env_ids] += asset.phase_successfully_completed[env_ids]
        if env_ids is None:
            asset.target_dwell_steps_batch = asset.target_dwell_steps[
                torch.arange(env.num_envs),
                asset.current_target_id[:, 0].clip(0, asset.num_targets - 1),
            ].reshape(-1, 1)
        else:
            asset.target_dwell_steps_batch[env_ids] = asset.target_dwell_steps[
                env_ids,
                asset.current_target_id[env_ids, 0].clip(0, asset.num_targets - 1),
            ].reshape(-1, 1)
        # # asset.current_target_id %= asset.num_targets  # wrap around to 0 after the last target

        # current_target_id = asset.current_target_id[:, 0]
        # phase_id_metric_completed = current_target_id > phase_id_metric

        # return phase_id_metric_completed

    def _inside_target(self, env, env_ids=None) -> torch.Tensor:
        asset = env.scene[_UNIVERSAL_ENTITY_NAME]
        model = asset.data.model
        if env_ids is None:
            current_target_id = asset.current_target_id[:, 0].clip(
                0, asset.num_targets - 1
            )
            env_ids = torch.arange(
                env.num_envs, device=self.asset.current_target_id.device
            )
        else:
            current_target_id = asset.current_target_id[env_ids, 0].clip(
                0, asset.num_targets - 1
            )

        target_pos_id = asset.target_pos_ids[current_target_id]
        target_size_id = asset.target_size_ids[current_target_id]

        ee_pos = _sites_pos(env)[env_ids]
        target_pos = asset.data.body_com_pos_w[env_ids, target_pos_id]
        target_size = model.geom_size[
            env_ids, target_size_id
        ][
            :, [0]
        ]  # assuming target_size is a radius for a spherical target; adjust if using different target shapes

        distance_to_target = torch.linalg.vector_norm(
            ee_pos - target_pos, dim=-1, keepdim=True
        )
        inside_target = distance_to_target < target_size
        return inside_target, distance_to_target, target_pos, target_size

    def _button_clicked(self, env, env_ids=None) -> torch.Tensor:
        asset = env.scene[_UNIVERSAL_ENTITY_NAME]
        if env_ids is None:
            current_target_id = asset.current_target_id[:, 0].clip(
                0, asset.num_targets - 1
            )
            env_ids = torch.arange(
                env.num_envs, device=self.asset.current_target_id.device
            )
        else:
            current_target_id = asset.current_target_id[env_ids, 0].clip(
                0, asset.num_targets - 1
            )

        sensor_data_complete = torch.cat(
            [
                env.scene[f"{_UNIVERSAL_ENTITY_NAME}/sensor_target_{i}"].data
                for i in range(asset.num_targets)
            ],
            dim=1,
        )  # WARNING: asserts that each sensor produces (batch_size, 1) sensordata; in case of vector sensordata, use torch.cat(..., dim=2)
        button_touch_force = sensor_data_complete[env_ids, current_target_id].reshape(
            -1, 1
        )
        button_clicked = (
            button_touch_force
            >= asset.target_min_touch_force[env_ids, current_target_id].reshape(-1, 1)
        )  # TODO: check if sensor_force is the right way to detect button press for all target types

        return button_clicked, button_touch_force


def _make_universal_env_cfg(config: Config, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Minimal ManagerBasedRlEnvCfg for myoUserUniversal-v0 (1 env, CPU/GPU)."""

    # Extract observation keys
    obs_keys = config.env.task_config.obs_keys
    omni_keys = config.env.task_config.omni_keys
    vision_enabled = config.env.vision.enabled

    # Wrap all muscle actuators defined in the XML so that mjlab marks the
    # entity as actuated and exposes valid ctrl indices for the leg muscles.
    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlActuatorCfg(
                target_names_expr=tuple(
                    f"{name}_tendon" for name in _MOBL_ARMS_MUSCLE_NAMES
                ),  # TODO: define mapping from config.env.model_path to muscle names, or parse from XML
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )

    entity_cfg = TaskEntityCfg(
        spec_fn=functools.partial(
            _universal_spec_fn,
            xml_file=_get_myouser_root() / config.env.model_path,
            config=config,
        ),
        articulation=articulation,
        base_config=config,
    )

    # Use a task-specific entity name to avoid collisions with other tasks.
    universal_entity_name = _UNIVERSAL_ENTITY_NAME

    scene_cfg = SceneCfg(
        num_envs=1024 if vision_enabled else 4096,  # 1024,
        entities={universal_entity_name: entity_cfg},
    )

    # Add vision based on an existing camera from the XML file
    if vision_enabled:
        camera_names = ["universal_robot/fixed-eye"]
        cam_kwargs = {
            "universal_robot/fixed-eye": {
                "height": 120,  # 500, #180, #120,
                "width": 160,  # 500, #240, #160,
            },
        }
        cam_types = (
            ["rgb", "depth"] if vision_enabled else []
        )  # TODO: infer from config
        shared_cam_kwargs = dict(
            data_types=cam_types,
            enabled_geom_groups=(0, 3),
            use_shadows=False,
            use_textures=True,
        )

        cam_terms = {}
        for cam_name in camera_names:
            cam_cfg = CameraSensorCfg(
                name=cam_name.split("/")[-1],
                camera_name=cam_name,
                **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
                **shared_cam_kwargs,
            )
            scene_cfg.sensors = (scene_cfg.sensors or ()) + (cam_cfg,)
            param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
            for cam_type in cam_types:
                if cam_type == "depth":
                    _param_kwargs = param_kwargs.copy()
                    _param_kwargs["cutoff_distance"] = 0.5
                    func = manipulation_mdp.camera_depth
                else:
                    _param_kwargs = param_kwargs
                    func = manipulation_mdp.camera_rgb
                cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
                    func=func, params=_param_kwargs
                )

    num_targets = config.env.task_config.targets.num_targets
    targets = [
        getattr(config.env.task_config.targets, f"target_{i}")
        for i in range(num_targets)
    ]
    # target_origin_rel = np.array(config.env.task_config.reach_settings.target_origin_rel)
    target_origin_rel = np.array([-0.0068, -0.1747, 1.0257])  # TODO: do not hardcode!

    ## TODO: fix this
    global _OBS_PROPRIO_ENTITY_CFG
    _OBS_PROPRIO_ENTITY_CFG = SceneEntityCfg(
        _UNIVERSAL_ENTITY_NAME,
        joint_names=_independent_joint_names,
        site_names=["fingertip"],
    )

    # Full observation vector matching MyoUniversalEnv.
    # Total: (nq-2) + nv + 2 + 4 + 2 + 1 + 6 + 1 + 80 + 80 + 80 + 80 = 403+ dims.
    # Each term function receives `env` and returns a (N, d) torch.Tensor.
    _obs_terms_complete = {
        "time": ObservationTermCfg(func=_time),
        "qpos": ObservationTermCfg(
            func=_qpos, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "qvel": ObservationTermCfg(
            func=_qvel, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "qacc": ObservationTermCfg(
            func=_qacc, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "act": ObservationTermCfg(
            func=_act, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        # "last_ctrl": ObservationTermCfg(func=_ctrl),
        "ee_pos": ObservationTermCfg(
            func=_sites_pos, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        # "sensor_data": ObservationTermCfg(func=_sensor_data),
        # "task_obs": ObservationTermCfg(func=SequentialTaskObservation, params={"task_config": config.env.task_config, "ctrl_dt": config.env.ctrl_dt}),
        "target_pos": ObservationTermCfg(
            func=_next_target_pos, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "target_size": ObservationTermCfg(
            func=_next_target_size, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "phase_progress": ObservationTermCfg(
            func=_phase_progress, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        "dwell_fraction": ObservationTermCfg(
            func=_dwell_fraction, params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG}
        ),
        # "vision_rgb": ObservationTermCfg(func=manipulation_mdp.camera_rgb, params={"sensor_name": "fixed_eye"}),
        # "vision_depth": ObservationTermCfg(func=manipulation_mdp.camera_depth, params={"sensor_name": "fixed_eye", "cutoff_distance": 0.5}),
    }

    if vision_enabled:
        observations = {
            "proprioception": ObservationGroupCfg(
                terms={k: v for k, v in _obs_terms_complete.items() if k in obs_keys},
            ),
            "vision_mono": ObservationGroupCfg(
                # terms = {"rgb": ObservationTermCfg(func=manipulation_mdp.camera_rgb, params={"sensor_name": "fixed_eye"}),
                #         "depth": ObservationTermCfg(func=manipulation_mdp.camera_depth, params={"sensor_name": "fixed_eye", "cutoff_distance": 0.5})},
                terms=cam_terms,
                enable_corruption=False,
                concatenate_dim=0,
                ## To stack histories of vision:  #TODO: fix
                # history_length=2, flatten_history_dim=False, concatenate_history_dim=True #TODO: infer from config
            ),
        }
    else:
        observations = {
            "proprioception": ObservationGroupCfg(
                terms={
                    k: v
                    for k, v in _obs_terms_complete.items()
                    if k in obs_keys + omni_keys
                },
                concatenate_terms=True,
            ),
        }

    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=universal_entity_name,
            actuator_names=(
                ".*",
            ),  # TODO: define mapping from config.env.model_path to muscle names, or parse from XML
        ),
        # "tendons": TendonEffortActionCfg(
        #     entity_name=universal_entity_name,
        #     actuator_names=(".*",),
        # )
    }

    rewards = {
        "distance": RewardTermCfg(
            # func=_distance_reward,
            func=_sequential_distance_reward,
            params={
                "asset_cfg": _OBS_PROPRIO_ENTITY_CFG,
                "exponential_distance_reward": config.env.task_config.exponential_distance_reward,
                "distance_metric": config.env.task_config.distance_metric,
            },
            weight=config.env.task_config.weighted_reward_keys.get("distance", 0.0),
        ),
        "neural_effort": RewardTermCfg(
            func=_neural_effort,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get(
                "neural_effort", 0.0
            ),
        ),
        "jac_effort": RewardTermCfg(
            func=_jac_effort,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get("jac_effort", 0.0),
        ),
        "phase_bonus": RewardTermCfg(
            func=_phase_bonus,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG},
            weight=config.env.task_config.weighted_reward_keys.get("phase_bonus", 0.0),
        ),
        "done": RewardTermCfg(
            func=_phase_successfully_completed,
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG, "phase_id": num_targets - 1},
            weight=config.env.task_config.weighted_reward_keys.get("done", 0.0),
        ),
    }

    events = {
        # # Reset all entities to their default state each episode.
        # "reset_scene": EventTermCfg(
        #     func=event_fns.reset_scene_to_default,
        #     mode="reset",
        # ),
        "reset_joints": EventTermCfg(
            func=event_fns.reset_joints_by_offset,
            params={
                "asset_cfg": _OBS_PROPRIO_ENTITY_CFG,
                "position_range": (-0.1, 0.1),
                "velocity_range": (-0.1, 0.1),
            },
            mode="reset",
        ),
        # Task logic update: check if current target is reached and update to the next target accordingly
        "task_logic_update": EventTermCfg(
            func=SequentialTaskLogic,
            mode="step",
        ),
        # Target location update
        "target_pos_dr": EventTermCfg(
            mode="reset",
            func=dr.body_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    _UNIVERSAL_ENTITY_NAME, body_names=["body_target_.*"]
                ),
                # "ranges": {".*knee.*": (0.5, 1.5), ".*hip.*": (0.8, 1.2)},
                "ranges": {
                    f"body_target_{i}": {
                        xyz: (
                            targets[i].position[0][xyz] + target_origin_rel[xyz],
                            targets[i].position[1][xyz] + target_origin_rel[xyz],
                        )
                        for xyz in range(3)
                    }
                    for i in range(num_targets)
                },
                "operation": "abs",
                # "ranges": {f"body_target_{i}": {xyz: (0, 0) for xyz in range(3)} for i in range(num_targets)},
            },
        ),
        # Target size update
        "target_size_dr": EventTermCfg(
            mode="reset",
            func=dr.geom_size,
            params={
                "asset_cfg": SceneEntityCfg(
                    _UNIVERSAL_ENTITY_NAME, geom_names=["geom_target_.*"]
                ),
                # "axes": [0],
                "ranges": {
                    f"geom_target_{i}": {
                        xyz: (targets[i].size[0][xyz], targets[i].size[1][xyz])
                        for xyz in range(3)
                    }
                    for i in range(num_targets)
                },
                "operation": "abs",
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp_terminations.time_out,
            time_out=True,
        ),
        "episode_success": TerminationTermCfg(
            func=_phase_successfully_completed,  # SequentialTaskSuccessTermination
            params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG, "phase_id": num_targets - 1},
        ),
    }

    metrics = {
        **{
            f"phase_{i}_success": MetricsTermCfg(
                func=_phase_successfully_completed,
                params={"asset_cfg": _OBS_PROPRIO_ENTITY_CFG, "phase_id": i},
            )
            for i in range(num_targets)
        },
        "inside_target": MetricsTermCfg(
            func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].inside_target[:, 0]
        ),
        # "button_touch_force": MetricsTermCfg(func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].button_touch_force[:, 0]),
        # "distance_to_next_target": MetricsTermCfg(func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].distance_to_target[:, 0]),
        "total_initial_distance": MetricsTermCfg(
            func=lambda env: env.scene[
                _UNIVERSAL_ENTITY_NAME
            ].between_target_distances.sum(dim=-1)
        ),
        "target_size": MetricsTermCfg(
            func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].target_size[:, 0]
        ),
        "current_target_id": MetricsTermCfg(
            func=lambda env: env.scene[_UNIVERSAL_ENTITY_NAME].current_target_id[:, 0]
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        decimation=int(round(config.env.ctrl_dt / config.env.sim_dt)),
        episode_length_s=config.env.task_config.max_duration,
        observations=observations,
        actions=actions,
        rewards=rewards,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=config.env.sim_dt),
        ),
        events=events,
        terminations=terminations,
        metrics=metrics,
    )


def _universal_ppo_runner_cfg(vision_enabled: bool) -> RslRlOnPolicyRunnerCfg:
    """Minimal PPO runner config for benchmarking."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(
                128,
                128,
            ),  ## TODO: match architecture with config file/default from MyoInteract
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
            cnn_cfg={
                "output_channels": [16, 32],
                "kernel_size": [5, 3],
                "stride": [2, 2],
                "padding": "zeros",
                "activation": "elu",
                "max_pool": False,
                "global_pool": "none",
                "spatial_softmax": True,
                "spatial_softmax_temperature": 1.0,
            }
            if vision_enabled
            else None,
            class_name="mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
            if vision_enabled
            else "MLPModel",
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256),
            activation="elu",
            obs_normalization=False,
            cnn_cfg={
                "output_channels": [16, 32],
                "kernel_size": [5, 3],
                "stride": [2, 2],
                "padding": "zeros",
                "activation": "elu",
                "max_pool": False,
                "global_pool": "none",
                "spatial_softmax": True,
                "spatial_softmax_temperature": 1.0,
            }
            if vision_enabled
            else None,
            class_name="mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
            if vision_enabled
            else "MLPModel",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="myo_universal",
        save_interval=150,
        num_steps_per_env=24 if vision_enabled else 24,
        max_iterations=1000,  # 750,  #3000
        obs_groups={
            "actor": ("proprioception", "vision_mono")
            if vision_enabled
            else ("proprioception",),
            "critic": ("proprioception", "vision_mono")
            if vision_enabled
            else ("proprioception",),
        },
    )


def register_mjlab_myouser_task(config: Config) -> None:
    """Register MyoUser env ids with mjlab.tasks.registry. Idempotent."""

    # --- Universal Task ---
    xml_file = _get_myouser_root() / config.env.model_path
    if xml_file.exists():
        # try:
        env_cfg = _make_universal_env_cfg(config, play=False)
        play_cfg = _make_universal_env_cfg(config, play=True)
        rl_cfg = _universal_ppo_runner_cfg(config.env.vision.enabled)
        register_mjlab_task(
            task_id="myoUserUniversal-v0",
            env_cfg=env_cfg,
            play_env_cfg=play_cfg,
            rl_cfg=rl_cfg,
            runner_cls=None,
        )
