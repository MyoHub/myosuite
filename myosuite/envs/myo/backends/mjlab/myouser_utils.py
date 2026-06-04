# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import mujoco
import numpy as np

from omegaconf import MISSING
from dataclasses import dataclass, field


@dataclass
class IndividualTargetConfig:
    name: str = MISSING
    rgb: list[float] = MISSING


@dataclass
class PointingTarget(IndividualTargetConfig):
    # penetrable: bool = False
    name: str = "pointing_target"
    # Position can either be a 3d vector or a 2 x list of 3d vectors specifying the min and max values for each dimension
    position: list[list[float]] = field(
        default_factory=lambda: [[0.225, -0.1, -0.3], [0.35, 0.1, 0.3]]
    )
    shape: str = "sphere"
    # Size can either be a single value or a list of 2 values specifying the min and max values
    size: list[float] = field(default_factory=lambda: [0.05, 0.15])
    # Any rewards received when inside the target
    reward_incentive: float = 0.0
    completion_bonus: float = 0.0
    dwell_duration: float = 0.25
    rgb: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])


@dataclass
class ButtonTarget(IndividualTargetConfig):
    position: list[list[float]] = MISSING
    name: str = "button_target"
    size: list[list[float]] = field(
        default_factory=lambda: [[0.025, 0.025, 0.01], [0.025, 0.025, 0.01]]
    )
    site_pos: list[float] = field(default_factory=lambda: [0, 0, 0.01])
    geom_margin: float = 0.001
    completion_bonus: float = 0.0
    min_touch_force: float = 1.0
    rgb: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    euler: list[float] = field(default_factory=lambda: [0, -0.79, 0])


def generate_target(
    target_pos_range: np.ndarray,
    target_radius_range: np.ndarray,
    target_coordinates_origin: np.ndarray,
    prev_pos: np.ndarray = None,
    min_distance: float = 0.1,
):
    target_pos = generate_target_pos(
        target_pos_range,
        target_coordinates_origin,
        prev_pos=prev_pos,
        min_distance=min_distance,
    )
    target_size = generate_target_size(target_radius_range)
    return target_pos, target_size


def generate_target_pos(
    target_pos_range,
    target_coordinates_origin,
    prev_pos: np.ndarray = None,
    min_distance: float = 0.1,
):
    if prev_pos is None:
        prev_pos = np.array([-999.0, -999.0, -999.0])
    new_pos = prev_pos.copy()
    while _unsatisfied_dist_constr(new_pos, prev_pos, min_distance=min_distance):
        prev_pos = new_pos
        new_pos = _target_pos_sampler(target_pos_range, target_coordinates_origin)
    return new_pos


def _unsatisfied_dist_constr(new_pos, prev_pos, min_distance=0.1):
    # distance = np.linalg.norm(
    #     new_pos - prev_pos, axis=-1
    # )
    # return distance < min_distance
    distance = np.abs(new_pos - prev_pos)
    return np.any(distance < min_distance)


def _target_pos_sampler(target_pos_range, target_coordinates_origin):
    sampled_pos = (
        np.random.rand(3) * (target_pos_range[1] - target_pos_range[0])
        + target_pos_range[0]
    )
    sampled_pos = target_coordinates_origin + sampled_pos
    return sampled_pos


def generate_target_size(target_radius_range):
    target_size = (
        np.random.rand(3) * (target_radius_range[1] - target_radius_range[0])
        + target_radius_range[0]
    )
    return target_size


def add_sphere_to_spec(
    spec: mujoco.MjSpec,
    target_cfg: PointingTarget,
    target_id: int,
    target_coordinates_origin: np.ndarray = np.zeros(3),
):
    ## TODO: unify usage of torch vs numpy
    target_body_name = f"body_target_{target_id}"
    target_geom_name = f"geom_target_{target_id}"
    target_site_name = f"site_target_{target_id}"
    target_sensor_name = f"sensor_target_{target_id}"

    worldbody = spec.worldbody
    target_pos, target_size = generate_target(
        np.array(target_cfg.position),
        np.array(target_cfg.size),
        target_coordinates_origin,
    )
    target_body = worldbody.add_body(name=target_body_name, pos=target_pos)
    rgba = np.ones(4)
    rgba[:3] = target_cfg.rgb
    target_size = np.ones(3) * target_size  ##TODO: deprecated -- remove
    target_body.add_geom(
        name=target_geom_name, pos=np.zeros(3), size=target_size, rgba=rgba
    )
    # print(f"Added target {target_geom_name} to spec")

    #### only required for consistency with add_button_to_spec
    target_body.add_site(
        name=target_site_name,
        type=mujoco._enums.mjtGeom(6),
        pos=target_cfg.site_pos,
        rgba=rgba,
        size=0.001 * np.ones(3),
    )
    spec.add_sensor(
        name=target_sensor_name,
        type=mujoco.mjtSensor.mjSENS_TOUCH,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=target_site_name,
    )

    return spec


def add_button_to_spec(
    spec: mujoco.MjSpec,
    target_cfg: ButtonTarget,
    target_id: int,
    target_coordinates_origin: np.ndarray = np.zeros(3),
):
    target_body_name = f"body_target_{target_id}"
    target_geom_name = f"geom_target_{target_id}"
    target_site_name = f"site_target_{target_id}"
    target_sensor_name = f"sensor_target_{target_id}"

    worldbody = spec.worldbody
    target_pos, target_size = generate_target(
        np.array(target_cfg.position),
        np.array(target_cfg.size),
        target_coordinates_origin,
    )
    target_body = worldbody.add_body(
        name=target_body_name, pos=target_pos, euler=target_cfg.euler
    )
    rgba = np.ones(4)
    rgba[:3] = target_cfg.rgb
    target_body.add_geom(
        name=target_geom_name,
        type=mujoco._enums.mjtGeom(6),
        size=target_size,
        margin=target_cfg.geom_margin,
        rgba=rgba,
        contype=1,
        conaffinity=1,
    )
    target_body.add_site(
        name=target_site_name,
        type=mujoco._enums.mjtGeom(6),
        pos=target_cfg.site_pos,
        rgba=rgba,
        size=target_size,
    )
    # print(f"Added target {target_geom_name} to spec")
    # Add touch sensor for the button
    spec.add_sensor(
        name=target_sensor_name,
        type=mujoco.mjtSensor.mjSENS_TOUCH,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=target_site_name,
    )
    # print(f"Added sensor {target_sensor_name} to spec")
    return spec
