#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Robot interface bridging MuJoCo simulation and optional hardware backends.

This module provides :class:`Robot`, a convenience wrapper used by legacy CPU envs
to:

- Build/manage a MuJoCo ``mj_model``/``mj_data`` pair.
- Read sensors (from sim or from hardware) and optionally propagate readings back
  into a simulation state.
- Apply controls with basic feasibility checks.

The API is intentionally kept stable because it is referenced by multiple envs.
"""

# Pylint in this repo is intentionally strict, but MuJoCo's Python bindings expose
# many symbols dynamically and several hardware backends are optional extras.
# Disable the specific false-positive categories for this legacy bridge module.
# pylint: disable=import-error,no-member,invalid-sequence-index

from __future__ import annotations

import ast
import logging
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.physics.quat_math import quat2euler


logger = logging.getLogger(__name__)

_ROBOT_VIZ = False

# Optional viz profiling: step deltas when ``_ROBOT_VIZ`` is enabled.
timing_SRV: list[float] = []
timing_SRV_t0: float = 0.0

# TODO ===========================================
# rename robot_config something more meaningful
# support loading multiple config files
# seperate ROBOT_VIZ as its own class
# remap_space() needs rigerous testing
# Support for sensors that provide multiple reading values. Sensor indexing might not directly follow the sensor's list index in this case. This support will potentilly allow us to also list cams as sensors
# Support for non uniform noise in sensor readings
# Support for noisy actions + separate noise_scale for sensor and actuator
# rename pos/vel to act/delta_act

# NOTE/ GOOD PRACTICES ===========================
# nq should be nv
# Order of sensors and actuators in config should follow XML order


class Robot:
    """A unified viewpoint of a robot in simulation and (optionally) hardware."""

    # Cached a persistent connection to the robot that is shared for the application's lifetime.
    robot_config: dict[str, dict[str, Any]] | None = None

    def __init__(
        self,
        robot_name: str = "default_robot",
        model_path: str | Path | None = None,  # model file to create sim
        mj_model: Any | None = None,  # pass mj_model directly (MuJoCo model)
        config_path: str | Path | None = None,  # config defining sensors/groups
        act_mode: str = "pos",  # pos / vel
        is_hardware: bool = None,  # use hardware
        sensor_cache_maxsize: int = 5,  # cache size for sensors
        noise_scale: float = 0.0,  # scale for sensor noise
        random_generator: np.random.Generator
        | np.random.RandomState
        | Any
        | None = None,
        **kwargs,
    ):
        if kwargs != {}:
            logger.warning("Unused kwargs found: %s", kwargs)
        self.name = robot_name + ("(hdr)" if is_hardware else "(sim)")
        if act_mode not in {"pos", "vel"}:
            raise ValueError(f"Unknown act_mode: {act_mode}. Expected 'pos' or 'vel'.")
        self._act_mode: str = act_mode
        self.is_hardware = bool(is_hardware)
        self._sensor_cache_maxsize = sensor_cache_maxsize
        self._noise_scale = noise_scale
        self.np_random = np.random if random_generator is None else random_generator

        # sensor cache
        self._sensor_cache = deque([], maxlen=self._sensor_cache_maxsize)

        # create robot sim
        if mj_model is None:
            if model_path is None:
                raise ValueError("Either 'mj_model' or 'model_path' must be provided.")
            model_path = str(Path(model_path))
            # (creates new robot everytime to facilitate parallelization)
            logger.info("Preparing robot-sim from %s", model_path)
            self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        else:
            self.mj_model = mj_model
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Configure the robot
        if self.robot_config is None:
            logger.info("Configuring a new session for %s", self.name)
            self.robot_config = self.configure_robot(self.mj_model, config_path)
            if _ROBOT_VIZ:
                self.configure_robot_viz(self.robot_config)
            # start the robot
            if self.is_hardware is True:
                logger.info("Initializing robot: %s", self.name)
                self.robot_config = self.hardware_init(self.robot_config)
        else:
            logger.info("Reusing a previous session of %s", self.name)

        # check robot health
        if self.is_hardware is True:
            self.hardware_okay(self.robot_config)

            # disable all collisions
            self.mj_model.geom_conaffinity[:] = 0
            self.mj_model.geom_contype[:] = 0

        # Robot's time
        self.time_start = time.time()
        self.time_wall = (
            time.time() - self.time_start
        )  # Wall time (used for realtime factors) for both sim and hardware

        # refresh the sensor cache
        self._sensor_cache_refresh()

    # Check if all hardware components are okay
    def hardware_okay(self, robot_config: dict[str, dict[str, Any]]) -> None:
        for name, device in robot_config.items():
            if not device["robot"].okay():
                logger.error("Please check device %s", name)

    # initialize all hardware components
    def hardware_init(
        self, robot_config: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        # initalize
        for name, device in robot_config.items():
            logger.info("Initializing device: %s", name)
            if device["interface"]["type"] == "dynamixel":
                # initialize dynamixels
                from dynamixel_py import dxl

                ids = np.unique(device["sensor_ids"] + device["actuator_ids"]).tolist()
                device["robot"] = dxl(
                    motor_id=ids,
                    motor_type=device["interface"]["motor_type"],
                    devicename=device["interface"]["name"],
                )

                # from .hardware_dynamixel import Dynamixels
                # motor_ids = np.unique([device['sensor_ids'] + device['actuator_ids']]).tolist()
                # device['robot'] = Dynamixels(name=name, motor_ids=motor_ids, motor_type=device['interface']['motor_type'], devicename= device['interface']['name'])

            elif device["interface"]["type"] == "optitrack":
                from .hardware_optitrack import OptiTrack

                device["robot"] = OptiTrack(
                    ip=device["interface"]["client_name"],
                    port=device["interface"]["port"],
                    packet_size=device["interface"]["packet_size"],
                )

            elif device["interface"]["type"] == "franka":
                from .hardware_franka import FrankaArm

                device["robot"] = FrankaArm(name=name, **device["interface"])

            elif device["interface"]["type"] == "realsense":
                try:
                    from .hardware_realsense import RealSense

                    device["robot"] = RealSense(name=name, **device["interface"])
                except ImportError:
                    from .hardware_realsense_single import RealsenseAPI

                    device["robot"] = RealsenseAPI(**device["interface"])

            elif device["interface"]["type"] == "robotiq":
                from .hardware_robotiq import Robotiq

                device["robot"] = Robotiq(name=name, **device["interface"])

            else:
                raise NotImplementedError(
                    f"Interface type not found: {device['interface']['type']}"
                )

        # start all hardware
        for name, device in robot_config.items():
            # Dynamixels
            if device["interface"]["type"] == "dynamixel":
                device["robot"].open_port()

                # set actuator mode
                for actuator in device["actuator"]:
                    device["robot"].set_operation_mode(
                        motor_id=[actuator["hdr_id"]], mode=actuator["mode"]
                    )

                # engage motors
                device["robot"].engage_motor(
                    motor_id=device["actuator_ids"], enable=True
                )

            # Other devices
            elif device["interface"]["type"] in [
                "optitrack",
                "franka",
                "realsense",
                "robotiq",
            ]:
                device["robot"].connect()

            else:
                raise NotImplementedError(
                    f"Interface type not found: {device['interface']['type']}"
                )

        return robot_config

    # get hardware sensors
    def hardware_get_sensors(self) -> dict[str, Any]:
        current_sensor_value: dict[str, Any] = {}
        current_sensor_value["time"] = time.time() - self.time_start
        for name, device in self.robot_config.items():
            if "sensor" in device.keys() and len(device["sensor"]) > 0:
                # get sensors
                if device["interface"]["type"] == "dynamixel":
                    # TODO: choose between pos, vel, or posvel
                    current_sensor_value[name] = device["robot"].get_pos(
                        device["sensor_ids"]
                    )
                    current_sensor_value[name + "_vel"] = device["robot"].get_vel(
                        device["sensor_ids"]
                    )

                elif device["interface"]["type"] == "optitrack":
                    data = device["robot"].get_sensors()
                    c, b, a = quat2euler(data["quat"])
                    rx = np.pi - a
                    rx = (rx - 2 * np.pi) if rx > np.pi else rx
                    ry = b
                    rz = -c
                    # print("Pos:", x, y, z)
                    # print("Rotations:", rx, ry, rz)
                    current_sensor_value[name] = np.concatenate(
                        [data["pos"], np.array([rx, ry, rz])]
                    )
                    # current_sensor_value[name] = np.array([x, y, z, 0, 0, 0])
                    # current_sensor_value[name] = np.array([x, y, z, -(a+np.pi/2), -c, -b])

                elif device["interface"]["type"] == "franka":
                    sensors = device["robot"].get_sensors()
                    current_sensor_value[name] = np.concatenate(
                        [sensors["joint_pos"], sensors["joint_vel"]]
                    )

                elif device["interface"]["type"] == "robotiq":
                    sensors = device["robot"].get_sensors()
                    current_sensor_value[name] = sensors

                else:
                    raise NotImplementedError(
                        f"Interface type not found: {device['interface']['type']}"
                    )

                # calibrate sensors
                for sensor_idx, sensor in enumerate(device["sensor"]):
                    current_sensor_value[name][sensor_idx] = (
                        current_sensor_value[name][sensor_idx] * sensor["scale"]
                        + sensor["offset"]
                    )
                device["sensor_data"] = current_sensor_value[name]
                device["sensor_time"] = current_sensor_value["time"]
        return current_sensor_value

    # apply controls to hardware
    def hardware_apply_controls(
        self, control: np.ndarray, is_reset: bool = False
    ) -> None:
        for name, device in self.robot_config.items():
            if "actuator" in device.keys() and len(device["actuator"]) > 0:
                if device["interface"]["type"] == "dynamixel":
                    # group as per mode
                    pos_ctrl = []
                    pos_ids = []
                    pwm_ctrl = []
                    pwm_ids = []
                    for actuator in device["actuator"]:
                        # calibrate
                        calib_ctrl = (
                            control[actuator["sim_id"]] * actuator["scale"]
                            + actuator["offset"]
                        )
                        if actuator["mode"] == "Position":
                            pos_ids.append(actuator["hdr_id"])
                            pos_ctrl.append(calib_ctrl)
                        elif actuator["mode"] == "PWM":
                            pwm_ids.append(actuator["hdr_id"])
                            pwm_ctrl.append(calib_ctrl)
                        else:
                            raise NotImplementedError(
                                f"Unsupported dynamixel mode: {actuator['mode']}"
                            )
                    # send controls
                    if pos_ids:
                        device["robot"].set_des_pos(pos_ids, pos_ctrl)
                    if pwm_ids:
                        device["robot"].set_des_pwm(pwm_ids, pwm_ctrl)

                elif device["interface"]["type"] == "franka":
                    franka_des_pos = []
                    for actuator in device["actuator"]:
                        # calibrate
                        franka_des_pos.append(
                            control[actuator["sim_id"]] * actuator["scale"]
                            + actuator["offset"]
                        )
                    if is_reset:
                        device["robot"].reset(franka_des_pos)
                    else:
                        device["robot"].apply_commands(franka_des_pos)

                elif device["interface"]["type"] == "robotiq":
                    robotiq_des_pos = []
                    for actuator in device["actuator"]:
                        # calibrate
                        robotiq_des_pos.append(
                            control[actuator["sim_id"]] * actuator["scale"]
                            + actuator["offset"]
                        )
                    if is_reset:
                        device["robot"].reset(robotiq_des_pos[0])
                    else:
                        device["robot"].apply_commands(robotiq_des_pos[0])
                else:
                    raise NotImplementedError(
                        f"Interface type not found: {device['interface']['type']}"
                    )

    # close hardware
    def hardware_close(self) -> bool:
        status: bool = True
        for name, device in self.robot_config.items():
            if device["interface"]["type"] == "dynamixel":
                if device["robot"]:
                    logging.getLogger(__name__).info("Closing dynamixel connection")
                    ids = np.unique(
                        device["sensor_ids"] + device["actuator_ids"]
                    ).tolist()
                    status = device["robot"].close(ids)
                    if status is True:
                        device["robot"] = None
            elif device["interface"]["type"] in [
                "optitrack",
                "franka",
                "realsense",
                "robotiq",
            ]:
                if device["robot"]:
                    logging.getLogger(__name__).info(
                        "Closing %s connection", device["interface"]["type"]
                    )
                    status = device["robot"].close()
                    if status is True:
                        device["robot"] = None
            else:
                raise NotImplementedError(
                    f"Interface type not found: {device['interface']['type']}"
                )

        return status

    # configure robot
    def configure_robot(
        self, mj_model: Any, config_path: str | Path | None
    ) -> dict[str, dict[str, Any]]:
        """Load and compile a robot configuration against a MuJoCo model.

        Args:
            mj_model: MuJoCo model used to resolve sensor/actuator IDs.
            config_path: Path to a Python-literal configuration dict.

        Returns:
            A fully-resolved robot configuration dict.

        Raises:
            ValueError: If the config cannot be parsed, or if it contains
                unsupported sensor/actuator types.
        """
        if config_path is None:
            return {
                "default_robot": {
                    "sensor": ["qpos", "qvel", "act"],
                    "actuator": "actuator",
                }
            }

        config_path = Path(config_path)
        logger.info("Reading robot-configurations from %s", config_path)
        try:
            robot_config = ast.literal_eval(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Failed to read/parse robot config at {config_path}"
            ) from exc

        for name, device in robot_config.items():
            logger.debug("Configuring component %s", name)

            # configure device sensors
            device["sensor_ids"] = []
            device["sensor_names"] = []
            for sensor in device["sensor"]:
                device["sensor_names"].append(sensor["name"])  # list of all ids
                device["sensor_ids"].append(sensor["hdr_id"])  # list of all ids
                sensor["sim_id"] = mj_model.sensor(sensor["name"]).id
                sensor_type = mj_model.sensor_type[sensor["sim_id"]]
                sensor_objid = mj_model.sensor_objid[sensor["sim_id"]]
                if (
                    sensor_type == mujoco.mjtSensor.mjSENS_JOINTPOS
                ):  # mjSENS_JOINTPOS,// scalar joint position (hinge and slide only)
                    sensor["data_type"] = "qpos"
                    sensor["data_id"] = mj_model.jnt_qposadr[sensor_objid]
                elif (
                    sensor_type == mujoco.mjtSensor.mjSENS_JOINTVEL
                ):  # mjSENS_JOINTVEL,// scalar joint position (hinge and slide only)
                    sensor["data_type"] = "qvel"
                    sensor["data_id"] = mj_model.jnt_dofadr[sensor_objid]
                else:
                    raise ValueError(
                        f"Sensor {sensor['name']} has unsupported sensor_type: {sensor_type}"
                    )

            # configure device actuators
            device["actuator_ids"] = []
            device["actuator_names"] = []
            for actuator in device["actuator"]:
                device["actuator_names"].append(actuator["name"])  # list of all ids
                device["actuator_ids"].append(actuator["hdr_id"])  # list of all ids
                actuator["sim_id"] = mj_model.actuator(actuator["name"]).id
                actuator_trntype = mj_model.actuator_trntype[actuator["sim_id"]]
                actuator_trnid = mj_model.actuator_trnid[actuator["sim_id"], 0]
                if actuator_trntype == 0:  # mjTRN_JOINT // force on joint
                    actuator["data_type"] = "qpos"
                    actuator["data_id"] = mj_model.jnt_dofadr[actuator_trnid]
                else:
                    raise ValueError(
                        f"Actuator {actuator['name']} has unsupported transmission_type: {actuator_trntype}"
                    )
        return robot_config

    # refresh the sensor cache
    def _sensor_cache_refresh(self) -> None:
        for _ in range(self._sensor_cache_maxsize):
            self.get_sensors()

    # get past sensor
    def get_sensor_from_cache(self, index: int = -1) -> dict[str, Any]:
        assert (index >= 0 and index < self._sensor_cache_maxsize) or (
            index < 0 and index >= -self._sensor_cache_maxsize
        ), "cache index out of bound. (cache size is %2d)" % self._sensor_cache_maxsize
        return self._sensor_cache[index]

    # get sensor data and update robot time accordingly
    def get_sensors(
        self,
        noise_scale: float | None = None,
        random_generator: Any | None = None,
    ) -> dict[str, Any]:
        """Get sensor readings and update internal time/cache."""
        current_sen: dict[str, Any] = {}
        noise_scale = self._noise_scale if noise_scale is None else noise_scale
        if random_generator is not None:
            self.np_random = random_generator

        if self.is_hardware:
            # record sensor*device['scale']+device['offset']
            current_sen = self.hardware_get_sensors()
            # update the sim as per the hardware observations
            self.sensor2sim(current_sen, self.mj_model, self.mj_data)
        else:
            current_sen["time"] = self.mj_data.time  # data time stamp
            for name, device in self.robot_config.items():
                if name == "default_robot":
                    sen = {}
                    sen["qpos"] = self.mj_data.qpos.copy()
                    sen["qvel"] = self.mj_data.qvel.copy()
                    sen["act"] = (
                        self.mj_data.act.copy() if self.mj_model.na > 0 else None
                    )
                    current_sen[name] = sen
                else:
                    sen = []
                    for sensor in device["sensor"]:
                        s = self.mj_data.sensordata[sensor["sim_id"]]
                        # ensure range
                        s = np.clip(s, sensor["range"][0], sensor["range"][1])
                        # add noise
                        if noise_scale != 0:
                            s += (
                                noise_scale
                                * sensor["noise"]
                                * self.np_random.uniform(low=-1.0, high=1.0)
                            )
                        sen.append(s)
                    current_sen[name] = np.array(sen)

                # create sensor reading
                device["sensor_data"] = current_sen[name]
                device["sensor_time"] = current_sen["time"]

            # VIK???: Propagating sensors back to sim can create trouble with contact stability in presence of noise
            # self.sensor2sim(current_sen, self.sim)

        # cache sensors
        self._sensor_cache.append(current_sen)

        # Update time
        self.time_wall = time.time() - self.time_start

        return current_sen

    # get sensor data and update robot time accordingly
    def get_visual_sensors(
        self, height: int, width: int, cameras: list, device_id: int, renderer: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.is_hardware:
            imgs = np.zeros((len(cameras), height, width, 3), dtype=np.uint8)
            depths = np.zeros((len(cameras), height, width), dtype=np.uint16)

            current_sensor_value = {}
            current_sensor_value["time"] = time.time() - self.time_start

            for ind, cam_name in enumerate(cameras):
                assert (
                    cam_name in self.robot_config.keys()
                ), f"{cam_name} camera not found"
                device = self.robot_config[cam_name]
                assert (
                    device["interface"]["type"] == "realsense"
                ), f"Check interface type for {cam_name}"
                data = device["robot"].get_sensors()
                data_height = data["rgb"].shape[0]
                assert (
                    data_height == height
                ), "Incorrect image height: required:{}, found:{}".format(
                    height, data_height
                )
                data_width = data["rgb"].shape[1]
                assert (
                    data_width == width
                ), "Incorrect image width: required:{}, found:{}".format(
                    width, data_width
                )
                current_sensor_value[cam_name] = data

                # calibrate sensors
                for cam in device["cam"]:
                    current_sensor_value[cam_name][cam["hdr_id"]] = (
                        current_sensor_value[cam_name][cam["hdr_id"]] * cam["scale"]
                        + cam["offset"]
                    )
                device["sensor_data"] = current_sensor_value[cam_name]
                device["sensor_time"] = current_sensor_value["time"]
                imgs[ind, :, :, :] = current_sensor_value[cam_name]["rgb"]
                depths[ind, :, :] = current_sensor_value[cam_name]["d"][
                    :, :, 0
                ]  # assumes single channel depth

        else:
            imgs = np.zeros((len(cameras), height, width, 3), dtype=np.uint8)
            depths = np.zeros((len(cameras), height, width))
            for ind, cam in enumerate(cameras):
                # img, depth = sim.render(width=width, height=height, depth=True, mode='offscreen', camera_name=cam, device_id=device_id)
                img, depth = renderer.render_offscreen(
                    width=width,
                    height=height,
                    depth=True,
                    camera_id=cam,
                    device_id=device_id,
                )
                # img = img[::-1, :, :] # Image has to be flipped
                imgs[ind, :, :, :] = img
                depths[ind, :, :] = depth

        return imgs, depths

    # Propagate sensor values back through the simulation.
    def sensor2sim(self, sensor: dict[str, Any], mj_model: Any, mj_data: Any) -> None:
        """Propagate sensor values back into the simulation.

        This operation is water-tight only where the system is fully observable
        (including velocities).

        Args:
            sensor: Sensor data dict.
            mj_model: MuJoCo model.
            mj_data: MuJoCo data to update.

        Note:
            Can be used to feed hardware sensors to robot-sim, or noisy sim sensors
            back into robot-sim (be careful: sim may be unstable afterward).
        """
        if not self.is_hardware and (self._noise_scale != 0):
            logging.getLogger(__name__).warning(
                "Propagating noisy sensors back to sim can destabilize simulation."
            )

        mj_data.time = sensor["time"]
        for name, device in self.robot_config.items():
            if name == "default_robot":
                mj_data.qpos[:] = device["sensor_data"]["qpos"]
                mj_data.qvel[:] = device["sensor_data"]["qvel"]
                if self.mj_model.na > 0:
                    mj_data.act[:] = device["sensor_data"]["act"]
            else:
                for s_id, s_val in enumerate(device["sensor"]):
                    # prompt(getattr(mj_data, s_val["data_type"])[s_val["data_id"]], sensor[name][s_id])
                    data = getattr(mj_data, s_val["data_type"])
                    data[s_val["data_id"]] = sensor[name][s_id]
        mujoco.mj_forward(mj_model, mj_data)

    # synchronize states between two sims
    def sync_sims(
        self,
        source_mj_model: Any,
        source_mj_data: Any,
        destination_mj_model: Any,
        destination_mj_data: Any,
        model: bool = True,
        data: bool = True,
    ) -> None:
        destination_mj_data.time = source_mj_data.time
        if data:
            destination_mj_data.qpos[:] = source_mj_data.qpos[:].copy()
            destination_mj_data.qvel[:] = source_mj_data.qvel[:].copy()
            if destination_mj_model.na > 0:
                destination_mj_data.act[:] = source_mj_data.act[:].copy()
            if destination_mj_model.nmocap > 0:
                destination_mj_data.mocap_pos[:] = source_mj_data.mocap_pos.copy()
                destination_mj_data.mocap_quat[:] = source_mj_data.mocap_quat.copy()

        if model:
            if destination_mj_model.nsite > 0:
                destination_mj_model.site_pos[:] = source_mj_model.site_pos[:].copy()
                destination_mj_model.site_quat[:] = source_mj_model.site_quat[:].copy()
            if destination_mj_model.nbody > 0:
                destination_mj_model.body_pos[:] = source_mj_model.body_pos[:].copy()
                destination_mj_model.body_quat[:] = source_mj_model.body_quat[:].copy()

        mujoco.mj_forward(destination_mj_model, destination_mj_data)

    # remap sensor/actuators spaces: sim<>hardware, TODO: Needs rigerous testing
    def remap_space(
        self,
        input_vec: np.ndarray,
        input_type: str,
        input_space: str,
        output_space: str,
    ) -> np.ndarray:
        assert input_type in ["sensor", "actuator"], "check input type"
        assert input_space in ["sim", "hdr"], "check input space"
        assert output_space in ["sim", "hdr"], "check output space"
        assert (
            input_space != output_space
        ), "Check: Input and output spaces are the same"

        input_space = input_space + "_id"
        output_space = output_space + "_id"
        output_vec = input_vec.copy()

        # sim => hdr
        if input_space == "sim_id" and output_space == "hdr_id":
            output_space = "data_id"  # No physical/logical ID; use model data index.
            for name, device in self.robot_config.items():
                if (
                    input_type == "actuator"
                    and "actuator" in device.keys()
                    and len(device["actuator"]) > 0
                ):
                    for _actuator_idx, actuator in enumerate(device["actuator"]):
                        output_vec[actuator[output_space]] = (
                            input_vec[actuator[input_space]] * actuator["scale"]
                            + actuator["offset"]
                        )

                if (
                    input_type == "sensor"
                    and "sensor" in device.keys()
                    and len(device["sensor"]) > 0
                ):
                    for _sensor_idx, sensor in enumerate(device["sensor"]):
                        output_vec[sensor[output_space]] = (
                            input_vec[sensor[input_space]] - sensor["offset"]
                        ) / sensor["scale"]

        # hdr => sim
        if input_space == "hdr_id" and output_space == "sim_id":
            input_space = "data_id"  # No physical/logical ID; use model data index.
            for name, device in self.robot_config.items():
                if (
                    input_type == "actuator"
                    and "actuator" in device.keys()
                    and len(device["actuator"]) > 0
                ):
                    for _actuator_idx, actuator in enumerate(device["actuator"]):
                        output_vec[actuator[output_space]] = (
                            input_vec[actuator[input_space]] - actuator["offset"]
                        ) / actuator["scale"]

                if (
                    input_type == "sensor"
                    and "sensor" in device.keys()
                    and len(device["sensor"]) > 0
                ):
                    for _sensor_idx, sensor in enumerate(device["sensor"]):
                        output_vec[sensor[output_space]] = (
                            input_vec[sensor[input_space]] * sensor["scale"]
                            + sensor["offset"]
                        )
        return output_vec

    # Normalize actions from absolute space to unit space
    def normalize_actions(
        self,
        controls: np.ndarray,
        out_space: str = "sim",
        unnormalize: bool = False,
    ) -> np.ndarray:
        """
        Normalize actions from absolute space to unit space
        Recover actions from unit space to absolute space; if unnormalize==True
        in_space for controls has to be 'sim'
        """
        act_id = -1
        controls_out = controls.copy()
        for name, device in self.robot_config.items():
            if name == "default_robot":
                if self._act_mode == "pos":
                    act_mid = np.mean(self.mj_model.actuator_ctrlrange, axis=-1)
                    act_rng = (
                        self.mj_model.actuator_ctrlrange[:, 1]
                        - self.mj_model.actuator_ctrlrange[:, 0]
                    ) / 2.0
                    controls_out = (
                        controls * act_rng + act_mid
                        if unnormalize
                        else (controls - act_mid) / act_rng
                    )
                else:
                    raise TypeError("only pos act supported")
            else:
                for actuator in device["actuator"]:
                    act_id += 1
                    in_id = actuator["sim_id"]
                    # output ordering is as per the config order for hdr
                    out_id = actuator["sim_id"] if out_space == "sim" else act_id

                    if self._act_mode == "pos":
                        act_mid = (
                            actuator["pos_range"][1] + actuator["pos_range"][0]
                        ) / 2.0
                        act_rng = (
                            actuator["pos_range"][1] - actuator["pos_range"][0]
                        ) / 2.0
                    elif self._act_mode == "vel":
                        act_mid = (
                            actuator["vel_range"][1] + actuator["vel_range"][0]
                        ) / 2.0
                        act_rng = (
                            actuator["vel_range"][1] - actuator["vel_range"][0]
                        ) / 2.0
                    else:
                        raise TypeError(f"Unknown act mode: {self._act_mode}")

                    # unnormalize/ normalize
                    control = controls[in_id]
                    if unnormalize:
                        control = np.clip(control, -1, 1)
                        control = control * act_rng + act_mid
                    else:
                        control = (control - act_mid) / act_rng
                        control = np.clip(control, -1, 1)

                    # remap to desired space
                    controls_out[out_id] = control
        return controls_out

    # enfoce limits
    def process_actuator(
        self,
        controls: np.ndarray,
        step_duration: float,
        normalized: bool = True,
        position_limits: bool = True,
        velocity_limits: bool = True,
        out_space: str = "sim",
    ) -> np.ndarray:
        """
        Process the actuation demands to
            (1) Remap provided controls to actuation space,
            (2) Enforces hardware position and velocity limits on the controls
        """
        # last_obs = self.get_sensor_from_cache(-1)
        processed_controls = controls.copy()
        act_id = -1
        for name, device in self.robot_config.items():
            if name == "default_robot":
                if self._act_mode == "pos":
                    if normalized:
                        processed_controls = (
                            np.mean(self.mj_model.actuator_ctrlrange, axis=-1)
                            + controls
                            * (
                                self.mj_model.actuator_ctrlrange[:, 1]
                                - self.mj_model.actuator_ctrlrange[:, 0]
                            )
                            / 2.0
                        )
                else:
                    raise TypeError("only pos act supported")
            else:
                for actuator in device["actuator"]:
                    act_id += 1
                    in_id = actuator["sim_id"]
                    # output ordering is as per the config order for hdr
                    out_id = actuator["sim_id"] if out_space == "sim" else act_id

                    control = controls[in_id]
                    if self._act_mode == "pos":
                        # remap to the limits if normalized
                        if normalized:
                            control = (
                                actuator["pos_range"][1] + actuator["pos_range"][0]
                            ) / 2.0 + control * (
                                actuator["pos_range"][1] - actuator["pos_range"][0]
                            ) / 2.0
                        # enforce velocity limits
                        # ALERT: This depends on previous sensor. This is not ideal as it breaks MDP addumptions. Be careful
                        if velocity_limits:
                            last_obs = getattr(self.mj_data, actuator["data_type"])[
                                actuator["data_id"]
                            ]
                            ctrl_desired_vel = (control - last_obs) / step_duration
                            ctrl_feasible_vel = np.clip(
                                ctrl_desired_vel,
                                actuator["vel_range"][0],
                                actuator["vel_range"][1],
                            )
                            control = last_obs + ctrl_feasible_vel * step_duration
                    elif self._act_mode == "vel":
                        # remap to the limits if normalized
                        if normalized:
                            control = (
                                actuator["vel_range"][1] + actuator["vel_range"][0]
                            ) / 2.0 + control * (
                                actuator["vel_range"][1] - actuator["vel_range"][0]
                            ) / 2.0
                        # enforce velocity limits
                        # ALERT: This depends on previous sensor. This is not ideal as it breaks MDP addumptions. Be careful
                        last_obs = getattr(self.mj_data, actuator["data_type"])[
                            actuator["data_id"]
                        ]
                        control = last_obs + control * step_duration
                    else:
                        raise TypeError(f"Unknown act mode: {self._act_mode}")

                    # enforce position limits
                    if position_limits:
                        control = np.clip(
                            control, actuator["pos_range"][0], actuator["pos_range"][1]
                        )

                    # remap to desired space
                    processed_controls[out_id] = control

        return processed_controls

    def _advance(
        self, model: Any, data: Any, n_frames: int = 1, render: Any | None = None
    ) -> None:
        # step the robot one step forward in time
        # compatibility with old code based on dm_control
        # TODO: Add rendering support
        for _ in range(n_frames):
            mujoco.mj_step(model, data)

    # step the robot one step forward in time
    def step(
        self,
        ctrl_desired: np.ndarray,
        step_duration: float,
        ctrl_normalized: bool = True,
        realTimeSim: bool = False,
        render_cbk: Any | None = None,
    ) -> np.ndarray:
        """Apply controls and step forward in time.

        Args:
            ctrl_desired: Desired control to be applied (sim_space).
            step_duration: Step duration in seconds.
            ctrl_normalized: If True, ctrl is normalized to [-1, 1].
            realTimeSim: If True, run simulation at real-world speed.
            render_cbk: Optional render callback.
        """

        # pick output space
        robot_type = "hdr" if self.is_hardware else "sim"

        # enforce limits
        ctrl_feasible = self.process_actuator(
            controls=ctrl_desired,
            step_duration=step_duration,
            normalized=ctrl_normalized,
            position_limits=True,
            velocity_limits=True,
            out_space=robot_type,
        )

        # Send controls to the robot
        if self.is_hardware:
            self.hardware_apply_controls(ctrl_feasible)
            if render_cbk:
                render_cbk()
        else:
            n_frames = int(step_duration / self.mj_model.opt.timestep)
            self.mj_data.ctrl[:] = ctrl_feasible
            self._advance(
                self.mj_model, self.mj_data, n_frames, render=(render_cbk is not None)
            )

        # update viz
        if _ROBOT_VIZ:
            for name, device in self.robot_config.items():
                device["controls"] = []
                for actuator in device["actuator"]:
                    device["controls"].append(ctrl_feasible[actuator["sim_id"]])
            self.update_robot_viz(update_sensor=True, update_control=True)

        # synchronize time to maintain step_duration
        if self.is_hardware or realTimeSim:
            time_now = time.time() - self.time_start
            time_left_in_step = step_duration - (time_now - self.time_wall)
            if time_left_in_step > 0.001:
                time.sleep(time_left_in_step)
            elif time_left_in_step < 0.0:
                logger.warning(
                    "Step duration %.4fs, step took %.4fs, time left %.4f",
                    step_duration,
                    time_now - self.time_wall,
                    time_left_in_step,
                )

        if _ROBOT_VIZ:
            global timing_SRV_t0
            timing_SRV_t = time.time()
            timing_SRV.append(timing_SRV_t - timing_SRV_t0)
            timing_SRV_t0 = timing_SRV_t
        return ctrl_feasible

    # Reset the robot
    def reset(
        self,
        reset_pos: np.ndarray,
        reset_vel: np.ndarray,
        blocking: bool = True,
        **_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.info("Resetting %s", self.name)

        # Enforce specs on the request
        #   for actuated dofs => actoator specs
        #   for passive dofs => sensor specs
        feasibe_pos = reset_pos.copy()
        feasibe_vel = reset_vel.copy()
        ctrl_feasible = []
        for name, device in self.robot_config.items():
            if name != "default_robot":
                if len(device["actuator"]) > 0:  # actuated dofs
                    for actuator in device["actuator"]:
                        if actuator["data_type"] == "qpos":
                            feasibe_pos[actuator["data_id"]] = np.clip(
                                reset_pos[actuator["data_id"]],
                                actuator["pos_range"][0],
                                actuator["pos_range"][1],
                            )
                            ctrl_feasible.append(feasibe_pos[actuator["data_id"]])
                else:  # passive dofs
                    for sensor in device["sensor"]:
                        if sensor["data_type"] == "qpos":
                            feasibe_pos[sensor["data_id"]] = np.clip(
                                reset_pos[sensor["data_id"]],
                                sensor["range"][0],
                                sensor["range"][1],
                            )
                        elif sensor["data_type"] == "qvel":
                            feasibe_vel[sensor["data_id"]] = np.clip(
                                reset_vel[sensor["data_id"]],
                                sensor["range"][0],
                                sensor["range"][1],
                            )

        if self.is_hardware:
            t_reset_start = time.time()
            logger.info("Rollout took: %.4f", t_reset_start - self.time_start)
            logger.info("Resetting %s (hardware)", self.name)
            # send request to the actuated dofs
            self.hardware_apply_controls(ctrl_feasible, is_reset=True)

            # engage other reset mechanisms for passive dofs
            # TODO raise NotImplementedError

            if blocking:
                input("press a key to start rollout")
            logger.info("Reset done in %.4fs", time.time() - t_reset_start)
        else:
            # Ideally we should use actuator/ reset mechanism as in the real world
            # but choosing to directly resetting sim for efficiency
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            self.mj_data.qpos[:] = feasibe_pos
            self.mj_data.qvel[:] = feasibe_vel
            mujoco.mj_forward(self.mj_model, self.mj_data)

            if _ROBOT_VIZ:
                input("press a key to start rollout")

        # clear viz
        if _ROBOT_VIZ:
            self.clear_robot_viz(clear_sensor=True, clear_control=True)

        # refresh sensor cache before exiting reset
        self._sensor_cache_refresh()

        # restart the robot clock
        self.time_start = time.time()
        self.time_wall = time.time() - self.time_start

        global timing_SRV_t0
        timing_SRV_t0 = time.time()

        return feasibe_pos, feasibe_vel

    # Clear the robot class. Note that it doesn't close the persistent connection
    def __del__(self):
        if self.robot_config is not None and self.is_hardware:
            warnings.warn(
                "MyoSuite:> Robot instance was garbage-collected while hardware "
                "connection is still active. Call robot.close() to terminate the "
                "persistent connection before exiting the program.",
                category=ResourceWarning,
                stacklevel=2,
            )

    # Close the persistnent connection to the robot. This should be called only once at the end when persistent connection is no longer needed.
    def close(self) -> None:
        if self.robot_config is not None:
            status = self.hardware_close() if self.is_hardware else True
            if status:
                logger.info("Closed %s (status: %s)", self.name, status)
                self.robot_config = None
            else:
                logger.error("Error closing %s (status: %s)", self.name, status)
        else:
            logger.warning("Trying to close a non-existent robot")

    # -------------------------------------------------------------------------
    # Optional visualization hooks (currently WIP / legacy).
    # -------------------------------------------------------------------------
    def configure_robot_viz(self, robot_config: dict[str, dict[str, Any]]) -> None:
        """Configure robot visualization.

        Note:
            Visualization is currently optional/legacy; this method is a stub so
            enabling `_ROBOT_VIZ` fails loudly with a helpful message instead of
            an `AttributeError`.
        """
        raise NotImplementedError(
            "Robot visualization is not implemented in myosuite4. "
            "Keep `_ROBOT_VIZ = False`."
        )

    def update_robot_viz(
        self, update_sensor: bool = True, update_control: bool = True
    ) -> None:
        """Update robot visualization (stub)."""
        raise NotImplementedError(
            "Robot visualization is not implemented in myosuite4. "
            "Keep `_ROBOT_VIZ = False`."
        )

    def clear_robot_viz(
        self, clear_sensor: bool = True, clear_control: bool = True
    ) -> None:
        """Clear robot visualization (stub)."""
        raise NotImplementedError(
            "Robot visualization is not implemented in myosuite4. "
            "Keep `_ROBOT_VIZ = False`."
        )


def demo_robot() -> None:
    """Demonstrate basic Robot usage against the FrankaReachFixed-v0 env."""
    from myosuite.utils import gym

    logger.info("Starting Robot")
    env = gym.make("FrankaReachFixed-v0")
    rob = env.env.robot

    logger.info("Getting sensor data")
    sen = rob.get_sensors()
    logger.info("Sensor data: %s", sen)

    logger.info("Stepping forward")
    ctrl = env.env.np_random.uniform(size=env.env.mj_model.nu)
    rob.step(ctrl, 1.0)

    logger.info("Resetting Robot")
    pos = env.env.np_random.uniform(size=env.env.mj_model.nq)
    vel = env.env.np_random.uniform(size=env.env.mj_model.nv)
    rob.reset(pos, vel)

    logger.info("Closing Robot")
    rob.close()


if __name__ == "__main__":
    demo_robot()
