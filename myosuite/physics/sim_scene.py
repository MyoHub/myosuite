""" =================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """


"""Simulated robot API backed by MuJoCo."""

import abc
import contextlib
import enum
from typing import Any, Union
import os
from myosuite.renderer.renderer import Renderer


class SimBackend(enum.Enum):
    """Simulation library types."""
    MUJOCO_PY = 0
    MUJOCO = 1

    # resolve sim backend
    @staticmethod
    def get_sim_backend()->'SimBackend':
        sim_backend = os.getenv('sim_backend')
        if sim_backend == 'MUJOCO_PY':
            return SimBackend.MUJOCO_PY
        elif sim_backend == 'MUJOCO' or sim_backend == None:
            return SimBackend.MUJOCO
        else:
            raise ValueError("Unknown sim_backend: {}. Available choices: MUJOCO_PY, MUJOCO")


class SimScene(metaclass=abc.ABCMeta):
    """Encapsulates a MuJoCo robotics simulation."""

    @staticmethod
    def create(*args, backend: Union[SimBackend, int], **kwargs) -> 'SimScene':
        """Creates a new simulation scene.

        Args:
            *args: Positional arguments to pass to the simulation.
            backend: The simulation backend to use to load the simulation.
            **kwargs: Keyword arguments to pass to the simulation.

        Returns:
            A SimScene object.
        """
        backend = SimBackend(backend)
        if backend == SimBackend.MUJOCO_PY:
            from myosuite.physics import mjpy_sim_scene  # type: ignore
            return mjpy_sim_scene.MjPySimScene(*args, **kwargs)
        elif backend == SimBackend.MUJOCO:
            from myosuite.physics import mj_sim_scene  # type: ignore
            return mj_sim_scene.DMSimScene(*args, **kwargs)
        else:
            raise NotImplementedError(backend)


    # Get sim as per the sim_backend
    @staticmethod
    def get_sim(model_handle: Any) -> 'SimScene':
        sim_backend = SimBackend.get_sim_backend()
        if sim_backend == SimBackend.MUJOCO_PY:
            return SimScene.create(model_handle=model_handle, backend=SimBackend.MUJOCO_PY)
        elif sim_backend == SimBackend.MUJOCO:
            return SimScene.create(model_handle=model_handle, backend=SimBackend.MUJOCO)
        else:
            raise ValueError("Unknown sim_backend: {}. Available choices: MUJOCO_PY, MUJOCO")


    def __init__(
            self,
            model_handle: Any,
    ):
        """Initializes a new simulation.

        Args:
            model_handle: The simulation model to load. This can be a XML file,
                or a format/object specific to the simulation backend.
        """
        self.sim = self._load_simulation(model_handle)
        self.model = self.sim.model
        self.data = self.sim.data
        self.lib = self.get_mjlib()

        self.renderer = self._create_renderer(self.sim)

        # Save initial values.
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

    @property
    def step_duration(self):
        """Returns the simulation step duration in seconds."""
        return self.model.opt.timestep

    def close(self):
        """Cleans up any resources used by the simulation."""
        self.renderer.close()

    def forward(self):
        """Run the simulation forward"""
        self.sim.forward()
        self.renderer.refresh_window()

    def reset(self):
        """Reset the simulation forward"""
        self.sim.reset()
        self.renderer.refresh_window()

    def disable_option(self,
                       constraint_solver: bool = False,
                       limits: bool = False,
                       contact: bool = False,
                       gravity: bool = False,
                       clamp_ctrl: bool = False,
                       actuation: bool = False):
        """Disables option(s) in the simulation."""
        # http://www.mujoco.org/book/APIreference.html#mjtDisableBit
        if constraint_solver:
            self.model.opt.disableflags |= (1 << 0)
        if limits:
            self.model.opt.disableflags |= (1 << 3)
        if contact:
            self.model.opt.disableflags |= (1 << 4)
        if gravity:
            self.model.opt.disableflags |= (1 << 6)
        if clamp_ctrl:
            self.model.opt.disableflags |= (1 << 7)
        if actuation:
            self.model.opt.disableflags |= (1 << 10)


    # get state of the scene
    def get_state(self):
        tt = self.data.time
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        act = self.data.act.ravel().copy() if self.model.na>0 else None
        return dict(time=tt,
                    qpos=qp,
                    qvel=qv,
                    act=act)

    # set state of the scene
    def set_state(self, time=None, qpos=None, qvel=None, act=None):
        if time:
            self.data.time = time
        if qpos is not None:
            assert qpos.shape == (self.model.nq,)
            self.sim.data.qpos[:] = qpos
        if qvel is not None:
            assert qvel.shape == (self.model.nv,)
            self.sim.data.qvel[:] = qvel
        if self.model.na>0 and act is not None:
            assert act.shape == (self.model.na,)
            self.sim.data.act[:] = act
        self.sim.forward()

    @contextlib.contextmanager
    def disable_option_context(self, **kwargs):
        """Disables options(s) in the simulation for the context."""
        original_flags = self.model.opt.disableflags
        self.disable_option(**kwargs)
        yield
        self.model.opt.disableflags = original_flags

    @abc.abstractmethod
    def copy_model(self) -> Any:
        """Returns a copy of the MjModel object."""

    @abc.abstractmethod
    def save_binary(self, path: str) -> str:
        """Saves the loaded model to a binary .mjb file.

        Returns:
            The file path that the binary was saved to.
        """

    @abc.abstractmethod
    def upload_height_field(self, hfield_id: int):
        """Uploads the height field to the rendering context."""

    @abc.abstractmethod
    def get_handle(self, value: Any) -> Any:
        """Returns a handle that can be passed to mjlib methods."""

    @abc.abstractmethod
    def get_mjlib(self) -> Any:
        """Returns an interface to the low-level MuJoCo API."""

    @abc.abstractmethod
    def _load_simulation(self, model_handle: Any) -> Any:
        """Loads the simulation from the given model handle."""

    @abc.abstractmethod
    def _create_renderer(self, sim: Any) -> Renderer:
        """Creates a renderer for the given simulation."""

    @abc.abstractmethod
    def advance(self, substeps: int, render:bool):
        """Advances the simulation substeps times forward."""

