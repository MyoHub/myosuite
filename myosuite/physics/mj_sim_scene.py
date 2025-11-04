"""=================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================="""

import mujoco

"""Simulation using DeepMind Control Suite."""

import copy
import logging
from typing import Any

import myosuite.utils.import_utils as import_utils
from myosuite.utils.prompt_utils import Prompt, prompt

import_utils.mujoco_isavailable()

from myosuite.physics.sim_scene import SimScene
from myosuite.renderer.mj_renderer import MJRenderer


class DMSimScene(SimScene):
    """Encapsulates a MuJoCo robotics simulation."""

    def _load_simulation(self, model_handle: Any) -> Any:
        """Loads the simulation from the given model handle.

        Args:
            model_handle: This can be a path to a Mujoco XML file, or an MJCF
                object.

        Returns:
            A mujoco MjModel object.
        """
        if isinstance(model_handle, str):
            if model_handle.endswith(".xml"):
                sim = mujoco.MjModel.from_xml_path(model_handle)
            elif isinstance(model_handle, str) and "<mujoco" in model_handle:
                sim = mujoco.MjModel.from_xml_string(model_handle)
            else:
                sim = mujoco.MjModel.from_binary_path(model_handle)
        elif isinstance(model_handle, mujoco.MjModel):
            sim = mujoco.MjModel.from_binary_path(model_handle)
        else:
            raise NotImplementedError(model_handle)

        return sim

    def advance(self, substeps: int = 1, render: bool = True):
        """Advances the simulation for one step."""
        # Step the simulation substeps (frame_skip) times.
        try:
            self.sim.step(substeps)
        except:
            prompt(
                "Simulation couldn't be stepped as intended. Issuing a reset",
                type=Prompt.WARN,
            )
            self.sim.reset()

        if render:
            # self.renderer.refresh_window()
            self.renderer.render_to_window()

    def _create_renderer(self, sim: Any) -> MJRenderer:
        """Creates a renderer for the given simulation."""
        return MJRenderer(sim)

    def copy_model(self) -> Any:
        """Returns a copy of the MjModel object."""
        # MuJoCo's MjModel defines __copy__.
        model_copy = copy.copy(self.model)
        return model_copy

    def save_binary(self, path: str) -> str:
        """Saves the loaded model to a binary .mjb file.

        Returns:
            The file path that the binary was saved to.
        """
        if not path.endswith(".mjb"):
            path = path + ".mjb"
        self.model.save_binary(path)
        return path

    def upload_height_field(self, hfield_id: int):
        """Uploads the height field to the rendering context."""
        if not self.sim.contexts:
            logging.warning("No rendering context; not uploading height field.")
            return
        with self.sim.contexts.gl.make_current() as ctx:
            ctx.call(
                self.get_mjlib().mjr_uploadHField,
                self.model.ptr,
                self.sim.contexts.mujoco.ptr,
                hfield_id,
            )

    def get_mjlib(self) -> Any:
        """Returns an interface to the low-level MuJoCo API."""
        return dm_mujoco.wrapper.mjbindings.mjlib

    def get_handle(self, value: Any) -> Any:
        """Returns a handle that can be passed to mjlib methods."""
        return value.ptr
