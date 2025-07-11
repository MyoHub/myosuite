""" =================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """
import mujoco

"""Simulation using DeepMind Control Suite."""

import copy
import logging
from typing import Any

import myosuite.utils.import_utils as import_utils
from myosuite.utils.prompt_utils import prompt, Prompt
import_utils.dm_control_isavailable()
import_utils.mujoco_isavailable()
import dm_control.mujoco as dm_mujoco
from dm_control.mujoco.wrapper import MjModel as dm_MjModel

from myosuite.renderer.mj_renderer import MJRenderer
from myosuite.physics.sim_scene import SimScene


class DMSimScene(SimScene):
    """Encapsulates a MuJoCo robotics simulation using dm_control."""

    def _load_simulation(self, model_handle: Any) -> Any:
        """Loads the simulation from the given model handle.

        Args:
            model_handle: This can be a path to a Mujoco XML file, or an MJCF
                object.

        Returns:
            A dm_control Physics object.
        """
        if isinstance(model_handle, str):
            if model_handle.endswith('.xml'):
                sim = dm_mujoco.Physics.from_xml_path(model_handle)
            elif isinstance(model_handle, str) and "<mujoco" in model_handle:
                sim = dm_mujoco.Physics.from_xml_string(model_handle)
            else:
                sim = dm_mujoco.Physics.from_binary_path(model_handle)
        elif isinstance(model_handle, mujoco.MjModel) or isinstance(model_handle, dm_MjModel):
            sim = dm_mujoco.Physics.from_model(model_handle)
        else:
            raise NotImplementedError(model_handle)
        self._patch_mjmodel_accessors(sim.model)
        self._patch_mjdata_accessors(sim.data)
        return sim

    def advance(self, substeps: int = 1, render:bool = True):
        """Advances the simulation for one step."""
        # Step the simulation substeps (frame_skip) times.
        try:
            self.sim.step(substeps)
        except:
            prompt("Simulation couldn't be stepped as intended. Issuing a reset", type=Prompt.WARN)
            self.sim.reset()

        if render:
            # self.renderer.refresh_window()
            self.renderer.render_to_window()

    def _create_renderer(self, sim: Any) -> MJRenderer:
        """Creates a renderer for the given simulation."""
        return MJRenderer(sim)

    def copy_model(self) -> Any:
        """Returns a copy of the MjModel object."""
        # dm_control's MjModel defines __copy__.
        model_copy = copy.copy(self.model)
        self._patch_mjmodel_accessors(model_copy)
        return model_copy

    def save_binary(self, path: str) -> str:
        """Saves the loaded model to a binary .mjb file.

        Returns:
            The file path that the binary was saved to.
        """
        if not path.endswith('.mjb'):
            path = path + '.mjb'
        self.model.save_binary(path)
        return path

    def upload_height_field(self, hfield_id: int):
        """Uploads the height field to the rendering context."""
        if not self.sim.contexts:
            logging.warning('No rendering context; not uploading height field.')
            return
        with self.sim.contexts.gl.make_current() as ctx:
            ctx.call(self.get_mjlib().mjr_uploadHField, self.model.ptr,
                     self.sim.contexts.mujoco.ptr, hfield_id)

    def get_mjlib(self) -> Any:
        """Returns an interface to the low-level MuJoCo API."""
        return dm_mujoco.wrapper.mjbindings.mjlib

    def get_handle(self, value: Any) -> Any:
        """Returns a handle that can be passed to mjlib methods."""
        return value.ptr

    def _patch_mjmodel_accessors(self, model):
        """Adds accessors to MjModel objects to support mujoco_py API.

        This adds `*_name2id` methods to a Physics object to have API
        consistency with mujoco_py.

        TODO(michaelahn): Deprecate this in favor of dm_control's named methods.
        """
        mjlib = self.get_mjlib()

        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(model.ptr,
                                      mjlib.mju_str2Type(type_name.encode()),
                                      name.encode())
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(
                    type_name, name))
            return obj_id

        def get_xml():
            from tempfile import TemporaryDirectory
            import os
            with TemporaryDirectory() as td:
                filename = os.path.join(td, 'model.xml')
                ret = mjlib.mj_saveLastXML(filename.encode(), model.ptr)
                if ret == 0:
                    raise Exception('Failed to save XML')
                with open(filename, 'r') as f:
                    return f.read()

        if not hasattr(model, 'body_name2id'):
            model.body_name2id = lambda name: name2id('body', name)

        if not hasattr(model, 'geom_name2id'):
            model.geom_name2id = lambda name: name2id('geom', name)

        if not hasattr(model, 'site_name2id'):
            model.site_name2id = lambda name: name2id('site', name)

        if not hasattr(model, 'joint_name2id'):
            model.joint_name2id = lambda name: name2id('joint', name)

        if not hasattr(model, 'actuator_name2id'):
            model.actuator_name2id = lambda name: name2id('actuator', name)

        if not hasattr(model, 'camera_name2id'):
            model.camera_name2id = lambda name: name2id('camera', name)

        if not hasattr(model, 'sensor_name2id'):
            model.sensor_name2id = lambda name: name2id('sensor', name)

        if not hasattr(model, 'get_xml'):

            model.get_xml = lambda : get_xml()


    def _patch_mjdata_accessors(self, data):
        """Adds accessors to MjData objects to support mujoco_py API."""
        if not hasattr(data, 'body_xpos'):
            data.body_xpos = data.xpos

        if not hasattr(data, 'body_xquat'):
            data.body_xquat = data.xquat
