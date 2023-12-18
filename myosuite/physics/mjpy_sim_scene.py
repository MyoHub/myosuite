""" =================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

"""Simulation using DeepMind Control Suite."""

import logging
import os
from typing import Any
import numpy as np
import myosuite.utils.import_utils as import_utils
import_utils.mujoco_py_isavailable()
import mujoco_py
from mujoco_py.builder import cymj, user_warning_raise_exception

from myosuite.renderer.mjpy_renderer import MjPyRenderer
from myosuite.physics.sim_scene import SimScene


# Custom handler for MuJoCo exceptions.
def _mj_warning_fn(warn_data: bytes):
    """Warning function override for mujoco_py."""
    try:
        user_warning_raise_exception(warn_data)
    except mujoco_py.MujocoException as e:
        logging.error('MuJoCo Exception: %s', str(e))


cymj.set_warning_callback(_mj_warning_fn)


class MjPySimScene(SimScene):
    """Encapsulates a MuJoCo robotics simulation using mujoco_py."""

    def _load_simulation(self, model_handle: Any) -> Any:
        """Loads the simulation from the given model handle.

        Args:
            model_handle: Path to the Mujoco XML file to load.

        Returns:
            A mujoco_py MjSim object.
        """
        if isinstance(model_handle, str):
            if not os.path.isfile(model_handle):
                raise ValueError(
                    '[MjPySimScene] Invalid model file path: {}'.format(
                        model_handle))

            model = mujoco_py.load_model_from_path(model_handle)
            sim = mujoco_py.MjSim(model)
        else:
            raise NotImplementedError(model_handle)

        return sim

    def _create_renderer(self, sim: Any) -> MjPyRenderer:
        """Creates a renderer for the given simulation."""
        return MjPyRenderer(sim)

    def copy_model(self) -> Any:
        """Returns a copy of the MjModel object."""
        null_model = self.get_mjlib().PyMjModel()
        model_copy = self.get_mjlib().mj_copyModel(null_model, self.model)
        return model_copy

    def save_binary(self, path: str) -> str:
        """Saves the loaded model to a binary .mjb file.

        Returns:
            The file path that the binary was saved to.
        """
        if not path.endswith('.mjb'):
            path = path + '.mjb'
        self.get_mjlib().mj_saveModel(self.model, path.encode(), None, 0)
        return path

    def upload_height_field(self, hfield_id: int):
        """Uploads the height field to the rendering context."""
        if not self.sim.render_contexts:
            logging.warning('No rendering context; not uploading height field.')
            return
        self.get_mjlib().mjr_uploadHField(
            self.model, self.sim.render_contexts[0].con, hfield_id)

    def _patch_mjlib_accessors(self, lib):
        class _mjtTrn:
            def __init__(self, lib):
                self.mjTRN_JOINT = lib.const.TRN_JOINT
                self.mjTRN_JOINTINPARENT = lib.const.TRN_JOINTINPARENT
                self.mjTRN_SLIDERCRANK = lib.const.TRN_SLIDERCRANK
                self.mjTRN_TENDON = lib.const.TRN_TENDON
                self.mjTRN_SITE = lib.const.TRN_SITE
                # self.mjTRN_BODY = lib.const.TRN_BODY # New MuJoCo version only

        class _mjtJoint:
            def __init__(self, lib):
                self.mjJNT_FREE = lib.const.JNT_FREE
                self.mjJNT_BALL = lib.const.JNT_BALL
                self.mjJNT_SLIDE = lib.const.JNT_SLIDE
                self.mjJNT_HINGE = lib.const.JNT_HINGE

        lib.mjtTrn = _mjtTrn(lib)
        lib.mjtJoint = _mjtJoint(lib)

        # patch jacobians to use non flattened arrays for compatibility with dm mujoco bindings
        lib.mj_jac_orig = lib.mj_jac # save before overwritting
        def _mj_jac(model, data, jacp, jacr, point, body):
            lib.mj_jac_orig(model, data, np.ravel(jacp), np.ravel(jacr), point, body)
        lib.mj_jac = _mj_jac

        lib.mj_jacBody_orig = lib.mj_jacBody
        def _mj_jacBody(model, data, jacp, jacr, body):
            lib.mj_jacBody_orig(model, data, np.ravel(jacp), np.ravel(jacr), body)
        lib.mj_jacBody = _mj_jacBody

        lib.mj_jacBodyCom_orig = lib.mj_jacBodyCom
        def _mj_jacBodyCom(model, data, jacp, jacr, body):
            lib.mj_jacBodyCom_orig(model, data, np.ravel(jacp), np.ravel(jacr), body)
        lib.mj_jacBodyCom = _mj_jacBodyCom

        # lib.mj_jacSubtreeCom_orig = lib.mj_jacSubtreeCom
        # def _mj_jacSubtreeCom(model, data, jacp, body):
        #     lib.mj_jacSubtreeCom_orig(model, data, np.ravel(jacp), body)
        # lib.mj_jacSubtreeCom = _mj_jacSubtreeCom

        lib.mj_jacGeom_orig = lib.mj_jacGeom
        def _mj_jacGeom(model, data, jacp, jacr, geom):
            lib.mj_jacGeom_orig(model, data, np.ravel(jacp), np.ravel(jacr), geom)
        lib.mj_jacGeom = _mj_jacGeom

        lib.mj_jacSite_orig = lib.mj_jacSite
        def _mj_jacSite(model, data, jacp, jacr, site):
            lib.mj_jacSite_orig(model, data, np.ravel(jacp), np.ravel(jacr), site)
        lib.mj_jacSite = _mj_jacSite

        lib.mj_jacPointAxis_orig = lib.mj_jacPointAxis
        def _mj_jacPointAxis(model, data, jacPoint, jacAxis, point, axis, body):
            lib.mj_jacPointAxis_orig(model, data, np.ravel(jacPoint), np.ravel(jacAxis), point, axis, body)
        lib.mj_jacPointAxis = _mj_jacPointAxis


    def get_mjlib(self) -> Any:
        """Returns an interface to the low-level MuJoCo API."""
        mjlib = _MjlibWrapper(mujoco_py.cymj)
        self._patch_mjlib_accessors(mjlib)
        return mjlib

    def get_handle(self, value: Any) -> Any:
        """Returns a handle that can be passed to mjlib methods."""
        return value

    def advance(self, substeps:int=1, render:bool = True):
        """Advances the simulation substeps times forward."""
        with mujoco_py.ignore_mujoco_warnings():
            functions = self.get_mjlib()
            model = self.get_handle(self.sim.model)
            data = self.get_handle(self.sim.data)
            for _ in range(substeps):
                functions.mj_step2(model, data)
                functions.mj_step1(model, data)
                if render:
                    # self.renderer.refresh_window()
                    self.renderer.render_to_window()


class _MjlibWrapper:
    """Wrapper that forwards mjlib calls."""

    def __init__(self, lib):
        self._lib = lib

    def __getattr__(self, name: str):
        if name.startswith('mj'):
            return getattr(self._lib, '_' + name)
        return getattr(self._lib, name)
