""" =================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

"""Domain-randomization for simulations."""

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from robel.robot_env import RobotEnv

# Define types.
Bound = Union[float, Sequence[float]]
Range = Tuple[Bound, Bound]


class SimRandomizer:
    """Randomizes simulation properties."""

    def __init__(self, env: RobotEnv):
        """Initializes a new randomizer.

        Args:
            env: The environment to randomize.
        """
        self.env = env
        self.sim_scene = env.sim_scene
        self.model = self.sim_scene.model
        self.orig_model = self.sim_scene.copy_model()

    @property
    def rand(self) -> np.random.RandomState:
        """Returns the random state."""
        return self.env.np_random

    def randomize_global(self,
                         total_mass_range: Optional[Range] = None,
                         height_field_range: Optional[Range] = None):
        """Randomizes global options in the scene.

        Args:
            total_mass_range: A range of mass values to which all bodies and
                inertias should be scaled to achieve.
            height_field_range: A range of values to generate a noise map
                for the height field.
        """
        if total_mass_range is not None:
            self.sim_scene.get_mjlib().mj_setTotalmass(
                self.sim_scene.get_handle(self.model),
                self.rand.uniform(*total_mass_range))

        if height_field_range is not None:
            self.model.hfield_data[:] = self.rand.uniform(
                *height_field_range, size=np.shape(self.model.hfield_data))

    def randomize_bodies(self,
                         names: Optional[Iterable[str]] = None,
                         all_same: bool = False,
                         position_perturb_range: Optional[Range] = None):
        """Randomizes the bodies in the scene.

        Args:
            names: The body names to randomize. If this is not provided, all
                bodies are randomized.
            all_same: If True, the bodies are assigned the same random value.
            position_perturb_range: A range to perturb the position of the body
                relative to its original position.
        """
        if names is None:
            body_ids = list(range(self.model.nbody))
        else:
            body_ids = [self.model.body_name2id(name) for name in names]
            body_ids = sorted(set(body_ids))
        num_bodies = len(body_ids)

        # Randomize the position relative to the original position.
        if position_perturb_range is not None:
            original_pos = self.orig_model.body_pos[body_ids]
            self.model.body_pos[body_ids] = original_pos + self.rand.uniform(
                *position_perturb_range,
                size=(3,) if all_same else (num_bodies, 3))

    def randomize_geoms(self,
                        names: Optional[Iterable[str]] = None,
                        parent_body_names: Optional[Iterable[str]] = None,
                        all_same: bool = False,
                        size_perturb_range: Optional[Range] = None,
                        color_range: Optional[Range] = None,
                        friction_slide_range: Optional[Range] = None,
                        friction_spin_range: Optional[Range] = None,
                        friction_roll_range: Optional[Range] = None):
        """Randomizes the geoms in the scene.

        Args:
            names: The geom names to randomize. If this nor `parent_body_names`
                is provided, all geoms are randomized.
            parent_body_names: A list of body names whose child geoms should be
                randomized.
            all_same: If True, the geoms are assigned the same random value.
            size_perturb_range: A range to perturb the size of the geom relative
                to its original size.
            color_range: A color range to assign an RGB color to the geom.
            friction_slide_range: The sliding friction that acts along both axes
                of the tangent plane.
            friction_spin_range: The torsional friction acting around the
                contact normal.
            friction_roll_range: The rolling friction acting around both axes of
                the tangent plane.
        """
        if names is None and parent_body_names is None:
            # Address all geoms if no names are given.
            geom_ids = list(range(self.model.ngeom))
        else:
            geom_ids = []
            if names is not None:
                geom_ids.extend(self.model.geom_name2id(name) for name in names)

            # Add child geoms of parent bodies.
            if parent_body_names is not None:
                for name in parent_body_names:
                    body_id = self.model.body_name2id(name)
                    geom_id_start = self.model.body_geomadr[body_id]
                    for i in range(self.model.body_geomnum[body_id]):
                        geom_ids.append(geom_id_start + i)

            geom_ids = sorted(set(geom_ids))
        num_geoms = len(geom_ids)

        # Randomize the size of the geoms relative to the original size.
        if size_perturb_range is not None:
            original_size = self.orig_model.geom_size[geom_ids]
            self.model.geom_size[geom_ids] = original_size + self.rand.uniform(
                *size_perturb_range, size=(3,) if all_same else (num_geoms, 3))

        # Randomize the color of the geoms.
        if color_range is not None:
            self.model.geom_rgba[geom_ids, :3] = self.rand.uniform(
                *color_range, size=(3,) if all_same else (num_geoms, 3))

        # Randomize the friction parameters.
        if friction_slide_range is not None:
            self.model.geom_friction[geom_ids, 0] = self.rand.uniform(
                *friction_slide_range, size=1 if all_same else num_geoms)

        if friction_spin_range is not None:
            self.model.geom_friction[geom_ids, 1] = self.rand.uniform(
                *friction_spin_range, size=1 if all_same else num_geoms)

        if friction_roll_range is not None:
            self.model.geom_friction[geom_ids, 2] = self.rand.uniform(
                *friction_roll_range, size=1 if all_same else num_geoms)

    def randomize_dofs(self,
                       indices: Optional[Iterable[int]] = None,
                       all_same: bool = False,
                       damping_range: Optional[Range] = None,
                       friction_loss_range: Optional[Range] = None):
        """Randomizes the DoFs in the scene.

        Args:
            indices: The DoF indices to randomize. If not provided, randomizes
                all DoFs.
            all_same: If True, the DoFs are assigned the same random value.
            damping_range: The range to assign a damping for the DoF.
            friction_loss_range: The range to assign a friction loss for the
                DoF.
        """
        if indices is None:
            dof_ids = list(range(self.model.nv))
        else:
            nv = self.model.nv
            assert all(-nv <= i < nv for i in indices), \
                'All DoF indices must be in [-{}, {}]'.format(nv, nv-1)
            dof_ids = sorted(set(indices))
        num_dofs = len(dof_ids)

        # Randomize the damping for each DoF.
        if damping_range is not None:
            self.model.dof_damping[dof_ids] = self.rand.uniform(
                *damping_range, size=1 if all_same else num_dofs)

        # Randomize the friction loss for each DoF.
        if friction_loss_range is not None:
            self.model.dof_frictionloss[dof_ids] = self.rand.uniform(
                *friction_loss_range, size=1 if all_same else num_dofs)

    def randomize_actuators(self,
                            indices: Optional[Iterable[int]] = None,
                            all_same: bool = False,
                            kp_range: Optional[Range] = None,
                            kv_range: Optional[Range] = None):
        """Randomizes the actuators in the scene.

        Args:
            indices: The actuator indices to randomize. If not provided,
                randomizes all actuators.
            all_same: If True, the actuators are assigned the same random value.
            kp_range: The position feedback gain range for the actuators.
                This assumes position control actuators.
            kv_range: The velocity feedback gain range for the actuators.
                This assumes velocity control actuators.
        """
        if indices is None:
            act_ids = list(range(self.model.nu))
        else:
            nu = self.model.nu
            assert all(-nu <= i < nu for i in indices), \
                'All actuator indices must be in [-{}, {}]'.format(nu, nu-1)
            act_ids = sorted(set(indices))
        num_acts = len(act_ids)

        # NOTE: For the values below, refer to:
        # http://mujoco.org/book/XMLreference.html#actuator

        # Randomize the Kp for each actuator.
        if kp_range is not None:
            kp = self.rand.uniform(*kp_range, size=1 if all_same else num_acts)
            self.model.actuator_gainprm[:, 0] = kp
            self.model.actuator_biasprm[:, 1] = -kp

        # Randomize the Kv for each actuator.
        if kv_range is not None:
            kv = self.rand.uniform(*kv_range, size=1 if all_same else num_acts)
            self.model.actuator_gainprm[:, 0] = kv
            self.model.actuator_biasprm[:, 2] = -kv
