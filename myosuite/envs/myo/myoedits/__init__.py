"""
Copyright (c) 2024 MyoSuite
Authors :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), James Heald (jamesbheald@gmail.com)
Source :: https://github.com/MyoHub/myosuite
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from myosuite.utils import gym; register=gym.register
from myosuite.envs.myo.myobase import register_env_with_variants

import os
import numpy as np

import mujoco

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Arm Reaching ==============================
def edit_fn_arm_reaching(spec: mujoco.MjSpec) -> None:

	# Get the positions of each body of each digit
	root_list = ['firstmc', 'secondmc', 'thirdmc', 'fourthmc', 'fifthmc']
	body_positions = {}
	IFtip_site = {}

	for root in root_list:
		body_positions[root] = []
		body = spec.body(root)
		child_body = body.first_body()
		while child_body is not None:
			mesh_names = [geom.name for geom in child_body.geoms if geom.type == mujoco.mjtGeom.mjGEOM_MESH]
			body_positions[root].append((child_body.name, child_body.pos.copy(), mesh_names))
			child_body = child_body.first_body()

	# Get the properties of the IFtip site
	site = spec.site('IFtip')
	IFtip_site = {
		attr: getattr(site, attr).copy() if hasattr(getattr(site, attr), 'copy') else getattr(site, attr)
		for attr in ['name', 'size', 'pos', 'rgba']
	}

	# Remove the digits
	for root in root_list:
		root_body = spec.body(root)
		child_body = root_body.first_body()
		if child_body is not None:
			spec.detach_body(child_body)

	# Add back simplified digits (each phalanx is a body with a mesh)
	for root in root_list:
		body = spec.body(root)
		for body_name, pos, mesh_names in body_positions[root]:
			# body_name = body_name + 'edited'
			body.add_body(name=body_name, pos=pos)
			for mesh_name in mesh_names:
				spec.body(body_name).add_geom(meshname=mesh_name, name=body_name, type=mujoco.mjtGeom.mjGEOM_MESH)
			if body_name == 'distph2':
				spec.body(body_name).add_site(name=IFtip_site['name'],
							 			 size=IFtip_site['size'] * 2,
										 pos=IFtip_site['pos'],
										 rgba=IFtip_site['rgba'])
			body = spec.body(body_name)

	# Add a reach target
	spec.body('world').add_site(name='IFtip_target',
							    type=mujoco.mjtGeom.mjGEOM_SPHERE,
								size=[0.02, 0.02, 0.02],
								pos=[-0.2, -0.2, 1.2],
								rgba=[0., 0., 1., 0.3])


register_env_with_variants(id='myoArmReachFixed-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=150,
        kwargs={
			'model_path': curr_dir+'/../../../simhive/myo_sim/arm/myoarm.xml',
            'target_reach_range': {
                'IFtip': ((-0.175, -0.245, 1.405), (-0.175, -0.245, 1.405)),
                },
            'normalize_act': True,
            'far_th': 1.,
            'edit_fn': edit_fn_arm_reaching,
            }
    )

register_env_with_variants(id='myoArmReachRandom-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=150,
        kwargs={
			'model_path': curr_dir+'/../../../simhive/myo_sim/arm/myoarm.xml',
            'target_reach_range': {
                'IFtip': ((-0.175-0.175, -0.245-0.175, 1.405-0.425), (-0.175+0.175, -0.245+0.175, 1.405+0.425)),
                },
            'normalize_act': True,
            'far_th': 1.,
			'edit_fn': edit_fn_arm_reaching,
            }
    )