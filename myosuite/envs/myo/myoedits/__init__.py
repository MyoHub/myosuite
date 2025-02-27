""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), James Heald (jamesbheald@gmail.com)
================================================= """

from myosuite.utils import gym; register=gym.register
from myosuite.envs.myo.myobase import register_env_with_variants

import os
import numpy as np

import mujoco

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Arm Reaching ==============================
def edit_fn_arm_reaching(spec: mujoco.MjSpec):

	root_list = ['firstmc', 'secondmc', 'thirdmc', 'fourthmc', 'fifthmc']
	body_positions = {}
	IFtip_site = {}
	for root in root_list:
		body_positions[root] = []
		body = spec.find_body(root)
		child_body = body.first_body()
		while child_body is not None:
			body_positions[root].append(child_body.pos.copy())
			if child_body.name == 'distph2':
				for site in child_body.sites:
					if site.name == 'IFtip':
						IFtip_site['name'] = site.name
						IFtip_site['size'] = site.size.copy()
						IFtip_site['pos'] = site.pos.copy()
						IFtip_site['rgba'] = site.rgba.copy()
			child_body = child_body.first_body()

	# remove fingers
	spec.detach_body(spec.find_body('proximal_thumb'))
	spec.detach_body(spec.find_body('proxph2'))
	spec.detach_body(spec.find_body('proxph3'))
	spec.detach_body(spec.find_body('proxph4'))
	spec.detach_body(spec.find_body('proxph5'))

	# add back simplified digits (each phalanx is a body with a mesh)
	
	# thumb
	spec.find_body('firstmc').add_body(name="thumbprox", pos=body_positions['firstmc'][0])
	spec.find_body('thumbprox').add_geom(meshname="thumbprox", name="thumbprox", type=mujoco.mjtGeom.mjGEOM_MESH)
	spec.find_body('thumbprox').add_body(name="thumbdist", pos=body_positions['firstmc'][1])
	spec.find_body('thumbdist').add_geom(meshname="thumbdist", name="thumbdist", type=mujoco.mjtGeom.mjGEOM_MESH)
	
	# non-thumb digits
	for i, root in enumerate(root_list):
		body = spec.find_body(root)
		for j, body_position in enumerate(body_positions[root]):
			if root != 'firstmc':
				if j == 0:
					phalanx = 'proxph'
				elif j == 1:
					phalanx = 'midph'
				elif j == 2:
					phalanx = 'distph'
				name = str(i+1) + phalanx
				body.add_body(name=name, pos=body_position)
				spec.find_body(name).add_geom(meshname=name, name=name, type=mujoco.mjtGeom.mjGEOM_MESH)
				body = spec.find_body(name)
				if name == '2distph':
					body.add_site(name=IFtip_site['name'],
								  size=IFtip_site['size']*2,
								  pos=IFtip_site['pos'],
								  rgba=IFtip_site['rgba'])

	spec.find_body('world').add_site(name='IFtip_target',
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
                'IFtip': ((-0.2, -0.2, 1.2), (-0.2, -0.2, 1.2)),
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
                'IFtip': ((-0.2-0.30, -0.2-0.30, 1.2-0.30), (-0.2+0.30, -0.2+0.30, 1.2+0.30)),
                },
            'normalize_act': True,
            'far_th': 1.,
			'edit_fn': edit_fn_arm_reaching,
            }
    )