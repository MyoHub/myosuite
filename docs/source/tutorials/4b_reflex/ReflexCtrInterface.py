# Author(s): Seungmoon Song <seungmoon.song@gmail.com>, Chun Kwang Tan <riodren.tan@gmail.com>
"""
adapted from:
- Song and Geyer. "A neural circuitry that emphasizes
spinal feedback generates diverse behaviours of human locomotion." The
Journal of physiology, 2015.
"""

from __future__ import division # '/' always means non-truncating division
import numpy as np
from reflexCtr import LocoCtrl

import myosuite
from myosuite.utils import gym

import numpy as np
import os

from myosuite.utils.quat_math import quat2euler
from myosuite.utils.quat_math import euler2quat

class MyoLegReflex(object):

    DEFAULT_INIT_POSE = {}
    DEFAULT_INIT_POSE['model_pose'] = {'yaw':np.deg2rad(0),'pitch':np.deg2rad(15),'roll':np.deg2rad(0)}
    DEFAULT_INIT_POSE['model_height'] = 0.92
    DEFAULT_INIT_POSE['joint_angles'] = {
        'hip_flexion_r': np.deg2rad(180-190),
        'hip_flexion_l': np.deg2rad(180-155),
        'knee_angle_r': np.deg2rad(180-165),
        'knee_angle_l': np.deg2rad(180-180),
        'ankle_angle_r': np.deg2rad(90-90),
        'ankle_angle_l': np.deg2rad(90-100),
    }
    DEFAULT_INIT_POSE['velocity'] = {'cartesian':[1.5, 0.0, 0.0]}

    def __init__(self, init_dict=DEFAULT_INIT_POSE, dt=0.01, mode='3D', sim_time=2.0, seed=0): # Default mode was '3D', currently defaulting to 2D (13 Mar 2023)
        self.dt = dt
        self.t = 0
        self.mode = mode
        
        self.n_par = len(LocoCtrl.cp_keys)
        control_dimension = 3
        
        self.cp_map = LocoCtrl.cp_map
        self.ReflexCtrl = LocoCtrl(self.dt, control_dimension=control_dimension, params=np.ones(self.n_par))

        # Myosuite setup
        self.sim_time = sim_time
        self.timestep_limit = int(self.sim_time/self.dt)

        self.init_dict = init_dict
        self.seed = seed

        self.env = gym.make('myoLegStandRandom-v0', normalize_act=False)

        print(f"Seed added - ", seed)
        self.env.reset()
        self.env.seed(seed)

        self.muscle_labels = {}
        self.muscles_dict = {}
        self.muscle_Fmax = {}
        self.muscle_L0 = {}

        self.init_pelvis = np.zeros(3,) # Variable to hold the initial position of the pelvis (for reward calculations)

        self.footstep = {}
        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = 0
        self.footstep['l_contact'] = 0
        
        # Accessor for LocoCtrl
        self.cp = self.ReflexCtrl.cp

# -----------------------------------------------------------------------------------------------------------------
    def reset(self):
        
        self.env.reset()
        self.env.seed(self.seed)
        
        self.ReflexCtrl.reset()
        
        self._set_muscle_groups()
        self._set_initial_pose(self.init_dict)

        
# -----------------------------------------------------------------------------------------------------------------
    def update(self):
        self.t += self.dt
        self.ReflexCtrl.update(self.get_obs_dict())
        return self.ReflexCtrl.stim.copy()
        

# -----------------------------------------------------------------------------------------------------------------
    def set_control_params(self, params):
        self.ReflexCtrl.set_control_params(params)

# -----------------------------------------------------------------------------------------------------------------
    def set_control_params_RL(self, s_leg, params):
        self.ReflexCtrl.set_control_params_RL(s_leg, params)

# -----------------------------------------------------------------------------------------------------------------
    def get_obs_dict(self):
        # Function translate Myosuite joint angle conventions into the conventions used by the reflex controller
        # refer to LocoCtrl.s_b_keys and LocoCtrl.s_l_keys
        # coordinate in body frame
        #   [0] x: forward
        #   [1] y: leftward
        #   [2] z: upward

        # Getting values directly from the Mujoco env, and translating them into the controller convention
        # Measurement is in world coordinates
        pel_euler = quat2euler(self.env.sim.data.body('pelvis').xquat.copy())
        pelvis_roll = pel_euler[0] - (np.pi/2)
        pelvis_pitch = pel_euler[2] * (-1)
        pelvis_yaw = pel_euler[1] * (-1)

        # Pelvis velocities and angular velocities
        temp_seg_vel = self.env.sim.data.object_velocity('pelvis','body', local_frame=False).copy()
        lin_seg_vel = temp_seg_vel[0]
        dx_local, dy_local = self.rotate_frame(lin_seg_vel[0], lin_seg_vel[1], pelvis_yaw)
        pelvis_vel = np.hstack((np.array([dx_local, dy_local, lin_seg_vel[2]]),
                                            temp_seg_vel[1] ))

        # GRF from foot contact sensor values
        # GRF from foot contact sensor values
        temp_right = (self.env.sim.data.sensor('r_foot').data[0].copy() + self.env.sim.data.sensor('r_toes').data[0].copy())
        temp_left = (self.env.sim.data.sensor('l_foot').data[0].copy() + self.env.sim.data.sensor('l_toes').data[0].copy())

        sensor_data = {'body':{}, 'r_leg':{}, 'l_leg':{}}
        sensor_data['body']['theta'] = [pelvis_roll, # around local x axis
                                        pelvis_pitch] # around local y axis

        sensor_data['body']['d_pos'] = [pelvis_vel[0], # local x (+) forward
                                        pelvis_vel[1]] # local y (+) leftward
        
        sensor_data['body']['dtheta'] = [pelvis_vel[3], # around local x axis
                                        pelvis_vel[4]] # around local y axis
        
        sensor_data['r_leg']['load_ipsi'] = temp_right / (np.sum(self.env.sim.model.body_mass)*9.8)
        sensor_data['l_leg']['load_ipsi'] = temp_left / (np.sum(self.env.sim.model.body_mass)*9.8)

        for s_leg, s_legc in zip(['r_leg', 'l_leg'], ['l_leg', 'r_leg']):

            sensor_data[s_leg]['contact_ipsi'] = 1 if sensor_data[s_leg]['load_ipsi'] > 0.1 else 0
            sensor_data[s_leg]['contact_contra'] = 1 if sensor_data[s_legc]['load_ipsi'] > 0.1 else 0
            sensor_data[s_leg]['load_contra'] = sensor_data[s_legc]['load_ipsi']

            sensor_data[s_leg]['phi_hip'] = (np.pi - self.env.sim.data.jnt(f"hip_flexion_{s_leg[0]}").qpos[0].copy())
            sensor_data[s_leg]['phi_knee'] = (np.pi - self.env.sim.data.jnt(f"knee_angle_{s_leg[0]}").qpos[0].copy())
            sensor_data[s_leg]['phi_ankle'] = (0.5*np.pi - self.env.sim.data.jnt(f"ankle_angle_{s_leg[0]}").qpos[0].copy())
            sensor_data[s_leg]['dphi_knee'] = self.env.sim.data.jnt(f"knee_angle_{s_leg[0]}").qvel[0].copy()

            # alpha = hip - 0.5*knee
            sensor_data[s_leg]['alpha'] = sensor_data[s_leg]['phi_hip'] - 0.5*sensor_data[s_leg]['phi_knee']
            dphi_hip = self.env.sim.data.jnt(f"hip_flexion_{s_leg[0]}").qvel[0].copy()
            sensor_data[s_leg]['dalpha'] = dphi_hip - 0.5*sensor_data[s_leg]['dphi_knee']

            sensor_data[s_leg]['alpha_f'] = (-1*self.env.sim.data.jnt(f"hip_adduction_{s_leg[0]}").qpos[0].copy()) + 0.5*np.pi

            temp_mus_force = self.env.sim.data.actuator_force.copy()

            sensor_data[s_leg]['F_RF'] = -1*np.mean( temp_mus_force[self.muscles_dict[s_leg]['RF']] / (self.muscle_Fmax[s_leg]['RF']) )
            sensor_data[s_leg]['F_VAS'] = -1*np.mean( temp_mus_force[self.muscles_dict[s_leg]['VAS']] / (self.muscle_Fmax[s_leg]['VAS']) )
            sensor_data[s_leg]['F_GAS'] = -1*np.mean( temp_mus_force[self.muscles_dict[s_leg]['GAS']] / (self.muscle_Fmax[s_leg]['GAS']) )
            sensor_data[s_leg]['F_SOL'] = -1*np.mean( temp_mus_force[self.muscles_dict[s_leg]['SOL']] / (self.muscle_Fmax[s_leg]['SOL']) )

        return sensor_data

# ---------------------------------------------------------------------------------------------------
# Integration of code with Myosuite control codes

    def run_reflex_step(self):
        # Run a step of the Mujoco env and Reflex controller
        is_done = False

        new_act = self.reflex2mujoco(self.update())
        self.env.step(new_act)

        self.update_footstep()
        
        # Have to collect observations after step, otherwise brain cmd would not have any values
        out_dict = self.get_obs_dict()
        
        temp_pel_euler = quat2euler(self.env.sim.data.body('root').xquat.copy())
        
        # Check if the simulation is still alive (height of pelvs still above threshold, has not fallen down yet)
        if self.env.sim.data.body('pelvis').xpos[2] < 0.65: # (Emprical testing) Even for very bent knee walking, height of pelvis is about 0.78
            is_done = True
        if temp_pel_euler[1] < np.deg2rad(-30) or temp_pel_euler[1] > np.deg2rad(30):
            # Punish for too much pitch of pelvis
            is_done = True
        
        return [ out_dict, is_done, np.round(self.env.sim.data.time,2), new_act]

    # ---------- Initialization Functions ----------
    def _set_muscle_groups(self):
        # ----- Gluteus group -----
        glu_r = [self.env.sim.model.actuator('glmax1_r').id,
        self.env.sim.model.actuator('glmax2_r').id,
        self.env.sim.model.actuator('glmax3_r').id,
        self.env.sim.model.actuator('glmed3_r').id]

        glu_l = [self.env.sim.model.actuator('glmax1_l').id,
        self.env.sim.model.actuator('glmax2_l').id,
        self.env.sim.model.actuator('glmax3_l').id,
        self.env.sim.model.actuator('glmed3_l').id]

        glu_r_lbl = ['glmax1_r','glmax2_r','glmax3_r','glmed3_r']
        glu_l_lbl = ['glmax1_l','glmax2_l','glmax3_l','glmed3_l']

        # ----- Hamstring (semitendinosus and semimembranosus) -----
        ham_r = [self.env.sim.model.actuator('semimem_r').id,
                self.env.sim.model.actuator('semiten_r').id,
                self.env.sim.model.actuator('bflh_r').id]

        ham_l = [self.env.sim.model.actuator('semimem_l').id,
                self.env.sim.model.actuator('semiten_l').id,
                self.env.sim.model.actuator('bflh_l').id]

        ham_r_lbl = ['semimem_r','semiten_r','bflh_r']
        ham_l_lbl = ['semimem_l','semiten_l','bflh_l']

        # ----- BF short head (biceps femoris) -----
        bfsh_r = [self.env.sim.model.actuator('bfsh_r').id]

        bfsh_l = [self.env.sim.model.actuator('bfsh_l').id]

        bfsh_r_lbl = ['bfsh_r']
        bfsh_l_lbl = ['bfsh_l']

        # ----- Gastrocnemius -----
        gas_r = [self.env.sim.model.actuator('gaslat_r').id,
                self.env.sim.model.actuator('gasmed_r').id]

        gas_l = [self.env.sim.model.actuator('gaslat_l').id,
                self.env.sim.model.actuator('gasmed_l').id]

        gas_r_lbl = ['gaslat_r','gasmed_r']
        gas_l_lbl = ['gaslat_l','gasmed_l']

        # ----- Soleus -----
        sol_r = [self.env.sim.model.actuator('soleus_r').id,
                self.env.sim.model.actuator('perbrev_r').id,
                self.env.sim.model.actuator('perlong_r').id,
                self.env.sim.model.actuator('tibpost_r').id]

        sol_l = [self.env.sim.model.actuator('soleus_l').id,
                self.env.sim.model.actuator('perbrev_l').id,
                self.env.sim.model.actuator('perlong_l').id,
                self.env.sim.model.actuator('tibpost_l').id]

        sol_r_lbl = ['soleus_r','perbrev_r','perlong_r','tibpost_r']
        sol_l_lbl = ['soleus_l','perbrev_l','perlong_l','tibpost_l']

        # ----- Hip Flexors (psoas and iliacus) -----
        hfl_r = [self.env.sim.model.actuator('psoas_r').id,
                self.env.sim.model.actuator('iliacus_r').id]

        hfl_l = [self.env.sim.model.actuator('psoas_l').id,
                self.env.sim.model.actuator('iliacus_l').id]

        hfl_r_lbl = ['psoas_r','iliacus_r']
        hfl_l_lbl = ['psoas_l','iliacus_l']

        # ----- Hip Abductors (piriformis, satorius and tensor fasciae latae) -----
        hab_r = [self.env.sim.model.actuator('piri_r').id,
        self.env.sim.model.actuator('sart_r').id,
        self.env.sim.model.actuator('glmed1_r').id,
        self.env.sim.model.actuator('glmed2_r').id,
        self.env.sim.model.actuator('glmin1_r').id,
        self.env.sim.model.actuator('glmin2_r').id,
        self.env.sim.model.actuator('glmin3_r').id]

        hab_l = [self.env.sim.model.actuator('piri_l').id,
        self.env.sim.model.actuator('sart_l').id,
        self.env.sim.model.actuator('glmed1_l').id,
        self.env.sim.model.actuator('glmed2_l').id,
        self.env.sim.model.actuator('glmin1_l').id,
        self.env.sim.model.actuator('glmin2_l').id,
        self.env.sim.model.actuator('glmin3_l').id]

        hab_r_lbl = ['piri_r','sart_r','glmed1_r','glmed2_r','glmin1_r','glmin2_r','glmin3_r']
        hab_l_lbl = ['piri_l','sart_l','glmed1_l','glmed2_l','glmin1_l','glmin2_l','glmin3_l']

        # ----- Hip Abbuctors (adductor [brevis, longus, magnus], gracilis) -----
        had_r = [self.env.sim.model.actuator('addbrev_r').id,
        self.env.sim.model.actuator('addlong_r').id,
        self.env.sim.model.actuator('addmagDist_r').id,
        self.env.sim.model.actuator('addmagIsch_r').id,
        self.env.sim.model.actuator('addmagMid_r').id,
        self.env.sim.model.actuator('addmagProx_r').id,
        self.env.sim.model.actuator('grac_r').id]

        had_l = [self.env.sim.model.actuator('addbrev_l').id,
        self.env.sim.model.actuator('addlong_l').id,
        self.env.sim.model.actuator('addmagDist_l').id,
        self.env.sim.model.actuator('addmagIsch_l').id,
        self.env.sim.model.actuator('addmagMid_l').id,
        self.env.sim.model.actuator('addmagProx_l').id,
        self.env.sim.model.actuator('grac_l').id]

        had_r_lbl = ['addbrev_r','addlong_r','addmagDist_r','addmagIsch_r','addmagMid_r','addmagProx_r','grac_r']
        had_l_lbl = ['addbrev_l','addlong_l','addmagDist_l','addmagIsch_l','addmagMid_l','addmagProx_l','grac_l']

        # ----- rectus femoris -----
        rf_r = [self.env.sim.model.actuator('recfem_r').id]

        rf_l = [self.env.sim.model.actuator('recfem_l').id]

        rf_r_lbl = ['recfem_r']
        rf_l_lbl = ['recfem_l']

        # ----- Vastius group -----
        vas_r = [self.env.sim.model.actuator('vasint_r').id,
        self.env.sim.model.actuator('vaslat_r').id,
        self.env.sim.model.actuator('vasmed_r').id]

        vas_l = [self.env.sim.model.actuator('vasint_l').id,
        self.env.sim.model.actuator('vaslat_l').id,
        self.env.sim.model.actuator('vasmed_l').id]

        vas_r_lbl = ['vasint_r','vaslat_r','vasmed_r']
        vas_l_lbl = ['vasint_l','vaslat_l','vasmed_l']

        # ----- tibialis anterior -----
        ta_r = [self.env.sim.model.actuator('tibant_r').id]

        ta_l = [self.env.sim.model.actuator('tibant_l').id]

        ta_r_lbl = ['tibant_r']
        ta_l_lbl = ['tibant_l']

        self.muscles_dict['r_leg'] = {}
        self.muscles_dict['r_leg']['HAB'] = hab_r
        self.muscles_dict['r_leg']['HAD'] = had_r
        self.muscles_dict['r_leg']['GLU'] = glu_r
        self.muscles_dict['r_leg']['HAM'] = ham_r
        self.muscles_dict['r_leg']['BFSH'] = bfsh_r
        self.muscles_dict['r_leg']['GAS'] = gas_r
        self.muscles_dict['r_leg']['SOL'] = sol_r
        self.muscles_dict['r_leg']['HFL'] = hfl_r
        self.muscles_dict['r_leg']['RF'] = rf_r
        self.muscles_dict['r_leg']['VAS'] = vas_r
        self.muscles_dict['r_leg']['TA'] = ta_r

        self.muscles_dict['l_leg'] = {}
        self.muscles_dict['l_leg']['HAB'] = hab_l
        self.muscles_dict['l_leg']['HAD'] = had_l
        self.muscles_dict['l_leg']['GLU'] = glu_l
        self.muscles_dict['l_leg']['HAM'] = ham_l
        self.muscles_dict['l_leg']['BFSH'] = bfsh_l
        self.muscles_dict['l_leg']['GAS'] = gas_l
        self.muscles_dict['l_leg']['SOL'] = sol_l
        self.muscles_dict['l_leg']['HFL'] = hfl_l
        self.muscles_dict['l_leg']['RF'] = rf_l
        self.muscles_dict['l_leg']['VAS'] = vas_l
        self.muscles_dict['l_leg']['TA'] = ta_l

        # Muscle labels
        self.muscle_labels['r_leg'] = {}
        self.muscle_labels['r_leg']['HAB'] = hab_r_lbl
        self.muscle_labels['r_leg']['HAD'] = had_r_lbl
        self.muscle_labels['r_leg']['GLU'] = glu_r_lbl
        self.muscle_labels['r_leg']['HAM'] = ham_r_lbl
        self.muscle_labels['r_leg']['BFSH'] = bfsh_r_lbl
        self.muscle_labels['r_leg']['GAS'] = gas_r_lbl
        self.muscle_labels['r_leg']['SOL'] = sol_r_lbl
        self.muscle_labels['r_leg']['HFL'] = hfl_r_lbl
        self.muscle_labels['r_leg']['RF'] = rf_r_lbl
        self.muscle_labels['r_leg']['VAS'] = vas_r_lbl
        self.muscle_labels['r_leg']['TA'] = ta_r_lbl

        self.muscle_labels['l_leg'] = {}
        self.muscle_labels['l_leg']['HAB'] = hab_l_lbl
        self.muscle_labels['l_leg']['HAD'] = had_l_lbl
        self.muscle_labels['l_leg']['GLU'] = glu_l_lbl
        self.muscle_labels['l_leg']['HAM'] = ham_l_lbl
        self.muscle_labels['l_leg']['BFSH'] = bfsh_l_lbl
        self.muscle_labels['l_leg']['GAS'] = gas_l_lbl
        self.muscle_labels['l_leg']['SOL'] = sol_l_lbl
        self.muscle_labels['l_leg']['HFL'] = hfl_l_lbl
        self.muscle_labels['l_leg']['RF'] = rf_l_lbl
        self.muscle_labels['l_leg']['VAS'] = vas_l_lbl
        self.muscle_labels['l_leg']['TA'] = ta_l_lbl

        #L0 = (actuator_lengthrange)
        temp_L0 = (self.env.sim.model.actuator_lengthrange[:,0] - self.env.sim.model.tendon_lengthspring[:,0]) / self.env.sim.model.actuator_biasprm[:,0]

        # --- Muscle Fmax normalizations ---
        for x in self.muscles_dict:
            self.muscle_Fmax[x] = {}
            self.muscle_L0[x] = {}
            for y in self.muscles_dict[x]:
                self.muscle_Fmax[x][y] = self.env.sim.model.actuator_biasprm[self.muscles_dict[x][y], 2].copy()
                #print(x, ' ', y, ' with', np.sum(self.env.sim.model.actuator_biasprm[self.muscles_dict[x][y],2]))
                self.muscle_L0[x][y] = temp_L0[self.muscles_dict[x][y]]


    def _set_initial_pose(self, init_dict):
        # Sets the initial pose of the Myoleg model based on an input dictionary of values
        
        # Setting the starting position for reward calculation
        self.init_pelvis = self.env.sim.data.body('pelvis').xpos.copy()

        # Converting from Euler to quaternions
        temp_quat_util = euler2quat([init_dict['model_pose']['roll'], 
                                         init_dict['model_pose']['pitch'], 
                                         init_dict['model_pose']['yaw']])

        self.env.sim.data.qpos[3] = temp_quat_util[0] # Setting no roll, pitch and yaw
        self.env.sim.data.qpos[4] = temp_quat_util[1]
        self.env.sim.data.qpos[5] = temp_quat_util[2]
        self.env.sim.data.qpos[6] = temp_quat_util[3]

        # Setting initial velocity
        # Pushes the free root joint, which propagates the velocities to all the joints and segments
        self.env.sim.data.qvel[0] = init_dict['velocity']['cartesian'][0]
        self.env.sim.data.qvel[1] = init_dict['velocity']['cartesian'][1]
        self.env.sim.data.qvel[2] = init_dict['velocity']['cartesian'][2]

        # Reusing the dict from above
        # Values in radians
        for joint_name in init_dict['joint_angles'].keys():
            self.env.sim.data.joint(joint_name).qpos[0] = init_dict['joint_angles'][joint_name]
        
        if 'height_offset' in init_dict.keys():
            height_offset = init_dict['height_offset']
        else:
            height_offset = 0

        # Lowering the height of the model by manipulating the free root joint
        self.env.sim.data.qpos[0] = 0 # X pos of free root joint
        self.env.sim.data.qpos[1] = 0 # Y pos of free root joint
        self.env.sim.data.qpos[2] =  init_dict['model_height'] + height_offset
        
        # From documentation: https://openai.github.io/mujoco-py/build/html/reference.html
        # Run forward() after modifying and joint angles or velocities
        self.env.sim.forward()

    # ---------- Internal functions ----------

    def update_footstep(self):
        
        # Getting only the heel contacts. Works better at detecting new steps, as compared to using both heel and toe
        r_contact = True if (self.env.sim.data.sensor('r_foot').data[0].copy()) > 0.1*(np.sum(self.env.sim.model.body_mass)*9.8) else False
        l_contact = True if (self.env.sim.data.sensor('l_foot').data[0].copy()) > 0.1*(np.sum(self.env.sim.model.body_mass)*9.8) else False

        self.footstep['new'] = False
        if ( (not self.footstep['r_contact'] and r_contact) or (not self.footstep['l_contact'] and l_contact) ):
            self.footstep['new'] = True
            self.footstep['n'] += 1

        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    
    def reflex2mujoco(self, output):
        
        mus_act = np.zeros((80,))
        mus_act[:] = 0 # Using non-normalized values of muscle activations

        legs = ['r_leg', 'l_leg']
        musc_idx = self.muscles_dict['r_leg'].keys()

        for s_leg in legs:
            for musc in musc_idx:
                mus_act[self.muscles_dict[s_leg][musc]] = output[s_leg][musc]
        
        return mus_act

    def rotate_frame(self, x, y, theta):
        x_rot = np.cos(theta)*x - np.sin(theta)*y
        y_rot = np.sin(theta)*x + np.cos(theta)*y
        return x_rot, y_rot