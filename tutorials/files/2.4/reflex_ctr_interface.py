# Author(s): Seungmoon Song <seungmoon.song@gmail.com>, Chun Kwang Tan <riodren.tan@gmail.com>
"""
adapted from:
- Song and Geyer. "A neural circuitry that emphasizes
spinal feedback generates diverse behaviours of human locomotion." The
Journal of physiology, 2015.
"""

import numpy as np
from reflex_ctr import LocoCtrl

import mujoco

from myosuite.utils import gym
from myosuite.physics.quat_math import euler2quat, quat2mat


class MyoLegReflex:
    DEFAULT_INIT_POSE = {}
    DEFAULT_INIT_POSE["model_pose"] = {
        "yaw": np.deg2rad(0),
        "pitch": np.deg2rad(0),
        "roll": np.deg2rad(0),
    }
    DEFAULT_INIT_POSE["model_height"] = 0.92
    DEFAULT_INIT_POSE["joint_angles"] = {
        "hip_flexion_r": np.deg2rad(180 - 190),
        "hip_flexion_l": np.deg2rad(180 - 155),
        "knee_angle_r": np.deg2rad(180 - 165),
        "knee_angle_l": np.deg2rad(180 - 180),
        "ankle_angle_r": np.deg2rad(90 - 90),
        "ankle_angle_l": np.deg2rad(90 - 100),
    }
    DEFAULT_INIT_POSE["velocity"] = {"cartesian": [1.5, 0.0, 0.0]}

    def __init__(
        self,
        init_dict=None,
        dt=0.01,
        mode="3D",
        sim_time=5.0,
        seed=0,
        render_mode="rgb_array",
    ):
        self.t = 0
        self.mode = mode
        self.sim_time = sim_time
        self.init_dict = self.DEFAULT_INIT_POSE if init_dict is None else init_dict
        self.seed = seed

        self.env = gym.make(
            "myoLegWalk-v0",
            normalize_act=False,
            reset_type="init",
            render_mode=render_mode,
        )
        self.dt = float(getattr(self.env.unwrapped, "_ctrl_dt", dt))
        self.timestep_limit = int(self.sim_time / self.dt)

        self.n_par = len(LocoCtrl.cp_keys)
        self.cp_map = LocoCtrl.cp_map
        self.ReflexCtrl = LocoCtrl(
            self.dt, control_dimension=3, params=np.ones(self.n_par)
        )

        print("Seed added - ", seed)
        self.env.reset(seed=seed)

        self.muscle_labels = {}
        self.muscles_dict = {}
        self.muscle_Fmax = {}
        self.muscle_L0 = {}

        self.init_pelvis = np.zeros(
            3,
        )  # Variable to hold the initial position of the pelvis (for reward calculations)

        self.footstep = {}
        self.footstep["n"] = 0
        self.footstep["new"] = False
        self.footstep["r_contact"] = 0
        self.footstep["l_contact"] = 0

        # Accessor for LocoCtrl
        self.cp = self.ReflexCtrl.cp

    # -----------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.env.reset(seed=self.seed)

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

    def _foot_load(self, sensor_names: tuple[str, ...], body_names: tuple[str, ...]) -> float:
        """Vertical support load from touch sensors, or contact forces if sensors are absent."""
        data = self.env.unwrapped.data
        model = self.env.unwrapped.model
        total = 0.0
        found = False
        for name in sensor_names:
            try:
                total += float(data.sensor(name).data[0])
                found = True
            except KeyError:
                continue
        if found:
            return total
        body_ids = set()
        for name in body_names:
            try:
                body_ids.add(int(model.body(name).id))
            except KeyError:
                continue
        cforce = np.zeros(6, dtype=np.float64)
        for i in range(int(data.ncon)):
            con = data.contact[i]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if b1 in body_ids or b2 in body_ids:
                mujoco.mj_contactForce(model, data, i, cforce)
                total += abs(float(cforce[0]))
        return total

    def _root_pose_vel(self):
        """Root attitude and velocity in the controller frame (x forward, y left, z up)."""
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data
        xmat = np.asarray(data.body("root").xmat, dtype=np.float64).reshape(3, 3)
        fwd = xmat[:, 0]
        left = xmat[:, 1]
        up = xmat[:, 2]
        pitch = float(np.arctan2(-fwd[2], np.hypot(fwd[0], fwd[1])))
        roll = float(np.arctan2(left[2], up[2]))
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.body("root").id),
            vel,
            0,
        )
        lin_body = xmat.T @ vel[3:6]
        ang_body = xmat.T @ vel[0:3]
        return roll, pitch, lin_body, ang_body

    def get_obs_dict(self):
        # Function translate Myosuite joint angle conventions into the conventions used by the reflex controller
        # refer to LocoCtrl.s_b_keys and LocoCtrl.s_l_keys
        # coordinate in body frame
        #   [0] x: forward
        #   [1] y: leftward
        #   [2] z: upward

        pelvis_roll, pelvis_pitch, lin_body, ang_body = self._root_pose_vel()

        temp_right = self._foot_load(("r_foot", "r_toes"), ("calcn_r", "toes_r"))
        temp_left = self._foot_load(("l_foot", "l_toes"), ("calcn_l", "toes_l"))

        sensor_data = {"body": {}, "r_leg": {}, "l_leg": {}}
        sensor_data["body"]["theta"] = [
            pelvis_roll,  # around local x axis
            pelvis_pitch,
        ]  # around local y axis

        sensor_data["body"]["d_pos"] = [
            float(lin_body[0]),  # local x (+) forward
            float(lin_body[1]),
        ]  # local y (+) leftward

        sensor_data["body"]["dtheta"] = [
            float(ang_body[0]),  # around local x axis
            float(ang_body[1]),
        ]  # around local y axis

        sensor_data["r_leg"]["load_ipsi"] = temp_right / (
            np.sum(self.env.unwrapped.model.body_mass) * 9.8
        )
        sensor_data["l_leg"]["load_ipsi"] = temp_left / (
            np.sum(self.env.unwrapped.model.body_mass) * 9.8
        )

        for s_leg, s_legc in zip(["r_leg", "l_leg"], ["l_leg", "r_leg"]):
            sensor_data[s_leg]["contact_ipsi"] = (
                1 if sensor_data[s_leg]["load_ipsi"] > 0.1 else 0
            )
            sensor_data[s_leg]["contact_contra"] = (
                1 if sensor_data[s_legc]["load_ipsi"] > 0.1 else 0
            )
            sensor_data[s_leg]["load_contra"] = sensor_data[s_legc]["load_ipsi"]

            sensor_data[s_leg]["phi_hip"] = (
                np.pi
                - self.env.unwrapped.data.jnt(f"hip_flexion_{s_leg[0]}").qpos[0].copy()
            )
            sensor_data[s_leg]["phi_knee"] = (
                np.pi
                - self.env.unwrapped.data.jnt(f"knee_angle_{s_leg[0]}").qpos[0].copy()
            )
            sensor_data[s_leg]["phi_ankle"] = (
                0.5 * np.pi
                - self.env.unwrapped.data.jnt(f"ankle_angle_{s_leg[0]}").qpos[0].copy()
            )
            sensor_data[s_leg]["dphi_knee"] = (
                self.env.unwrapped.data.jnt(f"knee_angle_{s_leg[0]}").qvel[0].copy()
            )

            # alpha = hip - 0.5*knee
            sensor_data[s_leg]["alpha"] = (
                sensor_data[s_leg]["phi_hip"] - 0.5 * sensor_data[s_leg]["phi_knee"]
            )
            dphi_hip = (
                self.env.unwrapped.data.jnt(f"hip_flexion_{s_leg[0]}").qvel[0].copy()
            )
            sensor_data[s_leg]["dalpha"] = (
                dphi_hip - 0.5 * sensor_data[s_leg]["dphi_knee"]
            )

            sensor_data[s_leg]["alpha_f"] = (
                -1
                * self.env.unwrapped.data.jnt(f"hip_adduction_{s_leg[0]}")
                .qpos[0]
                .copy()
            ) + 0.5 * np.pi

            temp_mus_force = self.env.unwrapped.data.actuator_force.copy()

            sensor_data[s_leg]["F_RF"] = -1 * np.mean(
                temp_mus_force[self.muscles_dict[s_leg]["RF"]]
                / (self.muscle_Fmax[s_leg]["RF"])
            )
            sensor_data[s_leg]["F_VAS"] = -1 * np.mean(
                temp_mus_force[self.muscles_dict[s_leg]["VAS"]]
                / (self.muscle_Fmax[s_leg]["VAS"])
            )
            sensor_data[s_leg]["F_GAS"] = -1 * np.mean(
                temp_mus_force[self.muscles_dict[s_leg]["GAS"]]
                / (self.muscle_Fmax[s_leg]["GAS"])
            )
            sensor_data[s_leg]["F_SOL"] = -1 * np.mean(
                temp_mus_force[self.muscles_dict[s_leg]["SOL"]]
                / (self.muscle_Fmax[s_leg]["SOL"])
            )

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

        data = self.env.unwrapped.data
        _, pitch, _, _ = self._root_pose_vel()
        if data.body("pelvis").xpos[2] < 0.65:
            is_done = True
        if abs(pitch) > np.deg2rad(30):
            is_done = True

        return [out_dict, is_done, np.round(data.time, 2), new_act]

    # ---------- Initialization Functions ----------
    def _set_muscle_groups(self):
        # ----- Gluteus group -----
        glu_r = [
            self.env.unwrapped.model.actuator("glmax1_r").id,
            self.env.unwrapped.model.actuator("glmax2_r").id,
            self.env.unwrapped.model.actuator("glmax3_r").id,
            self.env.unwrapped.model.actuator("glmed3_r").id,
        ]

        glu_l = [
            self.env.unwrapped.model.actuator("glmax1_l").id,
            self.env.unwrapped.model.actuator("glmax2_l").id,
            self.env.unwrapped.model.actuator("glmax3_l").id,
            self.env.unwrapped.model.actuator("glmed3_l").id,
        ]

        glu_r_lbl = ["glmax1_r", "glmax2_r", "glmax3_r", "glmed3_r"]
        glu_l_lbl = ["glmax1_l", "glmax2_l", "glmax3_l", "glmed3_l"]

        # ----- Hamstring (semitendinosus and semimembranosus) -----
        ham_r = [
            self.env.unwrapped.model.actuator("semimem_r").id,
            self.env.unwrapped.model.actuator("semiten_r").id,
            self.env.unwrapped.model.actuator("bflh_r").id,
        ]

        ham_l = [
            self.env.unwrapped.model.actuator("semimem_l").id,
            self.env.unwrapped.model.actuator("semiten_l").id,
            self.env.unwrapped.model.actuator("bflh_l").id,
        ]

        ham_r_lbl = ["semimem_r", "semiten_r", "bflh_r"]
        ham_l_lbl = ["semimem_l", "semiten_l", "bflh_l"]

        # ----- BF short head (biceps femoris) -----
        bfsh_r = [self.env.unwrapped.model.actuator("bfsh_r").id]

        bfsh_l = [self.env.unwrapped.model.actuator("bfsh_l").id]

        bfsh_r_lbl = ["bfsh_r"]
        bfsh_l_lbl = ["bfsh_l"]

        # ----- Gastrocnemius -----
        gas_r = [
            self.env.unwrapped.model.actuator("gaslat_r").id,
            self.env.unwrapped.model.actuator("gasmed_r").id,
        ]

        gas_l = [
            self.env.unwrapped.model.actuator("gaslat_l").id,
            self.env.unwrapped.model.actuator("gasmed_l").id,
        ]

        gas_r_lbl = ["gaslat_r", "gasmed_r"]
        gas_l_lbl = ["gaslat_l", "gasmed_l"]

        # ----- Soleus -----
        sol_r = [
            self.env.unwrapped.model.actuator("soleus_r").id,
            self.env.unwrapped.model.actuator("perbrev_r").id,
            self.env.unwrapped.model.actuator("perlong_r").id,
            self.env.unwrapped.model.actuator("tibpost_r").id,
        ]

        sol_l = [
            self.env.unwrapped.model.actuator("soleus_l").id,
            self.env.unwrapped.model.actuator("perbrev_l").id,
            self.env.unwrapped.model.actuator("perlong_l").id,
            self.env.unwrapped.model.actuator("tibpost_l").id,
        ]

        sol_r_lbl = ["soleus_r", "perbrev_r", "perlong_r", "tibpost_r"]
        sol_l_lbl = ["soleus_l", "perbrev_l", "perlong_l", "tibpost_l"]

        # ----- Hip Flexors (psoas and iliacus) -----
        hfl_r = [
            self.env.unwrapped.model.actuator("psoas_r").id,
            self.env.unwrapped.model.actuator("iliacus_r").id,
        ]

        hfl_l = [
            self.env.unwrapped.model.actuator("psoas_l").id,
            self.env.unwrapped.model.actuator("iliacus_l").id,
        ]

        hfl_r_lbl = ["psoas_r", "iliacus_r"]
        hfl_l_lbl = ["psoas_l", "iliacus_l"]

        # ----- Hip Abductors (piriformis, satorius and tensor fasciae latae) -----
        hab_r = [
            self.env.unwrapped.model.actuator("piri_r").id,
            self.env.unwrapped.model.actuator("sart_r").id,
            self.env.unwrapped.model.actuator("glmed1_r").id,
            self.env.unwrapped.model.actuator("glmed2_r").id,
            self.env.unwrapped.model.actuator("glmin1_r").id,
            self.env.unwrapped.model.actuator("glmin2_r").id,
            self.env.unwrapped.model.actuator("glmin3_r").id,
        ]

        hab_l = [
            self.env.unwrapped.model.actuator("piri_l").id,
            self.env.unwrapped.model.actuator("sart_l").id,
            self.env.unwrapped.model.actuator("glmed1_l").id,
            self.env.unwrapped.model.actuator("glmed2_l").id,
            self.env.unwrapped.model.actuator("glmin1_l").id,
            self.env.unwrapped.model.actuator("glmin2_l").id,
            self.env.unwrapped.model.actuator("glmin3_l").id,
        ]

        hab_r_lbl = [
            "piri_r",
            "sart_r",
            "glmed1_r",
            "glmed2_r",
            "glmin1_r",
            "glmin2_r",
            "glmin3_r",
        ]
        hab_l_lbl = [
            "piri_l",
            "sart_l",
            "glmed1_l",
            "glmed2_l",
            "glmin1_l",
            "glmin2_l",
            "glmin3_l",
        ]

        # ----- Hip Abbuctors (adductor [brevis, longus, magnus], gracilis) -----
        had_r = [
            self.env.unwrapped.model.actuator("addbrev_r").id,
            self.env.unwrapped.model.actuator("addlong_r").id,
            self.env.unwrapped.model.actuator("addmagDist_r").id,
            self.env.unwrapped.model.actuator("addmagIsch_r").id,
            self.env.unwrapped.model.actuator("addmagMid_r").id,
            self.env.unwrapped.model.actuator("addmagProx_r").id,
            self.env.unwrapped.model.actuator("grac_r").id,
        ]

        had_l = [
            self.env.unwrapped.model.actuator("addbrev_l").id,
            self.env.unwrapped.model.actuator("addlong_l").id,
            self.env.unwrapped.model.actuator("addmagDist_l").id,
            self.env.unwrapped.model.actuator("addmagIsch_l").id,
            self.env.unwrapped.model.actuator("addmagMid_l").id,
            self.env.unwrapped.model.actuator("addmagProx_l").id,
            self.env.unwrapped.model.actuator("grac_l").id,
        ]

        had_r_lbl = [
            "addbrev_r",
            "addlong_r",
            "addmagDist_r",
            "addmagIsch_r",
            "addmagMid_r",
            "addmagProx_r",
            "grac_r",
        ]
        had_l_lbl = [
            "addbrev_l",
            "addlong_l",
            "addmagDist_l",
            "addmagIsch_l",
            "addmagMid_l",
            "addmagProx_l",
            "grac_l",
        ]

        # ----- rectus femoris -----
        rf_r = [self.env.unwrapped.model.actuator("recfem_r").id]

        rf_l = [self.env.unwrapped.model.actuator("recfem_l").id]

        rf_r_lbl = ["recfem_r"]
        rf_l_lbl = ["recfem_l"]

        # ----- Vastius group -----
        vas_r = [
            self.env.unwrapped.model.actuator("vasint_r").id,
            self.env.unwrapped.model.actuator("vaslat_r").id,
            self.env.unwrapped.model.actuator("vasmed_r").id,
        ]

        vas_l = [
            self.env.unwrapped.model.actuator("vasint_l").id,
            self.env.unwrapped.model.actuator("vaslat_l").id,
            self.env.unwrapped.model.actuator("vasmed_l").id,
        ]

        vas_r_lbl = ["vasint_r", "vaslat_r", "vasmed_r"]
        vas_l_lbl = ["vasint_l", "vaslat_l", "vasmed_l"]

        # ----- tibialis anterior -----
        ta_r = [self.env.unwrapped.model.actuator("tibant_r").id]

        ta_l = [self.env.unwrapped.model.actuator("tibant_l").id]

        ta_r_lbl = ["tibant_r"]
        ta_l_lbl = ["tibant_l"]

        self.muscles_dict["r_leg"] = {}
        self.muscles_dict["r_leg"]["HAB"] = hab_r
        self.muscles_dict["r_leg"]["HAD"] = had_r
        self.muscles_dict["r_leg"]["GLU"] = glu_r
        self.muscles_dict["r_leg"]["HAM"] = ham_r
        self.muscles_dict["r_leg"]["BFSH"] = bfsh_r
        self.muscles_dict["r_leg"]["GAS"] = gas_r
        self.muscles_dict["r_leg"]["SOL"] = sol_r
        self.muscles_dict["r_leg"]["HFL"] = hfl_r
        self.muscles_dict["r_leg"]["RF"] = rf_r
        self.muscles_dict["r_leg"]["VAS"] = vas_r
        self.muscles_dict["r_leg"]["TA"] = ta_r

        self.muscles_dict["l_leg"] = {}
        self.muscles_dict["l_leg"]["HAB"] = hab_l
        self.muscles_dict["l_leg"]["HAD"] = had_l
        self.muscles_dict["l_leg"]["GLU"] = glu_l
        self.muscles_dict["l_leg"]["HAM"] = ham_l
        self.muscles_dict["l_leg"]["BFSH"] = bfsh_l
        self.muscles_dict["l_leg"]["GAS"] = gas_l
        self.muscles_dict["l_leg"]["SOL"] = sol_l
        self.muscles_dict["l_leg"]["HFL"] = hfl_l
        self.muscles_dict["l_leg"]["RF"] = rf_l
        self.muscles_dict["l_leg"]["VAS"] = vas_l
        self.muscles_dict["l_leg"]["TA"] = ta_l

        # Muscle labels
        self.muscle_labels["r_leg"] = {}
        self.muscle_labels["r_leg"]["HAB"] = hab_r_lbl
        self.muscle_labels["r_leg"]["HAD"] = had_r_lbl
        self.muscle_labels["r_leg"]["GLU"] = glu_r_lbl
        self.muscle_labels["r_leg"]["HAM"] = ham_r_lbl
        self.muscle_labels["r_leg"]["BFSH"] = bfsh_r_lbl
        self.muscle_labels["r_leg"]["GAS"] = gas_r_lbl
        self.muscle_labels["r_leg"]["SOL"] = sol_r_lbl
        self.muscle_labels["r_leg"]["HFL"] = hfl_r_lbl
        self.muscle_labels["r_leg"]["RF"] = rf_r_lbl
        self.muscle_labels["r_leg"]["VAS"] = vas_r_lbl
        self.muscle_labels["r_leg"]["TA"] = ta_r_lbl

        self.muscle_labels["l_leg"] = {}
        self.muscle_labels["l_leg"]["HAB"] = hab_l_lbl
        self.muscle_labels["l_leg"]["HAD"] = had_l_lbl
        self.muscle_labels["l_leg"]["GLU"] = glu_l_lbl
        self.muscle_labels["l_leg"]["HAM"] = ham_l_lbl
        self.muscle_labels["l_leg"]["BFSH"] = bfsh_l_lbl
        self.muscle_labels["l_leg"]["GAS"] = gas_l_lbl
        self.muscle_labels["l_leg"]["SOL"] = sol_l_lbl
        self.muscle_labels["l_leg"]["HFL"] = hfl_l_lbl
        self.muscle_labels["l_leg"]["RF"] = rf_l_lbl
        self.muscle_labels["l_leg"]["VAS"] = vas_l_lbl
        self.muscle_labels["l_leg"]["TA"] = ta_l_lbl

        # L0 = (actuator_lengthrange)
        temp_L0 = (
            self.env.unwrapped.model.actuator_lengthrange[:, 0]
            - self.env.unwrapped.model.tendon_lengthspring[:, 0]
        ) / self.env.unwrapped.model.actuator_biasprm[:, 0]

        # --- Muscle Fmax normalizations ---
        for x in self.muscles_dict:
            self.muscle_Fmax[x] = {}
            self.muscle_L0[x] = {}
            for y in self.muscles_dict[x]:
                self.muscle_Fmax[x][y] = self.env.unwrapped.model.actuator_biasprm[
                    self.muscles_dict[x][y], 2
                ].copy()
                # print(x, ' ', y, ' with', np.sum(self.env.unwrapped.model.actuator_biasprm[self.muscles_dict[x][y],2]))
                self.muscle_L0[x][y] = temp_L0[self.muscles_dict[x][y]]

    def _set_initial_pose(self, init_dict):
        # Preserve the model's keyframe heading (MyoLeg faces -Y in world) and
        # express cartesian velocity in that frame. Replacing the freejoint
        # quaternion with a from-scratch euler used to point the walker +X and
        # it fell immediately.
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data
        self.init_pelvis = data.body("pelvis").xpos.copy()

        heading = np.asarray(data.qpos[3:7], dtype=np.float64).copy()
        extra_euler = [
            init_dict["model_pose"]["roll"],
            init_dict["model_pose"]["pitch"],
            init_dict["model_pose"]["yaw"],
        ]
        if np.any(np.abs(extra_euler) > 1e-8):
            composed = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(composed, heading, euler2quat(extra_euler))
            data.qpos[3:7] = composed

        cart = np.asarray(init_dict["velocity"]["cartesian"], dtype=np.float64)
        data.qvel[0:3] = quat2mat(heading) @ cart

        for joint_name, angle in init_dict["joint_angles"].items():
            data.joint(joint_name).qpos[0] = angle

        height_offset = init_dict.get("height_offset", 0)
        data.qpos[0] = 0
        data.qpos[1] = 0
        data.qpos[2] = init_dict["model_height"] + height_offset

        mujoco.mj_forward(model, data)

    # ---------- Internal functions ----------

    def update_footstep(self):
        weight = np.sum(self.env.unwrapped.model.body_mass) * 9.8
        r_contact = self._foot_load(("r_foot",), ("calcn_r",)) > 0.1 * weight
        l_contact = self._foot_load(("l_foot",), ("calcn_l",)) > 0.1 * weight

        self.footstep["new"] = False
        if (not self.footstep["r_contact"] and r_contact) or (
            not self.footstep["l_contact"] and l_contact
        ):
            self.footstep["new"] = True
            self.footstep["n"] += 1

        self.footstep["r_contact"] = r_contact
        self.footstep["l_contact"] = l_contact

    def reflex2mujoco(self, output):
        n_act = int(self.env.action_space.shape[0])
        mus_act = np.zeros(n_act)

        legs = ["r_leg", "l_leg"]
        musc_idx = self.muscles_dict["r_leg"].keys()

        for s_leg in legs:
            for musc in musc_idx:
                mus_act[self.muscles_dict[s_leg][musc]] = output[s_leg][musc]

        return mus_act

    def rotate_frame(self, x, y, theta):
        x_rot = np.cos(theta) * x - np.sin(theta) * y
        y_rot = np.sin(theta) * x + np.cos(theta) * y
        return x_rot, y_rot
