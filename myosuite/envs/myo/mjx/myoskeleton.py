from __future__ import annotations

import os
from typing import Dict, List, Tuple

import mujoco
from decorators import info_property
from mujoco import MjSpec
from observation import Observation, ObservationType

## Adapted from loco_mujoco
## https://github.com/loco-mujoco/loco-mujoco/blob/main/loco_mujoco/environments/myoskeleton_mjx.py


class MyoSkeletonTorque:

    mjx_enabled = False

    def __init__(
        self,
        disable_fingers: bool = True,
        spec: MjSpec = None,
        observation_spec: List[Observation] = None,
        actuation_spec: List[str] = None,
        **kwargs,
    ) -> None:
        """
        Constructor.

        Args:
            disable_fingers (bool): If True, the fingers are disabled.
            spec (Union[str, MjSpec]): Specification of the environment.
                It can be a path to the xml file or a MjSpec object. If none, is provided, the default xml file is used.
            observation_spec (List[Observation]): Observation specification.
            actuation_spec (List[str]): Action specification.
            **kwargs: Additional arguments

        """

        self._disable_fingers = disable_fingers

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # apply changes to the MjSpec
        spec = self._apply_spec_changes(spec)

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        # Store the specification and related attributes
        self.spec = spec
        self.actuation_spec = actuation_spec
        self.observation_spec = observation_spec
        # Store any additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _get_observation_specification(self, spec: MjSpec) -> List[Observation]:
        """
        Getter for the observation space specification. This function reads all joint names from the xml and adds
        the prefix "q_" for the joint positions and "dq_" for the joint velocities. It also adds the free joint
        position (disregarding the x and y position) and velocity.

        Returns:
            List[Observation]: List of observations.

        """
        # get all joint names except the root
        j_names = [
            j.name for j in spec.joints if j.name != self.root_free_joint_xml_name
        ]

        # build observation spec
        observation_spec = []

        # add free joint observation
        observation_spec.append(
            ObservationType.FreeJointPosNoXY(
                "q_free_joint", self.root_free_joint_xml_name
            )
        )

        # add all joint positions
        observation_spec.append(ObservationType.JointPosArray("q_all_pos", j_names))

        # add free joint velocities
        observation_spec.append(
            ObservationType.FreeJointVel("dq_free_joint", self.root_free_joint_xml_name)
        )

        # add all joint velocities
        observation_spec.append(ObservationType.JointVelArray("dq_all_vel", j_names))

        return observation_spec

    def _get_action_specification(self, spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification. This function adds all actuator names found in the spec, which
        are the ones added in the _add_actuators method.

        Returns:
            List[str]: A list of tuples containing the specification of each action
            space entry.

        """
        action_spec = []
        for a in spec.actuators:
            action_spec.append(a.name)
        return action_spec

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        """
        This function reads the original myo_model spec and applies some changes to make it align with LocoMuJoCo.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.

        """

        def get_attributes(obj):
            EXCLUDE = {"id", "signature", "classname", "alt", "frame", "parent"}

            attrs = {}
            for attr in dir(obj):
                # exclude private / internal
                if attr.startswith("_"):
                    continue

                # exclude known invalid fields
                if attr in EXCLUDE:
                    continue

                try:
                    value = getattr(obj, attr)
                except Exception:
                    continue

                if callable(value):
                    continue

                attrs[attr] = value

            return attrs

        # remove floor and add ground plane
        for g in spec.geoms:
            if g.name == "floor":
                g.delete()
                # spec.delete(g)

        # remove old lights
        for b in spec.bodies:
            for l in b.lights:
                l.delete()
                # spec.delete(l)

        # load common specs
        scene_spec = mujoco.MjSpec.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "simhive",
                "myo_model",
                "scene",
                "basic_scene.xml",
            )
        )

        # add all textures, materials, geoms and lights
        for t in scene_spec.textures:
            if t.name not in [tt.name for tt in spec.textures]:
                spec.add_texture(**get_attributes(t))
        for m in scene_spec.materials:
            if m.name not in [mm.name for mm in spec.materials]:
                spec.add_material(**get_attributes(m))
        for g in scene_spec.geoms:
            if g.name not in [gg.name for gg in spec.geoms]:
                spec.worldbody.add_geom(**get_attributes(g))
        for light in scene_spec.lights:
            if light.name not in [ll.name for ll in spec.lights]:
                spec.worldbody.add_light(**get_attributes(light))

        # use default scene visuals
        spec.visual = scene_spec.visual

        # add mimic sites
        for body_name, site_name in self.body2sites_for_mimic.items():
            b = spec.body(body_name)
            pos = [0.0, 0.0, 0.0]
            # todo: can not load mimic sites attributes for now, so I add them manually
            b.add_site(
                name=site_name,
                group=4,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.075, 0.05, 0.025],
                rgba=[1.0, 0.0, 0.0, 0.5],
                pos=pos,
            )

        # add spot light
        for b in spec.bodies:
            if b.name == "pelvis":
                b.add_light(
                    name="spotlight",
                    mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
                    pos=[0, 50, -2],
                    dir=[0, -1, 0],
                )

        if self._disable_fingers:
            for j in spec.joints:
                if "finger" in self.finger_and_hand_joints:
                    j.delete()

        # add actuators
        # spec = self._add_actuators(spec)
        # add motor actuators with specific gear ratios
        spec = self._add_motor_actuators(spec)

        return spec

    def _add_actuators(self, spec: MjSpec) -> MjSpec:
        """
        Adds a generic actuator to each joint.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.

        """
        max_joint_forces = dict(
            L5_S1_Flex_Ext=200,
            L5_S1_Lat_Bending=200,
            L5_S1_axial_rotation=200,
            L4_L5_Flex_Ext=200,
            L4_L5_Lat_Bending=200,
            L4_L5_axial_rotation=200,
            L3_L4_Flex_Ext=200,
            L3_L4_Lat_Bending=200,
            L3_L4_axial_rotation=200,
            L2_L3_Flex_Ext=200,
            L2_L3_Lat_Bending=200,
            L2_L3_axial_rotation=200,
            L1_L2_Flex_Ext=200,
            L1_L2_Lat_Bending=200,
            L1_L2_axial_rotation=200,
            L1_T12_Flex_Ext=200,
            L1_T12_Lat_Bending=200,
            L1_T12_axial_rotation=200,
            c7_c6_FE=50,
            c7_c6_LB=50,
            c7_c6_AR=50,
            c6_c5_FE=50,
            c6_c5_LB=50,
            c6_c5_AR=50,
            c5_c4_FE=50,
            c5_c4_LB=50,
            c5_c4_AR=50,
            c4_c3_FE=50,
            c4_c3_LB=50,
            c4_c3_AR=50,
            c3_c2_FE=50,
            c3_c2_LB=50,
            c3_c2_AR=50,
            c2_c1_FE=50,
            c2_c1_LB=50,
            c2_c1_AR=50,
            c1_skull_FE=50,
            c1_skull_LB=50,
            c1_skull_AR=50,
            skull_FE=50,
            skull_LB=50,
            skull_AR=50,
            sternoclavicular_r2_r=80,
            sternoclavicular_r3_r=80,
            unrotscap_r3_r=80,
            unrotscap_r2_r=80,
            acromioclavicular_r2_r=80,
            acromioclavicular_r3_r=80,
            acromioclavicular_r1_r=80,
            unrothum_r1_r=80,
            unrothum_r3_r=80,
            unrothum_r2_r=80,
            elv_angle_r=80,
            shoulder_elv_r=80,
            shoulder1_r2_r=80,
            shoulder_rot_r=80,
            elbow_flex_r=80,
            pro_sup=80,
            deviation=80,
            flexion_r=80,
            sternoclavicular_r2_l=80,
            sternoclavicular_r3_l=80,
            unrotscap_r3_l=80,
            unrotscap_r2_l=80,
            acromioclavicular_r2_l=80,
            acromioclavicular_r3_l=80,
            acromioclavicular_r1_l=80,
            unrothum_r1_l=80,
            unrothum_r3_l=80,
            unrothum_r2_l=80,
            elv_angle_l=80,
            shoulder_elv_l=80,
            shoulder1_r2_l=80,
            shoulder_rot_l=80,
            elbow_flex_l=80,
            pro_sup_l=80,
            deviation_l=80,
            flexion_l=80,
            hip_flexion_r=200,
            hip_adduction_r=200,
            hip_rotation_r=200,
            knee_angle_r=200,
            knee_angle_r_rotation2=20,
            knee_angle_r_rotation3=20,
            ankle_angle_r=200,
            subtalar_angle_r=200,
            mtp_angle_r=200,
            knee_angle_r_beta_rotation1=20,
            hip_flexion_l=200,
            hip_adduction_l=200,
            hip_rotation_l=200,
            knee_angle_l=200,
            knee_angle_l_rotation2=20,
            knee_angle_l_rotation3=20,
            ankle_angle_l=200,
            subtalar_angle_l=200,
            mtp_angle_l=200,
            knee_angle_l_beta_rotation1=20,
        )

        for joint in spec.joints:
            # add an actuator for every joint except the pelvis
            if self.root_free_joint_xml_name not in joint.name:
                max_force = (
                    max_joint_forces[joint.name]
                    if joint.name in max_joint_forces.keys()
                    else 50
                )
                spec.add_actuator(
                    name="act_" + joint.name,
                    target=joint.name,
                    ctrlrange=[-max_force, max_force],
                    trntype=mujoco.mjtTrn.mjTRN_JOINT,
                    ctrllimited=True,
                )

        return spec

    def _add_motor_actuators(self, spec: MjSpec) -> MjSpec:
        """
        Adds motor actuators with specific gear ratios for key joints.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.
        """
        motor_configs = [
            # Lumbar motors
            ("mot_lumbar_ext", "lumbar_extension", 160),
            ("mot_lumbar_bend", "lumbar_bending", 160),
            ("mot_lumbar_rot", "lumbar_rotation", 100),
            # Right arm motors
            ("mot_shoulder_flex_r", "arm_flex_r", 250),
            ("mot_shoulder_add_r", "arm_add_r", 250),
            ("mot_shoulder_rot_r", "arm_rot_r", 250),
            ("mot_elbow_flex_r", "elbow_flex_r", 250),
            ("mot_pro_sup_r", "pro_sup_r", 250),
            ("mot_wrist_flex_r", "wrist_flex_r", 50),
            ("mot_wrist_dev_r", "wrist_dev_r", 50),
            # Left arm motors
            ("mot_shoulder_flex_l", "arm_flex_l", 250),
            ("mot_shoulder_add_l", "arm_add_l", 250),
            ("mot_shoulder_rot_l", "arm_rot_l", 250),
            ("mot_elbow_flex_l", "elbow_flex_l", 250),
            ("mot_pro_sup_l", "pro_sup_l", 250),
            ("mot_wrist_flex_l", "wrist_flex_l", 50),
            ("mot_wrist_dev_l", "wrist_dev_l", 50),
            # Right leg motors
            ("mot_hip_flexion_r", "hip_flexion_r", 275),
            ("mot_hip_adduction_r", "hip_adduction_r", 530),
            ("mot_hip_rotation_r", "hip_rotation_r", 600),
            ("mot_knee_angle_r", "knee_angle_r", 600),
            ("mot_ankle_angle_r", "ankle_angle_r", 500),
            ("mot_subtalar_angle_r", "subtalar_angle_r", 50),
            ("mot_mtp_angle_r", "mtp_angle_r", 50),
            # Left leg motors
            ("mot_hip_flexion_l", "hip_flexion_l", 275),
            ("mot_hip_adduction_l", "hip_adduction_l", 530),
            ("mot_hip_rotation_l", "hip_rotation_l", 600),
            ("mot_knee_angle_l", "knee_angle_l", 600),
            ("mot_ankle_angle_l", "ankle_angle_l", 500),
            ("mot_subtalar_angle_l", "subtalar_angle_l", 50),
            ("mot_mtp_angle_l", "mtp_angle_l", 50),
        ]

        for motor_name, joint_name, gear in motor_configs:
            # Check if joint exists in the spec
            joint_exists = any(j.name == joint_name for j in spec.joints)

            if joint_exists:
                # Check if actuator already exists
                actuator_exists = any(a.name == motor_name for a in spec.actuators)
                print(joint_name, joint_exists, actuator_exists, motor_name)
                if not actuator_exists:
                    # gear must be a 6-element array for joint actuators
                    gear_array = [0.0] * 6
                    gear_array[0] = gear  # Set gear for the first DOF
                    spec.add_actuator(
                        name=motor_name,
                        target=joint_name,
                        trntype=mujoco.mjtTrn.mjTRN_JOINT,
                        gear=gear_array,
                    )

        return spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.

        """

        return os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "simhive",
            "myo_model",
            "myoskeleton",
            "myoskeleton.xml",
        )

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco xml file.

        """
        return "thoracic_spine"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint in the Mujoco xml file.

        """
        return "myoskeleton_root"

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco xml file.

        """
        return "myoskeleton_root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.

        """
        return (0.6, 1.5)

    @info_property
    def body2sites_for_mimic(self) -> Dict[str, str]:
        """
        Returns a dictionary that maps body names to mimic site names.

        Returns:
            Dict[str, str]: Mapping from body names to mimic site names.

        """
        body2sitemimic = {
            "thoracic_spine": "upper_body_mimic",
            "skull": "head_mimic",  # Adding head mimic (likely attached to the skull)
            "pelvis": "pelvis_mimic",
            "humerus_l": "left_shoulder_mimic",
            "ulna_l": "left_elbow_mimic",
            "lunate_l": "left_hand_mimic",
            "femur_l": "left_hip_mimic",
            "tibia_l": "left_knee_mimic",
            "calcn_l": "left_foot_mimic",
            "humerus_r": "right_shoulder_mimic",
            "ulna_r": "right_elbow_mimic",
            "lunate_r": "right_hand_mimic",
            "femur_r": "right_hip_mimic",
            "tibia_r": "right_knee_mimic",
            "calcn_r": "right_foot_mimic",
        }

        return body2sitemimic

    @info_property
    def finger_and_hand_joints(self) -> List[str]:
        """
        Returns the names of the finger and hand joints.

        Returns:
            List[str]: List of finger and hand joint names.

        """
        finger_hand_joints = [
            # Thumb (Right)
            "cmc_flexion_r",
            "cmc_abduction_r",
            "mp_flexion_r",
            "ip_flexion_r",
            # Index Finger (Right)
            "mcp2_flexion_r",
            "mcp2_abduction_r",
            "pm2_flexion_r",
            "md2_flexion_r",
            # Middle Finger (Right)
            "mcp3_flexion_r",
            "mcp3_abduction_r",
            "pm3_flexion_r",
            "md3_flexion_r",
            # Ring Finger (Right)
            "mcp4_flexion_r",
            "mcp4_abduction_r",
            "pm4_flexion_r",
            "md4_flexion_r",
            # Little Finger (Right)
            "mcp5_flexion_r",
            "mcp5_abduction_r",
            "pm5_flexion_r",
            "md5_flexion_r",
            # Thumb (Left)
            "cmc_flexion_l",
            "cmc_abduction_l",
            "mp_flexion_l",
            "ip_flexion_l",
            # Index Finger (Left)
            "mcp2_flexion_l",
            "mcp2_abduction_l",
            "pm2_flexion_l",
            "md2_flexion_l",
            # Middle Finger (Left)
            "mcp3_flexion_l",
            "mcp3_abduction_l",
            "pm3_flexion_l",
            "md3_flexion_l",
            # Ring Finger (Left)
            "mcp4_flexion_l",
            "mcp4_abduction_l",
            "pm4_flexion_l",
            "md4_flexion_l",
            # Little Finger (Left)
            "mcp5_flexion_l",
            "mcp5_abduction_l",
            "pm5_flexion_l",
            "md5_flexion_l",
        ]

        return finger_hand_joints

    @info_property
    def sites_for_mimic(self) -> List[str]:
        """
        Returns a list of all mimic sites.

        """
        return list(self.body2sites_for_mimic.values())

    @info_property
    def goal_visualization_arrow_offset(self) -> List[float]:
        """
        Returns the offset for the goal visualization arrow.

        """
        return [0, 0, 0.4]


if __name__ == "__main__":
    env = MyoSkeletonTorque()
    model = env.spec.compile()

    with open("myoskeleton_edited.xml", "w") as f:
        f.write(env.spec.to_xml())

    mujoco.mj_saveModel(model, "myoskeleton_edited.mjb")
