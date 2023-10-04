"""
Script to process reference data into reference format
reference = collections.namedtuple('reference',
        ['time',        # int
         'robot',       # shape(N, n_robot_jnt) ==> robot trajectory
         'object',      # shape(M, n_objects_jnt) ==> object trajectory
         'robot_init',  # shape(n_objects_jnt) ==> initial robot pose
         'object_init'  # shape(n_objects_jnt) ==> initial object
         ])
"""
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))


sim_dt = 0.002
n_robot_jnt = 29

def load(reference_path):
    return {k:v for k, v in np.load(reference_path).items()}

def proces_n_save(data, new_path_name):

    # additional_resets :	 (3,)
    # human_joint_coords :	 (100, 16, 3)
    # human_obj_contact :	 (100,)

    # object_translation :	 (78, 3)
    # object_orientation :	 (78, 4)
    # s :	 (100, 35)
    # sdot :	 (100, 35)
    # eef_pos :	 (100, 16, 3)
    # eef_quat :	 (100, 16, 4)
    # eef_velp :	 (100, 16, 3)
    # eef_velr :	 (100, 16, 3)
    # s_0_dexman :	 (35,)
    # object_translation_dexman :	 (85, 3)
    # object_orientation_dexman :	 (85, 4)
    # length_dexman :	 ()
    # SIM_SUBSTEPS_dexman :	 ()
    # DATA_SUBSTEPS_dexman :	 ()
    # myo_s0_dexman :	 (26,)
    # object_translation_original :	 (100, 3)
    # object_orientation_original :	 (100, 4)

    # # prgrasp positions
    # s_pg-5 :	 (35,)
    # s_pg :	 (35,)
    # s_pg1 :	 (35,)
    # s_g :	     (35,)
    # s_pg10 :	 (35,)
    # s_g5 :	 (35,)
    # s_g10 :	 (35,)
    # s_0 :	     (35,)

    # print(f"Motion length = {data['length']}")


    # robot details
    robot = data['s'][:,:n_robot_jnt].copy()
    robot[:,0] = -1.0*data['s'][:,0]
    robot[:,1] = data['s'][:,2]
    robot[:,2] = data['s'][:,1]
    robot[:,3] = data['s'][:,3]
    robot[:,4] = data['s'][:,5]
    robot[:,5] = data['s'][:,4]

    # Object details
    data_obj = np.concatenate([data['object_translation_original'], data['object_orientation_original']], axis=1)
    obj = data['s'][:,n_robot_jnt:]

    # time details
    horizon = robot.shape[0]
    time = np.arange(0, horizon) * data['SUBSTEPS']*sim_dt

    np.savez(new_path_name,
             time=time,
             robot=robot,
             object=data_obj,
             robot_init=robot[0],
             object_init=obj[0],
             )

file_names = [
    ('airplane_fly1_myo.npz', 'MyoHand_airplane_fly1.npz'),
    ('airplane_lift_myo.npz', 'MyoHand_airplane_lift.npz'),
    ('airplane_pass1_myo.npz', 'MyoHand_airplane_pass1.npz'),
    ('alarmclock_lift_myo.npz', 'MyoHand_alarmclock_lift.npz'),
    ('alarmclock_pass1_myo.npz', 'MyoHand_alarmclock_pass1.npz'),
    ('alarmclock_see1_myo.npz', 'MyoHand_alarmclock_see1.npz'),
    ('apple_lift_myo.npz', 'MyoHand_apple_lift.npz'),
    ('apple_pass1_myo.npz', 'MyoHand_apple_pass1.npz'),
    ('banana_pass1_myo.npz', 'MyoHand_banana_pass1.npz'),
    ('binoculars_pass1_myo.npz', 'MyoHand_binoculars_pass1.npz'),

    # ('bowl_drink2_myo.npz', 'MyoHand_bowl_drink2.npz'),
    # ('bowl_pass1_myo.npz', 'MyoHand_bowl_pass1.npz'),
    # ('camera_pass1_myo.npz', 'MyoHand_camera_pass1.npz'),
    # ('cubelarge_pass1_myo.npz', 'MyoHand_cubelarge_pass1.npz'),
    # ('cubemedium_inspect1_myo.npz', 'MyoHand_cubemedium_inspect1.npz'),
    # ('cubesmall_lift_myo.npz', 'MyoHand_cubesmall_lift.npz'),
    # ('cubesmall_pass1_myo.npz', 'MyoHand_cubesmall_pass1.npz'),

    ('cup_drink1_myo.npz', 'MyoHand_cup_drink1.npz'),
    ('cup_pass1_myo.npz', 'MyoHand_cup_pass1.npz'),
    ('cup_pour1_myo.npz', 'MyoHand_cup_pour1.npz'),

    # ('cylindermedium_lift_myo.npz', 'MyoHand_cylindermedium_lift.npz'),
    # ('cylindermedium_pass1_myo.npz', 'MyoHand_cylindermedium_pass1.npz'),
    # ('cylindersmall_inspect1_myo.npz', 'MyoHand_cylindersmall_inspect1.npz'),
    # ('cylindersmall_lift_myo.npz', 'MyoHand_cylindersmall_lift.npz'),
    # ('cylindersmall_pass1_myo.npz', 'MyoHand_cylindersmall_pass1.npz'),

    ('duck_inspect1_myo.npz', 'MyoHand_duck_inspect1.npz'),
    ('duck_lift_myo.npz', 'MyoHand_duck_lift.npz'),
    ('duck_pass1_myo.npz', 'MyoHand_duck_pass1.npz'),

    ('elephant_lift_myo.npz', 'MyoHand_elephant_lift.npz'),
    ('elephant_pass1_myo.npz', 'MyoHand_elephant_pass1.npz'),
    ('eyeglasses_pass1_myo.npz', 'MyoHand_eyeglasses_pass1.npz'),
    ('flashlight_lift_myo.npz', 'MyoHand_flashlight_lift.npz'),
    ('flashlight_on1_myo.npz', 'MyoHand_flashlight_on1.npz'),
    ('flashlight_on2_myo.npz', 'MyoHand_flashlight_on2.npz'),
    ('flashlight_pass1_myo.npz', 'MyoHand_flashlight_pass1.npz'),

    ('flute_pass1_myo.npz', 'MyoHand_flute_pass1.npz'),
    ('fryingpan_cook2_myo.npz', 'MyoHand_fryingpan_cook2.npz'),
    ('hammer_pass1_myo.npz', 'MyoHand_hammer_pass1.npz'),
    ('hammer_use1_myo.npz', 'MyoHand_hammer_use1.npz'),
    ('hand_inspect1_myo.npz', 'MyoHand_hand_inspect1.npz'),
    ('hand_pass1_myo.npz', 'MyoHand_hand_pass1.npz'),
    ('headphones_pass1_myo.npz', 'MyoHand_headphones_pass1.npz'),

    ('knife_chop1_myo.npz', 'MyoHand_knife_chop1.npz'),
    ('knife_lift_myo.npz', 'MyoHand_knife_lift.npz'),

    ('lightbulb_pass1_myo.npz', 'MyoHand_lightbulb_pass1.npz'),
    ('mouse_lift_myo.npz', 'MyoHand_mouse_lift.npz'),
    ('mouse_pass1_myo.npz', 'MyoHand_mouse_pass1.npz'),
    ('mouse_use1_myo.npz', 'MyoHand_mouse_use1.npz'),
    ('mug_drink3_myo.npz', 'MyoHand_mug_drink3.npz'),
    ('mug_lift_myo.npz', 'MyoHand_mug_lift.npz'),
    ('mug_pass1_myo.npz', 'MyoHand_mug_pass1.npz'),
    ('phone_lift_myo.npz', 'MyoHand_phone_lift.npz'),
    ('piggybank_pass1_myo.npz', 'MyoHand_piggybank_pass1.npz'),
    ('piggybank_use1_myo.npz', 'MyoHand_piggybank_use1.npz'),
    ('pyramidlarge_pass1_myo.npz', 'MyoHand_pyramidlarge_pass1.npz'),
    ('pyramidmedium_pass1_myo.npz', 'MyoHand_pyramidmedium_pass1.npz'),
    ('pyramidsmall_inspect1_myo.npz', 'MyoHand_pyramidsmall_inspect1.npz'),
    ('scissors_use1_myo.npz', 'MyoHand_scissors_use1.npz'),
    ('spherelarge_pass1_myo.npz', 'MyoHand_spherelarge_pass1.npz'),
    ('spheremedium_inspect1_myo.npz', 'MyoHand_spheremedium_inspect1.npz'),
    ('spheremedium_lift_myo.npz', 'MyoHand_spheremedium_lift.npz'),
    ('spheresmall_inspect1_myo.npz', 'MyoHand_spheresmall_inspect1.npz'),
    ('spheresmall_lift_myo.npz', 'MyoHand_spheresmall_lift.npz'),
    ('spheresmall_pass1_myo.npz', 'MyoHand_spheresmall_pass1.npz'),
    ('stamp_lift_myo.npz', 'MyoHand_stamp_lift.npz'),
    ('stamp_stamp1_myo.npz', 'MyoHand_stamp_stamp1.npz'),
    ('stanfordbunny_inspect1_myo.npz', 'MyoHand_stanfordbunny_inspect1.npz'),
    ('stanfordbunny_pass1_myo.npz', 'MyoHand_stanfordbunny_pass1.npz'),
    ('stapler_lift_myo.npz', 'MyoHand_stapler_lift.npz'),
    ('stapler_staple1_myo.npz', 'MyoHand_stapler_staple1.npz'),
    ('stapler_staple2_myo.npz', 'MyoHand_stapler_staple2.npz'),
    ('teapot_pour2_myo.npz', 'MyoHand_teapot_pour2.npz'),
    ('toothbrush_brush1_myo.npz', 'MyoHand_toothbrush_brush1.npz'),
    ('toothbrush_lift_myo.npz', 'MyoHand_toothbrush_lift.npz'),
    ('toothpaste_lift_myo.npz', 'MyoHand_toothpaste_lift.npz'),
    ('toothpaste_squeeze1_myo.npz', 'MyoHand_toothpaste_squeeze1.npz'),
    ('toruslarge_inspect1_myo.npz', 'MyoHand_toruslarge_inspect1.npz'),
    ('toruslarge_lift_myo.npz', 'MyoHand_toruslarge_lift.npz'),
    ('torusmedium_lift_myo.npz', 'MyoHand_torusmedium_lift.npz'),
    ('torusmedium_pass1_myo.npz', 'MyoHand_torusmedium_pass1.npz'),
    ('torussmall_lift_myo.npz', 'MyoHand_torussmall_lift.npz'),
    ('torussmall_pass1_myo.npz', 'MyoHand_torussmall_pass1.npz'),
    ('train_play1_myo.npz', 'MyoHand_train_play1.npz'),
    ('watch_lift_myo.npz', 'MyoHand_watch_lift.npz'),
    ('watch_pass1_myo.npz', 'MyoHand_watch_pass1.npz'),
    ('waterbottle_lift_myo.npz', 'MyoHand_waterbottle_lift.npz'),
    ('waterbottle_pass1_myo.npz', 'MyoHand_waterbottle_pass1.npz'),
    ('waterbottle_shake1_myo.npz', 'MyoHand_waterbottle_shake1.npz'),
    ('wineglass_drink1_myo.npz', 'MyoHand_wineglass_drink1.npz'),
    ('wineglass_drink2_myo.npz', 'MyoHand_wineglass_drink2.npz'),
    ('wineglass_lift_myo.npz', 'MyoHand_wineglass_lift.npz'),
    ('wineglass_pass1_myo.npz', 'MyoHand_wineglass_pass1.npz'),
]

old_path_dir = '/home/vik/Libraries/mimic/trajectories_myo'
for old_name, new_name in file_names:
    print(f"{new_name}")
    old_path = os.path.join(old_path_dir, old_name)
    new_path = os.path.join(curr_dir, new_name)
    print(f"Processing: {old_path} as {new_path}", end="\n")
    data = load(old_path)
    proces_n_save(data, new_path)


