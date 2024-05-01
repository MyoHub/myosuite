from myosuite.utils import gym; register=gym.register

import collections
import os
import numpy as np
curr_dir = os.path.dirname(os.path.abspath(__file__))


# Task specification format
task_spec = collections.namedtuple('task_spec',
        ['name',        # task_name
         'robot',       # robot name
         'object',      # object name
         'motion',      # motion reference file path
         ])


# MyoDM tasks
MyoHand_task_spec = (

    task_spec(name='MyoHandAirplaneFly-v0', robot='MyoHand', object='airplane', motion='MyoHand_airplane_fly1.npz'),
    task_spec(name='MyoHandAirplaneLift-v0', robot='MyoHand', object='airplane', motion='MyoHand_airplane_lift.npz'),
    task_spec(name='MyoHandAirplanePass-v0', robot='MyoHand', object='airplane', motion='MyoHand_airplane_pass1.npz'),
    task_spec(name='MyoHandAlarmclockLift-v0', robot='MyoHand', object='alarmclock', motion='MyoHand_alarmclock_lift.npz'),
    task_spec(name='MyoHandAlarmclockSee-v0', robot='MyoHand', object='alarmclock', motion='MyoHand_alarmclock_see1.npz'),
    task_spec(name='MyoHandAppleLift-v0', robot='MyoHand', object='apple', motion='MyoHand_apple_lift.npz'),
    task_spec(name='MyoHandApplePass-v0', robot='MyoHand', object='apple', motion='MyoHand_apple_pass1.npz'),
    task_spec(name='MyoHandBananaPass-v0', robot='MyoHand', object='banana', motion='MyoHand_banana_pass1.npz'),
    task_spec(name='MyoHandCupDrink-v0', robot='MyoHand', object='cup', motion='MyoHand_cup_drink1.npz'),
    task_spec(name='MyoHandCupPass-v0', robot='MyoHand', object='cup', motion='MyoHand_cup_pass1.npz'),
    task_spec(name='MyoHandCupPour-v0', robot='MyoHand', object='cup', motion='MyoHand_cup_pour1.npz'),
    task_spec(name='MyoHandDuckInspect-v0', robot='MyoHand', object='duck', motion='MyoHand_duck_inspect1.npz'),
    task_spec(name='MyoHandDuckLift-v0', robot='MyoHand', object='duck', motion='MyoHand_duck_lift.npz'),
    task_spec(name='MyoHandElephantLift-v0', robot='MyoHand', object='elephant', motion='MyoHand_elephant_lift.npz'),
    task_spec(name='MyoHandElephantPass-v0', robot='MyoHand', object='elephant', motion='MyoHand_elephant_pass1.npz'),
    task_spec(name='MyoHandFlashlightLift-v0', robot='MyoHand', object='flashlight', motion='MyoHand_flashlight_lift.npz'),
    task_spec(name='MyoHandFlashlight1On-v0', robot='MyoHand', object='flashlight', motion='MyoHand_flashlight_on1.npz'),
    task_spec(name='MyoHandFlashlight2On-v0', robot='MyoHand', object='flashlight', motion='MyoHand_flashlight_on2.npz'),
    task_spec(name='MyoHandHammerUse-v0', robot='MyoHand', object='hammer', motion='MyoHand_hammer_use1.npz'),
    task_spec(name='MyoHandMouseLift-v0', robot='MyoHand', object='mouse', motion='MyoHand_mouse_lift.npz'),
    task_spec(name='MyoHandMugDrink3-v0', robot='MyoHand', object='mug', motion='MyoHand_mug_drink3.npz'),
    task_spec(name='MyoHandScissorsUse-v0', robot='MyoHand', object='scissors', motion='MyoHand_scissors_use1.npz'),
    task_spec(name='MyoHandSpheremediumLift-v0', robot='MyoHand', object='spheremedium', motion='MyoHand_spheremedium_lift.npz'),
    task_spec(name='MyoHandStampStamp-v0', robot='MyoHand', object='stamp', motion='MyoHand_stamp_stamp1.npz'),
    task_spec(name='MyoHandStanfordbunnyInspect-v0', robot='MyoHand', object='stanfordbunny', motion='MyoHand_stanfordbunny_inspect1.npz'),
    task_spec(name='MyoHandStaplerLift-v0', robot='MyoHand', object='stapler', motion='MyoHand_stapler_lift.npz'),
    task_spec(name='MyoHandToothpasteLift-v0', robot='MyoHand', object='toothpaste', motion='MyoHand_toothpaste_lift.npz'),
    task_spec(name='MyoHandToruslargeInspect-v0', robot='MyoHand', object='toruslarge', motion='MyoHand_toruslarge_inspect1.npz'),
    task_spec(name='MyoHandTrainPlay-v0', robot='MyoHand', object='train', motion='MyoHand_train_play1.npz'),
    task_spec(name='MyoHandWatchLift-v0', robot='MyoHand', object='watch', motion='MyoHand_watch_lift.npz'),
    task_spec(name='MyoHandWaterbottleShake-v0', robot='MyoHand', object='waterbottle', motion='MyoHand_waterbottle_shake1.npz'),
    task_spec(name='MyoHandWineglassDrink2-v0', robot='MyoHand', object='wineglass', motion='MyoHand_wineglass_drink2.npz'),
    task_spec(name='MyoHandWineglassLift-v0', robot='MyoHand', object='wineglass', motion='MyoHand_wineglass_lift.npz'),
    task_spec(name='MyoHandStampLift-v0', robot='MyoHand', object='stamp', motion='MyoHand_stamp_lift.npz'),
    task_spec(name='MyoHandStaplerStaple1-v0', robot='MyoHand', object='stapler', motion='MyoHand_stapler_staple1.npz'),
    task_spec(name='MyoHandStaplerStaple2-v0', robot='MyoHand', object='stapler', motion='MyoHand_stapler_staple2.npz'),
    task_spec(name='MyoHandTeapotPour2-v0', robot='MyoHand', object='teapot', motion='MyoHand_teapot_pour2.npz'),
    task_spec(name='MyoHandToothbrushBrush1-v0', robot='MyoHand', object='toothbrush', motion='MyoHand_toothbrush_brush1.npz'),
    task_spec(name='MyoHandBowlDrink2-v0', robot='MyoHand', object='bowl', motion='MyoHand_bowl_drink2.npz'),
    task_spec(name='MyoHandBowlPass-v0', robot='MyoHand', object='bowl', motion='MyoHand_bowl_pass1.npz'),
    task_spec(name='MyoHandToothpasteSqueeze1-v0', robot='MyoHand', object='toothpaste', motion='MyoHand_toothpaste_squeeze1.npz'),
    task_spec(name='MyoHandToruslargeLift-v0', robot='MyoHand', object='toruslarge', motion='MyoHand_toruslarge_lift.npz'),
    task_spec(name='MyoHandTorusmediumLift-v0', robot='MyoHand', object='torusmedium', motion='MyoHand_torusmedium_lift.npz'),
    task_spec(name='MyoHandCubesmallPass-v0', robot='MyoHand', object='cubesmall', motion='MyoHand_cubesmall_pass1.npz'),
    task_spec(name='MyoHandTorussmallLift-v0', robot='MyoHand', object='torussmall', motion='MyoHand_torussmall_lift.npz'),
    task_spec(name='MyoHandMugLift-v0', robot='MyoHand', object='mug', motion='MyoHand_mug_lift.npz'),
    task_spec(name='MyoHandTorussmallPass-v0', robot='MyoHand', object='torussmall', motion='MyoHand_torussmall_pass1.npz'),
    task_spec(name='MyoHandPhoneLift-v0', robot='MyoHand', object='phone', motion='MyoHand_phone_lift.npz'),
    task_spec(name='MyoHandCylindermediumLift-v0', robot='MyoHand', object='cylindermedium', motion='MyoHand_cylindermedium_lift.npz'),
    task_spec(name='MyoHandWatchPass-v0', robot='MyoHand', object='watch', motion='MyoHand_watch_pass1.npz'),
    task_spec(name='MyoHandCylindersmallInspect-v0', robot='MyoHand', object='cylindersmall', motion='MyoHand_cylindersmall_inspect1.npz'),
    task_spec(name='MyoHandCylindersmallPass-v0', robot='MyoHand', object='cylindersmall', motion='MyoHand_cylindersmall_pass1.npz'),
    task_spec(name='MyoHandPyramidsmallInspect-v0', robot='MyoHand', object='pyramidsmall', motion='MyoHand_pyramidsmall_inspect1.npz'),
    task_spec(name='MyoHandWineglassToast1-v0', robot='MyoHand', object='wineglass', motion='MyoHand_wineglass_toast1.npz'),
    task_spec(name='MyoHandSpheresmallInspect-v0', robot='MyoHand', object='spheresmall', motion='MyoHand_spheresmall_inspect1.npz'),
    task_spec(name='MyoHandSpheresmallLift-v0', robot='MyoHand', object='spheresmall', motion='MyoHand_spheresmall_lift.npz'),
    task_spec(name='MyoHandSpheresmallPass-v0', robot='MyoHand', object='spheresmall', motion='MyoHand_spheresmall_pass1.npz'),

    task_spec(name='MyoHandAlarmclockPass-v0', robot='MyoHand', object='alarmclock', motion='MyoHand_alarmclock_pass1.npz'),
    task_spec(name='MyoHandBinocularsPass-v0', robot='MyoHand', object='binoculars', motion='MyoHand_binoculars_pass1.npz'),
    task_spec(name='MyoHandDuckPass-v0', robot='MyoHand', object='duck', motion='MyoHand_duck_pass1.npz'),
    task_spec(name='MyoHandEyeglassesPass-v0', robot='MyoHand', object='eyeglasses', motion='MyoHand_eyeglasses_pass1.npz'),
    task_spec(name='MyoHandFlashlightPass-v0', robot='MyoHand', object='flashlight', motion='MyoHand_flashlight_pass1.npz'),
    task_spec(name='MyoHandFlutePass-v0', robot='MyoHand', object='flute', motion='MyoHand_flute_pass1.npz'),
    task_spec(name='MyoHandHammerPass-v0', robot='MyoHand', object='hammer', motion='MyoHand_hammer_pass1.npz'),
    task_spec(name='MyoHandHandInspect-v0', robot='MyoHand', object='hand', motion='MyoHand_hand_inspect1.npz'),
    task_spec(name='MyoHandHeadphonesPass-v0', robot='MyoHand', object='headphones', motion='MyoHand_headphones_pass1.npz'),
    task_spec(name='MyoHandKnifeChop-v0', robot='MyoHand', object='knife', motion='MyoHand_knife_chop1.npz'),
    task_spec(name='MyoHandLightbulbPass-v0', robot='MyoHand', object='lightbulb', motion='MyoHand_lightbulb_pass1.npz'),
    task_spec(name='MyoHandMouseUse-v0', robot='MyoHand', object='mouse', motion='MyoHand_mouse_use1.npz'),
    task_spec(name='MyoHandPiggybankUse-v0', robot='MyoHand', object='piggybank', motion='MyoHand_piggybank_use1.npz'),
    task_spec(name='MyoHandToothbrushLift-v0', robot='MyoHand', object='toothbrush', motion='MyoHand_toothbrush_lift.npz'),
    task_spec(name='MyoHandWaterbottleLift-v0', robot='MyoHand', object='waterbottle', motion='MyoHand_waterbottle_lift.npz'),
    task_spec(name='MyoHandWineglassPass-v0', robot='MyoHand', object='wineglass', motion='MyoHand_wineglass_pass1.npz'),
    task_spec(name='MyoHandStanfordbunnyPass-v0', robot='MyoHand', object='stanfordbunny', motion='MyoHand_stanfordbunny_pass1.npz'),
    task_spec(name='MyoHandCameraPass-v0', robot='MyoHand', object='camera', motion='MyoHand_camera_pass1.npz'),
    task_spec(name='MyoHandCubelargePass-v0', robot='MyoHand', object='cubelarge', motion='MyoHand_cubelarge_pass1.npz'),
    task_spec(name='MyoHandCubemediumLInspect-v0', robot='MyoHand', object='cubemedium', motion='MyoHand_cubemedium_inspect1.npz'),
    task_spec(name='MyoHandMousePass-v0', robot='MyoHand', object='mouse', motion='MyoHand_mouse_pass1.npz'),
    task_spec(name='MyoHandCubesmallLift-v0', robot='MyoHand', object='cubesmall', motion='MyoHand_cubesmall_lift.npz'),
    task_spec(name='MyoHandTorusmediumPass-v0', robot='MyoHand', object='torusmedium', motion='MyoHand_torusmedium_pass1.npz'),
    task_spec(name='MyoHandMugPass-v0', robot='MyoHand', object='mug', motion='MyoHand_mug_pass1.npz'),
    task_spec(name='MyoHandPiggybankPass-v0', robot='MyoHand', object='wineglass', motion='MyoHand_piggybank_pass1.npz'),
    task_spec(name='MyoHandCylindermediumPass-v0', robot='MyoHand', object='cylindermedium', motion='MyoHand_cylindermedium_pass1.npz'),
    task_spec(name='MyoHandPyramidlargePass-v0', robot='MyoHand', object='pyramidlarge', motion='MyoHand_pyramidlarge_pass1.npz'),
    task_spec(name='MyoHandWaterbottlePass-v0', robot='MyoHand', object='waterbottle', motion='MyoHand_waterbottle_pass1.npz'),
    task_spec(name='MyoHandPyramidmediumPass-v0', robot='MyoHand', object='pyramidmedium', motion='MyoHand_pyramidmedium_pass1.npz'),
    task_spec(name='MyoHandSpherelargePass-v0', robot='MyoHand', object='spherelarge', motion='MyoHand_spherelarge_pass1.npz'),
    task_spec(name='MyoHandSpheremediumInspect-v0', robot='MyoHand', object='spheremedium', motion='MyoHand_spheremedium_inspect1.npz'),
    task_spec(name='MyoHandCylinderlargeInspect-v0', robot='MyoHand', object='cylinderlarge', motion='MyoHand_cylinderlarge_inspect1.npz'),
    task_spec(name='MyoHandGamecontrollerPass-v0', robot='MyoHand', object='gamecontroller', motion='MyoHand_gamecontroller_pass1.npz'),

)

# Register MyoHand envs using reference motion
def register_myohand_object_trackref(task_name, object_name, motion_path=None):
    # Track reference motion
    # print("'"+task_name+"'", end=", ")
    register(
        id=task_name,
        entry_point='myosuite.envs.myo.myodm.myodm_v0:TrackEnv',
        max_episode_steps=75, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': '/../assets/hand/myohand_object.xml',
                'object_name': object_name,
                'reference':curr_dir+'/data/'+motion_path,
            }
    )
for task_name, robot_name, object_name, motion_path in MyoHand_task_spec:
    register_myohand_object_trackref(task_name, object_name, motion_path)


OBJECTS = ('airplane','alarmclock','apple','banana','binoculars','bowl','camera','coffeemug','cubelarge','cubemedium','cubesmall','cup','cylinderlarge','cylindermedium','cylindersmall','duck','elephant','eyeglasses','flashlight','flute','gamecontroller','hammer','hand','headphones','knife','lightbulb','mouse','mug','phone','piggybank', 'pyramidlarge','pyramidmedium','pyramidsmall','scissors','spherelarge','spheremedium','spheresmall','stamp','stanfordbunny','stapler','teapot','toothbrush','toothpaste','toruslarge','torusmedium','torussmall','train','watch','waterbottle','wineglass')

# Register object envs
def register_MyoHand_object(object_name):

    dof_robot = 29
    task_name = 'MyoHand{}Fixed-v0'.format(object_name.title())
    # print("'"+task_name+"'", end=", ")

    # Envs with fixed target
    register(
        id=task_name,
        entry_point='myosuite.envs.myo.myodm.myodm_v0:TrackEnv',
        max_episode_steps=50, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': '/../assets/hand/myohand_object.xml',
                'object_name': object_name,
                'reference': {'time':(0.0, 4.0),
                                'robot':np.zeros((1, dof_robot)),
                                'robot_vel':np.zeros((1,dof_robot)),
                                'object_init':np.array((-.2, -.2, 0.1, 1.0, 0.0, 0.0, 0.0)),
                                'object':np.reshape(np.array((.2, .2, 0.1, 1.0, 0.0, 0.0, 0.1)), (1,7))
                            }
            }
    )

    # Envs with random target
    task_name = 'MyoHand{}Random-v0'.format(object_name.title())
    # print("'"+task_name+"'", end=", ")
    register(
        id=task_name,
        entry_point='myosuite.envs.myo.myodm.myodm_v0:TrackEnv',
        max_episode_steps=50, #50steps*40Skip*2ms = 4s
        kwargs={
                'model_path': '/../assets/hand/myohand_object.xml',
                'object_name': object_name,
                'reference': {'time':(0.0, 4.0),
                                'robot':np.zeros((2, dof_robot)),
                                'robot_vel':np.zeros((2, dof_robot)),
                                'object_init':np.array((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)),
                                'object':np.array([ [-.2, -.2, 0.1, 1.0, 0.0, 0.0, -1.0],
                                                    [0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0]])
                            }
            }
    )
for obj in OBJECTS:
    register_MyoHand_object(obj)