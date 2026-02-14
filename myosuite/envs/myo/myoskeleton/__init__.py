
from myosuite.utils import gym
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Register MyoSkeleton Tracking
gym.register(
    id="MyoSkeletonTrack-v0",
    entry_point="myosuite.envs.myo.myoskeleton.track_v0:MyoSkeletonTrackEnv",
    max_episode_steps=1000,
    kwargs={
        "model_path": curr_dir + "/../mjx/myoskeleton_edited.xml", # Using the edits we made for MJX (meshes fixed)
        "reference_path": curr_dir + "/../mjx/standing_motion.h5",
        "normalize_act": True,
    }
)# Register MyoSkeleton Standing (Static)
gym.register(
    id="MyoSkeletonStand-v0",
    entry_point="myosuite.envs.myo.myoskeleton.stand_v0:MyoSkeletonStandEnv",
    max_episode_steps=1000,
    kwargs={
        "model_path": curr_dir + "/../mjx/myoskeleton_edited.xml",
        "reference_path": curr_dir + "/../mjx/standing_motion.h5",
        "normalize_act": True,
    }
)
