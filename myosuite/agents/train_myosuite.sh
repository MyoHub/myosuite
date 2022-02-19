# USAGE
# ./train_myosuite_suits.sh myo         # runs natively
# ./train_myosuite_suits.sh myo local    # use local launcher
# ./train_myosuite_suits.sh myo slurm    # use slurm launcher

# Configure launch
if [ "$#" -ne 2 ] ; then
    config=""
else
    config="--multirun hydra/output=$2 hydra/launcher=$2"
fi

# Configure envs
if [ "$1" == "myo" ] ; then
    envs="FingerReachMuscleFixed-v0,FingerReachMuscleRandom-v0,FingerPoseMuscleFixed-v0,FingerPoseMuscleRandom-v0,ElbowPose1D1MRandom-v0,ElbowPose1D6MRandom-v0,HandPoseMuscleFixed-v0,HandPoseMuscleRandom-v0,HandReachMuscleFixed-v0,HandReachMuscleRandom-v0,HandKeyTurnFixed-v0,HandKeyTurnRandom-v0,HandObjHoldFixed-v0,HandObjHoldRandom-v0,HandPenTwirlFixed-v0,HandPenTwirlRandom-v0,BaodingFixed-v1,BaodingRandom-v1"
    config="--config-name hydra_myo_config.yaml $config"

elif [ "$1" == "extra" ] ; then
    envs="FingerReachMotorFixed-v0,FingerReachMotorRandom-v0,FingerPoseMotorFixed-v0,FingerPoseMotorRandom-v0,BaodingFixed4th-v1,BaodingFixed8th-v1"
    config="--config-name hydra_myo_config.yaml $config"

else
    echo "Unknown task suite"
    exit 0
fi

# Disp NPG commands
echo "NPG: ======="
echo "python hydra_mjrl_launcher.py --config-path config $config env=$envs"