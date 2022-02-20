# USAGE
# ./train_myosuite.sh myo          # runs natively
# ./train_myosuite.sh myo local    # use local launcher
# ./train_myosuite.sh myo slurm    # use slurm launcher

# Configure launch
if [ "$#" -ne 2 ] ; then
    config=""
else
    config="--multirun hydra/output=$2 hydra/launcher=$2"
fi

# Configure envs
if [ "$1" == "myo" ] ; then
    envs="myoFingerReachFixed-v0,myoFingerReachRandom-v0,myoFingerPoseFixed-v0,myoFingerPoseRandom-v0,myoElbowPose1D6MFixed-v0,myoElbowPose1D6MRandom-v0,myoHandPoseFixed-v0,myoHandPoseRandom-v0,myoHandReachFixed-v0,myoHandReachRandom-v0,myoHandKeyTurnFixed-v0,myoHandKeyTurnRandom-v0,myoHandObjHoldFixed-v0,myoHandObjHoldRandom-v0,myoHandPenTwirlFixed-v0,myoHandPenTwirlRandom-v0,myoHandBaodingFixed-v1,myoHandBaodingRandom-v1"
    config="--config-name hydra_myo_config.yaml $config"

elif [ "$1" == "extra" ] ; then
    envs="motorFingerReachFixed-v0,motorFingerReachRandom-v0,motorFingerPoseFixed-v0,motorFingerPoseRandom-v0,myoHandBaodingFixed4th-v1,myoHandBaodingFixed8th-v1"
    config="--config-name hydra_myo_config.yaml $config"

else
    echo "Unknown task suite"
    exit 0
fi

# Disp NPG commands
echo "NPG: ======="
echo "python hydra_mjrl_launcher.py --config-path config $config env=$envs"