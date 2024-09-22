# USAGE
# ./train_myosuite.sh myo local mjrl   # use mjrl with local launcher
# ./train_myosuite.sh myo slurm mjrl   # use mjrl with slurm launcher
# ./train_myosuite.sh myo local sb3   # use stable-baselines3 with local launcher
# ./train_myosuite.sh myo slurm sb3   # use stable-baselines3 with slurm launcher

# Configure launch
if [ "$2" == 'local' ] ; then
    config=""
elif [ "$2" == "slurm" ] ; then
    config="--multirun hydra/output=$2 hydra/launcher=$2"
else
    echo "Unknown .. use local or slurm"
    exit 0
fi


# Configure envs
if [ "$1" == "myo" ] ; then
    envs="myoFingerReachFixed-v0,myoFingerReachRandom-v0,myoFingerPoseFixed-v0,myoFingerPoseRandom-v0,myoElbowPose1D6MFixed-v0,myoElbowPose1D6MRandom-v0,myoHandPoseFixed-v0,myoHandPoseRandom-v0,myoHandReachFixed-v0,myoHandReachRandom-v0,myoHandKeyTurnFixed-v0,myoHandKeyTurnRandom-v0,myoHandObjHoldFixed-v0,myoHandObjHoldRandom-v0,myoHandPenTwirlFixed-v0,myoHandPenTwirlRandom-v0,myoHandBaodingFixed-v1,myoHandBaodingRandom-v1"    
elif [ "$1" == "extra" ] ; then
    envs="motorFingerReachFixed-v0,motorFingerReachRandom-v0,motorFingerPoseFixed-v0,motorFingerPoseRandom-v0,myoHandBaodingFixed4th-v1,myoHandBaodingFixed8th-v1"
elif [ "$1" == "myochal" ] ; then
    envs="myoChallengeBimanual-v0"
else
    echo "Unknown task suite"
    exit 0
fi


if [ "$3" == "mjrl" ] ; then
    config="--config-name hydra_myo_config.yaml $config"
    # Disp NPG commands
    echo "NPG: ======="
    echo "python hydra_mjrl_launcher.py --config-path config $config env=$envs"
elif [ "$3" == "sb3" ] ; then
    if [ "$1" == "myochal" ] ; then
        config="--config-name hydra_myochal_sb3_ppo_config.yaml $config"
    else
        config="--config-name hydra_myo_sb3_ppo_config.yaml $config"
    fi
    # Disp SB3 commands
    echo "Stable-Baselines3: ======="
    echo "python hydra_sb3_launcher.py --config-path config $config env=$envs"
else
    echo "Unknown training framework"
    exit 0
fi

if [ "$4" == 'baseline' ]; then
    base_dir="baseline_SB3/myoChal24/${envs}"
    filename="checkpoint.pt"  # Specify the file name to save the checkpoint as

    mkdir -p $base_dir

    checkpoint_path="${base_dir}/${filename}"

    model_url="https://drive.google.com/uc?export=download&id=1P5ip5yjtL4ynbxwDmEOkuF163TG8Vaml"
    
    # Use curl with -L to follow redirects and -o to specify output file
    curl -L $model_url -o $checkpoint_path

    if [ $? -eq 0 ]; then
        echo "Download successful, checkpoint saved to $checkpoint_path."
        config="$config checkpoint=$checkpoint_path job_name=$job_name"
    else
        echo "Download failed."
        exit 1
    fi
fi



#python hydra_sb3_launcher.py --config-path config --config-name hydra_myo_sb3_ppo_config.yaml  env=myoHandPoseRandom-v0

#python hydra_sb3_launcher.py --config-path config --config-name hydra_myo_sb3_ppo_config.yaml --multirun hydra/output=slurm hydra/launcher=slurm env=myoHandPoseRandom-v0