#!/bin/bash
#SBATCH --job-name=Reach
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=24:00:0
#SBATCH --requeue
#SBATCH --exclude=gpu-sr670-22,gpu-sr670-20 # still compiling after 20 mins, gpu-350-02
##SBATCH --nodelist=gpu-380-17,gpu-350-04
##SBATCH --nodelist=gpu-380-17
##SBATCH --nodelist=gpu-350-04
##SBATCH --nodelist=gpu-380-14
##SBATCH --nodelist=gpu-350-04

echo "Launching a python run"
date

source /nfs/nhome/live/jheald/.bashrc

module load miniconda

conda deactivate
conda activate infosyn2

export WANDB_API_KEY=9ae130eea17d49e2bd1deafd27c8a8de06f66830

cd /tmp
export HOME=/tmp
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=0

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python3 -u /nfs/nhome/live/jheald/myosuite/myosuite/envs/myo/mjx/${1}Reach.py
# sbatch slurm_script_basic.sh 'infoSynSAC'

rm -rf /tmp/.bashrc
rm -rf /tmp/.mujoco/

# sbatch animal_slurm_script_basic.sh 'test65'
