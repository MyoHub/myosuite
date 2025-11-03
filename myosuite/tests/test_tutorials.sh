#!/bin/bash
set -e
GreenBK='\033[1;42m'
RedBK='\033[1;41m'
RC='\033[0m'

uv pip install scikit-learn

# Install potential missing packages needed for the tutorials
uv pip  install jupyter ipykernel tabulate matplotlib torch h5py
uv pip  install git+https://github.com/aravindr93/mjrl.git@pvr_beta_1vk # install from branch `pvr_beta_1vk` compatible with `mujoco` native binding
uv pip  install stable-baselines3

uv run python -m ipykernel install --user --name=myosuite_uv

# Tested tutorials
declare -a TUTORIALS=(
                         "1_Get_Started.ipynb" \
                         "2_Load_policy.ipynb" \
                         "3_Analyse_movements.ipynb" \
                        #  "4_Train_policy.ipynb" \
                        #  "4a_deprl.ipynb" \
                         "4c_Train_SB_policy.ipynb" \
                        #  "5_Move_Hand_Fingers.ipynb"
                         )

# Iterate the string array using for loop
for s in ${TUTORIALS[@]}; do
    uv run jupyter nbconvert --to notebook --execute "./docs/source/tutorials/$s" --ExecutePreprocessor.kernel_name=myosuite_uv

    if [ $? -eq 0 ]; then
        printf "${GreenBK}Tutorial $s!${RC} \n"
    else
        printf "${RedBK}Something is wrong with tutorial $s!${RC} \n"
    fi
done

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
rm ./docs/source/tutorials/*.nbconvert.ipynb
