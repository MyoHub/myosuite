#!/bin/bash
GreenBK='\033[1;42m'
RedBK='\033[1;41m'
RC='\033[0m'

# ## This routine tests the tutorials
# conda create --name $CONDA_DEFAULT_ENV python=3.7.1 -y
# conda init bash
# conda activate myosuite_test
# pip install -e .
pip install scikit-learn

# Install potential missing packages needed for the tutorials
pip install jupyter ipykernel tabulate matplotlib torch h5py
pip install git+https://github.com/aravindr93/mjrl.git
python -m ipykernel install --user --name=$CONDA_DEFAULT_ENV

# Tested tutorials
declare -a StringArray=(
                         "1_Get_Started.ipynb" \
                         "2_Load_policy.ipynb" \
                         "3_Analyse_movements.ipynb" \
                        #  "4_Train_policy.ipynb" \
                        # "5_Move_Hand_Fingers.ipynb" \
                         )

# Iterate the string array using for loop
for s in ${StringArray[@]}; do
    # ls "../../docs/source/tutorials/$s"
    jupyter nbconvert --to notebook --execute "./docs/source/tutorials/$s" --ExecutePreprocessor.kernel_name=$CONDA_DEFAULT_ENV

    if [ $? -eq 0 ]; then
        printf "${GreenBK}Tutorial $s!${RC} \n"
    else
        printf "${RedBK}Something is wrong with tutorial $s!${RC} \n"
    fi
done

# jupyter kernelspec remove $CONDA_DEFAULT_ENV
rm ./docs/source/tutorials/*.nbconvert.ipynb
# conda deactivate
# conda remove --name myosuite_test --all -y
