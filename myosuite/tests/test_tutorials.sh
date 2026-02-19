#!/bin/bash
set -e
GreenBK='\033[1;42m'
RedBK='\033[1;41m'
RC='\033[0m'

uv pip install scikit-learn

# Install potential missing packages needed for the tutorials
uv pip  install jupyter ipykernel tabulate matplotlib torch h5py tqdm osqp
uv pip  install git+https://github.com/aravindr93/mjrl.git@pvr_beta_1vk # install from branch `pvr_beta_1vk` compatible with `mujoco` native binding
uv pip  install stable-baselines3

uv run python -m ipykernel install --user --name=myosuite_uv

# Tested tutorials
TUTORIALS=(
                         "1_Get_Started.ipynb" \
                         "2_Load_policy.ipynb" \
                         "3_Analyse_movements.ipynb" \
                        #  "4_Train_policy.ipynb" \
                        #  "4a_deprl.ipynb" \
                         "4c_Train_SB_policy.ipynb" \
                         "5_Move_Hand_Fingers.ipynb"
                         "6_Inverse_Dynamics.ipynb"
                         "7_Fatigue_Modeling.ipynb"
                         "9_Computed_muscle_control.ipynb"
                         "10_PlaybackMotFile.ipynb"
                         )

# Run all tutorials in one Python process to avoid re-parsing uv.lock each iteration
uv run python -c "
import sys
from pathlib import Path
tutorials = [
    '1_Get_Started.ipynb',
    '2_Load_policy.ipynb',
    '3_Analyse_movements.ipynb',
    '4c_Train_SB_policy.ipynb',
    '5_Move_Hand_Fingers.ipynb',
    '6_Inverse_Dynamics.ipynb',
    '7_Fatigue_Modeling.ipynb',
    '9_Computed_muscle_control.ipynb',
    '10_PlaybackMotFile.ipynb',
]
from myosuite.tests.execute_tutorial import main as run_one
for s in tutorials:
    path = Path('tutorials') / s
    if not path.exists():
        print(f'Skip (not found): {s}')
        continue
    sys.argv = ['execute_tutorial', str(path)]
    try:
        run_one()
        print(f'\033[1;42mTutorial {s}!\033[0m')
    except Exception as e:
        print(f'\033[1;41mSomething is wrong with tutorial {s}!\033[0m')
        raise
"

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
rm ./tutorials/*.nbconvert.ipynb
