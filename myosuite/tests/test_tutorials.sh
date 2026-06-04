#!/bin/bash
set -euo pipefail

# Install packages required by tutorials.
uv pip install scikit-learn jupyter ipykernel tabulate matplotlib torch h5py tqdm osqp stable-baselines3

uv run python -m ipykernel install --user --name=myosuite_uv

# Run all tutorials under tutorials/ in one Python process to avoid re-parsing
# uv.lock each iteration. On GitHub Actions, skip notebooks that require large
# Hugging Face assets or long mjlab PPO runs (see SKIP_IN_CI_REL in the script
# embedded below).
uv run python -c "
import os
import sys
from pathlib import Path

from myosuite.tests.execute_tutorial import main as run_one

# Notebooks that need multi-GB HF checkpoints, fixed local cache paths, or
# long mjlab training; keep them for manual / release runs only.
SKIP_IN_CI_REL = frozenset(
    {
        'tutorials/6_Inverse_Dynamics.ipynb',
        'tutorials/4_Train_policy.ipynb',
        'tutorials/4b_reflex/MyoSuite_MyoReflex_Walk.ipynb',
        'tutorials/Walk_Backends_Demo.ipynb',
        'tutorials/mc26/mc26_PyTorch_Policies.ipynb',
        'tutorials/11a_MuscleMimic_Fullbody_Policy_Trajectory.ipynb',
        'tutorials/11b_MuscleMimic_Fullbody_Training.ipynb',
        'tutorials/11c_MuscleMimic_Fullbody_mjlab.ipynb',
    }
)
RUN_IN_CI_REL = frozenset(
    {
        'tutorials/1_Get_Started.ipynb',
        'tutorials/2_Load_policy.ipynb',
        'tutorials/3_Analyse_movements.ipynb',
        'tutorials/4a_deprl.ipynb',
        'tutorials/4c_Train_SB_policy.ipynb',
        'tutorials/5_Move_Hand_Fingers.ipynb',
        'tutorials/7_Fatigue_Modeling.ipynb',
        'tutorials/9_Computed_muscle_control.ipynb',
    }
)

root = Path('tutorials')
if not root.is_dir():
    print('No tutorials/ directory; nothing to run.')
    sys.exit(0)

paths = sorted(
    p
    for p in root.rglob('*.ipynb')
    if '.ipynb_checkpoints' not in p.parts
)

in_ci = bool(os.environ.get('GITHUB_ACTIONS'))
for path in paths:
    rel = path.as_posix()
    if in_ci:
        if rel in SKIP_IN_CI_REL or rel.startswith('tutorials/mc26/'):
            print(f'Skip in CI (optional / long run): {rel}')
            continue
        if rel not in RUN_IN_CI_REL:
            print(f'Skip in CI (not in stable tutorial allowlist): {rel}')
            continue
    sys.argv = ['execute_tutorial', str(path.resolve())]
    try:
        run_one()
        print(f'\033[1;42mOK {rel}\033[0m')
    except Exception:
        print(f'\033[1;41mFailed: {rel}\033[0m')
        raise
"

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
rm -f ./tutorials/*.nbconvert.ipynb
find tutorials -name '*.nbconvert.ipynb' -delete 2>/dev/null || true
