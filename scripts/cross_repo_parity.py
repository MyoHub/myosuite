#!/usr/bin/env python3
"""
Cross-repo parity test: verify myosuite4 (new) matches myosuite (old).

This script tests behavioral parity between the old and new implementations
by starting both environments from the SAME physics state and comparing
step outputs. This bypasses differences in random number generators between
the two repos.

Methodology:
1. Run OLD env to reset and collect initial physics state + task state.
2. Inject that exact state into NEW env.
3. Step both with the same fixed action sequence.
4. Compare obs, rewards, and termination flags.

The test verifies:
- Identical physics stepping (mj_step)
- Identical observation computation from the same physics state
- Identical reward computation from the same observations

Usage:
    cd /home/user/myosuite4
    uv run python scripts/cross_repo_parity.py [--verbose]

Exit codes:
    0 — all environments pass parity check
    1 — one or more environments fail
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

# Env IDs to test (must be registered in both repos)
ENV_IDS = [
    "myoFingerPoseFixed-v0",
    "myoFingerPoseRandom-v0",
    "myoElbowPose1D6MFixed-v0",
    "myoElbowPose1D6MRandom-v0",
    "myoHandPoseRandom-v0",
    "myoChallengeBaodingP2-v1",
    "myoLegWalk-v0",
]

# Paths to both repos
NEW_REPO = Path(__file__).resolve().parent.parent  # myosuite4
OLD_REPO = Path("/home/user/myosuite")
NEW_PY = NEW_REPO / ".venv" / "bin" / "python"
OLD_PY = OLD_REPO / ".venv" / "bin" / "python"

# ── Trajectory generator script (Phase 1: OLD env resets and records state) ──

_OLD_EXPORT_SCRIPT = """
import sys, pickle, numpy as np
sys.path.insert(0, '{old_repo}')
import myosuite
myosuite.register_all_envs()
import gymnasium as gym

ENV_ID = sys.argv[1]
OUTFILE = sys.argv[2]
N_STEPS = int(sys.argv[3])

env = gym.make(ENV_ID)
uw = env.unwrapped

# Use seed=0 for reproducibility within old code.
obs, _ = env.reset(seed=0)

# Collect physics state after reset
mj_data = uw.mj_data if hasattr(uw, 'mj_data') else getattr(uw, 'data', None)
mj_model = uw.mj_model if hasattr(uw, 'mj_model') else getattr(uw, 'model', None)

state = {{
    'qpos': mj_data.qpos[:].copy(),
    'qvel': mj_data.qvel[:].copy(),
    'site_pos': mj_model.site_pos[:].copy(),
    # Physical model properties (may be randomized per-reset)
    'body_mass': mj_model.body_mass[:].copy(),
    'geom_friction': mj_model.geom_friction[:].copy(),
    'geom_size': mj_model.geom_size[:].copy(),
    'obs_at_reset': obs.astype(np.float64),
    'n_act': int(mj_model.nu),
    'nq': int(mj_model.nq),
    'nv': int(mj_model.nv),
}}

# Task-specific state for pose envs
if hasattr(uw, 'target_jnt_value') and uw.target_jnt_value is not None:
    state['target_jnt_value'] = uw.target_jnt_value.copy()

# Task-specific state for baoding env
if hasattr(uw, 'which_task'):
    state['which_task'] = int(uw.which_task.value)
    state['counter'] = int(uw.counter)
    state['goal'] = uw.goal.copy()
    state['ball_1_starting_angle'] = float(uw.ball_1_starting_angle)
    state['ball_2_starting_angle'] = float(uw.ball_2_starting_angle)
    state['x_radius'] = float(uw.x_radius)
    state['y_radius'] = float(uw.y_radius)

# Walk-env task state (check for steps attribute)
if hasattr(uw, 'steps'):
    state['steps'] = int(uw.steps)

# Generate fixed action sequence
rng = np.random.RandomState(42)
actions = rng.uniform(-1.0, 1.0, (N_STEPS, state['n_act'])).astype(np.float32)
state['actions'] = actions

# Record OLD trajectories
records = []
for i in range(N_STEPS):
    obs, rwd, terminated, truncated, info = env.step(actions[i])
    mj_data_after = uw.mj_data if hasattr(uw, 'mj_data') else uw.data
    records.append({{
        'obs': obs.astype(np.float64),
        'rwd': float(rwd),
        'terminated': bool(terminated),
        'qpos': mj_data_after.qpos[:].copy(),
        'qvel': mj_data_after.qvel[:].copy(),
    }})
    if terminated or truncated:
        break

env.close()
data = {{'state': state, 'records': records}}
with open(OUTFILE, 'wb') as f:
    pickle.dump(data, f, protocol=4)
print(f"[old] {{ENV_ID}}: recorded {{len(records)}} steps, nq={{state['nq']}}")
""".format(old_repo=str(OLD_REPO))


_NEW_REPLAY_SCRIPT = """
import sys, pickle, numpy as np, mujoco
sys.path.insert(0, '{new_repo}')
import myosuite
myosuite.register_all_envs()
import gymnasium as gym

ENV_ID = sys.argv[1]
INFILE = sys.argv[2]
OUTFILE = sys.argv[3]

with open(INFILE, 'rb') as f:
    data = pickle.load(f)
state = data['state']
actions = state['actions']
N_STEPS = len(data['records'])

env = gym.make(ENV_ID)
uw = env.unwrapped

# Reset to initialise all internal data structures
env.reset()

# Use mj_data/mj_model — works for both GymnasiumEnv (via alias) and BaseV0 envs
mj_data = uw.mj_data
mj_model = uw.mj_model

# Inject the exact physics state from OLD env
mj_data.qpos[:state['nq']] = state['qpos'][:state['nq']]
mj_data.qvel[:state['nv']] = state['qvel'][:state['nv']]
mj_model.site_pos[:] = state['site_pos']
# Inject physical model properties (mass, friction, size) which may be randomized
mj_model.body_mass[:] = state['body_mass']
mj_model.geom_friction[:] = state['geom_friction']
mj_model.geom_size[:] = state['geom_size']

# Inject task-specific state
if 'target_jnt_value' in state and hasattr(uw, 'target_jnt_value'):
    uw.target_jnt_value = state['target_jnt_value'].copy()
    # Also update _task_state for reward computation
    if hasattr(uw, '_task_state'):
        uw._task_state['target_angles'] = uw.target_jnt_value.copy()

if 'which_task' in state and hasattr(uw, 'which_task'):
    from myosuite.envs.myo.tasks.challenge.baoding import Task
    uw.which_task = Task(state['which_task'])
    uw.counter = state['counter']
    uw.goal = state['goal'].copy()
    uw.ball_1_starting_angle = state['ball_1_starting_angle']
    uw.ball_2_starting_angle = state['ball_2_starting_angle']
    uw.x_radius = state['x_radius']
    uw.y_radius = state['y_radius']

if 'steps' in state and hasattr(uw, 'steps'):
    uw.steps = state['steps']

# Forward to sync kinematics
mujoco.mj_forward(mj_model, mj_data)

# Capture initial obs from injected state — handle both env architectures
if hasattr(uw, '_obs_dict_to_vec'):
    # GymnasiumEnv-style (pose envs, baoding, etc.)
    from myosuite.envs.gymnasium_env import CpuEnvAccessor
    uw._accessor = CpuEnvAccessor(uw.model, uw.data, uw._ctrl_dt)
    obs_injected = uw._obs_dict_to_vec(uw.get_obs_dict(uw._accessor)).astype(np.float64)
else:
    # BaseV0/MujocoEnv-style (WalkEnvV0, etc.)
    # obsd_mj_data == mj_data when no separate obs model
    obs_dict = uw.get_obs_dict(uw.obsd_mj_model, uw.obsd_mj_data)
    _, obs_injected = uw.obsdict2obsvec(obs_dict, uw.obs_keys)
    obs_injected = obs_injected.astype(np.float64)

# Run the same actions
records = []
for i in range(N_STEPS):
    obs, rwd, terminated, truncated, info = env.step(actions[i])
    records.append({{
        'obs': obs.astype(np.float64),
        'rwd': float(rwd),
        'terminated': bool(terminated),
        'qpos': mj_data.qpos[:].copy(),
        'qvel': mj_data.qvel[:].copy(),
    }})
    if terminated or truncated:
        break

env.close()
result = {{
    'records': records,
    'obs_injected': obs_injected,
    'obs_at_reset_old': state['obs_at_reset'],
}}
with open(OUTFILE, 'wb') as f:
    pickle.dump(result, f, protocol=4)
print(f"[new] {{ENV_ID}}: replayed {{len(records)}} steps")
""".format(new_repo=str(NEW_REPO))


def run_script(
    python_exe: str, script_code: str, args: list[str], timeout: int = 300
) -> bool:
    """Write a script to a temp file and run it."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_code)
        script_path = f.name
    try:
        result = subprocess.run(
            [python_exe, script_path] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.stdout:
            print(f"    {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[-2000:]}")
            return False
        return True
    finally:
        os.unlink(script_path)


def compare_trajectories(
    old_records: list[dict],
    new_records: list[dict],
    obs_injected: np.ndarray,
    obs_at_reset_old: np.ndarray,
    env_id: str,
    verbose: bool,
) -> bool:
    """Compare step-by-step trajectories from old and new repos."""
    n = min(len(old_records), len(new_records))
    if n == 0:
        print("    FAIL: no records to compare")
        return False

    # First check: initial obs should match after state injection
    reset_obs_diff = float(np.max(np.abs(obs_at_reset_old - obs_injected)))

    obs_diffs = []
    rwd_diffs = []
    term_mismatches = 0
    qpos_diffs = []

    for i in range(n):
        old_r = old_records[i]
        new_r = new_records[i]

        obs_diff = float(np.max(np.abs(old_r["obs"] - new_r["obs"])))
        rwd_diff = float(abs(old_r["rwd"] - new_r["rwd"]))

        obs_diffs.append(obs_diff)
        rwd_diffs.append(rwd_diff)
        if old_r["terminated"] != new_r["terminated"]:
            term_mismatches += 1

        n_common = min(len(old_r["qpos"]), len(new_r["qpos"]))
        qpos_diffs.append(
            float(np.max(np.abs(old_r["qpos"][:n_common] - new_r["qpos"][:n_common])))
        )

    max_obs = max(obs_diffs) if obs_diffs else 0.0
    max_rwd = max(rwd_diffs) if rwd_diffs else 0.0
    max_qpos = max(qpos_diffs) if qpos_diffs else 0.0

    # Generous tolerance for float32/float64 round-trips
    atol = 5e-4
    passed = (
        reset_obs_diff < atol
        and max_obs < atol
        and max_rwd < atol
        and term_mismatches == 0
    )

    status = "PASS" if passed else "FAIL"
    print(
        f"    {status}: reset_obs_diff={reset_obs_diff:.2e}  "
        f"max_obs={max_obs:.2e}  max_rwd={max_rwd:.2e}  "
        f"term_miss={term_mismatches}  max_qpos={max_qpos:.2e}"
    )

    if not passed and verbose:
        print(f"    [detail] obs at reset (old): {obs_at_reset_old[:5]}")
        print(f"    [detail] obs after inject (new): {obs_injected[:5]}")
        for i, (od, rd) in enumerate(zip(obs_diffs, rwd_diffs)):
            if od >= atol or rd >= atol:
                print(
                    f"    first divergence at step {i}: obs_diff={od:.2e} rwd_diff={rd:.2e}"
                )
                diff_idx = np.where(
                    np.abs(old_records[i]["obs"] - new_records[i]["obs"]) >= atol
                )[0]
                print(f"    diverging obs positions: {diff_idx[:10]}")
                print(f"    old: {old_records[i]['obs'][diff_idx[:5]]}")
                print(f"    new: {new_records[i]['obs'][diff_idx[:5]]}")
                break

    return passed


def test_env(env_id: str, n_steps: int, verbose: bool) -> bool:
    """Test parity for a single env. Returns True if passed."""
    print(f"\n[{env_id}]")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        old_file = f.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        new_file = f.name

    try:
        # Phase 1: Run OLD env, export state + trajectory
        ok = run_script(
            str(OLD_PY), _OLD_EXPORT_SCRIPT, [env_id, old_file, str(n_steps)]
        )
        if not ok:
            print("    SKIP: OLD env failed")
            return False

        # Phase 2: Inject into NEW env and replay
        ok = run_script(str(NEW_PY), _NEW_REPLAY_SCRIPT, [env_id, old_file, new_file])
        if not ok:
            print("    SKIP: NEW env failed")
            return False

        # Phase 3: Compare
        with open(old_file, "rb") as f:
            old_data = pickle.load(f)
        with open(new_file, "rb") as f:
            new_data = pickle.load(f)

        return compare_trajectories(
            old_data["records"],
            new_data["records"],
            new_data["obs_injected"],
            old_data["state"]["obs_at_reset"],
            env_id,
            verbose,
        )
    finally:
        for fp in [old_file, new_file]:
            try:
                os.unlink(fp)
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-steps", type=int, default=50, help="Steps to compare per env (default: 50)"
    )
    parser.add_argument(
        "--env-id", type=str, default=None, help="Single env to test (default: all)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed info on failure"
    )
    args = parser.parse_args()

    env_ids = [args.env_id] if args.env_id else ENV_IDS

    if not NEW_PY.exists():
        print(f"ERROR: myosuite4 venv python not found: {NEW_PY}")
        return 1
    if not OLD_PY.exists():
        print(f"ERROR: myosuite venv python not found: {OLD_PY}")
        return 1

    print(f"Cross-repo parity test ({len(env_ids)} envs, {args.n_steps} steps each)")
    print(f"  OLD: {OLD_REPO}")
    print(f"  NEW: {NEW_REPO}")

    failures = []
    skipped = []
    for env_id in env_ids:
        passed = test_env(env_id, args.n_steps, args.verbose)
        if passed is False:
            failures.append(env_id)
        elif passed is None:
            skipped.append(env_id)

    n_tested = len(env_ids) - len(skipped)
    print(f"\n{'='*60}")
    if failures:
        print(f"FAILED {len(failures)}/{n_tested} tested envs:")
        for e in failures:
            print(f"  {e}")
        return 1
    else:
        print(f"All {n_tested} tested environments PASS cross-repo parity.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
