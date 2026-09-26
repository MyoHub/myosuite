#!/usr/bin/env python3
"""Generate parity baselines for challenge envs from the public PyPI myosuite.

Creates an isolated virtual-env, installs the latest (or pinned) public
``myosuite`` package from PyPI, runs each challenge env for N steps with a
fixed seed, and pickles the trajectories to
``myosuite/tests/pypi_challenge_baselines/<env_id>.pkl``.

These baselines are consumed by ``test_challenge_pypi_regression.py``.

Usage::

    # latest public release (default)
    python scripts/generate_pypi_challenge_baselines.py

    # pin a specific version
    python scripts/generate_pypi_challenge_baselines.py --pypi-version 2.12.2

    # also include Modular / Fati / Sarc variants
    python scripts/generate_pypi_challenge_baselines.py --all-variants

Exit codes:
    0 — all baselines written
    1 — one or more envs failed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
import venv
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BASELINE_DIR = _REPO_ROOT / "myosuite" / "tests" / "pypi_challenge_baselines"

# Core env IDs that must exist in any public myosuite 2.x release.
_CORE_CHALLENGE_ENVS = [
    "myoChallengeChaseTagP1-v0",
    "myoChallengeChaseTagP2-v0",
    "myoChallengeChaseTagP2eval-v0",
    "myoChallengeOslRunFixed-v0",
    "myoChallengeOslRunRandom-v0",
    "myoChallengeSoccerP1-v0",
    "myoChallengeSoccerP2-v0",
]

_VARIANT_PREFIXES = ["myoFati", "myoSarc"]


def _build_venv(venv_dir: Path, pypi_version: str | None) -> Path:
    """Create a fresh venv and install the requested myosuite from PyPI."""
    print(f"[setup] Creating venv at {venv_dir} …")
    venv.create(str(venv_dir), with_pip=True, clear=True)

    pip = venv_dir / "bin" / "pip"
    pkg = f"myosuite=={pypi_version}" if pypi_version else "myosuite"
    print(f"[setup] Installing {pkg} …")
    subprocess.run(
        [str(pip), "install", "--quiet", pkg],
        check=True,
    )
    return venv_dir / "bin" / "python"


# Inner script executed inside the isolated venv.
_INNER_SCRIPT = textwrap.dedent(
    """
    import sys, pickle, json
    from pathlib import Path

    env_ids = json.loads(sys.argv[1])
    n_steps = int(sys.argv[2])
    seed    = int(sys.argv[3])
    out_dir = Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)

    import myosuite  # noqa: F401 — auto-registers envs on import for public releases
    try:
        myosuite.register_all_envs()
    except AttributeError:
        pass  # public releases (2.x) register envs in __init__
    import gymnasium as gym
    import numpy as np

    results = {}
    for env_id in env_ids:
        try:
            env = gym.make(env_id)
        except Exception as exc:
            results[env_id] = {"error": str(exc)}
            continue

        obs, _info = env.reset(seed=seed)
        episode_np_state = env.unwrapped.np_random.bit_generator.state

        # Collect obs dict keys if accessible (old BaseV0 envs expose obs_dict).
        obs_dict_keys: list[str] = []
        raw = getattr(env.unwrapped, "obs_dict", None)
        if isinstance(raw, dict):
            obs_dict_keys = sorted(raw.keys())

        # Run multiple episodes to get stable reward statistics.
        all_rewards: list[float] = []
        steps = []
        for step_i in range(n_steps):
            action = env.action_space.sample()
            obs_next, rwd, terminated, truncated, info = env.step(action)
            all_rewards.append(float(rwd))
            steps.append({
                "action": action,
                "obs": obs_next.astype(np.float32),
                "rwd": float(rwd),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            if terminated or truncated:
                obs, _info = env.reset()
        steps[0]["episode_np_state"] = episode_np_state

        baseline = {
            "steps": steps,
            "obs_dict_keys": obs_dict_keys,
            "reward_mean": float(np.mean(all_rewards)),
            "reward_std": float(np.std(all_rewards)),
            "obs_shape": list(steps[0]["obs"].shape),
            "action_shape": list(steps[0]["action"].shape),
        }
        env.close()

        out_path = out_dir / f"{env_id}.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(baseline, fh, protocol=4)
        results[env_id] = {"ok": True, "n_steps": len(steps),
                           "obs_shape": list(steps[0]["obs"].shape)}

    print(json.dumps(results))
    """
)


def run(
    env_ids: list[str],
    python: Path,
    n_steps: int,
    seed: int,
    out_dir: Path,
) -> dict[str, dict]:
    """Execute the inner script in the isolated venv."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="pypi_inner_"
    ) as tmp:
        tmp.write(_INNER_SCRIPT)
        script_path = tmp.name

    import os

    # Strip dev-repo paths from the child environment so the venv python
    # cannot import the in-tree myosuite (e.g. via editable install or
    # PYTHONPATH set by the outer shell).
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    repo_root = str(_REPO_ROOT)
    for var in ("PATH",):
        env[var] = os.pathsep.join(
            p for p in env.get(var, "").split(os.pathsep) if repo_root not in p
        )

    result = subprocess.run(
        [
            str(python),
            script_path,
            json.dumps(env_ids),
            str(n_steps),
            str(seed),
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        print("[inner stderr]", result.stderr[-3000:], file=sys.stderr)
        raise RuntimeError("Inner script failed")

    # Extract the last JSON line from stdout (warnings may precede it).
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"No JSON found in inner output:\n{result.stdout}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pypi-version",
        default=None,
        help="PyPI release to install (e.g. 2.12.2). Default: latest.",
    )
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Include Fati/Sarc muscle-condition variants.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_BASELINE_DIR,
    )
    args = parser.parse_args()

    env_ids = list(_CORE_CHALLENGE_ENVS)
    if args.all_variants:
        for prefix in _VARIANT_PREFIXES:
            for eid in _CORE_CHALLENGE_ENVS:
                env_ids.append(eid.replace("myo", prefix, 1))

    print(f"[plan] Generating baselines for {len(env_ids)} envs")
    print(f"[plan] Output → {args.out_dir}")

    with tempfile.TemporaryDirectory(prefix="pypi_myosuite_venv_") as venv_tmp:
        python = _build_venv(Path(venv_tmp), args.pypi_version)
        results = run(env_ids, python, args.n_steps, args.seed, args.out_dir)

    # Report
    failed = []
    for env_id, res in results.items():
        if "error" in res:
            print(f"  FAIL  {env_id}: {res['error']}", file=sys.stderr)
            failed.append(env_id)
        else:
            print(f"  OK    {env_id}  obs={res['obs_shape']}  steps={res['n_steps']}")

    if failed:
        print(f"\n{len(failed)} env(s) failed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(results)} baselines written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
