#!/usr/bin/env python3
"""
Generate parity baselines for all registered MyoSuite environments.

Runs each registered env for N steps with a fixed seed, recording
(obs, rwd, terminated, truncated, info) per step. Results are pickled
to myosuite/tests/parity_baselines/<env_id>.pkl.

These baselines are later used by test_parity.py to verify that refactored
environments produce byte-identical outputs.

Usage:
    python scripts/generate_parity_baselines.py [--n-steps 200] [--seed 42]
    python scripts/generate_parity_baselines.py --env-id myoElbowPose1D6MRandom-v0

Exit codes:
    0 — all baselines generated successfully
    1 — one or more envs failed (logged; others continue)
"""

from __future__ import annotations

import argparse
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

# Ensure the repo root (parent of this scripts/ dir) is first on sys.path so
# that the in-tree myosuite is used, not any installed version.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import myosuite at module level to trigger all env registrations
import myosuite  # noqa: E402

myosuite.register_all_envs()

BASELINE_DIR = Path(__file__).parent.parent / "myosuite" / "tests" / "parity_baselines"


def _get_all_env_ids() -> list[str]:
    """Return all env IDs registered by importing myosuite."""
    import gymnasium as gym

    # Import myosuite to trigger all registrations
    return [
        env_id
        for env_id in gym.envs.registry.keys()
        if env_id.startswith("myo")
        or env_id.startswith("Myo")
        or env_id.startswith("motor")
    ]


def generate_baseline(
    env_id: str,
    n_steps: int = 200,
    seed: int = 42,
    output_dir: Path = BASELINE_DIR,
) -> bool:
    """Generate and save parity baseline for a single environment.

    Args:
        env_id: Gymnasium environment identifier.
        n_steps: Number of steps to record.
        seed: Random seed for reproducibility.
        output_dir: Directory to write the pickle file.

    Returns:
        True on success, False on failure.
    """
    import gymnasium as gym

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{env_id}.pkl"

    try:
        env = gym.make(env_id)
        # Seed the env's internal RNG with the fixed seed for reproducibility.
        # env.reset(seed=seed) does NOT propagate through to env.np_random in
        # the current codebase, so we set np_random.bit_generator.state directly
        # after an unseeded reset to get a deterministic starting RNG state.
        env.reset()
        env.unwrapped.np_random.bit_generator.state = np.random.default_rng(
            seed
        ).bit_generator.state

        rng = np.random.default_rng(seed)

        # Records store the np_random state at the start of each episode plus
        # the actions taken so the test can reproduce the exact trajectory.
        # Plain dicts to avoid pickle class-resolution issues.
        records: list[dict] = []
        episode_np_state = env.unwrapped.np_random.bit_generator.state
        obs, _ = env.reset()
        for _ in range(n_steps):
            action = rng.uniform(
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
            obs, rwd, terminated, truncated, info = env.step(action)
            # Strip non-picklable info values
            safe_info = {
                k: v for k, v in info.items() if k not in ("obs_dict", "rwd_dict")
            }
            records.append(
                {
                    "action": np.array(action, dtype=np.float32),
                    "obs": np.array(obs, dtype=np.float64),
                    "rwd": float(rwd),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": safe_info,
                    "episode_np_state": episode_np_state,
                }
            )
            if terminated or truncated:
                episode_np_state = env.unwrapped.np_random.bit_generator.state
                obs, _ = env.reset()

        env.close()

        with open(out_path, "wb") as f:
            pickle.dump(records, f, protocol=5)

        print(f"  ✓ {env_id}  ({len(records)} steps → {out_path.name})")
        return True

    except Exception as exc:
        print(f"  ✗ {env_id}: {exc}", file=sys.stderr)
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-steps", type=int, default=200, help="Steps per environment (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Single env ID to generate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASELINE_DIR,
        help=f"Output directory (default: {BASELINE_DIR})",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print full tracebacks on failure"
    )
    args = parser.parse_args()

    if args.env_id:
        env_ids = [args.env_id]
    else:
        print("Discovering registered MyoSuite environments...")
        env_ids = _get_all_env_ids()
        print(f"Found {len(env_ids)} environments.")

    print(f"Generating baselines → {args.output_dir}\n")

    failures = []
    for env_id in sorted(env_ids):
        ok = generate_baseline(
            env_id,
            n_steps=args.n_steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        if not ok:
            failures.append(env_id)

    print()
    if failures:
        print(f"FAILED ({len(failures)}/{len(env_ids)}):", file=sys.stderr)
        for fid in failures:
            print(f"  {fid}", file=sys.stderr)
        return 1

    print(f"All {len(env_ids)} baselines generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
