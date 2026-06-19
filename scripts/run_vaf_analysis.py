# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Compute and save VAF curves for a MuscleMimic fullbody policy.

Pipeline:
  1. Resolve checkpoint (local path or ``hf://`` HuggingFace ref).
  2. Build the fullbody MuJoCo model.
  3. Load one or more AMASS motion clips.
  4. Roll out the policy and collect high-reward muscle activations.
  5. Compute VAF for n = 1…max_synergies and save:
     - ``<out_dir>/activations.npy``
     - ``<out_dir>/vaf_curve.json``
     - ``<out_dir>/vaf_curve.png``

Usage::

    python scripts/run_vaf_analysis.py \\
        --checkpoint hf://amathislab/mm-10m-2 \\
        --motions KIT/314/walking_medium09_poses \\
        --out_dir outputs/vaf_fullbody

    # multiple clips
    python scripts/run_vaf_analysis.py \\
        --checkpoint hf://amathislab/mm-10m-2 \\
        --motions KIT/314/walking_medium09_poses CMU/09/09_01_poses \\
        --max_synergies 60 \\
        --n_episodes 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect MuscleMimic activations and compute VAF curve."
    )
    p.add_argument(
        "--checkpoint",
        default="hf://amathislab/mm-10m-2",
        help="Checkpoint ref: local path or hf://<owner>/<repo> (default: %(default)s)",
    )
    p.add_argument(
        "--motions",
        nargs="+",
        default=["KIT/314/walking_medium09_poses"],
        metavar="MOTION",
        help="One or more AMASS motion keys or .mot/.npz file paths.",
    )
    p.add_argument(
        "--out_dir",
        default="outputs/vaf_analysis",
        help="Output directory (default: %(default)s)",
    )
    p.add_argument(
        "--n_episodes",
        type=int,
        default=50,
        help="Episodes per clip for activation collection (default: %(default)s)",
    )
    p.add_argument(
        "--reward_percentile",
        type=int,
        default=80,
        help="Reward percentile filter threshold (default: %(default)s)",
    )
    p.add_argument(
        "--max_synergies",
        type=int,
        default=40,
        help="Upper bound on synergy count for VAF curve (default: %(default)s)",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps per episode; None = full clip length (default: %(default)s)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: %(default)s)",
    )
    p.add_argument(
        "--no_cache",
        action="store_true",
        help="Ignore cached activations.npy and re-collect.",
    )
    return p.parse_args()


def _plot_vaf(vaf_curve: dict[int, float], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping VAF plot.")
        return

    ns = sorted(vaf_curve)
    vafs = [vaf_curve[n] for n in ns]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, vafs, marker="o", markersize=4, linewidth=1.5, color="steelblue")
    ax.axhline(0.95, color="tomato", linestyle="--", linewidth=1, label="95% VAF")
    ax.set_xlabel("Number of synergies")
    ax.set_ylabel("Variance Accounted For (VAF)")
    ax.set_title("VAF curve — MuscleMimic fullbody policy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("VAF plot saved to %s", out_path)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Imports (deferred so --help works without heavy deps)
    # ------------------------------------------------------------------
    import importlib.util

    if importlib.util.find_spec("mujoco") is None:
        log.error("mujoco not installed; run: pip install mujoco")
        sys.exit(1)

    from myosuite.integrations.musclemimic.activation_collector import (
        CollectionConfig,
        collect_activations,
    )
    from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
        resolve_checkpoint_ref,
    )
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        FullbodyObsAdapter,
        LocalPolicyRunner,
        load_local_policy_artifacts,
        read_checkpoint_config_metadata,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )
    from myosuite.core.trajectory_io import (
        load_motion_clip,
        resolve_motion_path,
    )
    from myosuite.integrations.musclemimic.sar_extraction import (
        compute_vaf_curve,
        select_n_synergies,
    )

    # ------------------------------------------------------------------
    # 2. Resolve checkpoint
    # ------------------------------------------------------------------
    log.info("Resolving checkpoint: %s", args.checkpoint)
    checkpoint = resolve_checkpoint_ref(args.checkpoint)
    checkpoint_root = checkpoint.local_path
    log.info("Checkpoint root: %s", checkpoint_root)

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    log.info("Compiling fullbody MuJoCo model…")
    cfg = default_mimic_fullbody_config()
    model, _spec, _xml = compile_mimic_fullbody_mjmodel(cfg)
    log.info("Model: nq=%d  nv=%d  nu=%d", model.nq, model.nv, model.nu)

    # ------------------------------------------------------------------
    # 4. Load motion clips
    # ------------------------------------------------------------------
    clips = []
    for motion_key in args.motions:
        log.info("Loading motion clip: %s", motion_key)
        motion_file = resolve_motion_path(motion_key, env_name="MyoFullBody")
        clip = load_motion_clip(motion_file, expected_nq=model.nq, expected_nv=model.nv)
        clips.append(clip)
        log.info(
            "  → %d frames  site_xpos=%s",
            clip.qpos.shape[0],
            clip.site_xpos.shape if clip.site_xpos is not None else None,
        )

    # ------------------------------------------------------------------
    # 5. Build policy runner
    # ------------------------------------------------------------------
    log.info("Loading policy artifacts…")
    artifacts = load_local_policy_artifacts(checkpoint_root)
    goal_params = (
        read_checkpoint_config_metadata(checkpoint_root)
        .get("experiment", {})
        .get("env_params", {})
        .get("goal_params", {})
    )
    obs_adapter = FullbodyObsAdapter(model, clips[0], goal_params)
    policy = LocalPolicyRunner(
        artifacts=artifacts,
        stochastic=False,
        seed=args.seed,
        obs_adapter=obs_adapter,
    )

    # ------------------------------------------------------------------
    # 6. Collect activations (with optional cache)
    # ------------------------------------------------------------------
    cache_path = None if args.no_cache else out_dir / "activations.npy"
    config = CollectionConfig(
        n_episodes_per_clip=args.n_episodes,
        reward_percentile=args.reward_percentile,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
    )

    log.info(
        "Collecting activations: %d episodes/clip, p%d reward filter",
        args.n_episodes,
        args.reward_percentile,
    )
    acts = collect_activations(policy, model, clips, config, cache_path=cache_path)
    log.info("Activations shape: %s", acts.shape)

    if cache_path is not None and not cache_path.exists():
        import numpy as np

        np.save(cache_path, acts)
        log.info("Saved activations to %s", cache_path)

    # ------------------------------------------------------------------
    # 7. Compute and save VAF curve
    # ------------------------------------------------------------------
    log.info("Computing VAF curve (max_synergies=%d)…", args.max_synergies)
    vaf_curve = compute_vaf_curve(acts, max_synergies=args.max_synergies)

    vaf_json = out_dir / "vaf_curve.json"
    with open(vaf_json, "w") as f:
        json.dump({str(k): v for k, v in vaf_curve.items()}, f, indent=2)
    log.info("VAF curve saved to %s", vaf_json)

    # Print summary table
    for threshold in (0.80, 0.90, 0.95, 0.99):
        n = select_n_synergies(vaf_curve, threshold=threshold)
        log.info("  VAF >= %.0f%%  →  %d synergies", threshold * 100, n)

    _plot_vaf(vaf_curve, out_dir / "vaf_curve.png")
    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
