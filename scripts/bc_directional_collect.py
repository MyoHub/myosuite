"""CLI: collect a real teacher-rollout BC dataset for directional locomotion.

Downloads the ``amathislab/mm-10m-2`` MuscleMimic teacher checkpoint and the
two circular walking clips (KIT/4 CW + CCW) from
``amathislab/musclemimic-retargeted``, rolls the teacher through both with
:func:`myosuite.integrations.musclemimic.bc_directional_collector.collect_bc_dataset`,
and writes ``(obs, actions)`` pairs to an ``.npz`` file.

Requires ``orbax-checkpoint`` (Linux only, see ``pyproject.toml`` platform
markers) — run on a Linux host / container, not macOS.

Usage::

    python scripts/bc_directional_collect.py \\
        --out runs/bc_directional_v1.npz \\
        --samples-per-clip 25000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-clip", type=int, default=25_000)
    parser.add_argument("--episode-len", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-repo", type=str, default="amathislab/mm-10m-2")
    parser.add_argument(
        "--gait-repo", type=str, default="amathislab/musclemimic-retargeted"
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download, hf_hub_download

    from myosuite.core.trajectory_io import load_motion_clip
    from myosuite.integrations.musclemimic.bc_directional_collector import (
        BcCollectionConfig,
        collect_bc_dataset,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )

    cfg = default_mimic_fullbody_config()
    model, _, _ = compile_mimic_fullbody_mjmodel(cfg)

    clip_files = [
        "MyoFullBody/gmr/KIT/4/WalkInClockwiseCircle01_poses.npz",
        "MyoFullBody/gmr/KIT/4/WalkInCounterClockwiseCircle08_poses.npz",
    ]
    clips = []
    for fname in clip_files:
        p = hf_hub_download(repo_id=args.gait_repo, filename=fname, repo_type="dataset")
        clips.append(
            load_motion_clip(Path(p), expected_nq=model.nq, expected_nv=model.nv)
        )
        print(f"Loaded clip {fname}: {len(clips[-1].qpos)} frames")

    teacher_root = Path(snapshot_download(repo_id=args.teacher_repo))
    print(f"Teacher checkpoint: {teacher_root}")

    collection_cfg = BcCollectionConfig(
        samples_per_clip=args.samples_per_clip,
        episode_len=args.episode_len,
        seed=args.seed,
    )
    dataset = collect_bc_dataset(teacher_root, model, clips, collection_cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        obs=dataset["obs"],
        actions=dataset["actions"],
        theta=dataset["theta"],
    )
    print(
        f"Saved {dataset['obs'].shape[0]} transitions to {args.out} "
        f"(obs={dataset['obs'].shape}, actions={dataset['actions'].shape})"
    )


if __name__ == "__main__":
    main()
