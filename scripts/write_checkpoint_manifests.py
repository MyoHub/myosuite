"""Write ``manifest.json`` into every ``checkpoints/<env_id>`` folder of a baselines checkout.

    python scripts/write_checkpoint_manifests.py <baselines dir> [--table docs/baseline_checkpoints.md]

The manifests record the env contract hash (see ``myosuite.utils.checkpoint_manifest``), the MyoSuite
version and commit, the deterministic success from the table and, for envs whose body comes from
``musclemimic_models``, that release (run with ``myosuite[musclemimic]`` installed, as for training). Upload the result to the Hugging
Face repo, then tag that commit with the release (for example ``v3.0``).
"""

import argparse
import re
from pathlib import Path

import gymnasium as gym

import myosuite  # noqa: F401  (registers the envs)
from myosuite.utils.checkpoint_manifest import write_manifest

ROW = re.compile(r"^\|\s*(\S+-v\d+)\s*\|\s*`(model_\d+\.pt)`\s*\|\s*([\d.]+)%", re.M)


def main() -> None:
    """Write the manifests."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root", type=Path, help="directory containing checkpoints/<env_id>/"
    )
    parser.add_argument(
        "--table", type=Path, default=Path("docs/baseline_checkpoints.md")
    )
    args = parser.parse_args()

    success = {env: float(pct) for env, _, pct in ROW.findall(args.table.read_text())}
    for folder in sorted((args.root / "checkpoints").iterdir()):
        if not folder.is_dir() or not any(folder.glob("model_*.pt")):
            continue
        if folder.name in gym.registry:
            path = write_manifest(folder, folder.name, success=success.get(folder.name))
        else:  # e.g. the MuscleMimic single-clip policy: mjlab-only, scored by clip completion
            env_id = folder.name.split("-walking")[0]
            path = write_manifest(
                folder,
                env_id,
                contract={},
                note="mjlab-only single-clip policy, scored by clip completion; no CPU env contract",
            )
        print(path)


if __name__ == "__main__":
    main()
