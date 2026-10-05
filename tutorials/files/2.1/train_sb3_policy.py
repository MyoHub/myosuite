"""Train the elbow policy of notebook 2.1 from the command line (wraps scripts/train_sb3.py).

    python tutorials/files/2.1/train_sb3_policy.py                   # 500k steps, saves ElbowPose_policy.zip here
    python tutorials/files/2.1/train_sb3_policy.py --timesteps 1000000 --tensorboard logs/tb

Any ``scripts/train_sb3.py`` flag is accepted and overrides these defaults.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))

from train_sb3 import main  # noqa: E402

if __name__ == "__main__":
    defaults = [
        "myoElbowPose1D6MRandom-v0",
        "--timesteps",
        "500000",
        "--out",
        str(HERE / "ElbowPose_policy"),
    ]
    main(defaults + sys.argv[1:])
