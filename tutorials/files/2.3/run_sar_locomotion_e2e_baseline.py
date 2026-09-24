# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""RL-E2E baseline companion to run_sar_full.py, for the SAR-RL vs RL-E2E
comparison plotted by ``plot_results(experiment='locomotion', ...)``.

Trains directly on the target terrain task without any SAR bottleneck.
Writes into the same ``sar_outputs/`` directory as run_sar_full.py so
``plot_results`` finds both runs.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "egl"
    if os.environ["MUJOCO_GL"] == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_REPO_ROOT = Path(__file__).parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import myosuite  # noqa: E402

from run_sar_full import SEED, TARGET_ENV, SAR_RL_STEPS, train  # noqa: E402

myosuite.register_all_envs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    os.makedirs("sar_outputs", exist_ok=True)
    os.chdir("sar_outputs")
    log.info("Working directory: %s", os.getcwd())

    model_path = Path(f"RL-E2E_model_{TARGET_ENV}_{SEED}.zip")
    env_path = Path(f"RL-E2E_env_{TARGET_ENV}_{SEED}")
    if model_path.exists() and env_path.exists():
        log.info("RL-E2E baseline already trained; found %s and %s", model_path, env_path)
        return
    train(TARGET_ENV, "RL-E2E", SAR_RL_STEPS, SEED)
    log.info("RL-E2E baseline complete.")


if __name__ == "__main__":
    main()
