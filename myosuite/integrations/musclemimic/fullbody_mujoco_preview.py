# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""MuJoCo viewer preview for MyoFullBody without MuscleMimic training code.

This is **not** HF policy evaluation: it compiles the same MJCF as MuscleMimic
(needs ``musclemimic_models`` / ``pip install 'MyoSuite[musclemimic]'``) and
runs the simulator with small random muscle excitations so you can inspect
the model. Used when ``fullbody_eval_cli`` cannot delegate to upstream
``fullbody.eval``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

import mujoco
import numpy as np

from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.integrations.musclemimic.playback_runners import (
    run_passive_viewer_loop,
)

logger = logging.getLogger(__name__)


def _parse_preview_argv(argv: Sequence[str]) -> argparse.Namespace:
    """Parse preview args and reject unsupported unknown flags."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--n_steps", type=int, default=1000)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--path", type=str, default="")
    p.add_argument("--motion_path", type=str, default="")
    return p.parse_args(list(argv))


def main(argv: list[str] | None = None) -> int:
    """Open a passive MuJoCo viewer on the MyoSuite MyoFullBody model."""
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_preview_argv(argv)

    logging.basicConfig(level=logging.INFO)
    logger.warning(
        "MyoSuite MuJoCo preview mode: physics + random muscle activity only. "
        "HF checkpoint / PPO policy is not loaded. "
        "Install MuscleMimic or set MYOSUITE_MUSCLEMIMIC_ROOT for full eval."
    )
    if args.path:
        logger.info("(Ignored for preview) --path %s", args.path)
    if args.motion_path:
        logger.info("(Ignored for preview) --motion_path %s", args.motion_path)

    try:
        cfg = default_mimic_fullbody_config()
        model, _spec, _xml = compile_mimic_fullbody_mjmodel(cfg)
    except ImportError as err:
        logger.error(
            "MyoFullBody MJCF requires packaged assets: pip install "
            "'MyoSuite[musclemimic]' (provides musclemimic_models). "
            "Original error: %s",
            err,
        )
        return 2

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(int(args.eval_seed))
    nu = int(model.nu)
    n_steps = max(1, int(args.n_steps))

    logger.info(
        "MuJoCo viewer open (passive). Close window or ESC to exit. "
        "steps=%d seed=%d",
        n_steps,
        args.eval_seed,
    )

    def _step(_idx: int, _model: mujoco.MjModel, _data: mujoco.MjData) -> None:
        if nu > 0:
            _data.ctrl[:] = rng.uniform(-0.15, 0.15, size=nu)
        mujoco.mj_step(_model, _data)

    run_passive_viewer_loop(
        model=model,
        data=data,
        n_steps=n_steps,
        step_fn=_step,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
