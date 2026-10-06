# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Download MuscleMimic-style demo motions from Hugging Face.

**Default:** uses :mod:`myosuite.integrations.musclemimic.hf_demo_cache` (only
needs ``huggingface_hub`` — **no** upstream ``musclemimic`` package).

**Fallbacks:** upstream ``musclemimic.utils.demo_cache`` if installed, then
``MYOSUITE_MUSCLEMIMIC_ROOT`` + subprocess (MuscleMimic tree's Python).
"""

from __future__ import annotations

import logging
import argparse

from myosuite.core.subprocess_orchestration import run_command
from myosuite.integrations.musclemimic.runtime import (
    build_demo_cache_command,
    resolve_musclemimic_root_from_env,
)

logger = logging.getLogger(__name__)

_HF_INSTALL = (
    "Install Hugging Face client: `pip install huggingface_hub` or "
    "`pip install 'MyoSuite[musclemimic]'`. "
    "You do **not** need the upstream MuscleMimic repo for this step."
)

_ALL_FAILED = (
    "Demo cache could not run. Install `huggingface_hub` (see above), or "
    "set `MYOSUITE_MUSCLEMIMIC_ROOT` to a MuscleMimic checkout with a "
    "working venv."
)


def main(argv: list[str] | None = None) -> int:
    """Run demo motion downloads for selected environment."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env_name",
        default="MyoFullBody",
        choices=["MyoFullBody", "MyoBimanualArm"],
        help="Environment demo set to download.",
    )
    args = parser.parse_args(argv)
    env_name = args.env_name

    try:
        from myosuite.integrations.musclemimic.hf_demo_cache import (
            setup_demo as native_setup,
        )
    except ImportError as err:
        logger.debug("MyoSuite hf_demo_cache import failed: %s", err)
        native_setup = None

    if native_setup is not None:
        try:
            native_setup(env_name)
            return 0
        except ImportError as err:
            msg = str(err).lower()
            if "huggingface" in msg:
                logger.error("%s Original error: %s", _HF_INSTALL, err)
                return 2
            raise
        except Exception:
            logger.exception("Demo cache setup failed.")
            return 1

    try:
        from musclemimic.utils.demo_cache import setup_demo as upstream_setup
    except ImportError:
        pass
    else:
        logger.info("Running demo cache via upstream musclemimic package.")
        try:
            upstream_setup(env_name)
        except Exception:
            logger.exception("Demo cache setup failed.")
            return 1
        return 0

    try:
        exec_cwd = resolve_musclemimic_root_from_env()
    except FileNotFoundError as err:
        logger.error("%s", err)
        return 2
    if exec_cwd is None:
        logger.error("%s %s", _HF_INSTALL, _ALL_FAILED)
        return 2

    cmd, env = build_demo_cache_command(exec_cwd, env_name=env_name)
    logger.info("Running demo cache via subprocess, cwd=%s", exec_cwd)
    return run_command(cmd=cmd, cwd=exec_cwd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
