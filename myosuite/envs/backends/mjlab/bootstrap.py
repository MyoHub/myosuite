# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Entry-point module that registers all MyoSuite twins with ``mjlab.tasks``.

Kept separate from :mod:`myosuite.envs.backends.mjlab` so co-located task modules
can import shared ``backends.mjlab`` builders in any order without triggering
recursive task registration. If a co-located task module is the very first module
to import ``mjlab`` (so ``mjlab``'s entry-point loader enters this module while
that task module is still mid-import), registration is deferred until the
outermost initializing ``myosuite.envs`` module finishes its definitions and
calls :func:`finish_deferred_mjlab_bootstrap`.
"""

from __future__ import annotations

import logging
import sys

_DEFERRED_BOOTSTRAP = False
_BOOTSTRAP_RUNNING = False


def _any_other_envs_module_initializing(*, exclude: str) -> bool:
    for mod_name, mod in list(sys.modules.items()):
        if mod_name in (exclude, __name__):
            continue
        if not (
            mod_name == "myosuite.envs.backends.mjlab"
            or mod_name.startswith("myosuite.envs.tasks.")
            or mod_name.startswith("myosuite.envs.backends.mjlab.")
        ):
            continue
        spec = getattr(mod, "__spec__", None)
        if getattr(spec, "_initializing", False):
            return True
    return False


def _run_bootstrap() -> None:
    global _BOOTSTRAP_RUNNING
    if _BOOTSTRAP_RUNNING:
        return
    _BOOTSTRAP_RUNNING = True
    try:
        from myosuite.envs.backends.mjlab.register_mjlab_tasks import (
            bootstrap_myosuite_mjlab_registry,
        )

        bootstrap_myosuite_mjlab_registry()
    except ImportError:
        # Package import must succeed on CPU-only installs without torch / mjlab.
        pass
    except Exception:
        logging.getLogger(__name__).warning(
            "myosuite mjlab backend: error during task registration",
            exc_info=True,
        )
    finally:
        _BOOTSTRAP_RUNNING = False


def finish_deferred_mjlab_bootstrap(caller_module: str) -> None:
    """Run deferred ``mjlab`` task registration once the outermost task module finishes definitions."""
    global _DEFERRED_BOOTSTRAP
    if not _DEFERRED_BOOTSTRAP or _BOOTSTRAP_RUNNING:
        return
    if _any_other_envs_module_initializing(exclude=caller_module):
        return
    _DEFERRED_BOOTSTRAP = False
    _run_bootstrap()


if _any_other_envs_module_initializing(exclude=__name__):
    _DEFERRED_BOOTSTRAP = True
else:
    _run_bootstrap()
