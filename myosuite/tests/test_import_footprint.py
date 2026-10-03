# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""What ``import myosuite`` and the colored-noise challenge envs import.

Every subprocess or vectorised-env worker pays these imports. The musclemimic
package loads its submodules on first access (PEP 562), and the Soccer /
ChaseTag opponents draw their noise with ``colorednoise``: ``import pink``
loads stable-baselines3 and torch whenever they are installed.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

_REPO = Path(__file__).resolve().parents[2]
_MUSCLEMIMIC = "myosuite.integrations.musclemimic"
# Where the names of the musclemimic package are defined.
_MUSCLEMIMIC_SOURCES = (
    f"{_MUSCLEMIMIC}.citation",
    f"{_MUSCLEMIMIC}.bimanual_model",
    f"{_MUSCLEMIMIC}.myotorso_bimanual_model",
    f"{_MUSCLEMIMIC}.fullbody_model",
    f"{_MUSCLEMIMIC}.fullbody_native_playback",
    "myosuite.core.playback_contract",
)


def _loaded_after(code: str, modules: tuple[str, ...]) -> list[str]:
    """Run *code* in a fresh interpreter; return which of *modules* it imported."""
    script = (
        f"{code}\nimport json, sys\n"
        f"print(json.dumps([m for m in {list(modules)!r} if m in sys.modules]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_import_myosuite_skips_heavy_modules() -> None:
    """Env registration loads no policy runtime, scipy.spatial or ml_collections."""
    heavy = (
        f"{_MUSCLEMIMIC}.fullbody_native_playback",
        f"{_MUSCLEMIMIC}.fullbody_local_policy",
        f"{_MUSCLEMIMIC}.fullbody_model",
        "scipy.spatial",
        "ml_collections",
        "torch",
    )
    assert _loaded_after("import myosuite", heavy) == []


@pytest.mark.parametrize(
    "env_id", ["myoChallengeSoccerP1-v0", "myoChallengeChaseTagP1-v0"]
)
def test_colored_noise_envs_skip_rl_frameworks(env_id: str) -> None:
    """make + reset + steps load neither pink, stable-baselines3 nor torch."""
    code = (
        "import gymnasium as gym, numpy as np, myosuite\n"
        f"env = gym.make({env_id!r})\n"
        "env.reset(seed=0)\n"
        "for _ in range(3):\n"
        "    env.step(np.zeros(env.action_space.shape, dtype=np.float32))\n"
    )
    assert _loaded_after(code, ("pink", "stable_baselines3", "torch")) == []


def test_musclemimic_submodules_load_on_first_access() -> None:
    """The package and one of its names leave the other submodules unimported.

    Submodules stay attributes of the package, as with the former eager imports.
    """
    fullbody = f"{_MUSCLEMIMIC}.fullbody_model"
    checkpoint_io = f"{_MUSCLEMIMIC}.fullbody_checkpoint_io"
    code = (
        f"import sys, {_MUSCLEMIMIC} as package\n"
        f"assert {fullbody!r} not in sys.modules\n"
        "package.build_mimic_fullbody_spec\n"
        f"assert {fullbody!r} in sys.modules\n"
        f"assert package.fullbody_checkpoint_io is sys.modules[{checkpoint_io!r}]\n"
    )
    playback = f"{_MUSCLEMIMIC}.fullbody_native_playback"
    assert _loaded_after(code, (playback, "scipy.spatial")) == []


def test_musclemimic_exports_are_the_submodule_objects() -> None:
    """Every name in ``__all__`` is the object its defining submodule holds."""
    package = importlib.import_module(_MUSCLEMIMIC)
    sources = [importlib.import_module(m) for m in _MUSCLEMIMIC_SOURCES]
    for name in package.__all__:
        owners = [m for m in sources if hasattr(m, name)]
        assert owners, name
        assert all(getattr(package, name) is getattr(m, name) for m in owners), name
    assert set(package.__all__) <= set(dir(package))
    with pytest.raises(AttributeError):
        getattr(package, "not_an_export")
