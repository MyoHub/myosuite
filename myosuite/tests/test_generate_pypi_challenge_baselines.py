# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``scripts/generate_pypi_challenge_baselines.py`` (no PyPI install)."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import venv
from pathlib import Path

import numpy as np
import pytest

from scripts.generate_pypi_challenge_baselines import _INNER_SCRIPT, _venv_python
from myosuite import make_env

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.tier1
def test_venv_python_is_the_interpreter_of_the_created_venv(tmp_path: Path) -> None:
    """``Scripts\\python.exe`` on Windows, ``bin/python`` elsewhere."""
    venv_dir = tmp_path / "venv"
    venv.create(str(venv_dir), with_pip=False)

    python = _venv_python(venv_dir)
    prefix = subprocess.run(
        [str(python), "-c", "import sys; print(sys.prefix)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert Path(prefix).resolve() == venv_dir.resolve()


@pytest.mark.tier2
def test_baseline_actions_are_the_seeded_stream_the_regression_test_replays(
    tmp_path: Path,
) -> None:
    """The recorded actions are ``default_rng(seed).uniform(low, high)`` draws, as in
    ``test_reward_mean_sign_vs_pypi``, not the never-seeded ``action_space.sample()``."""

    env_id, n_steps, seed = "myoElbowPose1D6MRandom-v0", 5, 42
    script = tmp_path / "inner.py"
    script.write_text(_INNER_SCRIPT, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(script), json.dumps([env_id]), str(n_steps), str(seed)]
        + [str(tmp_path)],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
    )
    baseline = pickle.loads((tmp_path / f"{env_id}.pkl").read_bytes())

    env = make_env(env_id)
    space = env.action_space
    env.close()
    rng = np.random.default_rng(seed)
    expected = [
        rng.uniform(space.low, space.high, size=space.shape) for _ in range(n_steps)
    ]
    np.testing.assert_array_equal([s["action"] for s in baseline["steps"]], expected)
