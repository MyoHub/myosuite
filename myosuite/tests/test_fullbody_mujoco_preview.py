# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for fullbody_mujoco_preview."""

from __future__ import annotations

import pytest

from myosuite.integrations.musclemimic.fullbody_eval_cli import (
    _parse_mode_config,
    _preview_requested,
)
from myosuite.integrations.musclemimic.fullbody_mujoco_preview import (
    _parse_preview_argv,
)


def test_preview_requested_requires_mujoco_flags() -> None:
    """MuJoCo preview path needs both viewer flags."""
    assert _preview_requested(
        _parse_mode_config(["--path", "x", "--use_mujoco", "--mujoco_viewer"])
    )
    assert _preview_requested(_parse_mode_config(["--use_mujoco", "--mujoco_viewer"]))
    assert not _preview_requested(_parse_mode_config(["--path", "x", "--use_mujoco"]))
    assert not _preview_requested(_parse_mode_config(["--path", "x"]))


def test_parse_preview_argv_ignores_unknown() -> None:
    """Unknown flags should fail fast in preview parser."""
    with pytest.raises(SystemExit):
        _parse_preview_argv(
            [
                "--path",
                "hf://x",
                "--motion_path",
                "KIT/a",
                "--stochastic",
                "--n_steps",
                "32",
                "--eval_seed",
                "1",
                "--extra-upstream-flag",
            ]
        )


def test_parse_preview_argv_parses_known_flags() -> None:
    """Known preview flags parse successfully."""
    ns = _parse_preview_argv(
        [
            "--path",
            "hf://x",
            "--motion_path",
            "KIT/a",
            "--n_steps",
            "32",
            "--eval_seed",
            "1",
        ]
    )
    assert ns.n_steps == 32
    assert ns.eval_seed == 1
    assert ns.path == "hf://x"
    assert ns.motion_path == "KIT/a"


def test_preview_compiles_mjmodel() -> None:
    """Preview path should compile the same MyoFullBody CPU model."""
    from myosuite.tests.support.optional_deps import require_musclemimic_models

    require_musclemimic_models()
    import mujoco

    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )

    cfg = default_mimic_fullbody_config()
    model, _spec, _xml = compile_mimic_fullbody_mjmodel(cfg)
    assert isinstance(model, mujoco.MjModel)
    assert model.nu > 0
