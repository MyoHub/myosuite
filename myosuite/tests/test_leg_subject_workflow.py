# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""The registered leg workflow consumes its fitted motion as the actual reset."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.core.subject_env import SubjectEnvConfig, subject_metadata
from myosuite.core.subject_model import ScaleEstimate, SubjectModel
from myosuite.integrations.capture.leg_workflow import ENV_ID, KNOWN_FACTORS, prepare
from myosuite.integrations.capture.retargeting import load_retargeted_clip

pytestmark = pytest.mark.tier2


def test_capture_estimate_and_target_fit_drive_the_registered_reset(
    tmp_path: Path,
) -> None:
    report = prepare(tmp_path)
    config = SubjectEnvConfig.load(tmp_path / "subject.json")
    assert config.reset_qpos is not None and config.reset_qvel is not None
    for name, factor in KNOWN_FACTORS.items():
        np.testing.assert_allclose(
            config.model.estimate.factors[name], [factor] * 3, atol=1e-10
        )
    assert report["quality"]["marker_max_m"] < 1e-4
    assert report["quality"]["converged_frames"] == 3
    env = make_env(EnvConfig(ENV_ID, subject=config))
    other = make_env(
        EnvConfig(
            ENV_ID,
            subject=replace(
                config,
                model=replace(
                    config.model,
                    estimate=ScaleEstimate(
                        {"femur_r": 1.1}, mass_kg=70.0, source="wrong subject"
                    ),
                ),
            ),
        )
    )
    try:
        clip = load_retargeted_clip(
            tmp_path / "retargeted_motion.npz",
            SubjectModel(
                env.unwrapped.model,
                env.unwrapped._mj_spec,
                subject_metadata(env),
            ),
        )
        _, info = env.reset(seed=7)
        np.testing.assert_allclose(env.unwrapped.data.qpos, clip.qpos[0], atol=1e-6)
        np.testing.assert_allclose(env.unwrapped.data.qvel, clip.qvel[0], atol=1e-6)
        assert info["subject_model"] == report["subject_model"]
        with pytest.raises(ValueError, match="different subject"):
            load_retargeted_clip(
                tmp_path / "retargeted_motion.npz",
                SubjectModel(
                    other.unwrapped.model,
                    other.unwrapped._mj_spec,
                    subject_metadata(other),
                ),
            )
    finally:
        env.close()
        other.close()
