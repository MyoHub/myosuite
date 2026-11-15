# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The task view uses its command's exact configured body without compiling again."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

pytestmark = pytest.mark.tier2


@pytest.mark.parametrize(
    "self_collisions,force_scale", [(True, None), (False, None), (True, 0.35)]
)
def test_task_view_reuses_the_configured_command_model(
    clips: Path, self_collisions: bool, force_scale: float | None, monkeypatch: Any
) -> None:
    from supermimic.task import TaskView

    from myosuite import make_env
    from myosuite.core.config import EnvConfig
    from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_body
    from myosuite.envs.wrappers import SarcopeniaWrapper

    features = (
        ()
        if force_scale is None
        else ((SarcopeniaWrapper, {"force_scale": force_scale}),)
    )
    env = make_env(
        EnvConfig(
            "myoMimic-v0",
            backend="mjlab",
            num_envs=1,
            features=features,
            task_kwargs={
                "overrides": {
                    "motion.clips": str(clips / "clip0.npz"),
                    "self_collisions": self_collisions,
                }
            },
        ),
        device="cpu",
    )
    cmd = env.command_manager.get_term("motion")
    m = cmd.info.m
    gains, bias, pose = (
        m.actuator_gainprm.copy(),
        m.actuator_biasprm.copy(),
        m.qpos0.copy(),
    )

    def no_rebuild() -> Any:
        raise AssertionError("TaskView must not rebuild the body spec")

    monkeypatch.setattr(cmd.robot.cfg, "spec_fn", no_rebuild)
    try:
        view = TaskView(env)
        assert view.info is cmd.info
        assert view.kin.info.m is m and view.geo.kin.info.m is m
        torch.testing.assert_close(
            view.geo.F0, torch.tensor(gains[:, 2], dtype=torch.float32)
        )
        torch.testing.assert_close(
            view.geo.b_F0, torch.tensor(bias[:, 2], dtype=torch.float32)
        )
        expected_scale = 1.0 if force_scale is None else force_scale
        np.testing.assert_allclose(
            gains[:, 2],
            compiled_body("myofullbody").actuator_gainprm[:, 2] * expected_scale,
        )
        assert bool(m.npair) == self_collisions
        np.testing.assert_array_equal(m.actuator_gainprm, gains)
        np.testing.assert_array_equal(m.actuator_biasprm, bias)
        np.testing.assert_array_equal(m.qpos0, pose)
        env.reset()
        env.step(torch.full((1, m.nu), 0.1))
    finally:
        env.close()
