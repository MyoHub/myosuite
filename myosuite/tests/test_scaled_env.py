# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``make_env(env_id, scales=..., total_mass=...)`` on the CPU: a subject's leg walk."""

from __future__ import annotations

import warnings
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.body_scaling import scale_model
from myosuite.core.config import EnvConfig
from myosuite.core.model_builder import ModelBuilder
from myosuite.utils import gym
from myosuite.utils.checkpoint_manifest import (
    check_contract,
    read_manifest,
    write_manifest,
)

pytestmark = pytest.mark.tier1

ENV_ID = "myoLegWalk-v0"
SCALES = {"femur_r": [1.02, 1.07, 0.98], "tibia_l": 1.04}
# Arrays the scaling changes; the env hides its terrain geom after compiling.
SCALED_FIELDS = (
    "body_pos",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "site_pos",
    "jnt_pos",
    "jnt_range",
    "eq_data",
    "actuator_lengthrange",
    "mesh_vert",
    "qpos0",
    "key_qpos",
)


def _model_bytes(model: mujoco.MjModel) -> bytes:
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer.tobytes()


def _builder() -> ModelBuilder:
    return ModelBuilder.from_xml_file(gym.spec(ENV_ID).kwargs["model_path"])


def _floor_clearance(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    """Distance of the body's collision geoms to the floor (MuJoCo's collider)."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    floor = model.geom("floor").id
    return min(
        mujoco.mj_geomDistance(model, data, g, floor, 10.0, None)
        for g in range(model.ngeom)
        if model.geom_bodyid[g] > 0 and model.geom_contype[g]
    )


def test_nominal_env_is_bit_identical() -> None:
    """Without scales the env builds its model exactly as before."""
    expected, _ = _builder().build()
    terrain = expected.geom("terrain").id
    expected.geom_rgba[terrain, 3] = 0.0
    expected.geom_pos[terrain] = [0.0, 0.0, -10.0]
    ones = {name: 1.0 for name in ("root", "femur_r", "tibia_l")}
    for kwargs in (
        {},
        {"scales": None, "total_mass": None},
        {"scales": {}},
        {"scales": ones},
    ):
        env = make_env(ENV_ID, **kwargs)
        try:
            assert _model_bytes(env.unwrapped.model) == _model_bytes(expected), kwargs
        finally:
            env.close()


def test_subject_env_builds_the_scale_model_model() -> None:
    nominal = make_env(ENV_ID)
    env = make_env(ENV_ID, scales=SCALES, total_mass=70.0)
    try:
        expected = scale_model(_builder().build_spec(), SCALES, total_mass=70.0)
        model = env.unwrapped.model
        for field in SCALED_FIELDS:
            np.testing.assert_array_equal(
                getattr(model, field), getattr(expected, field), err_msg=field
            )
        assert model.body_mass.sum() == pytest.approx(70.0)
        assert env.observation_space == nominal.observation_space
        assert env.action_space == nominal.action_space
    finally:
        nominal.close()
        env.close()


def test_subject_env_resets_reproducibly_on_its_aligned_keyframe() -> None:
    nominal = make_env(ENV_ID)
    env = make_env(ENV_ID, scales={b: 1.1 for b in ("root", "femur_r", "tibia_r")})
    try:
        first, _ = env.reset(seed=5)
        env.step(np.full(env.action_space.shape, 0.2, dtype=np.float32))
        second, _ = env.reset(seed=5)
        np.testing.assert_array_equal(first, second)
        assert first.dtype == np.float32 and np.all(np.isfinite(first))
        model, data = env.unwrapped.model, env.unwrapped.data
        np.testing.assert_array_equal(data.qpos, model.key_qpos[2])
        # The longer right leg still starts as far above the floor as nominal.
        ref = nominal.unwrapped.model
        assert _floor_clearance(model, data.qpos) == pytest.approx(
            _floor_clearance(ref, ref.key_qpos[2]), abs=1e-6
        )
        assert len(env.step(np.zeros(env.action_space.shape, np.float32))) == 5
    finally:
        nominal.close()
        env.close()


def test_scales_are_ordinary_task_kwargs(tmp_path: Path) -> None:
    """EnvConfig.task_kwargs and an OpenSim scale file reach the same model."""
    by_kwarg = make_env(ENV_ID, scales=SCALES)
    by_config = make_env(EnvConfig(ENV_ID, task_kwargs={"scales": SCALES}))
    scale_set = tmp_path / "subject.xml"
    scale_set.write_text(
        "<OpenSimDocument><ScaleSet><objects><Scale><scales>1.03 1.05 1.0</scales>"
        "<segment>femur_l</segment></Scale></objects></ScaleSet></OpenSimDocument>"
    )
    by_file = make_env(ENV_ID, scales=scale_set)
    try:
        assert _model_bytes(by_kwarg.unwrapped.model) == _model_bytes(
            by_config.unwrapped.model
        )
        expected = scale_model(_builder().build_spec(), scale_set)
        np.testing.assert_array_equal(
            by_file.unwrapped.model.body_pos, expected.body_pos
        )
    finally:
        for env in (by_kwarg, by_config, by_file):
            env.close()


@pytest.mark.parametrize(
    "env_id",
    ["myoLegRoughTerrainWalk-v0", "myoLegStandRandom-v0", "myoElbowPose1D6MRandom-v0"],
)
def test_tasks_without_a_subject_model_refuse_scales(env_id: str) -> None:
    with pytest.raises(NotImplementedError, match="subject"):
        make_env(env_id, scales=SCALES)
    with pytest.raises(NotImplementedError, match="subject"):
        make_env(env_id, total_mass=70.0)


def test_mjx_refuses_scales() -> None:
    with pytest.raises(NotImplementedError, match="cpu and mjlab"):
        make_env(ENV_ID, backend="mjx", scales=SCALES)


def test_manifest_records_the_subject_and_a_strict_check_rejects_another(
    tmp_path: Path,
) -> None:
    write_manifest(tmp_path, ENV_ID, scales=SCALES, total_mass=70.0)
    subject = read_manifest(tmp_path)["subject"]
    assert subject["scales"] == SCALES and subject["total_mass"] == 70.0
    assert len(subject["model_digest"]) == 64
    assert check_contract(ENV_ID, tmp_path, scales=SCALES, total_mass=70.0, strict=True)
    # The same body written differently is the same subject.
    same = {"femur_r": np.array([1.02, 1.07, 0.98]), "tibia_l": [1.04] * 3}
    assert check_contract(ENV_ID, tmp_path, scales=same, total_mass=70.0, strict=True)
    other = {"femur_r": 1.03}
    with pytest.raises(ValueError, match="bodies differ"):
        check_contract(ENV_ID, tmp_path, scales=other, strict=True)
    with pytest.warns(UserWarning, match="bodies differ"):
        assert not check_contract(ENV_ID, tmp_path, scales=other)
    with pytest.raises(ValueError, match="nominal body"):  # subject policy, nominal env
        check_contract(ENV_ID, tmp_path, strict=True)


def test_nominal_checkpoints_are_checked_as_before(tmp_path: Path) -> None:
    write_manifest(tmp_path, ENV_ID)
    assert "subject" not in read_manifest(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_contract(ENV_ID, tmp_path)
        assert check_contract(ENV_ID, tmp_path, strict=True)
        assert check_contract(ENV_ID, tmp_path / "no_manifest")
    with pytest.raises(ValueError, match="nominal body"):  # nominal policy, subject env
        check_contract(ENV_ID, tmp_path, scales=SCALES, strict=True)
