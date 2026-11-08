# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Live shadow model: composition, fatigue feedback, tracking gaps and logging."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.physics.shadow_model import ShadowModel, ShadowModelConfig
from myosuite.physics.static_optimization import StaticOptimizer
from myosuite.physics.streaming_kinematics import StreamingKinematics

pytestmark = pytest.mark.tier1


@pytest.fixture(scope="module")
def elbow() -> mujoco.MjModel:
    env = make_env("myoElbowPose1D6MRandom-v0")
    yield env.unwrapped.model
    env.close()


def _hold(model, load: float, limits: bool, seconds: float = 120.0, rate: float = 20.0):
    """Hold the forearm at 69 degrees against a hand load [N m]."""
    shadow = ShadowModel(
        model, ShadowModelConfig(window=5, fatigue_limits_activation=limits)
    )
    frames = [
        shadow.push(k / rate, np.array([1.2]), np.array([load]))
        for k in range(int(seconds * rate))
    ]
    frames = [f for f in frames if f is not None]
    act = np.array([f.optimization.activation for f in frames])
    mf = np.array([f.fatigue for f in frames])
    return act, mf, np.array([f.optimization.reserve[0] for f in frames])


def test_shadow_model_equals_its_parts(elbow: mujoco.MjModel) -> None:
    t = np.arange(0.0, 2.0, 1 / 90)
    qpos = (1.0 + 0.6 * np.sin(2.0 * t))[:, None]
    shadow = ShadowModel(elbow, ShadowModelConfig(fatigue=False))
    stream, so = StreamingKinematics(elbow, window=15), StaticOptimizer(elbow)
    for ti, q in zip(t, qpos):
        frame, state = shadow.push(ti, q), stream.push(ti, q)
        assert (frame is None) == (state is None)
        if state is not None:
            assert frame.state.time == state.time and frame.fatigue is None
            np.testing.assert_array_equal(
                frame.optimization.activation,
                so.solve(state.qpos, state.qvel, state.qacc).activation,
            )


def test_fatigue_caps_activation_and_shifts_the_load(elbow: mujoco.MjModel) -> None:
    act, mf, reserve = _hold(elbow, -20.0, limits=True)
    act0, mf0, reserve0 = _hold(elbow, -20.0, limits=False)
    assert mf[-1].max() > 0.5  # two minutes at about half of maximum
    # A fatigued muscle never exceeds what it has left; without the cap it does.
    assert (act[1:] - (1.0 - mf[:-1])).max() < 1e-6
    assert (act0[1:] - (1.0 - mf0[:-1])).max() > 0.15
    bic_long, bra = elbow.actuator("BIClong").id, elbow.actuator("BRA").id
    assert act[-1, bic_long] < act0[-1, bic_long] - 0.1
    assert act[-1, bra] > act0[-1, bra] + 0.05
    assert reserve[-1] > 1.0 > reserve0[-1]  # the fatigued arm starts to fall short


def test_tracking_gap_restarts_kinematics_and_holds_fatigue(
    elbow: mujoco.MjModel,
) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5))
    pose, load = np.array([1.2]), np.array([-20.0])
    before = [shadow.push(k / 20, pose, load) for k in range(40)][-1]
    after = [
        shadow.push(3.0 + k / 20, pose, load) for k in range(5)
    ]  # 1 s without tracking
    assert all(f is None for f in after[:4]) and after[4] is not None
    np.testing.assert_array_equal(after[4].fatigue, before.fatigue)


def test_external_load_belongs_to_the_estimated_frame(elbow: mujoco.MjModel) -> None:
    loaded, free = (
        ShadowModel(elbow, ShadowModelConfig(fatigue=False)),
        ShadowModel(elbow, ShadowModelConfig(fatigue=False)),
    )
    for k in range(40):
        q, ext = np.array([1.0 + 0.01 * k]), np.array([-0.5 * k])
        a, b = loaded.push(k / 50, q, ext), free.push(k / 50, q)
        if a is not None:
            k_est = round(a.state.time * 50)
            np.testing.assert_allclose(
                a.optimization.joint_torque - b.optimization.joint_torque, [0.5 * k_est]
            )


def test_log_records_every_frame(elbow: mujoco.MjModel, tmp_path: Path) -> None:
    body = elbow.body(elbow.nbody - 1).name
    config = ShadowModelConfig(window=5, log_contact_bodies=(body,))
    with ShadowModel(
        elbow, config, log_dir=tmp_path, metadata={"participant": "P00"}
    ) as shadow:
        frames = [shadow.push(k / 20, np.array([1.2])) for k in range(30)]
    rows = list(csv.reader(open(tmp_path / "shadow_model.csv", encoding="utf-8")))
    header, rows = rows[0], rows[1:]
    assert len(rows) == sum(f is not None for f in frames) == 26
    names = [elbow.actuator(i).name for i in range(elbow.nu)]
    assert header[5:] == [f"a_{n}" for n in names] + [f"mf_{n}" for n in names] + [
        f"contact_{body}_{x}" for x in "xyz"
    ]
    assert all(len(r) == len(header) for r in rows)
    assert float(rows[-1][0]) == frames[-1].state.time
    logged = np.array([float(v) for v in rows[-1][5 : 5 + 2 * elbow.nu]])
    np.testing.assert_allclose(logged[: elbow.nu], frames[-1].optimization.activation)
    np.testing.assert_allclose(logged[elbow.nu :], frames[-1].fatigue)
    record = json.loads((tmp_path / "shadow_model.json").read_text(encoding="utf-8"))
    buffer = np.zeros(mujoco.mj_sizeModel(elbow), dtype=np.uint8)
    mujoco.mj_saveModel(elbow, None, buffer)
    assert record["model_sha256"] == hashlib.sha256(buffer.tobytes()).hexdigest()
    assert record["participant"] == "P00" and record["config"]["window"] == 5
