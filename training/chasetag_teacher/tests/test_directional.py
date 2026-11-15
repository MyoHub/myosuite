# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The directional walker's observation, its ChaseTag adapter and the ONNX export (no data needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from chasetag_teacher.directional import (
    DIR_OBS_DIM,
    FBVS_OBS_DIM,
    ChaseTagHeadingAdapter,
    DirectionalScene,
    directional_obs,
    directional_obs_torch,
    export_chasetag_onnx,
    role_heading_torch,
)
from myosuite.integrations.musclemimic.steering import aim_heading_xy
from myosuite.envs.myo.tasks.mimic.policy import ActorCritic


@pytest.fixture(scope="module", name="scene")
def scene_fixture() -> DirectionalScene:
    return DirectionalScene()


def _random_state(scene: DirectionalScene, rng: np.random.Generator) -> None:
    scene.reset(float(rng.uniform(-math.pi, math.pi)), rng, joint_noise=0.05)
    d = scene.data
    d.qpos[:2] = rng.normal(size=2)
    d.qvel[:] = rng.normal(size=scene.nv)
    d.act[:] = rng.uniform(size=d.act.shape)


def test_layout(scene: DirectionalScene) -> None:
    rng = np.random.default_rng(0)
    _random_state(scene, rng)
    d = scene.data
    o = directional_obs(d.qpos, d.qvel, d.act, 0.3)
    assert o.shape == (DIR_OBS_DIM,) and o.dtype == np.float32
    np.testing.assert_allclose(o[:82], d.qpos[7:], atol=1e-6)
    np.testing.assert_allclose(o[82:164], d.qvel[6:], atol=1e-5)
    np.testing.assert_allclose(o[164:518], d.act, atol=1e-6)
    assert abs(np.linalg.norm(o[520:522]) - 1.0) < 1e-6
    np.testing.assert_allclose(o[524:527], d.qvel[3:6], atol=1e-5)


@pytest.mark.parametrize("delta", [0.7, -2.4, math.pi])
def test_yaw_invariance(scene: DirectionalScene, delta: float) -> None:
    """Turning the body, its velocity and the heading about z leaves the observation unchanged."""
    rng = np.random.default_rng(1)
    _random_state(scene, rng)
    d = scene.data
    q, v, a = d.qpos.copy(), d.qvel.copy(), d.act.copy()
    theta = 1.1
    o = directional_obs(q, v, a, theta)
    c, s = math.cos(delta), math.sin(delta)
    q2, v2 = q.copy(), v.copy()
    q2[:2] = [c * q[0] - s * q[1], s * q[0] + c * q[1]]
    v2[:2] = [
        c * v[0] - s * v[1],
        s * v[0] + c * v[1],
    ]  # world linear velocity; angular (3:6) is body frame
    w, x, y, z = q[3:7]
    h, hs = math.cos(delta / 2), math.sin(delta / 2)
    q2[3:7] = [h * w - hs * z, h * x - hs * y, h * y + hs * x, h * z + hs * w]
    np.testing.assert_allclose(directional_obs(q2, v2, a, theta + delta), o, atol=2e-5)


def test_role_heading_matches_planner() -> None:
    """The batched heading equals the teacher planner's (fence handling included)."""
    rng = np.random.default_rng(2)
    me = rng.uniform(-6.0, 6.0, size=(400, 2))
    opp = rng.uniform(-6.0, 6.0, size=(400, 2))
    runner = (rng.random(400) > 0.5).astype(np.float64)
    got = role_heading_torch(
        torch.as_tensor(me), torch.as_tensor(opp - me), torch.as_tensor(runner)[:, None]
    )
    want = np.stack(
        [aim_heading_xy(m, o, pursue=not r) for m, o, r in zip(me, opp, runner)]
    )
    np.testing.assert_allclose(got.numpy(), want, atol=1e-9)


def test_adapter_and_onnx(tmp_path) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(3)
    obs = rng.normal(size=(32, FBVS_OBS_DIM))
    qt = rng.normal(size=(32, 4))
    obs[:, 3:7] = qt / np.linalg.norm(qt, axis=1, keepdims=True)
    obs[:, 544:546] = np.eye(2)[rng.integers(0, 2, 32)]
    t = torch.as_tensor(obs, dtype=torch.float32)
    x = ChaseTagHeadingAdapter()(t)
    heading = role_heading_torch(t[:, 531:533], t[:, 537:539], t[:, 545:546])
    np.testing.assert_allclose(
        x.numpy(),
        directional_obs_torch(t[:, :89], t[:, 89:177], t[:, 177:531], heading).numpy(),
    )
    pt = tmp_path / "walker.pt"
    torch.manual_seed(0)
    torch.save(ActorCritic(DIR_OBS_DIM, 354).state_dict(), pt)
    path = export_chasetag_onnx(pt, tmp_path / "walker.onnx")
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = s.run(None, {"obs": obs.astype(np.float32)})[0]
    from chasetag_teacher.directional import load_walker

    with torch.no_grad():
        want = load_walker(pt)(x).numpy()
    assert got.shape == (32, 354)
    np.testing.assert_allclose(got, want, atol=1e-4)
