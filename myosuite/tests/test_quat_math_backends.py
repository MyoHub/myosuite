# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Quaternion helpers on every backend: MuJoCo parity, batching, Euler conventions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.physics import quat_math

pytestmark = pytest.mark.tier1


@pytest.fixture(params=["numpy", "torch", "jax"])
def backend(request: pytest.FixtureRequest) -> SimpleNamespace:
    """Quat module, array conversions and tolerance of one backend (float32 for torch/jax)."""
    if request.param == "numpy":
        return SimpleNamespace(qm=quat_math, arr=np.asarray, np=np.asarray, atol=1e-9)
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from myosuite.physics import quat_math_torch

        return SimpleNamespace(
            qm=quat_math_torch,
            arr=lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32)),
            np=lambda t: t.numpy().astype(np.float64),
            atol=2e-5,
        )
    jnp = pytest.importorskip("jax.numpy")
    from myosuite.physics import quat_math_jax

    return SimpleNamespace(
        qm=quat_math_jax,
        arr=lambda a: jnp.asarray(a, dtype=jnp.float32),
        np=lambda a: np.asarray(a, dtype=np.float64),
        atol=2e-5,
    )


def _unit_quats(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _wxyz(rot: Any) -> np.ndarray:
    return np.roll(rot.as_quat(), 1, axis=-1)  # scipy is (x, y, z, w)


def _assert_same_rotation(q: np.ndarray, want: np.ndarray, atol: float) -> None:
    """Quaternions equal up to the sign (q and -q are the same rotation)."""
    err = np.minimum(np.abs(q - want).max(-1), np.abs(q + want).max(-1))
    assert err.max() <= atol, err.max()


def test_quat2vel_matches_mju_quat2vel(backend: SimpleNamespace) -> None:
    """speed * axis equals mju_quat2Vel, also for w < 0 (it went the long way round)."""
    rng = np.random.default_rng(0)
    quats = _unit_quats(rng, 64)
    quats[:32] *= -np.sign(quats[:32, :1])  # same rotations, w < 0
    quats = np.vstack([quats, [[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0]]])
    dt = 0.5
    want = np.zeros((len(quats), 3))
    for q, res in zip(quats, want):
        mujoco.mju_quat2Vel(res, q, dt)

    speed, axis = backend.qm.quat2Vel(backend.arr(quats), dt)
    got = backend.np(speed)[:, None] * backend.np(axis)
    np.testing.assert_allclose(got, want, atol=max(backend.atol, 1e-6))

    # q and -q are the same orientation: zero angular velocity (was 2 pi / dt).
    speed, _ = backend.qm.quat_diff_to_vel(backend.arr(quats), backend.arr(-quats), 1.0)
    np.testing.assert_allclose(backend.np(speed), 0.0, atol=10 * backend.atol)


@pytest.mark.parametrize("n", [1, 3, 4, 7])
def test_quat_ops_batched_match_per_sample(backend: SimpleNamespace, n: int) -> None:
    """(N, 4) inputs give the per-sample results (torch returned (4, 4) for N >= 4)."""
    rng = np.random.default_rng(n)
    qa, qb = _unit_quats(rng, n), _unit_quats(rng, n)
    axis = _unit_quats(rng, n)[:, 1:]
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    cases: dict[str, tuple[np.ndarray, ...]] = {
        "mul_quat": (qa, qb),
        "neg_quat": (qa,),
        "diff_quat": (qa, qb),
        "axis_angle2quat": (axis, rng.uniform(-3.0, 3.0, n)),
        "rot_vec_mat_t": (rng.normal(size=(n, 3)), quat_math.quat2mat(qa)),
        "rot_vec_mat": (rng.normal(size=(n, 3)), quat_math.quat2mat(qa)),
        "rot_vec_quat": (rng.normal(size=(n, 3)), qa),
        "quat2euler_intrinsic": (qa,),
        "intrinsic_euler2quat": (rng.uniform(-1.2, 1.2, (n, 3)),),
        "quat2Vel": (qa,),
    }
    for name, args in cases.items():
        fn = getattr(backend.qm, name)
        batched = fn(*map(backend.arr, args))
        single = [fn(*(backend.arr(a[i]) for a in args)) for i in range(n)]
        if name == "quat2Vel":  # (speed, axis)
            pairs = [
                (batched[0], [s[0] for s in single]),
                (batched[1], [s[1] for s in single]),
            ]
        else:
            pairs = [(batched, single)]
        for got, per_sample in pairs:
            want = np.stack([backend.np(x) for x in per_sample])
            assert backend.np(got).shape == want.shape, name
            np.testing.assert_allclose(
                backend.np(got), want, atol=backend.atol, err_msg=name
            )


def test_euler_conventions_match_scipy(backend: SimpleNamespace) -> None:
    """Pin the documented conventions: euler2* / *2euler are intrinsic "XYZ",
    intrinsic_euler2quat / quat2euler_intrinsic are extrinsic "xyz" (scipy names)."""
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    qm, arr, to_np, atol = backend.qm, backend.arr, backend.np, backend.atol
    euler = np.random.default_rng(3).uniform([-3, -1.4, -3], [3, 1.4, 3], (32, 3))
    intrinsic, extrinsic = (
        rotation.from_euler("XYZ", euler),
        rotation.from_euler("xyz", euler),
    )

    np.testing.assert_allclose(
        to_np(qm.euler2mat(arr(euler))), intrinsic.as_matrix(), atol=atol
    )
    _assert_same_rotation(to_np(qm.euler2quat(arr(euler))), _wxyz(intrinsic), atol)
    np.testing.assert_allclose(
        to_np(qm.mat2euler(arr(intrinsic.as_matrix()))), euler, atol=10 * atol
    )
    np.testing.assert_allclose(
        to_np(qm.quat2euler(arr(_wxyz(intrinsic)))), euler, atol=10 * atol
    )
    _assert_same_rotation(
        to_np(qm.intrinsic_euler2quat(arr(euler))), _wxyz(extrinsic), atol
    )
    np.testing.assert_allclose(
        to_np(qm.quat2euler_intrinsic(arr(_wxyz(extrinsic)))), euler, atol=10 * atol
    )
    # At the pole the rounded sin(pitch) is clamped to pitch = pi / 2.
    pole = qm.intrinsic_euler2quat(arr([0.2, np.pi / 2, -0.4]))
    assert to_np(qm.quat2euler_intrinsic(pole))[1] == pytest.approx(np.pi / 2, abs=1e-3)
    if qm is quat_math:
        deg = [30.0, -40.0, 50.0]
        _assert_same_rotation(
            np.asarray(quat_math.quat_from_euler_xyz_deg(deg)),
            _wxyz(rotation.from_euler("xyz", deg, degrees=True)),
            1e-12,
        )
