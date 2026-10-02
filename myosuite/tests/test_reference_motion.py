# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""ReferenceMotion (NumPy and JAX twin) checked against an independent reference.

The NumPy and JAX implementations used to share the same interpolation bug, so a
NumPy-vs-JAX parity test could not catch it; these tests compare both against
``np.interp`` instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.logger.reference_motion import ReferenceMotion as NumpyReferenceMotion

pytestmark = pytest.mark.tier1

_BACKENDS = ("numpy", "jax")


def _reference_cls(backend: str) -> type:
    """Return the ReferenceMotion class for *backend* (skips when JAX is missing)."""
    if backend == "numpy":
        return NumpyReferenceMotion
    pytest.importorskip("jax")
    from myosuite.logger.reference_motion_jax import ReferenceMotion

    return ReferenceMotion


def _atol(backend: str) -> float:
    # JAX runs in float32 unless x64 is enabled.
    return 1e-9 if backend == "numpy" else 1e-5


def _track_data(rng: np.random.Generator, n: int = 7) -> dict[str, np.ndarray]:
    """Random TRACK clip with non-uniform frame times (4-decimal, like the class)."""
    times = np.round(
        np.concatenate([[0.0], np.cumsum(rng.uniform(0.05, 0.4, n - 1))]), 4
    )
    return {
        "time": times,
        "robot": rng.normal(size=(n, 3)),
        "robot_vel": rng.normal(size=(n, 3)),
        "object": rng.normal(size=(n, 2)),
    }


def _interp(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    return np.array([np.interp(t, times, values[:, j]) for j in range(values.shape[1])])


@pytest.mark.parametrize("backend", _BACKENDS)
def test_track_interpolation_matches_np_interp(backend: str) -> None:
    """Between frames, robot / robot_vel / object are linear in time (np.interp)."""
    cls = _reference_cls(backend)
    rng = np.random.default_rng(0)
    data = _track_data(rng)
    ref = cls({k: v.copy() for k, v in data.items()})
    times = data["time"]
    queries = np.round(rng.uniform(times[0], times[-1], 25), 4)
    # sorted queries walk the index cache; shuffled ones force the search path
    for sweep in (np.sort(queries), queries, times):
        ref.reset()
        for t in sweep:
            out = ref.get_reference(float(t))
            for key, field in (
                ("robot", out.robot),
                ("robot_vel", out.robot_vel),
                ("object", out.object),
            ):
                np.testing.assert_allclose(
                    np.asarray(field),
                    _interp(times, data[key], float(t)),
                    atol=_atol(backend),
                    err_msg=f"{backend} {key} at t={t}",
                )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_track_midpoint_is_mean_of_frames(backend: str) -> None:
    """Reviewer example: frames (0, 0.5, 1) -> (0, 2, 4) gives 3.0 at t=0.75 (was 0.5625)."""
    cls = _reference_cls(backend)
    values = np.array([[0.0], [2.0], [4.0]])
    ref = cls(
        {
            "time": np.array([0.0, 0.5, 1.0]),
            "robot": values,
            "robot_vel": values,
            "object": values,
        }
    )
    out = ref.get_reference(0.75)
    for field in (out.robot, out.robot_vel, out.object):
        np.testing.assert_allclose(np.asarray(field), [3.0], atol=_atol(backend))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_track_single_robot_frame_is_held(backend: str) -> None:
    """An object-only track holds its single robot frame at every exact object frame."""
    cls = _reference_cls(backend)
    data = _track_data(np.random.default_rng(2))
    data["robot"], data["robot_vel"] = data["robot"][:1], data["robot_vel"][:1]
    ref = cls({k: v.copy() for k, v in data.items()})
    for t in data["time"]:
        out = ref.get_reference(float(t))
        np.testing.assert_allclose(np.asarray(out.robot), data["robot"][0], atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(out.robot_vel), data["robot_vel"][0], atol=1e-6
        )


@pytest.mark.parametrize("as_jax", [False, True], ids=["numpy-arrays", "jax-arrays"])
def test_jax_track_holds_end_frames_outside_clip(as_jax: bool) -> None:
    """The JAX twin returns the end frames outside the clip (was NaN / IndexError)."""
    cls = _reference_cls("jax")
    import jax.numpy as jnp

    data = _track_data(np.random.default_rng(1))
    conv = jnp.asarray if as_jax else np.array
    ref = cls({k: conv(v) for k, v in data.items()})
    for t, frame in ((float(data["time"][-1]) + 0.3, -1), (-0.2, 0)):
        out = ref.get_reference(t)
        for key, field in (
            ("robot", out.robot),
            ("robot_vel", out.robot_vel),
            ("object", out.object),
        ):
            np.testing.assert_allclose(
                np.asarray(field),
                data[key][frame],
                atol=1e-5,
                err_msg=f"{key} at t={t}",
            )


def test_jax_random_draws_new_samples_each_call() -> None:
    """RANDOM references advance their PRNG key (was a fixed PRNGKey(0) per call)."""
    cls = _reference_cls("jax")
    import jax

    low, high = np.zeros(4), np.ones(4)
    data = {
        "time": np.array([0.0, 1.0]),
        "robot": np.stack([low, high]),
        "robot_vel": np.stack([low, high]),
        "object": np.stack([low[:2], high[:2]]),
    }
    ref = cls(data)
    draws = np.array([np.asarray(ref.get_reference(0.0).robot) for _ in range(5)])
    assert len({tuple(d) for d in draws}) == 5, draws
    assert np.all((draws >= low) & (draws <= high))

    # an explicit key is pure: same key, same sample
    key = jax.random.PRNGKey(7)
    a = np.asarray(ref.get_reference(0.0, key=key).object)
    b = np.asarray(ref.get_reference(0.0, key=key).object)
    np.testing.assert_array_equal(a, b)
    # the constructor key seeds the stream
    first = np.asarray(
        cls(data, random_key=jax.random.PRNGKey(3)).get_reference(0.0).robot
    )
    again = np.asarray(
        cls(data, random_key=jax.random.PRNGKey(3)).get_reference(0.0).robot
    )
    np.testing.assert_array_equal(first, again)
