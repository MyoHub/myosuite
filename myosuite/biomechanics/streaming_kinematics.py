# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Live joint kinematics: positions, velocities and accelerations of a pose stream.

:func:`~myosuite.biomechanics.inverse_dynamics.clip_kinematics` filters a whole clip
forward and backward, so it needs frames after the one it estimates. A live
shadow model (e.g. muscle estimates beside a user study) gets one pose at a time;
:class:`StreamingKinematics` fits a local polynomial to the last ``window`` poses
on their own timestamps (a Savitzky-Golay fit that tolerates jitter and dropped
frames) and evaluates it ``lag`` frames before the newest. A centred lag gives
the most accurate accelerations; ``lag=0`` gives no delay but noisier, biased
derivatives. Acceleration noise dominates the error of live muscle estimates, so
choose the longest delay the application tolerates::

    stream = StreamingKinematics(model, window=11)  # 90 Hz -> 56 ms delay
    for t, qpos in tracker:  # IK of each tracked pose onto the subject model
        state = stream.push(t, qpos)
        if state is not None:
            frame = optimizer.solve(state.qpos, state.qvel, state.qacc)

Free and ball joints are fitted in the tangent space of the evaluated frame
(``mj_differentiatePos``), where the derivatives equal MuJoCo's ``qvel`` and
``qacc``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings


@dataclass(frozen=True)
class KinematicState:
    """Estimated state of one past frame.

    Attributes:
        time: Timestamp of the estimated frame [s].
        qpos: ``(nq,)`` smoothed positions, couplings projected.
        qvel: ``(nv,)`` velocities (MuJoCo convention).
        qacc: ``(nv,)`` accelerations.
    """

    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray


class StreamingKinematics:
    """Fixed-lag local polynomial fit of a live ``qpos`` stream."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        window: int = 11,
        lag: int | None = None,
        polyorder: int = 3,
        project_couplings: bool = True,
    ) -> None:
        """Configure the fit.

        Args:
            model: Compiled model the poses are expressed in.
            window: Number of most recent poses in each fit.
            lag: Frames between the newest pose and the estimated one; the
                default ``window // 2`` centres the fit. The delay is ``lag``
                frame periods.
            polyorder: Degree of the local polynomial (at least 2).
            project_couplings: Put dependent joints of joint equalities on their
                polynomials, in each incoming pose and in the estimate.

        Raises:
            ValueError: If ``polyorder``, ``window`` and ``lag`` are inconsistent.
        """
        lag = window // 2 if lag is None else lag
        if not 2 <= polyorder < window or not 0 <= lag < window:
            raise ValueError(
                f"Need 2 <= polyorder < window and 0 <= lag < window, got "
                f"polyorder={polyorder}, window={window}, lag={lag}"
            )
        self.model, self.window, self.lag, self.polyorder = (
            model,
            window,
            lag,
            polyorder,
        )
        self._couplings = (
            JointCouplings.from_model(model) if project_couplings else None
        )
        self._times: deque[float] = deque(maxlen=window)
        self._poses: deque[np.ndarray] = deque(maxlen=window)
        self._disp = np.empty((window, model.nv))

    def reset(self) -> None:
        """Forget all poses (e.g. after a tracking loss)."""
        self._times.clear()
        self._poses.clear()

    def push(self, time: float, qpos: np.ndarray) -> KinematicState | None:
        """Add the newest pose and estimate the frame ``lag`` poses before it.

        Args:
            time: Timestamp of the pose [s], strictly increasing.
            qpos: ``(nq,)`` positions.

        Returns:
            The estimate, or ``None`` until ``window`` poses have arrived.

        Raises:
            ValueError: If the pose or timestamp is invalid.
        """
        qpos = np.array(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError(f"qpos must be finite with shape ({self.model.nq},)")
        if self._times and not time > self._times[-1]:
            raise ValueError(f"time must increase, got {time} after {self._times[-1]}")
        if self._couplings is not None:
            project_joint_couplings(self._couplings, qpos)
        self._times.append(float(time))
        self._poses.append(qpos)
        if len(self._poses) < self.window:
            return None

        e = self.window - 1 - self.lag
        ref = self._poses[e]
        for k, pose in enumerate(self._poses):
            mujoco.mj_differentiatePos(self.model, self._disp[k], 1.0, ref, pose)
        # Least-squares polynomial in time scaled by the window span.
        span = self._times[-1] - self._times[0]
        s = (np.asarray(self._times) - self._times[e]) / span
        coef = np.linalg.lstsq(
            np.vander(s, self.polyorder + 1, increasing=True), self._disp, rcond=None
        )[0]
        out = ref.copy()
        mujoco.mj_integratePos(self.model, out, coef[0], 1.0)
        qvel = coef[1] / span
        qacc = 2.0 * coef[2] / span**2
        if self._couplings is not None:
            project_joint_couplings(self._couplings, out, qvel)
            _project_coupled_acceleration(self._couplings, out, qvel, qacc)
        return KinematicState(time=self._times[e], qpos=out, qvel=qvel, qacc=qacc)


def _project_coupled_acceleration(
    c: JointCouplings, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray
) -> None:
    """Write coupled joints' accelerations from their masters, in place (chain rule).

    A dependent ``y = y0 + p(x - x0)`` has ``y'' = p'(x - x0) x'' + p''(x - x0) x'^2``;
    locked joints get zero. Passes follow the couplings' topological order, so a master
    that is itself coupled is written first. *qpos* and *qvel* must already be projected.
    """
    qacc[c.locked_dof] = 0.0
    start = 0
    for end in c.level_ends:
        rows = slice(start, end)
        x = qpos[c.master_qpos[rows]] - c.master_ref[rows]
        k = c.coef[rows]
        slope = k[:, 1] + 2 * k[:, 2] * x + 3 * k[:, 3] * x**2 + 4 * k[:, 4] * x**3
        curvature = 2 * k[:, 2] + 6 * k[:, 3] * x + 12 * k[:, 4] * x**2
        v, a = qvel[c.master_dof[rows]], qacc[c.master_dof[rows]]
        qacc[c.dep_dof[rows]] = slope * a + curvature * v**2
        start = end
