# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Shadow model: live muscle estimates beside an experiment.

Chains :class:`~myosuite.physics.streaming_kinematics.StreamingKinematics`,
:class:`~myosuite.physics.static_optimization.StaticOptimizer` and the 3CC-r
fatigue model (:class:`~myosuite.core.muscle_conditions.CumulativeFatigue`), and
logs every frame to CSV next to a JSON record of the model, versions and settings::

    with ShadowModel(model, log_dir="logs/P07", metadata={"participant": "P07"}) as shadow:
        for t, qpos in tracker:  # IK of each tracked pose onto the subject model
            frame = shadow.push(t, qpos)
            if frame is not None:
                show(frame.fatigue)

Fatigue takes the optimised activations as target loads. With
``fatigue_limits_activation`` a fatigued muscle delivers at most ``1 - MF`` of
its force, so later frames shift load to fresher muscles and, once those
saturate, to the reserves. Estimates are delayed by the kinematics lag.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.muscle_conditions import CumulativeFatigue
from myosuite.physics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizationResult,
    StaticOptimizer,
)
from myosuite.physics.streaming_kinematics import KinematicState, StreamingKinematics


@dataclass(frozen=True)
class ShadowModelConfig:
    """Settings of the shadow model.

    Attributes:
        window: Poses per kinematic fit.
        lag: Frames of delay; ``None`` centres the fit (``window // 2``).
        polyorder: Degree of the kinematic fit.
        max_gap: A pause between poses longer than this [s] restarts the
            kinematic fit; fatigue is held across it.
        optimization: Static-optimization settings (``contact_forces`` for
            standing and walking).
        fatigue: Track 3CC-r fatigue of every muscle.
        fatigue_limits_activation: Cap each muscle's activation at ``1 - MF``.
        log_contact_bodies: Bodies whose contact force is logged.
    """

    window: int = 15
    lag: int | None = None
    polyorder: int = 3
    max_gap: float = 0.25
    optimization: StaticOptimizationConfig = field(
        default_factory=StaticOptimizationConfig
    )
    fatigue: bool = True
    fatigue_limits_activation: bool = True
    log_contact_bodies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShadowFrame:
    """Estimates for one past frame.

    Attributes:
        state: Kinematics of the frame (``state.time`` is its timestamp).
        latest_time: Timestamp of the newest pose when it was computed [s].
        optimization: Activations, forces, reserves and contact forces.
        fatigue: ``(nu,)`` fatigued fraction ``MF`` per actuator (zero for
            non-muscles), or ``None`` without fatigue.
        compute_ms: Wall-clock time of this update [ms].
    """

    state: KinematicState
    latest_time: float
    optimization: StaticOptimizationResult
    fatigue: np.ndarray | None
    compute_ms: float


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


class ShadowModel:
    """Pose stream -> kinematics -> static optimization -> fatigue, logged per frame."""

    def __init__(
        self,
        model: mujoco.MjModel,
        config: ShadowModelConfig | None = None,
        *,
        log_dir: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Build the pipeline and, with *log_dir*, start the log.

        Args:
            model: Compiled (subject) model the poses are expressed in.
            config: Settings (defaults if ``None``).
            log_dir: Directory for ``shadow_model.csv`` and ``shadow_model.json``.
            metadata: Extra JSON-serialisable fields for the record (e.g. a
                participant or condition id).
        """
        self.model, self.config = model, config or ShadowModelConfig()
        cfg = self.config
        self.kinematics = StreamingKinematics(
            model, window=cfg.window, lag=cfg.lag, polyorder=cfg.polyorder
        )
        self.optimizer = StaticOptimizer(model, cfg.optimization)
        muscles = model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.fatigue = (
            CumulativeFatigue(model) if cfg.fatigue and muscles.any() else None
        )
        self._external: deque[tuple[float, np.ndarray | None]] = deque(
            maxlen=cfg.window
        )
        self._last_pose: float | None = None
        self._last_estimate: float | None = None
        self._bodies = [model.body(b).id for b in cfg.log_contact_bodies]
        self._file, self._writer = None, None
        if log_dir is not None:
            self._open_log(Path(log_dir), metadata or {})

    def _open_log(self, log_dir: Path, metadata: dict[str, Any]) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        m = self.model
        names = [m.actuator(i).name for i in range(m.nu)]
        buffer = np.zeros(mujoco.mj_sizeModel(m), dtype=np.uint8)
        mujoco.mj_saveModel(m, None, buffer)
        record = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {
                p: _package_version(p) for p in ("myosuite", "mujoco", "osqp")
            },
            "model_sha256": hashlib.sha256(buffer.tobytes()).hexdigest(),
            "model": {"nq": m.nq, "nv": m.nv, "nu": m.nu, "actuators": names},
            "config": dataclasses.asdict(self.config),
            "estimates": "minimum-effort static optimization; model estimates, not measurements",
            **metadata,
        }
        (log_dir / "shadow_model.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        header = ["time", "latest_time", "compute_ms", "status", "max_abs_reserve"]
        header += [f"a_{n}" for n in names]
        if self.fatigue is not None:
            header += [f"mf_{n}" for n in names]
        header += [
            f"contact_{b}_{x}" for b in self.config.log_contact_bodies for x in "xyz"
        ]
        self._file = open(
            log_dir / "shadow_model.csv", "w", newline="", encoding="utf-8"
        )  # noqa: SIM115
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)

    def reset_tracking(self) -> None:
        """Restart the kinematic fit after a tracking loss; fatigue is held."""
        self.kinematics.reset()
        self._external.clear()
        self._last_estimate = None

    def push(
        self, time_s: float, qpos: np.ndarray, external_qfrc: np.ndarray | None = None
    ) -> ShadowFrame | None:
        """Add the newest pose and estimate the frame ``lag`` poses before it.

        Args:
            time_s: Timestamp of the pose [s], increasing.
            qpos: ``(nq,)`` positions.
            external_qfrc: Optional ``(nv,)`` known external load at this pose
                (e.g. a hand-held weight mapped with ``mj_applyFT``).

        Returns:
            The estimate, or ``None`` while the kinematic window fills.
        """
        start = time.perf_counter()
        if (
            self._last_pose is not None
            and time_s - self._last_pose > self.config.max_gap
        ):
            self.reset_tracking()
        self._last_pose = time_s
        self._external.append((time_s, external_qfrc))
        state = self.kinematics.push(time_s, qpos)
        if state is None:
            return None
        external = next(e for t, e in self._external if t == state.time)

        f = self.fatigue
        cap = None
        if f is not None and self.config.fatigue_limits_activation:
            cap = np.full(self.model.nu, np.inf)
            cap[f.muscle_ids] = 1.0 - f.MF
        res = self.optimizer.solve(state.qpos, state.qvel, state.qacc, external, cap)
        mf = None
        if f is not None:
            if self._last_estimate is not None:
                f.compute_act(
                    res.activation[f.muscle_ids], dt=state.time - self._last_estimate
                )
            mf = np.zeros(self.model.nu)
            mf[f.muscle_ids] = f.MF
        self._last_estimate = state.time

        frame = ShadowFrame(
            state=state,
            latest_time=float(time_s),
            optimization=res,
            fatigue=mf,
            compute_ms=1e3 * (time.perf_counter() - start),
        )
        if self._writer is not None:
            row = [state.time, time_s, frame.compute_ms, str(res.status)]
            row += [float(np.abs(res.reserve).max())] + res.activation.tolist()
            row += [] if mf is None else mf.tolist()
            row += res.contact_force[self._bodies].ravel().tolist()
            self._writer.writerow(row)
            self._file.flush()
        return frame

    def close(self) -> None:
        """Close the log."""
        if self._file is not None:
            self._file.close()
            self._file, self._writer = None, None

    def __enter__(self) -> ShadowModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
