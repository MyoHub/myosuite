# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Target-model marker fitting, motion export and geometric quality diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.subject_model import SubjectModel, _geom_bottom, model_digest
from myosuite.core.trajectory_io import (
    MotionClip,
    expand_motion_clip_to_model,
    load_motion_clip,
)
from myosuite.integrations.capture.markers import MarkerTrial
from myosuite.physics.marker_ik import qpos_from_marker_targets


@dataclass(frozen=True)
class MarkerRetargetResult:
    """Named motion plus target identity, quality measures and geometric labels."""

    clip: MotionClip
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        """Write pickle-free NPZ motion and a readable JSON provenance sidecar."""
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Retargeted motion path must end in .npz.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            qpos=self.clip.qpos,
            qvel=self.clip.qvel,
            qpos_names=np.array(self.clip.qpos_joint_names),
            qvel_names=np.array(self.clip.qvel_joint_names),
            site_xpos=self.clip.site_xpos,
            site_names=np.array(self.clip.site_names),
            frequency=np.array(self.clip.frequency_hz),
            provenance_json=np.array(json.dumps(self.provenance, allow_nan=False)),
            contact_labels=self.diagnostics["contact_labels"],
            sole_bottom_m=self.diagnostics["sole_bottom_m"],
            root_trajectory_m=self.diagnostics["root_trajectory_m"],
            marker_errors_m=self.diagnostics["marker_errors_m"],
        )
        summary = {
            name: value
            for name, value in self.diagnostics.items()
            if np.isscalar(value)
        }
        path.with_suffix(".json").write_text(
            json.dumps(
                {"provenance": self.provenance, "quality": summary},
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return path


def motion_diagnostics(
    model: mujoco.MjModel,
    clip: MotionClip,
    trial: MarkerTrial,
    marker_sites: Mapping[str, str],
    *,
    free_joint: str,
    sole_geoms: Sequence[str] = (),
    root_marker: str | None = None,
    ground_height_m: float = 0.0,
    contact_height_m: float = 0.01,
    contact_speed_m_s: float = 0.2,
) -> dict[str, Any]:
    """Measure replay against markers; label near-ground, slow sole geometry.

    Contact labels are geometric heuristics, not measured forces. ``root_marker``
    must represent the free body's origin to make root tracking error meaningful.
    Missing marker observations are excluded from residual summaries.
    """
    if not marker_sites or not np.isfinite(ground_height_m):
        raise ValueError("Marker mapping and finite ground height are required.")
    for value in (contact_height_m, contact_speed_m_s):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Contact thresholds must be finite and positive.")
    clip = expand_motion_clip_to_model(clip, model)
    points = trial.in_model_frame()
    if len(clip.qpos) != len(points) or not clip.frequency_hz or clip.frequency_hz <= 0:
        raise ValueError(
            "Clip/trial frame counts and positive clip frequency must agree."
        )
    joint = model.joint(free_joint)
    if int(joint.type[0]) != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("free_joint must identify a free root joint.")
    root_body = int(joint.bodyid[0])
    labels = [trial.labels.index(label) for label in marker_sites]
    sites = [model.site(name).id for name in marker_sites.values()]
    geoms = [model.geom(name).id for name in sole_geoms]
    for geom in geoms:
        body = int(model.geom_bodyid[geom])
        while body and body != root_body:
            body = int(model.body_parentid[body])
        if body != root_body:
            raise ValueError(
                "Sole geometry must belong to the selected free-root tree."
            )
    data = mujoco.MjData(model)
    errors, bottoms, roots, centres = [], [], [], []
    for frame in clip.qpos:
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        errors.append(data.site_xpos[sites].copy())
        bottoms.append([_geom_bottom(model, data, geom) for geom in geoms])
        roots.append(data.xpos[root_body].copy())
        centres.append(data.geom_xpos[geoms].copy())
    marker_errors = np.linalg.norm(np.array(errors) - points[:, labels], axis=2)
    finite_errors = marker_errors[np.isfinite(marker_errors)]
    if not len(finite_errors):
        raise ValueError("No valid marker observations remain for diagnostics.")
    bottom = np.array(bottoms)
    root = np.array(roots)
    speed = np.zeros_like(bottom)
    if len(clip.qpos) > 1 and len(geoms):
        speed = np.linalg.norm(
            np.gradient(np.array(centres), 1 / clip.frequency_hz, axis=0), axis=2
        )
    contacts = (np.abs(bottom - ground_height_m) <= contact_height_m) & (
        speed <= contact_speed_m_s
    )
    diagnostics = {
        "marker_errors_m": marker_errors,
        "marker_rms_m": float(np.sqrt(np.mean(finite_errors**2))),
        "marker_max_m": float(finite_errors.max()),
        "sole_bottom_m": bottom,
        "sole_penetration_max_m": float(np.max(ground_height_m - bottom, initial=0.0)),
        "contact_labels": contacts,
        "root_trajectory_m": root,
    }
    if root_marker is not None:
        expected = points[:, trial.labels.index(root_marker)]
        valid = np.all(np.isfinite(expected), axis=1)
        if not np.any(valid):
            raise ValueError("root_marker has no valid observations.")
        diagnostics["root_tracking_rms_m"] = float(
            np.sqrt(np.mean(np.sum((root[valid] - expected[valid]) ** 2, axis=1)))
        )
    return diagnostics


def retarget_markers(
    subject: SubjectModel,
    trial: MarkerTrial,
    marker_sites: Mapping[str, str],
    *,
    frequency_hz: float | None = None,
    free_joint: str,
    sole_geoms: Sequence[str] = (),
    root_marker: str | None = None,
    joint_names: Sequence[str] | None = None,
    tolerance_m: float = 1e-4,
    max_evaluations: int = 200,
    strict: bool = True,
    scale_evidence: Mapping[str, Any] | None = None,
    ground_height_m: float = 0.0,
) -> MarkerRetargetResult:
    """Solve every frame against this subject; never substitute nominal replay.

    Frames warm-start from the previous fitted pose. NaN observations are omitted
    per frame; a frame without usable mapped markers fails. ``strict=False``
    retains failed fits for explicit negative controls and inspection only.
    """
    model = subject.model
    if model_digest(model) != subject.metadata["model_digest"]:
        raise ValueError(
            "Subject model changed after construction; rebuild its provenance."
        )
    frequency = trial.frequency_hz if frequency_hz is None else frequency_hz
    if frequency is None or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("A finite positive motion frequency is required.")
    if not marker_sites or len(set(marker_sites.values())) != len(marker_sites):
        raise ValueError("A nonempty one-to-one marker-to-site mapping is required.")
    names = [model.joint(index).name for index in range(model.njnt)]
    if any(not name for name in names):
        raise ValueError("Named motion export requires every model joint to be named.")
    indices = {label: trial.labels.index(label) for label in marker_sites}
    positions = trial.in_model_frame()
    state, poses, site_positions, success, equality_errors = (
        model.qpos0.copy(),
        [],
        [],
        [],
        [],
    )
    data = mujoco.MjData(model)
    site_ids = [model.site(name).id for name in marker_sites.values()]
    for frame_index, frame in enumerate(positions):
        targets = {
            site: frame[indices[label]]
            for label, site in marker_sites.items()
            if np.all(np.isfinite(frame[indices[label]]))
        }
        result = qpos_from_marker_targets(
            model,
            targets,
            qpos=state,
            joint_names=joint_names,
            tolerance_m=tolerance_m,
            max_evaluations=max_evaluations,
        )
        if strict and not result.success:
            raise ValueError(
                f"Frame {frame_index}: target IK did not converge; marker error {result.max_marker_error_m:g} m, equality error {result.max_equality_error:g}."
            )
        state = result.qpos
        poses.append(state)
        success.append(result.success)
        equality_errors.append(result.max_equality_error)
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        site_positions.append(data.site_xpos[site_ids].copy())
    qpos = np.array(poses)
    qvel = np.zeros((len(poses), model.nv))
    for index in range(len(poses) - 1):
        mujoco.mj_differentiatePos(
            model, qvel[index], 1 / frequency, qpos[index], qpos[index + 1]
        )
    if len(poses) > 1:
        qvel[-1] = qvel[-2]
    clip = MotionClip(
        qpos,
        qvel,
        np.array(site_positions),
        list(marker_sites.values()),
        qpos_joint_names=names,
        qvel_joint_names=names,
        frequency_hz=float(frequency),
    )
    diagnostics = motion_diagnostics(
        model,
        clip,
        trial,
        marker_sites,
        free_joint=free_joint,
        sole_geoms=sole_geoms,
        root_marker=root_marker,
        ground_height_m=ground_height_m,
    )
    diagnostics["converged_frames"] = int(sum(success))
    diagnostics["frames"] = len(poses)
    diagnostics["equality_error_max"] = max(equality_errors)
    provenance = {
        "schema": 1,
        "method": "simultaneous marker IK on scaled target",
        "subject_model": subject.metadata,
        "trial": trial.metadata(),
        "marker_sha256": hashlib.sha256(
            np.asarray(trial.positions).tobytes()
        ).hexdigest(),
        "marker_sites": dict(marker_sites),
        "joint_names": joint_names,
        "tolerance_m": tolerance_m,
        "free_joint": free_joint,
        "sole_geoms": list(sole_geoms),
        "ground_height_m": ground_height_m,
        "contact_definition": {
            "height_m": 0.01,
            "speed_m_s": 0.2,
            "kind": "geometric heuristic",
        },
        "scale_evidence": None if scale_evidence is None else dict(scale_evidence),
    }
    return MarkerRetargetResult(clip, diagnostics, provenance)


def load_retargeted_clip(path: str | Path, subject: SubjectModel) -> MotionClip:
    """Load named motion only when its recorded target matches the current model."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        provenance = json.loads(str(archive["provenance_json"]))
    if provenance["subject_model"]["model_digest"] != model_digest(subject.model):
        raise ValueError("Retargeted clip was fitted to a different subject model.")
    return expand_motion_clip_to_model(
        load_motion_clip(path, subject.model.nq, subject.model.nv), subject.model
    )
