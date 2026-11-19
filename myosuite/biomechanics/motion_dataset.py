# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Dataset layer for kinematic motion data (:class:`MotionDataset` and :class:`KinematicTrial`).

Composes dataset-level split/subject/licence metadata with format-level
:class:`~.retarget.ClipSource` adapters:

* :class:`MotionDataset` enumerates trials by split, exposes subject and licence
  metadata (:class:`SubjectMetadata`, :class:`DatasetLicense`), and resolves the
  :class:`~.retarget.ClipSource` for each trial's format.
* :class:`KinematicTrial` represents one path-backed, multi-file, or in-memory
  motion trial (including optional object poses/extras for hand-object interaction).
* :class:`JointSpaceClipSource` maps joint-space motions (OpenSim ``.mot``/``.sto``,
  retargeted ``.npz`` clips, or in-memory ``qpos`` arrays) onto a model's couplings
  through ``ClipTargets.qpos``, bypassing landmark IK while keeping projection,
  validation, and ``REFIT_META_KEY`` provenance identical.
* :class:`ManifestMotionDataset` loads declarative JSON manifests for lab datasets
  without custom Python adapters.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np

from myosuite.biomechanics.opensim_motion import (
    GAIT2392_SIGNS,
    opensim_motion_to_qpos,
    read_opensim_motion,
)
from myosuite.biomechanics.retarget import (
    ClipSource,
    ClipTargets,
    LandmarkIK,
    LandmarkMap,
    RefitConfig,
    RefitThresholds,
    file_sha256,
    refit_clip,
)


@dataclass(frozen=True)
class SubjectMetadata:
    """Anthropometric and demographic metadata for a recorded subject.

    Attributes:
        subject_id: Dataset-local subject identifier.
        height_m: Stature in metres, if known.
        mass_kg: Body mass in kilograms, if known.
        sex: Biological sex or body-model gender tag (e.g. ``"female"``, ``"male"``, ``"neutral"``).
        age_years: Age in years, if known.
        extra: Additional subject attributes (e.g. limb lengths or SMPL ``betas``).
    """

    subject_id: str = ""
    height_m: float | None = None
    mass_kg: float | None = None
    sex: str | None = None
    age_years: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)


@dataclass(frozen=True)
class DatasetLicense:
    """Licence, registration, and derivative-redistribution rules of a motion dataset.

    Attributes:
        name: Short licence name (e.g. ``"CC-BY-4.0"``, ``"AMASS Research License"``).
        spdx_or_url: SPDX identifier or canonical licence URL.
        citation: Primary bibliographic citation or BibTeX string.
        redistributable_derivatives: Whether derived retargeted clips may be published
            or redistributed publicly.
        requires_registration: Whether accessing the source dataset requires user
            registration / credentialed download.
    """

    name: str
    spdx_or_url: str = ""
    citation: str = ""
    redistributable_derivatives: bool = False
    requires_registration: bool = False

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)


def ensure_redistributable(
    license_meta: DatasetLicense | Mapping[str, Any] | None,
) -> None:
    """Raise ``PermissionError`` when *license_meta* forbids redistributing derived clips.

    Args:
        license_meta: Dataset licence dataclass or serialised dictionary.

    Raises:
        PermissionError: If *license_meta* is missing or ``redistributable_derivatives`` is false.
    """
    if license_meta is None:
        raise PermissionError(
            "Clip carries no dataset licence metadata; refusing redistributable export."
        )
    allowed = (
        license_meta.redistributable_derivatives
        if isinstance(license_meta, DatasetLicense)
        else bool(license_meta.get("redistributable_derivatives", False))
    )
    if not allowed:
        label = (
            license_meta.name
            if isinstance(license_meta, DatasetLicense)
            else str(license_meta.get("name", "unknown licence"))
        )
        raise PermissionError(
            f"Dataset licence {label!r} sets redistributable_derivatives=False; "
            "derived clips may be cached locally but must not be redistributed."
        )


@dataclass(frozen=True)
class KinematicTrial:
    """One kinematic motion trial from a :class:`MotionDataset`.

    Attributes:
        trial_id: Unique identifier within the dataset.
        format: Source format key understood by ``dataset.clip_source(format)``
            (e.g. ``"amass-smplh"``, ``"opensim-mot"``, ``"retargeted-npz"``, ``"markers"``).
        fps: Nominal frame rate in Hz (``None`` when read from the file header).
        path: Primary file path for single-file trials.
        paths: Named file map for multi-file trials (e.g. ``{"dynamic": ..., "static": ...}``).
        arrays: In-memory NumPy arrays when the trial is constructed without temp files.
        subject: Subject anthropometrics and metadata.
        axes: World axis convention of the raw source (e.g. ``"Z-up"``, ``"Y-up"``).
        split: Dataset split tag (e.g. ``"train"``, ``"val"``, ``"test"``).
        objects: Optional object pose trajectories or mesh descriptors for HOI datasets.
        extras: Additional per-trial metadata (curation flags, trim bounds, camera IDs).
        license: Optional trial-level licence override (defaults to the dataset's licence).
    """

    trial_id: str
    format: str
    fps: float | None = None
    path: Path | None = None
    paths: dict[str, Path] = field(default_factory=dict)
    arrays: dict[str, np.ndarray | Any] = field(default_factory=dict)
    subject: SubjectMetadata = field(default_factory=SubjectMetadata)
    axes: str = "Z-up"
    split: str = "train"
    objects: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    license: DatasetLicense | None = None

    def sha256(self) -> str:
        """Deterministic SHA-256 over the trial's backing file(s) or in-memory arrays."""
        if self.path is not None and not self.paths and not self.arrays:
            return file_sha256(self.path)
        h = hashlib.sha256()
        h.update(self.trial_id.encode("utf-8"))
        h.update(self.format.encode("utf-8"))
        if self.path is not None:
            h.update(file_sha256(self.path).encode("ascii"))
        for key in sorted(self.paths):
            h.update(key.encode("utf-8"))
            h.update(file_sha256(self.paths[key]).encode("ascii"))
        for key in sorted(self.arrays):
            h.update(key.encode("utf-8"))
            arr = np.asarray(self.arrays[key])
            h.update(str(arr.dtype).encode("ascii"))
            h.update(str(arr.shape).encode("ascii"))
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def source_label(self) -> str:
        """Human-readable provenance string recorded under ``meta['source']``."""
        if self.path is not None:
            return str(self.path)
        if "dynamic" in self.paths:
            return str(self.paths["dynamic"])
        return f"memory://{self.trial_id}"


@runtime_checkable
class MotionDataset(Protocol):
    """Protocol for kinematic motion datasets."""

    name: str
    license: DatasetLicense

    def trial_ids(self, split: str | None = None) -> tuple[str, ...]:
        """Ordered trial identifiers, optionally filtered by *split*."""
        ...

    def get_trial(self, trial_id: str) -> KinematicTrial:
        """Return the :class:`KinematicTrial` for *trial_id*."""
        ...

    def clip_source(self, format_name: str) -> ClipSource:
        """Return a :class:`~.retarget.ClipSource` capable of reading *format_name*."""
        ...


def _map_named_qpos(
    model: mujoco.MjModel,
    qpos_src: np.ndarray,
    joint_names: Sequence[str] | None,
) -> np.ndarray:
    """Map ``qpos_src`` onto *model* by joint name when layouts differ, else copy."""
    qpos_src = np.asarray(qpos_src, dtype=np.float64)
    if qpos_src.ndim != 2:
        raise ValueError(f"Expected 2-D qpos array (T, nq), got shape {qpos_src.shape}")
    model_jnames = [model.joint(j).name for j in range(model.njnt)]
    if joint_names is None or list(joint_names) == model_jnames:
        if qpos_src.shape[1] != model.nq:
            raise ValueError(
                f"qpos width {qpos_src.shape[1]} != model.nq {model.nq} without joint_names mapping"
            )
        return qpos_src.copy()

    # Reconstruct per-joint slices from a source joint_names list (hinge/slide/ball/free).
    out = np.tile(model.qpos0, (len(qpos_src), 1))
    src_names = [str(x) for x in joint_names]
    model_map = {model.joint(j).name: j for j in range(model.njnt)}
    # Infer source qpos addresses from model joint widths where names match, assuming standard widths.
    offset = 0
    for sname in src_names:
        if sname in model_map:
            mj = model_map[sname]
            jtype = int(model.jnt_type[mj])
            width = {
                int(mujoco.mjtJoint.mjJNT_FREE): 7,
                int(mujoco.mjtJoint.mjJNT_BALL): 4,
            }.get(jtype, 1)
            dst_adr = int(model.jnt_qposadr[mj])
            if offset + width <= qpos_src.shape[1]:
                out[:, dst_adr : dst_adr + width] = qpos_src[:, offset : offset + width]
            offset += width
        else:
            offset += 1
    return out


def _fk_targets_from_qpos(
    model: mujoco.MjModel,
    lmap: LandmarkMap,
    qpos: np.ndarray,
    fps: float,
    contact: np.ndarray | None = None,
) -> ClipTargets:
    """Evaluate *lmap* landmark positions and body orientations directly from ``qpos``."""
    data = mujoco.MjData(model)
    site_names = {model.site(i).name for i in range(model.nsite)}
    lm_body = np.empty(len(lmap.landmarks), np.int64)
    lm_off = np.empty((len(lmap.landmarks), 3))
    for i, lm in enumerate(lmap.landmarks):
        if lm.site is not None and lm.site in site_names:
            s = model.site(lm.site)
            lm_body[i], lm_off[i] = s.bodyid[0], s.pos
        else:
            lm_body[i], lm_off[i] = model.body(lm.body).id, lm.pos
    or_body = np.array([model.body(o.body).id for o in lmap.orientations], np.int64)

    n_frames = len(qpos)
    pos = np.empty((n_frames, len(lm_body), 3), dtype=np.float64)
    quat = np.empty((n_frames, len(or_body), 4), dtype=np.float64)
    for t in range(n_frames):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        for i in range(len(lm_body)):
            b = lm_body[i]
            pos[t, i] = data.xpos[b] + data.xmat[b].reshape(3, 3) @ lm_off[i]
        if len(or_body):
            quat[t] = data.xquat[or_body]
    if contact is None:
        contact_arr = np.zeros((n_frames, 2), dtype=bool)
    else:
        contact_arr = np.asarray(contact, dtype=bool)
    return ClipTargets(
        frequency=float(fps),
        pos=pos,
        quat=quat,
        seed=qpos[0].copy(),
        contact=contact_arr,
        qpos=qpos,
    )


@dataclass
class JointSpaceClipSource:
    """A :class:`~.retarget.ClipSource` for joint-space motions (``.mot``, ``.sto``, ``.npz``, or ``qpos`` arrays).

    Populates ``ClipTargets.qpos`` so :func:`~.retarget.refit_clip` projects couplings
    and clamps limits via :meth:`~.retarget.LandmarkIK.project` without running landmark
    Gauss-Newton iterations, while still computing :func:`~.retarget.validate_refit`
    metrics and ``REFIT_META_KEY`` provenance.

    Attributes:
        rate: Optional target frame rate in Hz for resampling OpenSim ``.mot`` files.
        root_body: Root body name placed by OpenSim pelvis coordinates.
        signs: Per-coordinate sign multipliers for OpenSim ``.mot`` files.
        name: Source format tag recorded in refit metadata.
    """

    rate: float | None = None
    root_body: str | None = "pelvis"
    signs: Mapping[str, float] = field(default_factory=lambda: dict(GAIT2392_SIGNS))
    name: str = "joint-space"
    _last_extra: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def targets(
        self,
        clip: str | Path | KinematicTrial,
        model: mujoco.MjModel,
        lmap: LandmarkMap,
    ) -> ClipTargets:
        """Build projected joint-space ``ClipTargets`` (with ``qpos`` populated) on *model*."""
        ik = LandmarkIK(model, lmap)
        contact: np.ndarray | None = None
        if isinstance(clip, KinematicTrial):
            if "qpos" in clip.arrays:
                jnames = clip.arrays.get("joint_names")
                raw_q = _map_named_qpos(model, np.asarray(clip.arrays["qpos"]), jnames)
                qpos = ik.project(raw_q)
                fps = float(clip.fps or clip.arrays.get("frequency", 100.0))
                if "contact" in clip.arrays:
                    contact = np.asarray(clip.arrays["contact"], dtype=bool)
                self._last_extra = {"joint_space_kind": "memory-qpos"}
                return _fk_targets_from_qpos(model, lmap, qpos, fps, contact)
            if clip.path is None:
                raise ValueError(
                    f"Trial {clip.trial_id!r} has neither arrays['qpos'] nor path"
                )
            path = clip.path
            fps_override = clip.fps
        else:
            path = Path(clip)
            fps_override = None

        suffix = path.suffix.lower()
        if suffix in (".mot", ".sto"):
            motion = read_opensim_motion(path)
            mapped = opensim_motion_to_qpos(
                model,
                motion,
                rate=self.rate or fps_override,
                root_body=self.root_body,
                signs=self.signs,
                project_couplings=True,
            )
            qpos = ik.project(mapped.qpos)
            fps = (
                float(round(1.0 / mapped.dt, 6))
                if mapped.dt > 0
                else float(fps_override or 100.0)
            )
            self._last_extra = {
                "joint_space_kind": "opensim",
                "mapped_coordinates": list(mapped.coordinates),
                "unmapped_coordinates": list(mapped.unmapped),
            }
            return _fk_targets_from_qpos(model, lmap, qpos, fps)

        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as d:
                jnames = (
                    [str(x) for x in d["joint_names"]]
                    if "joint_names" in d.files
                    else None
                )
                raw_q = _map_named_qpos(model, d["qpos"], jnames)
                fps = float(
                    fps_override
                    or (float(d["frequency"]) if "frequency" in d.files else 100.0)
                )
                if "contact" in d.files:
                    contact = np.asarray(d["contact"], dtype=bool)
            qpos = ik.project(raw_q)
            self._last_extra = {"joint_space_kind": "npz"}
            return _fk_targets_from_qpos(model, lmap, qpos, fps, contact)

        raise ValueError(f"Unsupported joint-space motion file: {path}")

    def finish(
        self, ik: LandmarkIK, targets: ClipTargets, qpos: np.ndarray
    ) -> tuple[np.ndarray, ClipTargets, dict[str, Any]]:
        """Return projected ``qpos`` and recorded joint-space mapping metadata."""
        return qpos, targets, dict(self._last_extra)


class ManifestMotionDataset:
    """Declarative JSON manifest-backed :class:`MotionDataset` for lab motion collections.

    Manifest schema::

        {
          "name": "lab_gait_pilot",
          "license": {
            "name": "CC-BY-4.0",
            "spdx_or_url": "CC-BY-4.0",
            "citation": "...",
            "redistributable_derivatives": true,
            "requires_registration": false
          },
          "axes": "Z-up",
          "subjects": {
            "S01": {"height_m": 1.75, "mass_kg": 72.0, "sex": "female"}
          },
          "trials": [
            {
              "trial_id": "S01_walk_01",
              "format": "opensim-mot",
              "path": "trials/S01_walk_01.mot",
              "fps": 100.0,
              "subject_id": "S01",
              "split": "train"
            }
          ]
        }
    """

    def __init__(
        self,
        manifest: str | Path | Mapping[str, Any],
        *,
        root_dir: str | Path | None = None,
        sources: Mapping[str, ClipSource] | None = None,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
            base = Path(root_dir) if root_dir is not None else manifest_path.parent
        else:
            doc = dict(manifest)
            base = Path(root_dir) if root_dir is not None else Path.cwd()

        self.name: str = str(doc["name"])
        lic = doc["license"]
        self.license = (
            lic
            if isinstance(lic, DatasetLicense)
            else DatasetLicense(
                name=str(lic["name"]),
                spdx_or_url=str(lic.get("spdx_or_url", "")),
                citation=str(lic.get("citation", "")),
                redistributable_derivatives=bool(
                    lic.get("redistributable_derivatives", False)
                ),
                requires_registration=bool(lic.get("requires_registration", False)),
            )
        )
        default_axes = str(doc.get("axes", "Z-up"))
        subjects: dict[str, SubjectMetadata] = {}
        for sid, srow in doc.get("subjects", {}).items():
            subjects[str(sid)] = SubjectMetadata(
                subject_id=str(sid),
                height_m=srow.get("height_m"),
                mass_kg=srow.get("mass_kg"),
                sex=srow.get("sex"),
                age_years=srow.get("age_years"),
                extra=dict(srow.get("extra", {})),
            )

        self._trials: dict[str, KinematicTrial] = {}
        for row in doc.get("trials", ()):
            tid = str(row["trial_id"])
            if tid in self._trials:
                raise ValueError(
                    f"Duplicate trial_id {tid!r} in manifest {self.name!r}"
                )
            sid = str(row.get("subject_id", ""))
            subj = subjects.get(sid, SubjectMetadata(subject_id=sid))
            rel_path = row.get("path")
            resolved_path = (base / rel_path) if rel_path is not None else None
            resolved_paths = {
                str(k): (base / v) for k, v in row.get("paths", {}).items()
            }
            self._trials[tid] = KinematicTrial(
                trial_id=tid,
                format=str(row["format"]),
                fps=float(row["fps"]) if row.get("fps") is not None else None,
                path=resolved_path,
                paths=resolved_paths,
                subject=subj,
                axes=str(row.get("axes", default_axes)),
                split=str(row.get("split", "train")),
                objects=dict(row.get("objects", {})),
                extras=dict(row.get("extras", {})),
                license=self.license,
            )

        default_joint = JointSpaceClipSource()
        self._sources: dict[str, ClipSource] = {
            "joint-space": default_joint,
            "opensim-mot": JointSpaceClipSource(name="opensim-mot"),
            "retargeted-npz": JointSpaceClipSource(name="retargeted-npz"),
        }
        if sources:
            self._sources.update(sources)

    def trial_ids(self, split: str | None = None) -> tuple[str, ...]:
        """Ordered trial IDs, optionally filtered by *split*."""
        if split is None:
            return tuple(self._trials)
        return tuple(tid for tid, tr in self._trials.items() if tr.split == split)

    def get_trial(self, trial_id: str) -> KinematicTrial:
        """Return the :class:`KinematicTrial` for *trial_id*."""
        try:
            return self._trials[trial_id]
        except KeyError:
            raise KeyError(
                f"Unknown trial_id {trial_id!r} in dataset {self.name!r}"
            ) from None

    def clip_source(self, format_name: str) -> ClipSource:
        """Return the registered :class:`~.retarget.ClipSource` for *format_name*."""
        try:
            return self._sources[format_name]
        except KeyError:
            raise KeyError(
                f"Dataset {self.name!r} has no ClipSource registered for format {format_name!r}"
            ) from None


def retarget_trial(
    dataset: MotionDataset,
    trial_id: str,
    model: mujoco.MjModel,
    model_name: str,
    model_version: str,
    lmap: LandmarkMap,
    *,
    config: RefitConfig | None = None,
    thresholds: RefitThresholds | None = None,
    ik: LandmarkIK | None = None,
    model_fingerprint: str | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Retarget (or project) one trial from *dataset* onto *model* with full provenance.

    Embeds ``dataset``, ``trial_id``, ``split``, ``subject``, and ``license`` into the
    returned ``meta`` dictionary so :func:`~.retarget.save_refit` can enforce
    ``redistributable_derivatives`` on downstream export.
    """
    trial = dataset.get_trial(trial_id)
    source = dataset.clip_source(trial.format)
    qpos, freq, meta = refit_clip(
        trial,
        model=model,
        model_name=model_name,
        model_version=model_version,
        lmap=lmap,
        source=source,
        config=config,
        thresholds=thresholds,
        ik=ik,
        model_fingerprint=model_fingerprint,
    )
    lic = trial.license or dataset.license
    meta["dataset"] = dataset.name
    meta["trial_id"] = trial.trial_id
    meta["split"] = trial.split
    meta["axes"] = trial.axes
    meta["subject"] = trial.subject.as_dict()
    meta["license"] = lic.as_dict()
    if trial.objects:
        meta["objects"] = list(trial.objects)
    return qpos, freq, meta


__all__ = [
    "DatasetLicense",
    "JointSpaceClipSource",
    "KinematicTrial",
    "ManifestMotionDataset",
    "MotionDataset",
    "SubjectMetadata",
    "ensure_redistributable",
    "retarget_trial",
]
