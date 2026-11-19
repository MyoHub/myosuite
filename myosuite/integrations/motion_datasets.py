# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Concrete :class:`~myosuite.biomechanics.motion_dataset.MotionDataset` adapters.

Provides ready-to-use dataset wrappers for the primary motion formats in issue #833:

* :class:`AMASSMotionDataset` — AMASS SMPL+H archives (e.g. KIT subset), carrying the
  AMASS research licence (``redistributable_derivatives=False``, ``requires_registration=True``).
* :class:`OpenSimMotDataset` — OpenSim ``.mot`` / ``.sto`` inverse-kinematics folders.
* :class:`MarkerClipSource` and :class:`MarkerMotionDataset` — optical motion-capture
  marker files (``.trc``, ``.c3d``, ``.csv``) via :mod:`myosuite.integrations.capture.markers`,
  supporting paired static calibration trials (``paths={"dynamic": ..., "static": ...}``).
* :class:`RetargetedNPZDataset` — already-retargeted ``.npz`` trajectory libraries
  (e.g. Gait120 or cached MuscleMimic clips).
* :func:`load_manifest_dataset` — loads JSON or YAML declarative manifests into
  :class:`~myosuite.biomechanics.motion_dataset.ManifestMotionDataset`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.biomechanics.motion_dataset import (
    DatasetLicense,
    JointSpaceClipSource,
    KinematicTrial,
    ManifestMotionDataset,
    SubjectMetadata,
)
from myosuite.biomechanics.opensim_motion import GAIT2392_SIGNS
from myosuite.biomechanics.retarget import (
    ClipSource,
    ClipTargets,
    LandmarkIK,
    LandmarkMap,
    collect_clip_files,
)
from myosuite.biomechanics.smplh import load_amass
from myosuite.integrations.capture.markers import _read_markers
from myosuite.integrations.citations import (
    AMASS_CITATION,
    KIT_MOTION_DATABASE_CITATION,
)

AMASS_RESEARCH_LICENSE = DatasetLicense(
    name="AMASS Research License",
    spdx_or_url=AMASS_CITATION.project_url,
    citation=AMASS_CITATION.bibtex,
    redistributable_derivatives=False,
    requires_registration=True,
)

KIT_AMASS_LICENSE = DatasetLicense(
    name="AMASS / KIT Motion Database License",
    spdx_or_url=KIT_MOTION_DATABASE_CITATION.project_url,
    citation=f"{AMASS_CITATION.bibtex}\n\n{KIT_MOTION_DATABASE_CITATION.bibtex}",
    redistributable_derivatives=False,
    requires_registration=True,
)


class AMASSMotionDataset:
    """Dataset adapter over an AMASS SMPL+H directory, clip list, or single ``.npz`` file."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str = "amass",
        smplh_source: ClipSource | None = None,
        splits: Mapping[str, str] | None = None,
        default_split: str = "train",
        license_meta: DatasetLicense | None = None,
        limit: int = 0,
    ) -> None:
        files, base, inferred_name = collect_clip_files(
            root, default_dataset=name, limit=limit
        )
        self.name = name or inferred_name
        self.base = base
        self.license = license_meta or (
            KIT_AMASS_LICENSE if "kit" in self.name.lower() else AMASS_RESEARCH_LICENSE
        )
        self._smplh_source = smplh_source
        self._files: dict[str, Path] = {}
        self._splits: dict[str, str] = {}
        for f in files:
            if f.name == "shape.npz":
                continue
            rel = (
                f.relative_to(base).with_suffix("").as_posix()
                if f.is_relative_to(base)
                else f.stem
            )
            self._files[rel] = f
            self._splits[rel] = (splits or {}).get(rel, default_split)

    def trial_ids(self, split: str | None = None) -> tuple[str, ...]:
        """Ordered AMASS trial IDs, optionally filtered by *split*."""
        if split is None:
            return tuple(self._files)
        return tuple(tid for tid in self._files if self._splits[tid] == split)

    def get_trial(self, trial_id: str) -> KinematicTrial:
        """Load lightweight subject/fps header metadata and return a :class:`KinematicTrial`."""
        try:
            path = self._files[trial_id]
        except KeyError:
            raise KeyError(
                f"Unknown AMASS trial {trial_id!r} in {self.name!r}"
            ) from None
        raw = load_amass(path)
        parts = trial_id.split("/")
        subj_id = parts[-2] if len(parts) >= 2 else parts[0]
        subject = SubjectMetadata(
            subject_id=subj_id,
            sex=str(raw["gender"]),
            extra={"num_betas": int(np.asarray(raw["betas"]).size)},
        )
        return KinematicTrial(
            trial_id=trial_id,
            format="amass-smplh",
            fps=float(raw["fps"]),
            path=path,
            subject=subject,
            axes="Z-up",
            split=self._splits[trial_id],
            license=self.license,
        )

    def clip_source(self, format_name: str) -> ClipSource:
        """Return the configured :class:`~myosuite.biomechanics.retarget_smplh.SMPLHSource`."""
        if format_name != "amass-smplh":
            raise KeyError(
                f"{self.name!r} only serves 'amass-smplh', got {format_name!r}"
            )
        if self._smplh_source is None:
            raise RuntimeError(
                f"{self.name!r} was initialised without smplh_source; pass "
                "retarget_smplh.smplh_source(model, body_models)[0]."
            )
        return self._smplh_source


class OpenSimMotDataset:
    """Dataset adapter over a folder of OpenSim ``.mot`` / ``.sto`` inverse-kinematics files."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str = "opensim-mot",
        license_meta: DatasetLicense,
        rate: float | None = None,
        signs: Mapping[str, float] = GAIT2392_SIGNS,
        splits: Mapping[str, str] | None = None,
        subjects: Mapping[str, SubjectMetadata] | None = None,
    ) -> None:
        base = Path(root)
        self.name = name
        self.license = license_meta
        self._source = JointSpaceClipSource(rate=rate, signs=signs, name="opensim-mot")
        files = (
            sorted([*base.rglob("*.mot"), *base.rglob("*.sto")])
            if base.is_dir()
            else [base]
        )
        self._trials: dict[str, KinematicTrial] = {}
        for f in files:
            tid = (
                f.relative_to(base).with_suffix("").as_posix()
                if base.is_dir()
                else f.stem
            )
            subj_key = tid.split("/")[0]
            subj = (subjects or {}).get(subj_key, SubjectMetadata(subject_id=subj_key))
            self._trials[tid] = KinematicTrial(
                trial_id=tid,
                format="opensim-mot",
                fps=rate,
                path=f,
                subject=subj,
                axes="Y-up",
                split=(splits or {}).get(tid, "train"),
                license=self.license,
            )

    def trial_ids(self, split: str | None = None) -> tuple[str, ...]:
        if split is None:
            return tuple(self._trials)
        return tuple(tid for tid, tr in self._trials.items() if tr.split == split)

    def get_trial(self, trial_id: str) -> KinematicTrial:
        return self._trials[trial_id]

    def clip_source(self, format_name: str) -> ClipSource:
        if format_name not in ("opensim-mot", "joint-space"):
            raise KeyError(f"Unsupported format {format_name!r} for {self.name!r}")
        return self._source


@dataclass
class MarkerClipSource:
    """A :class:`~myosuite.biomechanics.retarget.ClipSource` for optical marker files (``.trc``, ``.c3d``, ``.csv``).

    Reads 3-D marker trajectories in metres via :func:`~myosuite.integrations.capture.markers._read_markers`
    (or from ``trial.arrays['markers']`` / ``trial.arrays['marker_names']``) and aligns axes
    into the Z-up model world frame. When ``trial.paths`` includes a ``"static"`` trial,
    records static-trial marker medians in metadata for downstream segment-scale audits.
    """

    axes: str = "Z-up"
    default_fps: float = 100.0
    name: str = "markers"
    _last_extra: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _to_z_up(self, pos: np.ndarray, axes: str) -> np.ndarray:
        if axes.upper() == "Y-UP":
            # OpenSim Y-up (X forward, Y up, Z right) -> Z-up (X forward, Y left, Z up)
            return np.stack([pos[..., 0], -pos[..., 2], pos[..., 1]], axis=-1)
        return pos

    def targets(
        self,
        clip: str | Path | KinematicTrial,
        model: mujoco.MjModel,
        lmap: LandmarkMap,
    ) -> ClipTargets:
        """Assemble landmark positions ``(T, L, 3)`` on *model* from marker trajectories."""
        static_summary: dict[str, list[float]] = {}
        axes = self.axes
        if isinstance(clip, KinematicTrial):
            axes = clip.axes or self.axes
            if "markers" in clip.arrays and "marker_names" in clip.arrays:
                names = tuple(str(x) for x in clip.arrays["marker_names"])
                positions = np.asarray(clip.arrays["markers"], dtype=np.float64)
                fps = float(clip.fps or self.default_fps)
            else:
                dyn_path = clip.paths.get("dynamic") or clip.path
                if dyn_path is None:
                    raise ValueError(
                        f"Marker trial {clip.trial_id!r} lacks dynamic path or arrays"
                    )
                mdata = _read_markers(dyn_path)
                names, positions = mdata.names, mdata.positions
                fps = float(clip.fps or mdata.rate or self.default_fps)
            if "static" in clip.paths:
                sdata = _read_markers(clip.paths["static"])
                spos = self._to_z_up(sdata.positions, axes)
                med = np.nanmedian(spos, axis=0)
                static_summary = {
                    nm: [round(float(v), 5) for v in med[i]]
                    for i, nm in enumerate(sdata.names)
                }
        else:
            mdata = _read_markers(clip)
            names, positions = mdata.names, mdata.positions
            fps = float(mdata.rate or self.default_fps)

        positions = self._to_z_up(positions, axes)
        idx_map = {nm: i for i, nm in enumerate(names)}
        missing = [lm.source for lm in lmap.landmarks if lm.source not in idx_map]
        if missing:
            raise KeyError(f"Marker clip missing LandmarkMap sources: {missing}")

        pos = np.stack(
            [positions[:, idx_map[lm.source]] for lm in lmap.landmarks], axis=1
        )
        n_frames = len(pos)
        quat = np.zeros((n_frames, len(lmap.orientations), 4), dtype=np.float64)
        quat_mask: np.ndarray | None = None
        if len(lmap.orientations):
            quat[..., 0] = 1.0
            quat_mask = np.zeros((n_frames, len(lmap.orientations)), dtype=np.float64)
        seed = model.qpos0.copy()
        free_jnts = [
            j
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if free_jnts and pos.shape[1] > 0:
            root_body_id = int(model.jnt_bodyid[free_jnts[0]])
            adr = int(model.jnt_qposadr[free_jnts[0]])
            root_lms = [
                i
                for i, lm in enumerate(lmap.landmarks)
                if model.body(lm.body).id == root_body_id
                and np.isfinite(pos[0, i]).all()
            ]
            if root_lms:
                seed[adr : adr + 3] = pos[0, root_lms[0]]
            else:
                finite0 = pos[0, np.isfinite(pos[0]).all(axis=-1)]
                if len(finite0):
                    seed[adr : adr + 2] = finite0.mean(axis=0)[:2]
        contact = np.zeros((n_frames, 2), dtype=bool)
        self._last_extra = {
            "marker_count": len(names),
            "axes_source": axes,
            **({"static_marker_medians_m": static_summary} if static_summary else {}),
        }
        return ClipTargets(
            frequency=fps,
            pos=pos,
            quat=quat,
            seed=seed,
            contact=contact,
            quat_mask=quat_mask,
        )

    def finish(
        self, ik: LandmarkIK, targets: ClipTargets, qpos: np.ndarray
    ) -> tuple[np.ndarray, ClipTargets, dict[str, Any]]:
        return qpos, targets, dict(self._last_extra)


class MarkerMotionDataset:
    """Dataset adapter over ``.trc`` / ``.c3d`` / ``.csv`` optical motion-capture trials."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str = "markers",
        license_meta: DatasetLicense,
        axes: str = "Z-up",
        static_trials: Mapping[str, str | Path] | None = None,
        splits: Mapping[str, str] | None = None,
        subjects: Mapping[str, SubjectMetadata] | None = None,
    ) -> None:
        base = Path(root)
        self.name = name
        self.license = license_meta
        self._source = MarkerClipSource(axes=axes)
        exts = ("*.trc", "*.c3d", "*.csv")
        files = (
            sorted([f for pat in exts for f in base.rglob(pat)])
            if base.is_dir()
            else [base]
        )
        static_paths = {k: Path(v) for k, v in (static_trials or {}).items()}
        static_set = set(static_paths.values())
        self._trials: dict[str, KinematicTrial] = {}
        for f in files:
            if f in static_set or "static" in f.stem.lower():
                continue
            tid = (
                f.relative_to(base).with_suffix("").as_posix()
                if base.is_dir()
                else f.stem
            )
            subj_key = tid.split("/")[0]
            subj = (subjects or {}).get(subj_key, SubjectMetadata(subject_id=subj_key))
            paths = {"dynamic": f}
            if tid in static_paths:
                paths["static"] = static_paths[tid]
            elif subj_key in static_paths:
                paths["static"] = static_paths[subj_key]
            self._trials[tid] = KinematicTrial(
                trial_id=tid,
                format="markers",
                path=f,
                paths=paths,
                subject=subj,
                axes=axes,
                split=(splits or {}).get(tid, "train"),
                license=self.license,
            )

    def trial_ids(self, split: str | None = None) -> tuple[str, ...]:
        if split is None:
            return tuple(self._trials)
        return tuple(tid for tid, tr in self._trials.items() if tr.split == split)

    def get_trial(self, trial_id: str) -> KinematicTrial:
        return self._trials[trial_id]

    def clip_source(self, format_name: str) -> ClipSource:
        if format_name != "markers":
            raise KeyError(f"Unsupported format {format_name!r} for {self.name!r}")
        return self._source


class RetargetedNPZDataset:
    """Dataset adapter over already-retargeted ``.npz`` clips (e.g. Gait120 or refit caches)."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str = "retargeted-npz",
        license_meta: DatasetLicense,
        splits: Mapping[str, str] | None = None,
        limit: int = 0,
    ) -> None:
        files, base, inferred = collect_clip_files(
            root, default_dataset=name, limit=limit
        )
        self.name = name or inferred
        self.license = license_meta
        self._source = JointSpaceClipSource(name="retargeted-npz")
        self._trials: dict[str, KinematicTrial] = {}
        for f in files:
            tid = (
                f.relative_to(base).with_suffix("").as_posix()
                if f.is_relative_to(base)
                else f.stem
            )
            self._trials[tid] = KinematicTrial(
                trial_id=tid,
                format="retargeted-npz",
                path=f,
                axes="Z-up",
                split=(splits or {}).get(tid, "train"),
                license=self.license,
            )

    def trial_ids(self, split: str | None = None) -> tuple[str, ...]:
        if split is None:
            return tuple(self._trials)
        return tuple(tid for tid, tr in self._trials.items() if tr.split == split)

    def get_trial(self, trial_id: str) -> KinematicTrial:
        return self._trials[trial_id]

    def clip_source(self, format_name: str) -> ClipSource:
        if format_name not in ("retargeted-npz", "joint-space"):
            raise KeyError(f"Unsupported format {format_name!r} for {self.name!r}")
        return self._source


def load_manifest_dataset(
    manifest_path: str | Path,
    *,
    root_dir: str | Path | None = None,
    sources: Mapping[str, ClipSource] | None = None,
) -> ManifestMotionDataset:
    """Load a JSON or YAML motion-dataset manifest with default marker and joint-space sources."""
    path = Path(manifest_path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # type: ignore[import-untyped]

        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        doc = json.loads(path.read_text(encoding="utf-8"))
    merged_sources: dict[str, ClipSource] = {"markers": MarkerClipSource()}
    if sources:
        merged_sources.update(sources)
    return ManifestMotionDataset(
        doc,
        root_dir=Path(root_dir) if root_dir is not None else path.parent,
        sources=merged_sources,
    )


__all__ = [
    "AMASS_RESEARCH_LICENSE",
    "KIT_AMASS_LICENSE",
    "AMASSMotionDataset",
    "MarkerClipSource",
    "MarkerMotionDataset",
    "OpenSimMotDataset",
    "RetargetedNPZDataset",
    "load_manifest_dataset",
]
