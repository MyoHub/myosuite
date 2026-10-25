# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Pinned BoneHub bone references and reviewed Slicer landmarks for scaling."""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike

from myosuite.integrations.capture.markers import MarkerTrial

PINNED_REVISION = "7d520ffe410b3f79d82991b8368b2d51844feb80"
DATASET_ID = "BoneHub/visible-human-3d-models"
VSD_DATASET_ID = "BoneHub/vsd-lower-extremities-seg"
VSD_PINNED_REVISION = "85214b5a182b980a7869b57efba89a59a2445875"
_DATASET_ROOT = Path("visible_human_3d_models") / "CT"
_SUBJECTS = ("01_Male", "02_Female")
_ANATOMY = {
    "SKULL",
    "SPINAL_COLUMN",
    "THORAX",
    *(
        f"{region}_{side}"
        for region in ("UPPER_EXTREMITY", "LOWER_EXTREMITY", "HAND", "FOOT")
        for side in ("LEFT", "RIGHT")
    ),
}
_ATTRIBUTION = (
    "Alavi, S. H. and Asseln, M. (2026), visible-human-3d-models, "
    "https://doi.org/10.57967/hf/10464; "
    "Andreassen et al. (2023), Three Dimensional Lower Extremity "
    "Musculoskeletal Geometry of the Visible Human Female and Male, "
    "https://doi.org/10.1038/s41597-022-01905-2"
)
_VSD_SUBJECTS = (
    "002",
    "006",
    "010",
    "014",
    "015",
    "016",
    "017",
    "019",
    "023",
    "z001",
    "z004",
    "z009",
    "z013",
    "z019",
    "z023",
    "z027",
    "z035",
    "z036",
    "z042",
    "z046",
    "z049",
    "z050",
    "z055",
    "z056",
    "z057",
    "z061",
    "z062",
    "z063",
    "z064",
    "z066",
)
_VSD_QUALITY = {
    "z013": "Conflicting source metadata; intraosseous access in left tibia.",
    "z036": "Missing source height and mass.",
    "z049": "Source metadata sex differs from biological sex visible in CT.",
    "z050": "Hinged total knee replacement in right knee.",
    "z055": "Conflicting source weight information.",
    "z057": "Missing source height and mass.",
    "z061": "Right distal toe phalanges are cut off.",
    "z063": "Spinal fusion, bilateral hip replacements and right knee replacement.",
    "z064": "Missing source height and mass.",
    "z066": "Cut-off foot bones, metal artifacts and missing height/mass.",
}


@dataclass(frozen=True)
class _BoneHubMarkerTrial(MarkerTrial):
    _evidence_json: str = "{}"

    def metadata(self) -> dict[str, Any]:
        """Retain dataset and annotation provenance in scale-estimation evidence."""
        return {**super().metadata(), "bonehub": json.loads(self._evidence_json)}


@dataclass(frozen=True)
class _BoneReference(ABC):
    """Shared local paths and reviewed Slicer landmark conversion."""

    root: Path
    subject: str
    revision: str
    dataset_id: ClassVar[str]

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", self.revision
        ):
            raise ValueError("revision must be a full 40-character commit SHA.")
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "revision", self.revision.lower())

    @property
    def snapshot_root(self) -> Path:
        """Return the local directory reserved for this dataset revision."""
        return self.root / self.revision

    def _metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_id,
            "subject": self.subject,
            "revision": self.revision,
            "url": f"https://huggingface.co/datasets/{self.dataset_id}/tree/{self.revision}",
        }

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return source-specific identity and provenance."""

    def _local_path(self, relative: Path) -> Path:
        if not self.snapshot_root.resolve().is_relative_to(self.root):
            raise ValueError("Snapshot directory leaves the reference root.")
        candidate = (self.snapshot_root / relative).resolve()
        if not candidate.is_relative_to(self.snapshot_root.resolve()):
            raise ValueError("Reference path leaves the dataset snapshot.")
        return candidate

    def _download_file(self, relative: Path) -> Path:
        from huggingface_hub import hf_hub_download

        path = self._local_path(relative)
        hf_hub_download(
            filename=relative.as_posix(),
            repo_id=self.dataset_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=str(self.snapshot_root),
        )
        return path

    def read_markups(
        self,
        path: str | Path,
        *,
        units: Literal["m", "cm", "mm"],
        rotation_to_model: ArrayLike,
        annotation_source: str,
        definition_license: str,
        translation_to_model_m: ArrayLike = (0.0, 0.0, 0.0),
        confidence: float | None = None,
        uncertainty_m: float | None = None,
    ) -> MarkerTrial:
        """Read reviewed Slicer fiducials into the existing marker-scale adapter.

        Coordinates must already be exported in the reference's patient frame
        (harden any Slicer scene transforms before export). Each markup must
        declare LPS or RAS. RAS is rotated into LPS before the caller's proper
        LPS-to-model rotation. Units are explicitly confirmed against any file
        declaration. Undefined/preview points remain missing observations.

        The caller associates this annotation with the chosen subject/revision;
        coordinates cannot establish subject identity. ``definition_license``
        describes the user's annotation, separately from the geometry license;
        ``annotation_source`` identifies its author or review protocol.
        No joint centres, reference lengths or scale factors are inferred.
        """
        if units not in ("m", "cm", "mm"):
            raise ValueError("units must be explicitly m, cm, or mm.")
        if not isinstance(definition_license, str) or not definition_license.strip():
            raise ValueError("The annotation definition_license is required.")
        if not isinstance(annotation_source, str) or not annotation_source.strip():
            raise ValueError("The annotation_source is required.")
        path = Path(path).resolve()
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Expected a Slicer markups JSON object.")
        groups = payload.get("markups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("Expected a nonempty Slicer markups list.")
        labels, positions, frames = [], [], []
        for markup in groups:
            if not isinstance(markup, dict) or markup.get("type") != "Fiducial":
                raise ValueError("Only Slicer Fiducial markups are supported.")
            frame = markup.get("coordinateSystem")
            if frame not in ("LPS", "RAS"):
                raise ValueError("Each markup must explicitly declare LPS or RAS.")
            declared_units = markup.get("coordinateUnits")
            if isinstance(declared_units, list):
                if len(declared_units) != 3 or declared_units[1] != "UCUM":
                    raise ValueError("Unsupported Slicer coordinateUnits code.")
                declared_units = declared_units[0]
            if declared_units is not None and declared_units != units:
                raise ValueError("Confirmed units do not match markup coordinateUnits.")
            points = markup.get("controlPoints")
            if not isinstance(points, list) or not points:
                raise ValueError("Each fiducial markup needs controlPoints.")
            frames.append(frame)
            for point in points:
                if not isinstance(point, dict):
                    raise ValueError("Expected a fiducial control point object.")
                labels.append(point.get("label"))
                status = point.get("positionStatus", "defined")
                if status not in ("defined", "undefined", "preview"):
                    raise ValueError("Unsupported markup positionStatus.")
                position = np.full(3, np.nan)
                if status == "defined":
                    position = np.asarray(point.get("position"), dtype=float)
                    if position.shape != (3,) or not np.all(np.isfinite(position)):
                        raise ValueError(
                            "Defined fiducials need three finite coordinates."
                        )
                    if frame == "RAS":
                        position = position * (-1, -1, 1)
                positions.append(position)
        evidence = {
            **self.metadata(),
            "annotation": str(path),
            "annotation_source": annotation_source,
            "annotation_sha256": hashlib.sha256(raw).hexdigest(),
            "annotation_coordinate_systems": frames,
            "annotation_license": definition_license,
        }
        return _BoneHubMarkerTrial(
            labels=tuple(labels),
            positions=np.asarray(positions)[None],
            units=units,
            rotation_to_model=rotation_to_model,
            source=f"{evidence['url']}; subject {self.subject}; annotation {path}",
            translation_to_model_m=translation_to_model_m,
            confidence=confidence,
            uncertainty_m=uncertainty_m,
            definition_source=annotation_source,
            definition_license=definition_license,
            _evidence_json=json.dumps(evidence, sort_keys=True),
        )


@dataclass(frozen=True)
class BoneHubReference(_BoneReference):
    """Pinned Visible Human reference, with opt-in individual bone downloads.

    Args:
        root: Parent directory of revision-separated dataset mirrors.
        subject: Visible Human identifier, not a population template.
        revision: Full dataset commit SHA.
    """

    subject: Literal["01_Male", "02_Female"]
    revision: str = PINNED_REVISION
    dataset_id: ClassVar[str] = DATASET_ID

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.subject not in _SUBJECTS:
            raise ValueError(f"subject must be one of {_SUBJECTS}.")

    def metadata(self) -> dict[str, Any]:
        """Return source attribution and limitations without downloading data."""
        return {
            **self._metadata(),
            "geometry_license": "CC-BY-4.0",
            "attribution": _ATTRIBUTION,
            "coordinates": "LPS: +X left, +Y posterior, +Z superior",
            "limitations": [
                "Two cadaveric subjects do not establish population scaling priors.",
                "Elbow-region humerus, radius and ulna geometry is incomplete.",
                "Finger/toe phalanges are combined per digit; female subject has L6.",
                "Bone meshes do not supply joint centres or muscle attachments.",
            ],
        }

    def _mesh_relative(self, anatomy: str, bone: str) -> Path:
        if anatomy not in _ANATOMY:
            raise ValueError(f"Unknown BoneHub anatomy folder: {anatomy!r}.")
        if not isinstance(bone, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", bone):
            raise ValueError("bone must be an uppercase ASCII dataset identifier.")
        for side in ("LEFT", "RIGHT"):
            if anatomy.endswith(f"_{side}") and bone.endswith(
                f"_{'RIGHT' if side == 'LEFT' else 'LEFT'}"
            ):
                raise ValueError("Bone side does not match the anatomy folder.")
        return _DATASET_ROOT / "Mesh" / self.subject / anatomy / f"{bone}.stl"

    def mesh_path(self, anatomy: str, bone: str) -> Path:
        """Locate an existing STL without downloading or registering geometry."""
        path = self._local_path(self._mesh_relative(anatomy, bone))
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def download_mesh(self, anatomy: str, bone: str) -> Path:
        """Fetch one pinned STL and metadata; omit CT volumes and other bones."""
        relative = self._mesh_relative(anatomy, bone)
        self._local_path(relative)
        metadata = self._download_file(_DATASET_ROOT / "metadata.json")
        payload = json.loads(metadata.read_text("utf-8"))
        if not isinstance(payload, dict) or self.subject not in payload:
            raise ValueError("Downloaded metadata does not contain the subject.")
        self._download_file(relative)
        return self.mesh_path(anatomy, bone)


@dataclass(frozen=True)
class BoneHubVsdReference(_BoneReference):
    """Pinned VSD lower-extremity case with source quality and licence evidence.

    Args:
        root: Parent directory of dataset/revision-separated local mirrors.
        subject: BoneHub case identifier, from ``001`` through ``030``.
        revision: Full dataset commit SHA. Geometry is CC BY-NC-SA 4.0.

    Source metadata must be downloaded before reading reviewed markups, so the
    original VSD subject and known quality limitations accompany scale evidence.
    """

    revision: str = VSD_PINNED_REVISION
    dataset_id: ClassVar[str] = VSD_DATASET_ID

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.subject, str) or self.subject not in {
            f"{i:03d}" for i in range(1, 31)
        }:
            raise ValueError("subject must be a VSD case from '001' through '030'.")

    @property
    def snapshot_root(self) -> Path:
        """Keep the VSD source separate from the Visible Human geometry."""
        return self.root / "vsd-lower-extremities-seg" / self.revision

    def metadata(self) -> dict[str, Any]:
        """Read local subject identity and quality; do not infer scale factors."""
        path = self._local_path(Path("metadata") / f"{self.subject}.json")
        if not path.is_file():
            raise FileNotFoundError(f"Download the reference metadata first: {path}")
        raw = path.read_bytes()
        payload = json.loads(raw)
        expected_subject = _VSD_SUBJECTS[int(self.subject) - 1]
        if (
            not isinstance(payload, dict)
            or payload.get("source_subject") != expected_subject
        ):
            raise ValueError("VSD metadata source_subject does not match the case.")
        snapshot = payload.get("subjectSnapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("VSD metadata requires a subjectSnapshot object.")
        flags = (
            [_VSD_QUALITY[expected_subject]] if expected_subject in _VSD_QUALITY else []
        )
        return {
            **self._metadata(),
            "source_subject": expected_subject,
            "source_metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "source_measurements": {
                "height_m": snapshot.get("heightInMeters"),
                "mass_kg": snapshot.get("weightInKilograms"),
                "age_days": snapshot.get("ageInDays"),
            },
            "geometry_license": "CC-BY-NC-SA-4.0",
            "attribution": (
                "Alavi and Asseln (2026), vsd-lower-extremities-seg, "
                "https://doi.org/10.57967/hf/10471; Fischer (2023), "
                "https://doi.org/10.1038/s41597-023-02669-z"
            ),
            "coordinates": "Source CT physical coordinates; confirm mesh import units/frame.",
            "quality_flags": flags,
            "quality_source": "https://doi.org/10.1038/s41597-023-02669-z",
            "source_in_complete_cohort": not flags,
            "limitations": [
                "30 cadaveric cases; source paper selects 20 complete and flags 10 others.",
                "Case 026 lacks right distal toe phalanges 2-4; case 030 lacks all phalanges.",
                "The feet dataset reuses the same subjects; do not count it as another cohort.",
                "Bone geometry alone does not provide functional muscle parameters.",
            ],
        }

    def _mesh_relative(self, bone: str) -> Path:
        if not isinstance(bone, str) or not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", bone):
            raise ValueError("bone must be an ASCII dataset label such as 'Patella_R'.")
        return Path("meshes") / self.subject / f"{bone}.stl"

    def mesh_path(self, bone: str) -> Path:
        """Locate one existing VSD bone mesh without downloading it."""
        path = self._local_path(self._mesh_relative(bone))
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def download_mesh(self, bone: str) -> Path:
        """Fetch one labelled STL, its subject record and the small label map."""
        relative = self._mesh_relative(bone)
        self._local_path(relative)
        self._download_file(Path("metadata") / f"{self.subject}.json")
        self.metadata()
        labels_file = self._download_file(Path("lower_extremity_ct/dataset.json"))
        payload = json.loads(labels_file.read_text("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("labels"), dict):
            raise ValueError("VSD label dictionary must contain a labels object.")
        labels = payload["labels"]
        if bone not in labels or labels[bone] == 0:
            raise ValueError(f"Unknown VSD bone label: {bone!r}.")
        self._download_file(relative)
        return self.mesh_path(bone)
