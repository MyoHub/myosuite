# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight job and output adapters for external MOTOR runs.

`MOTOR <https://github.com/Babon-Innovations-b-v/motor>`_ solves offline kinematics,
residual reduction, ground reaction forces, and static muscle optimization on an
external ``newton`` + ``mujoco-warp`` Python 3.12 stack. Because MOTOR pins exact
solver versions outside MyoSuite's supported dependency range, MOTOR runs in its own
environment and exchanges plain ``job.json`` / ``out.json`` + ``out.npz`` artifacts
with MyoSuite through this module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from myosuite.core.trajectory_io import MotionClip
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

# MOTOR expresses person outputs in Y-up (+X left, +Y up, +Z forward), obtained from
# MuJoCo world coordinates (+X left, -Y forward, +Z up) via TO_OWN @ v_mj.
MUJOCO_TO_MOTOR_FRAME: np.ndarray = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
MOTOR_TO_MUJOCO_FRAME: np.ndarray = MUJOCO_TO_MOTOR_FRAME.T.copy()
MUJOCO_TO_MOTOR_FRAME.setflags(write=False)
MOTOR_TO_MUJOCO_FRAME.setflags(write=False)

DEFAULT_PART_BODIES: Mapping[str, str] = {
    "LeftFoot": "calcn_l",
    "RightFoot": "calcn_r",
}
_WRAPPER_SUFFIX = re.compile(r"__j\d+$")


def _norm_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_")


def unmangle_motor_joint_name(
    raw_name: str, model_joints: Sequence[tuple[str, str]] | None = None
) -> str:
    """Recover a canonical MuJoCo joint name from a Newton hierarchical label.

    MOTOR exports hinge labels from Newton's ``SolverMuJoCo`` model, which prefixes
    ancestor bodies and ``__j<k>`` multi-joint wrapper bodies onto each joint name.

    Args:
        raw_name: Raw coordinate or joint string from ``out.npz`` or ``out.json``.
        model_joints: Optional ``(body_name, joint_name)`` pairs from a target
            ``MjModel`` used to disambiguate joints whose names contain underscores.

    Returns:
        Unmangled joint name when uniquely matched against *model_joints*, or the
        trailing identifier otherwise.
    """
    if model_joints is not None:
        exact_body_hits: list[str] = []
        suffix_hits: list[str] = []
        for body_name, joint_name in model_joints:
            nb, nj = _norm_name(body_name), _norm_name(joint_name)
            base_body = _WRAPPER_SUFFIX.sub("", nb)
            if raw_name == nj or raw_name == joint_name:
                return joint_name
            if raw_name.endswith(f"_{nb}_{nj}") or re.search(
                rf"(?:^|_){re.escape(base_body)}__j\d+_{re.escape(nj)}$", raw_name
            ):
                exact_body_hits.append(joint_name)
            elif raw_name.endswith(f"_{nj}"):
                suffix_hits.append(joint_name)
        if len(exact_body_hits) == 1:
            return exact_body_hits[0]
        if len(suffix_hits) == 1:
            return suffix_hits[0]
    return raw_name.rsplit("_", 1)[-1]


@dataclass(frozen=True)
class TouchWrenchSeries:
    """Time series of ground reaction forces and centers of pressure for one body part.

    Attributes:
        part: MOTOR part key (for example ``"LeftFoot"`` or ``"RightFoot"``).
        force: ``(T, 3)`` MuJoCo Z-up world force applied by the ground [N].
        cop: ``(T, 3)`` MuJoCo Z-up world center of pressure [m], ``NaN`` off contact.
        in_contact: ``(T,)`` boolean mask indicating active contact frames.
    """

    part: str
    force: np.ndarray
    cop: np.ndarray
    in_contact: np.ndarray


@dataclass(frozen=True)
class ResidualCheckSummary:
    """Evaluation of floating-base residuals against Hicks et al. (2015) thresholds.

    Attributes:
        median_force_bw: Median residual force norm as a fraction of body weight.
        p95_force_bw: 95th-percentile residual force norm as a fraction of body weight.
        median_moment_bw_ht: Median residual moment norm over ``weight * com_height``.
        fraction_within_limits: Share of frames satisfying both force and moment bounds.
        passed_all_frames: Whether every frame satisfied both bounds.
    """

    median_force_bw: float
    p95_force_bw: float
    median_moment_bw_ht: float
    fraction_within_limits: float
    passed_all_frames: bool


@dataclass(frozen=True)
class MotorTrial:
    """Parsed single-subject trajectory and kinetic solution from a MOTOR run.

    All spatial vectors and poses on this dataclass are converted from MOTOR's Y-up
    person frame into MuJoCo's Z-up world frame (quaternions in ``wxyz`` order).

    Attributes:
        person: Subject identifier in ``out.json``.
        rate_hz: Recording frame rate [Hz].
        mass_kg: Final model mass reported by MOTOR [kg].
        weight_n: Body weight ``mass * gravity`` [N].
        com_height_m: Standing center-of-mass height over ground at frame 0 [m].
        coordinate_names: Raw coordinate names from ``<person>/coordinate names``.
        coordinates: ``(T, C)`` hinge angles [rad] from ``<person>/coordinates``.
        bone_names: Ordered mesh-bearing body names matching ``bone_poses``.
        bone_poses: ``(T, B, 7)`` Z-up body poses ``[x, y, z, qw, qx, qy, qz]``.
        ground_force: ``(T, 3)`` summed Z-up ground reaction force [N].
        touches: Per-part Z-up contact force and CoP series.
        pelvis_residual: ``(T, 6)`` Z-up pelvis residual ``[Fx, Fy, Fz, Mx, My, Mz]``.
        max_joint_reserve: ``(T,)`` peak absolute joint reserve moment [N m].
        joint_names: Raw driven-DoF joint labels from ``out.json``.
        coupled_torques: ``(T, J)`` coupled generalized moments [N m].
        own_torques: ``(T, J)`` single-joint inverse-dynamics moments [N m].
        muscle_names: Muscle actuator names in MOTOR's column order.
        activations: ``(T, M)`` muscle activations in ``[0, 1]``.
        muscle_tension: ``(T, M)`` positive muscle tensile forces [N].
        rides: Mapping from SOMA segment name to ridden MuJoCo body name.
        rra_mass_delta_pct: RRA percentage mass change reported in ``adjusted``.
        rra_torso_com_delta_m: ``(3,)`` RRA torso COM shift in torso frame [m].
        seconds: Per-stage wall-clock runtime dictionary [s].
        engine: Solver provenance string written by MOTOR.
        source_json: Path to ``out.json`` when loaded from disk.
    """

    person: str
    rate_hz: float
    mass_kg: float
    weight_n: float
    com_height_m: float
    coordinate_names: tuple[str, ...]
    coordinates: np.ndarray
    bone_names: tuple[str, ...]
    bone_poses: np.ndarray
    ground_force: np.ndarray
    touches: Mapping[str, TouchWrenchSeries]
    pelvis_residual: np.ndarray
    max_joint_reserve: np.ndarray
    joint_names: tuple[str, ...]
    coupled_torques: np.ndarray
    own_torques: np.ndarray
    muscle_names: tuple[str, ...]
    activations: np.ndarray
    muscle_tension: np.ndarray
    rides: Mapping[str, str] = field(default_factory=dict)
    rra_mass_delta_pct: float = 0.0
    rra_torso_com_delta_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    seconds: Mapping[str, float] = field(default_factory=dict)
    engine: str = ""
    source_json: Path | None = None

    @property
    def num_frames(self) -> int:
        """Number of frames ``T`` in the solved recording."""
        return int(self.coordinates.shape[0])

    @property
    def dt(self) -> float:
        """Frame period ``1 / rate_hz`` [s]."""
        return 1.0 / float(self.rate_hz)

    def check_residuals(
        self,
        *,
        force_limit_bw: float = 0.02,
        moment_limit_bw_ht: float = 0.01,
    ) -> ResidualCheckSummary:
        """Evaluate pelvis residuals against MOTOR / Hicks et al. (2015) bounds.

        Args:
            force_limit_bw: Maximum residual force as a fraction of ``weight_n``.
            moment_limit_bw_ht: Maximum residual moment over ``weight_n * com_height_m``.

        Returns:
            Summary statistics and frame pass fraction.
        """
        if force_limit_bw <= 0 or moment_limit_bw_ht <= 0:
            raise ValueError("Residual limits must be positive.")
        denom_f = max(float(self.weight_n), 1e-12)
        denom_m = max(float(self.weight_n * self.com_height_m), 1e-12)
        force_bw = np.linalg.norm(self.pelvis_residual[:, :3], axis=1) / denom_f
        moment_bw_ht = np.linalg.norm(self.pelvis_residual[:, 3:], axis=1) / denom_m
        within = (force_bw <= force_limit_bw) & (moment_bw_ht <= moment_limit_bw_ht)
        return ResidualCheckSummary(
            median_force_bw=float(np.median(force_bw)),
            p95_force_bw=float(np.percentile(force_bw, 95.0)),
            median_moment_bw_ht=float(np.median(moment_bw_ht)),
            fraction_within_limits=float(np.mean(within)),
            passed_all_frames=bool(np.all(within)),
        )


@dataclass(frozen=True)
class MotorPersonInput:
    """Per-subject input specification for writing a MOTOR ``job.json`` bundle.

    Attributes:
        name: Unique subject identifier (used as key in ``out.json`` and ``out.npz``).
        mass_kg: Subject body mass [kg].
        height_m: Subject standing height [m].
        rig_names: Ordered rig joint names (length ``R``).
        rig_parents: Parent index per rig joint (``-1`` for root, length ``R``).
        rig_bind: ``(R, 4, 4)`` bind-pose world transforms in Y-up meters.
        segments: Driven segment names (length ``S``), ordered to match ``world``.
        world: ``(T, S, 4, 4)`` posed segment transforms in Y-up meters.
        ground: ``(G, 3, 3)`` floor triangle vertices in Y-up meters.
        points: ``(V, 3)`` bind-pose skin mesh vertices in Y-up meters.
        faces: ``(F, 3)`` skin triangle vertex indices.
        weights: ``(V, R)`` linear-blend skinning weights over ``rig_names``.
    """

    name: str
    mass_kg: float
    height_m: float
    rig_names: tuple[str, ...]
    rig_parents: tuple[int, ...]
    rig_bind: np.ndarray
    segments: tuple[str, ...]
    world: np.ndarray
    ground: np.ndarray
    points: np.ndarray
    faces: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class MotorJobConfig:
    """Solver settings written into a MOTOR ``job.json`` file."""

    rate_hz: float = 30.0
    gravity: float = 9.81
    residual_force_bw: float = 0.02
    residual_moment_bw_ht: float = 0.01
    move_most_m: float = 0.02
    turn_most_deg: float = 5.0
    follows_deg: float = 5.0
    feet_kept: bool = True
    whole_trial: Literal["dynamics", "muscles"] | bool | None = None
    model_dir: Path | None = None


def write_motor_job(
    out_dir: str | Path,
    people: Sequence[MotorPersonInput],
    config: MotorJobConfig | None = None,
) -> Path:
    """Write ``job.json`` and per-person ``<name>_targets.npz`` files for MOTOR.

    Args:
        out_dir: Directory where ``job.json`` and target ``.npz`` files are written.
        people: One or more subject specifications.
        config: Solver and RRA configuration (defaults to :class:`MotorJobConfig`).

    Returns:
        Path to the written ``job.json``.

    Raises:
        ValueError: If *people* is empty, names collide, or array dimensions disagree.
    """
    if not people:
        raise ValueError("write_motor_job requires at least one MotorPersonInput.")
    cfg = config or MotorJobConfig()
    if not (np.isfinite(cfg.rate_hz) and cfg.rate_hz > 0):
        raise ValueError(f"rate_hz must be finite and positive, got {cfg.rate_hz}.")
    if not (np.isfinite(cfg.gravity) and cfg.gravity > 0):
        raise ValueError(f"gravity must be finite and positive, got {cfg.gravity}.")

    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    entries: list[dict[str, Any]] = []

    for person in people:
        if not person.name or person.name in seen_names:
            raise ValueError(f"Duplicate or empty person name {person.name!r}.")
        seen_names.add(person.name)
        if person.mass_kg <= 0 or person.height_m <= 0:
            raise ValueError(f"{person.name}: mass_kg and height_m must be positive.")

        n_rig = len(person.rig_names)
        n_seg = len(person.segments)
        bind = np.asarray(person.rig_bind, dtype=np.float64)
        world = np.asarray(person.world, dtype=np.float64)
        ground = np.asarray(person.ground, dtype=np.float64)
        points = np.asarray(person.points, dtype=np.float64)
        faces = np.asarray(person.faces, dtype=np.int32)
        weights = np.asarray(person.weights, dtype=np.float64)

        if len(person.rig_parents) != n_rig or bind.shape != (n_rig, 4, 4):
            raise ValueError(
                f"{person.name}: rig_parents ({len(person.rig_parents)}) and "
                f"rig_bind {bind.shape} must match {n_rig} rig_names."
            )
        if world.ndim != 4 or world.shape[1:] != (n_seg, 4, 4) or world.shape[0] < 1:
            raise ValueError(
                f"{person.name}: world must have shape (T >= 1, {n_seg}, 4, 4), got {world.shape}."
            )
        if ground.ndim != 3 or ground.shape[1:] != (3, 3) or ground.shape[0] < 1:
            raise ValueError(
                f"{person.name}: ground must have shape (G >= 1, 3, 3), got {ground.shape}."
            )
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError(f"{person.name}: points must have shape (V >= 3, 3).")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
            raise ValueError(f"{person.name}: faces must have shape (F >= 1, 3).")
        if weights.shape != (len(points), n_rig):
            raise ValueError(
                f"{person.name}: weights must have shape ({len(points)}, {n_rig}), got {weights.shape}."
            )

        target_path = folder / f"{person.name}_targets.npz"
        np.savez_compressed(
            target_path,
            world=world,
            ground=ground,
            points=points,
            faces=faces,
            weights=weights,
        )
        entries.append(
            {
                "name": person.name,
                "kg": float(person.mass_kg),
                "height": float(person.height_m),
                "rig": {
                    "names": list(person.rig_names),
                    "parents": [int(p) for p in person.rig_parents],
                    "bind": bind.tolist(),
                },
                "segments": list(person.segments),
                "targets": str(target_path.resolve()),
            }
        )

    job: dict[str, Any] = {
        "rate": float(cfg.rate_hz),
        "gravity": float(cfg.gravity),
        "residual": {
            "force": float(cfg.residual_force_bw),
            "moment": float(cfg.residual_moment_bw_ht),
        },
        "adjust": {
            "move most m": float(cfg.move_most_m),
            "turn most deg": float(cfg.turn_most_deg),
            "follows deg": float(cfg.follows_deg),
            "feet kept": bool(cfg.feet_kept),
        },
        "people": entries,
    }
    if cfg.whole_trial is not None:
        job["motor"] = cfg.whole_trial
    if cfg.model_dir is not None:
        job["model"] = str(Path(cfg.model_dir).resolve())

    job_path = folder / "job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job_path


def _bones_yup_to_zup_wxyz(bones_yup: np.ndarray) -> np.ndarray:
    """Convert MOTOR ``(T, B, 7)`` ``[x, y, z, qx, qy, qz, qw]`` Y-up poses to Z-up ``wxyz``."""
    arr = np.asarray(bones_yup, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 7:
        raise ValueError(f"Expected bones array of shape (T, B, 7), got {arr.shape}.")
    t_len, n_bones, _ = arr.shape
    pos_zup = arr[:, :, :3] @ MOTOR_TO_MUJOCO_FRAME.T
    flat_quat_xyzw = arr[:, :, 3:7].reshape(-1, 4)
    rot_yup = Rotation.from_quat(flat_quat_xyzw).as_matrix()
    rot_zup = np.einsum("ij,njk->nik", MOTOR_TO_MUJOCO_FRAME, rot_yup)
    quat_xyzw_zup = Rotation.from_matrix(rot_zup).as_quat()
    quat_wxyz_zup = np.roll(quat_xyzw_zup, 1, axis=1).reshape(t_len, n_bones, 4)
    return np.concatenate([pos_zup, quat_wxyz_zup], axis=2)


def _parse_person(
    person_name: str,
    entry: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    rate_hz: float,
    engine: str,
    source_json: Path | None,
) -> MotorTrial:
    coords = np.asarray(arrays[f"{person_name}/coordinates"], dtype=np.float64)
    coord_names = tuple(str(x) for x in arrays[f"{person_name}/coordinate names"])
    bones_zup = _bones_yup_to_zup_wxyz(arrays[f"{person_name}/bones"])
    bone_names = tuple(str(b) for b in entry["bones"])
    frames = entry["frames"]
    t_len = len(frames)
    if coords.shape != (t_len, len(coord_names)):
        raise ValueError(
            f"{person_name}: coordinates shape {coords.shape} disagrees with "
            f"{t_len} frames and {len(coord_names)} coordinate names."
        )
    if bones_zup.shape[:2] != (t_len, len(bone_names)):
        raise ValueError(
            f"{person_name}: bones shape {bones_zup.shape} disagrees with "
            f"{t_len} frames and {len(bone_names)} bone names."
        )

    ground_yup = np.asarray([fr["ground"] for fr in frames], dtype=np.float64)
    ground_zup = ground_yup @ MOTOR_TO_MUJOCO_FRAME.T

    res_yup = np.asarray([fr["residual"] for fr in frames], dtype=np.float64)
    res_zup = np.concatenate(
        [
            res_yup[:, :3] @ MOTOR_TO_MUJOCO_FRAME.T,
            res_yup[:, 3:6] @ MOTOR_TO_MUJOCO_FRAME.T,
        ],
        axis=1,
    )

    all_parts = sorted({part for fr in frames for part in (fr.get("touches") or {})})
    touches: dict[str, TouchWrenchSeries] = {}
    for part in all_parts:
        f_yup = np.zeros((t_len, 3), dtype=np.float64)
        cop_yup = np.full((t_len, 3), np.nan, dtype=np.float64)
        active = np.zeros(t_len, dtype=bool)
        for idx, fr in enumerate(frames):
            vals = (fr.get("touches") or {}).get(part)
            if vals is not None:
                f_yup[idx] = vals[:3]
                cop_yup[idx] = vals[3:6]
                active[idx] = True
        touches[part] = TouchWrenchSeries(
            part=part,
            force=f_yup @ MOTOR_TO_MUJOCO_FRAME.T,
            cop=cop_yup @ MOTOR_TO_MUJOCO_FRAME.T,
            in_contact=active,
        )

    adjusted = entry.get("adjusted") or {}
    torso_cm = np.asarray(
        adjusted.get("torso centre of mass cm", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    return MotorTrial(
        person=person_name,
        rate_hz=float(rate_hz),
        mass_kg=float(entry["kg"]),
        weight_n=float(entry["weight N"]),
        com_height_m=float(entry["com height"]),
        coordinate_names=coord_names,
        coordinates=coords,
        bone_names=bone_names,
        bone_poses=bones_zup,
        ground_force=ground_zup,
        touches=touches,
        pelvis_residual=res_zup,
        max_joint_reserve=np.asarray(
            [fr["reserve"] for fr in frames], dtype=np.float64
        ),
        joint_names=tuple(str(j) for j in entry["joints"]),
        coupled_torques=np.asarray([fr["torques"] for fr in frames], dtype=np.float64),
        own_torques=np.asarray([fr["own torques"] for fr in frames], dtype=np.float64),
        muscle_names=tuple(str(m) for m in entry["muscles"]),
        activations=np.asarray([fr["activations"] for fr in frames], dtype=np.float64),
        muscle_tension=np.asarray([fr["forces"] for fr in frames], dtype=np.float64),
        rides={str(k): str(v) for k, v in (entry.get("rides") or {}).items()},
        rra_mass_delta_pct=float(adjusted.get("mass %", 0.0)),
        rra_torso_com_delta_m=torso_cm / 100.0,
        seconds={str(k): float(v) for k, v in (entry.get("seconds") or {}).items()},
        engine=engine,
        source_json=source_json,
    )


def read_motor_outputs(
    out_json: str | Path,
    *,
    out_npz: str | Path | None = None,
) -> dict[str, MotorTrial]:
    """Load all subjects from a MOTOR ``out.json`` and companion ``out.npz``.

    Args:
        out_json: Path to ``out.json``.
        out_npz: Optional explicit path to ``out.npz`` (defaults to ``out_json`` with
            suffix ``.npz``).

    Returns:
        Mapping of ``person`` name to :class:`MotorTrial`.
    """
    json_path = Path(out_json)
    npz_path = Path(out_npz) if out_npz is not None else json_path.with_suffix(".npz")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rate_hz = float(payload["rate"])
    engine = str(payload.get("engine", ""))
    people = payload.get("people") or {}
    if not people:
        raise ValueError(f"No people entries found in {json_path}.")
    with np.load(npz_path, allow_pickle=False) as npz_data:
        arrays = {key: npz_data[key] for key in npz_data.files}
    return {
        name: _parse_person(
            name,
            entry,
            arrays,
            rate_hz=rate_hz,
            engine=engine,
            source_json=json_path,
        )
        for name, entry in people.items()
    }


def read_motor_output(
    out_json: str | Path,
    *,
    person: str | None = None,
    out_npz: str | Path | None = None,
) -> MotorTrial:
    """Load one subject's :class:`MotorTrial` from a MOTOR ``out.json`` bundle.

    Args:
        out_json: Path to ``out.json``.
        person: Subject key when ``out.json`` holds multiple people; optional when
            only one subject is present.
        out_npz: Optional companion ``.npz`` path.

    Returns:
        Parsed :class:`MotorTrial` in MuJoCo Z-up coordinates.
    """
    trials = read_motor_outputs(out_json, out_npz=out_npz)
    if person is not None:
        if person not in trials:
            known = ", ".join(sorted(trials))
            raise KeyError(f"Person {person!r} not in {out_json}; available: {known}")
        return trials[person]
    if len(trials) != 1:
        known = ", ".join(sorted(trials))
        raise ValueError(
            f"{out_json} contains {len(trials)} people ({known}); pass person=..."
        )
    return next(iter(trials.values()))


def _find_body_id(model: mujoco.MjModel, name: str) -> int:
    direct = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if direct >= 0:
        return int(direct)
    target = _norm_name(name)
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and _norm_name(bname) == target:
            return int(bid)
    return -1


def _reconstruct_free_root(
    model: mujoco.MjModel,
    trial: MotorTrial,
    qpos: np.ndarray,
    free_joint: int,
    root_anchor_body: str,
) -> None:
    root_qadr = int(model.jnt_qposadr[free_joint])
    root_bid = int(model.jnt_bodyid[free_joint])
    anchor_bid = _find_body_id(model, root_anchor_body)
    if anchor_bid < 0:
        raise KeyError(
            f"Anchor body {root_anchor_body!r} not found in model for free-root reconstruction."
        )
    if int(model.body_weldid[anchor_bid]) != int(model.body_weldid[root_bid]):
        raise ValueError(
            f"Anchor body {root_anchor_body!r} is not rigidly welded to free-joint body "
            f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_bid)!r}."
        )
    norm_bones = [_norm_name(b) for b in trial.bone_names]
    target_norm = _norm_name(root_anchor_body)
    if target_norm not in norm_bones:
        raise KeyError(
            f"Anchor body {root_anchor_body!r} not present in trial.bone_names {trial.bone_names}."
        )
    bone_idx = norm_bones.index(target_norm)

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[root_qadr : root_qadr + 7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_kinematics(model, data)
    pos_r2a = data.xpos[anchor_bid].copy()
    rot_r2a = data.xmat[anchor_bid].reshape(3, 3).copy()

    anchor_pos = trial.bone_poses[:, bone_idx, :3]
    anchor_wxyz = trial.bone_poses[:, bone_idx, 3:7]
    anchor_xyzw = np.roll(anchor_wxyz, -1, axis=1)
    rot_anchor = Rotation.from_quat(anchor_xyzw).as_matrix()
    rot_root = np.einsum("nij,kj->nik", rot_anchor, rot_r2a)
    pos_root = anchor_pos - np.einsum("nij,j->ni", rot_root, pos_r2a)
    root_wxyz = np.roll(Rotation.from_matrix(rot_root).as_quat(), 1, axis=1)

    qpos[:, root_qadr : root_qadr + 3] = pos_root
    qpos[:, root_qadr + 3 : root_qadr + 7] = root_wxyz


def motor_to_qpos(
    model: mujoco.MjModel,
    trial: MotorTrial,
    *,
    root_anchor_body: str = "pelvis",
    project_couplings: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Map a :class:`MotorTrial` onto ``(T, model.nq)`` generalized coordinates.

    Reconstructs any floating-base ``mjJNT_FREE`` joint from *root_anchor_body*
    in ``trial.bone_poses`` (accounting for rigid ``root -> anchor`` body offsets),
    matches hinge coordinates across Newton wrapper-body prefixes, normalizes
    quaternions, and optionally projects active ``mjEQ_JOINT`` polynomial couplings.

    Args:
        model: Target compiled ``MjModel`` (either MOTOR's sized subject XML or a
            standard MyoSuite musculoskeletal model).
        trial: Parsed :class:`MotorTrial`.
        root_anchor_body: Rigidly welded pelvis/root mesh body in ``trial.bone_names``.
        project_couplings: Project dependent coupled joints via :class:`JointCouplings`.

    Returns:
        ``(qpos, unmapped_coordinates)`` where ``qpos`` has shape ``(T, model.nq)``
        and ``unmapped_coordinates`` lists MOTOR coordinate names absent from *model*.
    """
    t_len = trial.num_frames
    qpos = np.tile(np.asarray(model.qpos0, dtype=np.float64), (t_len, 1))

    free_joints = [
        j
        for j in range(model.njnt)
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if free_joints:
        _reconstruct_free_root(model, trial, qpos, free_joints[0], root_anchor_body)

    hinge_pairs: list[tuple[int, str, str]] = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            bname = (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])
                )
                or ""
            )
            if jname:
                hinge_pairs.append((j, bname, jname))

    model_lookup = [(b, jn) for _, b, jn in hinge_pairs]
    jname_to_id = {jn: j for j, _, jn in hinge_pairs}
    norm_to_id = {_norm_name(jn): j for j, _, jn in hinge_pairs}

    unmapped: list[str] = []
    for col, raw_coord in enumerate(trial.coordinate_names):
        resolved = unmangle_motor_joint_name(raw_coord, model_lookup)
        jid = jname_to_id.get(resolved, norm_to_id.get(_norm_name(resolved), -1))
        if jid >= 0:
            qpos[:, int(model.jnt_qposadr[jid])] = trial.coordinates[:, col]
        else:
            unmapped.append(raw_coord)

    for row in range(t_len):
        mujoco.mj_normalizeQuat(model, qpos[row])

    if project_couplings:
        project_joint_couplings(JointCouplings.from_model(model), qpos)
    return qpos, tuple(unmapped)


def motor_to_motion_clip(
    model: mujoco.MjModel,
    trial: MotorTrial,
    *,
    root_anchor_body: str = "pelvis",
    project_couplings: bool = True,
) -> MotionClip:
    """Convert a :class:`MotorTrial` into a MyoSuite :class:`MotionClip`.

    Args:
        model: Target compiled ``MjModel``.
        trial: Parsed :class:`MotorTrial`.
        root_anchor_body: Mesh body in ``trial.bone_names`` welded to the free root.
        project_couplings: Project active ``mjEQ_JOINT`` polynomial couplings.

    Returns:
        :class:`MotionClip` ready for ``clip_kinematics``, ``inverse_dynamics``,
        ``static_optimization``, ``computed_muscle_control``, or mimic playback.
    """
    qpos, unmapped = motor_to_qpos(
        model,
        trial,
        root_anchor_body=root_anchor_body,
        project_couplings=project_couplings,
    )
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        for j in range(model.njnt)
    ]
    return MotionClip(
        qpos=qpos,
        qvel=None,
        site_xpos=None,
        site_names=None,
        qpos_joint_names=joint_names,
        frequency_hz=trial.rate_hz,
        source_path=trial.source_json,
        metadata={
            "person": trial.person,
            "engine": trial.engine,
            "unmapped_coordinates": list(unmapped),
            "rra_mass_delta_pct": trial.rra_mass_delta_pct,
            "rra_torso_com_delta_m": trial.rra_torso_com_delta_m.tolist(),
        },
    )


def motor_touches_to_qfrc(
    model: mujoco.MjModel,
    trial: MotorTrial,
    qpos: np.ndarray,
    *,
    part_bodies: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Map MOTOR contact forces and centers of pressure into ``(T, model.nv)`` ``qfrc``.

    Uses ``mj_fwdPosition`` (populating ``data.cdof``) followed by ``mj_applyFT`` at
    each touching part's Z-up world center of pressure, matching the ``external_qfrc``
    contract of :func:`myosuite.biomechanics.inverse_dynamics.inverse_dynamics`.

    Args:
        model: Compiled ``MjModel`` (unmodified).
        trial: Parsed :class:`MotorTrial`.
        qpos: ``(T, model.nq)`` generalized positions aligned with *trial*.
        part_bodies: Optional override mapping MOTOR touch keys (such as
            ``"LeftFoot"``) to MuJoCo body names. Defaults to
            :data:`DEFAULT_PART_BODIES` merged with ``trial.rides``.

    Returns:
        ``(T, model.nv)`` generalized contact forces.
    """
    q = np.asarray(qpos, dtype=np.float64)
    if q.shape != (trial.num_frames, model.nq) or not np.all(np.isfinite(q)):
        raise ValueError(
            f"qpos must be finite with shape ({trial.num_frames}, {model.nq}), got {q.shape}."
        )

    body_map: dict[str, str] = {**trial.rides, **DEFAULT_PART_BODIES}
    if part_bodies is not None:
        body_map.update(part_bodies)

    resolved_ids: dict[str, int] = {}
    for part in trial.touches:
        target_body = body_map.get(part, part)
        bid = _find_body_id(model, target_body)
        if bid < 0:
            raise KeyError(
                f"Touch part {part!r} mapped to body {target_body!r}, which is absent from model."
            )
        resolved_ids[part] = bid

    data = mujoco.MjData(model)
    zero_torque = np.zeros(3, dtype=np.float64)
    out = np.zeros((trial.num_frames, model.nv), dtype=np.float64)
    for frame in range(trial.num_frames):
        if not any(series.in_contact[frame] for series in trial.touches.values()):
            continue
        data.qpos[:] = q[frame]
        mujoco.mj_fwdPosition(model, data)
        for part, series in trial.touches.items():
            if series.in_contact[frame]:
                mujoco.mj_applyFT(
                    model,
                    data,
                    series.force[frame],
                    zero_torque,
                    series.cop[frame],
                    resolved_ids[part],
                    out[frame],
                )
    return out


def motor_activations_to_model(
    model: mujoco.MjModel,
    trial: MotorTrial,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align MOTOR muscle activations and tensile forces onto ``model.nu`` actuators.

    MOTOR reports muscle forces as positive tension ``[N]``, whereas MuJoCo's
    ``actuator_force`` convention uses negative values for pulling muscles; the
    returned ``actuator_force`` negates ``trial.muscle_tension`` to match MuJoCo and
    :class:`myosuite.biomechanics.static_optimization.StaticOptimizationResult`.

    Args:
        model: Target compiled ``MjModel``.
        trial: Parsed :class:`MotorTrial`.

    Returns:
        ``(activation, actuator_force, matched_mask)`` with shapes ``(T, nu)``,
        ``(T, nu)``, and ``(nu,)``. Unmatched actuators contain ``NaN``.
    """
    t_len = trial.num_frames
    act = np.full((t_len, model.nu), np.nan, dtype=np.float64)
    frc = np.full((t_len, model.nu), np.nan, dtype=np.float64)
    matched = np.zeros(model.nu, dtype=bool)

    motor_index = {_norm_name(name): idx for idx, name in enumerate(trial.muscle_names)}
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if not aname:
            continue
        src = motor_index.get(_norm_name(aname))
        if src is not None:
            act[:, aid] = trial.activations[:, src]
            frc[:, aid] = -trial.muscle_tension[:, src]
            matched[aid] = True
    return act, frc, matched


__all__ = [
    "DEFAULT_PART_BODIES",
    "MOTOR_TO_MUJOCO_FRAME",
    "MUJOCO_TO_MOTOR_FRAME",
    "MotorJobConfig",
    "MotorPersonInput",
    "MotorTrial",
    "ResidualCheckSummary",
    "TouchWrenchSeries",
    "motor_activations_to_model",
    "motor_to_motion_clip",
    "motor_to_qpos",
    "motor_touches_to_qfrc",
    "read_motor_output",
    "read_motor_outputs",
    "unmangle_motor_joint_name",
    "write_motor_job",
]
