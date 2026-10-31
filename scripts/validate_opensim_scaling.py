# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the LICENSE file.
"""Compare MyoSuite's OpenSim scaling route with OpenSim's own ModelScaler.

MyoSuite scales MyoLeg with ``scale_model(spec, scaleset_path)``. OpenSim scales
the pinned Rajagopal model with its ModelScaler, in a separate interpreter
because OpenSim is not a MyoSuite dependency. The committed reference stores both
halves; the fast tests rerun the MyoSuite half against the stored OpenSim half.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, TYPE_CHECKING
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

if TYPE_CHECKING:
    import mujoco

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "myosuite/tests/data/opensim_scaling"
REFERENCE = DATA / "reference.json"
MODEL_URL = (
    "https://raw.githubusercontent.com/opensim-org/opensim-models/"
    "d9b05d470b1a481c222372c85b75772faf8f7792/Models/Rajagopal/Rajagopal2016.osim"
)
MODEL_SHA256 = "3f5c5f23e486073f2ad2aa4a4967ffe2fcdd582b1e355512bc54f70c36376bf4"
LICENSE = "MIT, Copyright (c) 2015 Stanford University; see LICENSE-Rajagopal.txt"
LICENSE_SOURCE = "https://simtk.org/projects/full_body (Full Body Model, package1738)"
COORDINATES = ("hip_flexion_r", "knee_angle_r", "ankle_angle_r")
MUSCLES = ("iliacus_r", "recfem_r", "vasmed_r", "soleus_r", "gasmed_r", "tibant_r")
BODIES = ("pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r")
SOLE = np.array([[-0.04, -0.03, 0.0], [0.16, -0.03, 0.04], [0.16, -0.03, -0.04]])
METRICS = (
    "joint_centres_m",
    "sole_points_m",
    "sole_floor_offset_m",
    "moment_arms_m",
    "normalized_fibre_length",
    "normalized_tendon_force",
    "torque_nm",
)
# Power k of one length factor s in each metric: an exact similarity gives s**k.
LENGTH_POWER = {
    "joint_centres_m": 1,
    "sole_points_m": 1,
    "sole_floor_offset_m": 1,
    "moment_arms_m": 1,
    "normalized_fibre_length": 0,
    "normalized_tendon_force": 0,
    "torque_nm": 1,
}
# Files whose content determines the MyoSuite half.
SOURCES = ("scripts/validate_opensim_scaling.py", "myosuite/core/body_scaling.py")
# Deliberate errors: a factor 5% off along one OpenSim axis of one segment, a
# swapped femur axis pair and a 3% muscle lengthrange error.
FACTOR_ERROR = 1.05
LENGTHRANGE_ERROR = 1.03
MUTATED_SEGMENTS = (
    "pelvis",
    "femur_r",
    "tibia_r",
    "patella_r",
    "talus_r",
    "calcn_r",
    "torso",
)
AXIS_MUTATIONS = tuple(f"{seg}_{axis}" for seg in MUTATED_SEGMENTS for axis in "xyz")
MUTATIONS = ("tibia_r_xyz", "femur_r_swap_xy", "lengthrange", *AXIS_MUTATIONS)


def fixture() -> dict[str, Any]:
    """Return the fixed synthetic subject factors and pose grid."""
    return json.loads((DATA / "fixture.json").read_text())


def _dumps(value: Any, indent: int = 0) -> str:
    """JSON with one line per array, so the reference stays small and diffable."""
    if isinstance(value, dict) and value:
        pad = " " * (indent + 2)
        items = [
            f"{pad}{json.dumps(k)}: {_dumps(v, indent + 2)}" for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(value) + "\n")


def _inertia_matrix(inertia: Any) -> list[list[float]]:
    moments = inertia.getMoments().to_numpy()
    xy, xz, yz = inertia.getProducts().to_numpy()
    return [
        [float(moments[0]), float(xy), float(xz)],
        [float(xy), float(moments[1]), float(yz)],
        [float(xz), float(yz), float(moments[2])],
    ]


def download_model(path: Path) -> Path:
    """Fetch the pinned public model, refusing changed or corrupted bytes."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != MODEL_SHA256:
            raise ValueError(
                "Downloaded Rajagopal model does not match the pinned SHA-256."
            )
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Rajagopal model does not match the pinned SHA-256.")
    return path


def _opensim(path: Path) -> dict[str, Any]:
    # Conda's Windows package keeps dependent DLLs outside the Python package.
    dll = Path(sys.prefix) / "Library/bin"
    dll_handle = (
        os.add_dll_directory(str(dll))
        if sys.platform == "win32" and dll.exists()
        else None
    )
    import opensim as osim

    osim.Logger.setLevelString("error")
    config = fixture()
    output = {
        "version": osim.GetVersionAndDate(),
        "python_version": platform.python_version(),
        "cases": {},
    }
    output["inertia_probe"] = {}
    for mode in ("volume", "keep"):
        body = osim.Body("probe", 10.0, osim.Vec3(0), osim.Inertia(0.1, 0.2, 0.25))
        body.scaleInertialProperties(osim.Vec3(1.2, 0.8, 1.1), mode == "volume")
        output["inertia_probe"][mode] = {
            "mass_kg": body.getMass(),
            "inertia_kgm2": _inertia_matrix(body.getInertia()),
        }
    for case, recipe in config["cases"].items():
        model = osim.Model(str(path))
        factors = recipe["factors"]
        if factors:
            scaler = osim.ModelScaler()
            order = osim.ArrayStr()
            order.append("manualScale")
            scaler.setScalingOrder(order)
            scaler.setPreserveMassDist(recipe["mass"] == "keep")
            for name, values in factors.items():
                scale = osim.Scale()
                scale.setSegmentName(name)
                scale.setScaleFactors(osim.Vec3(*values))
                scale.setApply(True)
                scaler.addScale(scale)
            if not scaler.processModel(model, "", -1.0):
                raise RuntimeError(f"OpenSim ModelScaler failed for {case}.")
        state = model.initSystem()
        pelvis = model.getBodySet().get("pelvis")
        muscles = [model.getMuscles().get(n) for n in MUSCLES]
        coords = [model.getCoordinateSet().get(n) for n in COORDINATES]
        rows = {name: [] for name in METRICS}
        for angles in config["pose_grid_degrees"]:
            for coord, angle in zip(coords, angles):
                coord.setValue(state, np.deg2rad(angle), False)
            model.assemble(state)
            for muscle in muscles:
                muscle.setActivation(state, 1.0)
            model.equilibrateMuscles(state)
            model.realizeDynamics(state)
            centres = [
                model.getBodySet()
                .get(n)
                .findStationLocationInAnotherFrame(state, osim.Vec3(0), pelvis)
                .to_numpy()
                .tolist()
                for n in BODIES[1:4]
            ]
            sole_factors = np.array(factors.get("calcn_r", [1.0, 1.0, 1.0]))
            points = [
                model.getBodySet()
                .get("calcn_r")
                .findStationLocationInAnotherFrame(
                    state, osim.Vec3(*(p * sole_factors)), pelvis
                )
                .to_numpy()
                .tolist()
                for p in SOLE
            ]
            arms = np.array(
                [[m.computeMomentArm(state, c) for c in coords] for m in muscles]
            )
            forces = np.array([m.getTendonForce(state) for m in muscles])
            values = {
                "joint_centres_m": centres,
                "sole_points_m": points,
                "sole_floor_offset_m": -float(np.min(np.array(points)[:, 1])),
                "moment_arms_m": arms.tolist(),
                "normalized_fibre_length": [
                    m.getNormalizedFiberLength(state) for m in muscles
                ],
                "normalized_tendon_force": [
                    f / m.getMaxIsometricForce() for f, m in zip(forces, muscles)
                ],
                "torque_nm": (forces @ arms).tolist(),
            }
            for name, value in values.items():
                rows[name].append(value)
        rows["body_mass_kg"] = [model.getBodySet().get(n).getMass() for n in BODIES]
        rows["body_inertia_kgm2"] = [
            _inertia_matrix(model.getBodySet().get(n).getInertia()) for n in BODIES
        ]
        rows["muscle_lengths_m"] = [
            [m.getOptimalFiberLength(), m.getTendonSlackLength()] for m in muscles
        ]
        wrap = osim.WrapCylinder.safeDownCast(
            pelvis.getWrapObjectSet().get("Gmax1_at_pelvis_r")
        )
        rows["pelvis_wrap_dimensions_m"] = [wrap.get_radius(), wrap.get_length()]
        wrap = osim.WrapCylinder.safeDownCast(
            pelvis.getWrapObjectSet().get("PS_at_brim_r")
        )
        rows["psoas_wrap_dimensions_m"] = [wrap.get_radius(), wrap.get_length()]
        rows["total_mass_kg"] = model.getTotalMass(state)
        fibre = np.array(rows["normalized_fibre_length"])
        rows["normalized_fibre_operating_ranges"] = np.column_stack(
            [fibre.min(0), fibre.max(0)]
        ).tolist()
        output["cases"][case] = rows
    del dll_handle
    return output


def uniform_factor(recipe: Mapping[str, Any]) -> float | None:
    """The one factor of a recipe that scales every segment alike, else None."""
    values = {float(v) for factors in recipe["factors"].values() for v in factors}
    return values.pop() if len(values) == 1 else None


def compared_cases() -> list[str]:
    """Fixture cases that ``scale_model`` reproduces, nominal first.

    ``scale_model`` scales mass with volume and ``total_mass`` rescales the
    whole subject, so OpenSim's preserved mass distribution has a counterpart
    only for one uniform factor: rescaling to the nominal mass keeps every body's
    mass. The anisotropic preserved-mass OpenSim case stays in the reference.
    """
    return [
        name
        for name, recipe in fixture()["cases"].items()
        if recipe["mass"] == "volume" or uniform_factor(recipe) is not None
    ]


def nominal_spec() -> mujoco.MjSpec:
    """The registered myoLegWalk-v0 model, built as its env builds it."""
    import myosuite  # noqa: F401  (registers the envs)
    from myosuite.core.model_builder import ModelBuilder
    from myosuite.utils import gym

    path = gym.spec("myoLegWalk-v0").kwargs["model_path"]
    return ModelBuilder.from_xml_file(path).build_spec()


def write_scaleset(factors: Mapping[str, Sequence[float]], path: Path) -> Path:
    """Write segment factors as an OpenSim ScaleSet file (the Scale Tool format)."""
    document = ET.Element("OpenSimDocument", Version="40000")
    scale_set = ET.SubElement(document, "ScaleSet", name="opensim_scaling_fixture")
    objects = ET.SubElement(scale_set, "objects")
    for segment, values in factors.items():
        scale = ET.SubElement(objects, "Scale", name=segment)
        # repr keeps every factor exact.
        ET.SubElement(scale, "scales").text = " ".join(repr(float(v)) for v in values)
        ET.SubElement(scale, "segment").text = segment
        ET.SubElement(scale, "apply").text = "true"
    ET.ElementTree(document).write(path, encoding="utf-8", xml_declaration=True)
    return path


def mutated_factors(
    factors: Mapping[str, Sequence[float]], mutation: str | None
) -> dict[str, list[float]]:
    """Copy of *factors* with the deliberate factor error *mutation* applied."""
    out = {segment: [float(v) for v in values] for segment, values in factors.items()}
    if mutation in (None, "lengthrange"):
        return out
    if mutation == "femur_r_swap_xy":
        out["femur_r"][0], out["femur_r"][1] = out["femur_r"][1], out["femur_r"][0]
        return out
    segment, axis = mutation.rsplit("_", 1)
    values = out.setdefault(segment, [1.0, 1.0, 1.0])
    for index in range(3) if axis == "xyz" else ("xyz".index(axis),):
        values[index] *= FACTOR_ERROR
    return out


def build_myo_case(
    recipe: Mapping[str, Any], mutation: str | None = None
) -> mujoco.MjModel:
    """Scale MyoLeg with ``scale_model`` from a ScaleSet file of the recipe."""
    from myosuite.core.body_scaling import scale_model

    if mutation is not None and mutation not in MUTATIONS:
        raise ValueError(f"Unknown mutation {mutation!r}.")
    spec = nominal_spec()
    total_mass = None
    if recipe["mass"] == "keep":
        if uniform_factor(recipe) is None:
            raise ValueError("scale_model keeps no per-body mass distribution.")
        nominal = spec.copy().compile()  # compiling *spec* would freeze its frames
        subject = nominal.body_rootid == nominal.body("root").id
        total_mass = float(nominal.body_mass[subject].sum())
    with tempfile.TemporaryDirectory() as folder:
        path = write_scaleset(
            mutated_factors(recipe["factors"], mutation), Path(folder) / "scale.xml"
        )
        model = scale_model(spec, path, total_mass=total_mass)
    if mutation == "lengthrange":
        ids = [model.actuator(name).id for name in MUSCLES]
        model.actuator_lengthrange[ids] *= LENGTHRANGE_ERROR
    return model


def _myo_pose(model: mujoco.MjModel, angles: np.ndarray) -> mujoco.MjData:
    import mujoco

    data = mujoco.MjData(model)
    for name, angle in zip(COORDINATES, angles):
        data.qpos[model.joint(name).qposadr[0]] = angle
    # Apply the same fitted polynomial knee constraints used by the model.
    for _ in range(model.neq):
        for index in range(model.neq):
            if model.eq_type[index] == int(mujoco.mjtEq.mjEQ_JOINT):
                first, second = model.eq_obj1id[index], model.eq_obj2id[index]
                if second >= 0:
                    x = (
                        data.qpos[model.jnt_qposadr[second]]
                        - model.qpos0[model.jnt_qposadr[second]]
                    )
                    data.qpos[model.jnt_qposadr[first]] = model.qpos0[
                        model.jnt_qposadr[first]
                    ] + np.polynomial.polynomial.polyval(x, model.eq_data[index, :5])
    data.act[:] = 1.0
    mujoco.mj_forward(model, data)
    return data


def _inertia_in(model: mujoco.MjModel, body: int, frame: np.ndarray) -> np.ndarray:
    """Inertia of *body* about its centre of mass, in the world-aligned *frame*."""
    import mujoco

    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[body])
    rot = frame @ rot.reshape(3, 3)
    return rot @ np.diag(model.body_inertia[body]) @ rot.T


def sample_myo_case(
    model: mujoco.MjModel,
    calcn_factors: Sequence[float],
    *,
    moment_arm_step: float = 1e-5,
) -> dict[str, Any]:
    """Measure geometry, constrained moment arms and isometric muscle curves.

    Args:
        model: Scaled MyoLeg.
        calcn_factors: Fixture factors of ``calcn_r``, which place the calcaneus
            landmarks in that body's frame (the same arithmetic as the OpenSim
            side, so the landmarks measure the calcaneus pose only).
        moment_arm_step: Central finite-difference step (rad).

    Returns:
        Metric name -> sampled values over the pose grid, plus static records.
    """
    pelvis = model.body("pelvis").id
    muscle_ids = np.array([model.actuator(n).id for n in MUSCLES])
    rows: dict[str, Any] = {name: [] for name in METRICS}
    for degrees in fixture()["pose_grid_degrees"]:
        angles = np.deg2rad(degrees)
        data = _myo_pose(model, angles)
        rot = data.xmat[pelvis].reshape(3, 3)
        centres = (
            data.xpos[[model.body(n).id for n in BODIES[1:4]]] - data.xpos[pelvis]
        ) @ rot
        calcn = model.body("calcn_r").id
        points = (
            (SOLE * np.asarray(calcn_factors)) @ data.xmat[calcn].reshape(3, 3).T
            + data.xpos[calcn]
            - data.xpos[pelvis]
        ) @ rot
        arms = []
        for index in range(len(angles)):
            perturb = np.eye(3)[index] * moment_arm_step
            arms.append(
                -(
                    _myo_pose(model, angles + perturb).actuator_length[muscle_ids]
                    - _myo_pose(model, angles - perturb).actuator_length[muscle_ids]
                )
                / (2 * moment_arm_step)
            )
        arms = np.array(arms).T
        force = -data.actuator_force[muscle_ids]
        prm = model.actuator_gainprm[muscle_ids]
        lengthrange = model.actuator_lengthrange[muscle_ids]
        optimal = np.diff(lengthrange, axis=1)[:, 0] / (prm[:, 1] - prm[:, 0])
        slack = lengthrange[:, 0] - prm[:, 0] * optimal
        values = {
            "joint_centres_m": centres.tolist(),
            "sole_points_m": points.tolist(),
            "sole_floor_offset_m": -float(points[:, 1].min()),
            "moment_arms_m": arms.tolist(),
            "normalized_fibre_length": (
                (data.actuator_length[muscle_ids] - slack) / optimal
            ).tolist(),
            "normalized_tendon_force": (force / prm[:, 2]).tolist(),
            "torque_nm": (force @ arms).tolist(),
        }
        for name, value in values.items():
            rows[name].append(value)
    ids = [model.body(n).id for n in BODIES]
    rows["body_mass_kg"] = model.body_mass[ids].tolist()
    # In each body's own frame, like OpenSim's Body inertia.
    rows["body_inertia_kgm2"] = [
        _inertia_in(model, body, np.eye(3)).tolist() for body in ids
    ]
    rows["muscle_lengths_m"] = np.column_stack([optimal, slack]).tolist()
    wrap = model.geom("Gmax1_at_pelvis_wrap_r")
    rows["pelvis_wrap_dimensions_m"] = [float(wrap.size[0]), float(2 * wrap.size[1])]
    wrap = model.geom("PS_at_brim_wrap_r")
    rows["psoas_wrap_dimensions_m"] = [float(wrap.size[0]), float(2 * wrap.size[1])]
    rows["total_mass_kg"] = float(model.body_mass.sum())
    fibre = np.array(rows["normalized_fibre_length"])
    rows["normalized_fibre_operating_ranges"] = np.column_stack(
        [fibre.min(0), fibre.max(0)]
    ).tolist()
    return {**rows, **subject_records(model)}


def subject_records(model: mujoco.MjModel) -> dict[str, Any]:
    """MyoSuite-only records, which the OpenSim reference has no counterpart for.

    The root height of ``qpos0`` and of each keyframe comes from
    ``scale_model``'s alignment of the lowest collision geom (the feet). The
    torso and the root's head site are given in the pelvis frame (x anterior,
    y up, z right) of the neutral pose.
    """
    pelvis = model.body("pelvis").id
    neutral = _myo_pose(model, np.zeros(len(COORDINATES)))
    frame = neutral.xmat[pelvis].reshape(3, 3).T
    root = model.body("root").id
    free = model.jnt_qposadr[model.body_jntadr[root]]
    torso = model.body("torso").id
    geoms = np.flatnonzero(model.geom_bodyid == torso)
    head = model.site("head").id
    return {
        "root_heights_m": [float(model.qpos0[free + 2])]
        + model.key_qpos[:, free + 2].tolist(),
        "torso_mass_kg": float(model.body_mass[torso]),
        "torso_inertia_kgm2": _inertia_in(
            model, torso, frame @ neutral.xmat[torso].reshape(3, 3)
        ).tolist(),
        "torso_geom_offsets_m": (
            (neutral.geom_xpos[geoms] - neutral.xpos[torso]) @ frame.T
        ).tolist(),
        "head_site_m": (
            (neutral.site_xpos[head] - neutral.xpos[pelvis]) @ frame.T
        ).tolist(),
    }


def source_hashes() -> dict[str, str]:
    """SHA-256 of each measuring source with LF line endings, as git stores it.

    Working-tree bytes would differ between a Windows (CRLF) and a Linux
    checkout of the same commit.
    """
    return {
        name: hashlib.sha256(
            (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for name in SOURCES
    }


def _provenance() -> dict[str, Any]:
    """Platform and source hashes of a run."""
    return {"platform": platform.platform(), "code_sha256_lf": source_hashes()}


def sample_myo() -> dict[str, Any]:
    """Run the compared recipes using the installed MyoSuite/MuJoCo model assets."""
    import mujoco
    from myosuite.core.body_scaling import scale_model
    from myosuite.version import __version__

    probe = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="probe"><freejoint/><inertial mass="10" '
        'pos="0 0 0" diaginertia=".1 .2 .25"/></body></worldbody></mujoco>'
    )
    inertia_probe = {}
    # Rescaling to the nominal 10 kg keeps the mass, like OpenSim's "keep".
    for mode, total_mass in (("volume", None), ("keep", 10.0)):
        model = scale_model(probe, {"probe": [1.2, 0.8, 1.1]}, total_mass=total_mass)
        inertia_probe[mode] = {
            "mass_kg": float(model.body_mass[1]),
            "inertia_kgm2": _inertia_in(model, 1, np.eye(3)).tolist(),
        }
    cases = fixture()["cases"]
    return {
        "provenance": {
            **_provenance(),
            "python_version": platform.python_version(),
            "mujoco_version": mujoco.__version__,
            "myo_sim_version": importlib.metadata.version("myo-sim"),
            "myosuite_version": __version__,
        },
        "inertia_probe": inertia_probe,
        "cases": {
            name: sample_myo_case(
                build_myo_case(cases[name]),
                cases[name]["factors"].get("calcn_r", [1.0, 1.0, 1.0]),
            )
            for name in compared_cases()
        },
    }


def discrepancy(
    myo: Mapping[str, Any], opensim: Mapping[str, Any], case: str, metric: str
) -> np.ndarray:
    """Scaling discrepancy of *metric* in *case*, nominal conversion error removed.

    For one factor ``s`` it is ``(m_s - o_s) - s**k (m_n - o_n)``, zero when both
    pipelines are exact similarities (``k`` from :data:`LENGTH_POWER`). For
    several factors it is ``(m_s - m_n) - (o_s - o_n)``. ``m``/``o`` are the
    MyoSuite/OpenSim values, ``s``/``n`` the scaled and nominal cases.
    """
    m_s, o_s = (np.array(r["cases"][case][metric]) for r in (myo, opensim))
    m_n, o_n = (np.array(r["cases"]["nominal"][metric]) for r in (myo, opensim))
    s = uniform_factor(fixture()["cases"][case])
    weight = 1.0 if s is None else s ** LENGTH_POWER[metric]
    return (m_s - o_s) - weight * (m_n - o_n)


def category(case: str) -> str:
    """Tolerance category of a fixture case: ``uniform`` or ``anisotropic``."""
    if uniform_factor(fixture()["cases"][case]) is None:
        return "anisotropic"
    return "uniform"


def compare(myo: Mapping[str, Any], opensim: Mapping[str, Any]) -> dict[str, Any]:
    """Absolute differences and scaling discrepancies of every compared case."""
    output: dict[str, Any] = {}
    for case in compared_cases():
        output[case] = {}
        for metric in METRICS:
            difference = np.array(myo["cases"][case][metric]) - np.array(
                opensim["cases"][case][metric]
            )
            output[case][metric] = {
                "absolute_max": float(np.abs(difference).max()),
                "absolute_rms": float(np.sqrt(np.mean(difference**2))),
                "scaling_discrepancy_max": float(
                    np.abs(discrepancy(myo, opensim, case, metric)).max()
                ),
            }
        for metric in ("body_mass_kg", "body_inertia_kgm2"):
            ratios = []
            for runtime in (myo, opensim):
                scaled = np.array(runtime["cases"][case][metric])
                nominal = np.array(runtime["cases"]["nominal"][metric])
                if metric == "body_inertia_kgm2":
                    scaled = np.diagonal(scaled, axis1=1, axis2=2)
                    nominal = np.diagonal(nominal, axis1=1, axis2=2)
                ratios.append(scaled / nominal)
            output[case][metric] = {
                "scaling_ratio_difference_max": float(
                    np.abs(ratios[0] - ratios[1]).max()
                )
            }
        for metric in ("pelvis_wrap_dimensions_m", "psoas_wrap_dimensions_m"):
            difference = np.array(myo["cases"][case][metric]) - np.array(
                opensim["cases"][case][metric]
            )
            nominal = np.array(myo["cases"]["nominal"][metric]) - np.array(
                opensim["cases"]["nominal"][metric]
            )
            output[case][metric] = {
                "absolute_max": float(np.abs(difference).max()),
                "scaling_increment_max": float(np.abs(difference - nominal).max()),
            }
    return output


def assessment(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Check the measured discrepancies against the documented bounds."""
    limits = json.loads((DATA / "tolerances.json").read_text())
    violations = []
    for case, values in comparison.items():
        if case == "nominal":
            continue
        for metric, bound in limits[category(case)].items():
            measured = values[metric]["scaling_discrepancy_max"]
            if measured >= bound:
                violations.append(
                    {
                        "case": case,
                        "metric": metric,
                        "measured": measured,
                        "bound": bound,
                    }
                )
    return {
        "within_bounds": not violations,
        "violations": violations,
        "scope": "Fixed synthetic pose grid; nominal equivalence, subject validity "
        "and clinical accuracy are not established.",
    }


def main(argv: list[str] | None = None) -> None:
    """Regenerate the reference: both halves with OpenSim, or the MyoSuite half."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--opensim-python",
        type=Path,
        help="Python of an environment with OpenSim 4.6: regenerate both halves.",
    )
    mode.add_argument(
        "--refresh-myosuite",
        action="store_true",
        help="Keep the committed OpenSim half and regenerate the MyoSuite half.",
    )
    mode.add_argument("--opensim-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "tasks/opensim/Rajagopal2016.osim"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tasks/opensim/reference.json"
    )
    args = parser.parse_args(argv)
    if args.opensim_worker:
        _write(args.output, _opensim(download_model(args.model)))
        return
    if args.refresh_myosuite:
        committed = json.loads(REFERENCE.read_text())
        if committed["fixture"] != fixture():
            raise SystemExit("fixture.json changed: regenerate the OpenSim half too.")
        opensim = committed["opensim"]
    else:
        path = download_model(args.model)
        worker = ROOT / "tasks/opensim/opensim-result.json"
        subprocess.run(
            [
                str(args.opensim_python),
                str(Path(__file__).resolve()),
                "--opensim-worker",
                "--model",
                str(path.resolve()),
                "--output",
                str(worker),
            ],
            check=True,
        )
        opensim = {
            "provenance": {
                **_provenance(),
                "model_url": MODEL_URL,
                "model_sha256": MODEL_SHA256,
                "license": LICENSE,
                "license_source": LICENSE_SOURCE,
            },
            **json.loads(worker.read_text()),
        }
    reference = {
        "schema": 2,
        "fixture": fixture(),
        "opensim": opensim,
        "myosuite": sample_myo(),
    }
    _write(args.output, reference)
    comparison = compare(reference["myosuite"], reference["opensim"])
    result = assessment(comparison)
    report = ROOT / "tasks/opensim/comparison.json"
    _write(report, {**comparison, "assessment": result})
    print(f"Wrote {args.output} and {report}")
    if not result["within_bounds"]:
        raise SystemExit(
            "Measured scaling discrepancies exceeded the documented bounds."
        )


if __name__ == "__main__":
    main()
