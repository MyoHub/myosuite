# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the LICENSE file.
"""Reproduce the Rajagopal/OpenSim scaling comparison using two Python runtimes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, TYPE_CHECKING
import urllib.request

import numpy as np

if TYPE_CHECKING:
    import mujoco
    from myosuite.core.subject_model import SubjectModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "myosuite/tests/data/opensim_scaling"
MODEL_URL = (
    "https://raw.githubusercontent.com/opensim-org/opensim-models/"
    "d9b05d470b1a481c222372c85b75772faf8f7792/Models/Rajagopal/Rajagopal2016.osim"
)
MODEL_SHA256 = "3f5c5f23e486073f2ad2aa4a4967ffe2fcdd582b1e355512bc54f70c36376bf4"
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


def fixture() -> dict[str, Any]:
    """Return the fixed synthetic subject factors and pose grid."""
    return json.loads((DATA / "fixture.json").read_text())


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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


def build_myo_case(recipe: dict[str, Any], mutation: str | None = None) -> SubjectModel:
    """Build the registered MyoLeg source through the shared subject builder."""
    import mujoco
    import myosuite  # noqa: F401
    from myosuite.core.body_scaling import RAJAGOPAL_MYOLEG_SEGMENTS
    from myosuite.core.subject_model import (
        ScaleEstimate,
        SubjectModelConfig,
        build_subject_model,
    )
    from myosuite.utils import gym
    from myosuite.utils.asset_path_resolver import resolve_model_xml_path

    path = resolve_model_xml_path(gym.spec("myoLegWalk-v0").kwargs["model_path"])
    spec = mujoco.MjSpec.from_file(str(path))
    factors = {k: list(v) for k, v in recipe["factors"].items()}
    if factors and mutation == "wrong_factor":
        factors["tibia_r"] = (np.array(factors["tibia_r"]) * 1.05).tolist()
    if factors and mutation == "wrong_axis":
        factors["femur_r"][0], factors["femur_r"][1] = (
            factors["femur_r"][1],
            factors["femur_r"][0],
        )
    subject = build_subject_model(
        spec,
        SubjectModelConfig(
            ScaleEstimate(factors, source="opensim_scaling/fixture.json"),
            segments=RAJAGOPAL_MYOLEG_SEGMENTS,
            mass=recipe["mass"],
        ),
    )
    if mutation == "wrong_lengthrange":
        for name in MUSCLES:
            subject.model.actuator(name).lengthrange[:] *= 1.03
    return subject


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


def sample_myo_case(
    subject: SubjectModel, *, moment_arm_step: float = 1e-5
) -> dict[str, Any]:
    """Measure geometry, constrained moment arms and isometric muscle curves."""
    import mujoco
    from myosuite.core.subject_model import GroundAlignment, align_root_to_ground

    model = subject.model
    pelvis = model.body("pelvis").id
    muscle_ids = np.array([model.actuator(n).id for n in MUSCLES])
    rows = {name: [] for name in METRICS}
    for degrees in fixture()["pose_grid_degrees"]:
        angles = np.deg2rad(degrees)
        data = _myo_pose(model, angles)
        rot = data.xmat[pelvis].reshape(3, 3)
        centres = (
            data.xpos[[model.body(n).id for n in BODIES[1:4]]] - data.xpos[pelvis]
        ) @ rot
        calcn = model.body("calcn_r").id
        factors = np.array(subject.body_scales.get("calcn_r", [1.0, 1.0, 1.0]))
        points = (
            (SOLE * factors) @ data.xmat[calcn].reshape(3, 3).T
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
    inertia = []
    for body in ids:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.body_iquat[body])
        rot = rot.reshape(3, 3)
        inertia.append((rot @ np.diag(model.body_inertia[body]) @ rot.T).tolist())
    rows["body_mass_kg"] = model.body_mass[ids].tolist()
    rows["body_inertia_kgm2"] = inertia
    rows["muscle_lengths_m"] = np.column_stack([optimal, slack]).tolist()
    wrap = model.geom("Gmax1_at_pelvis_wrap_r")
    rows["pelvis_wrap_dimensions_m"] = [float(wrap.size[0]), float(2 * wrap.size[1])]
    wrap = model.geom("PS_at_brim_wrap_r")
    rows["psoas_wrap_dimensions_m"] = [float(wrap.size[0]), float(2 * wrap.size[1])]
    rows["total_mass_kg"] = float(model.body_mass.sum())
    _, rows["collision_root_z_offset_m"] = align_root_to_ground(
        model,
        GroundAlignment(
            "root",
            (
                "foot_col1_r",
                "foot_col3_r",
                "foot_col4_r",
                "bofoot_col1_r",
                "bofoot_col2_r",
            ),
        ),
    )
    rows["model_digest"] = subject.metadata["model_digest"]
    fibre = np.array(rows["normalized_fibre_length"])
    rows["normalized_fibre_operating_ranges"] = np.column_stack(
        [fibre.min(0), fibre.max(0)]
    ).tolist()
    return rows


def sample_myo(mutation: str | None = None) -> dict[str, Any]:
    """Run all recipes using the installed MyoSuite/MuJoCo model assets."""
    import mujoco
    from myosuite.core.body_scaling import scale_bodies
    from myosuite.version import __version__

    inertia_probe = {}
    for mode in ("volume", "keep"):
        spec = mujoco.MjSpec.from_string(
            '<mujoco><worldbody><body name="probe"><freejoint/><inertial mass="10" pos="0 0 0" diaginertia=".1 .2 .25"/></body></worldbody></mujoco>'
        )
        model = scale_bodies(spec, {"probe": [1.2, 0.8, 1.1]}, mass=mode).compile()
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.body_iquat[1])
        rotation = rotation.reshape(3, 3)
        inertia_probe[mode] = {
            "mass_kg": float(model.body_mass[1]),
            "inertia_kgm2": (
                rotation @ np.diag(model.body_inertia[1]) @ rotation.T
            ).tolist(),
        }
    return {
        "mujoco_version": mujoco.__version__,
        "myo_sim_version": importlib.metadata.version("myo-sim"),
        "myosuite_version": __version__,
        "python_version": platform.python_version(),
        "inertia_probe": inertia_probe,
        "cases": {
            name: sample_myo_case(build_myo_case(recipe, mutation))
            for name, recipe in fixture()["cases"].items()
        },
    }


def compare(myo: dict[str, Any], opensim: dict[str, Any]) -> dict[str, Any]:
    """Separate nominal conversion error from the additional scaling difference."""
    output = {}
    for case in fixture()["cases"]:
        output[case] = {}
        for metric in METRICS:
            m, o = (
                np.array(myo["cases"][case][metric]),
                np.array(opensim["cases"][case][metric]),
            )
            baseline = np.array(myo["cases"]["nominal"][metric]) - np.array(
                opensim["cases"]["nominal"][metric]
            )
            difference = m - o
            output[case][metric] = {
                "absolute_max": float(np.abs(difference).max()),
                "absolute_rms": float(np.sqrt(np.mean(difference**2))),
                "scaling_increment_max": float(np.abs(difference - baseline).max()),
            }
        for metric in ("body_mass_kg", "body_inertia_kgm2"):
            values = []
            for runtime in (myo, opensim):
                scaled = np.array(runtime["cases"][case][metric])
                nominal = np.array(runtime["cases"]["nominal"][metric])
                if metric == "body_inertia_kgm2":
                    scaled, nominal = (
                        np.diagonal(scaled, axis1=1, axis2=2),
                        np.diagonal(nominal, axis1=1, axis2=2),
                    )
                values.append(scaled / nominal)
            output[case][metric] = {
                "scaling_ratio_difference_max": float(
                    np.abs(values[0] - values[1]).max()
                )
            }
        for metric in ("pelvis_wrap_dimensions_m", "psoas_wrap_dimensions_m"):
            difference = np.array(myo["cases"][case][metric]) - np.array(
                opensim["cases"][case][metric]
            )
            baseline = np.array(myo["cases"]["nominal"][metric]) - np.array(
                opensim["cases"]["nominal"][metric]
            )
            output[case][metric] = {
                "absolute_max": float(np.abs(difference).max()),
                "scaling_increment_max": float(np.abs(difference - baseline).max()),
            }
    return output


def assessment(comparison: dict[str, Any]) -> dict[str, Any]:
    """Check measured increment bounds without claiming nominal equivalence."""
    limits = json.loads((DATA / "tolerances.json").read_text())
    violations = []
    for case, values in comparison.items():
        if case == "nominal":
            continue
        category = "uniform" if case.startswith("uniform") else "anisotropic"
        for metric, bound in limits[category].items():
            measured = values[metric]["scaling_increment_max"]
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
        "within_measured_increment_bounds": not violations,
        "violations": violations,
        "scope": "Fixed synthetic pose grid; nominal equivalence, subject validity and clinical accuracy are not established.",
    }


def main(argv: list[str] | None = None) -> None:
    """Run the comparison or its isolated OpenSim subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opensim-python", type=Path)
    parser.add_argument("--opensim-worker", action="store_true")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "tasks/opensim/Rajagopal2016.osim"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tasks/opensim/report.json"
    )
    args = parser.parse_args(argv)
    path = download_model(args.model)
    if args.opensim_worker:
        _write(args.output, _opensim(path))
        return
    if args.opensim_python is None:
        parser.error("--opensim-python must point to a Python with OpenSim installed.")
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
    opensim = json.loads(worker.read_text())
    myo = sample_myo()
    comparison = compare(myo, opensim)
    result = assessment(comparison)
    _write(
        args.output,
        {
            "schema": 1,
            "platform": platform.platform(),
            "code_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "scripts/validate_opensim_scaling.py",
                    "myosuite/core/body_scaling.py",
                    "myosuite/core/subject_model.py",
                )
            },
            "model_url": MODEL_URL,
            "model_sha256": MODEL_SHA256,
            "license": "MIT, Copyright (c) 2015 Stanford University; see LICENSE-Rajagopal.txt",
            "license_source": "https://simtk.org/projects/full_body (Full Body Model, package1738)",
            "fixture": fixture(),
            "opensim": opensim,
            "myosuite": myo,
            "comparison": comparison,
            "assessment": result,
        },
    )
    print(f"Wrote {args.output}")
    if not result["within_measured_increment_bounds"]:
        raise SystemExit(
            "Measured scaling differences exceeded the documented fixed-grid bounds."
        )


if __name__ == "__main__":
    main()
