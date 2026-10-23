#!/usr/bin/env python3
"""Compare authored MyoSuite models with Newton CPU/GPU trajectories.

This diagnostic requires Newton with MJCF DoF-label preservation. It does not
register a MyoSuite backend. Failures and numerical drift remain report results.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from scripts.newton_compat_probe import (
    DEFAULT_MODELS,
    _PACKAGES,
    _captured,
    _source,
    _version,
    authored_xml,
    converted_actuator_names,
)


def _joint_mapping(
    reference: Any, model: Any, solver: Any
) -> tuple[np.ndarray, dict[int, int]]:
    """Map converted MuJoCo positions to source positions through Newton DoFs."""
    converted = solver.mj_model
    labels = model.mujoco.joint_dof_label
    dofs = solver.mjc_jnt_to_newton_dof.numpy()[0]
    joints = solver.mjc_jnt_to_newton_jnt.numpy()[0]
    source = {reference.joint(row).name: row for row in range(reference.njnt)}
    mapping = np.full(converted.nq, -1, dtype=int)
    source_dofs = {}
    for row in range(converted.njnt):
        dof = int(dofs[row])
        label = (
            labels[dof]
            if dof >= 0 and labels[dof]
            else model.joint_label[int(joints[row])]
        )
        name = label.rsplit("/", 1)[-1]
        original = source[name]
        if reference.jnt_type[original] != converted.jnt_type[row]:
            raise ValueError(f"Joint type changed for {name}")
        size = {
            int(mujoco.mjtJoint.mjJNT_FREE): 7,
            int(mujoco.mjtJoint.mjJNT_BALL): 4,
        }.get(int(converted.jnt_type[row]), 1)
        start, target = converted.jnt_qposadr[row], reference.jnt_qposadr[original]
        mapping[start : start + size] = np.arange(target, target + size)
        if size == 1:
            source_dofs[original] = dof
    if len(mapping) != reference.nq or set(mapping) != set(range(reference.nq)):
        raise ValueError(
            "Source and converted joint positions do not form a complete correspondence"
        )
    return mapping, source_dofs


def _identity_mapping(
    reference: Any, model: Any, solver: Any, qpos_mapping: np.ndarray
) -> dict[str, Any]:
    """Record original identities despite Newton's hierarchical export names."""
    converted = solver.mj_model
    groups: dict[str, Any] = {}
    for kind, labels, indices in (
        ("body", model.body_label, solver.mjc_body_to_newton.numpy()[0]),
        ("site", model.shape_label, solver._mjc_site_to_newton_shape.numpy()[0]),
    ):
        entries = []
        for row, index in enumerate(indices):
            name = labels[int(index)].rsplit("/", 1)[-1] if index >= 0 else "world"
            source = getattr(reference, kind)(name).id
            entries.append(
                {
                    "source_index": source,
                    "source": name,
                    "converted_index": row,
                    "converted": getattr(converted, kind)(row).name,
                }
            )
        groups[kind] = entries
    groups["joint"] = []
    for row in range(converted.njnt):
        position = qpos_mapping[converted.jnt_qposadr[row]]
        source = int(np.searchsorted(reference.jnt_qposadr, position, side="right") - 1)
        groups["joint"].append(
            {
                "source_index": source,
                "source": reference.joint(source).name,
                "converted_index": row,
                "converted": converted.joint(row).name,
            }
        )
    for kind, entries in groups.items():
        sources = [entry["source_index"] for entry in entries]
        if len(sources) != getattr(
            reference, {"body": "nbody", "site": "nsite", "joint": "njnt"}[kind]
        ) or len(set(sources)) != len(sources):
            raise ValueError(
                f"The {kind} identity mapping is not complete and one-to-one"
            )
    return groups


def run_case(
    model_id: str, device: str, engine: str, poses: int, steps: int, workdir: Path
) -> dict[str, Any]:
    """Measure matched poses and trajectories, including a force mutation.

    Args:
        model_id: Registered env ID or ``musclemimic:<name>``.
        device: Warp device for the converted model.
        engine: ``cpu`` for native MuJoCo, or ``warp`` for MuJoCo-Warp.
        poses: Number of matched poses, including the default pose.
        steps: Number of free-running trajectory steps per pose.
        workdir: Directory for materialized model inputs.

    Returns:
        Identity maps, numerical errors and pass/fail results, or the exception
        that prevented the case from completing. Newton warnings are retained.
    """
    import newton
    import warp as wp

    result: dict[str, Any] = {"model": model_id, "device": device, "engine": engine}
    notes: dict[str, list[Any]] = {}
    try:
        reference = mujoco.MjModel.from_xml_path(str(authored_xml(model_id, workdir)))
        with _captured(notes):
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(authored_xml(model_id, workdir)), ctrl_direct=True)
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverMuJoCo(
                model,
                use_mujoco_cpu=engine == "cpu",
                use_mujoco_contacts=True,
                nconmax=2048,
                njmax=8192,
                strict_capacity=True,
            )
        mapping, source_dofs = _joint_mapping(reference, model, solver)
        result["identity_mapping"] = _identity_mapping(
            reference, model, solver, mapping
        )
        names = converted_actuator_names(solver, model)
        actuator_rows = [reference.actuator(name).id for name in names]
        qd_starts, q_starts = model.joint_qd_start.numpy(), model.joint_q_start.numpy()
        initial = model.joint_q.numpy().copy()
        outcomes = []
        for pose in range(poses):
            source_qpos = reference.qpos0.copy()
            joint_q = initial.copy()
            for joint, dof in source_dofs.items():
                width = (
                    float(np.ptp(reference.jnt_range[joint]))
                    if reference.jnt_limited[joint]
                    else 1.0
                )
                offset = min(0.01, width * 0.005) * np.sin(joint + 0.3) if pose else 0.0
                source_qpos[reference.jnt_qposadr[joint]] += offset
                owner = int(np.searchsorted(qd_starts, dof, side="right") - 1)
                joint_q[q_starts[owner] + dof - qd_starts[owner]] += offset
            model.joint_q.assign(joint_q)
            with _captured(notes):
                solver = newton.solvers.SolverMuJoCo(
                    model,
                    use_mujoco_cpu=engine == "cpu",
                    use_mujoco_contacts=True,
                    nconmax=2048,
                    njmax=8192,
                    strict_capacity=True,
                )
            baseline = mujoco.MjData(reference)
            baseline.qpos[:] = source_qpos
            baseline.act[:] = 0.5
            baseline.ctrl[:] = 0.5
            mujoco.mj_forward(reference, baseline)
            converted_data = mujoco.MjData(solver.mj_model)
            converted_data.qpos[:] = source_qpos[mapping]
            converted_data.act[:] = converted_data.ctrl[:] = 0.5
            mujoco.mj_forward(solver.mj_model, converted_data)
            expected = baseline.actuator_force[actuator_rows].copy()
            actual = converted_data.actuator_force.copy()
            if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
                raise ValueError(f"Non-finite static muscle force at pose {pose}")
            force_close = np.isclose(actual, expected, rtol=1e-3, atol=0.05)
            gain, bias = (
                reference.actuator_gainprm.copy(),
                reference.actuator_biasprm.copy(),
            )
            for row in range(reference.nu):
                slot = 2 if gain[row, 2] > 0 else 3
                reference.actuator_gainprm[row, slot] *= 1.1
                reference.actuator_biasprm[row, slot] *= 1.1
            mutant = mujoco.MjData(reference)
            mutant.qpos[:] = source_qpos
            mutant.act[:] = mutant.ctrl[:] = 0.5
            mujoco.mj_forward(reference, mutant)
            mutation_detected = not np.allclose(
                mutant.actuator_force[actuator_rows], expected, rtol=1e-3, atol=0.05
            )
            reference.actuator_gainprm[:] = gain
            reference.actuator_biasprm[:] = bias
            baseline = mujoco.MjData(reference)
            baseline.qpos[:] = source_qpos
            state, next_state = model.state(), model.state()
            control = model.control()
            max_position_error = 0.0
            max_force_error = 0.0
            trajectory_close = True
            largest_joint = ""
            largest_force = ""
            first_step_error = None
            first_failure = None
            max_contacts = 0
            for step in range(steps):
                command = 0.15 + 0.1 * np.sin(
                    np.arange(reference.nu) * 0.3 + step * 0.1
                )
                baseline.ctrl[:] = command
                control.mujoco.ctrl.assign(command[actuator_rows].astype(np.float32))
                mujoco.mj_step(reference, baseline)
                solver.step(
                    state, next_state, control, None, float(reference.opt.timestep)
                )
                state, next_state = next_state, state
                wp.synchronize_device(device)
                qpos = (
                    solver.mj_data.qpos
                    if engine == "cpu"
                    else solver.mjw_data.qpos.numpy()[0]
                )
                forces = (
                    solver.mj_data.actuator_force
                    if engine == "cpu"
                    else solver.mjw_data.actuator_force.numpy()[0]
                )
                if not all(
                    np.isfinite(values).all()
                    for values in (qpos, forces, baseline.qpos, baseline.actuator_force)
                ):
                    raise ValueError(
                        f"Non-finite state or muscle force at pose {pose}, step {step + 1}"
                    )
                position_error = float(
                    np.abs(qpos - baseline.qpos[mapping]).max(initial=0)
                )
                if first_step_error is None:
                    first_step_error = position_error
                if position_error > max_position_error:
                    source_position = mapping[
                        int(np.argmax(np.abs(qpos - baseline.qpos[mapping])))
                    ]
                    source_joint = int(
                        np.searchsorted(
                            reference.jnt_qposadr, source_position, side="right"
                        )
                        - 1
                    )
                    largest_joint = reference.joint(source_joint).name
                max_position_error = max(max_position_error, position_error)
                force_error = float(
                    np.abs(forces - baseline.actuator_force[actuator_rows]).max(
                        initial=0
                    )
                )
                if force_error > max_force_error:
                    largest_force = names[
                        int(
                            np.argmax(
                                np.abs(forces - baseline.actuator_force[actuator_rows])
                            )
                        )
                    ]
                max_force_error = max(max_force_error, force_error)
                position_close = bool(
                    np.allclose(qpos, baseline.qpos[mapping], rtol=1e-4, atol=1e-4)
                )
                force_close_step = bool(
                    np.allclose(
                        forces,
                        baseline.actuator_force[actuator_rows],
                        rtol=1e-3,
                        atol=0.05,
                    )
                )
                if first_failure is None and not (position_close and force_close_step):
                    first_failure = {
                        "step": step + 1,
                        "qpos_within_bounds": position_close,
                        "force_within_bounds": force_close_step,
                    }
                max_contacts = max(max_contacts, int(baseline.ncon))
                trajectory_close &= position_close and force_close_step
            outcomes.append(
                {
                    "pose": pose,
                    "static_force_max_abs_error": float(
                        np.abs(actual - expected).max(initial=0)
                    ),
                    "static_force_mismatches": int(np.count_nonzero(~force_close)),
                    "mutation_detected": mutation_detected,
                    "trajectory_max_qpos_error": max_position_error,
                    "first_step_qpos_error": first_step_error,
                    "largest_qpos_error_joint": largest_joint,
                    "largest_force_error_actuator": largest_force,
                    "trajectory_max_force_error": max_force_error,
                    "first_failure": first_failure,
                    "native_max_contacts": max_contacts,
                    "trajectory_within_bounds": trajectory_close,
                }
            )
        result["poses"] = outcomes
        result["passed"] = all(
            p["static_force_mismatches"] == 0
            and p["mutation_detected"]
            and p["trajectory_within_bounds"]
            for p in outcomes
        )
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        result["passed"] = False
    result["messages"] = [
        {"count": count, "example": example} for count, example in notes.values()
    ]
    return result


def main(argv: list[str] | None = None) -> int:
    """Write a versioned numerical report for the requested models and device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("newton_dynamics"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--engine", choices=("cpu", "warp"), default="cpu")
    parser.add_argument("--poses", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--newton-revision",
        help="Newton source commit for a checkout or unreleased build",
    )
    args = parser.parse_args(argv)
    if args.poses < 2 or args.steps < 1:
        parser.error("At least two poses and one trajectory step are required")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cases = []
    for model in args.models:
        print(f"Probing {model} on {args.device}/{args.engine}", flush=True)
        cases.append(
            run_case(model, args.device, args.engine, args.poses, args.steps, args.out)
        )
        (args.out / "newton_dynamics.json").write_text(
            json.dumps(
                {
                    "versions": {package: _version(package) for package in _PACKAGES},
                    "sources": {package: _source(package) for package in _PACKAGES},
                    "newton_revision": args.newton_revision,
                    "seconds": time.perf_counter() - start,
                    "settings": {
                        "device": args.device,
                        "engine": args.engine,
                        "poses": args.poses,
                        "steps": args.steps,
                        "qpos_rtol": 1e-4,
                        "qpos_atol": 1e-4,
                        "force_rtol": 1e-3,
                        "force_atol_N": 0.05,
                        "mutation": "10% muscle force/scale increase must exceed the force bounds",
                        "contacts": "native MuJoCo contacts enabled",
                        "nconmax": 2048,
                        "njmax": 8192,
                        "strict_capacity": True,
                    },
                    "cases": cases,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(cases, indent=2))
    return int(any(not case["passed"] for case in cases))


if __name__ == "__main__":
    raise SystemExit(main())
