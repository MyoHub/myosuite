#!/usr/bin/env python3
"""Measure how much of each MyoSuite model survives NVIDIA Newton's MJCF import.

Newton (https://github.com/newton-physics/newton) is not a MyoSuite backend; this
probe measures the gap. For every model it compiles an MJCF with MuJoCo (the
reference), imports the same file with ``newton.ModelBuilder.add_mjcf``, converts it
back with ``newton.solvers.SolverMuJoCo(use_mujoco_cpu=True)`` and compares
``solver.mj_model`` with the reference:

* element counts (bodies, joints, DoFs, tendons and wraps, equalities by type,
  sensors), muscles, and the actuators and muscles that were lost;
* ``actuator_lengthrange`` of every muscle present in both models;
* ``actuator_force`` after ``mj_forward`` at ``qpos0`` with every activation and
  control at 0.5.

Each model is probed from two inputs:

* ``authored``: the MJCF MyoSuite loads for the env (``resolve_model_xml_path``),
  the recipe written by ``materialize_recipe_xml``, or the MuscleMimic package XML.
* ``normalized``: the same model rewritten by MuJoCo (``MjSpec.to_xml``) as one
  explicit file: no includes or default classes, absolute asset paths, ``-`` in
  names replaced by ``_``, and without the geoms of MuJoCo ``.msh`` meshes
  (MyoSuite's scene backdrop and logo), which Newton's mesh loader cannot read.
  What this input loses is lost by Newton's model or its MuJoCo conversion, not by
  its MJCF parser.

Both inputs are imported with ``add_mjcf(ctrl_direct=True)``: MyoSuite drives every
actuator through ``ctrl``. When the import fails (Newton resolves some MyoSuite mesh
paths differently), the error is reported and the model imported again without
meshes, so the rest of the model can still be measured. Newton leaves the converted
actuators unnamed; they are matched to the reference through Newton's actuator
labels. Everything runs on the CPU.

Usage::

    python scripts/newton_compat_probe.py --out newton_probe
    python scripts/newton_compat_probe.py --models myoElbowPose1D6MRandom-v0 --strict

Exit codes: 0, or with ``--strict`` 1 when a probed model fails to import with its
meshes or to convert, loses an actuator, or changes an actuator force beyond
``--rtol``/``--atol``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterator
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import newton
import numpy as np
import warp as wp

import myosuite  # noqa: F401  (registers the envs)
from myosuite.core.model_recipes import materialize_recipe_xml
from myosuite.utils.asset_path_resolver import (
    _anchor_to_model_dir,
    resolve_model_xml_path,
)

DEFAULT_MODELS = (
    "myoElbowPose1D6MRandom-v0",
    "myoHandPoseRandom-v0",
    "myoArmReachRandom-v0",
    "myoLegWalk-v0",
    "musclemimic:myofullbody",
    "musclemimic:bimanual",
)
INPUTS = ("authored", "normalized")
_MUSCLEMIMIC = "musclemimic:"
_PACKAGES = (
    "newton",
    "warp-lang",
    "mujoco",
    "mujoco-warp",
    "musclemimic-models",
    "myosuite",
)
# MjSpec collections whose elements take a default class.
_CLASSED = (
    "bodies",
    "joints",
    "geoms",
    "sites",
    "cameras",
    "lights",
    "actuators",
    "equalities",
    "pairs",
    "materials",
    "meshes",
)
_NAMED = (
    ("body", mujoco.mjtObj.mjOBJ_BODY, "nbody"),
    ("joint", mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
    ("site", mujoco.mjtObj.mjOBJ_SITE, "nsite"),
    ("tendon", mujoco.mjtObj.mjOBJ_TENDON, "ntendon"),
    ("actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
)
_GEOM_WRAPS = [int(mujoco.mjtWrap.mjWRAP_SPHERE), int(mujoco.mjtWrap.mjWRAP_CYLINDER)]


def _slug(model: str) -> str:
    return re.sub(r"[^\w.-]", "_", model)


def authored_xml(model: str, workdir: Path) -> Path:
    """Return the MJCF MyoSuite loads for *model*.

    Args:
        model: A registered env id, or ``musclemimic:<name>`` for a
            ``musclemimic_models`` XML.
        workdir: Directory for the XML of recipe-built models.

    Returns:
        Path of the MJCF.

    Raises:
        ValueError: If the env edits a ``model_path`` model (not supported).
    """
    if model.startswith(_MUSCLEMIMIC):
        from musclemimic_models import get_xml_path

        return Path(get_xml_path(model.removeprefix(_MUSCLEMIMIC)))
    kwargs = gym.spec(model).kwargs
    recipe = kwargs.get("model_recipe")
    if recipe is not None:
        dest = workdir / f"{_slug(model)}.authored.xml"
        return materialize_recipe_xml(recipe, dest, edit_fn=kwargs.get("edit_fn"))
    if kwargs.get("edit_fn") is not None:
        raise ValueError(f"{model} edits its model_path model; not supported.")
    return resolve_model_xml_path(kwargs["model_path"])


def normalized_xml(source: Path, dest: Path) -> Path:
    """Rewrite *source* with MuJoCo as one explicit MJCF.

    Geoms of ``.msh`` meshes are dropped (Newton loads meshes with trimesh, which
    cannot read MuJoCo's own format). Every element moves to the main default
    class, so ``MjSpec.to_xml`` writes all of its attributes (Newton merges a class
    default per attribute, not per value), asset directories become absolute, and
    ``-`` in names becomes ``_`` (Newton sanitizes referenced names but not element
    names).

    Args:
        source: MJCF that MuJoCo compiles.
        dest: Output path.

    Returns:
        *dest*.

    Raises:
        ValueError: If replacing ``-`` would make two names equal.
    """
    spec = mujoco.MjSpec.from_file(str(source))
    msh = [mesh for mesh in spec.meshes if mesh.file.lower().endswith(".msh")]
    msh_names = {mesh.name for mesh in msh}
    for element in [g for g in spec.geoms if g.meshname in msh_names] + msh:
        spec.delete(element)
    spec.compile()  # to_xml needs a compiled spec
    for kind in _CLASSED:
        for element in getattr(spec, kind):
            element.classname = spec.default
    for element in (*spec.bodies, *spec.frames):
        element.childclass = ""
    root = ET.fromstring(spec.to_xml())
    _anchor_to_model_dir(root, source.parent)
    names = {element.get("name") for element in root.iter()} - {None}
    renamed = {name: name.replace("-", "_") for name in names if "-" in name}
    clashes = set(renamed.values()) & names
    if clashes:
        raise ValueError(f"Replacing '-' would merge names: {sorted(clashes)[:5]}")
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if value in renamed:
                element.set(key, renamed[value])
    dest.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return dest


def model_counts(m: mujoco.MjModel) -> dict[str, Any]:
    """Return the sizes and options of a model that a conversion can lose or change.

    Args:
        m: A compiled MuJoCo model.

    Returns:
        Element counts, muscles, equalities by type, total mass and integrator.
    """
    equalities = Counter(mujoco.mjtEq(t).name.removeprefix("mjEQ_") for t in m.eq_type)
    integrator = mujoco.mjtIntegrator(m.opt.integrator).name.removeprefix("mjINT_")
    return {
        "nbody": m.nbody,
        "njnt": m.njnt,
        "nq": m.nq,
        "nv": m.nv,
        "nu": m.nu,
        "muscles": int(
            np.count_nonzero(m.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
        ),
        "ntendon": m.ntendon,
        "nwrap": m.nwrap,
        "wrap_geoms": int(np.isin(m.wrap_type, _GEOM_WRAPS).sum()),
        "neq": {k.lower(): v for k, v in sorted(equalities.items())},
        "nsensor": m.nsensor,
        "mass": float(m.body_mass.sum()),
        "timestep": float(m.opt.timestep),
        "integrator": integrator.lower(),
    }


def newton_counts(model: Any) -> dict[str, int]:
    """Return the sizes of an imported Newton model.

    Args:
        model: A finalized ``newton.Model``.

    Returns:
        Bodies (without the world), joints, DoFs, MuJoCo actuators, muscles and
        tendons.
    """
    frequencies = model.custom_frequency_counts
    gaintype = getattr(getattr(model, "mujoco", None), "actuator_gaintype", None)
    muscle = int(mujoco.mjtGain.mjGAIN_MUSCLE)
    return {
        "bodies": model.body_count,
        "joints": model.joint_count,
        "dofs": model.joint_dof_count,
        "actuators": frequencies.get("mujoco:actuator", 0),
        "muscles": 0 if gaintype is None else int((gaintype.numpy() == muscle).sum()),
        "tendons": frequencies.get("mujoco:tendon", 0),
    }


def _names(m: mujoco.MjModel, obj: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(m, obj, i) or "" for i in range(count)]


def converted_actuator_names(solver: Any, model: Any) -> list[str]:
    """Return the MJCF names of the actuators in ``solver.mj_model``.

    Newton adds actuators to the MuJoCo spec unnamed; ``mjc_actuator_to_newton_idx``
    maps each direct-control one to its index in ``model.mujoco.actuator_label``.

    Args:
        solver: A ``newton.solvers.SolverMuJoCo``.
        model: The Newton model it converted.

    Returns:
        One name per converted actuator; ``""`` where none is known.
    """
    mj_model = solver.mj_model
    names = _names(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, mj_model.nu)
    labels = getattr(getattr(model, "mujoco", None), "actuator_label", None)
    index = getattr(solver, "mjc_actuator_to_newton_idx", None)
    source = getattr(solver, "mjc_actuator_ctrl_source", None)
    if labels is None or index is None or source is None:
        return names
    direct = int(newton.solvers.SolverMuJoCo.CtrlSource.CTRL_DIRECT)
    return [
        name or (labels[int(i)] if int(s) == direct else "")
        for name, i, s in zip(names, index.numpy(), source.numpy())
    ]


def actuator_forces(m: mujoco.MjModel) -> np.ndarray:
    """Return ``actuator_force`` at ``qpos0`` with every activation and control at 0.5.

    Args:
        m: A compiled MuJoCo model.

    Returns:
        One force per actuator.
    """
    data = mujoco.MjData(m)
    data.act[:] = 0.5
    data.ctrl[:] = 0.5
    mujoco.mj_forward(m, data)
    return data.actuator_force.copy()


def _finite(values: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in values]


def agreement(
    names: list[str], ref: np.ndarray, new: np.ndarray, rtol: float, atol: float
) -> dict[str, Any]:
    """Compare matched rows of *new* with *ref* (one row per name).

    Args:
        names: Name of each row.
        ref: Reference values.
        new: Values of the converted model.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Row counts (compared, mismatched beyond the tolerance, non-finite), the
        largest absolute and relative differences, and the mismatched rows.
    """
    ref = ref.reshape(len(names), -1) if names else np.zeros((0, 1))
    new = new.reshape(len(names), -1) if names else np.zeros((0, 1))
    close = np.isclose(new, ref, rtol=rtol, atol=atol).all(axis=1)
    finite = np.isfinite(new).all(axis=1)
    diff = np.abs(new - ref)[finite]
    scale = np.abs(ref[finite])
    rel = diff[scale > 0] / scale[scale > 0]
    return {
        "compared": len(names),
        "mismatched": int(np.count_nonzero(~close)),
        "nonfinite": int(np.count_nonzero(~finite)),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "max_rel_diff": float(rel.max()) if rel.size else 0.0,
        "mismatches": {
            n: {"reference": _finite(r), "newton": _finite(c)}
            for n, r, c, ok in zip(names, ref, new, close)
            if not ok
        },
    }


def compare(
    ref: mujoco.MjModel,
    new: mujoco.MjModel,
    new_actuators: list[str],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare the converted model *new* with the reference *ref*.

    Args:
        ref: Model MuJoCo compiled from the probed MJCF.
        new: ``SolverMuJoCo.mj_model`` of the same MJCF.
        new_actuators: MJCF names of the actuators of *new*.
        rtol: Relative tolerance of length ranges and forces.
        atol: Absolute tolerance of length ranges and forces.

    Returns:
        Lost actuators and muscles, names kept per element type, ``qpos0`` and body
        mass differences, and the length range and force agreement.
    """
    muscle = int(mujoco.mjtGain.mjGAIN_MUSCLE)
    ref_actuators = _names(ref, mujoco.mjtObj.mjOBJ_ACTUATOR, ref.nu)
    ref_id = {n: i for i, n in enumerate(ref_actuators)}
    new_id = {n: i for i, n in enumerate(new_actuators) if n}
    common = [n for n in ref_actuators if n in new_id]
    muscles = [n for n in ref_actuators if ref.actuator_gaintype[ref_id[n]] == muscle]
    kept = [
        n for n in muscles if n in new_id and new.actuator_gaintype[new_id[n]] == muscle
    ]
    rows = [ref_id[n] for n in common], [new_id[n] for n in common]
    lr_rows = [ref_id[n] for n in kept], [new_id[n] for n in kept]
    same_joints = np.array_equal(ref.jnt_type, new.jnt_type) and np.array_equal(
        ref.jnt_qposadr, new.jnt_qposadr
    )
    names_kept = {}
    for label, obj, size in _NAMED:
        ref_names = set(_names(ref, obj, getattr(ref, size))) - {""}
        new_names = set(_names(new, obj, getattr(new, size)))
        names_kept[label] = f"{len(ref_names & new_names)}/{len(ref_names)}"
    return {
        "actuators_lost": [n for n in ref_actuators if n not in new_id],
        "muscles_lost": [n for n in muscles if n not in kept],
        "names_kept": names_kept,
        "qpos0_max_abs_diff": (
            float(np.abs(new.qpos0 - ref.qpos0).max(initial=0.0))
            if same_joints
            else None
        ),
        "body_mass_max_abs_diff": (
            float(np.abs(new.body_mass - ref.body_mass).max())
            if new.nbody == ref.nbody
            else None
        ),
        "lengthrange": agreement(
            kept,
            ref.actuator_lengthrange[lr_rows[0]],
            new.actuator_lengthrange[lr_rows[1]],
            rtol,
            atol,
        ),
        "forces": agreement(
            common,
            actuator_forces(ref)[rows[0]],
            actuator_forces(new)[rows[1]],
            rtol,
            atol,
        ),
    }


@contextlib.contextmanager
def _timed(seconds: dict[str, float], stage: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        seconds[stage] = round(time.perf_counter() - start, 2)


@contextlib.contextmanager
def _captured(notes: dict[str, list[Any]]) -> Iterator[None]:
    """Collect Newton's warnings and its verbose ``Warning`` lines into *notes*.

    *notes* maps a message template (names and numbers blanked) to its count and a
    first example.
    """
    stdout = io.StringIO()
    with (
        warnings.catch_warnings(record=True) as caught,
        contextlib.redirect_stdout(stdout),
    ):
        warnings.simplefilter("always")
        try:
            yield
        finally:
            lines = [str(w.message).strip() for w in caught]
            lines += [
                line.strip()
                for line in stdout.getvalue().splitlines()
                if line.strip().startswith("Warning")
            ]
            for line in lines:
                template = re.sub(r"\d+", "#", re.sub(r"'[^']*'", "'...'", line))
                notes.setdefault(template, [0, line])[0] += 1


def newton_import(xml: Path, parse_meshes: bool) -> Any:
    """Import *xml* with Newton's MJCF importer and finalize it on the CPU.

    ``ctrl_direct=True`` keeps every actuator on ``ctrl``, as MyoSuite drives them.

    Args:
        xml: MJCF file.
        parse_meshes: Whether to load mesh geoms.

    Returns:
        The finalized ``newton.Model``.
    """
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(
        str(xml), ctrl_direct=True, parse_meshes=parse_meshes, verbose=True
    )
    return builder.finalize(device="cpu")


def _error(stage: str, err: Exception) -> dict[str, str]:
    return {"stage": stage, "type": type(err).__name__, "message": str(err)}


def probe(
    model: str, kind: str, workdir: Path, rtol: float, atol: float
) -> dict[str, Any]:
    """Probe one model from one input; a failure is recorded, not raised.

    Args:
        model: Env id or ``musclemimic:<name>``.
        kind: ``"authored"`` or ``"normalized"``.
        workdir: Directory for the generated MJCF files.
        rtol: Relative tolerance of length ranges and forces.
        atol: Absolute tolerance of length ranges and forces.

    Returns:
        The JSON-ready result: counts of the reference, the Newton model and the
        converted model, the comparison, stage timings and Newton's messages, or
        the stage and exception that stopped the probe. A failed import is retried
        without meshes, and its error kept as ``mesh_error``.
    """
    case: dict[str, Any] = {"model": model, "input": kind}
    seconds: dict[str, float] = {}
    notes: dict[str, list[Any]] = {}
    stage = "input"
    try:
        with _timed(seconds, stage):
            xml = authored_xml(model, workdir)
            if kind == "normalized":
                xml = normalized_xml(xml, workdir / f"{_slug(model)}.normalized.xml")
        case["xml"] = str(xml)
        stage = "reference"
        with _timed(seconds, stage):
            ref = mujoco.MjModel.from_xml_path(str(xml))
        case["reference"] = model_counts(ref)
        stage = "import"
        with _timed(seconds, stage), _captured(notes), wp.ScopedDevice("cpu"):
            try:
                newton_model = newton_import(xml, parse_meshes=True)
            except Exception as err:  # e.g. a mesh path Newton resolves differently
                case["mesh_error"] = _error(stage, err)
                newton_model = newton_import(xml, parse_meshes=False)
        case["newton"] = newton_counts(newton_model)
        stage = "convert"
        with _timed(seconds, stage), _captured(notes), wp.ScopedDevice("cpu"):
            solver = newton.solvers.SolverMuJoCo(newton_model, use_mujoco_cpu=True)
        case["converted"] = model_counts(solver.mj_model)
        stage = "compare"
        with _timed(seconds, stage):
            names = converted_actuator_names(solver, newton_model)
            case.update(compare(ref, solver.mj_model, names, rtol, atol))
    except Exception as err:  # every failure is a probe result
        case["error"] = _error(stage, err)
        if case.get("mesh_error", {}).get("message") == str(err):
            del case["mesh_error"]  # the import fails with or without meshes
    case["seconds"] = seconds
    case["messages"] = [
        {"template": t, "count": c, "example": e}
        for t, (c, e) in sorted(notes.items(), key=lambda item: -item[1][0])
    ]
    return case


def _pair(
    case: dict[str, Any],
    key: str,
    newton_key: str | None = None,
    fmt: Callable[[Any], str] = str,
) -> str:
    """``reference -> converted`` of *key*, or the Newton model's count if unconverted."""
    if "reference" not in case:
        return "-"
    ref = fmt(case["reference"][key])
    if "converted" in case:
        return f"{ref} -> {fmt(case['converted'][key])}"
    if newton_key is not None and "newton" in case:
        return f"{ref} -> {case['newton'][newton_key]} (Newton model)"
    return f"{ref} -> -"


def _equalities(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items()) or "0"


def _sig(value: float) -> str:
    return f"{value:.3g}"


def _number(case: dict[str, Any], section: str, key: str) -> str:
    value = case.get(section, {}).get(key)
    return "-" if value is None else _sig(value)


def _muscles_lost(case: dict[str, Any]) -> str:
    if "muscles_lost" in case:
        return str(len(case["muscles_lost"]))
    if "newton" in case:
        lost = case["reference"]["muscles"] - case["newton"]["muscles"]
        return f"{lost} (Newton model)"
    return "-"


def _qpos(case: dict[str, Any]) -> str:
    if "qpos0_max_abs_diff" not in case:
        return "-"
    value = case["qpos0_max_abs_diff"]
    return "joints differ" if value is None else f"{value:.3g}"


def _defective(case: dict[str, Any]) -> bool:
    forces = case.get("forces", {})
    return bool(
        "error" in case
        or "mesh_error" in case
        or case.get("actuators_lost")
        or forces.get("mismatched")
        or forces.get("nonfinite")
    )


def render(report: dict[str, Any]) -> str:
    """Return the report as Markdown.

    Args:
        report: Versions, settings, run time and the probed cases.

    Returns:
        Two tables (actuators and forces; structure), failures and Newton's messages.
    """
    settings = report["settings"]
    versions = ", ".join(f"{k} {v}" for k, v in report["versions"].items())
    lines = [
        "# Newton compatibility probe",
        "",
        f"{versions}. `SolverMuJoCo(use_mujoco_cpu=True)` on the CPU; "
        f"rtol {settings['rtol']:g}, atol {settings['atol']:g}; "
        f"{report['seconds']:.0f} s.",
        "",
        "Cells read MuJoCo reference -> Newton. `authored` is the MJCF MyoSuite loads; "
        "`normalized` is MuJoCo's explicit rewrite of it "
        "(see `scripts/newton_compat_probe.py`). "
        "*no meshes*: Newton failed to load the meshes and the model was imported "
        "without them.",
        "",
        "| model | input | result | actuators | muscles | muscles lost "
        "| length ranges off | max force diff (N) | max rel. force diff "
        "| non-finite forces |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    cases = report["cases"]
    for case in cases:
        error = case.get("error")
        result = f"failed at {error['stage']} ({error['type']})" if error else "ok"
        if "mesh_error" in case:
            result += ", *no meshes*"
        lr = case.get("lengthrange")
        lr_cell = (
            f"{lr['mismatched']}/{lr['compared']} (max {lr['max_abs_diff']:.3g} m)"
            if lr
            else "-"
        )
        lines.append(
            f"| {case['model']} | {case['input']} | {result} "
            f"| {_pair(case, 'nu', 'actuators')} | {_pair(case, 'muscles', 'muscles')} "
            f"| {_muscles_lost(case)} | {lr_cell} "
            f"| {_number(case, 'forces', 'max_abs_diff')} "
            f"| {_number(case, 'forces', 'max_rel_diff')} "
            f"| {case.get('forces', {}).get('nonfinite', '-')} |"
        )
    lines += [
        "",
        "| model | input | nbody | njnt | nq | nv | mass (kg) | tendons "
        "| wraps (geom wraps) | equalities | sensors | integrator | qpos0 max diff "
        "| names kept (body, joint, site, tendon, actuator) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in cases:
        wraps = _pair(case, "nwrap") + (
            f" ({_pair(case, 'wrap_geoms')})" if "reference" in case else ""
        )
        kept = case.get("names_kept")
        lines.append(
            f"| {case['model']} | {case['input']} | {_pair(case, 'nbody')} "
            f"| {_pair(case, 'njnt')} | {_pair(case, 'nq')} | {_pair(case, 'nv')} "
            f"| {_pair(case, 'mass', fmt=_sig)} "
            f"| {_pair(case, 'ntendon', 'tendons')} | {wraps} "
            f"| {_pair(case, 'neq', fmt=_equalities)} | {_pair(case, 'nsensor')} "
            f"| {_pair(case, 'integrator')} | {_qpos(case)} "
            f"| {', '.join(kept.values()) if kept else '-'} |"
        )
    failures = [
        (case, key, label)
        for case in cases
        for key, label in (("mesh_error", "import with meshes"), ("error", ""))
        if key in case
    ]
    if failures:
        lines += ["", "## Failures", ""]
        for case, key, label in failures:
            error = case[key]
            lines.append(
                f"- **{case['model']}** ({case['input']}), {label or error['stage']}: "
                f"`{_code(error['type'] + ': ' + error['message'], 400)}`"
            )
    noted = [c for c in cases if c["messages"]]
    if noted:
        lines += ["", "## Newton messages (count x first example)", ""]
        for case in noted:
            top = "; ".join(
                f"{m['count']} x `{_code(m['example'], 160)}`"
                for m in case["messages"][:4]
            )
            lines.append(f"- **{case['model']}** ({case['input']}): {top}")
    return "\n".join(lines) + "\n"


def _code(text: str, limit: int) -> str:
    """One line of *text* that fits in a Markdown code span."""
    return " ".join(text.split()).replace("`", "'")[:limit]


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def main(argv: list[str] | None = None) -> int:
    """Run the probe and write ``newton_probe.md`` and ``newton_probe.json``.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("newton_probe"),
        help="report directory (default: newton_probe)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="env ids or musclemimic:<name> (default: all six)",
    )
    parser.add_argument("--inputs", nargs="+", choices=INPUTS, default=list(INPUTS))
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when a model fails, loses actuators or changes a force",
    )
    args = parser.parse_args(argv)

    wp.config.quiet = True
    workdir = args.out / "inputs"
    workdir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cases = []
    for model in args.models:
        for kind in args.inputs:
            print(f"probing {model} ({kind})", flush=True)
            cases.append(probe(model, kind, workdir, args.rtol, args.atol))
    report = {
        "versions": {package: _version(package) for package in _PACKAGES},
        "settings": {
            "rtol": args.rtol,
            "atol": args.atol,
            "device": "cpu",
            "add_mjcf": "ctrl_direct=True; parse_meshes=False after a failed import",
            "solver": "SolverMuJoCo(use_mujoco_cpu=True)",
            "forces": "qpos0, act=0.5, ctrl=0.5, mj_forward",
        },
        "seconds": round(time.perf_counter() - start, 1),
        "cases": cases,
    }
    markdown = render(report)
    (args.out / "newton_probe.md").write_text(markdown, encoding="utf-8")
    text = json.dumps(report, indent=2)
    (args.out / "newton_probe.json").write_text(text + "\n", encoding="utf-8")
    print(markdown)
    return int(args.strict and any(_defective(case) for case in cases))


if __name__ == "__main__":
    raise SystemExit(main())
