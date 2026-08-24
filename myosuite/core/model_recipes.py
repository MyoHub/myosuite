# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Named model recipes registered via @model_recipe.

Import this module to populate the _RECIPES registry before calling
build_from_recipe(). myosuite/core/__init__.py imports this automatically.

.. warning::
    **Experimental.**  Challenge recipes are partial scene descriptions; they do
    not achieve full parity with the official challenge XML files.  For training
    or evaluation, use ``gym.make("<challenge-env-id>")`` instead.

Challenge environment recipes
------------------------------
ModelBuilder covers the body-part assembly and simple props natively.
The table below shows what each challenge recipe composes natively vs. what
requires an :meth:`~myosuite.core.model_builder.ModelBuilder.apply_transform`
escape hatch.

+------------------+--------------------------------+----------------------------------+
| Recipe           | Composed natively              | Needs apply_transform            |
+==================+================================+==================================+
| baoding          | hand fragment + 2 balls        | tendons, tracking sites          |
|                  | (mass, condim)                 |                                  |
+------------------+--------------------------------+----------------------------------+
| die_reorient     | hand fragment                  | 12-capsule + 3-box die body,     |
|                  |                                | cube-grid texture, slide joints  |
+------------------+--------------------------------+----------------------------------+
| tabletennis      | arm fragment + ball            | mirrored left arm, fluidcoef,    |
|                  | (mass, condim) + table/paddle/ | contact zone sites               |
|                  | net mesh bodies                |                                  |
+------------------+--------------------------------+----------------------------------+
| relocate         | arm fragment + box object      | static table, bin mocap body,    |
|                  | (mass, condim)                 | slide joints on object           |
+------------------+--------------------------------+----------------------------------+

Arm recipes
-----------

+------------------+--------------------------------+----------------------------------+
| Recipe           | Notes                          | Requirements                     |
+==================+================================+==================================+
| full_arm         | right arm (shoulder+elbow+hand)| pre-PR-#73 body names            |
+------------------+--------------------------------+----------------------------------+
| bimanual_arm     | left + right (77 DOF,          | myo_sim PR #73 merged into       |
|                  | 126 muscles)                   | the submodule                    |
+------------------+--------------------------------+----------------------------------+
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from myosuite.core.model_builder import ModelBuilder, build_from_recipe, model_recipe

# Root of the myosuite package — used to locate asset files.
_ASSETS = Path(__file__).parent.parent / "envs" / "myo" / "assets"


def materialize_recipe_xml(name: str, dest: Path | None = None) -> Path:
    """Write a recipe-built spec to a standalone MJCF file on disk.

    Some consumers (e.g. external packages like ``mjlab``, which read a
    plain ``model_path`` attribute off a task config rather than calling
    into myosuite's ``ModelBuilder``/recipe machinery) need a real file on
    disk, not a live ``MjSpec``. This bridges that gap: build the named
    recipe, then serialize and sanitize it via myo_sim's own
    ``sanitize_spec_xml`` (adds ``compiler meshdir`` pointing at myo_sim's
    package root — ``MjSpec.to_xml()`` emits mesh/texture ``file=``
    attributes as paths relative to that root but doesn't itself include a
    ``meshdir`` — and fixes up ``MjSpec.to_xml()``'s known empty/duplicate
    default-class serialization quirks).

    Args:
        name: Recipe name registered via ``@model_recipe``.
        dest: Output path. Default: an OS temp directory
            (``<tempdir>/myosuite_recipes/<name>.xml``).

    Returns:
        Absolute path to the written file.
    """
    import myo_sim  # type: ignore[import-untyped]
    from myo_sim.build.compose import sanitize_spec_xml  # type: ignore[import-untyped]

    model, spec = build_from_recipe(name)
    xml = sanitize_spec_xml(
        spec.to_xml(), asset_dir=str(myo_sim.MODELS_DIR), model=model
    )

    out = dest
    if out is None:
        import tempfile

        out = Path(tempfile.gettempdir()) / "myosuite_recipes" / f"{name}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    return out.resolve()


@model_recipe("elbow_standard")
def _elbow_standard(b: ModelBuilder) -> ModelBuilder:
    """Standard 2-DOF elbow model (flexion + supination, 6 muscles)."""
    return b.attach_fragment("elbow")


@model_recipe("elbow_sarcopenia")
def _elbow_sarcopenia(b: ModelBuilder) -> ModelBuilder:
    """Elbow model with sarcopenic (50% force) muscles."""
    return b.attach_fragment("elbow").apply_sarcopenia(force_scale=0.5)


def _calibrate_compose_arm_root(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Reposition the myo_sim-pip-composed arm+scaffold to the legacy world frame.

    ``myo_sim.load_spec("myoarm_r")`` returns a passive anatomical torso
    scaffold + right arm, rooted at ``"Full Body"``, which sits at
    ``pos=(0, 0, 1)`` with identity orientation by default. This module's
    arm-task furniture (targets, keyframes) uses absolute world positions
    calibrated against the legacy ``full_body``-rooted frame instead. Apply
    the rigid transform (fit via Kabsch on clavicle/humerus/ulna/radius/
    lunate/scaphoid/firstmc landmarks at qpos0, residual < 6e-16 m — the two
    arm chains are geometrically identical, just rigidly offset) that maps
    the new root onto the legacy frame, so existing task furniture lines up
    without every task recipe needing its own correction.
    """
    root = spec.body("Full Body")
    root.pos = [0.00685025, -0.00150371, 0.91886071]
    root.quat = [0.99999984, -0.00039816, 0.00000016, 0.00039816]
    return spec


def _arm_builder() -> ModelBuilder:
    """Return a ModelBuilder seeded with myo_sim's composed myoarm_r, or static fallback."""
    try:
        import myo_sim  # type: ignore[import-untyped]

        arm_spec = _calibrate_compose_arm_root(myo_sim.load_spec("myoarm_r"))
        return ModelBuilder().attach_spec(arm_spec, name="arm")
    except (ImportError, AttributeError):
        return ModelBuilder().attach_fragment("arm")


@model_recipe("full_arm")
def _full_arm(b: ModelBuilder) -> ModelBuilder:
    """Full arm: shoulder through hand, right arm only.

    Sourced from myo_sim's ``myoarm_r`` composition (passive anatomical
    torso scaffold + right arm, 63 muscles) when available, repositioned to
    the legacy world frame via :func:`_calibrate_compose_arm_root`; falls
    back to the bundled static ``"arm"`` fragment otherwise. For the
    bimanual (left + right) model, use the ``bimanual_arm`` recipe.
    """
    return _arm_builder()


def _add_reach_props(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add the forearm_tip site + forearm_tip_target site for arm-reach tasks.

    ``forearm_tip`` is a static (pose-independent) site — the legacy XML
    fixed it to the non-moving ``full_body`` wrapper body, not to any bone
    in the arm chain, so its world position never tracks the wrist as the
    arm moves. The local offset below reproduces that exact legacy world
    position on the newly-calibrated ``"Full Body"`` root (computed via
    forward kinematics: both models compiled, then
    ``local = R.T @ (target_world - full_body_world)``).
    """
    full_body = spec.body("Full Body")
    full_body.add_site(
        name="forearm_tip",
        size=[0.005],
        pos=[-0.01424121, -0.49090358, 0.52574885],
        rgba=[0.8, 0, 0, 1],
        group=4,
    )
    spec.worldbody.add_site(
        name="forearm_tip_target",
        pos=[-0.2, -0.2, 1.2],
        size=[0.02],
        rgba=[0, 1, 0, 0.3],
    )
    return spec


@model_recipe("arm_reach")
def _arm_reach(b: ModelBuilder) -> ModelBuilder:
    """Full arm + forearm_tip/forearm_tip_target sites for reach tasks."""
    return _arm_builder().apply_transform(_add_reach_props)


@model_recipe("bimanual_arm")
def _bimanual_arm(b: ModelBuilder) -> ModelBuilder:
    """Bimanual arm model (left + right combined).

    Requires the ``arms`` fragment introduced in **myo_sim PR #73**
    (``arm/myoarms.xml``: 77 DOF, 126 muscles).  The model will be available
    once the myo_sim submodule is updated to include PR #73.

    Right-arm body names carry an ``_r`` suffix (e.g. ``chest_r``);
    left-arm body names carry a ``_left`` suffix.

    Raises:
        FileNotFoundError: If the ``arm/myoarms.xml`` fragment is not yet
                           present in the myo_sim submodule.
    """
    return b.attach_fragment("arms")


def _myohand_r_path() -> Path | None:
    """Return path to myohand_r.xml from myo_sim pip package, or None."""
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "hand" / "myohand_r.xml"
        if p.exists():
            return p
    except ImportError:
        pass
    return None


def _calibrate_compose_hand_root(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Reposition the myo_sim-pip-composed hand to the legacy world frame.

    ``load_right_hand_from_arm_spec()`` returns the hand still rooted at
    ``myoarm_r_root``, which sits at the world origin in myo_sim pip's
    PR-#73 arm-chain convention. All of this module's hand-task furniture
    (key, pen, balls, targets) uses absolute world positions calibrated
    against the legacy ``thorax``-rooted frame instead. Apply the rigid
    transform (fit via Kabsch on S_grasp/IFtip/THtip/lunate/radius
    landmarks at qpos0, residual < 4e-5 m) that maps the new root onto the
    legacy frame, so existing task furniture lines up without every task
    recipe needing its own correction.
    """
    root = spec.body("myoarm_r_root")
    root.pos = [-0.02525155, 0.07146933, 1.39249298]
    root.quat = [0.52167746, 0.46981366, -0.47875126, -0.52718591]
    return spec


def _hand_builder() -> ModelBuilder:
    """Return a ModelBuilder seeded with myohand_r.xml, or fall back to compose/fragment."""
    p = _myohand_r_path()
    if p is not None:
        return ModelBuilder.from_xml_file(p)
    try:
        from myo_sim.build.compose import load_right_hand_from_arm_spec  # type: ignore[import-untyped]

        hand_spec = _calibrate_compose_hand_root(load_right_hand_from_arm_spec())
        return ModelBuilder().attach_spec(hand_spec, name="hand")
    except (ImportError, AttributeError):
        return ModelBuilder().attach_fragment("hand")


def _legacy_hand_builder() -> ModelBuilder:
    """Return the release-era hand seed used by public contact-task policies.

    Seeds from the shared ``hand/assets/myohand_{assets,body}.xml`` files plus
    the myo_sim scene via an in-memory MjSpec (no committed task/wrapper XML).
    Contact-task furniture is added by MjSpec transforms below.
    """
    from myosuite.utils.asset_path_resolver import (
        get_sim_asset_root,
        resolve_model_xml_path,
    )

    hand_dir = _ASSETS / "hand"
    sim_root = get_sim_asset_root("myo_sim")
    scene_path = resolve_model_xml_path(sim_root / "scene" / "myosuite_scene.xml")
    assets_bytes = (hand_dir / "assets" / "myohand_assets.xml").read_bytes()
    body_bytes = (hand_dir / "assets" / "myohand_body.xml").read_bytes()
    meshdir = sim_root.as_posix().rstrip("/") + "/"
    seed_xml = f"""
    <mujoco model="MyoHand legacy task seed">
      <include file="assets/myohand_assets.xml"/>
      <include file="{scene_path.as_posix()}"/>
      <compiler meshdir="{meshdir}" texturedir="{meshdir}"/>
      <worldbody>
        <include file="assets/myohand_body.xml"/>
      </worldbody>
    </mujoco>
    """
    return ModelBuilder.from_xml_string(
        seed_xml,
        include={
            "assets/myohand_assets.xml": assets_bytes,
            "assets/myohand_body.xml": body_bytes,
        },
    )


@model_recipe("hand_standard")
def _hand_standard(b: ModelBuilder) -> ModelBuilder:
    """Standard myohand model — from myohand_r.xml, compose pipeline, or static fallback."""
    return _hand_builder()


def _add_pose_props(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add fingertip target sites and error tendons for pose/reach visualisation."""
    tip_names = ["THtip_r", "IFtip_r", "MFtip_r", "RFtip_r", "LFtip_r"]
    tip_colors = [
        (0.8, 0.0, 0.0, 0.8),
        (0.0, 0.8, 0.0, 0.8),
        (0.0, 0.0, 0.8, 0.8),
        (0.8, 0.8, 0.0, 0.8),
        (0.8, 0.0, 0.8, 0.8),
    ]
    for tip, rgba in zip(tip_names, tip_colors):
        s = spec.worldbody.add_site()
        s.name = f"{tip}_target"
        s.pos = [0.0, 0.0, 0.002]
        s.size = [0.005, 0.005, 0.005]
        s.rgba = rgba
    return spec


@model_recipe("hand_pose")
def _hand_pose(b: ModelBuilder) -> ModelBuilder:
    """myohand_r + fingertip target sites for pose-tracking tasks."""
    return _hand_builder().apply_transform(_add_pose_props)


@model_recipe("hand_keyturn")
def _hand_keyturn(b: ModelBuilder) -> ModelBuilder:
    """myohand_r + key body/joint/site for the key-turning task."""

    def _add_key(spec: mujoco.MjSpec) -> mujoco.MjSpec:
        key_body = spec.worldbody.add_body()
        key_body.name = "key"
        key_body.pos = [-0.15, -0.55, 1.425]
        key_body.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER
        spec.compiler.degree = False
        key_body.alt.euler = [0.0, 0.0, 2.2]

        # Key head ellipsoid
        g_head = key_body.add_geom()
        g_head.name = "keyhead"
        g_head.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        g_head.size = [0.030, 0.030, 0.004]
        g_head.contype = 1
        g_head.conaffinity = 1
        g_head.rgba = [0.6, 0.6, 0.5, 1.0]

        # Key shaft capsule
        g_shaft = key_body.add_geom()
        g_shaft.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g_shaft.size = [0.005, 0.070, 0.0]
        g_shaft.pos = [-0.045, 0.0, 0.0]
        g_shaft.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER
        g_shaft.alt.euler = [0.0, 1.57, 0.0]
        g_shaft.contype = 1
        g_shaft.conaffinity = 1
        g_shaft.rgba = [0.6, 0.6, 0.5, 1.0]

        # Key bow box
        g_bow = key_body.add_geom()
        g_bow.type = mujoco.mjtGeom.mjGEOM_BOX
        g_bow.size = [0.015, 0.010, 0.004]
        g_bow.pos = [-0.1, 0.008, 0.0]
        g_bow.contype = 1
        g_bow.conaffinity = 1
        g_bow.rgba = [0.6, 0.6, 0.5, 1.0]

        # Key hinge joint
        j = key_body.add_joint()
        j.name = "keyjoint"
        j.type = mujoco.mjtJoint.mjJNT_HINGE
        j.axis = [1.0, 0.0, 0.0]
        j.frictionloss = 0.02
        j.damping = [0.1, 0.0, 0.0]

        # The release XML has an unnamed visual site plus the named reward /
        # observation site.
        visual_site = key_body.add_site()
        visual_site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        visual_site.size = [0.03, 0.03, 0.03]
        visual_site.rgba = [0.5, 0.7, 0.8, 0.3]

        s = key_body.add_site()
        s.name = "keyhead"
        s.size = [0.005, 0.005, 0.005]

        return spec

    return _legacy_hand_builder().apply_transform(_add_key)


@model_recipe("hand_hold")
def _hand_hold(b: ModelBuilder) -> ModelBuilder:
    """myohand_r + object body + goal site for object-holding tasks."""

    def _add_hold_props(spec: mujoco.MjSpec) -> mujoco.MjSpec:
        # Goal site in worldbody
        goal = spec.worldbody.add_site()
        goal.name = "goal"
        goal.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        goal.size = [0.025, 0.036, 0.030]
        goal.pos = [-0.240, -0.520, 1.470]
        goal.rgba = [0.0, 1.0, 0.0, 0.2]

        # Object body with freejoint
        obj_body = spec.worldbody.add_body()
        obj_body.name = "object"
        obj_body.pos = [-0.235, -0.51, 1.450]

        obj_geom = obj_body.add_geom()
        obj_geom.name = "object"
        obj_geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        obj_geom.size = [0.025, 0.036, 0.030]
        obj_geom.condim = 1
        obj_geom.conaffinity = 1
        obj_geom.rgba = [0.4, 0.6, 0.98, 1.0]

        obj_body.add_freejoint(name="object_free")

        obj_site = obj_body.add_site()
        obj_site.name = "object"
        obj_site.size = [0.005, 0.005, 0.005]

        return spec

    return _legacy_hand_builder().apply_transform(_add_hold_props)


def _add_pen_props(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add pen object + target body + eps_ball site to a hand spec."""
    # eps_ball site in worldbody
    eps = spec.worldbody.add_site()
    eps.name = "eps_ball"
    eps.type = mujoco.mjtGeom.mjGEOM_SPHERE
    eps.pos = [-0.230, -0.530, 1.445]
    eps.size = [0.075, 0.075, 0.075]
    eps.rgba = [1.0, 1.0, 0.0, 0.5]
    eps.group = 1

    # Object (pen) body
    pen_body = spec.worldbody.add_body()
    pen_body.name = "Object"
    pen_body.pos = [-0.230, -0.530, 1.445]
    pen_body.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER
    spec.compiler.degree = False
    pen_body.alt.euler = [0.0, 1.27, 0.0]

    for axis, jname, jtype in [
        ([1, 0, 0], "OBJTx", mujoco.mjtJoint.mjJNT_SLIDE),
        ([0, 1, 0], "OBJTy", mujoco.mjtJoint.mjJNT_SLIDE),
        ([0, 0, 1], "OBJTz", mujoco.mjtJoint.mjJNT_SLIDE),
        ([1, 0, 0], "OBJRx", mujoco.mjtJoint.mjJNT_HINGE),
        ([0, 1, 0], "OBJRy", mujoco.mjtJoint.mjJNT_HINGE),
        ([0, 0, 1], "OBJRz", mujoco.mjtJoint.mjJNT_HINGE),
    ]:
        j = pen_body.add_joint()
        j.name = jname
        j.type = jtype
        j.axis = axis
        j.limited = False
        j.damping = [0.0, 0.0, 0.0]

    pen_geom = pen_body.add_geom()
    pen_geom.name = "pen"
    pen_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    pen_geom.size = [0.015, 0.065, 0.0]
    pen_geom.condim = 4
    pen_geom.contype = 1
    pen_geom.conaffinity = 1
    pen_geom.rgba = [0.6, 0.6, 0.6, 0.6]
    pen_geom.density = 1500.0

    for gname, gtype, gsize, gpos, grgba in [
        (
            "top",
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.017, 0.020, 0.0],
            [0, 0, -0.0455],
            [0, 0.5, 1, 1],
        ),
        (
            "bot",
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.013, 0.002, 0.0],
            [0, 0, 0.067],
            [0, 0.5, 1, 1],
        ),
        (
            "cli",
            mujoco.mjtGeom.mjGEOM_BOX,
            [0.004, 0.006, 0.03],
            [-0.015, 0, -0.0255],
            [0, 0.5, 1, 1],
        ),
    ]:
        pg = pen_body.add_geom()
        pg.name = gname
        pg.type = gtype
        pg.size = gsize
        pg.pos = gpos
        pg.rgba = grgba
        pg.contype = 0
        pg.conaffinity = 0

    s_top = pen_body.add_site()
    s_top.name = "object_top"
    s_top.type = mujoco.mjtGeom.mjGEOM_SPHERE
    s_top.size = [0.005, 0.005, 0.005]
    s_top.pos = [0.0, 0.0, 0.065]
    s_top.rgba = [0.8, 0.2, 0.2, 1.0]

    s_bot = pen_body.add_site()
    s_bot.name = "object_bottom"
    s_bot.type = mujoco.mjtGeom.mjGEOM_SPHERE
    s_bot.size = [0.005, 0.005, 0.005]
    s_bot.pos = [0.0, 0.0, -0.065]
    s_bot.rgba = [0.2, 0.8, 0.2, 1.0]

    # Target body (static)
    tgt_body = spec.worldbody.add_body()
    tgt_body.name = "target"
    tgt_body.pos = [0.0, -0.54, 1.382]

    tgt_geom = tgt_body.add_geom()
    tgt_geom.name = "target"
    tgt_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    tgt_geom.size = [0.015, 0.065, 0.0]
    tgt_geom.condim = 4
    tgt_geom.rgba = [0.6, 0.6, 0.6, 0.3]

    for gname, gtype, gsize, gpos, grgba in [
        (
            "t_top",
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.017, 0.020, 0.0],
            [0, 0, -0.0455],
            [0, 1, 0.5, 1],
        ),
        (
            "t_bot",
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.013, 0.002, 0.0],
            [0, 0, 0.067],
            [0, 1, 0.5, 1],
        ),
        (
            "t_cli",
            mujoco.mjtGeom.mjGEOM_BOX,
            [0.004, 0.006, 0.03],
            [-0.015, 0, -0.0255],
            [0, 1, 0.5, 1],
        ),
    ]:
        tg = tgt_body.add_geom()
        tg.name = gname
        tg.type = gtype
        tg.size = gsize
        tg.pos = gpos
        tg.rgba = grgba
        tg.contype = 0
        tg.conaffinity = 0

    ts_top = tgt_body.add_site()
    ts_top.name = "target_top"
    ts_top.type = mujoco.mjtGeom.mjGEOM_SPHERE
    ts_top.size = [0.005, 0.005, 0.005]
    ts_top.pos = [0.0, 0.0, 0.065]
    ts_top.rgba = [0.8, 0.2, 0.2, 1.0]

    ts_bot = tgt_body.add_site()
    ts_bot.name = "target_bottom"
    ts_bot.type = mujoco.mjtGeom.mjGEOM_SPHERE
    ts_bot.size = [0.005, 0.005, 0.005]
    ts_bot.pos = [0.0, 0.0, -0.065]
    ts_bot.rgba = [0.2, 0.8, 0.2, 1.0]

    return spec


@model_recipe("hand_pen")
def _hand_pen(b: ModelBuilder) -> ModelBuilder:
    """Legacy hand seed + pen object and target built through MjSpec."""
    return _legacy_hand_builder().apply_transform(_add_pen_props)


def _add_sar_props(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add reorientation object, target body, and tracking sites."""
    # eps_ball site
    eps = spec.worldbody.add_site()
    eps.name = "eps_ball"
    eps.type = mujoco.mjtGeom.mjGEOM_SPHERE
    eps.pos = [-0.248, -0.532, 1.445]
    eps.size = [0.08, 0.08, 0.08]
    eps.rgba = [1.0, 1.0, 0.0, 0.0]
    eps.group = 1

    # Object body
    obj_body = spec.worldbody.add_body()
    obj_body.name = "Object"
    obj_body.pos = [-0.248, -0.532, 1.450]
    obj_body.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER
    spec.compiler.degree = False
    obj_body.alt.euler = [0.0, 1.27, 0.0]

    for axis, jname, jtype in [
        ([1, 0, 0], "OBJTx", mujoco.mjtJoint.mjJNT_SLIDE),
        ([0, 1, 0], "OBJTy", mujoco.mjtJoint.mjJNT_SLIDE),
        ([0, 0, 1], "OBJTz", mujoco.mjtJoint.mjJNT_SLIDE),
        ([1, 0, 0], "OBJRx", mujoco.mjtJoint.mjJNT_HINGE),
        ([0, 1, 0], "OBJRy", mujoco.mjtJoint.mjJNT_HINGE),
        ([0, 0, 1], "OBJRz", mujoco.mjtJoint.mjJNT_HINGE),
    ]:
        j = obj_body.add_joint()
        j.name = jname
        j.type = jtype
        j.axis = axis
        j.limited = False
        j.damping = [0.0, 0.0, 0.0]

    obj_geom = obj_body.add_geom()
    obj_geom.name = "obj"
    obj_geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    obj_geom.size = [0.015, 0.015, 0.045]
    obj_geom.condim = 4
    obj_geom.contype = 1
    obj_geom.conaffinity = 1
    obj_geom.rgba = [0.6, 0.6, 0.6, 0.6]
    obj_geom.density = 1500.0

    for gname, gpos in [("top", [0, 0, -0.035]), ("bot", [0, 0, 0.035])]:
        g = obj_body.add_geom()
        g.name = gname
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [0.013, 0.002, 0.0]
        g.pos = gpos
        g.rgba = [0.0, 0.5, 1.0, 0.0]
        g.contype = 0
        g.conaffinity = 0

    s_otop = obj_body.add_site()
    s_otop.name = "object_top"
    s_otop.type = mujoco.mjtGeom.mjGEOM_SPHERE
    s_otop.size = [0.005, 0.005, 0.005]
    s_otop.pos = [0.0, 0.0, 0.065]
    s_otop.rgba = [0.8, 0.2, 0.2, 1.0]

    s_obot = obj_body.add_site()
    s_obot.name = "object_bottom"
    s_obot.type = mujoco.mjtGeom.mjGEOM_SPHERE
    s_obot.size = [0.005, 0.005, 0.005]
    s_obot.pos = [0.0, 0.0, -0.065]
    s_obot.rgba = [0.2, 0.8, 0.2, 1.0]

    # Success site
    success = spec.worldbody.add_site()
    success.name = "success"
    success.type = mujoco.mjtGeom.mjGEOM_SPHERE
    success.pos = [-0.248, -0.54, 1.650]
    success.size = [0.07, 0.07, 0.07]
    success.rgba = [1.0, 1.0, 0.0, 0.1]
    success.group = 1

    # Target body
    tgt_body = spec.worldbody.add_body()
    tgt_body.name = "target"
    tgt_body.pos = [-0.248, -0.54, 1.650]
    tgt_body.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER
    spec.compiler.degree = False
    tgt_body.alt.euler = [0.0, 1.27, 0.0]

    tgt_geom = tgt_body.add_geom()
    tgt_geom.name = "target"
    tgt_geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    tgt_geom.size = [0.015, 0.015, 0.045]
    tgt_geom.condim = 4
    tgt_geom.rgba = [0.6, 0.6, 0.6, 0.6]

    for gname, gpos in [("t_top", [0, 0, -0.035]), ("t_bot", [0, 0, 0.035])]:
        tg = tgt_body.add_geom()
        tg.name = gname
        tg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        tg.size = [0.013, 0.002, 0.0]
        tg.pos = gpos
        tg.rgba = [0.0, 1.0, 0.5, 0.0]
        tg.contype = 0
        tg.conaffinity = 0

    ts_top = tgt_body.add_site()
    ts_top.name = "target_top"
    ts_top.type = mujoco.mjtGeom.mjGEOM_SPHERE
    ts_top.size = [0.005, 0.005, 0.005]
    ts_top.pos = [0.0, 0.0, 0.065]
    ts_top.rgba = [0.8, 0.2, 0.2, 1.0]

    ts_bot = tgt_body.add_site()
    ts_bot.name = "target_bottom"
    ts_bot.type = mujoco.mjtGeom.mjGEOM_SPHERE
    ts_bot.size = [0.005, 0.005, 0.005]
    ts_bot.pos = [0.0, 0.0, -0.065]
    ts_bot.rgba = [0.2, 0.8, 0.2, 1.0]

    return spec


@model_recipe("hand_sar")
def _hand_sar(b: ModelBuilder) -> ModelBuilder:
    """myohand_r + reorientation object + target body for SAR tasks."""
    return _hand_builder().apply_transform(_add_sar_props)


@model_recipe("walk_standard")
def _walk_standard(b: ModelBuilder) -> ModelBuilder:
    """Leg + OSL model for walking tasks."""
    return b.attach_fragment("leg").attach_fragment("osl")


# ---------------------------------------------------------------------------
# MyoChallenge recipes
# ---------------------------------------------------------------------------
# These recipes assemble the scene geometry that ModelBuilder handles natively.
# Items in the table above marked "needs apply_transform" are NOT included here;
# use apply_transform() on the returned builder to add tendons, sites, mirrored
# arms, or other features that require direct MjSpec manipulation.
# ---------------------------------------------------------------------------


def _disable_baoding_finger_collision(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Restore the legacy palm-only collision convention on the composed hand.

    myo_sim#111 added full per-bone finger/forearm collision capsules;
    baoding's balls are calibrated to rest across fingertips relying on the
    older pass-through-finger convention, so full collision diverges the
    trajectory replay baseline by ~5cm from step 0 (see
    https://github.com/MyoHub/myo_sim/issues/127). Delegates to myo_sim's
    own ``disable_finger_collision`` (added on ``feature/torso-build-api``
    alongside this fix) so myosuite doesn't duplicate the geom basename
    list.
    """
    from myo_sim.build.compose import disable_finger_collision  # type: ignore[import-untyped]

    disable_finger_collision(spec)
    # myo_sim's name-based geom lookup misses one unnamed fingernail-ellipsoid
    # geom on distph2_r (class "myohand_coll", contype=1/conaffinity=0) that
    # would otherwise still register contacts with the balls (default
    # conaffinity=1). Disable any remaining unnamed collision geom too.
    for g in spec.geoms:
        if g.name == "" and (g.contype != 0 or g.conaffinity != 0):
            g.contype = 0
            g.conaffinity = 0
    return spec


def _apply_baoding_inertia_floor(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Restore the legacy ``boundinertia``-clamped inertia floor on small hand bones.

    Originally root-caused and worked around locally (~30 bones, a
    near-singular mass matrix around the thumb), then filed upstream as
    https://github.com/MyoHub/myo_sim/issues/128. myo_sim's fix — an
    ``inertia_floor`` parameter on ``apply_inertia_floor()``/``build_spec()``
    /``load_spec()`` on ``feature/torso-build-api`` — also fixed a related
    bug found in the same investigation: a jointless scaffold body
    (``clavphant``, and siblings using the same "tiny mass + huge inertia"
    joint-constrained idiom) left with a physically-absurd fused inertia
    after the arm-to-hand pruning step removes its stabilizing joints.
    ``load_right_hand_from_arm_spec()`` doesn't expose the new parameter
    directly, so apply it here via myo_sim's standalone utility instead.
    """
    from myo_sim.build.compose import apply_inertia_floor  # type: ignore[import-untyped]

    return apply_inertia_floor(spec, 0.0001)


_LEGACY_SKIN_SOLIMP = [0.8, 0.8, 0.01, 0.5, 2.0]
_BAODING_PALM_SKIN_GEOMS = (
    "1mcskin_r",
    "2mcskin_coll_r",
    "3mcskin_coll_r",
    "4mcski_r",
    "5mcskin_r",
)


def _restore_legacy_skin_solimp(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Restore the legacy ``skin`` default class's softer contact impedance.

    The legacy ``myohand_baoding.xml`` puts the 5 palm/metacarpal-skin
    geoms in a custom ``class="skin"`` with
    ``solimp="0.8 0.8 0.01 0.5 2"`` — softer, wider-transition contact
    than MuJoCo's engine default (``0.9 0.95 0.001 0.5 2``, confirmed via
    the ball geoms, which use the ordinary default and already match).
    myo_sim's composed hand puts the same 5 geoms in its own
    ``myohand_coll`` class, which uses the engine default instead — solref
    /margin/gap/friction/condim/priority all otherwise match exactly.

    This is a real difference in the contact solver's force-vs-penetration
    curve, not a calibration nicety: it was the actual (and, after fixing
    collision geoms + lunate mass, the *only* remaining) source of
    trajectory divergence in a contact-critical task like balancing balls
    in the palm — root-caused by direct geom-attribute diffing after
    prior fixes left post-reset state identical but a zero-action rollout
    still diverging within 1 step.
    """
    for name in _BAODING_PALM_SKIN_GEOMS:
        geom = spec.geom(name)
        if geom is not None:
            geom.solimp = _LEGACY_SKIN_SOLIMP
    return spec


def _restore_lunate_phantom_mass(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Reproduce two mass-only geoms the legacy hand carried on ``lunate``.

    ``myohand_baoding.xml``'s ``WristExtensor_wrap``/``WristFlexor_wrap``
    cylinders (density 1000, ``contype=conaffinity=0``) are not used as a
    wrap surface by any tendon in that file (verified: no
    ``tendon_wrap_objid`` references either name) — they're vestigial
    muscle-routing markers from an earlier authoring pass, superseded by
    ``ExtensorEllipse_ellipsoid_wrap`` et al. But being real geoms with
    density, they still contribute ~14g (~8.6%) to the legacy ``lunate``
    body's mass, which myo_sim's cleaner fragment correctly omits.

    That's enough mass difference on the wrist bone to diverge
    ``test_parity.py``'s frozen baoding trajectory (a contact-critical,
    weight-sensitive task) from step 0, even with matching collision
    geoms, muscle calibration, and tendon routing (confirmed: this was the
    *entire* gap — reproducing just these two geoms makes ``lunate_r``'s
    compiled mass match the legacy value exactly, 0.16375769 kg).

    This function exists purely for byte-for-byte parity with the legacy
    baseline; the phantom mass is not anatomically meaningful.
    """
    lunate = spec.body("lunate_r")
    lunate.add_geom(
        name="WristExtensor_wrap_r",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0007, 0.0, 0.003],
        quat=[0.807649, -0.0333177, 0.588355, 0.0207694],
        size=[0.008, 0.02, 0.0],
        contype=0,
        conaffinity=0,
        density=1000.0,
    )
    lunate.add_geom(
        name="WristFlexor_wrap_r",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, -0.01, -0.004],
        quat=[0.807649, -0.0333177, 0.588355, 0.0207694],
        size=[0.007, 0.02, 0.0],
        contype=0,
        conaffinity=0,
        density=1000.0,
    )
    return spec


def _add_baoding_sites(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add ball1_site/ball2_site (on the balls) and target1_site/target2_site
    (rigidly attached to the palm's "trapezium_r" bone, matching the legacy
    myohand_baoding.xml local offsets exactly — verified against
    myosuite/envs/myo/assets/hand/assets/myohand_body.xml).

    Deliberately omits the two purely-decorative spatial tendons the legacy
    XML draws between each ball and its target ("string" visualisation) —
    baoding.py's reward/observation logic reads site positions directly
    (ball*_site, target*_site) and never references the tendons.
    """
    spec.body("ball1").add_site(
        name="ball1_site", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005]
    )
    spec.body("ball2").add_site(
        name="ball2_site", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005]
    )
    trapezium = spec.body("trapezium_r")
    trapezium.add_site(
        name="target1_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.007],
        rgba=[1.0, 0.8, 0.31, 1.0],
        pos=[-0.03, -0.05, -0.02],
        group=3,
    )
    trapezium.add_site(
        name="target2_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.007],
        rgba=[0.84, 0.59, 0.53, 1.0],
        pos=[0.005, -0.1, -0.02],
        group=3,
    )
    return spec


@model_recipe("challenge_baoding")
def _challenge_baoding(b: ModelBuilder) -> ModelBuilder:  # noqa: ARG001
    """MyoChallenge Baoding: hand + two baoding balls.

    Natively composed: myo_sim-native hand fragment (calibrated to the
    legacy world frame via :func:`_hand_builder`), with finger/forearm
    collision disabled to match the legacy palm-only convention (see
    ``myo_sim.build.compose.disable_finger_collision`` /
    https://github.com/MyoHub/myo_sim/issues/127), ball1 (yellow) and ball2
    (peach) with correct mass (43 g) and torsional-friction condim=4, plus
    ball/target sites (see :func:`_add_baoding_sites`).

    Verified against the legacy myohand_baoding.xml — down to the level of
    per-actuator force/length and per-DOF qacc diffing, not just parameter
    counts — by working through four distinct, individually root-caused
    upstream/legacy discrepancies:

    1. :func:`_disable_baoding_finger_collision` — full per-bone finger/
       forearm collision (myo_sim#111) vs. the legacy palm-only convention
       (https://github.com/MyoHub/myo_sim/issues/127).
    2. :func:`_apply_baoding_inertia_floor` — legacy's compiler
       ``boundinertia=".0001"`` floor clamps ~30 small wrist/finger bones'
       inertia up to a numerically-safe 1e-4 kg*m^2; without it several
       bones (including a jointless scaffold body left with a physically
       absurd fused inertia after arm-to-hand pruning) produce a
       near-singular local mass matrix. Originally worked around with ~30
       lines of local body-by-body patches; now a single call into
       myo_sim's own fix (https://github.com/MyoHub/myo_sim/issues/128,
       landed on ``feature/torso-build-api``).
    3. :func:`_restore_lunate_phantom_mass` — legacy's ``lunate`` carries
       ~14g/8.6% of mass from two tendon-wrap markers no tendon actually
       uses.
    4. :func:`_restore_legacy_skin_solimp` — legacy's palm-skin geoms use a
       softer custom contact-impedance curve than myo_sim's default.

    A 6-wrap-point-position bug (myo_sim#128) that was also part of this
    investigation — e.g. a 2mm offset that shortened the ``RI2`` tendon by
    5.5mm, flipping its resting passive force from -11.9N to -0.07N — was
    fixed directly upstream in myo_sim's asset data rather than patched
    here, since the raw (unpruned) arm fragment already had it wrong.

    These four fixes brought a matched-qpos/qvel, real-action, zero-contact
    single-step ``qacc`` discrepancy down from ~208 to a residual on the
    order of a few units, with every force-generating quantity checked
    (actuator gain/bias/lengthrange, joint damping/armature/frictionloss/
    stiffness/range, tendon wrap-point *counts* and, after myo_sim#128,
    positions, collision-geom local pose/size) now matching exactly or
    near-exactly.

    .. warning::
        **Not yet wired into myoChallengeBaodingP1/P2-v1.** Re-verify
        ``test_parity.py``'s frozen-trajectory replay (atol 1e-6) end to
        end with the upstream myo_sim#128 fixes pulled in before swapping
        the registration's ``model_path`` for this recipe.

    Gaps vs the full challenge XML (cosmetic only, no reward/observation
    impact — see :func:`_add_baoding_sites`):
        - The two decorative ball-to-target spatial tendons.
        - Scene lighting / floor from myosuite_scene.xml.
    """
    return (
        _hand_builder()
        .apply_transform(_disable_baoding_finger_collision)
        .apply_transform(_apply_baoding_inertia_floor)
        .apply_transform(_restore_lunate_phantom_mass)
        .apply_transform(_restore_legacy_skin_solimp)
        .add_free_body(
            "ball1",
            pos=[-0.227, -0.511, 1.452],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.022, 0, 0],
            rgba=[1.0, 0.8, 0.31, 1.0],
            mass=0.043,
            condim=4,
        )
        .add_free_body(
            "ball2",
            pos=[-0.256, -0.552, 1.442],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.022, 0, 0],
            rgba=[0.84, 0.59, 0.53, 1.0],
            mass=0.043,
            condim=4,
        )
        .apply_transform(_add_baoding_sites)
    )


def _tabletennis_left_arm_subtree_names(spec: mujoco.MjSpec) -> set[str]:
    """Body names in the ``myoarm_l_root`` subtree of a ``myotorso_arms`` spec."""
    root = spec.body("myoarm_l_root")
    names: set[str] = set()
    stack = [root]
    while stack:
        body = stack.pop()
        names.add(body.name)
        stack.extend(body.bodies)
    return names


def _strip_tabletennis_left_arm_muscles(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Delete every tendon/actuator touching the left-arm subtree.

    myo_sim's ``myotorso_arms`` composes torso + **real** left and right arms
    (not a runtime mirror) — both muscled. The legacy TableTennis convention
    only muscles the right arm; the left arm is a fully rigid, muscle-free
    kinematic double (see :func:`_immobilize_tabletennis_left_arm`).

    A name-pattern heuristic ("any actuator ending in ``_l``") is not
    sufficient: it misses passive tendons with no actuator at all (e.g. a
    shoulder ligament, ``s_glenohum_ligament_l``, found while root-causing an
    ``immobilize`` failure below). Delete by subtree *membership* instead —
    any tendon whose wrap path touches a site/geom belonging to a body in the
    left-arm subtree, then any actuator that targeted a deleted tendon.

    Verified: right-arm-muscle actuator names come out of ``myotorso_arms``
    completely unsided (matching the legacy convention exactly, no ``_r``
    suffix needed) and torso muscles stay bilaterally ``_r``/``_l`` suffixed
    on both sides — after this strip, the remaining actuator set is an exact
    275-name match (0 mismatches) against the legacy
    ``myoarm_tabletennis.xml``'s post-preprocessing actuator list (63 unsided
    right-arm muscles + 210 bilateral torso muscles + the 2
    ``pelvis_x``/``pelvis_y`` actuators added in
    :func:`_add_tabletennis_pelvis_actuators`), and 274/275 gainprm/biasprm
    pairs match exactly (the one exception, ``SUPSP``, is an isolated
    upstream calibration difference, not touched by this function).
    """
    subtree = _tabletennis_left_arm_subtree_names(spec)
    subtree_targets: set[str] = set()
    for site in spec.sites:
        if site.parent.name in subtree:
            subtree_targets.add(site.name)
    for geom in spec.geoms:
        if geom.parent.name in subtree:
            subtree_targets.add(geom.name)

    tendons_to_delete: set[str] = set()
    for tendon in spec.tendons:
        for i in range(len(tendon.path)):
            target = tendon.path[i].target
            name = getattr(target, "name", None)
            if name in subtree_targets:
                tendons_to_delete.add(tendon.name)
                break

    for actuator in list(spec.actuators):
        if actuator.target in tendons_to_delete:
            spec.delete(actuator)
    for tendon in list(spec.tendons):
        if tendon.name in tendons_to_delete:
            spec.delete(tendon)
    return spec


def _attach_tabletennis_legs(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Attach a muscle-free (bodies + joints only) leg chain under ``Full Body``.

    Matches the legacy ``myotorso_arm_chain_host.xml`` convention exactly: it
    includes myo_sim's leg *chain* and *assets* but not ``myolegs_tendon.xml``
    /``myolegs_muscle.xml`` — legs are pure scaffolding for this task (always
    immobilised, see :func:`_immobilize_tabletennis_legs`), so no muscles are
    needed. Uses myo_sim's own ``build_child_xml_from_components`` (which
    ``load_legs_spec()`` also calls, just with tendons/muscles included) with
    ``tendons_xml=None, muscles_xml=None``.
    """
    from myo_sim.build.compose import (  # type: ignore[import-untyped]
        LEGS_ASSETS_XML,
        LEGS_CHAIN_XML,
        ROOT,
        find_body,
    )
    from myo_sim.build.utils import (  # type: ignore[import-untyped]
        build_child_xml_from_components,
    )

    full_body = find_body(spec, "Full Body")
    legs_xml = build_child_xml_from_components(
        model_name="myolegs_bones_attach",
        compiler_meshdir=ROOT,
        assets_xml=LEGS_ASSETS_XML,
        tendons_xml=None,
        muscles_xml=None,
        chain_xml=LEGS_CHAIN_XML,
        root_body_name="myolegs_root",
        root_site_name="legs_root_attach",
    )
    legs_spec = mujoco.MjSpec.from_string(legs_xml)
    legs_spec.compiler.balanceinertia = True
    legs_frame = full_body.add_frame(name="legs_attach")
    spec.attach(legs_spec, prefix="", suffix="", frame=legs_frame)
    return spec


def _immobilize_tabletennis_legs_and_left_arm(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Freeze legs and the whole left-arm subtree, matching legacy exactly.

    Legacy's ``preprocess_tabletennis_spec()`` immobilises ``femur_l``/
    ``femur_r`` (legs are passive scaffolding) and — via its mirror-then-copy
    dance — ends up with a left arm that has **zero joints**: the mirrored
    copy has ``recursive_immobilize(spec_copy, temp_model,
    spec_copy.worldbody)`` called on it (freezing literally everything)
    *before* being pruned down to just the visual/collision double. Since
    myo_sim's ``myotorso_arms`` already gives a real (not mirrored) left arm,
    reproduce the same end state directly: freeze ``myoarm_l_root``'s whole
    subtree here instead of the copy-mirror-immobilize dance.
    """
    from myosuite.utils.spec_processing import recursive_immobilize

    temp_model = spec.compile()
    recursive_immobilize(spec, temp_model, spec.body("femur_l"), remove_eqs=True)
    recursive_immobilize(spec, temp_model, spec.body("femur_r"), remove_eqs=True)
    recursive_immobilize(spec, temp_model, spec.body("myoarm_l_root"), remove_eqs=True)
    return spec


# Kabsch-fit rigid transform (torso, humerus_r, radius_r, ulna_r, lunate_r,
# capitate_r, clavicle_r landmarks at qpos0) mapping myo_sim's composed
# ``myotorso_arms`` root onto the legacy ``myoarm_tabletennis.xml``'s
# post-preprocessing world frame. Residuals: torso ~1cm (the dominant one),
# all other landmarks 1.5-2.6mm — consistent with a small upstream anatomical
# revision between myo_sim's current data and the legacy chain (same class of
# discrepancy documented for baoding/hand elsewhere in this module), not a
# frame-fitting error: torso alone fits to ~1e-6 in isolation, and jointly
# optimising the whole landmark set trades a bit of torso precision for much
# better arm alignment overall.
_TABLETENNIS_ROOT_POS = [1.5998620381840716, 0.006932313674672224, 0.9424515202825139]
_TABLETENNIS_ROOT_QUAT = [
    0.7016473940459751,
    -0.00337963649369631,
    -0.0012230606481786134,
    -0.712515274649122,
]

# Legacy ``myoarm_tabletennis.xml`` wraps the chain in
# ``<body name="full_body" pos="1.6 0 0.95" euler="0 0 3.14">``.
_TABLETENNIS_LEGACY_ROOT_YAW = 3.14


def _tabletennis_pelvis_slide_axis(legacy_axis: list[float]) -> list[float]:
    """Re-express a legacy pelvis slide axis in the calibrated root frame.

    ``pelvis_x``/``pelvis_y`` are declared on the root body, so their axes are
    read in that body's frame. :data:`_TABLETENNIS_ROOT_QUAT` yaws the composed
    root ~-91 deg where legacy yaws it 180 deg, so the legacy axis literals
    would drive the actor sideways instead of toward the table.

    Args:
        legacy_axis: Axis as written in ``myoarm_tabletennis.xml``, i.e. in the
            legacy ``full_body`` frame.

    Returns:
        The same world direction expressed in the calibrated root frame, so
        keyframe qpos and actuator ranges transfer verbatim.
    """
    legacy_quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(
        legacy_quat, np.array([0.0, 0.0, 1.0]), _TABLETENNIS_LEGACY_ROOT_YAW
    )
    world = np.empty(3)
    mujoco.mju_rotVecQuat(world, np.asarray(legacy_axis, dtype=np.float64), legacy_quat)
    root_inverse = np.empty(4)
    mujoco.mju_negQuat(
        root_inverse, np.asarray(_TABLETENNIS_ROOT_QUAT, dtype=np.float64)
    )
    axis = np.empty(3)
    mujoco.mju_rotVecQuat(axis, world, root_inverse)
    return axis.tolist()


def _calibrate_tabletennis_root(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Reposition the composed ``Full Body`` root to the legacy world frame."""
    from myo_sim.build.compose import find_body  # type: ignore[import-untyped]

    full_body = find_body(spec, "Full Body")
    full_body.pos = _TABLETENNIS_ROOT_POS
    full_body.quat = _TABLETENNIS_ROOT_QUAT
    return spec


def _add_tabletennis_pelvis_actuators(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add the myosuite-original ``pelvis_x``/``pelvis_y`` joints + actuators.

    Not from myo_sim — copied exactly (damping/armature/kp/ranges) from the
    legacy ``myotorso_arm_chain_host.xml``'s ``pelvis_move`` default class.
    Axes go through :func:`_tabletennis_pelvis_slide_axis` because the
    calibrated root frame differs from legacy's.
    """
    from myo_sim.build.compose import find_body  # type: ignore[import-untyped]

    full_body = find_body(spec, "Full Body")
    full_body.add_joint(
        name="pelvis_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=_tabletennis_pelvis_slide_axis([1.0, 0.0, 0.0]),
        limited=True,
        range=[-1, -0.05],
        damping=1000,
        armature=10,
    )
    full_body.add_joint(
        name="pelvis_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=_tabletennis_pelvis_slide_axis([0.0, 1.0, 0.0]),
        limited=True,
        range=[-1, 1],
        damping=1000,
        armature=10,
    )
    spec.add_actuator(
        name="pelvis_x",
        target="pelvis_x",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[3500.0] + [0.0] * 9,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -3500.0, 0.0] + [0.0] * 7,
        ctrllimited=True,
        ctrlrange=[-1, 0.05],
    )
    spec.add_actuator(
        name="pelvis_y",
        target="pelvis_y",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[3500.0] + [0.0] * 9,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -3500.0, 0.0] + [0.0] * 7,
        ctrllimited=True,
        ctrlrange=[-1, 1],
    )
    return spec


def _add_tabletennis_contacts(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Strip body-body contacts under ``Full Body`` (radius-related excluded).

    Matches legacy's ``recursive_remove_contacts(full_body, return_condition=
    lambda b: "radius" in b.name)`` step exactly.
    """
    from myo_sim.build.compose import find_body  # type: ignore[import-untyped]

    from myosuite.utils.spec_processing import recursive_remove_contacts

    full_body = find_body(spec, "Full Body")
    recursive_remove_contacts(full_body, return_condition=lambda b: "radius" in b.name)
    return spec


def _match_tabletennis_compiler(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Apply legacy ``myoarm_tabletennis.xml``'s compiler and visual globals.

    Without ``boundinertia`` the ping-pong ball keeps its authored 7.2e-7
    rotational inertia instead of the 1e-4 floor the legacy model runs with,
    which changes how much spin a paddle hit imparts. The offscreen buffer
    matches legacy's ``<global offwidth="1280" offheight="1080"/>``.
    """
    spec.compiler.boundmass = 0.001
    spec.compiler.boundinertia = 0.0001
    spec.compiler.balanceinertia = True
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 1080
    return spec


def _spec_euler_rad(spec: mujoco.MjSpec, *radians: float) -> list[float]:
    """Return Euler angles in the spec's ``compiler.degree`` unit.

    Legacy ``myoarm_tabletennis.xml`` uses ``angle="radian"`` (e.g.
    ``euler="1.57 0 0"``). myo_sim composed specs compile in degrees, so
    passing those radian literals through ``add_geom(euler=...)`` rotates
    the table by ~1.57° instead of 90° and stands the visual mesh on edge.
    """
    if spec.compiler.degree:
        return [math.degrees(angle) for angle in radians]
    return list(radians)


# Legacy ``myoarm_tabletennis.xml``'s ``class="collision"`` default. ``mass=0``
# is load-bearing: without it MuJoCo derives mass from the geom density, which
# put the ping-pong ball at 5 kg and the paddle at 1.5 kg.
_TABLETENNIS_COLLISION_GEOM: dict[str, object] = {
    "group": 4,
    "condim": 3,
    "contype": 1,
    "conaffinity": 1,
    "solref": [0.002, 1.0],
    "solimp": [0.95, 0.95, 0.01, 0.5, 2.0],
    "mass": 0.0,
}


def _add_tabletennis_furniture(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add table/net/paddle/ball furniture, cameras, lights, and sensors.

    Reproduces ``myoarm_tabletennis.xml``'s furniture worldbody exactly
    (meshes/textures/materials/geoms/sites/sensors), since ``ModelBuilder``'s
    generic ``add_mesh_body``/``add_free_body`` helpers don't support the
    custom inertial/extra-collision-geom/site/freejoint combination the
    paddle and ball need.

    Mesh/texture ``file=`` paths are given relative to ``spec.compiler.
    meshdir`` (inherited from myo_sim's composed spec, so absolute paths
    under myosuite's own ``assets/`` need an explicit relative conversion).
    """
    import os

    wb = spec.worldbody
    meshdir = spec.compiler.meshdir

    def _rel(path: Path) -> str:
        return os.path.relpath(str(path), meshdir)

    # myo_sim's composed spec names its ground-plane geom "floor" and the
    # right-hand grasp site "S_grasp_r"; TableTennisEnv looks them up as
    # "ground"/"S_grasp" (legacy XML, unsided-right-arm convention).
    ground = spec.geom("floor")
    ground.name = "ground"
    spec.site("S_grasp_r").name = "S_grasp"
    # myo_sim's scene sinks its plane 0.4 m; legacy uses myosuite_quad.xml's
    # plane at z=0, which both the table and the actor's feet rest on.
    ground.pos = [0.0, 0.0, 0.0]
    ground.size = [6.0, 6.0, 0.1]

    wb.add_camera(
        name="default",
        pos=[-0.367, 0.451, 1.483],
        xyaxes=[-0.033, -0.999, 0.000, 0.207, -0.007, 0.978],
    )
    wb.add_camera(
        name="close_up",
        pos=[-0.031, -6.315, 0.984],
        xyaxes=[1.000, -0.009, -0.000, -0.001, -0.099, 0.995],
    )
    wb.add_light(
        name="glow",
        pos=[0, 0, 1.5],
        type=mujoco.mjtLightType.mjLIGHT_POINT,
        castshadow=1,
        diffuse=[0.7, 0.7, 0.7],
        specular=[0.5, 0.5, 0.5],
        ambient=[0.52, 0.52, 0.52],
    )

    spec.add_mesh(
        name="tabletennis_table", file=_rel(_ASSETS / "tabletennis_table.obj")
    )
    spec.add_mesh(
        name="tabletennis_net_mesh", file=_rel(_ASSETS / "tabletennis_net.obj")
    )
    spec.add_texture(
        name="tabletennis_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=_rel(_ASSETS / "tabletennis.png"),
    )
    tt_mat = spec.add_material(name="tabletennis_mat", specular=0.2, shininess=0.4)
    tt_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tabletennis_tex"

    paddle_mesh = spec.add_mesh(name="paddle_mesh", file=_rel(_ASSETS / "paddle.obj"))
    # Exact mesh inertia re-frames the geom onto principal axes and stands
    # paddle.obj (long in Z) as a stop-sign. Legacy keeps authored axes.
    paddle_mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_LEGACY
    spec.add_texture(
        name="paddle_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=_rel(_ASSETS / "paddle_1k.png"),
    )
    paddle_mat = spec.add_material(name="paddle_mat", specular=0.2, shininess=0.4)
    paddle_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "paddle_tex"

    table = wb.add_body(name="tabletennis_table")
    table.add_geom(
        name="coll_own_half",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.685, 0.76, 0.795],
        pos=[0.685, 0.04, 0],
        rgba=[0, 0, 0, 0],
        **_TABLETENNIS_COLLISION_GEOM,
    )
    table.add_geom(
        name="coll_opponent_half",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.685, 0.76, 0.796],
        pos=[-0.685, 0.04, 0],
        rgba=[0, 0, 0, 0],
        **_TABLETENNIS_COLLISION_GEOM,
    )
    table.add_geom(
        name="coll_net",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.005, 0.9125, 0.1525],
        pos=[0, 0.04, 0.795],
        rgba=[0, 0, 0, 0],
        **_TABLETENNIS_COLLISION_GEOM,
    )
    table.add_geom(
        name="mesh_tabletennis_table",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="tabletennis_table",
        material="tabletennis_mat",
        rgba=[1, 1, 1, 1],
        pos=[0, 0, 0],
        euler=_spec_euler_rad(spec, 1.57, 0.0, 0.0),
        contype=0,
        conaffinity=0,
    )
    table.add_geom(
        name="mesh_tabletennis_net",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="tabletennis_net_mesh",
        material="tabletennis_mat",
        rgba=[1, 1, 1, 1],
        pos=[0, 0, 0],
        euler=_spec_euler_rad(spec, 1.57, 0.0, 0.0),
        contype=0,
        conaffinity=0,
    )

    paddle = wb.add_body(
        name="paddle",
        pos=[1.8, 0.5, 1.13],
        euler=_spec_euler_rad(spec, -0.3, 1.57, 0.0),
    )
    paddle.mass = 0.15
    paddle.inertia = [0.001, 0.001, 0.001]
    # An unset ipos/iquat under explicitinertial resolves to the body's own
    # pos/quat, not the frame origin.
    paddle.ipos = [0.0, 0.0, 0.0]
    paddle.iquat = [1.0, 0.0, 0.0, 0.0]
    paddle.explicitinertial = True
    paddle.add_freejoint(name="paddle_freejoint")
    paddle.add_geom(
        name="paddle",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="paddle_mesh",
        material="paddle_mat",
        rgba=[1, 1, 1, 1],
        pos=[0, 0, 0],
        euler=[0, 0, 0],
        contype=0,
        conaffinity=0,
    )
    paddle.add_site(name="paddle", pos=[-0.06, 0.0, 0], group=4)
    paddle.add_geom(
        name="pad",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.093, 0.020, 0],
        pos=[-0.07, 0, 0],
        **_TABLETENNIS_COLLISION_GEOM,
    )
    paddle.add_geom(
        name="handle",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        euler=_spec_euler_rad(spec, 0.0, 1.57, 0.0),
        size=[0.016, 0.051, 0],
        pos=[0.04, 0, 0],
        **_TABLETENNIS_COLLISION_GEOM,
    )

    pingpong = wb.add_body(name="pingpong", pos=[0.95, 0.0, 1.252])
    pingpong.mass = 2.7e-3
    pingpong.inertia = [0.00000072, 0.00000072, 0.00000072]
    pingpong.ipos = [0.0, 0.0, 0.0]
    pingpong.iquat = [1.0, 0.0, 0.0, 0.0]
    pingpong.explicitinertial = True
    pingpong.add_freejoint(name="pingpong_freejoint")
    pingpong.add_site(name="pingpong", pos=[0, 0, 0], group=3)
    pingpong.add_geom(
        name="pingpong",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0, 0],
        typeinertia=mujoco.mjtGeomInertia.mjINERTIA_SHELL,
        fluid_coefs=[0.235, 0.25, 0.0, 1.0, 1.0],
        rgba=[0.98, 0.70, 0.015, 1],
        priority=2,
        fluid_ellipsoid=1,
        **{
            **_TABLETENNIS_COLLISION_GEOM,
            "group": 1,
            "solimp": [0.9, 0.95, 0.001, 0.5, 2],
            "solref": [-80000, -1],
        },
    )

    spec.add_sensor(
        name="pingpong_vel_sensor",
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="pingpong",
    )
    spec.add_sensor(
        name="paddle_vel_sensor",
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="paddle",
    )
    return spec


# Legacy "default" keyframe qpos, remapped from the post-preprocessing
# legacy model onto this recipe's joint set by NAME (not raw index — the
# joint order differs), stripping "_r" where needed for the arm's unsided
# convention. Verified: all 72 joints matched by exact name, 0 missing/
# dimension-mismatched. ``TableTennisEnv``/mjlab both read ``key_qpos[0]``
# for the episode's initial pose, so this keyframe is load-bearing, not
# cosmetic (unlike the legacy XML's "dribble" keyframe, which nothing reads
# and is intentionally not reproduced here).
# The paddle freejoint (qpos 58:65) is the legacy world spawn verbatim; the
# legacy arm pose already reaches it, so the handle lands in ``S_grasp``.
_TABLETENNIS_DEFAULT_KEY_QPOS = [
    -0.4205,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    -0.0971634,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.1368,
    0.14842,
    0.70695,
    -0.7405,
    -0.343225,
    0.805495,
    -0.10284,
    -0.08288,
    -0.730422,
    0.7,
    0.0632,
    0.07503,
    -0.296726,
    0.72266,
    -0.136136,
    0.26707,
    0.353475,
    0.65982,
    -0.151844,
    0.42417,
    0.51843,
    0.919035,
    -0.20944,
    0.259215,
    0.510575,
    0.793355,
    -0.204204,
    0.227795,
    0,
    1.91,
    0.65,
    1.18,
    0.699445,
    -0.105711,
    0.698888,
    -0.105627,
    -1.25,
    -0.5,
    1.2,
    1,
    0,
    0,
    0,
]


def _add_tabletennis_default_key(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add the "default" keyframe (initial pose) TableTennisEnv resets to."""
    spec.add_key(name="default", qpos=_TABLETENNIS_DEFAULT_KEY_QPOS)
    return spec


def _tabletennis_body_spec() -> mujoco.MjSpec:
    """Build the myo_sim-native torso + both-arms + bones-only-legs body.

    Replaces the legacy ``myoarm_tabletennis.xml`` → ``myotorso_arm_chain_
    host.xml`` → ``torso/assets/myotorso_arm_chain.xml`` chain (the last of
    which duplicated myo_sim body/muscle data with tendon/wrap-site names
    manually remapped). ``myo_sim.load_spec("myotorso_arms")`` already
    composes torso + **real, independently-defined** left and right arms
    (myo_sim's own build pipeline mirrors internally), which eliminates the
    entire runtime mirror-copy-and-hardcoded-correction-quaternion dance the
    legacy ``preprocess_tabletennis_spec()`` used to need.
    """
    import myo_sim  # type: ignore[import-untyped]

    spec = myo_sim.load_spec("myotorso_arms")
    _strip_tabletennis_left_arm_muscles(spec)
    _attach_tabletennis_legs(spec)
    _immobilize_tabletennis_legs_and_left_arm(spec)
    _calibrate_tabletennis_root(spec)
    _add_tabletennis_contacts(spec)
    _add_tabletennis_pelvis_actuators(spec)
    _match_tabletennis_compiler(spec)
    return spec


@model_recipe("challenge_tabletennis")
def _challenge_tabletennis(b: ModelBuilder) -> ModelBuilder:  # noqa: ARG001
    """MyoChallenge TableTennis: torso + both arms + legs + table/net/paddle/ball.

    Natively composed via :func:`_tabletennis_body_spec` (myo_sim's
    ``myotorso_arms`` — real bilateral anatomy, no runtime mirroring) +
    :func:`_add_tabletennis_furniture` (table, net, paddle, ping-pong ball,
    cameras, lights, velocimeter sensors).

    Verified against the legacy ``myoarm_tabletennis.xml`` (post its own
    ``preprocess_tabletennis_spec()``, for a fair post-processing-equivalent
    comparison):

    - ``nq``: 72 == 72 (exact).
    - ``nu``: 275 == 275 (exact), and the actuator **name set** matches
      exactly — 0 missing, 0 extra, no naming-convention normalisation
      needed (myo_sim's right-arm muscles come out unsided already, matching
      legacy verbatim; torso muscles are bilaterally ``_r``/``_l`` suffixed
      on both sides in both models).
    - ``gainprm``/``biasprm``: 274/275 exact matches. The one exception
      (``SUPSP``, a rotator-cuff muscle) has a real calibration difference
      upstream in myo_sim's current data vs. the legacy chain — not
      something this recipe can or should paper over.
    - Root/landmark site positions: Kabsch-fit (see
      :data:`_TABLETENNIS_ROOT_POS`/:data:`_TABLETENNIS_ROOT_QUAT`) to a
      few-mm residual (torso ~1cm, humerus/radius/ulna/lunate/capitate/
      clavicle 1.5-2.6mm) — consistent with a small upstream anatomical
      revision, not a frame-fitting bug.
    - ``nbody``: 107 vs. legacy's 104 (+3) — myo_sim's current fragment
      pipeline adds a few intermediate scaffold/attachment bodies (e.g.
      ``cervical_spine``, ``Arm_attachment``) that the older single-file
      legacy chain didn't have as separate bodies; same pattern documented
      for other body-count deltas throughout this module.

    There is no frozen trajectory-replay parity baseline for this task
    (unlike e.g. baoding) — physical plausibility (finite qpos/qvel over a
    stepped rollout) and the numeric checks above are the bar, not bit-exact
    replay.
    """
    return (
        b.attach_spec(_tabletennis_body_spec(), name="tabletennis_body")
        .apply_transform(_add_tabletennis_furniture)
        .apply_transform(_add_tabletennis_default_key)
    )


@model_recipe("boxing_heavybag")
def _boxing_heavybag(b: ModelBuilder) -> ModelBuilder:
    """Boxing heavy-bag scene: right arm + heavy bag with head/body scoring zones.

    Natively composed: right arm fragment (shoulder + elbow + hand), heavy bag
    free-floating cylinder (30 kg, condim=4 for realistic friction).

    Needs apply_transform:
        - ``fist_r`` site on the proximal phalanx of the middle finger
        - ``bag_head_zone`` site (upper bag — head-level target)
        - ``bag_body_zone`` site (middle bag — torso-level target)

    The heavy bag position (y=-0.7, z=1.1) places it directly in front of
    the extended arm.  Adjust via apply_transform if needed.

    Example::

        model, spec = build_from_recipe("boxing_heavybag")
    """

    def _add_boxing_sites(spec: mujoco.MjSpec) -> mujoco.MjSpec:
        fist_body = spec.body("proxph3")
        fist_site = fist_body.add_site()
        fist_site.name = "fist_r"
        fist_site.pos = [0.0, 0.0, 0.0]
        fist_site.size = [0.015, 0.0, 0.0]

        bag_body = spec.body("heavy_bag")
        head_site = bag_body.add_site()
        head_site.name = "bag_head_zone"
        head_site.pos = [0.0, 0.0, 0.15]

        body_site = bag_body.add_site()
        body_site.name = "bag_body_zone"
        body_site.pos = [0.0, 0.0, 0.0]

        # Gravity compensation: bag hangs stationary until hit.
        # A real heavy bag hangs from a chain; gravcomp=1.0 models this
        # implicitly without needing explicit chain geometry.
        bag_body.gravcomp = 1.0

        # Enable collisions: add_free_body leaves contype/conaffinity at 0.
        # Set contype=1, conaffinity=1 so the bag collides with hand geoms
        # (hand capsules have contype=1) and with the floor (conaffinity=1).
        bag_geom = spec.geom("heavy_bag")
        bag_geom.contype = 1
        bag_geom.conaffinity = 1

        return spec

    return (
        b.attach_fragment("arm")
        .add_free_body(
            "heavy_bag",
            pos=[-0.45, -0.3, 1.1],
            geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            geom_size=[0.15, 0.35, 0.0],
            rgba=[0.55, 0.27, 0.07, 1.0],
            mass=30.0,
            condim=4,
        )
        .apply_transform(_add_boxing_sites)
    )


@model_recipe("challenge_saber_p0")
def _challenge_saber_p0(b: ModelBuilder) -> ModelBuilder:
    """CPU saber prototype: arm + dual saber sites + two free targets.

    This first-step prototype reuses the right-arm fragment and defines two
    saber contact points on the hand so task logic can expose left/right-like
    hit channels without depending on the PR-#73 bimanual fragment.
    """

    def _add_saber_sites(spec: mujoco.MjSpec) -> mujoco.MjSpec:
        hand_body = spec.body("proxph3")

        left_site = hand_body.add_site()
        left_site.name = "left_saber_tip"
        left_site.pos = [0.0, 0.02, 0.10]
        left_site.size = [0.01, 0.0, 0.0]

        right_site = hand_body.add_site()
        right_site.name = "right_saber_tip"
        right_site.pos = [0.0, -0.02, 0.10]
        right_site.size = [0.01, 0.0, 0.0]

        target_a_body = spec.body("saber_target_a")
        target_a_site = target_a_body.add_site()
        target_a_site.name = "saber_target_a_site"
        target_a_site.pos = [0.0, 0.0, 0.0]
        target_a_site.size = [0.02, 0.0, 0.0]

        target_b_body = spec.body("saber_target_b")
        target_b_site = target_b_body.add_site()
        target_b_site.name = "saber_target_b_site"
        target_b_site.pos = [0.0, 0.0, 0.0]
        target_b_site.size = [0.02, 0.0, 0.0]

        for geom_name in ("saber_target_a", "saber_target_b"):
            geom = spec.geom(geom_name)
            geom.contype = 1
            geom.conaffinity = 1
        return spec

    return (
        b.attach_fragment("arm")
        .add_free_body(
            "saber_target_a",
            pos=[-0.25, -0.35, 1.15],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.03, 0.0, 0.0],
            rgba=[0.1, 0.7, 0.9, 0.9],
            mass=0.05,
            condim=4,
        )
        .add_free_body(
            "saber_target_b",
            pos=[-0.15, -0.45, 1.05],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.03, 0.0, 0.0],
            rgba=[0.9, 0.3, 0.3, 0.9],
            mass=0.05,
            condim=4,
        )
        .apply_transform(_add_saber_sites)
    )


@model_recipe("challenge_relocate")
def _challenge_relocate(b: ModelBuilder) -> ModelBuilder:
    """MyoChallenge Relocate: arm + a free-floating object to manipulate.

    Natively composed: arm fragment + cube object (mass 50–300 g range;
    recipe uses the nominal 100 g, condim 4).

    Gaps vs the full challenge XML:
        - Static table geometry (box) — a mocap/static body not a freejoint
        - Target bin (4-wall mocap body) — need apply_transform
        - Slide joints on object (original XML uses 6 separate slide/hinge
          joints rather than a freejoint) — affects joint indexing
        - Human body mesh (human_lowpoly_norighthand.stl)

    The box here uses a freejoint for simplicity; replicate the original
    six-DOF joint setup with apply_transform if exact parity is needed.
    """
    return b.attach_fragment("arm").add_free_body(
        "object",
        pos=[0.0, -0.25, 0.95],
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
        geom_size=[0.0284, 0.0284, 0.0284],
        rgba=[0.5, 0.2, 0.7, 1.0],
        mass=0.1,
        condim=4,
    )


# ---------------------------------------------------------------------------
# MuscleMimic recipes
# ---------------------------------------------------------------------------


def _musclemimic_build(name: str) -> tuple:
    """Delegate to the appropriate MuscleMimic compile function.

    Args:
        name: Recipe name (e.g. ``"musclemimic_fullbody"``).

    Returns:
        Tuple of (MjModel, MjSpec).
    """
    if name == "musclemimic_fullbody":
        from myosuite.integrations.musclemimic.fullbody_model import (
            compile_mimic_fullbody_mjmodel,
            default_mimic_fullbody_config,
        )

        model, spec, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
        return model, spec
    if name == "musclemimic_bimanual":
        from myosuite.integrations.musclemimic.bimanual_model import (
            compile_mimic_bimanual_mjmodel,
            default_mimic_config,
        )

        model, spec, _ = compile_mimic_bimanual_mjmodel(default_mimic_config())
        return model, spec
    if name == "musclemimic_bimanual_fingers":
        from myosuite.integrations.musclemimic.bimanual_model import (
            compile_mimic_bimanual_mjmodel,
            default_mimic_config,
        )

        cfg = default_mimic_config()
        cfg.disable_fingers = False
        model, spec, _ = compile_mimic_bimanual_mjmodel(cfg)
        return model, spec
    if name == "musclemimic_myotorso_bimanual":
        from myosuite.integrations.musclemimic.myotorso_bimanual_model import (
            compile_myotorso_bimanual_mimic_mjmodel,
            default_myotorso_bimanual_mimic_config,
        )

        model, spec, _ = compile_myotorso_bimanual_mimic_mjmodel(
            default_myotorso_bimanual_mimic_config()
        )
        return model, spec
    if name == "musclemimic_myotorso_bimanual_fingers":
        from myosuite.integrations.musclemimic.myotorso_bimanual_model import (
            compile_myotorso_bimanual_mimic_mjmodel,
            default_myotorso_bimanual_mimic_config,
        )

        cfg = default_myotorso_bimanual_mimic_config()
        cfg.disable_fingers = False
        model, spec, _ = compile_myotorso_bimanual_mimic_mjmodel(cfg)
        return model, spec
    raise KeyError(f"Unknown MuscleMimic recipe: {name!r}")


_MUSCLEMIMIC_NAMES = (
    "musclemimic_fullbody",
    "musclemimic_bimanual",
    "musclemimic_bimanual_fingers",
    "musclemimic_myotorso_bimanual",
    "musclemimic_myotorso_bimanual_fingers",
)

# Register each MuscleMimic name as a recipe whose builder raises immediately;
# _build_model_from_task_model in modular_env.py is updated to call
# build_from_recipe() which dispatches to _musclemimic_build via the override.
for _mm_name in _MUSCLEMIMIC_NAMES:
    _name = _mm_name  # capture loop variable

    def _make_mm_recipe(n: str):  # type: ignore[no-untyped-def]
        @model_recipe(n)
        def _mm_recipe(b: ModelBuilder) -> ModelBuilder:  # noqa: ARG001
            # This builder function is never called directly.
            # modular_env._build_model_from_task_model detects musclemimic names
            # and delegates to _musclemimic_build. This registration exists so
            # that list_recipes() and get_recipe() are aware of these names.
            raise NotImplementedError(
                f"MuscleMimic recipe {n!r} cannot be built via ModelBuilder; "
                "call myosuite.core.model_recipes._musclemimic_build(name) directly."
            )

    _make_mm_recipe(_name)
