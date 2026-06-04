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

from pathlib import Path

import mujoco

from myosuite.core.model_builder import ModelBuilder, model_recipe

# Root of the myosuite package — used to locate asset files.
_ASSETS = Path(__file__).parent.parent / "envs" / "myo" / "assets"


@model_recipe("elbow_standard")
def _elbow_standard(b: ModelBuilder) -> ModelBuilder:
    """Standard 2-DOF elbow model (flexion + supination, 6 muscles)."""
    return b.attach_fragment("elbow")


@model_recipe("elbow_sarcopenia")
def _elbow_sarcopenia(b: ModelBuilder) -> ModelBuilder:
    """Elbow model with sarcopenic (50% force) muscles."""
    return b.attach_fragment("elbow").apply_sarcopenia(force_scale=0.5)


@model_recipe("full_arm")
def _full_arm(b: ModelBuilder) -> ModelBuilder:
    """Full arm: shoulder + elbow + hand (right arm only).

    .. note::
        **myo_sim PR #73 breaking change.** When the submodule is updated to
        include PR #73, right-arm body names gain an ``_r`` suffix
        (e.g. ``thorax`` → ``chest_r``).  The ``parent`` strings below reference
        the pre-PR-#73 names; update them once the submodule is upgraded.
        For the bimanual (left + right) model, use the ``bimanual_arm`` recipe.
    """
    return (
        b.attach_fragment("shoulder")
        .attach_fragment("elbow", parent="shoulder_distal")
        .attach_fragment("hand", parent="elbow_distal")
    )


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


@model_recipe("hand_standard")
def _hand_standard(b: ModelBuilder) -> ModelBuilder:
    """Standard myohand model."""
    return b.attach_fragment("hand")


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


@model_recipe("challenge_baoding")
def _challenge_baoding(b: ModelBuilder) -> ModelBuilder:
    """MyoChallenge Baoding: hand + two baoding balls.

    Natively composed: hand fragment, ball1 (yellow) and ball2 (peach) with
    correct mass (43 g) and torsional-friction condim=4.

    Gaps vs the full challenge XML:
        - Tendons connecting ball sites to target sites (need apply_transform)
        - Target tracking sites (need apply_transform)
        - Scene lighting / floor from myosuite_scene.xml (need apply_transform)

    Example::

        model, spec = build_from_recipe("challenge_baoding")
        # Then add tendons:
        def add_tendons(spec):
            # ... spec.add_tendon() calls ...
            return spec
        model, spec = (
            ModelBuilder()
            .attach_fragment("hand")
            ... (balls) ...
            .apply_transform(add_tendons)
            .build()
        )
    """
    return (
        b.attach_fragment("hand")
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
    )


@model_recipe("challenge_tabletennis")
def _challenge_tabletennis(b: ModelBuilder) -> ModelBuilder:
    """MyoChallenge TableTennis: arm + ping-pong ball + table + net + paddle.

    Natively composed: right arm fragment, ping-pong ball (mass 2.7 g, condim 4),
    and the three mesh props (table, net, paddle) with their textures.

    Gaps vs the full challenge XML:
        - Mirrored left arm (requires spec processing / apply_transform; or use
          the ``arm_left`` fragment once myo_sim PR #73 is merged)
        - Ball fluidcoef (aerodynamic drag) — set via apply_transform on the geom
        - Contact zone sites on own/opponent halves (need apply_transform)
        - Immobilised legs / torso included in the full XML scene

    .. note::
        **myo_sim PR #73 breaking change.** Once the submodule is updated to
        PR #73, right-arm body names gain an ``_r`` suffix.  Parent references
        inside the ``arm`` fragment will change accordingly.

    Example of adding the remaining physics::

        def add_fluidcoef(spec):
            ball_geom = spec.geom("pingpong")
            ball_geom.fluidcoef = [0.235, 0.25, 0.0, 1.0, 1.0]
            return spec

        model, spec = (
            ModelBuilder()
            .attach_fragment("arm")
            .add_free_body("pingpong", ...)
            .apply_transform(add_fluidcoef)
            .build()
        )
    """
    asset_dir = _ASSETS  # mesh/texture files live directly in assets/
    return (
        b.attach_fragment("arm")
        .add_free_body(
            "pingpong",
            pos=[0.0, -0.35, 1.5],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.02, 0, 0],
            rgba=[0.9, 0.9, 0.9, 1.0],
            mass=0.0027,
            condim=4,
        )
        .add_mesh_body(
            "tabletennis_table",
            mesh_file=asset_dir / "tabletennis_table.obj",
            texture_file=asset_dir / "tabletennis.png",
            pos=[0.0, 0.0, 0.0],
            material_shininess=0.4,
            material_specular=0.2,
        )
        .add_mesh_body(
            "tabletennis_net",
            mesh_file=asset_dir / "tabletennis_net.obj",
            texture_file=asset_dir / "tabletennis.png",
            pos=[0.0, 0.0, 0.0],
            material_shininess=0.4,
            material_specular=0.2,
        )
        .add_mesh_body(
            "paddle",
            mesh_file=asset_dir / "paddle.obj",
            texture_file=asset_dir / "paddle_1k.png",
            pos=[0.0, -0.2, 1.3],
            material_shininess=0.4,
            material_specular=0.2,
        )
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
