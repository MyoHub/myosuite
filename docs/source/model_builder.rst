ModelBuilder *(experimental)*
==============================

.. warning::

   **ModelBuilder is experimental and does not fully replicate the official
   MyoChallenge environments.**

   The challenge XML files include tendons, sites, mocap bodies, contact
   exclusions, keyframes, and composite multi-geom objects that ModelBuilder
   does not currently support natively.  **For training or evaluating agents
   against a benchmark, always use the registered Gymnasium entry points**
   (e.g. ``gym.make("myoChallengeBaodingP2-v1")``) or instantiate the
   environment class directly.

   ModelBuilder is intended for rapid prototyping, ablation studies, and
   domain randomisation pipelines where partial scenes are sufficient.

``ModelBuilder`` is the programmatic interface for composing MuJoCo MJCF models
in MyoSuite.  It records a **lazy recipe** — fragment attachments, object
placements, and transform functions — and compiles everything into a
``(MjModel, MjSpec)`` pair on the first call to :meth:`build`.
Results are content-hashed and cached, so identical recipes compile only once.

.. contents:: Contents
   :local:
   :depth: 2


Current limitations
-------------------

The table below lists scene features that are **not** natively supported.
Use the :ref:`apply_transform <apply-transform>` escape hatch or load from
XML directly for anything in this list.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Unsupported feature
     - Affected challenge envs
   * - Tendons / spatial constraints
     - Baoding (ball tracking tendons)
   * - Sites (tracking markers, sensor anchors)
     - Baoding, Die, Relocate, TableTennis
   * - Mocap / static kinematic bodies
     - Relocate (bin), Soccer (goal posts), ChaseTag (opponent)
   * - Non-freejoint object kinematics (slide / hinge joints on props)
     - Relocate, Die (6-DOF slide joints)
   * - Multi-geom composite bodies
     - Die (12 capsule edges + 3 box faces)
   * - Cube-map / grid-layout textures
     - Soccer ball, Die faces
   * - Aerodynamic fluid coefficients (``fluidcoef``)
     - TableTennis (ping-pong ball drag)
   * - Mirrored body assembly
     - TableTennis (left arm via spec processing)
   * - Heightfield terrain
     - RunTrack, ChaseTag
   * - Contact filtering / exclusion pairs
     - All locomotion envs
   * - Keyframes (initial pose snapshots)
     - TableTennis, Soccer
   * - Sensors (force, touch, accelerometer, …)
     - Any env with sensor obs

Anything marked above can be added via :meth:`apply_transform` — see the
:ref:`escape hatch <apply-transform>` section below.


Alternatives
------------

Direct XML loading
~~~~~~~~~~~~~~~~~~

For full challenge parity, load the environment's existing XML directly.
This is always the most reliable approach:

.. code-block:: python

   import mujoco

   # load from the challenge XML directly
   model = mujoco.MjModel.from_xml_path(
       "myosuite/envs/myo/assets/hand/myohand_baoding.xml"
   )

Gymnasium entry points (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The safest way to get a fully-configured challenge environment:

.. code-block:: python

   import gymnasium as gym

   env = gym.make("myoChallengeBaodingP2-v1")
   env = gym.make("myoChallengeTableTennisP2-v0")
   env = gym.make("myoChallengeSoccerP2-v0")
   env = gym.make("myoChallengeRelocateP2-v0")
   env = gym.make("myoChallengeDieReorientP2-v0")

These environments load their own XML scene files which include all tendons,
sites, contacts, and other features that ModelBuilder does not cover.

apply_transform escape hatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _apply-transform:

Use ``apply_transform`` to add any unsupported feature to a ModelBuilder scene:

.. code-block:: python

   import mujoco
   from myosuite.core.model_builder import ModelBuilder

   def add_tracking_sites(spec: mujoco.MjSpec) -> mujoco.MjSpec:
       """Add target tracking sites missing from the native API."""
       site = spec.worldbody.add_site(name="target1_site")
       site.pos = [-0.25, -0.5, 1.45]
       site.size = [0.005]
       return spec

   def add_tendons(spec: mujoco.MjSpec) -> mujoco.MjSpec:
       """Link ball sites to target sites via spatial tendons."""
       t = spec.add_tendon()
       t.name = "tendon1"
       t.add_site(site="ball1_site")
       t.add_site(site="target1_site")
       return spec

   model, spec = (
       ModelBuilder()
       .attach_fragment("hand")
       .add_free_body("ball1", pos=[-0.227, -0.511, 1.452],
                      geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      geom_size=[0.022, 0, 0], rgba=[1, 0.8, 0.31, 1],
                      mass=0.043, condim=4)
       .apply_transform(add_tracking_sites)
       .apply_transform(add_tendons)
       .build()
   )


Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``attach_fragment(name, parent)``
     - Attach a body-part XML at the parent body's origin
   * - ``place_fragment(name, pos, quat, parent)``
     - Attach a body-part at an explicit position / orientation
   * - ``add_free_body(name, pos, quat, geom_type, geom_size, rgba, mass, condim)``
     - Add a free-floating primitive prop (ball, box, cylinder, …) at a world position
   * - ``add_mesh_body(name, mesh_file, pos, quat, scale, texture_file, rgba, …)``
     - Add a free-floating body loaded from a mesh file with optional texture
   * - ``apply_transform(fn)``
     - Register an arbitrary ``MjSpec → MjSpec`` transform (escape hatch)
   * - ``set_timestep(dt)``
     - Override the simulation timestep (seconds)
   * - ``apply_sarcopenia(force_scale)``
     - Scale all muscle peak forces (e.g. 0.5 → 50 % of nominal)
   * - ``.build()``
     - Compile and return ``(mujoco.MjModel, mujoco.MjSpec)``


Fragment names
--------------

The following names are recognised out of the box:

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 55

   * - Name
     - DoF
     - Muscles
     - Description
   * - ``"elbow"``
     - 2
     - 6
     - 2-DoF elbow (flexion + supination), 3 flexors + 3 extensors
   * - ``"finger"``
     - 4
     - 5
     - Single finger with DIP, PIP, MCP, abduction
   * - ``"hand"``
     - 23
     - 39
     - Full myoHand — 29 bones, 23 joints
   * - ``"arm"``
     - 27
     - 63
     - Full myoArm — shoulder through fingertips
   * - ``"leg"``
     - 20
     - 80
     - Full myoLeg — hip, knee, ankle, 80-muscle model
   * - ``"osl"``
     - —
     - —
     - Leg + Osseointegrated Prosthetic Limb (OSL) add-on
   * - ``"torso"``
     - 18
     - 210
     - Lumbar spine / torso (210-muscle model)
   * - ``"body"``
     - —
     - —
     - Full-body scaffold

Custom fragment paths (absolute ``*.xml``) can be added via
``myo_sim.FragmentRegistry.register(name, path)`` when the ``myo_sim`` package
is installed.


Positioning objects
-------------------

attach_fragment — default (no offset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attaches the fragment at the parent body's origin.  Use this when the XML
already positions the body correctly, or when you don't need a specific offset:

.. code-block:: python

   from myosuite.core.model_builder import ModelBuilder

   model, spec = ModelBuilder().attach_fragment("elbow").build()

place_fragment — explicit offset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``place_fragment`` when you need the fragment root to appear at a specific
position and/or orientation **relative to the parent body frame**.
The quaternion follows the MJCF convention ``[w, x, y, z]``:

.. code-block:: python

   import numpy as np
   from myosuite.core.model_builder import ModelBuilder

   # Place hand 35 cm below the elbow distal body
   model, spec = (
       ModelBuilder()
       .attach_fragment("elbow")
       .place_fragment("hand", pos=[0, 0, -0.35], parent="elbow_distal")
       .build()
   )

   # Rotate a fragment 90° around the Z axis
   angle = np.pi / 2
   quat_z90 = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]  # [w, x, y, z]
   model, spec = (
       ModelBuilder()
       .place_fragment("elbow", pos=[0, 0, 0], quat=quat_z90)
       .build()
   )

add_free_body — free-floating props
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``add_free_body`` creates a body with a 6-DoF freejoint and a single collision
geom.  Use it for balls, boxes, paddles, obstacles, and any other scene prop
whose position you want to control at runtime.

.. code-block:: python

   import mujoco
   from myosuite.core.model_builder import ModelBuilder

   model, spec = (
       ModelBuilder()
       .attach_fragment("hand")
       .add_free_body(
           "my_ball",
           pos=[0.1, 0.0, 1.3],                 # world-frame position (m)
           geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
           geom_size=[0.025, 0, 0],              # [radius, 0, 0] for sphere
           rgba=[1.0, 0.4, 0.0, 1.0],            # orange
       )
       .build()
   )

Each free body adds **7 dof** to ``model.nq`` (3 translation + 4 quaternion).
Set the initial pose at runtime:

.. code-block:: python

   import mujoco, numpy as np

   data = mujoco.MjData(model)
   jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "my_ball_free")
   qadr = model.jnt_qposadr[jid]
   data.qpos[qadr:qadr+3] = [0.2, 0.0, 1.5]   # new position
   data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]     # identity quaternion
   mujoco.mj_forward(model, data)

**Supported geom types and size conventions:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``geom_type``
     - ``geom_size`` meaning
   * - ``mjGEOM_SPHERE``
     - ``[radius, 0, 0]``
   * - ``mjGEOM_BOX``
     - ``[half-x, half-y, half-z]``
   * - ``mjGEOM_CYLINDER``
     - ``[radius, half-height, 0]``
   * - ``mjGEOM_CAPSULE``
     - ``[radius, half-height, 0]``
   * - ``mjGEOM_ELLIPSOID``
     - ``[semi-x, semi-y, semi-z]``


apply_transform — arbitrary MjSpec edits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For anything not covered by the methods above, pass a callable that receives
and returns ``mujoco.MjSpec``:

.. code-block:: python

   from myosuite.core.model_builder import ModelBuilder

   def _move_target(spec):
       spec.body("target").pos = [0.2, 0.0, 0.8]
       return spec

   model, spec = (
       ModelBuilder()
       .attach_fragment("elbow")
       .apply_transform(_move_target)
       .build()
   )


Challenge environment examples *(partial parity)*
---------------------------------------------------

The examples below show what ModelBuilder assembles natively for each challenge
environment.  Each example notes what is missing vs. the full XML scene.
For the complete scene, use the Gymnasium entry point or load from XML directly.

.. list-table:: Parity summary
   :header-rows: 1
   :widths: 20 40 40

   * - Environment
     - Natively composed
     - Missing (needs ``apply_transform`` or XML)
   * - Baoding
     - hand + 2 balls (mass, condim)
     - tendons, target sites, floor/lighting
   * - TableTennis
     - arm + ball + table/net/paddle meshes
     - mirrored left arm, ``fluidcoef``, contact sites
   * - Relocate
     - arm + box object (mass, condim)
     - table, bin mocap body, slide joints
   * - Soccer
     - leg + sphere ball
     - torso/head, goal meshes, cube-map ball texture
   * - Die Reorient
     - hand fragment only
     - 12-capsule die body, dice texture, slide joints

Baoding balls (``myoChallengeBaodingP2-v1``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hand + two spheres with correct mass (43 g) and torsional-friction contact.
**Missing:** tendons linking ball sites to target sites; target tracking sites;
scene lighting and floor (``myosuite_scene.xml``).

.. code-block:: python

   import mujoco
   from myosuite.core.model_builder import ModelBuilder

   model, spec = (
       ModelBuilder()
       .attach_fragment("hand")
       .add_free_body(
           "ball1",
           pos=[-0.227, -0.511, 1.452],
           geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
           geom_size=[0.022, 0, 0],    # 22 mm radius (nominal)
           rgba=[1.0, 0.8, 0.31, 1.0], # yellow
           mass=0.043,                 # 43 g — matches challenge XML
           condim=4,                   # torsional friction
       )
       .add_free_body(
           "ball2",
           pos=[-0.256, -0.552, 1.442],
           geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
           geom_size=[0.022, 0, 0],
           rgba=[0.84, 0.59, 0.53, 1.0],  # peach
           mass=0.043,
           condim=4,
       )
       .build()
   )
   # model.nq = hand_nq + 7 + 7  (two freejoints)

Or via the named recipe (equivalent):

.. code-block:: python

   from myosuite.core.model_builder import build_from_recipe
   import myosuite.core.model_recipes  # registers recipes

   model, spec = build_from_recipe("challenge_baoding")

Per-episode ball-size randomisation (P2 variant, radius ∈ [0.018, 0.024]):

.. code-block:: python

   import random, mujoco
   from myosuite.core.model_builder import ModelBuilder

   def make_baoding_model(radius: float) -> mujoco.MjModel:
       model, _ = (
           ModelBuilder()
           .attach_fragment("hand")
           .add_free_body("ball1", pos=[-0.227, -0.511, 1.452],
                          geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                          geom_size=[radius, 0, 0], rgba=[1.0, 0.8, 0.31, 1.0],
                          mass=0.043, condim=4)
           .add_free_body("ball2", pos=[-0.256, -0.552, 1.442],
                          geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                          geom_size=[radius, 0, 0], rgba=[0.84, 0.59, 0.53, 1.0],
                          mass=0.043, condim=4)
           .build()
       )
       return model

   r = random.uniform(0.018, 0.024)
   model = make_baoding_model(r)

TableTennis (``myoChallengeTableTennisP2-v0``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Right arm + ping-pong ball + table / net / paddle as mesh bodies with textures.
**Missing:** mirrored left arm (requires spec processing); ball aerodynamic drag
(``fluidcoef``); contact zone sites on own/opponent halves; immobilised legs.

.. code-block:: python

   import mujoco
   from pathlib import Path
   from myosuite.core.model_builder import ModelBuilder

   assets = Path("myosuite/envs/myo/assets")

   model, spec = (
       ModelBuilder()
       .attach_fragment("arm")
       .add_free_body(
           "pingpong",
           pos=[0.0, -0.35, 1.5],
           geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
           geom_size=[0.02, 0, 0],        # 40 mm diameter
           rgba=[0.9, 0.9, 0.9, 1.0],
           mass=0.0027,                   # 2.7 g — matches challenge XML
           condim=4,
       )
       .add_mesh_body(
           "tabletennis_table",
           mesh_file=assets / "tabletennis_table.obj",
           texture_file=assets / "tabletennis.png",
           material_shininess=0.4, material_specular=0.2,
       )
       .add_mesh_body(
           "tabletennis_net",
           mesh_file=assets / "tabletennis_net.obj",
           texture_file=assets / "tabletennis.png",
           material_shininess=0.4, material_specular=0.2,
       )
       .add_mesh_body(
           "paddle",
           mesh_file=assets / "paddle.obj",
           texture_file=assets / "paddle_1k.png",
           pos=[0.0, -0.2, 1.3],
           material_shininess=0.4, material_specular=0.2,
       )
       .build()
   )

To add the missing ball aerodynamics via ``apply_transform``:

.. code-block:: python

   def add_fluidcoef(spec: mujoco.MjSpec) -> mujoco.MjSpec:
       """Add aerodynamic drag matching the challenge XML (fluidcoef)."""
       g = spec.geom("pingpong")
       g.fluidcoef = [0.235, 0.25, 0.0, 1.0, 1.0]
       return spec

   model, spec = (
       ModelBuilder()
       .attach_fragment("arm")
       .add_free_body("pingpong", pos=[0.0, -0.35, 1.5],
                      geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      geom_size=[0.02, 0, 0], rgba=[0.9, 0.9, 0.9, 1.0],
                      mass=0.0027, condim=4)
       .apply_transform(add_fluidcoef)
       .build()
   )

Relocate (``myoChallengeRelocateP2-v0``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Arm + free-floating cube object.
**Missing:** static table geometry; 4-wall target bin (mocap body); the original
XML uses 6 separate slide/hinge joints on the object rather than a freejoint.

.. code-block:: python

   import mujoco
   from myosuite.core.model_builder import ModelBuilder

   model, spec = (
       ModelBuilder()
       .attach_fragment("arm")
       .add_free_body(
           "object",
           pos=[0.0, -0.25, 0.95],
           geom_type=mujoco.mjtGeom.mjGEOM_BOX,
           geom_size=[0.0284, 0.0284, 0.0284],
           rgba=[0.5, 0.2, 0.7, 1.0],
           mass=0.1,      # nominal 100 g (P2 range: 50–300 g)
           condim=4,
       )
       .build()
   )

Soccer (``myoChallengeSoccerP2-v0``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leg fragment + FIFA size-5 ball.
**Missing:** torso / head composite; goal post mesh bodies; cube-map soccer-ball
texture; full scene lighting.

.. code-block:: python

   import mujoco
   from myosuite.core.model_builder import ModelBuilder

   FIFA_RADIUS = 0.117   # size-5 ball, 450 g
   model, spec = (
       ModelBuilder()
       .attach_fragment("leg")
       .add_free_body(
           "soccer_ball",
           pos=[2.0, 0.0, FIFA_RADIUS],
           geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
           geom_size=[FIFA_RADIUS, 0, 0],
           rgba=[0.95, 0.95, 0.95, 1.0],
           mass=0.45,   # 450 g
           condim=4,
       )
       .build()
   )

Die Reorient (``myoChallengeDieReorientP2-v0``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hand fragment only — the die body itself cannot be composed natively.
The die is 12 capsule edges + 3 overlapping boxes with a 3×4 grid-layout
dice texture, which requires ``apply_transform``:

.. code-block:: python

   import mujoco
   from pathlib import Path
   from myosuite.core.model_builder import ModelBuilder

   DICE_PNG = Path("myosuite/envs/myo/assets/hand/dice.png")

   def add_die(spec: mujoco.MjSpec) -> mujoco.MjSpec:
       """Add the 12-capsule + 3-box die body via direct MjSpec manipulation."""
       # Register cube-grid texture (not supported by add_mesh_body)
       tex = spec.add_texture()
       tex.name = "dice"
       tex.file = str(DICE_PNG)
       tex.gridsize = [3, 4]
       tex.gridlayout = "..U.LFRB..D."

       mat = spec.add_material()
       mat.name = "MatDice"
       mat.textures[0] = "dice"
       mat.specular = 0.3
       mat.shininess = 1.0

       body = spec.worldbody.add_body(name="Object")
       body.pos = [-0.240, -0.535, 1.46]
       body.ipos = [0, 0, 0]
       body.imass = 0.108
       body.idiaginertia = [6.48e-5, 6.48e-5, 6.48e-5]

       half = 0.0235
       for e in [  # 12 edges as capsules
           ([ half, -half, -half], [ half,  half, -half]),
           ([-half, -half, -half], [-half,  half, -half]),
           ([-half,  half, -half], [ half,  half, -half]),
           ([-half, -half, -half], [ half, -half, -half]),
           ([ half, -half,  half], [ half,  half,  half]),
           ([-half, -half,  half], [-half,  half,  half]),
           ([-half,  half,  half], [ half,  half,  half]),
           ([-half, -half,  half], [ half, -half,  half]),
           ([ half, -half, -half], [ half, -half,  half]),
           ([ half,  half, -half], [ half,  half,  half]),
           ([-half,  half, -half], [-half,  half,  half]),
           ([-half, -half, -half], [-half, -half,  half]),
       ]:
           g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
           g.fromto = e[0] + e[1]
           g.size = [0.005]
           g.rgba = [1, 1, 1, 1]
           g.group = 2

       for size in ([0.0284, 0.0236, 0.0236],
                    [0.0236, 0.0284, 0.0236],
                    [0.0236, 0.0236, 0.0284]):
           g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
           g.size = size
           g.material = "MatDice"
           g.group = 2

       for ax, nm in [([1,0,0], "OBJTx"), ([0,1,0], "OBJTy"), ([0,0,1], "OBJTz")]:
           j = body.add_joint(name=nm, type=mujoco.mjtJoint.mjJNT_SLIDE)
           j.axis = ax; j.range = [-0.25, 0.25]; j.damping = 0.001
       for ax, nm in [([1,0,0], "OBJRx"), ([0,1,0], "OBJRy"), ([0,0,1], "OBJRz")]:
           j = body.add_joint(name=nm, type=mujoco.mjtJoint.mjJNT_HINGE)
           j.axis = ax; j.limited = False; j.damping = 0.001

       return spec

   model, spec = (
       ModelBuilder()
       .attach_fragment("hand")
       .apply_transform(add_die)
       .build()
   )


Named recipes
-------------

For the most common configurations, pre-built recipes are registered via
:func:`myosuite.core.model_builder.model_recipe` and can be loaded with
:func:`build_from_recipe`:

.. code-block:: python

   from myosuite.core.model_builder import build_from_recipe

   model, spec = build_from_recipe("elbow_standard")
   model, spec = build_from_recipe("elbow_sarcopenia")   # 50 % muscle force
   model, spec = build_from_recipe("hand_standard")
   model, spec = build_from_recipe("full_arm")           # shoulder + elbow + hand
   model, spec = build_from_recipe("walk_standard")      # leg + OSL

Register your own recipe to share it across a project:

.. code-block:: python

   from myosuite.core.model_builder import ModelBuilder, model_recipe

   @model_recipe("my_reach_scene")
   def _my_scene(b: ModelBuilder) -> ModelBuilder:
       return (
           b.attach_fragment("elbow")
            .add_free_body("target", pos=[0.3, 0.0, 0.8],
                           geom_type=__import__("mujoco").mjtGeom.mjGEOM_SPHERE,
                           geom_size=[0.015, 0, 0], rgba=[0, 1, 0, 0.5])
       )

   model, spec = build_from_recipe("my_reach_scene")


Pathological conditions
-----------------------

.. code-block:: python

   from myosuite.core.model_builder import ModelBuilder

   # Sarcopenia: all muscles generate only 60 % of their peak force
   model, spec = (
       ModelBuilder()
       .attach_fragment("hand")
       .apply_sarcopenia(force_scale=0.6)
       .build()
   )

   # Combine: Baoding balls + sarcopenic hand
   import mujoco
   model, spec = (
       ModelBuilder()
       .attach_fragment("hand")
       .apply_sarcopenia(force_scale=0.5)
       .add_free_body("ball1", pos=[-0.227, -0.511, 1.452],
                      geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      geom_size=[0.022, 0, 0], rgba=[1.0, 0.8, 0.0, 1.0])
       .add_free_body("ball2", pos=[-0.256, -0.552, 1.442],
                      geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      geom_size=[0.022, 0, 0], rgba=[0.8, 0.6, 0.4, 1.0])
       .build()
   )


Caching
-------

``build()`` caches the compiled model by a SHA-256 hash of:

* fragment file contents + parent body name
* per-fragment ``pos`` and ``quat``
* per-free-body ``pos``, ``quat``, ``geom_size``, ``rgba``, ``geom_type``, ``mass``, ``condim``
* per-mesh-body mesh file bytes + texture file bytes + ``scale``, ``pos``, ``quat``
* timestep and contact settings
* number of registered transforms

Identical recipes always return the same cached ``(MjModel, MjSpec)`` object.
Different positions always produce different cache entries.


See also
--------

* :doc:`architecture` — three-path design (CPU / MJX / mjlab)
* :doc:`environments` — complete environment listing
* ``myosuite/core/model_builder.py`` — source
* ``myosuite/core/model_recipes.py`` — built-in recipes
* ``myosuite/tests/test_model_builder.py`` — unit tests
* ``myosuite/tests/test_model_builder_scenes.py`` — challenge-environment scene tests
