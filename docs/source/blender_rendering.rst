Render a rollout in Blender
===========================

``scripts/render_blender.py`` exports one CPU episode to animated USD, then
runs Blender in the background to produce a studio render and editable scene.
It accepts an explicit SB3 ``.zip`` or RSL-RL ``.pt`` checkpoint. Use ``--random``
for a demonstration with random actions; incompatible checkpoints fail instead
of falling back to random actions. Tutorial ``tutorials/3.6_Blender_Rendering.ipynb``
walks through an export, the muscle volumes and the render options.

From Python, :class:`myosuite.viz.blender_render.RenderConfig` holds the same settings
and :func:`myosuite.viz.blender_render.render_rollout` runs the export, Blender and the
video; the muscle geometry is in :mod:`myosuite.viz.muscle_tubes`.

.. code-block:: python

   from pathlib import Path
   from myosuite.viz.blender_render import RenderConfig, render_rollout

   render_rollout(RenderConfig(env="myoHandReorient8-v0", output=Path("renders/hand"),
                               seconds=1, preview=True))

.. list-table::
   :widths: 33 33 33

   * - .. image:: images/blender/leg.jpg
     - .. image:: images/blender/hand.jpg
     - .. image:: images/blender/elbow.jpg
   * - ``myoLegWalk-v0``
     - ``myoHandReorient8-v0``
     - ``myoElbowPose1D6MRandom-v0``
   * - .. image:: images/blender/tabletennis.jpg
     - .. image:: images/blender/chasetag.jpg
     -
   * - ``myoChallengeTableTennisP1-v0``
     - ``myoChallengeChaseTagP1-v0`` (``--scene mujoco``)
     -

Random-action frames rendered with Blender 5.2 at 64 samples.

Install Blender separately and add the USD exporter dependencies to your
MyoSuite environment. The sandbox validation used MuJoCo 3.15.0 and Blender
4.5.3 LTS. Tendon tessellation uses MuJoCo's USD exporter internals, so other
versions require verification.

.. code-block:: bash

   pip install 'mujoco[usd]==3.15.0'
   # Optional policy dependencies: MyoSuite[rl] for SB3; rsl-rl-lib for RSL-RL.
   python scripts/render_blender.py \
       --env myoHandReorient8-v0 --random --seconds 1 \
       --output renders/hand --blender /path/to/blender
   python scripts/render_blender.py \
       --env myoLegWalk-v0 --checkpoint /path/to/model_1355.pt \
       --seconds 3 --output renders/walking --blender /path/to/blender

Use ``--preview`` for one frame, ``--export-only`` to export without Blender,
``--fps`` to select the video rate, and ``--resolution WIDTH HEIGHT`` for image
size, and ``--samples`` for Cycles samples per frame (default 64). Use a new
output directory for each run. Recording stops at episode
termination, even when ``--seconds`` requests a longer clip.

The output contains ``video.mp4`` (or ``preview.png``), ``scene.blend``, the
``usd/`` animation and textures, ``render.json`` settings and visibility, and
``reference.npz`` with recorded state for verification, and ``muscles.npz``
with muscle activations. Keep the whole output
folder together: Blender cache paths are relative and images are packed.

Look
----

The default look follows anatomical illustration, after the volumetric muscle
visualiser of `MuSkeMo <https://github.com/PashavanBijlert/MuSkeMo>`_:

- **Volumetric muscles** (``--muscles volumetric``). Each tendon-driven muscle
  becomes a tube along its MuJoCo path: a fusiform belly and tendons at 0.2 of
  the belly radius.

  - The peak cross-section is ``F_max / 1.6 MPa``, a stress fitted to measured
    cross-sections. MuSkeMo's ``F_max / 300 kPa * L0`` volume assumes
    anatomical forces and overestimates MyoSuite's forearm muscles 5-15 times.
  - The belly is as long as the optimal fibre length ``L0``, or longer when its
    cross-section needs it, as in pennate leg muscles. It covers 40-80% of the
    path, towards the origin, so finger muscles end in the forearm with long
    tendons. The peak radius is at most 12% of the belly length.
  - Forearm volumes are 0.5-1.4 times measured adult volumes (Holzbaur et al.
    2007). Large pennate leg muscles match (soleus, vastus lateralis). Strap- and
    fan-shaped leg muscles come out leaner than measured (Handsfield et al. 2014).
  - The volume stays constant during the episode, so a shortening muscle
    thickens.

  ``--muscle-scale`` multiplies all radii. ``--muscles paths`` keeps MuJoCo's
  thin tendon paths instead.
- **Activation colour.** Muscles shade from a relaxed rose to a deep red as
  their activation rises.
- **Materials.** Bones (meshes in a kinematic tree that holds muscle
  attachments) get a waxy ivory shader. Muscles get a glossy, translucent one.
  Task objects keep their MuJoCo colours.
- **Studio** (``--scene studio``). A seamless backdrop (cyclorama) replaces the
  task's floor planes. The scene uses warm key, cool fill and rim area
  lights, and an 85 mm perspective camera. Use ``--scene mujoco`` to keep the
  task's own floor, for example a soccer pitch or an arena.

Collidable world geometry, such as goals and arena fences, stays
visible; visual-only world meshes (room shells and wall props, with
``contype`` and ``conaffinity`` 0) are hidden because the studio replaces them.
Static world geometry is excluded from camera fitting. Camera fitting
samples three poses with a margin; check longer clips for framing before final use.

Muscle volumes are a visual estimate, not a fitted anatomical shape, and
muscle tubes may intersect bones and each other.
Skin/flex models and variable timesteps are rejected. Changes to primitive
geometry during an episode are not supported. Output duration rounds up to
one video frame. A CPU rollout of a GPU-trained policy requires a compatible
CPU task contract.

Sandbox validation covered a published walking checkpoint (0.5 seconds) and
random-action hand motion (0.18 seconds). Saved Blender scenes matched MuJoCo
body-geometry positions at two timestamps to within 6.1e-8 metres; visible
anatomy stayed inside the camera throughout those clips. These checks do not
establish compatibility with every environment or policy success rates.
