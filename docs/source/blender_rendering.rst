Render a rollout in Blender
==========================

``scripts/render_blender.py`` exports one CPU episode to animated USD, then
runs Blender in the background to produce a studio render and editable scene.
It accepts an explicit SB3 ``.zip`` or RSL-RL ``.pt`` checkpoint. Use ``--random``
for a demonstration with random actions; incompatible checkpoints fail instead
of falling back to random actions.

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
size. Use a new output directory for each run. Recording stops at episode
termination, even when ``--seconds`` requests a longer clip.

The output contains ``video.mp4`` (or ``preview.png``), ``scene.blend``, the
``usd/`` animation and textures, ``render.json`` settings and visibility, and
``reference.npz`` with recorded state for verification. Keep the whole output
folder together: Blender cache paths are relative and images are packed.

Bones and task geometry retain their imported appearance; tendon paths get a
red material. The studio view hides static world meshes used as scenery and
excludes static world geometry from camera fitting. Camera fitting samples
three poses with a margin; check longer clips for framing before final use.

This is a small rendering bridge, not a skin or anatomical-muscle generator.
Skin/flex models and variable timesteps are rejected. Changes to primitive
geometry during an episode are not supported. Output duration rounds up to
one video frame. A CPU rollout of a GPU-trained policy requires a compatible
CPU task contract.

Sandbox validation covered a published walking checkpoint (0.5 seconds) and
random-action hand motion (0.18 seconds). Saved Blender scenes matched MuJoCo
body-geometry positions at two timestamps to within 6.1e-8 metres; visible
anatomy stayed inside the camera throughout those clips. These checks do not
establish compatibility with every environment or policy success rates.
