Athletic Blender rendering
==========================

``render_athletic`` rolls an athletic episode on the CPU plant and renders it
in Blender Cycles with a studio light rig and the running ring. It uses the
same constructor arguments as ``make_athletic``: task, skeleton (``model``),
and tendon variant (``tendons``).

Quick start
-----------

.. code-block:: python

   from myosuite.envs.myo.athletic import render_athletic

   # Reset pose of myofullbody / rigid tendons (myoFullbodySprint-v0).
   render_athletic("sprint", "out/sprint-preview", preview=True)

   # Short clip; requires Blender on PATH.
   render_athletic("sprint", "out/sprint", seconds=2.0, resolution=(1280, 720))

Export the stage without launching Blender (writes ``geoms.obj`` and
``render.json``)::

   render_athletic("sprint", "out/stage", preview=True, export_only=True)

From the shell, once Blender is installed::

   python -c "from myosuite.envs.myo.athletic import render_athletic; \
       render_athletic('sprint', 'out/sprint-preview', preview=True)"

Constructor and env ids
-----------------------

``render_athletic`` resolves the same bundle as ``make_athletic``:

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - ``model``
     - ``tendons``
     - Registered id
   * - ``myofullbody`` (default)
     - ``rigid`` (default)
     - ``myoFullbodySprint-v0``
   * - ``myofullbody``
     - ``elastic``
     - ``myoFullbodySprintElastic-v0``
   * - ``myofullbody_simple``
     - ``rigid``
     - ``myoFullbodySimpleSprint-v0``
   * - ``myosprinter``
     - ``rigid``
     - ``myoSprint-v0``

Elastic ids declare their own physics. Building them raises
``NotImplementedError`` until the series-elastic plugin is applied; use
``tendons="rigid"`` for renders today.

Studio rig and ring
-------------------

The world stays dark (strength 0.035). Five area lights model the body: a
large warm key, a larger cool fill, a narrow rim, a floor bounce, and a hair
light. Lamps are sized from the runner's mesh **before** the oval is added, so
the ring does not wash the figure. The shot frames the rubber ring with a
white line on every lane boundary; the runner stands in the inside lane.

``render.json`` records ``env_id``, ``model``, ``tendons``, camera offsets,
lighting, and per-frame geom poses so a stage can be re-lit offline.

Tests
-----

Lighting contract and stage export (no Blender binary)::

   pytest myosuite/tests/test_athletic_blender.py
