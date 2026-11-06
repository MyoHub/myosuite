Renderer
========

Sport offscreen frames
----------------------

``MJRenderer.render_offscreen(scene="sport")`` adds a navy floor, neutral key
light, orange/cyan side lights and shadows to RGB frames. The preset only edits
the rendered scene; it leaves the shared model, simulation state, depth and
segmentation outputs unchanged. The light colours are scene accents, independent
of agent roles and activation values. Camera settings remain explicit:

.. code-block:: python

   from myosuite.viz.mj_renderer import MJRenderer

   renderer = MJRenderer(env.unwrapped.model, env.unwrapped.data)
   try:
       renderer.set_free_camera_settings(
           distance=3, azimuth=22, elevation=-13, lookat=[0, 0, 1]
       )
       frame = renderer.render_offscreen(width=640, height=360, scene="sport")
   finally:
       renderer.close()

The default remains ``scene="default"``. The preset applies to ``MJRenderer``
offscreen RGB rendering; the interactive viewer and ``env.render()`` keep their
existing appearance. Choose dimensions within the model's offscreen framebuffer.

.. automodule:: myosuite.viz.renderer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: myosuite.viz.mj_renderer
   :members:
   :undoc-members:
   :show-inheritance:
