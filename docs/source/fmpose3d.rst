Motion from video (FMPose3D)
============================

`FMPose3D <https://github.com/AdaptiveMotorControlLab/FMPose3D>`_ estimates the 17
Human3.6M joints of a person in 3D from ordinary video frames.
:func:`myosuite.integrations.fmpose3d.retarget_fmpose3d` fits MyoLeg or the
MuscleMimic MyoFullBody to those joints and returns a
:class:`~myosuite.core.trajectory_io.MotionClip`.

FMPose3D runs in an environment of its own. It needs ``torch<2.11`` and Python
3.10 to 3.12, while ``myosuite[mjlab]`` needs ``torch>=2.13``, so it is not a
MyoSuite dependency. Save its output there and load the file in MyoSuite:

.. code-block:: bash

    ffmpeg -i walk.mp4 frames/%05d.png   # FMPose3D reads images, not videos

.. code-block:: python

    # With fmpose3d installed (pip install fmpose3d); weights download on first use.
    import numpy as np
    from fmpose3d import FMPose3DInference

    result = FMPose3DInference().predict("frames/")
    np.savez("walk.npz", poses_3d=result.poses_3d,
             valid_frames_mask=result.valid_frames_mask, fps=30)

.. code-block:: python

    # In MyoSuite
    import mujoco
    from myosuite.integrations.fmpose3d import retarget_fmpose3d

    spec = mujoco.MjSpec.from_file("myolegs_with_torso.xml")
    model, clip = retarget_fmpose3d(spec, "walk.npz")
    # A subject model first: any form scale_model accepts.
    model, clip = retarget_fmpose3d(spec, "walk.npz", scales="subject_scale.xml")

``poses`` can also be FMPose3D's ``Pose3DResult`` itself, a ``(T, 17, 3)`` array, or
the ``pose3d`` file its demo writes for each frame.

How the poses are fitted
------------------------

* FMPose3D's ``poses_3d`` are root-relative, in metres, in the camera frame (x
  right, y down, z forward). With ``upright=True`` (the default) the median
  direction from the hip to the thorax joint is turned to the model's own trunk
  direction, which assumes the trunk is upright on average, as in walking. With
  ``upright=False`` the camera is taken to be level. The first valid frame is then
  turned to face the way the model faces.
* Each frame is rebuilt with the model's segment lengths along the estimated
  segment directions. One camera fixes directions far better than lengths, and
  Human3.6M's hip joints sit about 0.13 m either side of the pelvis, wider than the
  models' anatomical hip centres. FMPose3D's bone lengths are not used to scale
  the model; pass ``scales`` or ``total_mass`` for a subject.
* :func:`~myosuite.core.retargeting.retarget` fits joint-centre sites added to a copy
  of the model (``h36m_<joint>``): the hip centres and their midpoint, the knees,
  the ankles and a trunk point, and on MyoFullBody the shoulders, elbows and wrists.
  Every frame's lowest sole point is put on the ground (``ground_each_frame``).

What the joints determine
-------------------------

* **Fitted:** hip flexion, adduction and rotation (the last needs a bent knee), knee
  flexion and trunk lean. On MyoFullBody also lumbar bending and rotation, the
  shoulder angles and elbow flexion; humeral rotation needs a bent elbow.
* **Held at the seed pose:** the ankle, subtalar and toe joints and the wrist,
  which move none of the joints (Human3.6M has no foot or hand points); forearm
  rotation, which moves the wrist joint by only 14 mm per radian; and
  MyoFullBody's lumbar flexion, which the joints cannot tell from pelvis tilt. A
  forward lean of the trunk shows as pelvis tilt. ``clip.metadata["fmpose3d"]["held"]``
  lists the held joints.
* **Not available:** horizontal travel. The poses are root-relative, so a walk is
  fitted in place. The spine, nose and head joints are not used: FMPose3D
  synthesises them from other keypoints in its 2D stage.

``myosuite/tests/test_fmpose3d.py`` checks this on synthetic poses of both models
with a camera turned 120 degrees and pitched 15 degrees, hips 5.5 cm wider and every
bone 10% longer. Every joint angle comes back within 1 degree on MyoLeg and 1.5
degrees on MyoFullBody, and the held joints keep the seed pose exactly. With 1 cm
of noise on the joints, hip rotation reaches its limits where the knee is nearly
straight.

Citation
--------

``myosuite.integrations.citations.get_integration_citation("fmpose3d")`` returns the
BibTeX entry of the FMPose3D paper (Wang, Yu and Mathis, CVPR 2026).
