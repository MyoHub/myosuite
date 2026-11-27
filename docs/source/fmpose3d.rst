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
  right, y down, z forward). Its root and thorax are replaced by the midpoints of
  its hip and of its shoulder joints, the points the models mark. FMPose3D
  synthesises both in its 2D stage, and on its demo footage the thorax, a
  neck-base point, sat 4 to 5 cm above and 2 to 3 cm in front of the shoulder
  midpoint. Taken as given, it left MyoFullBody's shoulders 3 cm from their
  targets and its elbow flexion 7 degrees from FMPose3D's.
* With ``upright=True`` (the default) the median direction from the hip midpoint
  to the shoulder midpoint is turned to the model's own, which assumes the trunk
  is upright on average, as in walking. With ``upright=False`` the camera is taken
  to be level. The first valid frame is then turned to face the way the model
  faces.
* Each frame is rebuilt with the model's segment lengths along the estimated
  segment directions. One camera fixes directions far better than lengths, and
  Human3.6M's hip joints sit about 0.13 m either side of the pelvis, wider than the
  models' anatomical hip centres. FMPose3D's bone lengths are not used to scale
  the model; pass ``scales`` or ``total_mass`` for a subject.
* :func:`~myosuite.core.retargeting.retarget` fits joint-centre sites added to a copy
  of the model (``h36m_<joint>``): the hip centres and their midpoint, the knees,
  the ankles and the shoulder midpoint (on MyoLeg's torso, where MyoFullBody has
  it), and on MyoFullBody the shoulders, elbows and wrists. Every frame's lowest
  sole point is put on the ground (``ground_each_frame``).
* MyoFullBody's shoulder centres follow the arms through its shoulder girdle, by
  24 mm against the thorax at 1 rad of elevation. It is fitted twice: the second
  time with the trunk and shoulder lengths of the first fit's poses and, with
  ``upright``, with the trunk's own axis upright on average.

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
  fitted in place. The spine, nose and head joints are not used, nor the root and
  thorax as given: FMPose3D synthesises all five from other keypoints in its 2D
  stage.

``myosuite/tests/test_fmpose3d.py`` checks this on synthetic poses of both models
with a camera turned 120 degrees and pitched 15 degrees, the root and thorax where
FMPose3D puts them, hips 5.5 cm wider and every bone 10% longer. Every joint angle
comes back within 0.5 degrees on both models, and the held joints keep the seed
pose exactly. With 1 cm of noise on the joints, hip rotation reaches its limits
where the knee is nearly straight.

On real video
-------------

FMPose3D follows one tracked person per frame, and in a crowd it can switch to a
bystander. Check its 2D keypoints and mark such frames invalid in
``valid_frames_mask``; the fit holds the last pose over them.

On 68 frames of a runner from FMPose3D's own demo, filmed from the side, the fit
follows FMPose3D's thigh and shank directions within 0.3 to 0.9 degrees (median,
95th percentile 1.5 to 3 degrees) and its knee flexion within 0.4 to 1 degree. On
MyoFullBody, elbow flexion follows within 1.0 and 1.2 degrees (95th percentile 2.7
and 7.1). Where FMPose3D's knee flexion passes the models' 120 degree limit, the
fit stops at the limit. These figures say how closely the fit follows FMPose3D,
not how accurate FMPose3D is.

Citation
--------

``myosuite.integrations.citations.get_integration_citation("fmpose3d")`` returns the
BibTeX entry of the FMPose3D paper (Wang, Yu and Mathis, CVPR 2026).
