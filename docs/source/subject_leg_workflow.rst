A complete MyoLeg subject workflow
==================================

``myosuite.integrations.capture.leg_workflow`` runs a registered MyoLeg subject
from labelled markers through target-specific motion fitting, a portable subject
recipe, CUDA PPO training, native checkpoint resume, ONNX export and CPU replay.
It uses the same ``myoLegWalk-v0`` registration and subject JSON on CPU and mjlab.
It creates no separate task registration.

Run the example
---------------

Use a MyoSuite source checkout with the capture, mjlab, training and ONNX
dependencies installed. The GPU interpreter needs CUDA-enabled PyTorch and a
supported NVIDIA GPU. The preparation stage needs only the CPU/capture packages:

.. code-block:: text

    python -m myosuite.integrations.capture.leg_workflow --stage prepare --output tasks/subject-leg

Run all stages, using a separate CUDA environment if desired:

.. code-block:: text

    python -m myosuite.integrations.capture.leg_workflow --stage all --output tasks/subject-leg-run --gpu-python <cuda-python>

Use a fresh output directory. The training stages refuse to overwrite existing
``runs/initial`` or ``runs/resumed`` directories. To execute the stages separately,
run ``prepare``, ``train``, ``resume`` and ``replay`` in that order with the same
``--output`` directory. Run the middle stages in the CUDA interpreter.

The training smoke run deliberately uses two environments, four rollout steps,
one PPO iteration and one resumed iteration, with small networks and local
TensorBoard logging. This is a pipeline check; it does not train a competent
walking policy or establish subject validity. Larger runs should use the
standard training CLI and an appropriate training budget.

What the preparation stage does
-------------------------------

The example generates synthetic markers from an independently constructed
subject with known isotropic femur/tibia factors. It uses existing named MyoLeg
anatomical sites: the hip centre and lateral femoral condyle for each femur,
and the knee centre and lateral malleolus for each tibia. A marker pair must lie
on the same body. Reference lengths come from the nominal model; source units
and the proper rotation into model-world coordinates are explicit.

The marker estimator recovers the four factors and records each measured
distance, calibration, isotropic assumption and source. All other bodies retain
their nominal factors. Synthetic markers are Apache-2.0 example outputs; they
contain no captured person, SMPL, SMPL-X or AMASS data. The stored known factors
are a negative-control reference for the example, not estimator inputs.

The target model is scaled before fitting the motion. Three moving frames are
generated with coupled knee coordinates and explicit sole-ground alignment.
Simultaneous marker IK fits pelvis, thigh, shank and foot markers against the
scaled target. Two existing off-origin toe muscle-path sites per foot are used
as synthetic markers so toe orientation is observable; a toe-joint origin alone
cannot determine that rotation. Sampled coupled knee poses are checked against
the model's helper-joint limits. A fitted first frame supplies ``reset_qpos`` and ``reset_qvel``
in ``SubjectEnvConfig``. The final model is built and the clip is fitted/saved
against that final identity, which also includes its reset keyframe. Native
clip loading then verifies that exact target model. The registered CPU reset is
checked against the fitted first pose and velocity.

The recipe explicitly names both feet's collision soles and the free root,
requests 2 mm clearance, supplies a 1 mm contact margin, a 0.70 m fall threshold,
and a 70 kg total mass with preserved relative mass distribution. These numbers
are example task choices in metres and kilograms; they are not inferred from
segment factors. The selected reset keyframe is realigned in its fitted pose.
The walking reward remains the registered walking task; the fitted clip supplies
initial state and provenance rather than a new tracking reward.

For real data, replace the synthetic ``MarkerTrial`` positions and calibrated
site correspondence with the measurements you are licensed to use. Review the
marker definitions, units, frame transform, missing-marker evidence, model
residuals, initial velocity and contact assumptions before training. One distance
per segment supports the example's stated isotropic assumption only.

Training and policy admission
-----------------------------

The example calls the same ``TrainConfig.from_task`` and ``run_train`` functions
as ``scripts/train_mjlab.py``. The first run records the shared subject model and
task identity in its checkpoint manifest. The second run actually loads that
native checkpoint, checks its morphology/task identity and continues PPO.

For ordinary training, the equivalent subject entry point is:

.. code-block:: text

    python scripts/train_mjlab.py myoLegWalk-v0 --subject tasks/subject-leg/subject.json --env.scene.num-envs 1024 --agent.logger tensorboard

Resume with the same subject recipe and the standard ``--agent.resume`` and
``--agent.load-run`` options. Passing a different subject, reset state, fall
threshold or muscle-feature configuration must fail admission even when the
observation/action spaces agree.

The replay stage exports the resumed checkpoint with its observation
normalization and embedded manifest. It loads both the native checkpoint and
ONNX through ``load_policy(..., strict=True)`` on the CPU subject, performs finite
rollouts, and verifies that both formats reject a nominal environment and a
different subject task. No nominal fallback policy is accepted.

The standard CPU evaluation entry point also consumes this recipe:

.. code-block:: text

    python scripts/eval_mjlab_policy.py myoLegWalk-v0 --subject tasks/subject-leg/subject.json --checkpoint <native-checkpoint-or-policy.onnx> --episodes 1

Output artifacts
----------------

.. list-table:: Files in the requested output directory
   :header-rows: 1
   :widths: 34 66

   * - Artifact
     - Contents
   * - ``static_markers.json``, ``motion_markers.json``
     - Synthetic labelled coordinates, units, frame transform and provenance.
   * - ``scale_estimate.json``
     - Recovered factors, measurements, stated assumptions and synthetic truth.
   * - ``subject.json``
     - Portable model/task recipe including fitted reset pose and velocity.
   * - ``retargeted_motion.npz`` and ``.json``
     - Named motion, target identity, marker residuals and geometric contact labels.
   * - ``runs/initial`` and ``runs/resumed``
     - Native checkpoints, model/task manifests, parameters and local training logs.
   * - ``policy.onnx``
     - Resumed actor with embedded normalization and subject manifest.
   * - ``preparation_report.json``, ``training_report.json``
     - Fitting quality, CPU rollout results, artifacts and mismatch rejections.
   * - ``train.log``, ``resume.log``, ``replay.log``
     - Complete stage output, including checkpoint loading and optimizer updates.
   * - ``smoke_summary.json``
     - Compact numeric proof with dependency/device versions and model/task identities.

Each run also writes ``smoke_report.json`` with its device and rollout size. The
resumed run verifies a nonzero change in actor network parameters after loading
the initial checkpoint; it does not rely only on checkpoint filenames or the
runner's iteration counter.

An actual RTX 4050 run is recorded in
``myosuite/integrations/capture/fixtures/leg_workflow_smoke.json``. This compact
reference contains numbers and provenance only; checkpoints and ONNX binaries
are generated locally. CPU replay also checks native versus ONNX actions on the
same reset observation, with a ``1e-5`` maximum absolute action tolerance.

Contact labels in the clip are geometric heuristics, not measured contact
forces. Finite short rollouts prove that the workflow runs; they do not prove
balance, successful walking, biomechanical accuracy or policy transfer to a
different person.
