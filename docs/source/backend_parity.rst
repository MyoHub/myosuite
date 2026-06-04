Backend Parity & Cross-Backend Playback
=========================================

This page documents the plan and tooling to validate that the three simulation
backends — **cpu-gym**, **MJX**, and **mjlab** — produce identical
action→observation sequences, and to export trained policies to **ONNX** so
that a model trained on any backend can be played back on the others.

.. contents::
   :local:
   :depth: 2


Why this matters
----------------

The three backends use different array libraries, precision, and physics
solvers:

.. list-table::
   :header-rows: 1
   :widths: 15 20 15 20 30

   * - Backend
     - Framework
     - Precision
     - Algorithm
     - Physics
   * - **cpu-gym**
     - Gymnasium / numpy
     - float64
     - SB3 SAC
     - MuJoCo C++ (sequential)
   * - **MJX XLA**
     - JAX / Brax
     - float32
     - SB3 SAC (single env) or Brax PPO
     - MuJoCo MJX on XLA
   * - **MJX Warp**
     - JAX / MuJoCo Warp
     - float32
     - SB3 SAC
     - MuJoCo Warp GPU
   * - **mjlab**
     - Isaac Lab / PyTorch
     - float32
     - RSL-RL PPO
     - MuJoCo Warp GPU (vectorised)

If each backend computes a *different* observation or reward for the same
physical state, a policy trained on one backend will degrade when played back
on another.  The goal is to detect and fix any such divergence.


What is expected to match
--------------------------

Given the same initial state and the same deterministic action sequence:

- **Observation vector** at each step: all 12 components must match to
  within float32 rounding (≈ 1e-3 absolute for most terms).
- **Per-step reward**: must match to within 0.05 absolute.
- **Episode length**: may differ by at most 2 steps due to float32/float64
  threshold comparisons near ``min_height = 0.8``.

What is **not** expected to match exactly:

- Episode-to-episode variance from exploration noise (tests use fixed actions).
- Algorithm-level hyperparameters (SAC vs PPO learning rates, batch sizes).
- SAR / SynergyWrapper pre-processing (excluded from raw physics comparison).


Current parity test suite
--------------------------

Three layers of tests are already in place for the Walk environment:

.. list-table::
   :header-rows: 1
   :widths: 45 35 20

   * - File
     - What it checks
     - When it runs
   * - ``benchmarks/sar_backends/test_obs_parity.py``
     - Observation + reward key names and formulas via AST inspection
     - Every CI run (no env needed)
   * - ``benchmarks/sar_backends/test_mjlab_training_invariants.py``
     - Training pipeline uses RSL-RL + SARTorchTransform (not SB3)
     - Every CI run (no GPU needed)
   * - ``benchmarks/sar_backends/test_backend_parity.py`` *(new)*
     - Numerical obs/reward values over 50 steps with fixed actions
     - On-demand; ``MYOSUITE_GPU_TESTS=1`` for mjlab

Run the structural (fast) tests::

    pytest benchmarks/sar_backends/test_obs_parity.py \
           benchmarks/sar_backends/test_mjlab_training_invariants.py -v

Run the numerical runtime tests (needs mujoco + jax)::

    pytest benchmarks/sar_backends/test_backend_parity.py -v

Enable mjlab parity (needs GPU)::

    MYOSUITE_GPU_TESTS=1 pytest benchmarks/sar_backends/test_backend_parity.py -v


Known divergence sources and mitigations
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Root cause
     - Impact
     - Mitigation
   * - float64 (CPU) vs float32 (MJX/mjlab)
     - Height/rotation thresholds differ by 1 step
     - Allow ±2 step episode-length tolerance; use short N_STEPS=50
   * - ``cvel`` sign convention in MJX vs CPU
     - ``com_vel`` observation flipped
     - ``_walk_obs_com_vel`` negates ``data.cvel`` (tested in test_obs_parity)
   * - ``actuator_force ÷ 1000`` float32 precision
     - Muscle force obs differs by ~5e-3
     - Tolerance 5e-3 applied to ``muscle_force`` component
   * - Contact model approximation (MJX vs C++)
     - Ground reaction forces differ slightly
     - Observed difference < 1e-3 on flat terrain; larger on stairs
   * - Action normalization (sigmoid vs linear)
     - Large divergence if backends differ
     - ``sigmoid(5*(a-0.5))`` verified in all three backends by test suite
   * - XML model path (wrong model for mjlab)
     - Completely different physics
     - ``test_xml_model_parity`` enforces ``simhive/myo_sim/leg/myolegs.xml``


ONNX export and cross-backend playback
----------------------------------------

ONNX provides a **framework-agnostic** policy representation that any backend
can run for inference.  The workflow is:

.. code-block:: text

    Train on any backend
           │
           ▼
    Export to .onnx
     (export_onnx.py)
           │
     ┌─────┼──────┐
     ▼     ▼      ▼
    CPU   MJX  mjlab
   playback verification

Step 1 — Export
~~~~~~~~~~~~~~~~

**SB3 (SAC / TD3 / PPO)** checkpoint::

    myosuite-export-onnx export \
        --framework sb3 \
        --checkpoint walk_sac.zip \
        --obs-dim 243 --act-dim 80 \
        --output walk_policy.onnx

**RSL-RL (PPO)** checkpoint::

    myosuite-export-onnx export \
        --framework rslrl \
        --checkpoint walk_ppo.pt \
        --obs-dim 243 --act-dim 80 \
        --hidden-dims 512 256 128 \
        --output walk_policy.onnx

**JAX/Brax PPO** — see :ref:`jax-onnx-export` below.

Step 2 — Verify on CPU
~~~~~~~~~~~~~~~~~~~~~~~

::

    myosuite-export-onnx verify \
        --onnx walk_policy.onnx \
        --steps 200 \
        --min-reward 0.5

Step 3 — Compare across backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    myosuite-export-onnx compare \
        --onnx walk_policy.onnx \
        --steps 50 \
        --atol 0.1

This runs the ONNX policy on CPU and MJX using the same action sequence and
reports the maximum absolute observation difference.

ONNX model interface
~~~~~~~~~~~~~~~~~~~~~

All exported models share the same interface:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 45

   * - Port
     - Name
     - Shape
     - Description
   * - Input
     - ``obs``
     - ``(batch, obs_dim)`` float32
     - Normalised observation from any backend
   * - Output
     - ``action``
     - ``(batch, act_dim)`` float32
     - Deterministic muscle activations in [0, 1]

The ``batch`` axis is dynamic, so the same model file works for both
single-env inference (batch=1) and vectorised inference (batch=N).

.. _jax-onnx-export:

JAX/Brax policy export
~~~~~~~~~~~~~~~~~~~~~~~

JAX policies require an extra conversion step because PyTorch's
``torch.onnx.export`` does not accept JAX arrays.  Two options:

**Option A — jax2tf + tf2onnx** (recommended for Flax networks)::

    # requires: pip install tensorflow tf2onnx
    import jax
    import jax.experimental.jax2tf as jax2tf
    import tensorflow as tf
    import tf2onnx

    apply_fn = jax.jit(lambda params, obs: policy.apply(params, obs))
    tf_fn = jax2tf.convert(apply_fn, enable_xla=False)

    @tf.function(input_signature=[
        tf.TensorSpec([None, obs_dim], tf.float32),
    ])
    def serving(obs):
        return tf_fn(params, obs)

    tf2onnx.convert.from_function(serving, output_path="walk_policy.onnx",
                                   opset=17)

**Option B — Intermediate PyTorch wrapper** *(planned — not yet implemented)*:

    Convert JAX/Flax params → numpy → PyTorch, then use ``torch.onnx.export``.
    This requires a ``flax_to_torch_mlp`` conversion utility (tracked in Phase 3
    of the roadmap below). Use Option A until this is available.

Option A is more robust for arbitrary Flax architectures; Option B will be simpler
for standard MLP networks once the conversion utility is implemented.


Extending parity tests to other environments
---------------------------------------------

The Walk environment is fully covered.  To add parity tests for other envs:

1. **Add a CPU fixture** in ``test_backend_parity.py`` that calls
   ``gym.make("<env-id>")`` and runs ``N_STEPS`` with fixed actions.

2. **Add an MJX fixture** calling the MJX equivalent env.

3. **Add parity assertions** using the same ``TestCpuVsMjxParity`` pattern.

4. **Add structural AST tests** in ``test_obs_parity.py`` verifying that
   the observation keys in the MJX/mjlab env source match the CPU env's
   ``DEFAULT_OBS_KEYS``.

Environments not yet covered:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Environment
     - CPU env ID
     - Gap / blocker
   * - Reach (elbow)
     - ``myoElbowPose1D6MRandom-v0``
     - No MJX equivalent yet
   * - Baoding
     - ``myoChallengeBaodingP2-v1``
     - No MJX/mjlab equivalent
   * - TableTennis
     - ``myoChallengeTableTennisP2-v0``
     - No MJX/mjlab equivalent
   * - Relocate
     - ``myoChallengeRelocateP2-v0``
     - No MJX/mjlab equivalent


Roadmap
-------

Phase 1 — Walk parity (current)
  Walk environment fully covered: structural AST tests + numerical runtime tests.
  ONNX export implemented for SB3 and RSL-RL.

Phase 2 — ONNX roundtrip CI gate
  Add a CI step that (a) exports a reference checkpoint to ONNX, (b) runs
  ``compare_onnx_across_backends`` on CPU vs MJX, and (c) fails if max obs
  diff > 0.1.  Prevents regressions when observation definitions change.

Phase 3 — JAX export and mjlab playback
  Implement ``flax_to_torch_mlp`` utility and the JAX→ONNX pipeline.
  Add mjlab ONNX inference runner (load ONNX in Torch, run in mjlab step loop).

Phase 4 — Additional environments
  Extend parity tests to Reach, Pose, and any new MJX-ported environments.

Phase 5 — Quantitative reward threshold
  Instead of just checking obs difference, assert that the ONNX policy achieves
  at least 80% of its training-environment reward when played back on the other
  two backends.  This is the definitive cross-backend compatibility criterion.


See also
--------

* ``benchmarks/sar_backends/test_backend_parity.py`` — runtime numerical tests
* ``myosuite/utils/export_onnx.py`` — ONNX export + playback CLI
* ``benchmarks/sar_backends/test_obs_parity.py`` — structural AST tests
* ``benchmarks/sar_backends/run_benchmark.py`` — full training comparison
* ``myosuite/core/protocols.py`` — ``EnvAccessor`` / ``PhysicsPath`` shared protocol
