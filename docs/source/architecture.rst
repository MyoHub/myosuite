Architecture
=============

MyoSuite 4 supports three independent execution paths that share the same
**term functions** (observation, reward, termination logic).
The diagram below shows how the pieces fit together.

.. code-block:: text

                         myosuite/terms/           ← SHARED across all three paths
                     ┌────────────────────────┐
                     │  myo_obs_terms.py      │
                     │  myo_action_terms.py   │  pure functions
                     │  myo_reward_terms.py   │  receive EnvAccessor
                     │  myo_event_terms.py    │  return native array type
                     │  myo_termination_terms │  (numpy | jax.Array | torch.Tensor)
                     └────────┬───────────────┘
                              │ EnvAccessor protocol
           ┌──────────────────┼──────────────────────┐
           ▼                  ▼                      ▼
    PATH 1: CPU          PATH 2: MJX            PATH 3: mjlab
    ─────────────        ────────────────        ──────────────────────────────
    gymnasium.Env        mujoco_playground        mjlab.ManagerBasedRlEnv
                         .MjxEnv                  (Isaac Lab manager API)
         │                    │                         │
    MyoGymnasiumEnv      MyoMjxEnvBase            ManagerBasedRlEnvCfg subclass
    (CPU, numpy)         (JAX, jax.Array)         (MuJoCo Warp, torch.Tensor)
         │                    │                         │
    gym.make(env_id)     myosuite.envs.myo.backends.mjx    mjlab.envs.make(env_id)
                         .make(env_id)            + entry-point autodiscovery
                         + pg_wrapper.wrap_for_
                           brax_training()

Paths 1 and 2 are separate class hierarchies — MJX inherits
``mujoco_playground.MjxEnv``, **not** ``MyoGymnasiumEnv``.
All three paths share the same term functions via a thin ``EnvAccessor``.


Path 1 — CPU (Gymnasium)
--------------------------

The standard single-environment path for CPU-based training and analysis.

* **Base class**: ``myosuite.envs.gymnasium_env.MyoGymnasiumEnv``
* **Backend**: MuJoCo CPU (``mujoco`` Python bindings)
* **Array type**: ``numpy.ndarray``
* **API**: Gymnasium ≥ 0.26 — 5-tuple ``(obs, reward, terminated, truncated, info)``
* **Entry point**: ``gym.make("myoElbowPose1D6MRandom-v0")``

All environments registered via :func:`myosuite.core.registry.register_env`
are available here.  Pathological variants (sarcopenia, fatigue,
reafferentation) are auto-registered for every ``myo*`` base environment.


Path 2 — MJX (JAX / GPU)
--------------------------

Massively parallel execution via `MuJoCo MJX <https://mujoco.readthedocs.io/en/stable/mjx.html>`_
and the `mujoco-playground <https://github.com/google-deepmind/mujoco_playground>`_ framework.

* **Base class**: ``myosuite.envs.myo.backends.mjx.MyoMjxEnvBase``
  (inherits ``mujoco_playground.MjxEnv``, **not** ``MyoGymnasiumEnv``)
* **Backend**: MuJoCo MJX — JIT-compiled JAX.  Default is **JAX/XLA** (``mjx_impl=None``,
  sometimes called *mjx_xla*): runs on CPU, GPU, or TPU with no extra dependencies.
  Optional **MuJoCo Warp** (``mjx_impl="warp"``) is GPU-only and requires ``warp-lang``.
* **Array type**: ``jax.Array``
* **Entry point**: ``myosuite.envs.myo.backends.mjx.make("MjxElbowPoseFixed-v0")``
* **Brax wrapper**: ``mujoco_playground.wrapper.wrap_for_brax_training(env)``
* **JAX PPO**: ``scripts/train_sar_jax_ppo.py`` trains with Brax PPO on MJX; it uses
  **mjx_xla** (JAX/XLA) by default and prefers it (works without GPU; use ``JAX_PLATFORMS=cpu``
  if no CUDA is available).

Install: ``pip install -e ".[mjx]"``  (add ``mjx-cuda`` for CUDA support).

Available MJX environments: ``MjxElbowPoseFixed-v0``, ``MjxElbowPoseRandom-v0``,
``MjxFingerPoseFixed-v0``, ``MjxFingerPoseRandom-v0``,
``MjxHandReachFixed-v0``, ``MjxHandReachRandom-v0``, ``MjxLegWalk-v0``.


Path 3 — mjlab (MuJoCo Warp / Isaac Lab)
------------------------------------------

Isaac Lab–style manager-based environments running on MuJoCo Warp for
GPU-accelerated rigid-body simulation with PyTorch tensors.

* **Base class**: ``mjlab.ManagerBasedRlEnv`` (Isaac Lab manager API)
* **Config class**: a ``ManagerBasedRlEnvCfg`` dataclass subclass per task
* **Backend**: MuJoCo Warp
* **Array type**: ``torch.Tensor``
* **Entry point**: ``mjlab.envs.make("myoElbowPose1D6MRandom-v0")``
* **Discovery**: entry-point autodiscovery via
  ``myosuite.envs.myo.backends.mjlab.REGISTERED_TASKS``

Install: ``pip install -e ".[mjlab]"``

Currently registered mjlab tasks: ``myoElbowPose1D6MRandom-v0``,
``myoElbowPose1D6MFixed-v0``, ``myoHandPoseRandom-v0``,
``myoLegWalk-v0``, ``myoChallengeBaodingP2-v1``.


Terms and EnvAccessor
-----------------------

Term functions live in ``myosuite/terms/`` and are **backend-agnostic**:

.. code-block:: text

    myosuite/terms/
    ├── myo_obs_terms.py          # observation constructors
    ├── myo_action_terms.py       # action constructors
    ├── myo_reward_terms.py       # reward functions → {"dense": float, "done": bool, ...}
    ├── myo_event_terms.py        # episode-level events (resets, noise injection)
    └── myo_termination_terms.py  # termination conditions

Rules:

* Term functions are **pure** — no side effects, no global state.
* They receive an ``EnvAccessor`` protocol object, never ``mujoco.MjData`` directly.
* They use ``accessor.array_module()`` for all array ops (returns ``numpy``,
  ``jnp``, or ``torch`` depending on the active path).
* They return the **native array type** of the current path, so the same
  function works on CPU, MJX, and mjlab without modification.


ModelBuilder
-------------

``myosuite.utils.model_builder.ModelBuilder`` is the only supported way to
compose MJCF models programmatically.  Hard-coded e.g. ``curr_dir + "/../../../simhive/..."``,
path strings are not permitted.

.. code-block:: python

    from myosuite.utils.model_builder import ModelBuilder

    builder = ModelBuilder()
    builder.add_model("myohand")       # resolved via myo_sim asset registry
    builder.add_model("baoding_balls")
    xml_string = builder.build()


Registry
---------

All CPU environments are registered via:

.. code-block:: python

    from myosuite.core.registry import register_env

    register_env(
        env_id="myoMyNewTask-v0",
        entry_point="myosuite.envs.myo.myobase.my_task:MyTaskEnv",
        max_episode_steps=200,
        kwargs={...},
    )

This also auto-registers the sarcopenia (``myoSarc…``), fatigue (``myoFati…``),
and — for hand environments — reafferentation (``myoReaf…``) variants.

See :doc:`environments` for the complete listing of all registered environments.
