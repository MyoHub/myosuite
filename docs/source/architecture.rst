Architecture
=============

A task has two matched halves under one ``env_id``:

* **CPU** — :class:`~myosuite.envs.gymnasium_env.MyoGymnasiumEnv`, via
  ``gym.make(env_id)``. Playback, debug, SB3.
* **GPU** — mjlab ``ManagerBasedRlEnvCfg``, via ``scripts/train_mjlab.py``.
  Parallel training on MuJoCo Warp.

They share observation order, action mapping, and ``ctrl_dt``
(see :doc:`backend_parity` and ``docs/wiki/cross-backend-contract.md``).

An MJX (JAX) backend also exists. It is experimental — do not start new work
on it.

.. code-block:: text

                         myosuite/terms/          ← shared obs/reward/action terms
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
        CPU Gymnasium      mjlab (Warp)      MJX (experimental)
        MyoGymnasiumEnv    ManagerBasedRlEnv  mujoco_playground
        gym.make(env_id)   same env_id        Mjx*-v0 / mjx.make()


CPU
---

* Base: ``myosuite.envs.gymnasium_env.MyoGymnasiumEnv``
* Register with ``myosuite.core.registry.register_env`` (never ``gym.register``)
* ``step()`` returns ``(obs, reward, terminated, truncated, info)``
* ``info["obs_dict"]`` / ``info["rwd_dict"]`` hold the named breakdowns

Canonical example: ``myosuite/envs/myo/tasks/basic/arm/reach.py``.

Muscle-condition variants (``myoSarc…``, ``myoFati…``, hand ``myoReaf…``) are
auto-registered for ``myo*`` CPU IDs.


mjlab
-----

Install ``pip install -e ".[mjlab]"`` (see :doc:`install` for matching the torch
build to your driver's CUDA version). Tasks are discovered through the
``mjlab.tasks`` entry point (``myosuite.envs.myo.backends.mjlab``).
Train with ``python scripts/train_mjlab.py <env_id>``.


Terms
-----

Pure functions in ``myosuite/terms/``:

* ``base_obs.py`` — observation terms (``foo`` → ``foo_obs``)
* ``base_reward.py`` — reward terms (``foo`` → ``foo_reward``)
* ``base_action.py``, ``base_event.py``, ``base_termination.py``

Use ``accessor.array_module()``; do not hard-code numpy / jax / torch in shared
terms. See ``docs/wiki/writing-term-functions.md``.


ModelBuilder
------------

``myosuite.core.model_builder.ModelBuilder`` composes MJCF. Prefer named
recipes in ``myosuite.core.model_recipes``. For official challenge evals, use
``gym.make`` — recipes do not fully reproduce every challenge XML.
See :doc:`model_builder`.
