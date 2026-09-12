Installation
============

Requires Python 3.10 or later.

From PyPI
~~~~~~~~~

.. code-block:: bash

   pip install -U myosuite

From source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/MyoHub/myosuite.git
   cd myosuite
   pip install -e .
   # CPU RL:    pip install -e ".[rl]"
   # GPU (Linux + CUDA): pip install -e ".[mjlab]"

Or with `uv <https://docs.astral.sh/uv/>`_::

   uv sync -p 3.10 --extra rl

Sim assets ship as pip packages (``myo-sim``, ``furniture-sim``, ``mpl-sim``,
``object-sim``, ``ycb-sim``). No git submodule checkout is required.
A source install uses the ``myo-sim`` pin in ``pyproject.toml``.

Verify
~~~~~~

.. code-block:: bash

   python -c "import myosuite; print(len(myosuite.myosuite_env_suite), 'envs')"
   python -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0

On macOS the viewer needs ``mjpython`` instead of ``python``.

Minimal usage
~~~~~~~~~~~~~

.. code-block:: python

   import gymnasium as gym
   import myosuite

   env = gym.make("myoElbowPose1D6MRandom-v0")
   obs, info = env.reset(seed=0)
   obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
   env.close()
