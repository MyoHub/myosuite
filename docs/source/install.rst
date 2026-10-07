Installation
============

Python 3.10–3.14 is supported (the GPU ``[mjlab]`` extra needs Python ≤3.13).

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

For GPU, install a torch build matching your driver's CUDA version *before*
the ``.[mjlab]`` install, e.g. ``pip install torch --index-url
https://download.pytorch.org/whl/cu128`` (see `pytorch.org/get-started/locally
<https://pytorch.org/get-started/locally/>`_ for the right tag). Otherwise an
unconstrained resolve may pick a torch build requiring a newer CUDA runtime
than your driver supports (or, in some environments, a CPU-only wheel).

Or with `uv <https://docs.astral.sh/uv/>`_::

   uv sync -p 3.10 --extra rl

Musculoskeletal models ship in the ``myo-sim`` pip package; the few MPL, YCB and
furniture assets MyoSuite uses are bundled in the package. No git submodule checkout is required.
A source install uses the ``myo-sim`` pin in ``pyproject.toml``.

Verify
~~~~~~

.. code-block:: bash

   python -c "import myosuite; print(len(myosuite.myosuite_env_suite), 'envs')"
   python -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0

On macOS the viewer needs ``mjpython`` instead of ``python``.

.. note::

   Run these commands from a directory that does not directly contain a folder named ``myosuite``, such as
   the parent of your clone. There Python imports that folder as a namespace package instead of the installed
   package, and fails with ``ImportError: cannot import name ... from 'myosuite' (unknown location)``, for
   editable and regular installs alike. Inside the clone, or anywhere else, it works.

Minimal usage
~~~~~~~~~~~~~

.. code-block:: python

   from myosuite import make_env

   env = make_env("myoElbowPose1D6MRandom-v0")
   obs, info = env.reset(seed=0)
   obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
   env.close()
