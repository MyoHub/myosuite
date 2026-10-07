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

For GPU, the torch build has to match your NVIDIA driver: the default torch wheel on PyPI is built
for the newest CUDA, and on an older driver it installs fine but cannot use the GPU ("The NVIDIA driver
on your system is too old"). pip cannot see the driver, so either let `uv <https://docs.astral.sh/uv/>`_
pick the build for you::

   uv pip install -e ".[mjlab]" --torch-backend=auto     # or: export UV_TORCH_BACKEND=auto

or install a matching torch *before* the ``.[mjlab]`` install with pip, e.g. ``pip install torch
--index-url https://download.pytorch.org/whl/cu129`` for a CUDA 12.8 / 12.9 driver (``nvidia-smi``
shows the CUDA version; see `pytorch.org/get-started/locally <https://pytorch.org/get-started/locally/>`_
for the right tag).

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
