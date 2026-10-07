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
   pip install -e ".[rl]"        # CPU training (Stable-Baselines3)
   # or with uv: uv sync -p 3.10 --extra rl

GPU training (mjlab)
~~~~~~~~~~~~~~~~~~~~

An NVIDIA GPU on Linux (tested) or Windows; on macOS mjlab runs on the CPU only. The torch and Warp wheels on PyPI are built for the newest CUDA; on an
older driver they install but cannot use the GPU. `uv <https://docs.astral.sh/uv/>`_ picks the torch
build that matches your driver:

.. code-block:: bash

   uv pip install -e ".[mjlab]" --torch-backend=auto
   # with pip: install a matching torch first (nvidia-smi shows the driver's CUDA version), e.g.
   # pip install torch --index-url https://download.pytorch.org/whl/cu129 && pip install -e ".[mjlab]"

On a CUDA 12 driver, also install the CUDA 12 build of Warp from its
`releases <https://github.com/NVIDIA/warp/releases>`_ (otherwise mjlab fails with
``Invalid device identifier: cuda:0``), e.g. on Linux
``pip install --force-reinstall --no-deps https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl``.

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
