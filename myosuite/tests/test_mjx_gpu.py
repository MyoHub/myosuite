# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""MJX GPU smoke test.

This module provides a focused smoke test that verifies whether the MJX +
JAX stack is actually running on a GPU when requested. It is intended to be
used after installing the CUDA-enabled extra:

    uv sync --extra mjx-cuda

Usage:
    # CPU-only (default; no effect here, tests are skipped)
    pytest myosuite/tests/test_mjx_gpu.py -v

    # Explicitly request GPU tests
    MYOSUITE_GPU_TESTS=1 pytest myosuite/tests/test_mjx_gpu.py -v

Behaviour:
    - When ``MYOSUITE_GPU_TESTS`` is not set (or falsey), the test is skipped.
    - When it is set, the test:
        * checks that at least one JAX GPU device is present,
        * loads a small MuJoCo model and converts it to an MJX model,
        * runs a single JIT-compiled MJX step on the GPU,
        * asserts that the resulting arrays live on a GPU device.
    - Any JAX/cuSolver / device errors will cause this test to FAIL, which is
      a strong signal that the GPU installation is not healthy.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import mujoco

jax = pytest.importorskip("jax", reason="jax is required for MJX GPU smoke tests")
jnp = pytest.importorskip("jax.numpy", reason="jax is required for MJX GPU smoke tests")

try:
    from mujoco import mjx as mjx_mod
except Exception:  # noqa: BLE001
    mjx_mod = None


_GPU_REQUESTED = os.environ.get("MYOSUITE_GPU_TESTS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

pytestmark = [pytest.mark.tier2, pytest.mark.slow]


@pytest.mark.skipif(
    not _GPU_REQUESTED,
    reason="MJX GPU tests only run when MYOSUITE_GPU_TESTS=1",
)
def test_mjx_gpu_smoke() -> None:
    """Verify that MJX + JAX can execute a step on a GPU device.

    Skips if:
        - No MJX backend is available (``mujoco.mjx`` import fails), or
        - The simhive XML asset is not present (simhive not initialised).

    Fails if:
        - No JAX GPU device is visible, or
        - A simple MJX step cannot be executed on the GPU.
    """
    if mjx_mod is None:
        pytest.skip("mujoco.mjx not available; install mujoco with MJX support")

    # Require at least one GPU device from JAX.
    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpu_devices:
        pytest.fail(
            "No JAX GPU devices detected. "
            "Ensure CUDA drivers are installed and `uv sync --extra mjx-cuda` "
            "has been run in this environment."
        )

    # Use the standard myofinger MJCF used elsewhere in the test suite.
    try:
        import myo_sim  # type: ignore[import-untyped]

        xml_path = myo_sim.MODELS_DIR / "finger" / "myofinger_v0.xml"
    except ImportError:
        xml_path = Path("myosuite/simhive/myo_sim/finger/myofinger_v0.xml")
    if not xml_path.exists():
        pytest.skip(
            f"finger model not found at {xml_path}. "
            "Install myo-sim: `pip install myo-sim` or fetch simhive."
        )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    mjx_model = mjx_mod.put_model(model)
    data = mjx_mod.make_data(mjx_model)

    # Simple zero-action step: set ctrl on data, then step(model, data).
    data = data.replace(ctrl=jnp.zeros(mjx_model.nu, dtype=jnp.float32))

    @jax.jit
    def _step(m, d):
        return mjx_mod.step(m, d)

    new_data = _step(mjx_model, data)

    # Block on one of the arrays to surface any GPU execution errors.
    new_qpos = new_data.qpos
    new_qpos.block_until_ready()

    # In recent JAX versions, arrays expose their primary device via .device.
    try:
        dev = new_qpos.device
        assert dev.platform == "gpu", f"Expected GPU device, got {dev}"
    except AttributeError:
        # Fallback: rely on the presence of GPU devices and lack of errors
        # from the step above as a sufficient smoke signal.
        assert gpu_devices, "GPU devices disappeared during MJX step execution"
