# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import pytest

# Guard: skip entire module if JAX or MJX is not available
try:
    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    from mujoco import mjx

    MJX_AVAILABLE = True
except (ImportError, AttributeError) as _err:
    MJX_AVAILABLE = False
    mjx = None
    jax = None
    jp = None
    mujoco = None
    pytest.skip(
        f"JAX/MJX not available ({_err}); install with uv sync --extra mjx",
        allow_module_level=True,
    )

# Configure JAX to use CPU for consistent testing
jax.config.update("jax_platform_name", "cpu")

pytestmark = pytest.mark.tier2


class TestMjxFunctions:
    @classmethod
    def setup_class(cls):
        """Set up test data and model"""
        if not MJX_AVAILABLE:
            pytest.skip("MJX is not available")

        # Load a standard MuJoCo model for comparison and MJX model creation
        cls.mujoco_model = mujoco.MjModel.from_xml_path(
            "myosuite/simhive/myo_sim/finger/myofinger_v0.xml"
        )
        # Convert to MJX model
        cls.mjx_model = mjx.put_model(cls.mujoco_model)

    def test_model_loading(self):
        """Test that the MJX model is loaded correctly"""
        assert self.mjx_model is not None
        # Check some basic properties, e.g., number of degrees of freedom
        assert self.mjx_model.nq == self.mujoco_model.nq
        assert self.mjx_model.nv == self.mujoco_model.nv
        assert self.mjx_model.nu == self.mujoco_model.nu

    def test_data_creation(self):
        """Test creating MJX data from the MJX model"""
        mjx_data = mjx.make_data(self.mjx_model)
        assert mjx_data is not None
        # Check some basic properties of the data
        assert mjx_data.qpos.shape[0] == self.mjx_model.nq
        assert mjx_data.qvel.shape[0] == self.mjx_model.nv
        assert mjx_data.act.shape[0] == self.mjx_model.nu

    def test_step_simulation(self):
        """Test performing a single simulation step with MJX"""
        mjx_data = mjx.make_data(self.mjx_model)

        # Store initial state
        initial_qpos = mjx_data.qpos
        initial_qvel = mjx_data.qvel

        # Define a JIT-compiled step function.
        # In mujoco.mjx, ctrl is set on data before calling step (not passed separately).
        @jax.jit
        def run_step(model, data, action):
            data = data.replace(ctrl=action)
            return mjx.step(model, data)

        # Perform a step with zero control input
        act = jp.zeros(self.mjx_model.nu)
        new_mjx_data = run_step(self.mjx_model, mjx_data, act)

        # Assert that the new data object is different from the initial data object
        assert mjx_data is not new_mjx_data

        # For a model with gravity (like myofinger_v0), qpos or qvel should change after a step
        qpos_changed = not jp.allclose(initial_qpos, new_mjx_data.qpos, atol=1e-6)
        qvel_changed = not jp.allclose(initial_qvel, new_mjx_data.qvel, atol=1e-6)

        assert (
            qpos_changed or qvel_changed
        ), "qpos or qvel should change after a step with gravity"

    def test_forward_kinematics(self):
        """Test mjx.forward function and compare with MuJoCo's mj_forward"""
        mjx_data = mjx.make_data(self.mjx_model)

        # Define a JIT-compiled forward function
        @jax.jit
        def run_forward(model, data):
            return mjx.forward(model, data)

        # Run forward kinematics
        new_mjx_data = run_forward(self.mjx_model, mjx_data)

        # Check if some kinematic properties are computed
        assert new_mjx_data.xpos is not None
        assert new_mjx_data.xquat is not None
        assert new_mjx_data.subtree_com is not None

        # Check shape of xpos (one entry per body)
        assert new_mjx_data.xpos.shape == (self.mjx_model.nbody, 3)

        # Compare with MuJoCo's forward
        mujoco_data = mujoco.MjData(self.mujoco_model)
        mujoco.mj_forward(self.mujoco_model, mujoco_data)

        # Compare specific fields like xpos (body positions)
        np.testing.assert_allclose(
            np.array(new_mjx_data.xpos),
            mujoco_data.xpos,
            atol=1e-5,  # Allow for small floating point differences
            err_msg="mjx.forward xpos does not match mujoco.mj_forward xpos",
        )

        # Compare xquat (body orientations)
        np.testing.assert_allclose(
            np.array(new_mjx_data.xquat),
            mujoco_data.xquat,
            atol=1e-5,
            err_msg="mjx.forward xquat does not match mujoco.mj_forward xquat",
        )

        # Verify that qpos and qvel are unchanged by mjx.forward
        np.testing.assert_allclose(
            np.array(new_mjx_data.qpos),
            np.array(mjx_data.qpos),
            err_msg="mjx.forward should not change qpos",
        )
        np.testing.assert_allclose(
            np.array(new_mjx_data.qvel),
            np.array(mjx_data.qvel),
            err_msg="mjx.forward should not change qvel",
        )
