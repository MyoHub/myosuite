import unittest

import jax
import jax.numpy as jp
import mujoco
import numpy as np

# Configure JAX to use CPU for consistent testing
jax.config.update("jax_platform_name", "cpu")

# Try to import mjx, skip tests if not available
try:
    import mjx

    MJX_AVAILABLE = True
except ImportError:
    MJX_AVAILABLE = False
    mjx = None


class TestMjxFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model"""
        if not MJX_AVAILABLE:
            raise unittest.SkipTest("MJX is not available")

        # Load a standard MuJoCo model for comparison and MJX model creation
        cls.mujoco_model = mujoco.MjModel.from_xml_path(
            "myosuite/simhive/myo_sim/finger/myofinger_v0.xml"
        )
        # Convert to MJX model
        cls.mjx_model = mjx.device_put(cls.mujoco_model)

    def test_model_loading(self):
        """Test that the MJX model is loaded correctly"""
        self.assertIsNotNone(self.mjx_model)
        # Check some basic properties, e.g., number of degrees of freedom
        self.assertEqual(self.mjx_model.nq, self.mujoco_model.nq)
        self.assertEqual(self.mjx_model.nv, self.mujoco_model.nv)
        self.assertEqual(self.mjx_model.nu, self.mujoco_model.nu)

    def test_data_creation(self):
        """Test creating MJX data from the MJX model"""
        mjx_data = mjx.make_data(self.mjx_model)
        self.assertIsNotNone(mjx_data)
        # Check some basic properties of the data
        self.assertEqual(mjx_data.qpos.shape[0], self.mjx_model.nq)
        self.assertEqual(mjx_data.qvel.shape[0], self.mjx_model.nv)
        self.assertEqual(mjx_data.act.shape[0], self.mjx_model.nu)

    def test_step_simulation(self):
        """Test performing a single simulation step with MJX"""
        mjx_data = mjx.make_data(self.mjx_model)

        # Store initial state
        initial_qpos = mjx_data.qpos
        initial_qvel = mjx_data.qvel

        # Define a JIT-compiled step function
        @jax.jit
        def run_step(model, data, action):
            return mjx.step(model, data, action)

        # Perform a step with zero control input
        act = jp.zeros(self.mjx_model.nu)
        new_mjx_data = run_step(self.mjx_model, mjx_data, act)

        # Assert that the new data object is different from the initial data object
        self.assertIsNot(mjx_data, new_mjx_data)

        # For a model with gravity (like myofinger_v0), qpos or qvel should change after a step
        qpos_changed = not jp.allclose(initial_qpos, new_mjx_data.qpos, atol=1e-6)
        qvel_changed = not jp.allclose(initial_qvel, new_mjx_data.qvel, atol=1e-6)

        self.assertTrue(
            qpos_changed or qvel_changed,
            "qpos or qvel should change after a step with gravity",
        )

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
        self.assertIsNotNone(new_mjx_data.xpos)
        self.assertIsNotNone(new_mjx_data.xquat)
        self.assertIsNotNone(new_mjx_data.subtree_com)

        # For myofinger, xpos should not be all zeros
        self.assertTrue(
            jp.any(new_mjx_data.xpos != 0.0),
            "xpos should not be all zeros after forward kinematics",
        )

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
