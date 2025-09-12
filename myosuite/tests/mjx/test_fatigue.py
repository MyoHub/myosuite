import unittest
import numpy as np
import jax
import jax.numpy as jp
import mujoco
from jax import tree_util

from myosuite.envs.myo.fatigue import CumulativeFatigue as NumpyCumulativeFatigue
from myosuite.envs.myo.fatigue_jax import CumulativeFatigue as JaxCumulativeFatigue

# Configure JAX to use CPU for consistent testing
jax.config.update("jax_platform_name", "cpu")


class TestFatigue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model"""
        cls.model = mujoco.MjModel.from_xml_path(
            "myosuite/simhive/myo_sim/finger/myofinger_v0.xml"
        )
        cls.frame_skip = 5
        cls.test_act = np.array([0.5] * 5, dtype=np.float32)
        cls.test_fatigue_vec = np.array([0.5] * 5, dtype=np.float32)

    def test_pytree_structure(self):
        """Test that the class works correctly as a PyTree"""
        fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)

        # Test flattening and unflattening
        flat, treedef = tree_util.tree_flatten(fatigue)
        restored = tree_util.tree_unflatten(treedef, flat)

        # Check that all attributes are preserved
        np.testing.assert_allclose(restored.MA, fatigue.MA)
        np.testing.assert_allclose(restored.MR, fatigue.MR)
        np.testing.assert_allclose(restored.MF, fatigue.MF)
        self.assertEqual(restored.na, fatigue.na)

        # Test that the restored object works correctly
        MA1, _, _ = fatigue.compute_act(self.test_act)
        MA2, _, _ = restored.compute_act(self.test_act)
        np.testing.assert_allclose(MA1, MA2)

    def test_random_reset(self):
        """Test random reset with explicit key handling"""
        fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)
        key = jax.random.PRNGKey(0)

        # Test random reset
        fatigue.reset(key=key, fatigue_reset_random=True)

        # Verify states sum to 1
        total = fatigue.MA + fatigue.MR + fatigue.MF
        np.testing.assert_allclose(total, jp.ones_like(total))

        # Test deterministic behavior with same key
        fatigue2 = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)
        fatigue2.reset(key=key, fatigue_reset_random=True)

        np.testing.assert_allclose(fatigue.MA, fatigue2.MA)
        np.testing.assert_allclose(fatigue.MR, fatigue2.MR)
        np.testing.assert_allclose(fatigue.MF, fatigue2.MF)

    def test_compute_act_vmap(self):
        """Test that compute_act works with vmap"""
        batch_size = 2
        batch_acts = jp.stack([self.test_act] * batch_size)

        # Define a batched computation using vmap and JIT
        @jax.jit
        def batch_compute(acts):
            def single_compute(act):
                fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)
                return fatigue.compute_act(act)

            return jax.vmap(single_compute)(acts)

        # Run batch computation
        batch_MA, batch_MR, batch_MF = batch_compute(batch_acts)

        # Verify shapes
        self.assertEqual(batch_MA.shape, (batch_size, 5))
        self.assertEqual(batch_MR.shape, (batch_size, 5))
        self.assertEqual(batch_MF.shape, (batch_size, 5))

        # Verify against sequential computation
        for i in range(batch_size):
            fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)
            MA, MR, MF = fatigue.compute_act(batch_acts[i])
            np.testing.assert_allclose(MA, batch_MA[i])
            np.testing.assert_allclose(MR, batch_MR[i])
            np.testing.assert_allclose(MF, batch_MF[i])

    def test_random_reset_vmap(self):
        """Test that random reset works with vmap"""
        batch_size = 3
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, batch_size)

        # Define batched reset function
        @jax.jit
        def batch_reset(keys):
            def single_reset(key):
                fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)
                fatigue.reset(key=key, fatigue_reset_random=True)
                return fatigue.MA, fatigue.MR, fatigue.MF

            return jax.vmap(single_reset)(keys)

        # Run batch reset
        batch_MA, batch_MR, batch_MF = batch_reset(keys)

        # Verify shapes
        self.assertEqual(batch_MA.shape, (batch_size, 5))
        self.assertEqual(batch_MR.shape, (batch_size, 5))
        self.assertEqual(batch_MF.shape, (batch_size, 5))

        # Verify states sum to 1 for each instance
        totals = batch_MA + batch_MR + batch_MF
        np.testing.assert_allclose(totals, jp.ones_like(totals))

    def test_get_effort_vmap(self):
        """Test that get_effort works with vmap"""
        batch_size = 2
        batch_acts = jp.stack([self.test_act] * batch_size)

        # Define a batched computation using vmap and JIT
        @jax.jit
        def batch_effort(acts):
            def single_effort(act):
                fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)
                fatigue.compute_act(act)
                return fatigue.get_effort()

            return jax.vmap(single_effort)(acts)

        # Run batch computation
        batch_efforts = batch_effort(batch_acts)

        # Verify shapes
        self.assertEqual(batch_efforts.shape, (batch_size,))

        # Verify against sequential computation
        for i in range(batch_size):
            fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)
            fatigue.compute_act(batch_acts[i])
            effort = fatigue.get_effort()
            np.testing.assert_allclose(effort, batch_efforts[i])

    def test_numpy_jax_compute_act(self):
        """Test that JAX and NumPy compute_act produce the same results"""
        numpy_fatigue = NumpyCumulativeFatigue(self.model, frame_skip=self.frame_skip)
        jax_fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)

        # Test sequence of activations
        test_acts = [
            np.zeros(5, dtype=np.float32),  # Zero activation
            np.ones(5, dtype=np.float32),  # Full activation
            np.array([0.3, 0.5, 0.7, 0.2, 0.8], dtype=np.float32),  # Mixed activation
            np.array([0.5] * 5, dtype=np.float32),  # Uniform activation
        ]

        for act in test_acts:
            numpy_MA, numpy_MR, numpy_MF = numpy_fatigue.compute_act(act)
            jax_MA, jax_MR, jax_MF = jax_fatigue.compute_act(
                jp.array(act, dtype=jp.float32)
            )

            np.testing.assert_allclose(
                numpy_MA,
                np.array(jax_MA),
                rtol=1e-5,
                err_msg=f"MA mismatch for activation {act}",
            )
            np.testing.assert_allclose(
                numpy_MR,
                np.array(jax_MR),
                rtol=1e-5,
                err_msg=f"MR mismatch for activation {act}",
            )
            np.testing.assert_allclose(
                numpy_MF,
                np.array(jax_MF),
                rtol=1e-5,
                err_msg=f"MF mismatch for activation {act}",
            )

    def test_numpy_jax_effort(self):
        """Test that JAX and NumPy effort calculations match"""
        numpy_fatigue = NumpyCumulativeFatigue(self.model, frame_skip=self.frame_skip)
        jax_fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)

        # Test with different activations
        test_acts = [
            np.zeros(5, dtype=np.float32),
            np.ones(5, dtype=np.float32),
            np.array([0.3, 0.5, 0.7, 0.2, 0.8], dtype=np.float32),
            np.array([0.5] * 5, dtype=np.float32),
        ]

        for act in test_acts:
            # Update states
            numpy_fatigue.compute_act(act)
            jax_fatigue.compute_act(act)

            # Compare efforts
            numpy_effort = numpy_fatigue.get_effort()
            jax_effort = float(jax_fatigue.get_effort())

            np.testing.assert_allclose(
                numpy_effort,
                jax_effort,
                rtol=1e-5,
                err_msg=f"Effort mismatch for activation {act}",
            )

    def test_numpy_jax_parameters(self):
        """Test that JAX and NumPy parameter updates behave the same"""
        numpy_fatigue = NumpyCumulativeFatigue(self.model, frame_skip=self.frame_skip)
        jax_fatigue = JaxCumulativeFatigue(self.model, frame_skip=self.frame_skip)

        # Test Fatigue coefficient
        new_F = 0.02
        numpy_fatigue.set_FatigueCoefficient(new_F)
        jax_fatigue.set_FatigueCoefficient(new_F)
        np.testing.assert_allclose(numpy_fatigue.F, float(jax_fatigue.F))

        # Test Recovery coefficient
        new_R = 0.003
        numpy_fatigue.set_RecoveryCoefficient(new_R)
        jax_fatigue.set_RecoveryCoefficient(new_R)
        np.testing.assert_allclose(numpy_fatigue.R, float(jax_fatigue.R))

        # Test Recovery multiplier
        new_r = 12
        numpy_fatigue.set_RecoveryMultiplier(new_r)
        jax_fatigue.set_RecoveryMultiplier(new_r)
        np.testing.assert_allclose(numpy_fatigue.r, float(jax_fatigue.r))

        # Verify behavior with updated parameters
        act = np.array([0.5] * 5, dtype=np.float32)
        numpy_MA, _, _ = numpy_fatigue.compute_act(act)
        jax_MA, _, _ = jax_fatigue.compute_act(act)
        np.testing.assert_allclose(
            numpy_MA,
            np.array(jax_MA),
            rtol=1e-5,
            err_msg="MA mismatch after parameter updates",
        )


if __name__ == "__main__":
    unittest.main()
