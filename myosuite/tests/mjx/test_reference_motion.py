import unittest
import numpy as np
import glob

from myosuite.logger.reference_motion_jax import ReferenceMotion as JaxReferenceMotion
from myosuite.logger.reference_motion import ReferenceMotion as NumpyReferenceMotion
from myosuite.logger.reference_motion import ReferenceType


class TestReferenceMotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Path to test data
        cls.data_dir = "./myosuite/envs/myo/myodm/data/"
        cls.reference_files = glob.glob(f"{cls.data_dir}*.npz")
        cls.ignore_files = [
            f"{cls.data_dir}MyoHand_cylindersmall_lift.npz",
            f"{cls.data_dir}MyoHand_fryingpan_cook2.npz",
            f"{cls.data_dir}MyoHand_hand_pass1.npz",
            f"{cls.data_dir}MyoHand_knife_lift.npz",
            f"{cls.data_dir}MyoHand_wineglass_drink1.npz",
        ]
        cls.fixed_ref_data = {
            "time": (0.0, 4.0),
            "robot": np.zeros((1, 29)),
            "robot_vel": np.zeros((1, 29)),
            "object_init": np.array((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)),
            "object": np.array([[-0.2, -0.2, 0.1, 1.0, 0.0, 0.0, -1.0]]),
        }

        cls.random_ref_data = {
            "time": (0.0, 4.0),
            "robot": np.zeros((2, 29)),
            "robot_vel": np.zeros((2, 29)),
            "object_init": np.array((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)),
            "object": np.array(
                [
                    [-0.2, -0.2, 0.1, 1.0, 0.0, 0.0, -1.0],
                    [0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0],
                ]
            ),
        }
        cls.file_path = "./myosuite/envs/myo/myodm/data/MyoHand_airplane_fly1.npz"

    def test_load_npz_file(self):
        """Test loading .npz files in both implementations"""
        for file_path in self.reference_files:
            if file_path in self.ignore_files:
                print(f"Skipping {file_path} because it is in the ignore list")
                continue

            # Load with both implementations
            numpy_ref = NumpyReferenceMotion(file_path)
            jax_ref = JaxReferenceMotion(file_path)

            # Compare basic properties
            self.assertEqual(jax_ref.horizon, numpy_ref.horizon)
            self.assertEqual(jax_ref.robot_dim, numpy_ref.robot_dim)
            self.assertEqual(jax_ref.object_dim, numpy_ref.object_dim)
            self.assertEqual(jax_ref.type.value, numpy_ref.type.value)

            # Compare reference data
            np.testing.assert_allclose(
                np.array(jax_ref.reference["time"]),
                numpy_ref.reference["time"],
                rtol=1e-5,
            )
            if jax_ref.reference["robot"] is not None:
                np.testing.assert_allclose(
                    np.array(jax_ref.reference["robot"]),
                    numpy_ref.reference["robot"],
                    rtol=1e-5,
                )
            if jax_ref.reference["object"] is not None:
                np.testing.assert_allclose(
                    np.array(jax_ref.reference["object"]),
                    numpy_ref.reference["object"],
                    rtol=1e-5,
                )

    def test_fixed_reference(self):
        """Test fixed reference type in both implementations"""
        # Create references

        jax_ref = JaxReferenceMotion(self.fixed_ref_data)
        numpy_ref = NumpyReferenceMotion(self.fixed_ref_data)

        # Check type
        self.assertEqual(jax_ref.type.value, ReferenceType.FIXED.value)
        self.assertEqual(numpy_ref.type.value, ReferenceType.FIXED.value)

        # Check initialization
        robot_init_jax, object_init_jax = jax_ref.get_init()
        robot_init_numpy, object_init_numpy = numpy_ref.get_init()

        np.testing.assert_allclose(np.array(robot_init_jax), robot_init_numpy)
        np.testing.assert_allclose(np.array(object_init_jax), object_init_numpy)

    def test_random_reference(self):
        """Test random reference type in both implementations"""
        # Create references
        jax_ref = JaxReferenceMotion(self.random_ref_data)
        numpy_ref = NumpyReferenceMotion(self.random_ref_data)

        # Check type
        self.assertEqual(jax_ref.type.value, ReferenceType.RANDOM.value)
        self.assertEqual(numpy_ref.type.value, ReferenceType.RANDOM.value)

        # Check initialization
        robot_init_jax, object_init_jax = jax_ref.get_init()
        robot_init_numpy, object_init_numpy = numpy_ref.get_init()

        np.testing.assert_allclose(np.array(robot_init_jax), robot_init_numpy)
        np.testing.assert_allclose(np.array(object_init_jax), object_init_numpy)

    def test_track_reference(self):
        """Test tracking reference type using actual motion files"""

        # Create references
        jax_ref = JaxReferenceMotion(self.file_path)
        numpy_ref = NumpyReferenceMotion(self.file_path)

        # Check type
        self.assertEqual(jax_ref.type.value, ReferenceType.TRACK.value)
        self.assertEqual(numpy_ref.type.value, ReferenceType.TRACK.value)

        # Test time slot finding
        test_times = [0.0, 0.1, 0.5, 1.0]
        for time in test_times:
            jax_indices = jax_ref.find_timeslot_in_reference(time)
            numpy_indices = numpy_ref.find_timeslot_in_reference(time)
            self.assertEqual(jax_indices, numpy_indices)

    def test_reset(self):
        """Test reset functionality"""

        # Create references
        jax_ref = JaxReferenceMotion(self.file_path)
        numpy_ref = NumpyReferenceMotion(self.file_path)

        # Advance index cache
        _ = jax_ref.find_timeslot_in_reference(0.5)
        _ = numpy_ref.find_timeslot_in_reference(0.5)

        # Reset
        jax_ref.reset()
        numpy_ref.reset()

        # Check if index cache is reset
        self.assertEqual(jax_ref.index_cache, 0)
        self.assertEqual(numpy_ref.index_cache, 0)

    def test_error_handling(self):
        """Test error handling in both implementations"""
        # Test invalid reference data
        invalid_ref = {
            "time": np.array([0.0]),
            "robot": np.array([0.0, 0.1, 0.2]),  # Wrong shape
        }

        with self.assertRaises(AssertionError):
            _ = JaxReferenceMotion(invalid_ref)
        with self.assertRaises(AssertionError):
            _ = NumpyReferenceMotion(invalid_ref)

        # Test missing time key
        invalid_ref = {"robot": np.array([[0.0, 0.1, 0.2]])}

        with self.assertRaises(AssertionError):
            _ = JaxReferenceMotion(invalid_ref)
        with self.assertRaises(AssertionError):
            _ = NumpyReferenceMotion(invalid_ref)

    def test_get_init_fixed(self):
        """Test get_init for fixed reference type"""
        jax_ref = JaxReferenceMotion(self.fixed_ref_data)
        numpy_ref = NumpyReferenceMotion(self.fixed_ref_data)

        # Get initial states
        jax_robot_init, jax_object_init = jax_ref.get_init()
        numpy_robot_init, numpy_object_init = numpy_ref.get_init()

        # Compare results
        np.testing.assert_allclose(
            np.array(jax_robot_init),
            numpy_robot_init,
            rtol=1e-5,
            err_msg="Robot init states don't match",
        )
        np.testing.assert_allclose(
            np.array(jax_object_init),
            numpy_object_init,
            rtol=1e-5,
            err_msg="Object init states don't match",
        )

    def test_get_init_random(self):
        """Test get_init for random reference type"""
        jax_ref = JaxReferenceMotion(self.random_ref_data)
        numpy_ref = NumpyReferenceMotion(self.random_ref_data)

        # Get initial states
        jax_robot_init, jax_object_init = jax_ref.get_init()
        numpy_robot_init, numpy_object_init = numpy_ref.get_init()

        np.testing.assert_allclose(
            np.array(jax_robot_init),
            numpy_robot_init,
            rtol=1e-5,
            err_msg="Robot init states don't match",
        )
        np.testing.assert_allclose(
            np.array(jax_object_init),
            numpy_object_init,
            rtol=1e-5,
            err_msg="Object init states don't match",
        )

    def test_get_reference_fixed(self):
        """Test get_reference for fixed reference type"""
        jax_ref = JaxReferenceMotion(self.fixed_ref_data)
        numpy_ref = NumpyReferenceMotion(self.fixed_ref_data)

        # Test at different times (should all give same result for fixed type)
        test_times = [0.0, 0.5, 1.0]

        for time in test_times:
            jax_ref_struct = jax_ref.get_reference(time)
            numpy_ref_struct = numpy_ref.get_reference(time)

            # Compare results
            np.testing.assert_allclose(
                np.array(jax_ref_struct.robot),
                numpy_ref_struct.robot,
                rtol=1e-5,
                err_msg=f"Robot refs don't match at time {time}",
            )
            np.testing.assert_allclose(
                np.array(jax_ref_struct.object),
                numpy_ref_struct.object,
                rtol=1e-5,
                err_msg=f"Object refs don't match at time {time}",
            )

    def test_get_reference_track(self):
        """Test get_reference for tracking reference type"""
        jax_ref = JaxReferenceMotion(self.file_path)
        numpy_ref = NumpyReferenceMotion(self.file_path)

        # Test exact timestamps
        time = jax_ref.reference["time"][1]  # Use second timestamp
        jax_ref_struct = jax_ref.get_reference(time)
        numpy_ref_struct = numpy_ref.get_reference(time)

        # Compare results at exact timestamp
        np.testing.assert_allclose(
            np.array(jax_ref_struct.robot),
            numpy_ref_struct.robot,
            rtol=1e-5,
            err_msg="Robot refs don't match at exact timestamp",
        )

        # Test interpolation
        time = (jax_ref.reference["time"][1] + jax_ref.reference["time"][2]) / 2
        jax_ref_struct = jax_ref.get_reference(time)
        numpy_ref_struct = numpy_ref.get_reference(time)

        # Compare interpolated results
        np.testing.assert_allclose(
            np.array(jax_ref_struct.robot),
            numpy_ref_struct.robot,
            rtol=1e-5,
            err_msg="Robot refs don't match during interpolation",
        )

    def test_get_reference_extrapolation(self):
        """Test get_reference with extrapolation"""
        # Enable extrapolation
        jax_ref = JaxReferenceMotion(self.file_path, motion_extrapolation=True)
        numpy_ref = NumpyReferenceMotion(self.file_path, motion_extrapolation=True)

        # Test beyond final timestamp
        max_time = jax_ref.reference["time"][-1]
        test_time = max_time + 1.0

        jax_ref_struct = jax_ref.get_reference(test_time)
        numpy_ref_struct = numpy_ref.get_reference(test_time)

        # Compare extrapolated results
        np.testing.assert_allclose(
            np.array(jax_ref_struct.robot),
            numpy_ref_struct.robot,
            rtol=1e-5,
            err_msg="Robot refs don't match during extrapolation",
        )

        # Should match final position when extrapolating
        np.testing.assert_allclose(
            np.array(jax_ref_struct.robot),
            jax_ref.reference["robot"][-1],
            rtol=1e-5,
            err_msg="Extrapolation doesn't match final position",
        )
        
    def test_missing_init_fixed(self):
        """Test initialization when robot_init and object_init are missing for fixed reference"""
        # Create reference data without init values
        fixed_ref_no_init = {
            "time": np.array([0.0]),
            "robot": np.array([[1.0, 2.0, 3.0]]),
            "robot_vel": np.array([[0.1, 0.2, 0.3]]),
            "object": np.array([[4.0, 5.0, 6.0]]),
        }

        # Create references
        jax_ref = JaxReferenceMotion(fixed_ref_no_init)
        numpy_ref = NumpyReferenceMotion(fixed_ref_no_init)

        # Get initial states
        jax_robot_init, jax_object_init = jax_ref.get_init()
        numpy_robot_init, numpy_object_init = numpy_ref.get_init()

        # For fixed type, init should be first frame
        expected_robot_init = fixed_ref_no_init["robot"][0]
        expected_object_init = fixed_ref_no_init["object"][0]

        # Compare results
        np.testing.assert_allclose(
            np.array(jax_robot_init),
            expected_robot_init,
            rtol=1e-5,
            err_msg="Robot init not using first frame",
        )
        np.testing.assert_allclose(
            np.array(jax_object_init),
            expected_object_init,
            rtol=1e-5,
            err_msg="Object init not using first frame",
        )

        # Compare JAX and NumPy implementations
        np.testing.assert_allclose(
            np.array(jax_robot_init),
            numpy_robot_init,
            rtol=1e-5,
            err_msg="Robot init states don't match between implementations",
        )
        np.testing.assert_allclose(
            np.array(jax_object_init),
            numpy_object_init,
            rtol=1e-5,
            err_msg="Object init states don't match between implementations",
        )

    def test_missing_init_random(self):
        """Test initialization when robot_init and object_init are missing for random reference"""
        # Create reference data without init values
        random_ref_no_init = {
            "time": np.array([0.0, 1.0]),
            "robot": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "robot_vel": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "object": np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        }

        # Create references
        jax_ref = JaxReferenceMotion(random_ref_no_init)
        numpy_ref = NumpyReferenceMotion(random_ref_no_init)

        # Get initial states
        jax_robot_init, jax_object_init = jax_ref.get_init()
        numpy_robot_init, numpy_object_init = numpy_ref.get_init()

        # For random type, init should be mean of bounds
        expected_robot_init = random_ref_no_init["robot"][0]
        expected_object_init = random_ref_no_init["object"][0]

        # Compare results with expected means
        np.testing.assert_allclose(
            np.array(jax_robot_init),
            expected_robot_init,
            rtol=1e-5,
            err_msg="Robot init not using mean of bounds",
        )
        np.testing.assert_allclose(
            np.array(jax_object_init),
            expected_object_init,
            rtol=1e-5,
            err_msg="Object init not using mean of bounds",
        )

        # Compare JAX and NumPy implementations
        np.testing.assert_allclose(
            np.array(jax_robot_init),
            numpy_robot_init,
            rtol=1e-5,
            err_msg="Robot init states don't match between implementations",
        )
        np.testing.assert_allclose(
            np.array(jax_object_init),
            numpy_object_init,
            rtol=1e-5,
            err_msg="Object init states don't match between implementations",
        )

    def test_missing_init_track(self):
        """Test initialization when robot_init and object_init are missing for track reference"""
        # Load tracking data
        track_ref = {k: v for k, v in np.load(self.file_path).items()}

        # Remove init values if they exist
        track_ref.pop("robot_init", None)
        track_ref.pop("object_init", None)

        # Create references
        jax_ref = JaxReferenceMotion(track_ref)
        numpy_ref = NumpyReferenceMotion(track_ref)

        # Get initial states
        jax_robot_init, jax_object_init = jax_ref.get_init()
        numpy_robot_init, numpy_object_init = numpy_ref.get_init()

        # For track type, init should be first frame
        expected_robot_init = track_ref["robot"][0] if "robot" in track_ref else None
        expected_object_init = track_ref["object"][0] if "object" in track_ref else None

        # Compare results with first frame
        if expected_robot_init is not None:
            np.testing.assert_allclose(
                np.array(jax_robot_init),
                expected_robot_init,
                rtol=1e-5,
                err_msg="Robot init not using first frame",
            )
        if expected_object_init is not None:
            np.testing.assert_allclose(
                np.array(jax_object_init),
                expected_object_init,
                rtol=1e-5,
                err_msg="Object init not using first frame",
            )

        # Compare JAX and NumPy implementations
        np.testing.assert_allclose(
            np.array(jax_robot_init),
            numpy_robot_init,
            rtol=1e-5,
            err_msg="Robot init states don't match between implementations",
        )
        if jax_object_init is not None and numpy_object_init is not None:
            np.testing.assert_allclose(
                np.array(jax_object_init),
                numpy_object_init,
                rtol=1e-5,
                err_msg="Object init states don't match between implementations",
            )


if __name__ == "__main__":
    unittest.main()
