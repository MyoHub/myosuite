import unittest
import numpy as np
import jax
import jax.numpy as jp

# Configure JAX to use CPU to avoid potential GPU-related issues
jax.config.update("jax_platform_name", "cpu")

from myosuite.utils.quat_math import (
    mulQuat as np_mulQuat,
    negQuat as np_negQuat,
    quat2Vel as np_quat2Vel,
    diffQuat as np_diffQuat,
    quatDiff2Vel as np_quatDiff2Vel,
    axis_angle2quat as np_axis_angle2quat,
    euler2mat as np_euler2mat,
    intrinsic_euler2quat as np_euler2quat,
    mat2euler as np_mat2euler,
    mat2quat as np_mat2quat,
    quat2euler as np_quat2euler,
    quat2mat as np_quat2mat,
    rotVecMatT as np_rotVecMatT,
    rotVecMat as np_rotVecMat,
    rotVecQuat as np_rotVecQuat,
    quat2euler_intrinsic as np_quat2euler_intrinsic,
)
from myosuite.utils.quat_math_jax import (
    mulQuat as jax_mulQuat,
    negQuat as jax_negQuat,
    quat2Vel as jax_quat2Vel,
    diffQuat as jax_diffQuat,
    quatDiff2Vel as jax_quatDiff2Vel,
    axis_angle2quat as jax_axis_angle2quat,
    euler2mat as jax_euler2mat,
    intrinsic_euler2quat as jax_euler2quat,
    mat2euler as jax_mat2euler,
    mat2quat as jax_mat2quat,
    quat2euler as jax_quat2euler,
    quat2mat as jax_quat2mat,
    rotVecMatT as jax_rotVecMatT,
    rotVecMat as jax_rotVecMat,
    rotVecQuat as jax_rotVecQuat,
    quat2euler_intrinsic as jax_quat2euler_intrinsic,
)


class TestQuatMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize JAX"""
        # Ensure JAX is initialized on CPU
        cls.device = jax.devices("cpu")[0]

    def setUp(self):
        # Define some test quaternions with explicit dtype
        self.test_cases = [
            # Identity quaternion
            (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32),
            ),
            # 90-degree rotations around different axes
            (
                np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32),
                np.array([0.7071, 0.0, 0.7071, 0.0], dtype=np.float32),
            ),
            # Arbitrary quaternions
            (
                np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32),
            ),
            # Zero rotation
            (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
        ]

        # Additional test cases specifically for negQuat
        self.neg_test_cases = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),  # Pure i
            np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),  # Pure j
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # Pure k
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),  # Equal components
            np.array(
                [0.7071, 0.7071, 0.0, 0.0], dtype=np.float32
            ),  # 90-degree rotation
        ]

        # Add test cases for quat2Vel
        self.vel_test_cases = [
            # No rotation (identity quaternion)
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            # 90-degree rotation around x-axis
            np.array(
                [0.7071067811865476, 0.7071067811865476, 0.0, 0.0], dtype=np.float32
            ),
            # 90-degree rotation around y-axis
            np.array(
                [0.7071067811865476, 0.0, 0.7071067811865476, 0.0], dtype=np.float32
            ),
            # 90-degree rotation around z-axis
            np.array(
                [0.7071067811865476, 0.0, 0.0, 0.7071067811865476], dtype=np.float32
            ),
            # 45-degree rotation around x-axis
            np.array(
                [0.9238795325112867, 0.3826834323650898, 0.0, 0.0], dtype=np.float32
            ),
            # Arbitrary rotation
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        ]

        # Expected results for dt=1.0
        self.vel_expected_results = [
            # For identity quaternion: no rotation
            (0.0, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
            # For 90-degree rotation around x: pi/2 speed, [1,0,0] axis
            (np.pi / 2, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            # For 90-degree rotation around y: pi/2 speed, [0,1,0] axis
            (np.pi / 2, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            # For 90-degree rotation around z: pi/2 speed, [0,0,1] axis
            (np.pi / 2, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            # For 45-degree rotation around x: pi/4 speed, [1,0,0] axis
            (np.pi / 4, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            # For arbitrary rotation: specific values
            (
                2 * np.arccos(0.5),
                np.array([1.0, 1.0, 1.0] / np.sqrt(3), dtype=np.float32),
            ),
        ]

        # Add test cases for diffQuat
        self.diff_test_cases = [
            # Same quaternions (should result in identity)
            (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
            # 90-degree difference around x-axis
            (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity
                np.array(
                    [0.7071067811865476, 0.7071067811865476, 0.0, 0.0], dtype=np.float32
                ),
            ),
            # 90-degree difference around y-axis
            (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity
                np.array(
                    [0.7071067811865476, 0.0, 0.7071067811865476, 0.0], dtype=np.float32
                ),
            ),
            # 180-degree difference around z-axis
            (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
            # Arbitrary rotations
            (
                np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32),
            ),
        ]

        # Expected results for diffQuat
        self.diff_expected_results = [
            np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),  # Identity (no difference)
            np.array(
                [0.7071067811865476, 0.7071067811865476, 0.0, 0.0], dtype=np.float32
            ),  # 90-deg x
            np.array(
                [0.7071067811865476, 0.0, 0.7071067811865476, 0.0], dtype=np.float32
            ),  # 90-deg y
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # 180-deg z
            np.array(
                [0.0, -1.0, 0.0, 0.0], dtype=np.float32
            ),  # Result for arbitrary case
        ]

        # Add test cases for quatDiff2Vel
        self.diff2vel_test_cases = [
            # No rotation difference
            (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),  # Identity
            # 90-degree rotation difference around x-axis
            (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity
                np.array(
                    [0.7071067811865476, 0.7071067811865476, 0.0, 0.0], dtype=np.float32
                ),
            ),
            # 90-degree rotation difference around y-axis
            (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity
                np.array(
                    [0.7071067811865476, 0.0, 0.7071067811865476, 0.0], dtype=np.float32
                ),
            ),
            # 180-degree rotation difference around z-axis
            (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
            # Arbitrary rotation difference
            (
                np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32),
            ),
        ]

        # Expected results for quatDiff2Vel (dt=1.0)
        self.diff2vel_expected_results = [
            # No rotation: zero angular velocity
            (0.0, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
            # 90-degree rotation around x: pi/2 speed, [1,0,0] axis
            (np.pi / 2, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            # 90-degree rotation around y: pi/2 speed, [0,1,0] axis
            (np.pi / 2, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            # 180-degree rotation around z: pi speed, [0,0,1] axis
            (np.pi, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            # Arbitrary rotation: specific values
            (np.pi, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ]

        # Add test cases for axis_angle2quat
        self.axis_angle_test_cases = [
            # No rotation
            (np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0),
            # 90-degree rotations around principal axes
            (np.array([1.0, 0.0, 0.0], dtype=np.float32), np.pi / 2),
            (np.array([0.0, 1.0, 0.0], dtype=np.float32), np.pi / 2),
            (np.array([0.0, 0.0, 1.0], dtype=np.float32), np.pi / 2),
            # 180-degree rotation
            (np.array([0.0, 0.0, 1.0], dtype=np.float32), np.pi),
            # 45-degree rotation around arbitrary axis
            (np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3), np.pi / 4),
        ]

        # Expected results for axis_angle2quat
        self.axis_angle_expected_results = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array(
                [0.7071067811865476, 0.7071067811865476, 0.0, 0.0], dtype=np.float32
            ),
            np.array(
                [0.7071067811865476, 0.0, 0.7071067811865476, 0.0], dtype=np.float32
            ),
            np.array(
                [0.7071067811865476, 0.0, 0.0, 0.7071067811865476], dtype=np.float32
            ),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.92388, 0.220942, 0.220942, 0.220942], dtype=np.float32),
        ]

        # Add test cases for euler2mat
        self.euler_test_cases = [
            # No rotation
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            # 90-degree rotations around single axes
            np.array([np.pi / 2, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, np.pi / 2, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, np.pi / 2], dtype=np.float32),
            # Combined rotations
            np.array([np.pi / 4, np.pi / 4, 0.0], dtype=np.float32),
            np.array([np.pi / 3, np.pi / 6, np.pi / 2], dtype=np.float32),
        ]

        # Add test cases for euler2quat
        self.euler2quat_test_cases = [
            # No rotation
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            # Single axis rotations - 90 degrees
            np.array([np.pi / 2, 0.0, 0.0], dtype=np.float32),  # Roll
            np.array([0.0, np.pi / 2, 0.0], dtype=np.float32),  # Pitch
            np.array([0.0, 0.0, np.pi / 2], dtype=np.float32),  # Yaw
            # Single axis rotations - 45 degrees
            np.array([np.pi / 4, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, np.pi / 4, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, np.pi / 4], dtype=np.float32),
            # Combined rotations
            np.array([np.pi / 4, np.pi / 4, 0.0], dtype=np.float32),
            np.array([np.pi / 4, 0.0, np.pi / 4], dtype=np.float32),
            np.array([0.0, np.pi / 4, np.pi / 4], dtype=np.float32),
            np.array([np.pi / 6, np.pi / 4, np.pi / 3], dtype=np.float32),
        ]

        # Expected results for euler2quat
        self.euler2quat_expected_results = [
            # Identity quaternion for no rotation
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            # 90-degree rotations around principal axes
            np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float32),  # Roll
            np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float32),  # Pitch
            np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32),  # Yaw
            # 45-degree rotations around principal axes
            np.array([0.9238795, 0.3826834, 0.0, 0.0], dtype=np.float32),
            np.array([0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float32),
            np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float32),
            # Combined rotations (pre-computed values)
            np.array([0.853553, 0.353553, 0.353553, -0.146447], dtype=np.float32),
            np.array([0.853553, 0.353553, 0.146447, 0.353553], dtype=np.float32),
            np.array([0.85355335, -0.14644663, 0.3535534, 0.3535534], dtype=np.float32),
            np.array([0.822363, 0.02226, 0.43968, 0.360423], dtype=np.float32),
        ]

        # Add test cases for mat2euler
        self.mat2euler_test_cases = [
            # Identity matrix (no rotation)
            np.eye(3, dtype=np.float32),
            # 90-degree rotations around principal axes
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
            ),  # 90° around x
            np.array(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
            ),  # 90° around y
            np.array(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            ),  # 90° around z
            # 45-degree rotations
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.7071068, -0.7071068],
                    [0.0, 0.7071068, 0.7071068],
                ],
                dtype=np.float32,
            ),  # 45° around x
            np.array(
                [
                    [0.7071068, 0.0, 0.7071068],
                    [0.0, 1.0, 0.0],
                    [-0.7071068, 0.0, 0.7071068],
                ],
                dtype=np.float32,
            ),  # 45° around y
            np.array(
                [
                    [0.7071068, -0.7071068, 0.0],
                    [0.7071068, 0.7071068, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),  # 45° around z
            # Combined rotations
            np.array(
                [
                    [0.3536, -0.6124, 0.7071],
                    [0.866007, 0.500033, 0.0],
                    [-0.353576, 0.612359, 0.707107],
                ],
                dtype=np.float32,
            ),  # Combined rotation
        ]

        # Expected Euler angles for each rotation matrix
        self.mat2euler_expected_results = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),  # No rotation
            np.array([np.pi / 2, 0.0, 0.0], dtype=np.float32),  # 90° x
            np.array([0.0, np.pi / 2, 0.0], dtype=np.float32),  # 90° y
            np.array([0.0, 0.0, np.pi / 2], dtype=np.float32),  # 90° z
            np.array([np.pi / 4, 0.0, 0.0], dtype=np.float32),  # 45° x
            np.array([0.0, np.pi / 4, 0.0], dtype=np.float32),  # 45° y
            np.array([0.0, 0.0, np.pi / 4], dtype=np.float32),  # 45° z
            np.array([0.0, np.pi / 4, np.pi / 3], dtype=np.float32),  # Combined
        ]

        # Add test cases for mat2quat
        self.mat2quat_test_cases = [
            # Identity matrix (no rotation)
            np.eye(3, dtype=np.float32),
            # 90-degree rotations around principal axes
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
            ),  # 90° around x
            np.array(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
            ),  # 90° around y
            np.array(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            ),  # 90° around z
            # 45-degree rotations
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.7071068, -0.7071068],
                    [0.0, 0.7071068, 0.7071068],
                ],
                dtype=np.float32,
            ),  # 45° around x
            np.array(
                [
                    [0.7071068, 0.0, 0.7071068],
                    [0.0, 1.0, 0.0],
                    [-0.7071068, 0.0, 0.7071068],
                ],
                dtype=np.float32,
            ),  # 45° around y
            np.array(
                [
                    [0.7071068, -0.7071068, 0.0],
                    [0.7071068, 0.7071068, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),  # 45° around z
            # Combined rotation
            np.array(
                [
                    [0.3536, -0.6124, 0.7071],
                    [0.866007, 0.500033, 0.0],
                    [-0.353576, 0.612359, 0.707107],
                ],
                dtype=np.float32,
            ),
        ]

        # Expected quaternions for each rotation matrix
        self.mat2quat_expected_results = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity
            np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float32),  # 90° x
            np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float32),  # 90° y
            np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32),  # 90° z
            np.array([0.9238795, 0.3826834, 0.0, 0.0], dtype=np.float32),  # 45° x
            np.array([0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float32),  # 45° y
            np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float32),  # 45° z
            np.array(
                [0.80011504, 0.19133107, 0.33140944, 0.46192655], dtype=np.float32
            ),  # Combined
        ]

        # Add test cases for quat2euler
        self.quat2euler_test_cases = [
            # Identity quaternion (no rotation)
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            # 90-degree rotations around principal axes
            np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float32),  # x-axis
            np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float32),  # y-axis
            np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32),  # z-axis
            # 45-degree rotations around principal axes
            np.array([0.9238795, 0.3826834, 0.0, 0.0], dtype=np.float32),  # x-axis
            np.array([0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float32),  # y-axis
            np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float32),  # z-axis
            # Combined rotation
            np.array(
                [0.80011504, 0.19133107, 0.33140944, 0.46192655], dtype=np.float32
            ),
        ]

        # Expected Euler angles for each quaternion
        self.quat2euler_expected_results = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),  # No rotation
            np.array([np.pi / 2, 0.0, 0.0], dtype=np.float32),  # 90° x
            np.array([0.0, np.pi / 2, 0.0], dtype=np.float32),  # 90° y
            np.array([0.0, 0.0, np.pi / 2], dtype=np.float32),  # 90° z
            np.array([np.pi / 4, 0.0, 0.0], dtype=np.float32),  # 45° x
            np.array([0.0, np.pi / 4, 0.0], dtype=np.float32),  # 45° y
            np.array([0.0, 0.0, np.pi / 4], dtype=np.float32),  # 45° z
            np.array([0.0, np.pi / 4, np.pi / 3], dtype=np.float32),  # Combined
        ]

        # Add test cases for quat2mat
        self.quat2mat_test_cases = [
            # Identity quaternion (no rotation)
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            # 90-degree rotations around principal axes
            np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float32),  # x-axis
            np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float32),  # y-axis
            np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32),  # z-axis
            # 45-degree rotations around principal axes
            np.array([0.9238795, 0.3826834, 0.0, 0.0], dtype=np.float32),  # x-axis
            np.array([0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float32),  # y-axis
            np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float32),  # z-axis
            # Combined rotation
            np.array(
                [0.80011504, 0.19133107, 0.33140944, 0.46192655], dtype=np.float32
            ),
        ]

        # Expected rotation matrices for each quaternion
        self.quat2mat_expected_results = [
            # Identity matrix
            np.eye(3, dtype=np.float32),
            # 90-degree rotation matrices around principal axes
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
            ),  # x-axis
            np.array(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
            ),  # y-axis
            np.array(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            ),  # z-axis
            # 45-degree rotation matrices
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.7071068, -0.7071068],
                    [0.0, 0.7071068, 0.7071068],
                ],
                dtype=np.float32,
            ),  # x-axis
            np.array(
                [
                    [0.7071068, 0.0, 0.7071068],
                    [0.0, 1.0, 0.0],
                    [-0.7071068, 0.0, 0.7071068],
                ],
                dtype=np.float32,
            ),  # y-axis
            np.array(
                [
                    [0.7071068, -0.7071068, 0.0],
                    [0.7071068, 0.7071068, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),  # z-axis
            # Combined rotation matrix
            np.array(
                [
                    [0.3536, -0.6124, 0.7071],
                    [0.866007, 0.500033, 0.0],
                    [-0.353576, 0.612359, 0.707107],
                ],
                dtype=np.float32,
            ),
        ]

        # Add test cases for rotVecMatT
        self.rotVecMatT_test_cases = [
            # Identity rotation (no change)
            (
                np.array([1.0, 0.0, 0.0], dtype=np.float32),  # x-axis vector
                np.eye(3, dtype=np.float32),
            ),  # identity matrix
            (
                np.array([0.0, 1.0, 0.0], dtype=np.float32),  # y-axis vector
                np.eye(3, dtype=np.float32),
            ),  # identity matrix
            # 90-degree rotations
            (
                np.array([1.0, 0.0, 0.0], dtype=np.float32),  # x-axis vector
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
            ),  # 90° around x
            (
                np.array([0.0, 1.0, 0.0], dtype=np.float32),  # y-axis vector
                np.array(
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                    dtype=np.float32,
                ),
            ),  # 90° around y
            # 45-degree rotation
            (
                np.array([1.0, 1.0, 1.0], dtype=np.float32),  # diagonal vector
                np.array(
                    [
                        [0.7071068, -0.7071068, 0.0],
                        [0.7071068, 0.7071068, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ),  # 45° around z
            # Arbitrary vector and rotation
            (
                np.array([0.5, -0.3, 0.8], dtype=np.float32),
                np.array(
                    [
                        [0.3536, -0.6124, 0.7071],
                        [0.866007, 0.500033, 0.0],
                        [-0.353576, 0.612359, 0.707107],
                    ],
                    dtype=np.float32,
                ),
            ),
        ]

        # Expected results for rotVecMatT
        self.rotVecMatT_expected_results = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # No change for identity
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # No change for identity
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # 90° x rotation
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # 90° y rotation
            np.array([1.414214, 0.0, 1.0], dtype=np.float32),  # 45° z rotation
            np.array(
                [-0.365863, 0.033677, 0.919236], dtype=np.float32
            ),  # Arbitrary rotation
        ]

        # Add test cases for rotVecMat
        self.rotVecMat_test_cases = [
            # Identity rotation (no change)
            (
                np.array([1.0, 0.0, 0.0], dtype=np.float32),  # x-axis vector
                np.eye(3, dtype=np.float32),
            ),  # identity matrix
            (
                np.array([0.0, 1.0, 0.0], dtype=np.float32),  # y-axis vector
                np.eye(3, dtype=np.float32),
            ),  # identity matrix
            # 90-degree rotations
            (
                np.array([1.0, 0.0, 0.0], dtype=np.float32),  # x-axis vector
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
            ),  # 90° around x
            (
                np.array([0.0, 1.0, 0.0], dtype=np.float32),  # y-axis vector
                np.array(
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                    dtype=np.float32,
                ),
            ),  # 90° around y
            # 45-degree rotation
            (
                np.array([1.0, 1.0, 1.0], dtype=np.float32),  # diagonal vector
                np.array(
                    [
                        [0.7071068, -0.7071068, 0.0],
                        [0.7071068, 0.7071068, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ),  # 45° around z
            # Arbitrary vector and rotation
            (
                np.array([0.5, -0.3, 0.8], dtype=np.float32),
                np.array(
                    [
                        [0.3536, -0.6124, 0.7071],
                        [0.866007, 0.500033, 0.0],
                        [-0.353576, 0.612359, 0.707107],
                    ],
                    dtype=np.float32,
                ),
            ),
        ]

        # Expected results for rotVecMat
        self.rotVecMat_expected_results = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # No change for identity
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # No change for identity
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # 90° x rotation
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # 90° y rotation
            np.array([0.0, 1.414214, 1.0], dtype=np.float32),  # 45° z rotation
            np.array(
                [0.9262, 0.282994, 0.20519], dtype=np.float32
            ),  # Arbitrary rotation
        ]

        # Add test cases for rotVecQuat
        self.rotVecQuat_test_cases = [
            # Identity rotation (no change)
            (
                np.array([1.0, 0.0, 0.0], dtype=np.float32),  # x-axis vector
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),  # identity quaternion
            # 90-degree rotations
            (
                np.array([1.0, 0.0, 0.0], dtype=np.float32),  # x-axis vector
                np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float32),
            ),  # 90° around x
            (
                np.array([0.0, 1.0, 0.0], dtype=np.float32),  # y-axis vector
                np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float32),
            ),  # 90° around y
            # 45-degree rotation
            (
                np.array([1.0, 1.0, 1.0], dtype=np.float32),  # diagonal vector
                np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float32),
            ),  # 45° around z
            # Arbitrary vector and rotation
            (
                np.array([0.5, -0.3, 0.8], dtype=np.float32),
                np.array(
                    [0.80011504, 0.19133107, 0.33140944, 0.46192655], dtype=np.float32
                ),
            ),
        ]

        # Expected results for rotVecQuat (same as corresponding matrix rotations)
        self.rotVecQuat_expected_results = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # No change for identity
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # 90° x rotation
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # 90° y rotation
            np.array([0.0, 1.414214, 1.0], dtype=np.float32),  # 45° z rotation
            np.array(
                [0.9262, 0.282994, 0.20519], dtype=np.float32
            ),  # Arbitrary rotation
        ]

        # Add test cases for quat2euler_intrinsic
        self.quat2euler_intrinsic_test_cases = [
            # Identity quaternion (no rotation)
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            # 90-degree rotations around principal axes
            np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float32),  # x-axis
            np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float32),  # y-axis
            np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32),  # z-axis
            # 45-degree rotations around principal axes
            np.array([0.9238795, 0.3826834, 0.0, 0.0], dtype=np.float32),  # x-axis
            np.array([0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float32),  # y-axis
            np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float32),  # z-axis
        ]

        # Expected Euler angles for each quaternion
        self.quat2euler_intrinsic_expected_results = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),  # No rotation
            np.array([np.pi / 2, 0.0, 0.0], dtype=np.float32),  # 90° x
            np.array([np.pi, np.pi / 2, np.pi], dtype=np.float32),  # 90° y
            np.array([0.0, 0.0, np.pi / 2], dtype=np.float32),  # 90° z
            np.array([np.pi / 4, 0.0, 0.0], dtype=np.float32),  # 45° x
            np.array([0.0, np.pi / 4, 0.0], dtype=np.float32),  # 45° y
            np.array([0.0, 0.0, np.pi / 4], dtype=np.float32),  # 45° z
        ]

    def test_mulQuat_implementations_match(self):
        """Test that JAX and NumPy implementations give the same results"""
        try:
            for qa, qb in self.test_cases:
                # Convert inputs to the appropriate type
                qa_jax = jp.array(qa, dtype=jp.float32)
                qb_jax = jp.array(qb, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_mulQuat(qa, qb)
                result_jax = jax_mulQuat(qa_jax, qb_jax)

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results
                np.testing.assert_allclose(
                    result_np,
                    result_jax,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for qa={qa}, qb={qb}",
                )
        except Exception as e:
            print(f"Error in mulQuat test: {str(e)}")
            raise

    def test_negQuat_implementations_match(self):
        """Test that JAX and NumPy implementations of negQuat give the same results"""
        try:
            for q in self.neg_test_cases:
                # Convert input to JAX array
                print(f"Testing negQuat with input: {q} (type: {q.dtype})")
                q_jax = jp.array(q, dtype=jp.float32)
                print(f"JAX array: {q_jax} (type: {q_jax.dtype})")

                # Compute results from both implementations
                result_np = np_negQuat(q)
                print(f"NumPy result: {result_np}")

                result_jax = jax_negQuat(q_jax)
                print(f"JAX result: {result_jax}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results
                np.testing.assert_allclose(
                    result_np,
                    result_jax,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for q={q}",
                )
        except Exception as e:
            print(f"Error in negQuat test: {str(e)}")
            raise

    def test_negQuat_properties(self):
        """Test mathematical properties of quaternion negation"""
        for q in self.neg_test_cases:
            q_jax = jp.array(q)

            # Test double negation returns original quaternion
            neg_neg_q = jax_negQuat(jax_negQuat(q_jax))
            np.testing.assert_allclose(
                np.array(neg_neg_q),
                q,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Double negation failed for q={q}",
            )

            # Test that negation preserves norm
            neg_q = jax_negQuat(q_jax)
            orig_norm = jp.sqrt(jp.sum(q_jax * q_jax))
            neg_norm = jp.sqrt(jp.sum(neg_q * neg_q))
            np.testing.assert_allclose(
                float(orig_norm),
                float(neg_norm),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Norm not preserved for q={q}",
            )

    def test_mulQuat_properties(self):
        """Test mathematical properties of quaternion multiplication"""

        # Test identity quaternion property
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        identity_jax = jp.array(identity)

        for qa, _ in self.test_cases:
            qa_jax = jp.array(qa)

            # Identity * q = q
            result_jax = jax_mulQuat(identity_jax, qa_jax)
            np.testing.assert_allclose(
                np.array(result_jax),
                qa,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Identity property failed for q={qa}",
            )

            # q * Identity = q
            result_jax = jax_mulQuat(qa_jax, identity_jax)
            np.testing.assert_allclose(
                np.array(result_jax),
                qa,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Identity property failed for q={qa}",
            )

    def test_mulQuat_norm_preservation(self):
        """Test that quaternion multiplication preserves norm"""
        for qa, qb in self.test_cases:
            qa_jax = jp.array(qa)
            qb_jax = jp.array(qb)

            # Compute result
            result = jax_mulQuat(qa_jax, qb_jax)

            # Check that the result is still a unit quaternion
            norm = jp.sqrt(jp.sum(result * result))
            np.testing.assert_allclose(
                float(norm),
                1.0,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Norm not preserved for qa={qa}, qb={qb}",
            )

    def test_quat2Vel_implementations_match(self):
        """Test that JAX and NumPy implementations of quat2Vel give the same results"""
        try:
            for q, (expected_speed, expected_axis) in zip(
                self.vel_test_cases, self.vel_expected_results
            ):
                # Convert input to JAX array
                print(f"\nTesting quat2Vel with input: {q} (type: {q.dtype})")
                q_jax = jp.array(q, dtype=jp.float32)

                # Test with different dt values
                for dt in [1.0, 0.5, 0.1]:
                    # Compute results from both implementations
                    speed_np, axis_np = np_quat2Vel(q, dt)
                    speed_jax, axis_jax = jax_quat2Vel(q_jax, dt)

                    print(f"dt={dt}")
                    print(f"NumPy result: speed={speed_np}, axis={axis_np}")
                    print(f"JAX result: speed={speed_jax}, axis={axis_jax}")

                    # Convert JAX results to numpy for comparison
                    speed_jax = float(speed_jax)
                    axis_jax = np.array(axis_jax)

                    # Compare results
                    np.testing.assert_allclose(
                        speed_np,
                        speed_jax,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Speed doesn't match for q={q}, dt={dt}",
                    )
                    np.testing.assert_allclose(
                        axis_np,
                        axis_jax,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Axis doesn't match for q={q}, dt={dt}",
                    )

                    # For dt=1.0, also check against expected results
                    if dt == 1.0:
                        np.testing.assert_allclose(
                            speed_jax,
                            expected_speed,
                            rtol=1e-5,
                            atol=1e-5,
                            err_msg=f"Speed doesn't match expected for q={q}",
                        )
                        np.testing.assert_allclose(
                            np.abs(axis_jax),
                            np.abs(expected_axis),
                            rtol=1e-5,
                            atol=1e-5,
                            err_msg=f"Axis doesn't match expected for q={q}",
                        )

        except Exception as e:
            print(f"Error in quat2Vel test: {str(e)}")
            raise

    def test_diffQuat_implementations_match(self):
        """Test that JAX and NumPy implementations of diffQuat give the same results"""
        try:
            for (q1, q2), expected_result in zip(
                self.diff_test_cases, self.diff_expected_results
            ):
                # Convert inputs to JAX arrays
                print(f"\nTesting diffQuat with inputs: q1={q1}, q2={q2}")
                q1_jax = jp.array(q1, dtype=jp.float32)
                q2_jax = jp.array(q2, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_diffQuat(q1, q2)
                result_jax = jax_diffQuat(q1_jax, q2_jax)

                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_result}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results
                np.testing.assert_allclose(
                    result_np,
                    result_jax,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for q1={q1}, q2={q2}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    np.abs(result_jax),
                    np.abs(expected_result),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Result doesn't match expected for q1={q1}, q2={q2}",
                )

        except Exception as e:
            print(f"Error in diffQuat test: {str(e)}")
            raise

    def test_diffQuat_properties(self):
        """Test mathematical properties of quaternion difference"""
        try:
            for q1, q2 in self.diff_test_cases:
                q1_jax = jp.array(q1, dtype=jp.float32)
                q2_jax = jp.array(q2, dtype=jp.float32)

                # Property 1: diff(q, q) should be identity quaternion
                result = jax_diffQuat(q1_jax, q1_jax)
                identity = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
                np.testing.assert_allclose(
                    np.array(result),
                    identity,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Self-difference not identity for q={q1}",
                )

                # Property 2: Norm preservation
                result = jax_diffQuat(q1_jax, q2_jax)
                norm = jp.sqrt(jp.sum(result * result))
                np.testing.assert_allclose(
                    float(norm),
                    1.0,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Norm not preserved for q1={q1}, q2={q2}",
                )

                # Property 3: diff(q1,q2) * q1 = q2 (up to numerical precision)
                diff_result = jax_diffQuat(q1_jax, q2_jax)
                reconstructed = jax_mulQuat(diff_result, q1_jax)
                np.testing.assert_allclose(
                    np.abs(np.array(reconstructed)),
                    np.abs(q2),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Reconstruction failed for q1={q1}, q2={q2}",
                )

        except Exception as e:
            print(f"Error in diffQuat properties test: {str(e)}")
            raise

    def test_quatDiff2Vel_implementations_match(self):
        """Test that JAX and NumPy implementations of quatDiff2Vel give the same results"""
        try:
            for (q1, q2), (expected_speed, expected_axis) in zip(
                self.diff2vel_test_cases, self.diff2vel_expected_results
            ):
                # Convert inputs to JAX arrays
                print(f"\nTesting quatDiff2Vel with inputs: q1={q1}, q2={q2}")
                q1_jax = jp.array(q1, dtype=jp.float32)
                q2_jax = jp.array(q2, dtype=jp.float32)

                # Test with different dt values
                for dt in [1.0, 0.5, 0.1]:
                    # Compute results from both implementations
                    speed_np, axis_np = np_quatDiff2Vel(q1, q2, dt)
                    speed_jax, axis_jax = jax_quatDiff2Vel(q1_jax, q2_jax, dt)

                    print(f"dt={dt}")
                    print(f"NumPy result: speed={speed_np}, axis={axis_np}")
                    print(f"JAX result: speed={speed_jax}, axis={axis_jax}")

                    # Convert JAX results to numpy for comparison
                    speed_jax = float(speed_jax)
                    axis_jax = np.array(axis_jax)

                    # Compare results
                    np.testing.assert_allclose(
                        speed_np,
                        speed_jax,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Speed doesn't match for q1={q1}, q2={q2}, dt={dt}",
                    )
                    np.testing.assert_allclose(
                        np.abs(axis_np),
                        np.abs(axis_jax),
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Axis doesn't match for q1={q1}, q2={q2}, dt={dt}",
                    )

                    # For dt=1.0, also check against expected results
                    if dt == 1.0:
                        np.testing.assert_allclose(
                            speed_jax,
                            expected_speed,
                            rtol=1e-5,
                            atol=1e-5,
                            err_msg=f"Speed doesn't match expected for q1={q1}, q2={q2}",
                        )
                        np.testing.assert_allclose(
                            np.abs(axis_jax),
                            np.abs(expected_axis),
                            rtol=1e-5,
                            atol=1e-5,
                            err_msg=f"Axis doesn't match expected for q1={q1}, q2={q2}",
                        )

        except Exception as e:
            print(f"Error in quatDiff2Vel test: {str(e)}")
            raise

    def test_quatDiff2Vel_properties(self):
        """Test mathematical properties of quaternion difference to velocity conversion"""
        try:
            for q1, q2 in self.diff2vel_test_cases:
                q1_jax = jp.array(q1, dtype=jp.float32)
                q2_jax = jp.array(q2, dtype=jp.float32)

                # Property 1: Zero velocity for identical quaternions
                speed, axis = jax_quatDiff2Vel(q1_jax, q1_jax, dt=1.0)
                np.testing.assert_allclose(
                    float(speed),
                    0.0,
                    atol=1e-5,
                    err_msg=f"Non-zero speed for identical quaternions: q={q1}",
                )

                # Property 2: Scaling with dt
                speed1, axis1 = jax_quatDiff2Vel(q1_jax, q2_jax, dt=1.0)
                speed2, axis2 = jax_quatDiff2Vel(q1_jax, q2_jax, dt=2.0)
                np.testing.assert_allclose(
                    float(speed1),
                    2 * float(speed2),
                    rtol=1e-5,
                    err_msg=f"Speed scaling with dt failed for q1={q1}, q2={q2}",
                )
                np.testing.assert_allclose(
                    np.abs(np.array(axis1)),
                    np.abs(np.array(axis2)),
                    rtol=1e-5,
                    err_msg=f"Axis changed with dt for q1={q1}, q2={q2}",
                )

        except Exception as e:
            print(f"Error in quatDiff2Vel properties test: {str(e)}")
            raise

    def test_axis_angle2quat_implementations_match(self):
        """Test that JAX and NumPy implementations of axis_angle2quat give the same results"""
        try:
            for (axis, angle), expected_result in zip(
                self.axis_angle_test_cases, self.axis_angle_expected_results
            ):
                # Convert inputs to JAX arrays
                print(
                    f"\nTesting axis_angle2quat with inputs: axis={axis}, angle={angle}"
                )
                axis_jax = jp.array(axis, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_axis_angle2quat(axis, angle)
                result_jax = jax_axis_angle2quat(axis_jax, angle)

                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_result}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results
                np.testing.assert_allclose(
                    np.abs(result_np),
                    np.abs(result_jax),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for axis={axis}, angle={angle}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    np.abs(result_jax),
                    np.abs(expected_result),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Result doesn't match expected for axis={axis}, angle={angle}",
                )

        except Exception as e:
            print(f"Error in axis_angle2quat test: {str(e)}")
            raise

    def test_euler2mat_implementations_match(self):
        """Test that JAX and NumPy implementations of euler2mat give the same results"""
        try:
            for euler in self.euler_test_cases:
                # Convert input to JAX array
                print(f"\nTesting euler2mat with input: euler={euler}")
                euler_jax = jp.array(euler, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_euler2mat(euler)
                result_jax = jax_euler2mat(euler_jax)

                print(f"NumPy result:\n{result_np}")
                print(f"JAX result:\n{result_jax}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results
                np.testing.assert_allclose(
                    result_np,
                    result_jax,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for euler={euler}",
                )

                # Test orthogonality property
                mat_t = result_jax.T
                identity = np.eye(3, dtype=np.float32)
                np.testing.assert_allclose(
                    result_jax @ mat_t,
                    identity,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Matrix not orthogonal for euler={euler}",
                )

        except Exception as e:
            print(f"Error in euler2mat test: {str(e)}")
            raise

    def test_axis_angle2quat_properties(self):
        """Test mathematical properties of axis-angle to quaternion conversion"""
        try:
            for axis, angle in self.axis_angle_test_cases:
                axis_jax = jp.array(axis, dtype=jp.float32)

                # Property 1: Result should be a unit quaternion
                result = jax_axis_angle2quat(axis_jax, angle)
                norm = jp.sqrt(jp.sum(result * result))
                np.testing.assert_allclose(
                    float(norm),
                    1.0,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Result not unit quaternion for axis={axis}, angle={angle}",
                )

                # Property 2: Zero angle should give identity quaternion
                result_zero = jax_axis_angle2quat(axis_jax, 0.0)
                identity = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
                np.testing.assert_allclose(
                    np.array(result_zero),
                    identity,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Zero angle doesn't give identity for axis={axis}",
                )

        except Exception as e:
            print(f"Error in axis_angle2quat properties test: {str(e)}")
            raise

    def test_euler2mat_properties(self):
        """Test mathematical properties of Euler angles to rotation matrix conversion"""
        try:
            for euler in self.euler_test_cases:
                euler_jax = jp.array(euler, dtype=jp.float32)

                # Property 1: Determinant should be 1 (proper rotation)
                result = jax_euler2mat(euler_jax)
                det = jp.linalg.det(result)
                np.testing.assert_allclose(
                    float(det),
                    1.0,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Determinant not 1 for euler={euler}",
                )

                # Property 2: Zero angles should give identity matrix
                zero_euler = jp.zeros(3, dtype=jp.float32)
                result_zero = jax_euler2mat(zero_euler)
                identity = jp.eye(3, dtype=jp.float32)
                np.testing.assert_allclose(
                    np.array(result_zero),
                    identity,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg="Zero angles don't give identity matrix",
                )

        except Exception as e:
            print(f"Error in euler2mat properties test: {str(e)}")
            raise

    def test_mat2euler_implementations_match(self):
        """Test that JAX and NumPy implementations of mat2euler give the same results"""
        try:
            for mat, expected_euler in zip(
                self.mat2euler_test_cases, self.mat2euler_expected_results
            ):
                # Convert input to JAX array
                print(f"\nTesting mat2euler with input matrix:\n{mat}")
                mat_jax = jp.array(mat, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_mat2euler(mat)
                result_jax = jax_mat2euler(mat_jax)

                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_euler}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results
                # Note: We need to handle angle wrapping (e.g., -π and π are equivalent)
                np.testing.assert_allclose(
                    np.sin(result_np),
                    np.sin(result_jax),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for matrix:\n{mat}",
                )
                np.testing.assert_allclose(
                    np.cos(result_np),
                    np.cos(result_jax),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for matrix:\n{mat}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    np.sin(result_jax),
                    np.sin(expected_euler),
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Result doesn't match expected for matrix:\n{mat}",
                )
                np.testing.assert_allclose(
                    np.cos(result_jax),
                    np.cos(expected_euler),
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Result doesn't match expected for matrix:\n{mat}",
                )

        except Exception as e:
            print(f"Error in mat2euler test: {str(e)}")
            raise

    def test_mat2euler_properties(self):
        """Test mathematical properties of rotation matrix to Euler angles conversion"""
        try:
            for mat, expected_euler in zip(
                self.mat2euler_test_cases, self.mat2euler_expected_results
            ):
                mat_jax = jp.array(mat, dtype=jp.float32)

                # Property 1: Converting back to matrix should give original matrix
                euler = jax_mat2euler(mat_jax)
                reconstructed_mat = jax_euler2mat(euler)
                np.testing.assert_allclose(
                    np.array(reconstructed_mat),
                    mat,
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Reconstruction failed for matrix:\n{mat}",
                )

                # Property 2: Identity matrix should give zero angles
                if jp.allclose(mat_jax, jp.eye(3)):
                    np.testing.assert_allclose(
                        np.array(euler),
                        np.zeros(3),
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg="Identity matrix doesn't give zero angles",
                    )

        except Exception as e:
            print(f"Error in mat2euler properties test: {str(e)}")
            raise

    def test_mat2quat_implementations_match(self):
        """Test that JAX and NumPy implementations of mat2quat give the same results"""
        try:
            for mat, expected_quat in zip(
                self.mat2quat_test_cases, self.mat2quat_expected_results
            ):
                # Convert input to JAX array
                print(f"\nTesting mat2quat with input matrix:\n{mat}")
                mat_jax = jp.array(mat, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_mat2quat(mat)
                result_jax = jax_mat2quat(mat_jax)

                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_quat}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results (using absolute values due to possible sign differences)
                np.testing.assert_allclose(
                    np.abs(result_np),
                    np.abs(result_jax),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for matrix:\n{mat}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    np.abs(result_jax),
                    np.abs(expected_quat),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Result doesn't match expected for matrix:\n{mat}",
                )

        except Exception as e:
            print(f"Error in mat2quat test: {str(e)}")
            raise

    def test_mat2quat_properties(self):
        """Test mathematical properties of rotation matrix to quaternion conversion"""
        try:
            for mat, expected_quat in zip(
                self.mat2quat_test_cases, self.mat2quat_expected_results
            ):
                mat_jax = jp.array(mat, dtype=jp.float32)

                # Property 1: Result should be a unit quaternion
                result = jax_mat2quat(mat_jax)
                norm = jp.sqrt(jp.sum(result * result))
                np.testing.assert_allclose(
                    float(norm),
                    1.0,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Result not unit quaternion for matrix:\n{mat}",
                )

                # Property 2: Identity matrix should give identity quaternion
                if jp.allclose(mat_jax, jp.eye(3)):
                    identity_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
                    np.testing.assert_allclose(
                        np.abs(np.array(result)),
                        np.abs(identity_quat),
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg="Identity matrix doesn't give identity quaternion",
                    )

                # Property 3: Converting back to matrix should give original matrix
                quat = jax_mat2quat(mat_jax)
                reconstructed_mat = jax_quat2mat(quat)
                np.testing.assert_allclose(
                    np.array(reconstructed_mat),
                    mat,
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Reconstruction failed for matrix:\n{mat}",
                )

        except Exception as e:
            print(f"Error in mat2quat properties test: {str(e)}")
            raise

    def test_quat2euler_implementations_match(self):
        """Test that JAX and NumPy implementations of quat2euler give the same results"""
        try:
            for quat, expected_euler in zip(
                self.quat2euler_test_cases, self.quat2euler_expected_results
            ):
                # Convert input to JAX array
                print(f"\nTesting quat2euler with input quaternion: {quat}")
                quat_jax = jp.array(quat, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_quat2euler(quat)
                result_jax = jax_quat2euler(quat_jax)

                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_euler}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results (using sin/cos to handle angle wrapping)
                np.testing.assert_allclose(
                    np.sin(result_np),
                    np.sin(result_jax),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for quaternion: {quat}",
                )
                np.testing.assert_allclose(
                    np.cos(result_np),
                    np.cos(result_jax),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for quaternion: {quat}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    np.sin(result_jax),
                    np.sin(expected_euler),
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Result doesn't match expected for quaternion: {quat}",
                )
                np.testing.assert_allclose(
                    np.cos(result_jax),
                    np.cos(expected_euler),
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Result doesn't match expected for quaternion: {quat}",
                )

        except Exception as e:
            print(f"Error in quat2euler test: {str(e)}")
            raise

    def test_quat2euler_properties(self):
        """Test mathematical properties of quaternion to Euler angles conversion"""
        try:
            for quat, expected_euler in zip(
                self.quat2euler_test_cases, self.quat2euler_expected_results
            ):
                quat_jax = jp.array(quat, dtype=jp.float32)

                # Property 1: Converting back to quaternion should give original quaternion
                euler = jax_quat2euler(quat_jax)
                reconstructed_quat = jax_euler2quat(euler)

                # Compare quaternions (using absolute values due to possible sign differences)
                np.testing.assert_allclose(
                    np.abs(np.array(reconstructed_quat)),
                    np.abs(quat),
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Reconstruction failed for quaternion: {quat}",
                )

                # Property 2: Identity quaternion should give zero angles
                if jp.allclose(quat_jax, jp.array([1.0, 0.0, 0.0, 0.0])):
                    np.testing.assert_allclose(
                        np.array(euler),
                        np.zeros(3),
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg="Identity quaternion doesn't give zero angles",
                    )

        except Exception as e:
            print(f"Error in quat2euler properties test: {str(e)}")
            raise

    def test_quat2mat_implementations_match(self):
        """Test that JAX and NumPy implementations of quat2mat give the same results"""
        try:
            for quat, expected_mat in zip(
                self.quat2mat_test_cases, self.quat2mat_expected_results
            ):
                # Convert input to JAX array
                print(f"\nTesting quat2mat with input quaternion: {quat}")
                quat_jax = jp.array(quat, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_quat2mat(quat)
                result_jax = jax_quat2mat(quat_jax)

                print(f"NumPy result:\n{result_np}")
                print(f"JAX result:\n{result_jax}")
                print(f"Expected result:\n{expected_mat}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results
                np.testing.assert_allclose(
                    result_np,
                    result_jax,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for quaternion: {quat}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    result_jax,
                    expected_mat,
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Result doesn't match expected for quaternion: {quat}",
                )

        except Exception as e:
            print(f"Error in quat2mat test: {str(e)}")
            raise

    def test_quat2mat_properties(self):
        """Test mathematical properties of quaternion to rotation matrix conversion"""
        try:
            for quat, expected_mat in zip(
                self.quat2mat_test_cases, self.quat2mat_expected_results
            ):
                quat_jax = jp.array(quat, dtype=jp.float32)

                # Property 1: Result should be orthogonal (R * R^T = I)
                result = jax_quat2mat(quat_jax)
                result_t = jp.transpose(result)
                identity = jp.eye(3, dtype=jp.float32)
                np.testing.assert_allclose(
                    np.array(result @ result_t),
                    identity,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Matrix not orthogonal for quaternion: {quat}",
                )

                # Property 2: Determinant should be 1 (proper rotation)
                det = jp.linalg.det(result)
                np.testing.assert_allclose(
                    float(det),
                    1.0,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Determinant not 1 for quaternion: {quat}",
                )

                # Property 3: Identity quaternion should give identity matrix
                if jp.allclose(quat_jax, jp.array([1.0, 0.0, 0.0, 0.0])):
                    np.testing.assert_allclose(
                        np.array(result),
                        identity,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg="Identity quaternion doesn't give identity matrix",
                    )

                # Property 4: Converting back to quaternion should give original quaternion
                reconstructed_quat = jax_mat2quat(result)
                np.testing.assert_allclose(
                    np.abs(np.array(reconstructed_quat)),
                    np.abs(quat),
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Reconstruction failed for quaternion: {quat}",
                )

        except Exception as e:
            print(f"Error in quat2mat properties test: {str(e)}")
            raise

    def test_rotVecMatT_implementations_match(self):
        """Test that JAX and NumPy implementations of rotVecMatT give the same results"""
        try:
            for (vec, mat), expected_result in zip(
                self.rotVecMatT_test_cases, self.rotVecMatT_expected_results
            ):
                # Convert inputs to JAX arrays
                print(f"\nTesting rotVecMatT with inputs:\nvec={vec}\nmat=\n{mat}")
                vec_jax = jp.array(vec, dtype=jp.float32)
                mat_jax = jp.array(mat, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_rotVecMatT(vec, mat)
                result_jax = jax_rotVecMatT(vec_jax, mat_jax)

                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_result}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results
                np.testing.assert_allclose(
                    result_np,
                    result_jax,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for vec={vec}, mat=\n{mat}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    result_jax,
                    expected_result,
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Result doesn't match expected for vec={vec}, mat=\n{mat}",
                )

        except Exception as e:
            print(f"Error in rotVecMatT test: {str(e)}")
            raise

    def test_rotVecMatT_properties(self):
        """Test mathematical properties of vector rotation by matrix transpose"""
        try:
            for (vec, mat), expected_result in zip(
                self.rotVecMatT_test_cases, self.rotVecMatT_expected_results
            ):
                vec_jax = jp.array(vec, dtype=jp.float32)
                mat_jax = jp.array(mat, dtype=jp.float32)

                # Property 1: Length preservation
                result = jax_rotVecMatT(vec_jax, mat_jax)
                orig_length = jp.sqrt(jp.sum(vec_jax * vec_jax))
                result_length = jp.sqrt(jp.sum(result * result))
                np.testing.assert_allclose(
                    float(result_length),
                    float(orig_length),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Length not preserved for vec={vec}, mat=\n{mat}",
                )

                # Property 2: Identity matrix should not change the vector
                if jp.allclose(mat_jax, jp.eye(3)):
                    np.testing.assert_allclose(
                        np.array(result),
                        vec,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg="Identity matrix changed the vector",
                    )

                # Property 3: Double rotation by mat and mat.T should give original vector
                # First rotate by mat.T, then by mat
                intermediate = jax_rotVecMatT(vec_jax, mat_jax)
                final = jax_rotVecMat(intermediate, mat_jax)
                np.testing.assert_allclose(
                    np.array(final),
                    vec,
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Double rotation failed for vec={vec}, mat=\n{mat}",
                )

        except Exception as e:
            print(f"Error in rotVecMatT properties test: {str(e)}")
            raise

    def test_rotVecMat_implementations_match(self):
        """Test that JAX and NumPy implementations of rotVecMat give the same results"""
        try:
            for (vec, mat), expected_result in zip(
                self.rotVecMat_test_cases, self.rotVecMat_expected_results
            ):
                # Convert inputs to JAX arrays
                print(f"\nTesting rotVecMat with inputs:\nvec={vec}\nmat=\n{mat}")
                vec_jax = jp.array(vec, dtype=jp.float32)
                mat_jax = jp.array(mat, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_rotVecMat(vec, mat)
                result_jax = jax_rotVecMat(vec_jax, mat_jax)

                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_result}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results
                np.testing.assert_allclose(
                    result_np,
                    result_jax,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for vec={vec}, mat=\n{mat}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    result_jax,
                    expected_result,
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Result doesn't match expected for vec={vec}, mat=\n{mat}",
                )

        except Exception as e:
            print(f"Error in rotVecMat test: {str(e)}")
            raise

    def test_rotVecQuat_implementations_match(self):
        """Test that JAX and NumPy implementations of rotVecQuat give the same results"""
        try:
            for (vec, quat), expected_result in zip(
                self.rotVecQuat_test_cases, self.rotVecQuat_expected_results
            ):
                # Convert inputs to JAX arrays
                print(f"\nTesting rotVecQuat with inputs:\nvec={vec}\nquat={quat}")
                vec_jax = jp.array(vec, dtype=jp.float32)
                quat_jax = jp.array(quat, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_rotVecQuat(vec, quat)
                result_jax = jax_rotVecQuat(vec_jax, quat_jax)

                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_result}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results
                np.testing.assert_allclose(
                    result_np,
                    result_jax,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for vec={vec}, quat={quat}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    result_jax,
                    expected_result,
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Result doesn't match expected for vec={vec}, quat={quat}",
                )

        except Exception as e:
            print(f"Error in rotVecQuat test: {str(e)}")
            raise

    def test_rotation_properties(self):
        """Test mathematical properties of vector rotations"""
        try:
            # Test that rotVecQuat gives same results as rotVecMat with equivalent rotation
            for (vec, quat), expected_result in zip(
                self.rotVecQuat_test_cases, self.rotVecQuat_expected_results
            ):
                vec_jax = jp.array(vec, dtype=jp.float32)
                quat_jax = jp.array(quat, dtype=jp.float32)

                # Convert quaternion to matrix
                mat_jax = jax_quat2mat(quat_jax)

                # Rotate vector using both methods
                result_quat = jax_rotVecQuat(vec_jax, quat_jax)
                result_mat = jax_rotVecMat(vec_jax, mat_jax)

                # Results should match
                np.testing.assert_allclose(
                    np.array(result_quat),
                    np.array(result_mat),
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Quaternion and matrix rotations don't match for vec={vec}",
                )

                # Length preservation
                orig_length = jp.sqrt(jp.sum(vec_jax * vec_jax))
                result_length = jp.sqrt(jp.sum(result_quat * result_quat))
                np.testing.assert_allclose(
                    float(result_length),
                    float(orig_length),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Length not preserved for vec={vec}",
                )

                # Identity rotation should not change vector
                if jp.allclose(quat_jax, jp.array([1.0, 0.0, 0.0, 0.0])):
                    np.testing.assert_allclose(
                        np.array(result_quat),
                        vec,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg="Identity rotation changed the vector",
                    )

        except Exception as e:
            print(f"Error in rotation properties test: {str(e)}")
            raise

    def test_quat2euler_intrinsic_implementations_match(self):
        """Test that JAX and NumPy implementations of quat2euler_intrinsic give the same results"""
        try:
            for quat, expected_euler in zip(
                self.quat2euler_intrinsic_test_cases,
                self.quat2euler_intrinsic_expected_results,
            ):
                # Convert input to JAX array
                print(f"\nTesting quat2euler_intrinsic with input quaternion: {quat}")
                quat_jax = jp.array(quat, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_quat2euler_intrinsic(quat)
                result_jax = jax_quat2euler_intrinsic(quat_jax)

                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_euler}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results (using sin/cos to handle angle wrapping)
                np.testing.assert_allclose(
                    np.sin(result_np),
                    np.sin(result_jax),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for quaternion: {quat}",
                )
                np.testing.assert_allclose(
                    np.cos(result_np),
                    np.cos(result_jax),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for quaternion: {quat}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    np.sin(result_jax),
                    np.sin(expected_euler),
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Result doesn't match expected for quaternion: {quat}",
                )
                np.testing.assert_allclose(
                    np.cos(result_jax),
                    np.cos(expected_euler),
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Result doesn't match expected for quaternion: {quat}",
                )

        except Exception as e:
            print(f"Error in quat2euler_intrinsic test: {str(e)}")
            raise

    def test_quat2euler_intrinsic_properties(self):
        """Test mathematical properties of quaternion to intrinsic Euler angles conversion"""
        try:
            for quat, expected_euler in zip(
                self.quat2euler_intrinsic_test_cases,
                self.quat2euler_intrinsic_expected_results,
            ):
                quat_jax = jp.array(quat, dtype=jp.float32)

                # Property 1: Converting back to quaternion should give original quaternion
                euler = jax_quat2euler_intrinsic(quat_jax)
                reconstructed_quat = jax_euler2quat(euler)

                # Compare quaternions (using absolute values due to possible sign differences)
                np.testing.assert_allclose(
                    np.abs(np.array(reconstructed_quat)),
                    np.abs(quat),
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Reconstruction failed for quaternion: {quat}",
                )

                # Property 2: Identity quaternion should give zero angles
                if jp.allclose(quat_jax, jp.array([1.0, 0.0, 0.0, 0.0])):
                    np.testing.assert_allclose(
                        np.array(euler),
                        np.zeros(3),
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg="Identity quaternion doesn't give zero angles",
                    )

                # Property 3: Angles should be in valid ranges
                euler_np = np.array(euler)
                self.assertTrue(
                    np.all(euler_np >= -np.pi) and np.all(euler_np <= np.pi),
                    f"Euler angles out of range [-π, π] for quaternion: {quat}",
                )

        except Exception as e:
            print(f"Error in quat2euler_intrinsic properties test: {str(e)}")
            raise

    def test_euler2quat_implementations_match(self):
        """Test that JAX and NumPy implementations of intrinsic_euler2quat give the same results"""
        try:
            for euler, expected_quat in zip(
                self.euler2quat_test_cases, self.euler2quat_expected_results
            ):
                # Convert input to JAX array
                print(f"\nTesting euler2quat with input Euler angles: {euler}")
                euler_jax = jp.array(euler, dtype=jp.float32)

                # Compute results from both implementations
                result_np = np_euler2quat(euler)
                result_jax = jax_euler2quat(euler_jax)

                print(f"NumPy result: {result_np}")
                print(f"JAX result: {result_jax}")
                print(f"Expected result: {expected_quat}")

                # Convert JAX result to numpy for comparison
                result_jax = np.array(result_jax)

                # Compare results (using absolute values due to possible sign differences)
                np.testing.assert_allclose(
                    np.abs(result_np),
                    np.abs(result_jax),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Results don't match for Euler angles: {euler}",
                )

                # Compare with expected results
                np.testing.assert_allclose(
                    np.abs(result_jax),
                    np.abs(expected_quat),
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Result doesn't match expected for Euler angles: {euler}",
                )

        except Exception as e:
            print(f"Error in euler2quat test: {str(e)}")
            raise

    def test_euler2quat_properties(self):
        """Test mathematical properties of Euler angles to quaternion conversion"""
        try:
            for euler, expected_quat in zip(
                self.euler2quat_test_cases, self.euler2quat_expected_results
            ):
                euler_jax = jp.array(euler, dtype=jp.float32)

                # Property 1: Result should be a unit quaternion
                result = jax_euler2quat(euler_jax)
                norm = jp.sqrt(jp.sum(result * result))
                np.testing.assert_allclose(
                    float(norm),
                    1.0,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Result not unit quaternion for Euler angles: {euler}",
                )

                # Property 2: Zero angles should give identity quaternion
                if jp.allclose(euler_jax, jp.zeros(3)):
                    identity_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
                    np.testing.assert_allclose(
                        np.abs(np.array(result)),
                        np.abs(identity_quat),
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg="Zero angles don't give identity quaternion",
                    )

                # Property 3: Converting back to Euler angles should give original angles
                reconstructed_euler = jax_quat2euler_intrinsic(result)
                np.testing.assert_allclose(
                    np.sin(np.array(reconstructed_euler)),
                    np.sin(euler),
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Reconstruction failed for Euler angles: {euler}",
                )
                print(f"Reconstructed euler: {reconstructed_euler}")
                print(f"Euler: {euler}")
                np.testing.assert_allclose(
                    np.cos(np.array(reconstructed_euler)),
                    np.cos(euler),
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Reconstruction failed for Euler angles: {euler}",
                )

        except Exception as e:
            print(f"Error in euler2quat properties test: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
