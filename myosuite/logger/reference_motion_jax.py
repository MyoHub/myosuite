import enum
from typing import Union
import collections
from jax import numpy as jp
import jax
import jax.random as jrandom
import pickle
from myosuite.utils.quat_math_jax import euler2quat

# Time precision to use. Avoids rounding/resolution errors during comparisons
_TIME_PRECISION = 4

# Reference structure
ReferenceStruct = collections.namedtuple(
    "ReferenceStruct",
    [
        "time",  # float(N)
        "robot",  # shape(N, n_robot_jnt) ==> robot trajectory
        "robot_vel",  # shape(N, n_robot_jnt) ==> robot velocity
        "object",  # shape(M, n_objects_jnt) ==> object trajectory
        "robot_init",  # shape(n_objects_jnt) ==> initial robot pose (can be different from robot(0,n_robot_jnt))
        "object_init",  # shape(n_objects_jnt) ==> initial object (can be different from object(0,n_object_jnt))
    ],
)


# Reference type
class ReferenceType(enum.Enum):
    """Reference types"""

    FIXED = 0
    RANDOM = 1
    TRACK = 2


# Reference motion
class ReferenceMotion:
    def __init__(
        self, reference_data: Union[str, dict], motion_extrapolation: bool = False
    ):
        """
        Reference Type
            Fixed  :: N==1, M==1  :: input is <Fixed target dict>
            Random :: N==2, M==2  :: input is <Randomization range dict> :: ind0:low_limit, ind1:high_limit
            Track  :: N>2 or M>2  :: input as <Motion file to be tracked>
        """
        self.motion_extrapolation = motion_extrapolation

        # load reference
        self.reference = self.load(reference_data)
        # check reference for format
        self.check_format(self.reference)
        self.reference["time"] = jp.around(
            self.reference["time"], _TIME_PRECISION
        )  # round to help with comparisons
        robot_shape = (
            self.reference["robot"].shape
            if self.reference["robot"] is not None
            else (0, 0)
        )
        object_shape = (
            self.reference["object"].shape
            if self.reference["object"] is not None
            else (0, 0)
        )
        self.robot_dim = robot_shape[1]
        self.object_dim = object_shape[1]

        # identify type
        if robot_shape[0] > 2 or object_shape[0] > 2:
            self.type = ReferenceType.TRACK
        elif robot_shape[0] == 2 or object_shape[0] == 2:
            self.type = ReferenceType.RANDOM
        elif robot_shape[0] == 1 or object_shape[0] == 1:
            self.type = ReferenceType.FIXED
        else:
            raise ValueError("Reference values not per specs")

        # identify length
        self.robot_horizon = robot_shape[0]
        self.object_horizon = object_shape[0]
        self.horizon = max(robot_shape[0], object_shape[0])

        # Add missing values
        if self.type == ReferenceType.RANDOM:
            # use mean postures from the randomization range
            if "robot_init" not in self.reference.keys():
                self.reference["robot_init"] = jp.mean(self.reference["robot"], axis=0)
            if "object_init" not in self.reference.keys():
                self.reference["object_init"] = jp.mean(self.reference["object"][0])
        else:
            # use first timeset
            if "robot_init" not in self.reference.keys():
                self.reference["robot_init"] = self.reference["robot"][0]
            if "object_init" not in self.reference.keys():
                self.reference["object_init"] = self.reference["object"][0]

        # cache to help with finding index
        self.index_cache = 0

    def check_format(self, reference):
        """
        Check reference format
        """
        if reference["robot"] is not None:
            assert (
                reference["robot"].ndim == 2
            ), "Check robot reference, must be shape(N, n_robot_jnt)"
        if reference["object"] is not None:
            assert (
                reference["object"].ndim == 2
            ), "Check object reference, must be shape(N, n_object_jnt)"
        if reference["robot_init"] is not None:
            assert (
                reference["robot_init"].ndim == 1
            ), "Check robot_init reference, must be shape(n_robot_jnt)"
        if reference["robot"] is not None and reference["robot_init"] is not None:
            assert (
                reference["robot_init"].shape[0] == reference["robot"].shape[1]
            ), "n_robot_jnt different between motion and init"
        if reference["object_init"] is not None:
            assert (
                reference["object_init"].ndim == 1
            ), "Check object_init reference, must be shape(n_object_jnt)"
        if reference["object_init"] is not None and reference["object"] is not None:
            assert (
                reference["object_init"].shape[0] == reference["object"].shape[1]
            ), "n_object_jnt different between motion and init"

    def load(self, reference_data):
        """
        Load reference motion
        """
        if isinstance(reference_data, str):
            if reference_data.endswith("npz"):
                reference = {k: jp.array(v) for k, v in jp.load(reference_data).items()}
            elif reference_data.endswith(("pkl", "pickle")):
                with open(reference_data, "rb") as data:
                    reference = pickle.load(data)
        elif isinstance(reference_data, dict):
            reference = reference_data.copy()
        else:
            raise TypeError("Unknown reference type")

        # resolve values
        assert "time" in reference.keys(), "Missing key (time) in reference"
        reference.setdefault(
            "robot_init", reference["robot"][0] if "robot" in reference else None
        )
        reference.setdefault(
            "object_init", reference["object"][0] if "object" in reference else None
        )

        # Handle conversion from euler to quaternion format for object_init if needed
        if reference["object_init"] is not None and "object" in reference and reference["object"] is not None:
            if reference["object_init"].shape[0] == 6 and reference["object"].shape[1] == 7:
                # Convert [x, y, z, euler_x, euler_y, euler_z] to [x, y, z, quat_w, quat_x, quat_y, quat_z]
                pos = reference["object_init"][:3]
                euler = reference["object_init"][3:]
                quat = euler2quat(euler)
                reference["object_init"] = jp.concatenate([pos, quat])

        result = ReferenceStruct(
            time=jp.array(reference["time"]),
            robot=reference.get("robot"),
            robot_vel=reference.get("robot_vel"),
            object=reference.get("object"),
            robot_init=reference["robot_init"],
            object_init=reference["object_init"],
        )._asdict()

        return result

    def find_timeslot_in_reference(self, time):
        """
        Find the timeslot interval for the provided time in the reference motion.
        """
        time = jp.around(time, _TIME_PRECISION)
        
        # Use jax.lax.cond to handle the conditional logic
        def fixed_case(_):
            return (0, 0)

        def extrapolation_case(_):
            return (self.horizon - 1, self.horizon - 1)

        def normal_case(_):
            # Use jax.lax.cond for conditional logic
            def check_current_index(_):
                return (self.index_cache, self.index_cache)

            def check_next_index(_):
                return (self.index_cache + 1, self.index_cache + 1)

            def search_index(_):
                new_index = jp.searchsorted(self.reference["time"], time, side="right") - 1
                return jax.lax.cond(
                    time == self.reference["time"][new_index],
                    lambda _: (new_index, new_index),
                    lambda _: (new_index, new_index + 1),
                    operand=None
                )

            return jax.lax.cond(
                time == self.reference["time"][self.index_cache],
                check_current_index,
                lambda _: jax.lax.cond(
                    (self.index_cache < (self.horizon - 1)) & (time == self.reference["time"][self.index_cache + 1]),
                    check_next_index,
                    search_index,
                    operand=None
                ),
                operand=None
            )

        # Use jax.lax.cond to handle the top-level conditional logic
        return jax.lax.cond(
            self.type == ReferenceType.FIXED,
            fixed_case,
            lambda _: jax.lax.cond(
                self.motion_extrapolation & (time >= self.reference["time"][-1]),
                extrapolation_case,
                normal_case,
                operand=None
            ),
            operand=None
        )

    def reset(self):
        self.index_cache = 0

    def get_init(self):
        """Return the initial posture of the robot and the object."""
        result = self.reference["robot_init"], self.reference["object_init"]

        return result

    def get_reference(self, time):
        """
        Return the reference at the given time, with linear interpolation if needed.
        """
        if self.type == ReferenceType.FIXED:
            robot_ref = self.reference["robot"][0]
            robot_vel_ref = self.reference["robot_vel"][0]
            object_ref = self.reference["object"][0]
        elif self.type == ReferenceType.RANDOM:
            rng_key, rng1, rng2 = jrandom.split(jax.random.PRNGKey(0), 3)
            robot_ref = jrandom.uniform(
                rng1,
                shape=self.reference["robot"][
                    0, :
                ].shape,  # Specify the shape explicitly
                minval=self.reference["robot"][0, :],
                maxval=self.reference["robot"][1, :],
            )
            robot_vel_ref = jrandom.uniform(
                rng2,
                shape=self.reference["robot_vel"][
                    0, :
                ].shape,  # Specify the shape explicitly
                minval=self.reference["robot_vel"][0, :],
                maxval=self.reference["robot_vel"][1, :],
            )
            object_ref = jrandom.uniform(
                rng_key,
                shape=self.reference["object"][0, :].shape,
                minval=self.reference["object"][0, :],
                maxval=self.reference["object"][1, :],
            )
        elif self.type == ReferenceType.TRACK:
            ind, ind_next = self.find_timeslot_in_reference(time=time)
            # Use jax.lax.cond to handle the conditional logic
            def exact_frame_case(_):
                # Exact frame[time] found for reference
                robot_ref = (
                    self.reference["robot"][ind]
                    if self.robot_horizon > 1
                    else self.reference["robot"][0]
                )
                robot_vel_ref = (
                    None
                    if self.reference["robot_vel"] is None
                    else self.reference["robot_vel"][ind]
                )
                object_ref = (
                    None
                    if self.reference["object"] is None
                    else self.reference["object"][ind]
                )
                return robot_ref, robot_vel_ref, object_ref

            def interpolate_case(_):
                # Linearly interpolate between frames to get references
                blend = time - self.reference["time"][ind] / (
                    self.reference["time"][ind_next] - self.reference["time"][ind]
                )

                # robot motion
                if self.robot_horizon > 1:
                    robot_ref = (1.0 - blend) ** self.reference["robot"][
                        ind
                    ] + blend * self.reference["robot"][ind_next]
                    robot_vel_ref = (
                        None
                        if self.reference["robot_vel"] is None
                        else (1.0 - blend) ** self.reference["robot_vel"][ind]
                        + blend * self.reference["robot_vel"][ind_next]
                    )
                else:
                    robot_ref = self.reference["robot"][0]
                    robot_vel_ref = (
                        None
                        if self.reference["robot_vel"] is None
                        else self.reference["robot_vel"][0]
                    )
                # object motion
                if self.reference["object"] is None:
                    object_ref = None
                elif self.object_horizon > 1:
                    object_ref = (1.0 - blend) * self.reference["object"][
                        ind
                    ] + blend * self.reference["object"][ind_next]
                else:
                    object_ref = self.reference["object"][0]

                return robot_ref, robot_vel_ref, object_ref

            robot_ref, robot_vel_ref, object_ref = jax.lax.cond(
                ind == ind_next,
                exact_frame_case,
                interpolate_case,
                operand=None
            )

        return ReferenceStruct(
            time=time,
            robot=robot_ref,
            robot_vel=robot_vel_ref,
            object=object_ref,
            robot_init=self.reference["robot_init"],
            object_init=self.reference["object_init"],
        )

    def __repr__(self) -> str:
        return repr(self.reference)
