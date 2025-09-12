import jax.numpy as jp
import jax.random as jrandom
import mujoco
from typing import Dict, Tuple, Any
from jax import tree_util
import numpy as np

# Constants for floating-point precision
_FLOAT_EPS = jp.finfo(jp.float32).eps
_EPS4 = _FLOAT_EPS * 4.0


class CumulativeFatigue:
    """
    JAX implementation of the 3CC-r muscle fatigue model
    Adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701
    Based on implementation from Aleksi Ikkala and Florian Fischer
    """

    def __init__(self, mj_model, frame_skip=1):
        # Get muscle actuator indices
        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        # Convert to concrete integer value
        self.na = int(np.sum(muscle_act_ind))  # Use numpy here since we're in __init__

        # Dynamic arrays (will be included in tree_flatten children)
        self.MA = jp.zeros(self.na, dtype=jp.float32)  # Muscle Active
        self.MR = jp.ones(self.na, dtype=jp.float32)  # Muscle Resting
        self.MF = jp.zeros(self.na, dtype=jp.float32)  # Muscle Fatigue
        self.TL = jp.zeros(self.na, dtype=jp.float32)  # Target Load

        # Parameters (will be included in tree_flatten children)
        self.F = jp.array(0.00912, dtype=jp.float32)  # Fatigue coefficient
        self.R = jp.array(0.1 * 0.00094, dtype=jp.float32)  # Recovery coefficient
        self.r = jp.array(10 * 15, dtype=jp.float32)  # Recovery multiplier
        self.dt = jp.array(mj_model.opt.timestep * frame_skip, dtype=jp.float32)

        # Get muscle parameters
        self.tauact = jp.array(
            [
                mj_model.actuator_dynprm[i][0]
                for i in range(len(muscle_act_ind))
                if muscle_act_ind[i]
            ],
            dtype=jp.float32,
        )
        self.taudeact = jp.array(
            [
                mj_model.actuator_dynprm[i][1]
                for i in range(len(muscle_act_ind))
                if muscle_act_ind[i]
            ],
            dtype=jp.float32,
        )

    def _tree_flatten(self) -> Tuple[Tuple[Any, ...], Dict]:
        """Flatten the class into children and auxiliary data"""
        # Dynamic values (arrays that change during computation)
        children = (
            self.MA,
            self.MR,
            self.MF,
            self.TL,
            self.F,
            self.R,
            self.r,
            self.dt,
            self.tauact,
            self.taudeact,
        )

        # Static values
        aux_data = {"na": self.na}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """Reconstruct class from flattened data"""
        obj = cls.__new__(cls)  # Create new instance without __init__

        # Restore dynamic values
        (
            obj.MA,
            obj.MR,
            obj.MF,
            obj.TL,
            obj.F,
            obj.R,
            obj.r,
            obj.dt,
            obj.tauact,
            obj.taudeact,
        ) = children

        # Restore static values
        obj.na = aux_data["na"]
        return obj

    # @jax.jit
    def compute_act(self, act):
        """
        Compute muscle activation considering fatigue

        Args:
            act: Target activation levels

        Returns:
            tuple: (MA, MR, MF) states
        """
        # Set target load
        self.TL = jp.array(act, dtype=jp.float32)

        # Calculate effective time constants
        LD = 1 / self.tauact * (0.5 + 1.5 * self.MA)
        LR = (0.5 + 1.5 * self.MA) / self.taudeact

        # Calculate C(t) - transfer rate between MR and MA
        C = jp.zeros_like(self.MA)

        # Case 1: MA < TL and MR > (TL - MA)
        mask1 = (self.MA < self.TL) & (self.MR > (self.TL - self.MA))
        C = jp.where(mask1, LD * (self.TL - self.MA), C)

        # Case 2: MA < TL and MR <= (TL - MA)
        mask2 = (self.MA < self.TL) & (self.MR <= (self.TL - self.MA))
        C = jp.where(mask2, LD * self.MR, C)

        # Case 3: MA >= TL
        mask3 = self.MA >= self.TL
        C = jp.where(mask3, LR * (self.TL - self.MA), C)

        # Calculate recovery rate
        rR = jp.where(self.MA >= self.TL, self.r * self.R, self.R)

        # Clip C(t) to ensure states remain between 0 and 1
        C_min = jp.maximum(
            -self.MA / self.dt + self.F * self.MA,
            (self.MR - 1) / self.dt + rR * self.MF,
        )
        C_max = jp.minimum(
            (1 - self.MA) / self.dt + self.F * self.MA, self.MR / self.dt + rR * self.MF
        )
        C = jp.clip(C, C_min, C_max)

        # Update states
        dMA = (C - self.F * self.MA) * self.dt
        dMR = (-C + rR * self.MF) * self.dt
        dMF = (self.F * self.MA - rR * self.MF) * self.dt

        self.MA = self.MA + dMA
        self.MR = self.MR + dMR
        self.MF = self.MF + dMF

        return self.MA, self.MR, self.MF

    # @jax.jit
    def get_effort(self):
        """Calculate effort as norm of difference between actual and target activation"""
        return jp.linalg.norm(self.MA - self.TL)

    def reset(self, fatigue_reset_vec=None, fatigue_reset_random=False, key=None):
        """Reset fatigue states"""
        if fatigue_reset_random:
            assert (
                fatigue_reset_vec is None
            ), "Cannot use fatigue_reset_vec if fatigue_reset_random=True"
            key1, key2 = jrandom.split(key)
            non_fatigued_muscles = jrandom.uniform(key1, (self.na,))
            active_percentage = jrandom.uniform(key2, (self.na,))
            self.MA = non_fatigued_muscles * active_percentage
            self.MR = non_fatigued_muscles * (1 - active_percentage)
            self.MF = 1 - non_fatigued_muscles
        else:
            if fatigue_reset_vec is not None:
                assert (
                    len(fatigue_reset_vec) == self.na
                ), f"Invalid length of fatigue vector (expected {self.na}, got {len(fatigue_reset_vec)})"
                self.MF = jp.array(fatigue_reset_vec, dtype=jp.float32)
                self.MR = 1 - self.MF
                self.MA = jp.zeros(self.na, dtype=jp.float32)
            else:
                self.MA = jp.zeros(self.na, dtype=jp.float32)
                self.MR = jp.ones(self.na, dtype=jp.float32)
                self.MF = jp.zeros(self.na, dtype=jp.float32)

    def set_FatigueCoefficient(self, F):
        """Set Fatigue coefficient"""
        self.F = jp.array(F, dtype=jp.float32)

    def set_RecoveryCoefficient(self, R):
        """Set Recovery coefficient"""
        self.R = jp.array(R, dtype=jp.float32)

    def set_RecoveryMultiplier(self, r):
        """Set Recovery time multiplier"""
        self.r = jp.array(r, dtype=jp.float32)


# Register the class as a PyTree
tree_util.register_pytree_node(
    CumulativeFatigue,
    CumulativeFatigue._tree_flatten,
    CumulativeFatigue._tree_unflatten,
)
