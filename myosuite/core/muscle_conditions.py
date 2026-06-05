# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Muscle condition helpers: cumulative fatigue, sarcopenia, reafferentation.

These are pure transform functions that modify MuJoCo MjSpec or MjModel
to represent physiological muscle conditions. They are used by ModelBuilder
as transform callbacks and by term functions via EnvAccessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import mujoco


# ---------------------------------------------------------------------------
# Per-muscle fatigue parameters (Rakshit et al. 2021, Looft & Frey-Law 2020)
# ---------------------------------------------------------------------------

MUSCLE_FATIGUE_PARAMS: dict[str, dict[str, float]] = {
    "Ankle-Dorsiflexor-F": {"F": 0.00746, "R": 0.00081, "r": 4.97},
    "Ankle-Dorsiflexor-M": {"F": 0.00725, "R": 0.00096, "r": 10.36},
    "Ankle-Dorsiflexor": {"F": 0.00828, "R": 0.00204, "r": 7.07},
    "Ankle-Plantarflexor-F": {"F": 0.00702, "R": 0.00098},
    "Ankle-Plantarflexor-M": {"F": 0.00683, "R": 0.00093},
    "Ankle-Plantarflexor": {"F": 0.00695, "R": 0.00096},
    "Elbow-Extensor-F": {"F": 0.01874, "R": 0.00206, "r": 21.22},
    "Elbow-Extensor-M": {"F": 0.01269, "R": 0.00085, "r": 30.21},
    "Elbow-Extensor": {"F": 0.01559, "R": 0.00125, "r": 25.52},
    "Elbow-Flexor-F": {"F": 0.00965, "R": 0.00197, "r": 6.22},
    "Elbow-Flexor-M": {"F": 0.01302, "R": 0.00188, "r": 8.99},
    "Elbow-Flexor": {"F": 0.01703, "R": 0.00494, "r": 4.68},
    "Hand-Adductor-Pollicis-F": {"F": 0.00476, "R": 0.00093, "r": 6.62},
    "Hand-Adductor-Pollicis-M": {"F": 0.00586, "R": 0.00202, "r": 1.00},
    "Hand-Adductor-Pollicis": {"F": 0.00558, "R": 0.00283, "r": 1.00},
    "Hand-First-Dorsal-Interossei-F": {"F": 0.03999, "R": 0.03983},
    "Hand-First-Dorsal-Interossei-M": {"F": 0.01637, "R": 0.00360, "r": 3.66},
    "Hand-First-Dorsal-Interossei": {"F": 0.02686, "R": 0.00656, "r": 3.41},
    "Wrist-Flexor-F": {
        "F": 0.01159,
        "R": 0.00217,
        "r": 7.39,
    },  # from Hand G/Grip group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Wrist-Flexor-M": {
        "F": 0.01238,
        "R": 0.00178,
        "r": 8.00,
    },  # from Hand G/Grip group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Wrist-Flexor": {
        "F": 0.01235,
        "R": 0.00135,
        "r": 12.51,
    },  # from Hand G/Grip group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Knee-Extensor-F": {"F": 0.01407, "R": 0.00185, "r": 6.32},
    "Knee-Extensor-M": {"F": 0.01420, "R": 0.00153, "r": 10.96},
    "Knee-Extensor": {"F": 0.00825, "R": 0.00076, "r": 14.85},
    "Ankle": {"F": 0.01485, "R": 0.00333, "r": 9.31},
    "Toe": {
        "F": 0.01485,
        "R": 0.00333,
        "r": 9.31,
    },  # from Ankle group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Elbow": {"F": 0.01086, "R": 0.00225, "r": 4.93},
    "Hand": {"F": 0.01227, "R": 0.00134, "r": 9.10},
    "Wrist": {
        "F": 0.01227,
        "R": 0.00134,
        "r": 9.10,
    },  # from Hand group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Finger": {
        "F": 0.01227,
        "R": 0.00134,
        "r": 9.10,
    },  # from Hand group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Knee": {"F": 0.00825, "R": 0.00076, "r": 14.85},
    "Shoulder": {
        "F": 0.00825,
        "R": 0.00076,
        "r": 14.85,
    },  # Shoulder values from Looft & Frey-Law 2020 (https://doi.org/10.1016/j.jbiomech.2020.109762)
    # default: Looft et al. 2018 / Looft & Frey-Law 2020
    "Default": {"F": 0.00970, "R": 0.00091, "r": 15},
    # v2.4 fallback params (Looft et al. 2018, r compensated for 0.1 R/F ratio)
    "Default_v2_4": {"F": 0.00912, "R": 0.1 * 0.00094, "r": 10 * 15},
}

_DEFAULT_F = MUSCLE_FATIGUE_PARAMS["Default"]["F"]
_DEFAULT_R = MUSCLE_FATIGUE_PARAMS["Default"]["R"]
_DEFAULT_r = MUSCLE_FATIGUE_PARAMS["Default"]["r"]


def _per_muscle_params(
    mj_model: Any, sex: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive per-muscle F, R, r arrays from MFG lookup.

    Returns arrays of shape ``(na,)`` where ``na`` is the number of muscle
    actuators in *mj_model*.
    """
    import mujoco as _mujoco  # noqa: PLC0415

    from myosuite.core.muscle_groups import MUSCLE_FMG  # noqa: PLC0415

    muscle_act_ind = mj_model.actuator_dyntype == _mujoco.mjtDyn.mjDYN_MUSCLE
    na = int(sum(muscle_act_ind))
    actuator_names = [
        mj_model.actuator(i).name for i in range(mj_model.nu) if muscle_act_ind[i]
    ]

    F_arr = np.zeros(na)
    R_arr = np.zeros(na)
    r_arr = np.zeros(na)

    for idx, name in enumerate(actuator_names):
        mfg = MUSCLE_FMG.get(name, "Default")
        if sex is not None:
            mfg_sex = f"{mfg}-{sex}"
            if mfg_sex not in MUSCLE_FATIGUE_PARAMS:
                mfg_sex = mfg
        else:
            mfg_sex = mfg
        # Walk up to parent group if needed
        group = mfg_sex if mfg_sex in MUSCLE_FATIGUE_PARAMS else mfg_sex.split("-")[0]
        if group not in MUSCLE_FATIGUE_PARAMS:
            group = "Default"
        p = MUSCLE_FATIGUE_PARAMS[group]
        F_arr[idx] = p.get("F", _DEFAULT_F)
        R_arr[idx] = p.get("R", _DEFAULT_R)
        r_arr[idx] = p.get("r", _DEFAULT_r)

    return F_arr, R_arr, r_arr


# ---------------------------------------------------------------------------
# Sarcopenia helpers
# ---------------------------------------------------------------------------


def apply_sarcopenia_to_spec(
    spec: mujoco.MjSpec, force_scale: float = 0.5
) -> mujoco.MjSpec:
    """Scale muscle peak forces to simulate sarcopenia (age-related muscle loss).

    Multiplies the ``forcerange`` attribute of every actuator in the spec
    by *force_scale*.

    Args:
        spec: MuJoCo model spec to modify in-place.
        force_scale: Fraction of original peak force to retain (0–1).

    Returns:
        The modified spec (same object, modified in-place).

    Example:
        >>> import mujoco
        >>> spec = mujoco.MjSpec.from_file("elbow.xml")
        >>> apply_sarcopenia_to_spec(spec, force_scale=0.5)
    """
    for actuator in spec.actuators:
        if hasattr(actuator, "forcerange"):
            actuator.forcerange[0] *= force_scale
            actuator.forcerange[1] *= force_scale
    return spec


def apply_sarcopenia_to_model(model: mujoco.MjModel, force_scale: float = 0.5) -> None:
    """Scale muscle peak forces on a compiled MjModel in-place.

    Multiplies ``actuator_gainprm[:, 2]`` (the maximum isometric force Fmax)
    by *force_scale* for every actuator.  This matches the original
    ``BaseV0.initializeConditions`` implementation.

    Args:
        model: Compiled MuJoCo model to modify.
        force_scale: Fraction of original peak force to retain (0–1).
    """
    model.actuator_gainprm[:, 2] *= force_scale


# ---------------------------------------------------------------------------
# CumulativeFatigue — 3CC-r model (numpy, CPU)
# ---------------------------------------------------------------------------


class CumulativeFatigue:
    """3CC-r cumulative fatigue model (Xia & Frey Law 2008, Rakshit et al. 2021).

    Tracks the active (MA), fatigued (MF), and resting (MR) compartments
    for each muscle actuator.  By default uses per-muscle fatigue / recovery
    constants derived from biomechanical functional muscle groups (FMG).

    Args:
        mj_model: Compiled MuJoCo model.  Used to read muscle actuator
            time constants (tauact / taudeact) and to look up per-muscle
            fatigue parameters via :data:`MUSCLE_FATIGUE_PARAMS`.
        frame_skip: Number of physics steps per control step.  Together
            with ``mj_model.opt.timestep`` this defines the stored ``_dt``
            used when *dt* is not provided to :meth:`compute_act`.
        sex: Optional sex specifier (``"F"`` or ``"M"``) selects
            sex-specific rows in :data:`MUSCLE_FATIGUE_PARAMS`.
        seed: Random seed for stochastic resets via
            :meth:`reset(fatigue_reset_random=True)`.
        use_uniform_params: If ``True``, bypass per-muscle lookup and use
            uniform F/R/r for all muscles (matches the original
            ``use_fatigue_model_v2_4`` behaviour in ``physics/fatigue.py``).

    Example:
        >>> fatigue = CumulativeFatigue(mj_model, frame_skip=5)
        >>> ma, _, _ = fatigue.compute_act(excitation)
        >>> # or use the convenience wrapper:
        >>> eff = fatigue.step(excitation, dt=0.01)
    """

    def __init__(
        self,
        mj_model: Any,
        frame_skip: int = 1,
        sex: str | None = None,
        seed: int | None = None,
        use_uniform_params: bool = False,
    ) -> None:
        import mujoco as _mujoco  # noqa: PLC0415

        self._dt = float(mj_model.opt.timestep) * int(frame_skip)
        muscle_act_ind = mj_model.actuator_dyntype == _mujoco.mjtDyn.mjDYN_MUSCLE
        self.na: int = int(sum(muscle_act_ind))
        self._tauact: np.ndarray = np.array(
            [
                mj_model.actuator_dynprm[i][0]
                for i in range(mj_model.nu)
                if muscle_act_ind[i]
            ]
        )
        self._taudeact: np.ndarray = np.array(
            [
                mj_model.actuator_dynprm[i][1]
                for i in range(mj_model.nu)
                if muscle_act_ind[i]
            ]
        )
        self._MA = np.zeros(self.na)
        self._MR = np.ones(self.na)
        self._MF = np.zeros(self.na)
        self.TL = np.zeros(self.na)

        if use_uniform_params:
            self._F = _DEFAULT_F * np.ones(self.na)
            self._R = _DEFAULT_R * np.ones(self.na)
            self._r = _DEFAULT_r * np.ones(self.na)
        else:
            self._F, self._R, self._r = _per_muscle_params(mj_model, sex)

        self.seed(seed)

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def step(self, excitation: np.ndarray, dt: float | None = None) -> np.ndarray:
        """One control-step wrapper: returns effective muscle activation (MA).

        This is the interface used by ``ModularTaskEnv`` and other pipeline
        callers that just want the effective excitation after fatigue.

        Args:
            excitation: Commanded muscle excitations, shape ``(na,)``.
            dt: Control timestep in seconds.  If ``None``, uses ``self._dt``.

        Returns:
            Effective excitation (= MA after update), shape ``(na,)``.
        """
        ma, _, _ = self.compute_act(excitation, dt=dt)
        return ma

    # ------------------------------------------------------------------
    # Core 3CC-r dynamics
    # ------------------------------------------------------------------

    def compute_act(
        self, act: np.ndarray, dt: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance fatigue state by one step and return updated compartments.

        Args:
            act: Target activation (commanded excitation), shape ``(na,)``.
            dt: Timestep in seconds.  Defaults to ``self._dt``.

        Returns:
            ``(MA, MR, MF)`` — updated active, resting, and fatigued
            compartments, each shape ``(na,)``.
        """
        _dt = dt if dt is not None else self._dt
        self.TL = act.copy()

        # Activation/deactivation rates (MuJoCo Hill-type dynamics)
        LD = (1.0 / self._tauact) * (0.5 + 1.5 * self._MA)
        LR = (0.5 + 1.5 * self._MA) / self._taudeact

        # Transfer rate C between MR and MA
        C = np.zeros_like(self._MA)
        mask = (self._MA < self.TL) & (self._MR > self.TL - self._MA)
        C[mask] = LD[mask] * (self.TL[mask] - self._MA[mask])
        mask = (self._MA < self.TL) & (self._MR <= self.TL - self._MA)
        C[mask] = LD[mask] * self._MR[mask]
        mask = self._MA >= self.TL
        C[mask] = LR[mask] * (self.TL[mask] - self._MA[mask])

        # Recovery rate: faster during rest (MA >= TL)
        rR = np.where(self._MA >= self.TL, self._r * self._R, self._R)

        # Clip C to keep compartments in [0, 1]
        C = np.clip(  # type: ignore[assignment]
            C,
            np.maximum(
                -self._MA / _dt + self._F * self._MA,
                (self._MR - 1.0) / _dt + rR * self._MF,
            ),
            np.minimum(
                (1.0 - self._MA) / _dt + self._F * self._MA,
                self._MR / _dt + rR * self._MF,
            ),
        )

        dMA = (C - self._F * self._MA) * _dt
        dMR = (-C + rR * self._MF) * _dt
        dMF = (self._F * self._MA - rR * self._MF) * _dt
        self._MA += dMA
        self._MR += dMR
        self._MF += dMF

        return self._MA, self._MR, self._MF

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        fatigue_reset_vec: np.ndarray | None = None,
        fatigue_reset_random: bool = False,
    ) -> None:
        """Reset fatigue state.

        Args:
            fatigue_reset_vec: If provided, set MF to this vector and
                MR = 1 - fatigue_reset_vec.  Shape ``(na,)``.
            fatigue_reset_random: If ``True``, sample a random fatigue
                state using the stored RNG.  Cannot be combined with
                *fatigue_reset_vec*.
        """
        if fatigue_reset_random:
            if fatigue_reset_vec is not None:
                raise ValueError(
                    "Cannot pass fatigue_reset_vec when fatigue_reset_random=True."
                )
            nf = self.np_random.random(size=(self.na,))
            ap = self.np_random.random(size=(self.na,))
            self._MA = nf * ap
            self._MR = nf * (1.0 - ap)
            self._MF = 1.0 - nf
        elif fatigue_reset_vec is not None:
            if len(fatigue_reset_vec) != self.na:
                raise ValueError(
                    f"fatigue_reset_vec length {len(fatigue_reset_vec)} != na={self.na}"
                )
            self._MF = np.asarray(fatigue_reset_vec, dtype=float)  # type: ignore[assignment]
            self._MR = 1.0 - self._MF  # type: ignore[assignment]
            self._MA = np.zeros(self.na)
        else:
            self._MA[:] = 0.0
            self._MR[:] = 1.0
            self._MF[:] = 0.0

    def state_dict(self) -> dict[str, list]:
        """Return serialisable snapshot of the fatigue compartments.

        Returns:
            Dict with keys ``"MA"``, ``"MR"``, ``"MF"``, each a flat list of
            length ``na``.
        """
        return {
            "MA": self._MA.tolist(),
            "MR": self._MR.tolist(),
            "MF": self._MF.tolist(),
        }

    def load_state_dict(self, state: dict[str, list]) -> None:
        """Restore fatigue compartments from a :meth:`state_dict` snapshot.

        Args:
            state: Dict produced by :meth:`state_dict`.
        """
        self._MA = np.array(state["MA"], dtype=float)  # type: ignore[assignment]
        self._MR = np.array(state["MR"], dtype=float)  # type: ignore[assignment]
        self._MF = np.array(state["MF"], dtype=float)  # type: ignore[assignment]

    def seed(self, seed: int | None = None) -> list[int]:
        """Set random seed used by stochastic reset."""
        from myosuite.utils import gym  # noqa: PLC0415

        self.input_seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]  # type: ignore[list-item]

    # ------------------------------------------------------------------
    # Properties (legacy compatibility)
    # ------------------------------------------------------------------

    @property
    def n_muscles(self) -> int:
        return self.na

    @property
    def MA(self) -> np.ndarray:
        return self._MA

    @property
    def MF(self) -> np.ndarray:
        return self._MF

    @property
    def MR(self) -> np.ndarray:
        return self._MR

    @property
    def F(self) -> np.ndarray:
        return self._F

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def r(self) -> np.ndarray:
        return self._r

    # ------------------------------------------------------------------
    # Legacy coefficient setters
    # ------------------------------------------------------------------

    def set_FatigueCoefficient(self, F: float | np.ndarray) -> None:
        self._F = F * np.ones(self.na) if np.isscalar(F) else np.asarray(F)  # type: ignore[assignment, operator]

    def set_RecoveryCoefficient(self, R: float | np.ndarray) -> None:
        self._R = R * np.ones(self.na) if np.isscalar(R) else np.asarray(R)  # type: ignore[assignment, operator]

    def set_RecoveryMultiplier(self, r: float | np.ndarray) -> None:
        self._r = r * np.ones(self.na) if np.isscalar(r) else np.asarray(r)  # type: ignore[assignment, operator]


# ---------------------------------------------------------------------------
# TorchFatigueState — batched 3CC-r model (torch, mjlab)
# ---------------------------------------------------------------------------


class TorchFatigueState:
    """Batched 3CC-r cumulative fatigue state for torch action terms.

    Mirrors :class:`CumulativeFatigue` but operates on torch tensors of shape
    ``(num_envs, n_muscles)`` so it can be used directly in mjlab action terms
    without looping over environments.  Uses the same per-muscle
    :data:`MUSCLE_FATIGUE_PARAMS` and recovery-multiplier dynamics as the
    numpy class.

    Prefer :meth:`from_mj_model` when a MuJoCo model is available; the plain
    constructor accepts scalar or per-muscle-array F / R / r.

    Args:
        num_envs: Number of parallel environments.
        n_muscles: Number of muscle actuators.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda:0"``).
        F: Fatigue rate — scalar or array of length *n_muscles*.
        R: Recovery rate — scalar or array of length *n_muscles*.
        r: Recovery multiplier (active-phase rate boost) — scalar or array.
    """

    def __init__(
        self,
        num_envs: int,
        n_muscles: int,
        device: str = "cpu",
        F: float | np.ndarray = _DEFAULT_F,
        R: float | np.ndarray = _DEFAULT_R,
        r: float | np.ndarray = _DEFAULT_r,
        tauact: float | np.ndarray = 0.01,
        taudeact: float | np.ndarray = 0.04,
    ) -> None:
        import torch  # noqa: PLC0415

        def _to_tensor(v: float | np.ndarray) -> Any:
            arr = np.broadcast_to(np.asarray(v, dtype=np.float32), (n_muscles,)).copy()
            return torch.tensor(arr, device=device)

        self._F: Any = _to_tensor(F)  # (n_muscles,)
        self._R: Any = _to_tensor(R)  # (n_muscles,)
        self._r: Any = _to_tensor(r)  # (n_muscles,)
        self.MA: Any = torch.zeros(num_envs, n_muscles, device=device)
        self.MF: Any = torch.zeros(num_envs, n_muscles, device=device)
        self.MR: Any = torch.ones(num_envs, n_muscles, device=device)
        self._tauact: Any = _to_tensor(tauact)  # (n_muscles,)
        self._taudeact: Any = _to_tensor(taudeact)  # (n_muscles,)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_mj_model(
        cls,
        mj_model: Any,
        num_envs: int,
        device: str = "cpu",
        sex: str | None = None,
        use_uniform_params: bool = False,
    ) -> TorchFatigueState:
        """Build a :class:`TorchFatigueState` from a compiled MuJoCo model.

        Extracts per-muscle F / R / r from :data:`MUSCLE_FATIGUE_PARAMS` via
        the muscle functional group (MFG) lookup — the same logic used by
        :class:`CumulativeFatigue`.

        Args:
            mj_model: Compiled MuJoCo model (``mujoco.MjModel``).
            num_envs: Number of parallel environments.
            device: Torch device string.
            sex: Optional ``"F"`` / ``"M"`` for sex-specific parameters.
            use_uniform_params: Use default uniform F/R/r instead of per-muscle.
        """
        import mujoco as _mujoco  # noqa: PLC0415

        muscle_act_ind = mj_model.actuator_dyntype == _mujoco.mjtDyn.mjDYN_MUSCLE
        na = int(sum(muscle_act_ind))
        if use_uniform_params or na == 0:
            return cls(num_envs=num_envs, n_muscles=na or mj_model.nu, device=device)
        F_arr, R_arr, r_arr = _per_muscle_params(mj_model, sex)
        tauact: np.ndarray = np.array(
            [
                mj_model.actuator_dynprm[i][0]
                for i in range(mj_model.nu)
                if muscle_act_ind[i]
            ]
        )
        taudeact: np.ndarray = np.array(
            [
                mj_model.actuator_dynprm[i][1]
                for i in range(mj_model.nu)
                if muscle_act_ind[i]
            ]
        )
        return cls(
            num_envs=num_envs,
            n_muscles=na,
            device=device,
            F=F_arr,
            R=R_arr,
            r=r_arr,
            tauact=tauact,
            taudeact=taudeact,
        )

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, excitation: Any, dt: float) -> Any:
        """Advance fatigue state by one control step.

        Implements the same 3CC-r dynamics as :meth:`CumulativeFatigue.step`
        (without the tauact / taudeact transfer-rate correction, which is not
        available in the batched torch setting without additional model data).

        Args:
            excitation: Muscle excitation tensor, shape ``(num_envs, n_muscles)``.
            dt: Control timestep in seconds.

        Returns:
            Effective excitation after fatigue scaling, same shape as
            *excitation*.
        """
        import torch  # noqa: PLC0415

        # Activation/deactivation rates (MuJoCo Hill-type dynamics)
        LD = 1.0 / (self._tauact * (0.5 + 1.5 * self.MA))
        LR = (0.5 + 1.5 * self.MA) / self._taudeact

        # Recovery rate: boosted during rest (MA >= TL)
        rR = torch.where(self.MA >= excitation, self._r * self._R, self._R)

        # Clip C to keep compartments in [0, 1]
        C = torch.clamp(
            (LD * torch.minimum(excitation - self.MA, self.MR)) * (self.MA < excitation)
            + (LR * (excitation - self.MA)) * (self.MA >= excitation),
            min=torch.maximum(
                -self.MA / dt + self._F * self.MA, (self.MR - 1.0) / dt + rR * self.MF
            ),
            max=torch.minimum(
                (1.0 - self.MA) / dt + self._F * self.MA, self.MR / dt + rR * self.MF
            ),
        )
        dMA_dt = C - self._F * self.MA
        dMF_dt = self._F * self.MA - rR * self.MF
        dMR_dt = -C + rR * self.MF
        self.MA = self.MA + dMA_dt * dt
        self.MF = self.MF + dMF_dt * dt
        self.MR = self.MR + dMR_dt * dt
        return torch.clamp(excitation, max=self.MA)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: Any = None) -> None:
        """Reset fatigue to unfatigued state for the given environments.

        Args:
            env_ids: Indices or slice of environments to reset.  If ``None``,
                resets all environments.
        """
        if env_ids is None:
            self.MA.zero_()
            self.MF.zero_()
            self.MR.fill_(1.0)
        else:
            self.MA[env_ids] = 0.0
            self.MF[env_ids] = 0.0
            self.MR[env_ids] = 1.0

    def state_dict(self) -> dict[str, list]:
        """Return serialisable snapshot of the fatigue compartments.

        Returns:
            Dict with keys ``"MA"``, ``"MR"``, ``"MF"``, each a nested list
            of shape ``(num_envs, n_muscles)``.
        """
        return {
            "MA": self.MA.cpu().numpy().tolist(),
            "MR": self.MR.cpu().numpy().tolist(),
            "MF": self.MF.cpu().numpy().tolist(),
        }

    def load_state_dict(self, state: dict[str, list]) -> None:
        """Restore fatigue compartments from a :meth:`state_dict` snapshot.

        Args:
            state: Dict produced by :meth:`state_dict`.  Shape must be
                compatible with the current ``(num_envs, n_muscles)`` tensors.
        """
        import torch  # noqa: PLC0415

        self.MA.copy_(
            torch.tensor(state["MA"], dtype=torch.float32, device=self.MA.device)
        )
        self.MR.copy_(
            torch.tensor(state["MR"], dtype=torch.float32, device=self.MR.device)
        )
        self.MF.copy_(
            torch.tensor(state["MF"], dtype=torch.float32, device=self.MF.device)
        )
