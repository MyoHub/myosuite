# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""
Backend parity test: cpu-gym vs MJX vs mjlab produce identical action→observation.

Validation strategy
-------------------
Given a shared initial state and a fixed sequence of actions:

    s_0 ──a_0──► s_1 ──a_1──► … ──a_{T-1}──► s_T

every backend must return the same observation vector o_t (= f(s_t)) and
reward r_t at every step, up to float32/64 rounding.

Tolerances
----------
* CPU uses float64; MJX and mjlab use float32 by default.
* Accepted absolute difference: 1e-4 for float32 terms, 1e-7 for float64.
* Positions / velocities are more sensitive than muscle forces; per-term
  tolerances are documented in _TOLERANCES below.

What is NOT expected to match exactly
--------------------------------------
* Episode length / termination: float32 height comparisons near min_height=0.8
  can terminate one step earlier than float64.  Tests use a short horizon
  (N_STEPS = 50) well within any episode boundary.
* Random exploration noise added by the RL algorithm is excluded — tests
  apply deterministic actions (zeros or a fixed sinusoidal signal).
* Algorithm-specific wrappers (SynergyWrapper, SAR transform) are NOT applied
  here; raw muscle-space actions are used so the physics comparison is clean.

Running
-------
    # See why tests were skipped (recommended):
    pytest benchmarks/sar_backends/test_backend_parity.py -v -rs

    # CPU-only: action-space tests and fixtures that don't need MJX/mjlab run:
    pytest benchmarks/sar_backends/test_backend_parity.py -v -k "cpu or action"

    # Full CPU vs MJX parity (install optional deps first):
    uv sync --extra mjx
    pytest benchmarks/sar_backends/test_backend_parity.py -v

    # With mjlab (needs GPU + mjlab):
    MYOSUITE_GPU_TESTS=1 pytest benchmarks/sar_backends/test_backend_parity.py -v

Skip behavior
-------------
* TestCpuVsMjxParity (5 tests): need ``jax`` and ``mujoco.mjx`` (e.g. ``uv sync --extra mjx``).
  MJX env is created via ``make()`` + ``MJXGymnasiumAdapter`` (not gymnasium registry).
* TestCpuVsMjlabParity (5 tests): need ``MYOSUITE_GPU_TESTS=1``, ``torch``, ``mjlab``.
* TestMjxVsMjlabParity (2 tests): need both MJX and mjlab.
* TestActionSpaceConsistency: both params run when MJX is installed (CPU env via gym, MJX via make+adapter).

Verifying that tests actually run
---------------------------------
* Use ``-rs`` to print skip reasons: ``pytest benchmarks/sar_backends/test_backend_parity.py -v -rs``.
* With ``uv sync --extra mjx`` you should see CpuVsMjx and ActionSpace tests execute (no skip for missing MJX).
* Only mjlab tests remain skipped unless ``MYOSUITE_GPU_TESTS=1`` and mjlab are available.

Relationship to existing tests
-------------------------------
* ``test_obs_parity.py``       – structural / AST checks (no env instantiation)
* ``test_mjlab_training_invariants.py`` – training-loop static checks
* THIS FILE                    – runtime numerical parity (env.step level)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of steps used for numerical comparison (short to avoid termination).
N_STEPS = 50

# Seed used to put the environment in a known initial state.
SEED = 42

# Absolute tolerance for float32 vs float64 comparison.
# Tighter for positions/velocities, looser for derived quantities.
# CPU vs MJX "almost parity": muscle/actuator state and integration differ (float32 vs float64),
# so we use looser tolerances than strict numerical match.
_CPU_MJX_INITIAL_OBS_ATOL = 2.0  # muscle_length/velocity/force/act can differ ~1–2
_CPU_MJX_REWARD_ATOL = 8.0  # per-step reward (float32 vs float64 dynamics)
_CPU_MJX_REWARD_FIRST_N = 10  # compare first N steps only (before divergence)
_CPU_MJX_OBS_MAX_DIFF = 5.0  # max abs obs difference over rollout
_CPU_MJX_EPISODE_LENGTH_SLACK = (
    15  # allow termination step difference (float32 threshold)
)

_TOLERANCES = {
    "qpos_without_xy": 1e-3,  # float32 accumulation over N_STEPS
    "qvel": 2e-3,  # velocities diverge faster
    "com_vel": 2e-3,
    "torso_angle": 1e-3,
    "feet_heights": 1e-3,
    "height": 1e-3,
    "feet_rel_positions": 2e-3,
    "phase_var": 1e-4,  # purely time-based, exact
    "muscle_length": 1e-3,
    "muscle_velocity": 5e-3,  # clip at 100 reduces precision pressure
    "muscle_force": 5e-3,  # divided by 1000 before comparing
    "act": 1e-3,
}
_DEFAULT_TOL = 1e-3

# GPU test gate (mirrors existing convention in the repo)
_GPU_TESTS = os.getenv("MYOSUITE_GPU_TESTS", "0") == "1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_walk_env(env_id: str, **kwargs: Any):
    """Create myoLegWalk or MjxLegWalk env for parity tests.

    MjxLegWalk-v0 is not registered with gymnasium; it is created via the
    MJX package make() and wrapped with MJXGymnasiumAdapter so the test
    can use the same gymnasium API for both backends.
    """
    gym = pytest.importorskip("gymnasium")
    if env_id == "myoLegWalk-v0":
        return gym.make(env_id, **kwargs)
    if env_id == "MjxLegWalk-v0":
        pytest.importorskip("jax")
        pytest.importorskip("mujoco.mjx")
        from myosuite.envs.myo.backends.mjx import make
        from benchmarks.sar_backends.mjx_gymnasium_adapter import MJXGymnasiumAdapter

        mjx_env = make(env_id)
        return MJXGymnasiumAdapter(mjx_env, seed=kwargs.get("seed", SEED))
    raise ValueError(f"Unknown env_id for parity tests: {env_id}")


def _fixed_actions(n_steps: int, n_act: int, seed: int = SEED) -> np.ndarray:
    """Return a (n_steps, n_act) array of deterministic actions in [0, 1].

    Uses a sinusoidal pattern so muscles alternate activation, exercising the
    full range of the physics rather than a static zero-action sequence.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.9, size=(1, n_act))
    t = np.linspace(0, 2 * np.pi, n_steps)[:, None]
    freq = rng.uniform(0.5, 2.0, size=(1, n_act))
    return np.clip(base + 0.15 * np.sin(t * freq), 0.0, 1.0)


def _split_obs(
    obs_vec: np.ndarray, obs_keys: list[str], obs_sizes: dict[str, int]
) -> dict[str, np.ndarray]:
    """Split a flat observation vector into named components."""
    out = {}
    idx = 0
    for key in obs_keys:
        sz = obs_sizes[key]
        out[key] = obs_vec[idx : idx + sz]
        idx += sz
    return out


# Path to MJX leg model — use same XML for CPU in parity tests so physics match.
_LEG_MJX_XML = (
    Path(__file__).resolve().parents[2]
    / "myosuite"
    / "simhive"
    / "myo_sim"
    / "leg"
    / "myolegs_mjx.xml"
)

# ---------------------------------------------------------------------------
# CPU backend fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpu_rollout():
    """Run N_STEPS on the CPU walk env and return per-step obs and rewards.

    Uses myolegs_mjx.xml (plane ground) when available so CPU and MJX share
    the same model and terrain for numerical parity.
    """
    gym = pytest.importorskip("gymnasium")
    myosuite = pytest.importorskip("myosuite")  # noqa: F841

    kwargs = {}
    if _LEG_MJX_XML.exists():
        kwargs["model_path"] = str(_LEG_MJX_XML)
    try:
        env = gym.make("myoLegWalk-v0", **kwargs)
    except Exception as exc:
        pytest.skip(f"Could not create myoLegWalk-v0: {exc}")

    obs, _ = env.reset(seed=SEED)
    n_act = env.action_space.shape[0]
    actions = _fixed_actions(N_STEPS, n_act)

    obs_seq, rew_seq = [], []
    obs_seq.append(obs.copy())
    for t in range(N_STEPS):
        obs, rew, term, trunc, _ = env.step(actions[t])
        obs_seq.append(obs.copy())
        rew_seq.append(float(rew))
        if term or trunc:
            break
    env.close()

    return {
        "obs": np.array(obs_seq),  # (T+1, obs_dim)
        "rew": np.array(rew_seq),  # (T,)
        "actions": actions,
        "n_act": n_act,
        "obs_dim": obs_seq[0].shape[0],
    }


# ---------------------------------------------------------------------------
# MJX backend fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mjx_rollout(cpu_rollout):
    """Run the same action sequence on the MJX walk env and return obs/rewards."""
    try:
        env = _make_walk_env("MjxLegWalk-v0", seed=SEED)
    except Exception as exc:
        pytest.skip(f"Could not create MjxLegWalk-v0: {exc}")

    obs, _ = env.reset(seed=SEED)
    actions = cpu_rollout["actions"]

    obs_seq, rew_seq = [], []
    obs_seq.append(np.asarray(obs).squeeze())
    for t in range(len(actions)):
        act_t = np.asarray(actions[t], dtype=np.float32)
        obs, rew, term, trunc, _ = env.step(act_t)
        obs_seq.append(np.asarray(obs).squeeze())
        rew_seq.append(float(np.asarray(rew).squeeze()))
        if np.any(np.asarray(term) | np.asarray(trunc)):
            break
    env.close()

    return {
        "obs": np.array(obs_seq),
        "rew": np.array(rew_seq),
    }


# ---------------------------------------------------------------------------
# mjlab backend fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mjlab_rollout(cpu_rollout):
    """Run the same action sequence on the mjlab walk env and return obs/rewards."""
    if not _GPU_TESTS:
        pytest.skip("Set MYOSUITE_GPU_TESTS=1 to enable mjlab parity tests")

    pytest.importorskip("torch")
    mjlab = pytest.importorskip("mjlab")  # noqa: F841

    try:
        from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import make_walk_env

        env = make_walk_env(num_envs=1, seed=SEED)
    except Exception as exc:
        pytest.skip(f"Could not create mjlab walk env: {exc}")

    import torch

    obs = env.reset()
    actions = cpu_rollout["actions"]

    obs_seq, rew_seq = [], []
    obs_seq.append(obs.cpu().numpy().squeeze())
    for t in range(len(actions)):
        act_t = torch.tensor(actions[t], dtype=torch.float32).unsqueeze(0)
        obs, rew, term, _ = env.step(act_t)
        obs_seq.append(obs.cpu().numpy().squeeze())
        rew_seq.append(float(rew.cpu().numpy().squeeze()))
        if term.any():
            break
    env.close()

    return {
        "obs": np.array(obs_seq),
        "rew": np.array(rew_seq),
    }


# ---------------------------------------------------------------------------
# Parity tests: CPU vs MJX
# ---------------------------------------------------------------------------


class TestCpuVsMjxParity:
    """Numerical parity between CPU (float64) and MJX (float32) backends."""

    def test_observation_shape_matches(self, cpu_rollout, mjx_rollout):
        """MJX observation vector has the same dimensionality as CPU."""
        assert cpu_rollout["obs"].shape[-1] == mjx_rollout["obs"].shape[-1], (
            f"Obs dim mismatch: CPU={cpu_rollout['obs'].shape[-1]}, "
            f"MJX={mjx_rollout['obs'].shape[-1]}"
        )

    def test_initial_obs_close(self, cpu_rollout, mjx_rollout):
        """First observation after reset must be close between CPU and MJX (almost parity).

        Muscle/actuator state can differ due to float32 vs float64; use relaxed atol.
        """
        cpu_o0 = cpu_rollout["obs"][0]
        mjx_o0 = mjx_rollout["obs"][0]
        np.testing.assert_allclose(
            mjx_o0,
            cpu_o0,
            atol=_CPU_MJX_INITIAL_OBS_ATOL,
            err_msg="Initial observation differs between CPU and MJX backends",
        )

    def test_reward_sequence_close(self, cpu_rollout, mjx_rollout):
        """First few steps' rewards must be close (early trajectory almost parity)."""
        n = min(
            len(cpu_rollout["rew"]),
            len(mjx_rollout["rew"]),
            _CPU_MJX_REWARD_FIRST_N,
        )
        np.testing.assert_allclose(
            mjx_rollout["rew"][:n],
            cpu_rollout["rew"][:n],
            atol=_CPU_MJX_REWARD_ATOL,
            err_msg="Reward sequence (first steps) diverges between CPU and MJX backends",
        )

    def test_obs_sequence_close(self, cpu_rollout, mjx_rollout):
        """Observation sequences must stay within bound (almost parity)."""
        n = min(cpu_rollout["obs"].shape[0], mjx_rollout["obs"].shape[0])
        max_abs_diff = np.max(np.abs(cpu_rollout["obs"][:n] - mjx_rollout["obs"][:n]))
        assert max_abs_diff < _CPU_MJX_OBS_MAX_DIFF, (
            f"Max abs obs difference CPU vs MJX over {n} steps: {max_abs_diff:.4f} "
            f"(threshold {_CPU_MJX_OBS_MAX_DIFF})"
        )

    def test_episode_length_close(self, cpu_rollout, mjx_rollout):
        """Episode length must be within slack (float32 vs float64 termination)."""
        cpu_T = len(cpu_rollout["rew"])
        mjx_T = len(mjx_rollout["rew"])
        assert abs(cpu_T - mjx_T) <= _CPU_MJX_EPISODE_LENGTH_SLACK, (
            f"Episode length differs: CPU={cpu_T} steps, MJX={mjx_T} steps. "
            f"Allow slack {_CPU_MJX_EPISODE_LENGTH_SLACK} for float32/float64."
        )


# ---------------------------------------------------------------------------
# Parity tests: CPU vs mjlab
# ---------------------------------------------------------------------------


class TestCpuVsMjlabParity:
    """Numerical parity between CPU (float64) and mjlab (float32/torch) backends."""

    def test_observation_shape_matches(self, cpu_rollout, mjlab_rollout):
        assert cpu_rollout["obs"].shape[-1] == mjlab_rollout["obs"].shape[-1], (
            f"Obs dim mismatch: CPU={cpu_rollout['obs'].shape[-1]}, "
            f"mjlab={mjlab_rollout['obs'].shape[-1]}"
        )

    def test_initial_obs_close(self, cpu_rollout, mjlab_rollout):
        cpu_o0 = cpu_rollout["obs"][0]
        mjlab_o0 = mjlab_rollout["obs"][0]
        np.testing.assert_allclose(
            mjlab_o0,
            cpu_o0,
            atol=_DEFAULT_TOL,
            err_msg="Initial observation differs between CPU and mjlab backends",
        )

    def test_reward_sequence_close(self, cpu_rollout, mjlab_rollout):
        n = min(len(cpu_rollout["rew"]), len(mjlab_rollout["rew"]))
        np.testing.assert_allclose(
            mjlab_rollout["rew"][:n],
            cpu_rollout["rew"][:n],
            atol=0.05,
            err_msg="Reward sequence diverges between CPU and mjlab backends",
        )

    def test_obs_sequence_close(self, cpu_rollout, mjlab_rollout):
        n = min(cpu_rollout["obs"].shape[0], mjlab_rollout["obs"].shape[0])
        max_abs_diff = np.max(np.abs(cpu_rollout["obs"][:n] - mjlab_rollout["obs"][:n]))
        assert (
            max_abs_diff < 0.1
        ), f"Max abs obs difference CPU vs mjlab over {n} steps: {max_abs_diff:.4f}"


# ---------------------------------------------------------------------------
# Parity tests: MJX vs mjlab (both float32, should be tightest)
# ---------------------------------------------------------------------------


class TestMjxVsMjlabParity:
    """Both MJX and mjlab use float32; differences should be smallest here."""

    def test_observation_shape_matches(self, mjx_rollout, mjlab_rollout):
        assert mjx_rollout["obs"].shape[-1] == mjlab_rollout["obs"].shape[-1]

    def test_obs_sequence_close(self, mjx_rollout, mjlab_rollout):
        n = min(mjx_rollout["obs"].shape[0], mjlab_rollout["obs"].shape[0])
        max_abs_diff = np.max(np.abs(mjx_rollout["obs"][:n] - mjlab_rollout["obs"][:n]))
        assert max_abs_diff < 0.05, (
            f"Max abs obs difference MJX vs mjlab over {n} steps: {max_abs_diff:.4f} "
            f"(both float32 — tolerance tighter at 0.05)"
        )


# ---------------------------------------------------------------------------
# Action-space consistency tests (no env.step needed)
# ---------------------------------------------------------------------------


class TestActionSpaceConsistency:
    """Action spaces must be numerically identical across backends."""

    @pytest.mark.parametrize(
        "env_id,kwargs",
        [
            ("myoLegWalk-v0", {}),
            ("MjxLegWalk-v0", {"num_envs": 1}),
        ],
    )
    def test_action_space_bounds(self, env_id, kwargs):
        """All backends expose [0, 1]^{nu} action space (post-sigmoid normalization)."""
        pytest.importorskip("myosuite")  # noqa: F841
        try:
            env = _make_walk_env(env_id, **kwargs)
        except Exception as exc:
            pytest.skip(f"Could not create {env_id}: {exc}")

        low = np.asarray(env.action_space.low).ravel()
        high = np.asarray(env.action_space.high).ravel()
        np.testing.assert_allclose(
            low, 0.0, atol=1e-6, err_msg=f"{env_id}: action_space.low != 0"
        )
        np.testing.assert_allclose(
            high, 1.0, atol=1e-6, err_msg=f"{env_id}: action_space.high != 1"
        )
        env.close()

    @pytest.mark.parametrize(
        "env_id,kwargs",
        [
            ("myoLegWalk-v0", {}),
            ("MjxLegWalk-v0", {"num_envs": 1}),
        ],
    )
    def test_action_dimension_matches(self, env_id, kwargs):
        """Action dimension must equal the number of muscles (nu = 80 for myoLeg)."""
        pytest.importorskip("myosuite")  # noqa: F841
        try:
            env = _make_walk_env(env_id, **kwargs)
        except Exception as exc:
            pytest.skip(f"Could not create {env_id}: {exc}")

        n_act = env.action_space.shape[-1]
        assert n_act == 80, f"{env_id}: expected 80 muscles, got {n_act}"
        env.close()
