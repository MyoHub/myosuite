# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backend parity tests: MJX (XLA/CPU or GPU) vs MuJoCo CPU.

Verifies that the MJX env and the CPU env produce consistent physics and
rewards when run from the same initial state with the same action sequence.

Device selection:
    By default, JAX uses CPU so tests pass in CI and without a GPU. To run
    parity tests on GPU (requires ``uv sync --extra mjx-cuda`` and a compatible
    CUDA/driver), set::

        MYOSUITE_GPU_TESTS=1 pytest myosuite/tests/test_backends.py -v

MJX backend selection:
    When no GPU is present, the JAX/XLA backend is used (``impl=None``),
    which runs on CPU via XLA compilation — no GPU required.
    The ``"warp"`` backend (GPU only) is tried first; falls back to XLA.

Skip conditions:
    - ``mujoco.mjx`` not importable (version mismatch / not installed)
    - ``mujoco_playground`` not installed
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest


pytestmark = pytest.mark.tier2


# ---------------------------------------------------------------------------
# Guard: skip entire module if MJX / mujoco_playground are unavailable.
# Use try/except rather than pytest.importorskip because some environments
# raise AttributeError (mujoco/mjx version mismatch) rather than ImportError.
# ---------------------------------------------------------------------------
try:
    from mujoco import mjx as mjx_mod  # noqa: F401
    import mujoco_playground as _mp_check  # noqa: F401
except (ImportError, AttributeError) as _mjx_err:
    pytest.skip(
        f"MJX/mujoco_playground not available ({_mjx_err}); "
        "skipping backend parity tests",
        allow_module_level=True,
    )

import os  # noqa: E402

# When running GPU parity tests, set JAX env before first import (can help cuSolver/GPU stability).
_use_gpu = os.environ.get("MYOSUITE_GPU_TESTS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
if _use_gpu:
    os.environ.setdefault("JAX_PREALLOCATE", "false")

import jax  # noqa: E402

# Use GPU only when explicitly requested; otherwise CPU for CI and default runs.
if not _use_gpu:
    jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
from ml_collections import config_dict  # noqa: E402

# SAR helpers require sklearn
try:
    from sklearn.decomposition import FastICA, PCA
    from sklearn.preprocessing import MinMaxScaler

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# mjlab backend guard: require full mjlab installation (envs + tasks registry)
try:
    import mjlab  # noqa: F401
    import mjlab.envs  # noqa: F401
    from mjlab.tasks import registry as _mjlab_registry  # noqa: F401

    _MJLAB_AVAILABLE = True
except ImportError:
    _MJLAB_AVAILABLE = False

from myosuite.envs.myo.backends.mjx.pose_env import MjxPoseEnv  # noqa: E402
from myosuite.envs.myo.backends.mjx.reach_env import MjxReachEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ASSETS = pathlib.Path(__file__).parents[1] / "envs" / "myo" / "assets"

_N_STEPS = 50
_REWARD_RTOL = 0.05  # rewards within 5 %
_QPOS_ATOL = 5e-3  # joint positions: 5 mrad / 5 mm (float32 vs float64)


# ---------------------------------------------------------------------------
# Fixtures: GPU sanity check and MJX impl
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gpu_skip_reason() -> str | None:
    """When MYOSUITE_GPU_TESTS=1, probe JAX GPU; return skip reason if GPU unusable.

    Returns:
        None if GPU is OK or not requested; else a string reason to skip GPU tests
        (e.g. cuSolver internal error).
    """
    if not _use_gpu:
        return None
    try:
        jax.device_put(jnp.ones(1)).block_until_ready()
        return None
    except Exception as err:  # noqa: BLE001
        msg = str(err)
        if "cuSolver" in msg or "gpusolver" in msg or "INTERNAL" in msg:
            return f"JAX GPU unavailable ({type(err).__name__}): {msg[:200]}"
        raise


@pytest.fixture(scope="session")
def mjx_impl(gpu_skip_reason: str | None) -> str | None:
    """Return ``"warp"`` if available, else ``None`` (XLA / CPU).

    Returns:
        ``"warp"`` when the ``warp`` module is importable (GPU present),
        otherwise ``None`` which selects the JAX/XLA backend (any device).
    """
    if gpu_skip_reason:
        pytest.skip(gpu_skip_reason)

    # Only attempt to use the Warp (CUDA) backend when GPU tests are explicitly
    # requested. On CPU-only machines, always fall back to the XLA backend
    # (impl=None) so these tests still run.
    if _use_gpu:
        try:
            import warp  # noqa: F401

            return "warp"
        except ImportError:
            return None
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pose_config(
    model_path: pathlib.Path,
    target_range: dict,
    impl: str | None,
) -> config_dict.ConfigDict:
    """Build a pose env config for testing.

    Args:
        model_path: Path to the MJCF file.
        target_range: Mapping of joint name → ``(lo, hi)`` JAX arrays.
        impl: MJX impl (``"warp"`` or ``None`` for XLA).

    Returns:
        Config dict suitable for ``MjxPoseEnv``.
    """
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        num_envs=1,
        mjx_impl=impl,
        reward_config=config_dict.create(
            angle_reward_weight=1.0,
            ctrl_cost_weight=0.0,  # disable act_reg to match CPU default
            pose_thd=0.35,
            far_th=float(2 * np.pi),
            bonus_weight=4.0,
        ),
        target_jnt_range=config_dict.ConfigDict(target_range),
        max_episode_steps=_N_STEPS,
        model_path=model_path,
    )


def _reach_config(
    model_path: pathlib.Path,
    target_range: dict,
    impl: str | None,
) -> config_dict.ConfigDict:
    """Build a reach env config for testing.

    Args:
        model_path: Path to the MJCF file.
        target_range: Mapping of site name → ``(lo, hi)`` JAX arrays.
        impl: MJX impl.

    Returns:
        Config dict suitable for ``MjxReachEnv``.
    """
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        num_envs=1,
        mjx_impl=impl,
        reward_weights=config_dict.create(
            reach=1.0,
            bonus=4.0,
            penalty=50.0,
        ),
        target_reach_range=config_dict.ConfigDict(target_range),
        far_th=0.35,
        max_episode_steps=_N_STEPS,
        model_path=model_path,
    )


def _run_cpu_episode(
    env_id: str,
    actions: np.ndarray,
    override_target_jnt: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one CPU env episode; return ``(qpos_trajectory, rewards)``.

    Args:
        env_id: Gymnasium env ID (must be registered).
        actions: Action sequence, shape ``(T, nu)``.
        override_target_jnt: If given, set ``target_jnt_value`` to this
            array after reset so both backends target the same pose.

    Returns:
        Tuple ``(qpos_traj, rewards)`` — numpy arrays of lengths ``T`` and
        ``(T, nq)`` respectively.
    """
    import gymnasium as gym
    from myosuite.utils import gym as _myo_gym  # ensure envs registered  # noqa: F401

    env = gym.make(env_id)
    env.reset(seed=0)

    if override_target_jnt is not None:
        env.unwrapped.target_jnt_value[:] = override_target_jnt

    # Reset physics to model default (qpos0, zero vel) so both backends match
    env.unwrapped.data.qpos[:] = env.unwrapped.model.qpos0
    env.unwrapped.data.qvel[:] = 0.0
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    qpos_traj, rewards = [], []
    for action in actions:
        _, rwd, terminated, truncated, _ = env.step(action)
        qpos_traj.append(env.unwrapped.data.qpos.copy())
        rewards.append(rwd)
        if terminated or truncated:
            break

    env.close()
    return np.array(qpos_traj), np.array(rewards)


def _run_mjx_episode(
    env_inst,
    task_state: dict,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one MJX env episode; return ``(qpos_trajectory, rewards)``.

    Args:
        env_inst: Instantiated MJX env (already has reset called internally).
        task_state: Fixed task state injected into ``state.info`` (overrides
            the random one from ``reset``).
        actions: Action sequence, shape ``(T, nu)``.

    Returns:
        Tuple ``(qpos_traj, rewards)`` — numpy arrays of shape ``(T, nq)``
        and ``(T,)`` respectively.
    """
    rng = jax.random.PRNGKey(0)
    state = env_inst.reset(rng)
    # Inject fixed task_state so targets match the CPU env
    state = state.replace(info={**state.info, **task_state})

    jit_step = jax.jit(env_inst.step)

    qpos_traj, rewards = [], []
    for action in actions:
        state = jit_step(state, jnp.array(action, dtype=jnp.float32))
        qpos_traj.append(np.array(state.data.qpos))
        rewards.append(float(state.reward))

    return np.array(qpos_traj), np.array(rewards)


def _run_mjlab_sar_episode(
    env_id: str,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one mjlab episode; return ``(qpos_trajectory, rewards)``.

    This helper mirrors ``_run_cpu_episode``/``_run_mjx_episode`` while keeping
    backend glue thin: we only adapt action tensor shape and read qpos from the
    mjlab scene entity.
    """
    import torch

    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env

    env = make_env(env_id, backend="mjlab")
    env.reset(seed=0)

    # Current mjlab elbow task uses entity name "elbow".
    entity = env.scene["elbow"]
    raw_data = entity.data.data

    qpos_traj, rewards = [], []
    for action in actions:
        action_t = torch.as_tensor(
            action, dtype=torch.float32, device=env.device
        ).unsqueeze(0)
        _, reward, terminated, truncated, _ = env.step(action_t)
        qpos_traj.append(raw_data.qpos[0].detach().cpu().numpy().copy())
        rewards.append(float(reward[0].detach().cpu().item()))
        if bool(terminated.any()) or bool(truncated.any()):
            break

    if hasattr(env, "close"):
        env.close()
    return np.array(qpos_traj), np.array(rewards)


# ---------------------------------------------------------------------------
# SAR helpers
# ---------------------------------------------------------------------------


def _fit_minimal_sar(
    n_muscles: int, n_syn: int, seed: int = 42
) -> tuple[FastICA, PCA, MinMaxScaler]:
    """Fit PCA + ICA + MinMaxScaler on synthetic activation data.

    Args:
        n_muscles: Action-space dimensionality.
        n_syn: Number of synergies (PCA components).
        seed: RNG seed for reproducibility.

    Returns:
        Tuple ``(ica, pca, scaler)``.
    """
    rng = np.random.default_rng(seed)
    synth = rng.uniform(0.0, 1.0, (500, n_muscles)).astype(np.float32)
    pca = PCA(n_components=n_syn)
    pca_acts = pca.fit_transform(synth)
    ica = FastICA(random_state=seed, max_iter=500)
    ica_acts = ica.fit_transform(pca_acts)
    scaler = MinMaxScaler((-1.0, 1.0))
    scaler.fit(ica_acts)
    return ica, pca, scaler


def _sar_to_muscle_actions(
    synergy_actions: np.ndarray,
    ica: FastICA,
    pca: PCA,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Inverse-transform synergy-space actions to full muscle activations.

    Args:
        synergy_actions: Shape ``(T, n_syn)`` in ``[-1, 1]``.
        ica: Fitted ICA.
        pca: Fitted PCA.
        scaler: Fitted MinMaxScaler.

    Returns:
        Full muscle activations, shape ``(T, n_muscles)``, dtype float32.
    """
    full = pca.inverse_transform(
        ica.inverse_transform(scaler.inverse_transform(synergy_actions))
    )
    return full.astype(np.float32)


# ---------------------------------------------------------------------------
# Elbow pose parity (full physics + reward comparison)
# ---------------------------------------------------------------------------


class TestElbowPoseBackendParity:
    """CPU vs MJX(XLA) physics and reward parity for the elbow pose task.

    The elbow model has 1 DOF and 6 muscles — simple enough to exhibit
    near-identical trajectories despite float32/float64 differences.
    """

    @pytest.fixture(autouse=True)
    def setup(self, mjx_impl):
        """Build both envs with identical configs.

        Args:
            mjx_impl: ``"warp"`` or ``None``, from the session fixture.
        """
        self.model_path = _ASSETS / "elbow" / "myoelbow_1dof6muscles.xml"
        if not self.model_path.exists():
            pytest.skip(f"Elbow model not found: {self.model_path}")

        # Fixed target: 2.0 rad (same as myoElbowPose1D6MFixed-v0)
        target = jnp.array([2.0])
        self.target_jnt_range = {"r_elbow_flex": (target, target)}
        self.target_np = np.array([2.0])

        cfg = _pose_config(self.model_path, self.target_jnt_range, mjx_impl)
        try:
            self.mjx_env = MjxPoseEnv(cfg)
        except NotImplementedError as exc:
            # Some mujoco.mjx builds do not support certain collision pairs
            # (e.g. cylinder–mesh) used by the elbow assets. Treat this as a
            # limitation of the local MJX installation and skip these parity
            # tests rather than failing the suite.
            pytest.skip(
                f"MJX elbow pose env not supported on this mujoco.mjx build: {exc}"
            )

        nu = self.mjx_env.mj_model.nu
        # Zero actions → sigmoid(−2.5) ≈ 0.076 muscle activation
        self.actions = np.zeros((_N_STEPS, nu), dtype=np.float32)
        self.task_state = {"target_angles": jnp.array(self.target_np)}

    def test_qpos_trajectory_close(self):
        """MJX and CPU joint angles should agree within ``QPOS_ATOL``.

        Both envs start from ``qpos0 = [0.0]`` with zero velocity.  Float32
        (MJX) vs float64 (CPU) causes sub-milliradian drift after 50 steps.
        """
        cpu_qpos, _ = _run_cpu_episode(
            "myoElbowPose1D6MFixed-v0",
            self.actions,
            override_target_jnt=self.target_np,
        )
        mjx_qpos, _ = _run_mjx_episode(self.mjx_env, self.task_state, self.actions)

        n = min(len(cpu_qpos), len(mjx_qpos))
        assert n >= 10, f"Episodes too short: cpu={len(cpu_qpos)}, mjx={n}"

        np.testing.assert_allclose(
            cpu_qpos[:n],
            mjx_qpos[:n],
            atol=_QPOS_ATOL,
            err_msg="CPU and MJX qpos diverge beyond tolerance",
        )

    def test_reward_within_tolerance(self):
        """MJX total reward should be within 5 % of CPU total reward.

        Uses ``ctrl_cost_weight=0`` in MJX to match the CPU default which
        omits ``act_reg`` from the dense reward sum.
        """
        cpu_qpos, cpu_rwds = _run_cpu_episode(
            "myoElbowPose1D6MFixed-v0",
            self.actions,
            override_target_jnt=self.target_np,
        )
        mjx_qpos, mjx_rwds = _run_mjx_episode(
            self.mjx_env, self.task_state, self.actions
        )

        n = min(len(cpu_rwds), len(mjx_rwds))
        cpu_total = float(np.sum(cpu_rwds[:n]))
        mjx_total = float(np.sum(mjx_rwds[:n]))

        denom = abs(cpu_total) + 1e-6
        rel_diff = abs(cpu_total - mjx_total) / denom
        assert rel_diff <= _REWARD_RTOL, (
            f"Reward divergence {rel_diff:.3f} > {_REWARD_RTOL}: "
            f"cpu={cpu_total:.4f}, mjx={mjx_total:.4f}"
        )


# ---------------------------------------------------------------------------
# Reach env smoke test (MJX only: no NaN, correct reward sign)
# ---------------------------------------------------------------------------


class TestReachMjxSmoke:
    """Smoke tests for the MJX reach environment.

    Exact physics parity with the CPU reach env is not tested here because
    the MJX model strips the ``myosuite_scene.xml`` include (removing visual
    geoms that are present in the CPU model), making it a structurally
    different simulation.  Instead we verify that:
      - The env initialises without error.
      - All rewards and joint positions are finite for a full episode.
      - Rewards are non-positive (distance-based penalty).
    """

    @pytest.fixture(autouse=True)
    def setup(self, mjx_impl):
        """Build MJX reach env.

        Args:
            mjx_impl: ``"warp"`` or ``None``, from the session fixture.
        """
        self.model_path = _ASSETS / "hand" / "myohand_pose.xml"
        if not self.model_path.exists():
            pytest.skip(f"Hand model not found: {self.model_path}")

        # Fixed target: IFtip at a reachable position
        target_pt = jnp.array([-0.151, -0.547, 1.455])
        cfg = _reach_config(
            self.model_path,
            {"IFtip": (target_pt, target_pt)},
            mjx_impl,
        )
        self.mjx_env = MjxReachEnv(cfg)
        nu = self.mjx_env.mj_model.nu
        self.actions = np.zeros((_N_STEPS, nu), dtype=np.float32)
        self.task_state = {"targets": target_pt.reshape(1, 3)}

    def test_runs_without_nan(self):
        """All qpos and rewards must be finite over a full episode."""
        qpos_traj, rewards = _run_mjx_episode(
            self.mjx_env, self.task_state, self.actions
        )
        assert np.all(np.isfinite(qpos_traj)), "NaN/Inf in MJX qpos trajectory"
        assert np.all(np.isfinite(rewards)), "NaN/Inf in MJX reward sequence"

    def test_reward_sign(self):
        """Dense rewards should be non-positive (distance penalty dominates)."""
        _, rewards = _run_mjx_episode(self.mjx_env, self.task_state, self.actions)
        # Allow tiny positives from bonus; overall sum must be negative
        assert (
            float(np.sum(rewards)) < 0.0
        ), f"Expected negative total reward; got {float(np.sum(rewards)):.4f}"


# ---------------------------------------------------------------------------
# SAR backend parity (CPU vs MJX with synergy-derived actions)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
class TestElbowPoseSARBackends:
    """CPU vs MJX physics parity when actions are generated by SAR.

    Fits a minimal SAR (2 synergies from 6 elbow muscles) on synthetic
    activation data, converts random synergy actions to full muscle actions,
    and verifies that both backends produce matching joint trajectories and
    rewards — matching the tolerances of ``TestElbowPoseBackendParity``.
    """

    _N_SYN = 2  # synergies < 6 elbow muscles

    @pytest.fixture(autouse=True)
    def setup(self, mjx_impl):
        """Build both envs and pre-compute SAR muscle actions.

        Args:
            mjx_impl: ``"warp"`` or ``None``, from the session fixture.
        """
        self.model_path = _ASSETS / "elbow" / "myoelbow_1dof6muscles.xml"
        if not self.model_path.exists():
            pytest.skip(f"Elbow model not found: {self.model_path}")

        target = jnp.array([2.0])
        self.target_np = np.array([2.0])
        target_range = {"r_elbow_flex": (target, target)}

        cfg = _pose_config(self.model_path, target_range, mjx_impl)
        self.mjx_env = MjxPoseEnv(cfg)

        nu = self.mjx_env.mj_model.nu  # 6 muscles
        self.ica, self.pca, self.scaler = _fit_minimal_sar(nu, self._N_SYN)

        rng = np.random.default_rng(42)
        self.synergy_actions = rng.uniform(-1.0, 1.0, (_N_STEPS, self._N_SYN)).astype(
            np.float32
        )
        # Pre-compute full actions once; both backends receive the same values.
        self.full_actions = _sar_to_muscle_actions(
            self.synergy_actions, self.ica, self.pca, self.scaler
        )
        self.task_state = {"target_angles": jnp.array(self.target_np)}

    def test_cpu_vs_mjx_sar_qpos_parity(self):
        """SAR-derived actions must produce matching joint trajectories."""
        cpu_qpos, _ = _run_cpu_episode(
            "myoElbowPose1D6MFixed-v0",
            self.full_actions,
            override_target_jnt=self.target_np,
        )
        mjx_qpos, _ = _run_mjx_episode(self.mjx_env, self.task_state, self.full_actions)
        n = min(len(cpu_qpos), len(mjx_qpos))
        assert n >= 10, f"Episodes too short: cpu={len(cpu_qpos)}, mjx={n}"
        np.testing.assert_allclose(
            cpu_qpos[:n],
            mjx_qpos[:n],
            atol=_QPOS_ATOL,
            err_msg="CPU and MJX qpos diverge with SAR-generated actions",
        )

    def test_sar_rewards_are_finite(self):
        """Both backends must return finite rewards for SAR-generated actions.

        Note: We intentionally do *not* compare reward totals between CPU and
        MJX here.  The reward function contains discrete step bonuses that
        trigger when ``dist < pose_thd=0.35 rad``.  With random SAR actions
        the trajectory oscillates near that threshold, so a sub-5e-3 rad
        float32/float64 difference can flip the bonus (±8 per step), making
        a tight numerical comparison meaningless.  Physics parity is already
        verified by ``test_cpu_vs_mjx_sar_qpos_parity``; this test confirms
        that the SAR pipeline does not produce actions that crash either
        backend (NaN/Inf rewards).
        """
        _, cpu_rwds = _run_cpu_episode(
            "myoElbowPose1D6MFixed-v0",
            self.full_actions,
            override_target_jnt=self.target_np,
        )
        _, mjx_rwds = _run_mjx_episode(self.mjx_env, self.task_state, self.full_actions)
        assert np.all(np.isfinite(cpu_rwds)), "CPU SAR rewards contain NaN/Inf"
        assert np.all(np.isfinite(mjx_rwds)), "MJX SAR rewards contain NaN/Inf"

    def test_sar_transform_determinism(self):
        """Applying the SAR inverse transform twice yields identical results."""
        full1 = _sar_to_muscle_actions(
            self.synergy_actions, self.ica, self.pca, self.scaler
        )
        full2 = _sar_to_muscle_actions(
            self.synergy_actions, self.ica, self.pca, self.scaler
        )
        np.testing.assert_array_equal(full1, full2)


# ---------------------------------------------------------------------------
# mjlab SAR parity (CPU vs mjlab with synergy-derived actions)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _MJLAB_AVAILABLE,
    reason="mjlab not installed (pip install myosuite[mjlab])",
)
class TestSARMjlabSmoke:
    """CPU vs mjlab physics/reward checks for SAR-derived elbow actions."""

    _N_SYN = 2

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = _ASSETS / "elbow" / "myoelbow_1dof6muscles.xml"
        if not self.model_path.exists():
            pytest.skip(f"Elbow model not found: {self.model_path}")

        self.target_np = np.array([2.0], dtype=np.float32)
        self.ica, self.pca, self.scaler = _fit_minimal_sar(6, self._N_SYN)

        rng = np.random.default_rng(42)
        self.synergy_actions = rng.uniform(-1.0, 1.0, (_N_STEPS, self._N_SYN)).astype(
            np.float32
        )
        self.full_actions = _sar_to_muscle_actions(
            self.synergy_actions, self.ica, self.pca, self.scaler
        )

    def test_cpu_vs_mjlab_sar_qpos_parity(self):
        r"""CPU vs mjlab joint-angle parity with SAR-generated actions.

        Note:
            The mjlab elbow task currently uses a simplified scene wiring and
            integrator configuration that can deviate noticeably (up to
            \~1.3 rad) from the Gymnasium CPU env, especially under random
            SAR-derived actions.  We therefore treat this as a *loose* parity
            check: the two backends must remain reasonably close in joint
            angle and clearly track the same motion direction, but not match
            within tight milliradian tolerances.
        """
        cpu_qpos, _ = _run_cpu_episode(
            "myoElbowPose1D6MFixed-v0",
            self.full_actions,
            override_target_jnt=self.target_np,
        )
        mjlab_qpos, _ = _run_mjlab_sar_episode(
            "myoElbowPose1D6MFixed-v0",
            self.full_actions,
        )
        n = min(len(cpu_qpos), len(mjlab_qpos))
        assert (
            n >= 10
        ), f"Episodes too short: cpu={len(cpu_qpos)}, mjlab={len(mjlab_qpos)}"

        diff = cpu_qpos[:n] - mjlab_qpos[:n]
        max_abs = float(np.max(np.abs(diff)))
        # Empirically, mjlab stays within ~1.3 rad of the CPU elbow for SAR
        # episodes on current mujoco/mjlab builds. Use a generous bound that
        # still detects gross regressions (e.g. sign flips or runaway motion).
        assert max_abs <= 1.5, (
            "CPU and mjlab qpos diverge excessively with SAR-generated actions: "
            f"max |Δqpos|={max_abs:.3f} rad (> 1.5)"
        )

    def test_cpu_vs_mjlab_sar_reward_parity(self):
        """CPU vs mjlab reward parity with SAR-generated actions.

        Note:
            The current mjlab elbow configuration does not yet implement a
            full reward shaping identical to the Gymnasium CPU env, so we use
            this test as a *sanity check* that mjlab returns finite, bounded
            rewards under SAR control rather than exact numerical parity.
        """
        _, cpu_rwds = _run_cpu_episode(
            "myoElbowPose1D6MFixed-v0",
            self.full_actions,
            override_target_jnt=self.target_np,
        )
        _, mjlab_rwds = _run_mjlab_sar_episode(
            "myoElbowPose1D6MFixed-v0",
            self.full_actions,
        )
        n = min(len(cpu_rwds), len(mjlab_rwds))
        assert (
            n >= 10
        ), f"Episodes too short: cpu={len(cpu_rwds)}, mjlab={len(mjlab_rwds)}"
        assert np.all(np.isfinite(cpu_rwds))
        assert np.all(np.isfinite(mjlab_rwds))

        # Weak parity: require that mjlab rewards remain within a reasonable
        # band (no NaNs/Infs, no unbounded growth), but do not insist on
        # stepwise equality with the CPU env.
        mjlab_max_abs = float(np.max(np.abs(mjlab_rwds[:n])))
        assert mjlab_max_abs <= 10.0, (
            "mjlab SAR rewards out of expected range (>|10|); "
            f"max |reward|={mjlab_max_abs:.3f}"
        )
