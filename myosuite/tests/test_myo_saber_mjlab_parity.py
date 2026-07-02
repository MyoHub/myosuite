from __future__ import annotations

import types

import numpy as np
import pytest

pytestmark = pytest.mark.tier2

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _TORCH_AVAILABLE = False

try:
    import mjlab  # noqa: F401

    _MJLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _MJLAB_AVAILABLE = False


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE),
    reason="mjlab and torch not installed (pip install myosuite[mjlab])",
)
def test_myo_challenge_saber_mjlab_matches_cpu_control_contract() -> None:
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env
    from myosuite.envs.myo.backends.mjlab.saber_mjlab_env import (
        _saber_joint_pos_obs,
    )

    cpu_env = gym.make("myoChallengeSaberP0-v0")
    mj_env = make_env("myoChallengeSaberP0-v0", backend="mjlab", device="cpu")
    mj_entity = mj_env.scene["saber_p0_robot"]

    try:
        cpu_obs, _ = cpu_env.reset(seed=0)
        cpu = cpu_env.unwrapped
        mj_obs, _ = mj_env.reset(seed=0)
        mj_obs_policy = mj_obs["policy"][0].detach().cpu().numpy()

        # Compute target-pool slice from the task config obs key order so the
        # offset stays correct if model dimensions change.
        from myosuite.envs.myo.tasks.challenge.saber_task_spec import SABER_P0_ENV_ID  # noqa: F401

        _task_cfg = cpu._task_config  # noqa: SLF001
        _obs_keys = list(_task_cfg.obs.keys)
        _obs_dims = {
            "joint_pos": cpu.model.nq,
            "joint_vel": cpu.model.nv,
            "muscle_act": cpu.model.na,
            "saber_target_pool": int(_task_cfg.obs.extra["num_visible"]) * 8,
            "time": 1,
            "health_status": 1,
        }
        _pool_start = sum(
            _obs_dims.get(k, 0)
            for k in _obs_keys[: _obs_keys.index("saber_target_pool")]
        )
        _pool_end = _pool_start + _obs_dims["saber_target_pool"]

        def _target_pool_slice(obs: np.ndarray) -> np.ndarray:
            return obs[_pool_start:_pool_end]

        assert np.asarray(cpu_obs).shape == tuple(mj_obs["policy"].shape[1:])
        assert cpu_env.action_space.shape[0] == mj_env.action_space.shape[-1]
        assert np.allclose(
            cpu.data.qpos.copy(),
            _saber_joint_pos_obs(mj_env)[0].detach().cpu().numpy(),
            atol=1e-6,
        )
        assert np.allclose(np.asarray(cpu_obs), mj_obs_policy, atol=1e-6)

        ctrl_action = np.full(cpu_env.action_space.shape[0], 0.3, dtype=np.float32)
        cpu_env.step(ctrl_action)
        mj_env.step(
            torch.full((1, mj_env.action_space.shape[-1]), 0.3, dtype=torch.float32)
        )
        ctrl_cpu = cpu.data.ctrl.copy()
        ctrl_mj = mj_entity.data.data.ctrl[0].detach().cpu().numpy()
        assert ctrl_cpu.shape == ctrl_mj.shape
        assert np.allclose(ctrl_cpu, ctrl_mj, atol=1e-6)

        zero_action = np.zeros(cpu_env.action_space.shape[0], dtype=np.float32)
        for seed in (0, 1, 7):
            cpu_obs, _ = cpu_env.reset(seed=seed)
            cpu = cpu_env.unwrapped
            mj_obs, _ = mj_env.reset(seed=seed)
            mj_obs_policy = mj_obs["policy"][0].detach().cpu().numpy()

            assert np.allclose(np.asarray(cpu_obs), mj_obs_policy, atol=1e-6)
            assert np.allclose(
                _target_pool_slice(np.asarray(cpu_obs)),
                _target_pool_slice(mj_obs_policy),
                atol=0.0,
            )

            for _ in range(5):
                cpu_obs, reward_cpu, term_cpu, trunc_cpu, _ = cpu_env.step(zero_action)
                mj_obs, reward_mj, term_mj, trunc_mj, info_mj = mj_env.step(
                    torch.zeros(1, mj_env.action_space.shape[-1], dtype=torch.float32)
                )
                mj_obs_policy = mj_obs["policy"][0].detach().cpu().numpy()

                assert bool(term_cpu) is bool(term_mj[0])
                assert bool(trunc_cpu) is bool(trunc_mj[0])
                assert np.isfinite(reward_cpu)
                assert np.isfinite(float(reward_mj[0]))
                assert np.allclose(
                    _target_pool_slice(np.asarray(cpu_obs)),
                    _target_pool_slice(mj_obs_policy),
                    atol=0.0,
                )
                assert cpu._task_state["health_status"] == pytest.approx(0.5)  # noqa: SLF001
                assert float(
                    info_mj["task_state"]["health_status"][0]
                ) == pytest.approx(0.5)
                assert int(cpu._task_state["target_pool_total_spawned"]) == int(  # noqa: SLF001
                    info_mj["task_state"]["target_pool_total_spawned"][0]
                )
    finally:
        cpu_env.close()
        if hasattr(mj_env, "close"):
            mj_env.close()


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE),
    reason="mjlab and torch not installed (pip install myosuite[mjlab])",
)
def test_myo_challenge_saber_mjlab_updates_target_pool_visuals() -> None:
    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env
    from myosuite.envs.myo.backends.mjlab.saber_mjlab_env import (
        _get_saber_logic,
    )

    env = make_env("myoChallengeSaberP0-v0", backend="mjlab", device="cpu")
    try:
        env.reset(seed=0)
        assert "geom_rgba" in env.sim.expanded_fields
        logic = _get_saber_logic(env)

        def _geom_rgba(geom_id: int) -> np.ndarray:
            geom_rgba = env.sim.model.geom_rgba
            if len(geom_rgba.shape) == 3:
                return geom_rgba[0, geom_id].detach().cpu().numpy()
            return geom_rgba[geom_id].detach().cpu().numpy()

        first_cube = int(logic.target_pool_geom_ids[0])
        first_markers = [int(v) for v in logic.target_pool_marker_geom_ids[0].tolist()]
        assert np.allclose(
            _geom_rgba(first_cube), np.array([0.55, 0.55, 0.55, 0.85]), atol=1e-6
        )
        for marker_gid in first_markers:
            assert np.allclose(_geom_rgba(marker_gid), np.zeros(4), atol=1e-6)

        zero_action = torch.zeros(1, env.action_space.shape[-1], dtype=torch.float32)
        for _ in range(10):
            env.step(zero_action)

        active_targets = torch.nonzero(logic.pool_active[0], as_tuple=False).squeeze(-1)
        assert len(active_targets) > 0
        active_idx = int(active_targets[0])
        cube_rgba = _geom_rgba(int(logic.target_pool_geom_ids[active_idx]))
        assert np.allclose(cube_rgba, np.array([0.80, 0.80, 0.80, 1.0]), atol=1e-6)

        marker_ids = [
            int(v) for v in logic.target_pool_marker_geom_ids[active_idx].tolist()
        ]
        marker_rgba = np.stack([_geom_rgba(gid) for gid in marker_ids], axis=0)
        visible = np.where(marker_rgba[:, 3] > 0.99)[0]
        assert visible.tolist() == [int(logic.pool_preferred_hit_area[0, active_idx])]
        expected_color = (
            np.array([0.05, 0.25, 1.0, 1.0])
            if int(logic.pool_preferred_hand[0, active_idx]) == 0
            else np.array([1.0, 0.10, 0.10, 1.0])
        )
        assert np.allclose(marker_rgba[visible[0]], expected_color, atol=1e-6)
    finally:
        env.close()


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE),
    reason="mjlab and torch not installed (pip install myosuite[mjlab])",
)
def test_myo_challenge_saber_mjlab_has_nan_termination() -> None:
    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env

    env = make_env("myoChallengeSaberP0-v0", backend="mjlab", device="cpu")
    try:
        assert "nan_detection" in env.cfg.terminations
        assert env.cfg.terminations["nan_detection"].func.__name__ == "nan_detection"
    finally:
        env.close()


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE),
    reason="mjlab and torch not installed (pip install myosuite[mjlab])",
)
def test_myo_challenge_saber_mjlab_matches_cpu_seeded_first_spawn() -> None:
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env
    from myosuite.envs.myo.backends.mjlab.saber_mjlab_env import (
        _get_saber_logic,
    )

    cpu_env = gym.make("myoChallengeSaberP0-v0")
    mj_env = make_env("myoChallengeSaberP0-v0", backend="mjlab", device="cpu")
    zero_action = np.zeros(cpu_env.action_space.shape[0], dtype=np.float32)
    zero_action_t = torch.zeros(1, mj_env.action_space.shape[-1], dtype=torch.float32)

    try:
        cpu_env.reset(seed=0)
        mj_env.reset(seed=0)
        cpu = cpu_env.unwrapped
        logic = _get_saber_logic(mj_env)

        for _ in range(10):
            cpu_env.step(zero_action)
            mj_env.step(zero_action_t)

        cpu_runtime = cpu._target_pool_runtime  # noqa: SLF001
        assert cpu_runtime is not None
        cpu_active = np.flatnonzero(cpu_runtime.active)
        mj_active = (
            torch.nonzero(logic.pool_active[0], as_tuple=False)
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
        assert cpu_active.tolist() == mj_active.tolist()
        assert cpu_active.size > 0

        active_idx = int(cpu_active[0])
        cpu_pos = cpu.data.mocap_pos[int(cpu._target_pool_mocap_ids[active_idx])].copy()  # noqa: SLF001
        mj_pos = logic.target_pos[0, active_idx].detach().cpu().numpy()
        np.testing.assert_allclose(cpu_pos, mj_pos, atol=1e-6)
        assert int(cpu_runtime.preferred_hand[active_idx]) == int(
            logic.pool_preferred_hand[0, active_idx]
        )
        assert int(cpu_runtime.preferred_hit_area[active_idx]) == int(
            logic.pool_preferred_hit_area[0, active_idx]
        )
        assert bool(cpu_runtime.is_obstacle[active_idx]) is bool(
            logic.pool_is_obstacle[0, active_idx]
        )
    finally:
        cpu_env.close()
        mj_env.close()


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE),
    reason="mjlab and torch not installed (pip install myosuite[mjlab])",
)
def test_myo_challenge_saber_mjlab_mimic_mix_preserves_native_reset() -> None:
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()
    from mjlab.tasks.registry import register_mjlab_task
    from myosuite.core.registry import make_env
    from myosuite.core.trajectory_io import MotionClip
    from myosuite.envs.myo.backends.mjlab.register_mjlab_saber import (
        register_saber_p0_mjlab_task_with_mimic_mix,
    )
    from myosuite.integrations.musclemimic.bimanual_model import BODY2SITES_FOR_MIMIC

    def _make_clip(offset: float) -> MotionClip:
        n_frames = 8
        site_names = list(BODY2SITES_FOR_MIMIC.values())
        qpos = np.zeros((n_frames, 108), dtype=np.float32)
        qvel = np.zeros((n_frames, 106), dtype=np.float32)
        site_xpos = np.zeros((n_frames, len(site_names), 3), dtype=np.float32)
        for frame in range(n_frames):
            qpos[frame, :3] = np.asarray(
                [offset + frame * 0.01, offset, 1.0 + offset],
                dtype=np.float32,
            )
            qpos[frame, 3] = 1.0
            site_xpos[frame] = offset + frame * 0.05
        return MotionClip(
            qpos=qpos,
            qvel=qvel,
            site_xpos=site_xpos,
            site_names=site_names,
            frequency_hz=100.0,
            source_path=None,
        )

    env_only_id = "myoChallengeSaberP0NativeMimicObsTest-v0"
    aug_id = "myoChallengeSaberP0NativeMimicMixTest-v0"
    clips = (_make_clip(0.25), _make_clip(0.75))
    for task_id, reward_mode, mimic_reward_weight in (
        (env_only_id, "env", 0.5),
        (aug_id, "augmented", 0.5),
    ):
        try:
            register_saber_p0_mjlab_task_with_mimic_mix(
                register_mjlab_task,
                lambda **_: types.SimpleNamespace(),
                task_id=task_id,
                clips=clips,
                reward_mode=reward_mode,
                mimic_reward_weight=mimic_reward_weight,
                env_reward_weight=1.0,
            )
        except ValueError:
            pass

    cpu_env = gym.make("myoChallengeSaberP0-v0")
    native_env = make_env("myoChallengeSaberP0-v0", backend="mjlab", device="cpu")
    env_only = make_env(env_only_id, backend="mjlab", device="cpu")
    aug_env = make_env(aug_id, backend="mjlab", device="cpu")
    zero_action = np.zeros(cpu_env.action_space.shape[0], dtype=np.float32)
    zero_action_t = torch.zeros(
        1, native_env.action_space.shape[-1], dtype=torch.float32
    )

    try:
        cpu_obs, _ = cpu_env.reset(seed=3)
        native_obs, _ = native_env.reset(seed=3)
        env_only_obs, _ = env_only.reset(seed=3)
        aug_obs, _ = aug_env.reset(seed=3)

        native_policy = native_obs["policy"][0].detach().cpu().numpy()
        env_only_policy = env_only_obs["policy"][0].detach().cpu().numpy()
        aug_policy = aug_obs["policy"][0].detach().cpu().numpy()

        assert native_policy.shape[0] == 632
        assert env_only_policy.shape[0] == 1600
        assert aug_policy.shape[0] == 1600
        assert np.allclose(np.asarray(cpu_obs), native_policy, atol=1e-6)
        assert np.allclose(env_only_policy[:632], native_policy, atol=1e-6)
        assert np.allclose(aug_policy[:632], native_policy, atol=1e-6)

        _, reward_cpu, _, _, _ = cpu_env.step(zero_action)
        _, reward_native, _, _, _ = native_env.step(zero_action_t)
        _, reward_env_only, _, _, _ = env_only.step(zero_action_t)
        _, reward_aug, _, _, _ = aug_env.step(zero_action_t)

        assert float(reward_env_only[0]) == pytest.approx(
            float(reward_native[0]),
            abs=5e-5,
        )
        assert float(reward_aug[0]) > float(reward_env_only[0])
    finally:
        cpu_env.close()
        native_env.close()
        env_only.close()
        aug_env.close()


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE),
    reason="mjlab and torch not installed (pip install myosuite[mjlab])",
)
def test_myo_challenge_saber_mjlab_mimic_mix_supports_reference_state_init() -> None:
    import myosuite

    myosuite.register_all_envs()
    from mjlab.tasks.registry import register_mjlab_task
    from myosuite.core.registry import make_env
    from myosuite.core.trajectory_io import MotionClip
    from myosuite.envs.myo.backends.mjlab.register_mjlab_saber import (
        register_saber_p0_mjlab_task_with_mimic_mix,
    )
    from myosuite.envs.myo.backends.mjlab.saber_mjlab_env import (
        _saber_mimic_qpos_obs,
        _saber_mimic_qvel_state,
    )
    from myosuite.integrations.musclemimic.bimanual_model import BODY2SITES_FOR_MIMIC

    def _make_clip(offset: float) -> MotionClip:
        n_frames = 6
        site_names = list(BODY2SITES_FOR_MIMIC.values())
        qpos = np.zeros((n_frames, 108), dtype=np.float32)
        qvel = np.zeros((n_frames, 106), dtype=np.float32)
        site_xpos = np.zeros((n_frames, len(site_names), 3), dtype=np.float32)
        for frame in range(n_frames):
            qpos[frame, :3] = np.asarray(
                [offset + frame * 0.02, offset * 0.5, 1.0 + offset],
                dtype=np.float32,
            )
            qpos[frame, 3] = 1.0
            qvel[frame, :3] = np.asarray([0.1 + offset, 0.0, 0.0], dtype=np.float32)
            site_xpos[frame] = offset + frame * 0.05
        return MotionClip(
            qpos=qpos,
            qvel=qvel,
            site_xpos=site_xpos,
            site_names=site_names,
            frequency_hz=100.0,
            source_path=None,
        )

    task_id = "myoChallengeSaberP0NativeMimicRSITest-v0"
    try:
        register_saber_p0_mjlab_task_with_mimic_mix(
            register_mjlab_task,
            lambda **_: types.SimpleNamespace(),
            task_id=task_id,
            clips=(_make_clip(0.1), _make_clip(0.4)),
            reward_mode="mimic",
            env_reward_weight=0.0,
            use_early_termination=True,
            use_reference_state_initialization=True,
        )
    except ValueError:
        pass

    env = make_env(task_id, backend="mjlab", device="cpu")
    try:
        obs, _ = env.reset(seed=11)
        policy_obs = obs["policy"][0].detach().cpu().numpy()
        ref_qpos_obs = policy_obs[1246:1354]
        ref_qvel_obs = policy_obs[1354:1460]
        sim_qpos = _saber_mimic_qpos_obs(env)[0].detach().cpu().numpy()
        sim_qvel = _saber_mimic_qvel_state(env)[0].detach().cpu().numpy()

        assert np.allclose(sim_qpos, ref_qpos_obs, atol=1e-5)
        assert np.allclose(sim_qvel, ref_qvel_obs, atol=1e-5)
        assert "mimic_rsi" in env.cfg.events
        assert "keyframe_reset" not in env.cfg.events
    finally:
        env.close()
