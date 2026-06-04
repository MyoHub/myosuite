# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for mjlab trajectory-playback (ClipTrajectorySource).

No mjlab or musclemimic_models installation required — all physics state is
mocked with lightweight ``torch.Tensor`` namespaces identical to those used in
``test_mjlab_compat.py``.

Test classes
------------
``TestClipTrajectorySourceBasics``
    Unit tests for :class:`ClipTrajectorySource` in isolation:
    frame-index arithmetic, per-env reset detection, device placement.

``TestClipTrajectorySourceAdvance``
    Verifies that targets advance frame-by-frame as simulation time progresses.

``TestClipTrajectorySourceReset``
    Verifies that start offsets are resampled independently when individual
    environments reset (their time regresses).

``TestMimicMjlabCacheDispatch``
    Verifies that :func:`_sync_mimic_mjlab_targets` dispatches to the clip
    source when one is present in the cache, leaving the random path intact.

``TestMimicMjlabClosures``
    Verifies the obs and reward closure factories using a fully mocked mjlab
    environment and an injected ``ClipTrajectorySource``.

``TestInitialPoseHelpers``
    Verifies ``initial_qpos`` / ``initial_qvel`` and ``make_init_state_fn``.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest
import torch

from myosuite.core.trajectory_io import MotionClip
from myosuite.envs.myo.backends.mjlab.clip_trajectory_source import (
    ClipTrajectorySource,
    MultiClipTrajectorySource,
)
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
    _init_state_from_model,
    _mimic_keyframe_reset_event,
    _mimic_mjlab_cache,
    _mimic_obs_act,
    _mimic_obs_clip_phase,
    _mimic_obs_clip_ref_qpos,
    _mimic_obs_clip_ref_qvel,
    _mimic_obs_err,
    _mimic_obs_qpos,
    _mimic_obs_qvel,
    _mimic_obs_site_pos,
    _mimic_obs_target,
    _mimic_tracking_reward,
    _mjlab_sim_time_to_tensor,
    _normalize_mimic_reward_mode,
    _sync_mimic_mjlab_targets,
)
from myosuite.envs.myo.backends.mjlab.register_mjlab_saber import (
    SaberMimicMixCfg,
    register_saber_p0_mjlab_task_with_mimic_mix,
)

pytestmark = pytest.mark.tier2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N = 4  # number of parallel envs
_T = 50  # clip length (frames)
_N_SITES = 7  # number of tracked sites
_NQ = 17
_NV = 16
_NA = 80
_CTRL_DT = 0.01


def _make_clip(
    T: int = _T,
    n_model_sites: int = _N_SITES,
    nq: int = _NQ,
    nv: int = _NV,
    with_qpos: bool = True,
    with_qvel: bool = True,
) -> MotionClip:
    """Build a synthetic MotionClip with deterministic data."""
    rng = np.random.default_rng(0)
    site_xpos = rng.uniform(0.1, 2.0, (T, n_model_sites, 3)).astype(np.float32)
    qpos = rng.uniform(-0.1, 0.1, (T, nq)).astype(np.float32) if with_qpos else None
    qvel = rng.uniform(-0.05, 0.05, (T, nv)).astype(np.float32) if with_qvel else None
    return MotionClip(
        qpos=qpos,
        qvel=qvel,
        site_xpos=site_xpos,
        site_names=None,
        frequency_hz=100.0,
        source_path=None,
    )


def _make_site_ids(n: int = _N_SITES) -> np.ndarray:
    return np.arange(n, dtype=np.int32)


def _make_source(
    T: int = _T,
    n_sites: int = _N_SITES,
    nq: int = _NQ,
    nv: int = _NV,
    ctrl_dt: float = _CTRL_DT,
) -> ClipTrajectorySource:
    clip = _make_clip(T=T, n_model_sites=n_sites, nq=nq, nv=nv)
    return ClipTrajectorySource(
        clip=clip,
        tracked_site_ids=_make_site_ids(n_sites),
        ctrl_dt=ctrl_dt,
    )


def _make_multi_clip_source() -> MultiClipTrajectorySource:
    clip_a = _make_clip(T=5, n_model_sites=_N_SITES, nq=_NQ, nv=_NV)
    clip_b = _make_clip(T=7, n_model_sites=_N_SITES, nq=_NQ, nv=_NV)
    clip_a = MotionClip(
        qpos=clip_a.qpos + 10.0,
        qvel=clip_a.qvel + 100.0,
        site_xpos=clip_a.site_xpos + 1.0,
        site_names=clip_a.site_names,
        qpos_model_indices=clip_a.qpos_model_indices,
        qvel_model_indices=clip_a.qvel_model_indices,
        frequency_hz=clip_a.frequency_hz,
        source_path=clip_a.source_path,
    )
    clip_b = MotionClip(
        qpos=clip_b.qpos + 20.0,
        qvel=clip_b.qvel + 200.0,
        site_xpos=clip_b.site_xpos + 2.0,
        site_names=clip_b.site_names,
        qpos_model_indices=clip_b.qpos_model_indices,
        qvel_model_indices=clip_b.qvel_model_indices,
        frequency_hz=clip_b.frequency_hz,
        source_path=clip_b.source_path,
    )
    return MultiClipTrajectorySource(
        clips=(clip_a, clip_b),
        tracked_site_ids=_make_site_ids(_N_SITES),
        ctrl_dt=_CTRL_DT,
    )


def _make_mock_env(
    n_envs: int = _N,
    nq: int = _NQ,
    nv: int = _NV,
    na: int = _NA,
    n_sites: int = _N_SITES,
    t: float | torch.Tensor = 0.0,
    entity_name: str = "robot",
) -> Any:
    """Build a minimal namespace that mimics the mjlab env + scene layout.

    In real mjlab the physics data is accessed as::

        env.scene[entity_name].data.data   # → MjData-like object

    The double ``.data`` nesting is an mjlab convention:
    - ``entity.data``      — the entity's simulation-data *holder*
    - ``entity.data.data`` — the actual physics arrays (qpos, qvel, …)
    """

    class _MockScene(dict):
        def __init__(
            self, *args: Any, env_origins: torch.Tensor, **kwargs: Any
        ) -> None:
            super().__init__(*args, **kwargs)
            self.env_origins = env_origins

    class _MockSim:
        def __init__(self) -> None:
            self.forward_calls = 0

        def forward(self) -> None:
            self.forward_calls += 1

    if isinstance(t, float):
        time_tensor = torch.full((n_envs,), t, dtype=torch.float32)
    else:
        time_tensor = t.float()

    # Innermost: actual physics arrays
    physics = types.SimpleNamespace(
        qpos=torch.zeros(n_envs, nq),
        qvel=torch.zeros(n_envs, nv),
        act=torch.zeros(n_envs, na),
        site_xpos=torch.zeros(n_envs, n_sites, 3),
        time=time_tensor,
        ctrl_range=torch.zeros(na, 2),
    )
    # mjlab double-nesting: entity.data.data → physics
    data_holder = types.SimpleNamespace(data=physics)
    entity = types.SimpleNamespace(data=data_holder)
    scene = _MockScene(
        {entity_name: entity},
        env_origins=torch.zeros(n_envs, 3, dtype=torch.float32),
    )
    return types.SimpleNamespace(
        scene=scene,
        sim=_MockSim(),
        physics_dt=0.002,
        cfg=types.SimpleNamespace(decimation=5),
    )


def test_register_saber_p0_clip_task_registers_expected_env(monkeypatch) -> None:
    """Clip-mode registration should add a split-scene mjlab saber mimic env."""
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "myosuite.envs.myo.backends.mjlab.register_mjlab_saber._make_saber_mimic_env_cfg",
        lambda *, play, mimic_mix: {"play": play, "mimic_mix": mimic_mix},
    )

    def _register_mjlab_task(**kwargs: Any) -> None:
        captured.update(kwargs)

    clip = _make_clip(nq=17, nv=16)
    register_saber_p0_mjlab_task_with_mimic_mix(
        _register_mjlab_task,
        lambda **_: "rl-cfg",
        mimic_mix=SaberMimicMixCfg(
            clips=(clip,),
            reward_mode="augmented",
            mimic_reward_weight=0.25,
            env_reward_weight=2.0,
        ),
    )

    from myosuite.envs.myo.tasks.challenge.saber_task_spec import SABER_P0_MIMIC_ENV_ID

    assert captured["task_id"] == SABER_P0_MIMIC_ENV_ID
    mm = captured["env_cfg"]["mimic_mix"]
    assert mm.reward_mode == "augmented"
    assert mm.mimic_reward_weight == pytest.approx(0.25)
    assert mm.env_reward_weight == pytest.approx(2.0)
    assert captured["play_env_cfg"]["play"] is True


def test_normalize_mimic_reward_mode_rejects_invalid() -> None:
    """Reward-mode validation should reject unknown composition modes."""
    assert _normalize_mimic_reward_mode("ENV") == "env"
    with pytest.raises(ValueError, match="reward_mode must be one of"):
        _normalize_mimic_reward_mode("hybrid")


def test_init_state_from_model_handles_fixed_base_with_free_props(monkeypatch) -> None:
    """Fixed-base models must not assume a leading free root joint."""
    from myosuite.tests.support.optional_deps import require_mjlab

    require_mjlab()
    import mujoco

    names = {0: "hinge0", 1: "slide1", 2: "prop_free"}
    monkeypatch.setattr(
        mujoco,
        "mj_id2name",
        lambda _model, _obj, idx: names.get(int(idx)),
    )
    model = types.SimpleNamespace(
        nkey=1,
        njnt=3,
        key_qpos=np.array([[0.1, -0.2, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
        key_qvel=np.array([[0.3, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        jnt_type=np.array(
            [
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
                int(mujoco.mjtJoint.mjJNT_FREE),
            ],
            dtype=np.int32,
        ),
        jnt_qposadr=np.array([0, 1, 2], dtype=np.int32),
        jnt_dofadr=np.array([0, 1, 2], dtype=np.int32),
    )

    init = _init_state_from_model(model)

    assert init.pos == (0.0, 0.0, 0.0)
    assert init.rot == (1.0, 0.0, 0.0, 0.0)
    assert init.joint_pos == {"hinge0": 0.1, "slide1": -0.2}
    assert init.joint_vel == {"hinge0": 0.3, "slide1": -0.4}


def test_keyframe_reset_event_restores_aux_free_joints() -> None:
    """Keyframe reset must restore prop free-joint state and apply env offsets."""
    import mujoco

    model = types.SimpleNamespace(
        nkey=1,
        nq=15,
        nv=13,
        njnt=3,
        nu=4,
        key_qpos=np.array(
            [
                [
                    0.25,
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    -2.0,
                    -3.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ],
            dtype=np.float32,
        ),
        key_qvel=np.array([[0.5] + [0.0] * 12], dtype=np.float32),
        key_act=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        jnt_type=np.array(
            [
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_FREE),
                int(mujoco.mjtJoint.mjJNT_FREE),
            ],
            dtype=np.int32,
        ),
        jnt_qposadr=np.array([0, 1, 8], dtype=np.int32),
    )
    env = _make_mock_env(nq=15, nv=13, na=4)
    env.scene.env_origins = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
            [0.0, 0.0, 0.0],
            [-5.0, -6.0, -7.0],
        ],
        dtype=torch.float32,
    )

    event = _mimic_keyframe_reset_event("robot", model)
    event(env, torch.tensor([1, 3], dtype=torch.long))

    qpos = env.scene["robot"].data.data.qpos
    qvel = env.scene["robot"].data.data.qvel
    act = env.scene["robot"].data.data.act

    np.testing.assert_allclose(qvel[1].numpy(), model.key_qvel[0], atol=1e-6)
    np.testing.assert_allclose(qvel[3].numpy(), model.key_qvel[0], atol=1e-6)
    np.testing.assert_allclose(act[1].numpy(), model.key_act[0], atol=1e-6)
    np.testing.assert_allclose(act[3].numpy(), model.key_act[0], atol=1e-6)
    assert qpos[1, 0].item() == pytest.approx(0.25)
    np.testing.assert_allclose(
        qpos[1, 1:4].numpy(), np.array([11.0, 22.0, 33.0]), atol=1e-6
    )
    np.testing.assert_allclose(
        qpos[1, 8:11].numpy(), np.array([9.0, 18.0, 27.0]), atol=1e-6
    )
    np.testing.assert_allclose(
        qpos[3, 1:4].numpy(), np.array([-4.0, -4.0, -4.0]), atol=1e-6
    )
    np.testing.assert_allclose(
        qpos[3, 8:11].numpy(), np.array([-6.0, -8.0, -10.0]), atol=1e-6
    )
    assert env.sim.forward_calls == 1


def _make_cache_with_source(
    source: ClipTrajectorySource,
    n_sites: int = _N_SITES,
) -> dict[str, Any]:
    """Build a cache dict that routes through ClipTrajectorySource."""
    rng = np.random.default_rng(1)
    return dict(
        site_ids=_make_site_ids(n_sites),
        lo=rng.uniform(-1.0, 0.0, 3),
        hi=rng.uniform(0.0, 1.0, 3),
        tracking=types.SimpleNamespace(reward_scale=20.0, success_threshold=0.04),
        clip_source=source,
        last_sim_time=None,
        target_torch=None,
    )


def _make_cache_random(n_sites: int = _N_SITES) -> dict[str, Any]:
    """Build a cache dict that uses random box sampling (no clip)."""
    np.random.default_rng(2)
    return dict(
        site_ids=_make_site_ids(n_sites),
        lo=np.array([-0.5, -0.5, 0.0]),
        hi=np.array([0.5, 0.5, 2.0]),
        tracking=types.SimpleNamespace(reward_scale=20.0, success_threshold=0.04),
        clip_source=None,
        last_sim_time=None,
        target_torch=None,
    )


# ---------------------------------------------------------------------------
# TestClipTrajectorySourceBasics
# ---------------------------------------------------------------------------


class TestClipTrajectorySourceBasics:
    def test_raises_without_site_xpos(self) -> None:
        clip = MotionClip(
            qpos=np.zeros((10, 5)),
            qvel=None,
            site_xpos=None,
            site_names=None,
            frequency_hz=100.0,
            source_path=None,
        )
        with pytest.raises(ValueError, match="site_xpos"):
            ClipTrajectorySource(clip=clip, tracked_site_ids=np.arange(5), ctrl_dt=0.01)

    def test_n_frames(self) -> None:
        src = _make_source(T=60)
        assert src.n_frames == 60

    def test_n_tracked(self) -> None:
        src = _make_source(n_sites=9)
        assert src.n_tracked == 9

    def test_update_initialises_device(self) -> None:
        src = _make_source()
        t = torch.zeros(_N)
        src.update(t)
        assert src._site_tensor is not None
        assert src._start_offsets is not None
        assert src._last_t is not None

    def test_site_tensor_shape(self) -> None:
        src = _make_source(T=_T, n_sites=_N_SITES)
        t = torch.zeros(_N)
        src.update(t)
        assert src._site_tensor is not None
        assert src._site_tensor.shape == (_T, _N_SITES, 3)

    def test_qpos_tensor_shape(self) -> None:
        src = _make_source(T=_T, nq=_NQ)
        t = torch.zeros(_N)
        src.update(t)
        assert src._qpos_tensor is not None
        assert src._qpos_tensor.shape == (_T, _NQ)

    def test_no_qpos_tensor_when_missing(self) -> None:
        clip = _make_clip(with_qpos=False, with_qvel=False)
        src = ClipTrajectorySource(
            clip=clip, tracked_site_ids=_make_site_ids(), ctrl_dt=_CTRL_DT
        )
        t = torch.zeros(_N)
        src.update(t)
        assert src._qpos_tensor is None
        assert src.ref_qpos(t) is None
        assert src.ref_qvel(t) is None

    def test_start_offsets_in_range(self) -> None:
        src = _make_source(T=_T)
        t = torch.zeros(_N)
        src.update(t)
        assert src._start_offsets is not None
        assert (src._start_offsets >= 0).all()
        assert (src._start_offsets < _T).all()

    def test_site_targets_shape(self) -> None:
        src = _make_source()
        t = torch.zeros(_N)
        src.update(t)
        out = src.site_targets(t)
        assert out.shape == (_N, _N_SITES, 3)

    def test_ref_qpos_shape(self) -> None:
        src = _make_source(nq=_NQ)
        t = torch.zeros(_N)
        src.update(t)
        out = src.ref_qpos(t)
        assert out is not None
        assert out.shape == (_N, _NQ)

    def test_phase_shape_and_range(self) -> None:
        src = _make_source()
        t = torch.zeros(_N)
        src.update(t)
        ph = src.phase(t)
        assert ph.shape == (_N, 1)
        assert (ph >= 0.0).all()
        assert (ph < 1.0).all()

    def test_multi_clip_source_uses_per_env_clip_assignments(self) -> None:
        src = _make_multi_clip_source()
        t = torch.tensor([0.00, 0.02, 0.04, 0.06], dtype=torch.float32)
        src.update(t)
        src._clip_indices = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        src._start_offsets = torch.tensor([1, 2, 3, 4], dtype=torch.long)

        frame_idx = src.frame_indices(t)
        clip_lengths = src.clip_lengths(t)
        targets = src.site_targets(t)
        qpos = src.ref_qpos(t)
        qvel = src.ref_qvel(t)

        torch.testing.assert_close(frame_idx, torch.tensor([1, 4, 2, 3]))
        torch.testing.assert_close(clip_lengths, torch.tensor([5, 7, 5, 7]))
        np.testing.assert_allclose(
            targets[0].cpu().numpy(),
            np.asarray(src.clips[0].site_xpos[1], dtype=np.float32),
        )
        np.testing.assert_allclose(
            targets[1].cpu().numpy(),
            np.asarray(src.clips[1].site_xpos[4], dtype=np.float32),
        )
        assert qpos is not None
        assert qvel is not None
        np.testing.assert_allclose(
            qpos[2].cpu().numpy(),
            np.asarray(src.clips[0].qpos[2], dtype=np.float32),
        )
        np.testing.assert_allclose(
            qvel[3].cpu().numpy(),
            np.asarray(src.clips[1].qvel[3], dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# TestClipTrajectorySourceAdvance
# ---------------------------------------------------------------------------


class TestClipTrajectorySourceAdvance:
    def test_targets_change_as_time_advances(self) -> None:
        """Targets must change when simulation time advances by ctrl_dt."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        # Pin start_offsets to 0 for determinism
        t0 = torch.zeros(_N)
        src.update(t0)
        src._start_offsets = torch.zeros(_N, dtype=torch.long)

        tgt0 = src.site_targets(t0).clone()

        t1 = torch.full((_N,), _CTRL_DT)  # one frame later
        src.update(t1)
        tgt1 = src.site_targets(t1)

        assert not torch.allclose(
            tgt0, tgt1
        ), "Targets must differ after advancing one frame"

    def test_targets_match_clip_data(self) -> None:
        """Targets at frame k must equal clip.site_xpos[k]."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t0 = torch.zeros(_N)
        src.update(t0)
        # Force all envs to start at frame 0
        src._start_offsets = torch.zeros(_N, dtype=torch.long)

        for frame_k in [0, 1, 5, _T - 1]:
            t_k = torch.full((_N,), frame_k * _CTRL_DT)
            expected = (
                torch.as_tensor(src.clip.site_xpos[frame_k], dtype=torch.float32)
                .unsqueeze(0)
                .expand(_N, -1, -1)
            )
            actual = src.site_targets(t_k)
            assert torch.allclose(
                actual, expected, atol=1e-6
            ), f"Frame {frame_k}: targets don't match clip"

    def test_clip_wraps_at_end(self) -> None:
        """After T frames the clip wraps back to frame 0."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t0 = torch.zeros(_N)
        src.update(t0)
        src._start_offsets = torch.zeros(_N, dtype=torch.long)

        # Frame 0 and frame T should give identical targets
        tgt0 = src.site_targets(torch.zeros(_N)).clone()
        t_wrap = torch.full((_N,), _T * _CTRL_DT)
        src.update(t_wrap)
        tgt_wrap = src.site_targets(t_wrap)
        assert torch.allclose(tgt0, tgt_wrap, atol=1e-6)

    def test_phase_advances(self) -> None:
        """Phase must increase as time progresses."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t0 = torch.zeros(_N)
        src.update(t0)
        src._start_offsets = torch.zeros(_N, dtype=torch.long)

        ph0 = src.phase(torch.zeros(_N))
        ph1 = src.phase(torch.full((_N,), 5 * _CTRL_DT))
        assert (ph1 > ph0).all()

    def test_different_start_offsets_give_different_targets(self) -> None:
        """Two envs at the same time but different offsets must have different targets."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t = torch.zeros(_N)
        src.update(t)
        # Give each env a different offset
        src._start_offsets = torch.arange(_N, dtype=torch.long)

        tgt = src.site_targets(t)
        # Row 0 and row 1 are different frames → different positions
        assert not torch.allclose(tgt[0], tgt[1])


# ---------------------------------------------------------------------------
# TestClipTrajectorySourceReset
# ---------------------------------------------------------------------------


class TestClipTrajectorySourceReset:
    def test_reset_resamples_offsets_for_regressed_envs(self) -> None:
        """Envs whose time regresses must get new start offsets."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t_forward = torch.full((_N,), 5 * _CTRL_DT)
        src.update(t_forward)
        offsets_before = src._start_offsets.clone()

        # Env 0 and env 2 reset (time goes back to 0)
        t_mixed = t_forward.clone()
        t_mixed[0] = 0.0
        t_mixed[2] = 0.0
        src.update(t_mixed)
        offsets_after = src._start_offsets

        # Non-resetting envs (1, 3) keep their offsets
        assert offsets_after[1] == offsets_before[1]
        assert offsets_after[3] == offsets_before[3]

    def test_non_reset_envs_keep_offsets(self) -> None:
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t = torch.full((_N,), 3 * _CTRL_DT)
        src.update(t)
        offsets_before = src._start_offsets.clone()

        # All envs continue forward
        t2 = torch.full((_N,), 6 * _CTRL_DT)
        src.update(t2)
        assert torch.all(src._start_offsets == offsets_before)

    def test_full_batch_reset(self) -> None:
        """All envs reset simultaneously — all offsets should be resampled."""
        torch.manual_seed(99)
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t_fwd = torch.full((_N,), 10 * _CTRL_DT)
        src.update(t_fwd)
        offsets_before = src._start_offsets.clone()

        torch.manual_seed(42)
        t_reset = torch.zeros(_N)
        src.update(t_reset)
        offsets_after = src._start_offsets

        # With 50 possible values, chance of all 4 matching is (1/50)^4 ≈ 0
        # (might theoretically fail but probability is negligible)
        assert not torch.all(offsets_after == offsets_before)


# ---------------------------------------------------------------------------
# TestMimicMjlabCacheDispatch
# ---------------------------------------------------------------------------


class TestMimicMjlabCacheDispatch:
    def test_sync_with_clip_source_updates_target(self) -> None:
        """_sync_mimic_mjlab_targets must call clip_source when present."""
        src = _make_source()
        cache = _make_cache_with_source(src)
        env = _make_mock_env(t=0.0)

        _sync_mimic_mjlab_targets(env, "robot", cache)

        assert cache["target_torch"] is not None
        assert cache["target_torch"].shape == (_N, _N_SITES, 3)

    def test_sync_with_clip_source_changes_on_next_step(self) -> None:
        """Targets from clip must differ between consecutive time steps."""
        src = _make_source()
        cache = _make_cache_with_source(src)

        env0 = _make_mock_env(t=0.0)
        _sync_mimic_mjlab_targets(env0, "robot", cache)
        tgt0 = cache["target_torch"].clone()

        # Force consistent offsets before second step
        assert src._start_offsets is not None
        src._start_offsets = torch.zeros(_N, dtype=torch.long)

        env1 = _make_mock_env(t=_CTRL_DT)
        _sync_mimic_mjlab_targets(env1, "robot", cache)
        tgt1 = cache["target_torch"]

        assert not torch.allclose(tgt0, tgt1)

    def test_sync_random_fallback_still_works(self) -> None:
        """Random path must be unaffected when no clip_source is in the cache."""
        cache = _make_cache_random()
        env = _make_mock_env(t=0.0)

        _sync_mimic_mjlab_targets(env, "robot", cache)

        tgt = cache["target_torch"]
        assert tgt is not None
        assert tgt.shape == (_N, _N_SITES, 3)
        # Values must be within the box
        lo = torch.as_tensor(cache["lo"], dtype=torch.float32)
        hi = torch.as_tensor(cache["hi"], dtype=torch.float32)
        assert (tgt >= lo).all()
        assert (tgt <= hi).all()

    def test_sync_random_resamples_on_episode_reset(self) -> None:
        """Random path must resample targets when time regresses."""
        cache = _make_cache_random()

        env_fwd = _make_mock_env(t=5 * _CTRL_DT)
        _sync_mimic_mjlab_targets(env_fwd, "robot", cache)
        tgt_before = cache["target_torch"].clone()

        # Simulate episode reset (time goes back)
        env_reset = _make_mock_env(t=0.0)
        _sync_mimic_mjlab_targets(env_reset, "robot", cache)
        tgt_after = cache["target_torch"]

        # Resampled targets will almost certainly differ
        assert not torch.allclose(tgt_before, tgt_after)


# ---------------------------------------------------------------------------
# TestMimicMjlabClosures
# ---------------------------------------------------------------------------


class TestMimicMjlabClosures:
    """Tests for obs / reward closure factories using injected cache."""

    def _inject_cache(
        self,
        env: Any,
        entity_name: str,
        variant: str,
        cache: dict[str, Any],
    ) -> None:
        key = (id(env), entity_name, variant)
        _mimic_mjlab_cache[key] = cache
        # Ensure targets are populated
        _sync_mimic_mjlab_targets(env, entity_name, cache)

    def _setup(self, with_clip: bool = True) -> tuple[Any, dict[str, Any], str, str]:
        entity = "robot"
        variant = "bimanual"
        src = _make_source() if with_clip else None
        cache = _make_cache_with_source(src) if with_clip else _make_cache_random()
        env = _make_mock_env(t=0.0, entity_name=entity)
        self._inject_cache(env, entity, variant, cache)
        return env, cache, entity, variant

    # --- Core obs terms ---

    def test_obs_qpos_shape(self) -> None:
        env, _, entity, _ = self._setup()
        fn = _mimic_obs_qpos(entity)
        out = fn(env)
        assert out.shape == (_N, _NQ)

    def test_obs_qvel_shape(self) -> None:
        env, _, entity, _ = self._setup()
        fn = _mimic_obs_qvel(entity)
        out = fn(env)
        assert out.shape == (_N, _NV)

    def test_obs_act_shape(self) -> None:
        env, _, entity, _ = self._setup()
        fn = _mimic_obs_act(entity)
        out = fn(env)
        assert out.shape == (_N, _NA)

    def test_obs_site_pos_shape(self) -> None:
        env, _, entity, variant = self._setup()
        fn = _mimic_obs_site_pos(entity, variant)
        out = fn(env)
        assert out.shape == (_N, _N_SITES * 3)

    def test_obs_target_shape(self) -> None:
        env, _, entity, variant = self._setup()
        fn = _mimic_obs_target(entity, variant)
        out = fn(env)
        assert out.shape == (_N, _N_SITES * 3)

    def test_obs_err_shape(self) -> None:
        env, _, entity, variant = self._setup()
        fn = _mimic_obs_err(entity, variant)
        out = fn(env)
        assert out.shape == (_N, _N_SITES * 3)

    def test_tracking_reward_shape_and_range(self) -> None:
        env, _, entity, variant = self._setup()
        fn = _mimic_tracking_reward(entity, variant)
        rwd = fn(env)
        assert rwd.shape == (_N,)
        assert (rwd >= 0.0).all()
        assert (rwd <= 1.0).all()

    # --- Trajectory-mode obs terms ---

    def test_obs_clip_ref_qpos_shape(self) -> None:
        src = _make_source(nq=_NQ)
        cache = _make_cache_with_source(src)
        env = _make_mock_env(t=0.0)
        entity, variant = "robot", "bimanual"
        self._inject_cache(env, entity, variant, cache)

        clip = src.clip
        fn = _mimic_obs_clip_ref_qpos(entity, variant, clip, _CTRL_DT)
        out = fn(env)
        assert out is not None
        assert out.shape == (_N, _NQ)

    def test_obs_clip_ref_qvel_shape(self) -> None:
        src = _make_source(nv=_NV)
        cache = _make_cache_with_source(src)
        env = _make_mock_env(t=0.0)
        entity, variant = "robot", "bimanual"
        self._inject_cache(env, entity, variant, cache)

        fn = _mimic_obs_clip_ref_qvel(entity, variant, src.clip, _CTRL_DT)
        out = fn(env)
        assert out is not None
        assert out.shape == (_N, _NV)

    def test_obs_clip_phase_shape_and_range(self) -> None:
        src = _make_source()
        cache = _make_cache_with_source(src)
        env = _make_mock_env(t=0.0)
        entity, variant = "robot", "bimanual"
        self._inject_cache(env, entity, variant, cache)

        fn = _mimic_obs_clip_phase(entity, variant, src.clip, _CTRL_DT)
        out = fn(env)
        assert out.shape == (_N, 1)
        assert (out >= 0.0).all()
        assert (out < 1.0).all()

    def test_obs_target_comes_from_clip(self) -> None:
        """In trajectory mode the target obs must match clip.site_xpos at frame 0."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        cache = _make_cache_with_source(src)
        env = _make_mock_env(t=0.0)
        entity, variant = "robot", "bimanual"
        self._inject_cache(env, entity, variant, cache)

        # Force all envs to start at frame 0
        assert src._start_offsets is not None
        src._start_offsets = torch.zeros(_N, dtype=torch.long)
        _sync_mimic_mjlab_targets(env, entity, cache)

        fn = _mimic_obs_target(entity, variant)
        tgt = fn(env).reshape(_N, _N_SITES, 3)

        expected = torch.as_tensor(src.clip.site_xpos[0], dtype=torch.float32)
        assert torch.allclose(tgt[0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# TestInitialPoseHelpers
# ---------------------------------------------------------------------------


class TestInitialPoseHelpers:
    def test_initial_qpos_shape(self) -> None:
        src = _make_source(nq=_NQ)
        t = torch.zeros(_N)
        src.update(t)
        q = src.initial_qpos()
        assert q is not None
        assert q.shape == (_N, _NQ)

    def test_initial_qpos_none_when_missing(self) -> None:
        clip = _make_clip(with_qpos=False, with_qvel=False)
        src = ClipTrajectorySource(
            clip=clip, tracked_site_ids=_make_site_ids(), ctrl_dt=_CTRL_DT
        )
        src.update(torch.zeros(_N))
        assert src.initial_qpos() is None
        assert src.initial_qvel() is None

    def test_initial_qpos_matches_start_offset(self) -> None:
        src = _make_source(T=_T, nq=_NQ)
        t = torch.zeros(_N)
        src.update(t)
        # Pin offset for env 0 to frame 5
        assert src._start_offsets is not None
        src._start_offsets[0] = 5
        q = src.initial_qpos()
        assert q is not None
        expected = torch.as_tensor(src.clip.qpos[5], dtype=torch.float32)
        assert torch.allclose(q[0], expected, atol=1e-6)

    def test_make_init_state_fn_returns_callable(self) -> None:
        src = _make_source()
        fn = src.make_init_state_fn()
        qpos, qvel = fn(n_envs=8, device=torch.device("cpu"))
        assert qpos is not None
        assert qpos.shape == (8, _NQ)
        assert qvel is not None
        assert qvel.shape == (8, _NV)

    def test_make_init_state_fn_frames_in_clip_range(self) -> None:
        src = _make_source(T=_T)
        fn = src.make_init_state_fn()
        qpos, _ = fn(n_envs=32, device=torch.device("cpu"))
        assert qpos is not None
        clip_qpos = torch.as_tensor(src.clip.qpos, dtype=torch.float32)
        # Each row of qpos must be a row in clip_qpos
        for i in range(32):
            matches = torch.all(
                torch.isclose(qpos[i].unsqueeze(0), clip_qpos, atol=1e-6), dim=1
            )
            assert matches.any(), f"Row {i} of qpos does not match any clip frame"

    def test_make_init_state_fn_no_qpos(self) -> None:
        clip = _make_clip(with_qpos=False, with_qvel=False)
        src = ClipTrajectorySource(
            clip=clip, tracked_site_ids=_make_site_ids(), ctrl_dt=_CTRL_DT
        )
        fn = src.make_init_state_fn()
        qpos, qvel = fn(n_envs=4, device=torch.device("cpu"))
        assert qpos is None
        assert qvel is None


# ---------------------------------------------------------------------------
# TestMjlabSimTimeToTensor
# ---------------------------------------------------------------------------


class TestMjlabSimTimeToTensor:
    def test_from_tensor(self) -> None:
        t = torch.tensor([0.1, 0.2, 0.3])
        out = _mjlab_sim_time_to_tensor(t)
        assert out.shape == (3,)
        assert out.dtype == torch.float32
        assert torch.allclose(out, t.float())

    def test_from_scalar_float(self) -> None:
        out = _mjlab_sim_time_to_tensor(0.05)
        assert out.shape == (1,)
        assert float(out[0]) == pytest.approx(0.05)

    def test_from_numpy_scalar(self) -> None:
        out = _mjlab_sim_time_to_tensor(np.float32(0.1))
        assert out.shape == (1,)
        assert float(out[0]) == pytest.approx(0.1, abs=1e-6)
