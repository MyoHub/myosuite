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
    Verifies that targets advance frame-by-frame as the step counter advances.

``TestClipTrajectorySourceReset``
    Verifies that start offsets are resampled independently when individual
    environments reset (their step counter regresses).

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
    _mimic_cache_key,
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
    _mimic_rsi_event,
    _mimic_tracking_reward,
    _normalize_mimic_reward_mode,
    _sync_mimic_mjlab_targets,
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


def _steps(k: int) -> torch.Tensor:
    """Per-env control-step counter (mjlab ``episode_length_buf``) at step *k*."""
    return torch.full((_N,), k, dtype=torch.long)


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

    The Mimic state observations read the entity API (``entity.data.joint_pos``,
    ``joint_vel``; the entity is fixed-base here); the other terms read the raw
    physics arrays, which mjlab nests one level deeper::

        env.scene[entity_name].data.data   # → MjData-like object

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
    # mjlab's integer step counter, which indexes the clip frames.
    episode_length_buf = torch.round(time_tensor / _CTRL_DT).long()

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
    data_holder = types.SimpleNamespace(
        data=physics, joint_pos=physics.qpos, joint_vel=physics.qvel
    )
    entity = types.SimpleNamespace(data=data_holder, is_fixed_base=True)
    scene = _MockScene(
        {entity_name: entity},
        env_origins=torch.zeros(n_envs, 3, dtype=torch.float32),
    )
    return types.SimpleNamespace(
        scene=scene,
        sim=_MockSim(),
        physics_dt=0.002,
        cfg=types.SimpleNamespace(decimation=5),
        episode_length_buf=episode_length_buf,
    )


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
    assert init.joint_pos == {"^hinge0$": 0.1, "^slide1$": -0.2}
    assert init.joint_vel == {"^hinge0$": 0.3, "^slide1$": -0.4}


def test_init_state_from_model_world_frame_root_angular_velocity(monkeypatch) -> None:
    """The free joint's body-frame angular velocity is rotated to the world frame."""
    from myosuite.tests.support.optional_deps import require_mjlab

    require_mjlab()
    import mujoco

    monkeypatch.setattr(mujoco, "mj_id2name", lambda _model, _obj, _idx: "root")
    half = np.sqrt(0.5)  # +90 deg about z: body x-axis points along world y
    model = types.SimpleNamespace(
        nkey=1,
        njnt=1,
        key_qpos=np.array([[0.0, 0.0, 1.0, half, 0.0, 0.0, half]]),
        key_qvel=np.array([[0.1, 0.2, 0.3, 1.0, 0.0, 0.0]]),
        jnt_type=np.array([int(mujoco.mjtJoint.mjJNT_FREE)], dtype=np.int32),
        jnt_qposadr=np.array([0], dtype=np.int32),
        jnt_dofadr=np.array([0], dtype=np.int32),
    )

    init = _init_state_from_model(model)

    assert init.lin_vel == (0.1, 0.2, 0.3)
    np.testing.assert_allclose(init.ang_vel, (0.0, 1.0, 0.0), atol=1e-12)


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
        last_step=None,
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
        last_step=None,
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
        t = torch.zeros(_N, dtype=torch.long)
        src.update(t)
        assert src._site_tensor is not None
        assert src._start_offsets is not None
        assert src._last_step is not None

    def test_site_tensor_shape(self) -> None:
        src = _make_source(T=_T, n_sites=_N_SITES)
        t = torch.zeros(_N, dtype=torch.long)
        src.update(t)
        assert src._site_tensor is not None
        assert src._site_tensor.shape == (_T, _N_SITES, 3)

    def test_qpos_tensor_shape(self) -> None:
        src = _make_source(T=_T, nq=_NQ)
        t = torch.zeros(_N, dtype=torch.long)
        src.update(t)
        assert src._qpos_tensor is not None
        assert src._qpos_tensor.shape == (_T, _NQ)

    def test_no_qpos_tensor_when_missing(self) -> None:
        clip = _make_clip(with_qpos=False, with_qvel=False)
        src = ClipTrajectorySource(
            clip=clip, tracked_site_ids=_make_site_ids(), ctrl_dt=_CTRL_DT
        )
        t = torch.zeros(_N, dtype=torch.long)
        src.update(t)
        assert src._qpos_tensor is None
        assert src.ref_qpos(t) is None
        assert src.ref_qvel(t) is None

    def test_start_offsets_in_range(self) -> None:
        src = _make_source(T=_T)
        t = torch.zeros(_N, dtype=torch.long)
        src.update(t)
        assert src._start_offsets is not None
        assert (src._start_offsets >= 0).all()
        assert (src._start_offsets < _T).all()

    def test_site_targets_shape(self) -> None:
        src = _make_source()
        t = torch.zeros(_N, dtype=torch.long)
        src.update(t)
        out = src.site_targets(t)
        assert out.shape == (_N, _N_SITES, 3)

    def test_ref_qpos_shape(self) -> None:
        src = _make_source(nq=_NQ)
        t = torch.zeros(_N, dtype=torch.long)
        src.update(t)
        out = src.ref_qpos(t)
        assert out is not None
        assert out.shape == (_N, _NQ)

    def test_phase_shape_and_range(self) -> None:
        src = _make_source()
        t = torch.zeros(_N, dtype=torch.long)
        src.update(t)
        ph = src.phase(t)
        assert ph.shape == (_N, 1)
        assert (ph >= 0.0).all()
        assert (ph < 1.0).all()

    def test_clip_end_flags_episodes_past_the_last_frame(self) -> None:
        src = _make_source(T=10)
        src.update(_steps(0))
        src._start_offsets = torch.tensor([0, 5, 8, 9][:_N], dtype=torch.long)
        assert src.clip_end(_steps(1)).tolist() == [False, False, False, True][:_N]
        assert src.clip_end(_steps(5)).tolist() == [False, True, True, True][:_N]
        # The truncating step reads the last frame (not wrapped frame 0).
        assert src.frame_indices(_steps(1)).tolist() == [1, 6, 9, 9][:_N]

    def test_multi_clip_end_uses_each_envs_clip_length(self) -> None:
        src = _make_multi_clip_source()
        src.update(torch.zeros(_N, dtype=torch.long))
        src._clip_indices = torch.tensor([0, 1, 0, 1][:_N], dtype=torch.long)
        src._start_offsets = torch.tensor([3, 3, 4, 6][:_N], dtype=torch.long)
        # lengths 5 / 7 / 5 / 7
        assert src.clip_end(_steps(2)).tolist() == [True, False, True, True][:_N]
        # Past its end each env holds its own clip's last frame.
        assert src.frame_indices(_steps(2)).tolist() == [4, 5, 4, 6][:_N]

    def test_multi_clip_source_uses_per_env_clip_assignments(self) -> None:
        src = _make_multi_clip_source()
        t = torch.tensor([0, 2, 0, 1], dtype=torch.long)
        src.update(t)
        src._clip_indices = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        src._start_offsets = torch.tensor([1, 2, 3, 4], dtype=torch.long)

        frame_idx = src.frame_indices(t)
        clip_lengths = src.clip_lengths(t)
        targets = src.site_targets(t)
        qpos = src.ref_qpos(t)
        qvel = src.ref_qvel(t)

        torch.testing.assert_close(frame_idx, torch.tensor([1, 4, 3, 5]))
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
            np.asarray(src.clips[0].qpos[3], dtype=np.float32),
        )
        np.testing.assert_allclose(
            qvel[3].cpu().numpy(),
            np.asarray(src.clips[1].qvel[5], dtype=np.float32),
        )

    def test_multi_clip_gather_is_one_sync_free_read_of_each_envs_clip(self) -> None:
        """Bank reads (all envs, or the envs of a partial reset) make no host sync.

        The gather looped over the clips with ``mask.any()`` and bool-mask
        indexing: three host syncs per clip and read.
        """
        from myosuite.tests.support.host_sync import HostSyncCounter

        src = _make_multi_clip_source()
        src.update(torch.zeros(_N, dtype=torch.long))
        src._clip_indices = torch.tensor([1, 0, 1, 0][:_N], dtype=torch.long)
        frames = torch.tensor([6, 4, 0, 2][:_N], dtype=torch.long)  # last frames too
        env_ids = torch.tensor([2, 0], dtype=torch.long)
        with HostSyncCounter(package_only=True) as syncs:
            sites = src.site_targets_at_frames(frames)
            qpos = src.ref_qpos_at_frames(frames)
            qvel = src.ref_qvel_at_frames(frames)
            reset_qpos = src.ref_qpos_at_frames(frames[env_ids], env_ids)
        assert syncs.total == 0, syncs.report()
        assert qpos is not None and qvel is not None and reset_qpos is not None
        rows = list(zip(src._clip_indices.tolist(), frames.tolist()))
        for name, got in (("site_xpos", sites), ("qpos", qpos), ("qvel", qvel)):
            expected = np.stack([getattr(src.clips[c], name)[f] for c, f in rows])
            np.testing.assert_array_equal(got.numpy(), expected.astype(np.float32))
        assert torch.equal(reset_qpos, qpos[env_ids])

    def test_multi_clip_reads_never_cross_into_a_neighbouring_clip(self) -> None:
        """Past a clip's end, and for wrapped lookahead frames, the bank stays in-clip.

        The clips share one concatenated tensor, so the row of an out-of-range
        frame would belong to the next clip. Sweep every clip and start offset
        beyond the end and compare with the per-clip arrays.
        """
        src = _make_multi_clip_source()
        src.update(torch.zeros(_N, dtype=torch.long))
        lengths = [len(c.site_xpos) for c in src.clips]
        for clip_idx in (0, 1):
            for offset in range(lengths[clip_idx]):
                src._clip_indices = torch.full((_N,), clip_idx, dtype=torch.long)
                src._start_offsets = torch.full((_N,), offset, dtype=torch.long)
                clip = src.clips[clip_idx]
                for k in range(2 * max(lengths)):
                    step = _steps(k)
                    frame = min(offset + k, lengths[clip_idx] - 1)
                    assert src.frame_indices(step).tolist() == [frame] * _N
                    assert (
                        src.clip_end(step).tolist()
                        == [offset + k >= lengths[clip_idx]] * _N
                    )
                    for name, got in (
                        ("site_xpos", src.site_targets(step)),
                        ("qpos", src.ref_qpos(step)),
                        ("qvel", src.ref_qvel(step)),
                    ):
                        expected = np.asarray(getattr(clip, name)[frame], np.float32)
                        np.testing.assert_array_equal(got[0].numpy(), expected)
                    # Lookahead wraps inside the clip (modulo its own length).
                    ahead = (src.frame_indices(step) + 3) % src.clip_lengths(step)
                    np.testing.assert_array_equal(
                        src.site_targets_at_frames(ahead)[0].numpy(),
                        np.asarray(clip.site_xpos[int(ahead[0])], np.float32),
                    )

    @pytest.mark.parametrize("bad_frame", [5, -1])
    def test_multi_clip_gather_rejects_a_frame_outside_its_clip(
        self, bad_frame: int
    ) -> None:
        """A frame past its env's clip (or negative) would read a neighbouring
        clip's rows in the concatenated bank; the gather refuses it."""
        src = _make_multi_clip_source()
        src.update(torch.zeros(_N, dtype=torch.long))
        src._clip_indices = torch.tensor([0, 1, 1, 1][:_N], dtype=torch.long)
        # Env 0 plays clip 0 (5 frames): frame 5 would be clip 1's first row.
        frames = torch.tensor([bad_frame, 0, 0, 0][:_N], dtype=torch.long)
        with pytest.raises(RuntimeError, match="outside its env's clip"):
            src.site_targets_at_frames(frames)
        with pytest.raises(RuntimeError, match="outside its env's clip"):
            src.ref_qpos_at_frames(frames[:1], torch.tensor([0]))


# ---------------------------------------------------------------------------
# TestClipTrajectorySourceAdvance
# ---------------------------------------------------------------------------


class TestClipTrajectorySourceAdvance:
    def test_targets_change_as_step_advances(self) -> None:
        """Targets must change when the step counter advances by one."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        # Pin start_offsets to 0 for determinism
        t0 = _steps(0)
        src.update(t0)
        src._start_offsets = torch.zeros(_N, dtype=torch.long)

        tgt0 = src.site_targets(t0).clone()

        t1 = _steps(1)  # one frame later
        src.update(t1)
        tgt1 = src.site_targets(t1)

        assert not torch.allclose(
            tgt0, tgt1
        ), "Targets must differ after advancing one frame"

    def test_targets_match_clip_data(self) -> None:
        """Targets at frame k must equal clip.site_xpos[k]."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t0 = _steps(0)
        src.update(t0)
        # Force all envs to start at frame 0
        src._start_offsets = torch.zeros(_N, dtype=torch.long)

        for frame_k in [0, 1, 5, _T - 1]:
            t_k = _steps(frame_k)
            expected = (
                torch.as_tensor(src.clip.site_xpos[frame_k], dtype=torch.float32)
                .unsqueeze(0)
                .expand(_N, -1, -1)
            )
            actual = src.site_targets(t_k)
            assert torch.allclose(
                actual, expected, atol=1e-6
            ), f"Frame {frame_k}: targets don't match clip"

    def test_clip_end_step_scores_the_last_frame(self) -> None:
        """The step that truncates at the clip end reads frame T-1, not wrapped frame 0."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        src.update(_steps(0))
        src._start_offsets = torch.zeros(_N, dtype=torch.long)

        t_end = _steps(_T)
        src.update(t_end)
        assert src.clip_end(t_end).all()
        assert src.frame_indices(t_end).tolist() == [_T - 1] * _N
        last = torch.as_tensor(src.clip.site_xpos[_T - 1], dtype=torch.float32)
        assert torch.allclose(src.site_targets(t_end), last.expand(_N, -1, -1))
        ref_qpos = src.ref_qpos(t_end)
        assert ref_qpos is not None
        last_qpos = torch.as_tensor(src.clip.qpos[_T - 1], dtype=torch.float32)
        assert torch.allclose(ref_qpos, last_qpos.expand(_N, -1))
        assert torch.allclose(src.phase(t_end), torch.full((_N, 1), (_T - 1) / _T))

    def test_phase_advances(self) -> None:
        """Phase must increase as the step counter advances."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t0 = _steps(0)
        src.update(t0)
        src._start_offsets = torch.zeros(_N, dtype=torch.long)

        ph0 = src.phase(_steps(0))
        ph1 = src.phase(_steps(5))
        assert (ph1 > ph0).all()

    def test_float_time_is_rejected(self) -> None:
        """Float sim time drifts, so it must not be accepted as a frame counter."""
        src = _make_source()
        with pytest.raises(TypeError, match="episode_length_buf"):
            src.update(torch.zeros(_N))
        src.update(_steps(0))
        with pytest.raises(TypeError, match="episode_length_buf"):
            src.site_targets(torch.full((_N,), 0.07))

    def test_frames_follow_step_counter_not_float32_time(self) -> None:
        """Frame k after k control steps, unlike floor(float32 time / ctrl_dt).

        mujoco_warp accumulates ``time += timestep`` in float32 every physics
        substep; that clock reads 0.06999 s after 7 steps of 5 x 2 ms, so the
        old ``floor(time / ctrl_dt)`` index fell one frame behind.
        """
        src = _make_source(T=1001, ctrl_dt=_CTRL_DT)
        src.update(_steps(0))
        src._start_offsets = torch.zeros(_N, dtype=torch.long)
        time = np.float32(0.0)
        lagging = 0
        for k in range(1, 1001):
            for _ in range(5):
                time = np.float32(time + np.float32(0.002))
            lagging += int(np.floor(time / _CTRL_DT)) != k
            src.update(_steps(k))
            assert int(src.frame_indices(_steps(k))[0]) == k
        assert lagging > 600  # the float32 clock is behind on most steps

    def test_different_start_offsets_give_different_targets(self) -> None:
        """Two envs at the same time but different offsets must have different targets."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t = torch.zeros(_N, dtype=torch.long)
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
        """Envs whose step counter regresses must get new start offsets."""
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t_forward = _steps(5)
        src.update(t_forward)
        offsets_before = src._start_offsets.clone()

        # Env 0 and env 2 reset (step counter goes back to 0)
        t_mixed = t_forward.clone()
        t_mixed[0] = 0
        t_mixed[2] = 0
        src.update(t_mixed)
        offsets_after = src._start_offsets

        # Non-resetting envs (1, 3) keep their offsets
        assert offsets_after[1] == offsets_before[1]
        assert offsets_after[3] == offsets_before[3]

    def test_non_reset_envs_keep_offsets(self) -> None:
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t = _steps(3)
        src.update(t)
        offsets_before = src._start_offsets.clone()

        # All envs continue forward
        t2 = _steps(6)
        src.update(t2)
        assert torch.all(src._start_offsets == offsets_before)

    def test_full_batch_reset(self) -> None:
        """All envs reset simultaneously — all offsets should be resampled."""
        torch.manual_seed(99)
        src = _make_source(T=_T, ctrl_dt=_CTRL_DT)
        t_fwd = _steps(10)
        src.update(t_fwd)
        offsets_before = src._start_offsets.clone()

        torch.manual_seed(42)
        t_reset = _steps(0)
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

    def test_sync_random_resamples_only_the_reset_envs(self) -> None:
        """A reset of env 2 must not resample the other envs' targets."""
        cache = _make_cache_random()
        _sync_mimic_mjlab_targets(_make_mock_env(t=5 * _CTRL_DT), "robot", cache)
        before = cache["target_torch"].clone()

        steps = torch.full((_N,), 6.0) * _CTRL_DT
        steps[2] = 0.0  # only env 2 restarts its episode
        _sync_mimic_mjlab_targets(_make_mock_env(t=steps), "robot", cache)
        after = cache["target_torch"]

        keep = torch.arange(_N) != 2
        assert torch.equal(after[keep], before[keep])
        assert not torch.allclose(after[2], before[2])


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
        key = _mimic_cache_key(env, entity_name, variant)
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
# TestMimicTargetSync: one sync per step phase, never a stale target
# ---------------------------------------------------------------------------


class TestMimicTargetSync:
    """The closures of one step share one target sync.

    They used to re-sync on every call (about 30 per env step, each a host sync
    on a GPU); the sync now runs when its inputs change, and the reset check
    only when the step counter did more than mjlab's per-step increment.
    """

    _ENTITY, _VARIANT = "robot", "bimanual"

    def _setup(self, cache: dict[str, Any]) -> Any:
        env = _make_mock_env(t=5 * _CTRL_DT, entity_name=self._ENTITY)
        env.common_step_counter = 5
        _mimic_mjlab_cache[_mimic_cache_key(env, self._ENTITY, self._VARIANT)] = cache
        _sync_mimic_mjlab_targets(env, self._ENTITY, cache)
        return env

    def _closures(self, src: ClipTrajectorySource) -> list[Any]:
        args = (self._ENTITY, self._VARIANT)
        return [
            _mimic_obs_target(*args),
            _mimic_obs_err(*args),
            _mimic_obs_site_pos(*args),
            _mimic_tracking_reward(*args),
            _mimic_obs_clip_phase(*args, src.clip, _CTRL_DT),
            _mimic_obs_clip_ref_qpos(*args, src.clip, _CTRL_DT),
            _mimic_obs_clip_ref_qvel(*args, src.clip, _CTRL_DT),
        ]

    @staticmethod
    def _step(env: Any) -> None:
        """mjlab's per-step counter update."""
        env.episode_length_buf += 1
        env.common_step_counter += 1

    def _assert_current(self, env: Any, src: ClipTrajectorySource) -> None:
        """Every closure agrees with a direct evaluation of the clip source."""
        step = env.episode_length_buf
        target, err, _, _, phase, ref_qpos, ref_qvel = (
            fn(env) for fn in self._closures(src)
        )
        expected = src.site_targets(step).reshape(_N, -1)
        assert torch.equal(target, expected)
        assert torch.equal(err, expected)  # mock sites sit at the origin
        assert torch.equal(phase, src.phase(step))
        assert torch.equal(ref_qpos, src.ref_qpos(step))
        assert torch.equal(ref_qvel, src.ref_qvel(step))

    def test_closures_of_a_step_share_one_update(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _make_source()
        env = self._setup(_make_cache_with_source(src))
        calls: list[bool] = []
        real = src.update

        def counting(step: torch.Tensor, **kwargs: Any) -> None:
            calls.append(kwargs.get("check_resets", True))
            real(step, **kwargs)

        monkeypatch.setattr(src, "update", counting)
        for fn in self._closures(src):
            fn(env)
        assert calls == []  # synced in _setup, nothing changed since
        self._step(env)
        for _ in range(3):
            for fn in self._closures(src):
                fn(env)
        # One update, without the reset check: a plain step cannot start an episode.
        assert calls == [False]
        self._assert_current(env, src)

    def test_targets_follow_every_input_change(self) -> None:
        src = _make_source()
        env = self._setup(_make_cache_with_source(src))
        self._assert_current(env, src)
        self._step(env)
        self._assert_current(env, src)
        env.episode_length_buf = env.episode_length_buf + 3  # rsl_rl-style rebind
        self._assert_current(env, src)
        src._start_offsets[1] = 7  # in place, as the RSI event does
        self._assert_current(env, src)
        src._start_offsets = torch.zeros(_N, dtype=torch.long)
        self._assert_current(env, src)
        before = src._start_offsets.clone()
        torch.manual_seed(0)
        env.episode_length_buf[2] = 0  # env 2 starts a new episode
        self._assert_current(env, src)
        assert torch.equal(src._start_offsets[[0, 1, 3]], before[[0, 1, 3]])
        # Inference tensors keep no version counter: every call re-syncs.
        with torch.inference_mode():
            env.episode_length_buf = torch.full((_N,), 9, dtype=torch.long)
        self._assert_current(env, src)
        with torch.inference_mode():
            env.episode_length_buf[0] = 4
        self._assert_current(env, src)

    def test_restart_after_a_step_is_detected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A step and a reset between two calls still run the reset check."""
        src = _make_source()
        env = self._setup(_make_cache_with_source(src))
        checks: list[torch.Tensor] = []
        real = src._detect_and_resample_resets

        def counting(step: torch.Tensor) -> None:
            checks.append(step.clone())
            real(step)

        monkeypatch.setattr(src, "_detect_and_resample_resets", counting)
        self._step(env)
        env.episode_length_buf[1] = 0  # the env reset after the step
        _mimic_obs_target(self._ENTITY, self._VARIANT)(env)
        assert len(checks) == 1 and checks[0].tolist() == [6, 0, 6, 6]
        self._assert_current(env, src)

    def test_random_targets_resample_only_on_new_episodes(self) -> None:
        cache = _make_cache_random()
        env = self._setup(cache)
        target_fn = _mimic_obs_target(self._ENTITY, self._VARIANT)
        before = target_fn(env).clone()
        self._step(env)
        assert torch.equal(target_fn(env), before)
        assert torch.equal(cache["last_step"], env.episode_length_buf)
        env.episode_length_buf[3] = 0  # env 3 starts a new episode
        after = target_fn(env)
        assert torch.equal(after[:3], before[:3])
        assert not torch.equal(after[3], before[3])


# ---------------------------------------------------------------------------
# TestInitialPoseHelpers
# ---------------------------------------------------------------------------


class TestInitialPoseHelpers:
    def test_initial_qpos_shape(self) -> None:
        src = _make_source(nq=_NQ)
        t = torch.zeros(_N, dtype=torch.long)
        src.update(t)
        q = src.initial_qpos()
        assert q is not None
        assert q.shape == (_N, _NQ)

    def test_initial_qpos_none_when_missing(self) -> None:
        clip = _make_clip(with_qpos=False, with_qvel=False)
        src = ClipTrajectorySource(
            clip=clip, tracked_site_ids=_make_site_ids(), ctrl_dt=_CTRL_DT
        )
        src.update(_steps(0))
        assert src.initial_qpos() is None
        assert src.initial_qvel() is None

    def test_initial_qpos_matches_start_offset(self) -> None:
        src = _make_source(T=_T, nq=_NQ)
        t = torch.zeros(_N, dtype=torch.long)
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
# Env-wrapper / env_ids regressions
# ---------------------------------------------------------------------------


class _EnvProxy:
    """Minimal attribute-forwarding env wrapper."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_mimic_cache_key_stable_across_env_wrappers() -> None:
    """A wrapped env must resolve to the same cache entry as the raw env.

    Keying on ``id(env)`` gave each wrapper its own ClipTrajectorySource, and
    hence a different random start frame.
    """
    env = _make_mock_env()
    key = _mimic_cache_key(env, "robot", "bimanual")
    assert _mimic_cache_key(_EnvProxy(env), "robot", "bimanual") == key
    assert _mimic_cache_key(_EnvProxy(_EnvProxy(env)), "robot", "bimanual") == key


def test_rsi_event_accepts_env_ids_none() -> None:
    """``env_ids=None`` means "every env" in mjlab's event contract."""
    from myosuite.tests.support.optional_deps import require_mjlab

    require_mjlab()  # RSI rotates the root velocity with mjlab's quat_apply
    entity, variant = "robot", "bimanual"
    env = _make_mock_env(t=0.0, entity_name=entity)
    _mimic_mjlab_cache[_mimic_cache_key(env, entity, variant)] = (
        _make_cache_with_source(_make_source())
    )
    seen: dict[str, torch.Tensor] = {}
    ent = env.scene[entity]
    ent.write_root_state_to_sim = lambda state, env_ids: seen.update(root=env_ids)
    ent.write_joint_state_to_sim = lambda pos, vel, env_ids: seen.update(joint=env_ids)

    _mimic_rsi_event(entity, variant, _make_clip(), _CTRL_DT)(env, None)

    assert seen["root"].tolist() == list(range(_N))
    assert seen["joint"].tolist() == list(range(_N))


def test_check_clip_rate_warns_only_on_mismatch() -> None:
    import warnings

    from myosuite.core.trajectory_io import check_clip_rate

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_clip_rate(None, _CTRL_DT)
        check_clip_rate(1.0 / _CTRL_DT, _CTRL_DT)
    with pytest.warns(UserWarning, match="control rate"):
        check_clip_rate(30.0, _CTRL_DT)
