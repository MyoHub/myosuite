# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Device storage of a multi-clip bank (``ClipBankCfg``): exact by default, opt-in lean.

The clips drive a free root, a ball joint and a hinge; their qvel is MuJoCo's
``mj_differentiatePos`` from the previous frame (forward at the first one), the
convention of the MuscleMimic clips that ``store_qvel=False`` reproduces. The root
travels 20 m, so float16 storage must keep the root position in float32 and the
sites around a centroid.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

pytestmark = pytest.mark.tier2

from myosuite.core.trajectory_io import (  # noqa: E402
    MotionClip,
    expand_motion_clip_to_model,
    load_motion_clip,
)
from myosuite.envs.myo.backends.mjlab import clip_trajectory_source as cts  # noqa: E402
from myosuite.envs.myo.backends.mjlab.clip_trajectory_source import (  # noqa: E402
    ClipBankCfg,
    ClipJointLayout,
    MotionClipBank,
    MultiClipTrajectorySource,
    qvel_derivation_error,
)
from myosuite.tests.support.host_sync import HostSyncCounter  # noqa: E402

_DT = 0.01
_XML = """
<mujoco>
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom size="0.1"/>
      <site name="pelvis"/>
      <body name="arm" pos="0 0 0.3">
        <joint name="ball" type="ball"/>
        <geom size="0.05"/>
        <site name="elbow" pos="0 0 0.2"/>
        <body name="hand" pos="0 0 0.3">
          <joint name="hinge" axis="0 1 0"/>
          <geom size="0.05"/>
          <site name="tip" pos="0.1 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
_SITES = ("pelvis", "elbow", "tip")


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_XML)


def _quat(axis: tuple[float, ...], angle: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis) / np.linalg.norm(axis), angle)
    return quat


def _clip(model: mujoco.MjModel, n: int, phase: float) -> MotionClip:
    """A clip whose qvel is mj_differentiatePos of its qpos and sites its kinematics."""
    data = mujoco.MjData(model)
    t = np.arange(n) * _DT
    qpos = np.zeros((n, model.nq))
    qpos[:, 0] = 20.0 * t / t[-1]  # the root travels 20 m
    qpos[:, 1] = 0.3 * np.sin(t + phase)
    qpos[:, 2] = 1.0
    sites = np.zeros((n, len(_SITES), 3))
    for i, ti in enumerate(t):
        qpos[i, 3:7] = _quat((0.2, 0.3, 1.0), 0.8 * ti + phase)
        qpos[i, 7:11] = _quat((1.0, 0.0, 0.5), 0.5 * np.sin(2.0 * ti) + 0.1)
        qpos[i, 11] = 0.7 * np.sin(3.0 * ti + phase)
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        sites[i] = [data.site(name).xpos for name in _SITES]
    qvel = np.zeros((n, model.nv))
    for i in range(n):
        a, b = (i - 1, i) if i > 0 else (0, 1)
        mujoco.mj_differentiatePos(model, qvel[i], _DT, qpos[a], qpos[b])
    return MotionClip(
        qpos=qpos,
        qvel=qvel,
        site_xpos=sites,
        site_names=list(_SITES),
        frequency_hz=100.0,
    )


@pytest.fixture(scope="module")
def clips(model: mujoco.MjModel) -> tuple[MotionClip, ...]:
    return (_clip(model, 40, 0.0), _clip(model, 70, 1.3))


def _source(
    clips: tuple[MotionClip, ...], model: mujoco.MjModel, **cfg: object
) -> MultiClipTrajectorySource:
    src = MultiClipTrajectorySource(
        clips=clips,
        tracked_site_ids=np.arange(len(_SITES)),
        ctrl_dt=_DT,
        bank_cfg=ClipBankCfg(**cfg),
        joint_layout=ClipJointLayout.from_model(model),
    )
    src.update(torch.zeros(4, dtype=torch.long))
    return src


def _every_frame(
    src: MultiClipTrajectorySource, name: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(gathered, clip array)`` of every frame of every clip."""
    out = []
    for c, clip in enumerate(src.clips):
        n = int(clip.site_xpos.shape[0])
        src._clip_indices = torch.full((n,), c, dtype=torch.long)
        frames = torch.arange(n)
        got = {
            "site_xpos": src.site_targets_at_frames,
            "qpos": src.ref_qpos_at_frames,
            "qvel": src.ref_qvel_at_frames,
        }[name](frames)
        out.append((got.numpy(), getattr(clip, name)))
    src._clip_indices = torch.zeros(4, dtype=torch.long)
    return out


def test_default_bank_is_exact(clips, model) -> None:
    src = _source(clips, model)
    assert src.bank_cfg.exact and src.n_frames == 70
    for name in ("site_xpos", "qpos", "qvel"):
        for got, clip_values in _every_frame(src, name):
            assert got.dtype == np.float32
            np.testing.assert_array_equal(got, clip_values.astype(np.float32))
    assert src.storage_error == {"site_xpos": 0.0, "qpos": 0.0, "qvel": 0.0}
    rows = sum(len(c.qpos) for c in clips)
    assert src.bank_nbytes == rows * (3 * len(_SITES) + model.nq + model.nv) * 4


def test_float16_bank_halves_the_bytes_within_its_error_bound(clips, model) -> None:
    src = _source(clips, model, dtype="float16")
    rows = sum(len(c.qpos) for c in clips)
    # sites: float32 centroid + float16 offsets; qpos: root x (20 m) float32, rest float16
    sites, qpos, qvel = (
        3 * 4 + 3 * len(_SITES) * 2,
        4 + (model.nq - 1) * 2,
        model.nv * 2,
    )
    assert src.bank_nbytes == rows * (sites + qpos + qvel)
    err = src.storage_error
    assert err["site_xpos"] < 1e-3 and err["qpos"] <= 4.0 * 2.0**-11
    for name in ("site_xpos", "qpos", "qvel"):
        for got, clip_values in _every_frame(src, name):
            assert got.dtype == np.float32
            exact = clip_values.astype(np.float32)
            assert np.abs(got - exact).max() <= err[name] + 1e-6
            if name == "qpos":  # the travelled distance stays exact
                np.testing.assert_array_equal(got[:, 0], exact[:, 0])
    qvel_scale = max(np.abs(c.qvel).max() for c in clips)
    assert err["qvel"] <= qvel_scale * 2.0**-11


def test_derived_qvel_matches_mujoco_and_drops_the_qvel_bank(clips, model) -> None:
    exact = _source(clips, model)
    derived = _source(clips, model, store_qvel=False)
    rows = sum(len(c.qpos) for c in clips)
    assert exact.bank_nbytes - derived.bank_nbytes == rows * model.nv * 4
    assert "qvel" not in derived.storage_error
    for got, clip_qvel in _every_frame(derived, "qvel"):  # first frames included
        np.testing.assert_allclose(got, clip_qvel, atol=2e-3, rtol=1e-4)
    assert qvel_derivation_error(clips, model, _DT) < 1e-9


def test_float16_with_derived_qvel_keeps_qpos_exact(clips, model) -> None:
    """Differencing float16 qpos over ctrl_dt would amplify its error ~2/ctrl_dt times."""
    src = _source(clips, model, dtype="float16", store_qvel=False)
    rows = sum(len(c.qpos) for c in clips)
    sites = 3 * 4 + 3 * len(_SITES) * 2
    assert src.bank_nbytes == rows * (sites + model.nq * 4)
    assert src.storage_error["qpos"] == 0.0
    for got, clip_qvel in _every_frame(src, "qvel"):
        np.testing.assert_allclose(got, clip_qvel, atol=2e-3, rtol=1e-4)


def test_the_musclemimic_clip_stores_the_backward_difference() -> None:
    """The retargeted MuscleMimic clips store qvel as the backward difference that
    ``store_qvel=False`` derives (a forward one is off by ~20 % RMS of joint speed)."""
    from myosuite.tests.support.optional_deps import require_musclemimic_models

    require_musclemimic_models()
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="amathislab/musclemimic-retargeted",
            filename="MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz",
            repo_type="dataset",
        )
    except (ImportError, OSError, RuntimeError):
        pytest.skip(
            "MuscleMimic motion clip not available (gated Hugging Face dataset)"
        )
    from ml_collections import config_dict

    from myosuite.integrations.musclemimic.fullbody_model import (
        build_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )

    cfg = config_dict.create(**dict(default_mimic_fullbody_config()))
    full = build_mimic_fullbody_spec(cfg)[0].compile()
    clip = load_motion_clip(Path(path), full.nq, full.nv)
    clip = expand_motion_clip_to_model(clip, full)
    assert qvel_derivation_error([clip], full, float(cfg.ctrl_dt)) < 1e-6


@pytest.mark.parametrize("convention", ["forward", "central"])
def test_qvel_derivation_error_flags_another_convention(
    clips, model, convention: str
) -> None:
    """Forward or central differences are not the backward ones the bank derives."""
    other = []
    for clip in clips:
        qvel = clip.qvel.copy()
        if convention == "forward":  # the backward difference of the next frame
            qvel[:-1] = clip.qvel[1:]
        else:
            qvel[1:-1] = 0.5 * (clip.qvel[1:-1] + clip.qvel[2:])
        other.append(MotionClip(clip.qpos, qvel, clip.site_xpos, clip.site_names))
    assert qvel_derivation_error(other, model, _DT) > 1e-3


def test_lean_banks_gather_without_host_syncs(clips, model) -> None:
    src = _source(clips, model, dtype="float16", store_qvel=False)
    step = torch.full((4,), 3, dtype=torch.long)
    src.update(step, check_resets=False)
    with HostSyncCounter(package_only=True) as syncs:
        src.site_targets(step)
        src.ref_qpos(step)
        src.ref_qvel(step)
        src.initial_qvel()
    assert syncs.total == 0, syncs.report()


def test_store_qvel_false_needs_a_joint_layout(clips) -> None:
    with pytest.raises(ValueError, match="joint_layout"):
        MultiClipTrajectorySource(
            clips=clips,
            tracked_site_ids=np.arange(len(_SITES)),
            ctrl_dt=_DT,
            bank_cfg=ClipBankCfg(store_qvel=False),
        )
    with pytest.raises(ValueError, match="dtype"):
        ClipBankCfg(dtype="bfloat16")  # type: ignore[arg-type]


def test_a_new_env_count_and_the_init_fn_reuse_the_bank(
    clips, model, monkeypatch
) -> None:
    src = _source(clips, model)
    uploads = []
    monkeypatch.setattr(cts.MultiClipTrajectorySource, "_upload", uploads.append)
    src.update(torch.zeros(7, dtype=torch.long))  # another batch size
    torch.manual_seed(5)
    qpos, qvel = src.make_init_state_fn()(6, torch.device("cpu"))
    assert uploads == [] and qpos.shape == (6, model.nq) and qvel.shape == (6, model.nv)
    # The draws are those of a fresh source with its own upload.
    monkeypatch.undo()
    torch.manual_seed(5)
    fresh = MultiClipTrajectorySource(
        clips=clips, tracked_site_ids=np.arange(len(_SITES)), ctrl_dt=_DT
    )
    fresh._ensure_device(torch.device("cpu"), 6)
    np.testing.assert_array_equal(qpos.numpy(), fresh.initial_qpos().numpy())
    np.testing.assert_array_equal(qvel.numpy(), fresh.initial_qvel().numpy())


def test_motion_clip_bank_is_a_tuple_that_keeps_its_cfg(clips) -> None:
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _normalize_motion_clip_bank,
    )

    bank = MotionClipBank(clips, ClipBankCfg(dtype="float16"))
    assert isinstance(bank, tuple) and len(bank) == 2 and bank[1] is clips[1]
    assert _normalize_motion_clip_bank(bank) is bank
    restored = pickle.loads(pickle.dumps(bank))
    assert restored.bank_cfg == ClipBankCfg(dtype="float16") and len(restored) == 2
    assert MotionClipBank(clips).bank_cfg.exact


def test_float32_loading_gives_the_same_bank_at_half_the_host_bytes(
    clips, model, tmp_path
) -> None:
    paths = []
    for i, clip in enumerate(clips):
        path = tmp_path / f"clip{i}.npz"
        np.savez(
            path,
            qpos=clip.qpos,
            qvel=clip.qvel,
            site_xpos=clip.site_xpos,
            site_names=np.asarray(_SITES),
            frequency=100.0,
        )
        paths.append(path)
    banks, host = [], []
    for dtype in (np.float64, np.float32):
        loaded = tuple(
            expand_motion_clip_to_model(
                load_motion_clip(p, model.nq, model.nv, dtype=dtype), model
            )
            for p in paths
        )
        host.append(
            sum(c.qpos.nbytes + c.qvel.nbytes + c.site_xpos.nbytes for c in loaded)
        )
        assert all(c.qpos.dtype == dtype for c in loaded)
        banks.append(_source(loaded, model))
    assert host[1] * 2 == host[0]
    for name in ("site_xpos", "qpos", "qvel"):
        for (a, _), (b, _) in zip(
            _every_frame(banks[0], name), _every_frame(banks[1], name), strict=True
        ):
            np.testing.assert_array_equal(a, b)


def test_expanding_a_partial_float32_clip_keeps_float32(model) -> None:
    """Columns the clip does not name take the model default (zeros without a
    keyframe), in the clip's dtype."""
    hinge = np.linspace(0.0, 1.0, 5, dtype=np.float32)[:, None]
    clip = MotionClip(
        qpos=hinge,
        qvel=None,
        site_xpos=None,
        site_names=None,
        qpos_joint_names=["hinge"],
    )
    expanded = expand_motion_clip_to_model(clip, model)
    assert expanded.qpos.dtype == np.float32 and expanded.qpos.shape == (5, model.nq)
    np.testing.assert_array_equal(expanded.qpos[:, 11], hinge[:, 0])
    np.testing.assert_array_equal(expanded.qpos[:, :11], np.zeros((5, 11), np.float32))
