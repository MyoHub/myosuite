# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reset bins contain transitions, while terminal frames remain scoring targets."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.envs.myo.tasks.mimic.tracking.config import ResetCfg
from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_body
from myosuite.envs.myo.tasks.mimic.tracking.motion import (
    MotionLib,
    build_motion_lib,
    write_synthetic_clip,
)
from myosuite.envs.myo.tasks.mimic.tracking.sampling import ResetSampler

pytestmark = pytest.mark.tier2


def _library(
    tmp_path: Path, lengths: tuple[int, ...], bin_frames: int, cache_dir: str = ""
) -> MotionLib:
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body name="root"><freejoint name="root_joint"/>
          <geom size=".1"/><body name="limb"><joint name="hinge" range="-1 1"/>
          <geom size=".1"/></body></body></worldbody></mujoco>
    """)
    paths = []
    for i, n in enumerate(lengths):
        path = tmp_path / f"clip_{i}.npz"
        np.savez(
            path,
            fps=100.0,
            root_pos=np.zeros((n, 3)),
            root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
            joint_pos=np.zeros((n, 1)),
            joint_names=np.array(["hinge"]),
        )
        paths.append(str(path))
    return build_motion_lib(
        paths,
        model,
        ("root", "limb"),
        0.01,
        bin_s=bin_frames * 0.01,
        workers=1,
        cache_dir=cache_dir,
    )


@pytest.mark.parametrize("bin_frames", [1, 100])
def test_bins_cover_only_transition_starts(tmp_path: Path, bin_frames: int) -> None:
    """Exact bin boundaries and consecutive clips never create an empty draw."""
    lengths = (2, 100, 101, 102, 200, 201, 202)
    lib = _library(tmp_path, lengths, bin_frames)
    np.testing.assert_array_equal(lib.clip_len, lengths)
    for clip, (first, last) in enumerate(zip(lib.clip_start, lib.clip_end)):
        bins = np.flatnonzero(lib.bin_clip == clip)
        starts = lib.bin_start[bins]
        assert starts[0] == first and np.all(starts < last)
        assert len(np.unique(starts)) == len(starts)
        frames = np.arange(first, last)
        assigned = lib.frame_bin[frames]
        assert np.all(lib.bin_clip[assigned] == clip)
        assert np.all(lib.bin_start[assigned] <= frames)
        assert np.all(frames < lib.bin_start[assigned] + bin_frames)
        assert lib.frame_bin[last] == lib.frame_bin[last - 1]
        assert lib.bin_clip[lib.frame_bin[last]] == clip


@pytest.mark.parametrize("mode", ["uniform", "return"])
def test_cpu_sampler_excludes_terminal_frames(tmp_path: Path, mode: str) -> None:
    """Both mixtures leave a successor in the same clip, including failure updates."""
    lib = _library(tmp_path, (2, 101, 102, 201), 100)
    sampler = ResetSampler(lib, ResetCfg(mode=mode, fail_reset=0.3), 0.24)
    rng = np.random.default_rng(0)
    for _ in range(300):
        frame, bin_id = sampler.sample(rng)
        clip = lib.bin_clip[bin_id]
        assert lib.clip_start[clip] <= frame < lib.clip_end[clip]
        assert lib.clip_id[frame + 1] == clip
        assert lib.frame_bin[frame] == bin_id
    for clip, terminal in enumerate(lib.clip_end):
        bin_id = lib.frame_bin[terminal]
        sampler.finish_episode(bin_id, int(terminal), 1, 0.0, 0.0, True)
        assert lib.bin_clip[bin_id] == clip
    assert np.isfinite(sampler.probs()).all()


def test_resampled_clip_requires_a_transition(tmp_path: Path) -> None:
    """A valid source that is shorter than ctrl_dt gets an actionable error."""
    model = compiled_body("myofullbody")
    path = tmp_path / "too_short.npz"
    write_synthetic_clip(path, model, 0.01, 0.0, 0.0, 0)
    with pytest.raises(ValueError, match="resampled clip needs at least two frames"):
        build_motion_lib([str(path)], model, ("pelvis",), 0.02, workers=1)


def test_disk_hit_reconstructs_transition_bins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The data-only cache restores corrected bins without reprocessing frames."""
    from myosuite.envs.myo.tasks.mimic.tracking import motion

    directory = str(tmp_path / "cache")
    cold = _library(tmp_path, (2, 101, 201), 100, directory)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("cache hit must not preprocess clips")

    monkeypatch.setattr(motion, "_build_motion_lib", forbidden)
    warm = _library(tmp_path, (2, 101, 201), 100, directory)
    np.testing.assert_array_equal(cold.bin_start, warm.bin_start)
    np.testing.assert_array_equal(cold.frame_bin, warm.frame_bin)
    assert np.all(warm.bin_start < warm.clip_end[warm.bin_clip])


@pytest.mark.parametrize("mode", ["uniform", "return"])
@pytest.mark.parametrize("seconds", [0.01, 1.0])
def test_cpu_reset_steps_to_the_terminal_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, seconds: float
) -> None:
    """The one-transition clip and the original 1 s/10 ms crash both step safely."""
    path = tmp_path / "synthetic.npz"
    write_synthetic_clip(path, compiled_body("myofullbody"), seconds, 0.0, 0.0, 0)
    env = make_env(
        EnvConfig(
            "myoMimic-v0",
            ctrl_dt=0.01,
            task_kwargs={"overrides": {"motion.clips": str(path), "reset.mode": mode}},
        )
    )
    try:
        env.reset(seed=0)
        u = env.unwrapped
        last = int(u.motion.clip_end[0])
        assert u.frame < last
        env.step(np.zeros(u.model.nu))
        monkeypatch.setattr(
            u._sampler,
            "sample",
            lambda rng: (last - 1, int(u.motion.frame_bin[last - 1])),
        )
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(u.model.nu))
        assert u.frame == last and info["clip_end"]
        assert truncated and not terminated
        assert np.isfinite(obs).all() and np.isfinite(reward)
    finally:
        env.close()
