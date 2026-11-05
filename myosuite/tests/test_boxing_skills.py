"""Checks for the boxing skill library manifest and cutter (scripts/boxing_skills.py)."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "boxing_skills.py"

spec = importlib.util.spec_from_file_location("boxing_skills", SCRIPT)
boxing_skills = importlib.util.module_from_spec(spec)
sys.modules["boxing_skills"] = boxing_skills  # dataclasses look the module up here
spec.loader.exec_module(boxing_skills)

pytestmark = pytest.mark.skipif(
    not boxing_skills.CLIP_DIR.is_dir(), reason="boxing mocap clips not checked out"
)


def test_manifest_ranges_fit_their_clips() -> None:
    skills = boxing_skills.load_manifest()
    assert len({s.name for s in skills}) == len(skills)
    for s in skills:
        n = np.load(boxing_skills.CLIP_DIR / f"{s.clip}.npz")["qpos"].shape[0]
        assert 0 <= s.start < s.end <= n, s.name
        assert s.kind, s.name


def test_cut_slices_and_pads() -> None:
    skill = next(s for s in boxing_skills.load_manifest() if s.kind == "punch_R")
    src = np.load(boxing_skills.CLIP_DIR / f"{skill.clip}.npz")
    out = boxing_skills.cut_skill(skill)
    n = skill.end - skill.start
    for k in boxing_skills.PER_FRAME:
        assert out[k].shape[0] == n + skill.pad, k
        np.testing.assert_array_equal(out[k][:n], src[k][skill.start : skill.end])
    np.testing.assert_array_equal(
        out["qpos"][n:], np.repeat(out["qpos"][n - 1 : n], skill.pad, 0)
    )
    assert not out["qvel"][n:].any() and not out["cvel"][n:].any()
    assert list(out["split_points"]) == [0, n + skill.pad]


def test_cut_clip_loads(tmp_path: Path) -> None:
    from myosuite.core.trajectory_io import load_motion_clip

    skill = boxing_skills.load_manifest()[0]
    path = tmp_path / f"{skill.name}.npz"
    np.savez(path, **boxing_skills.cut_skill(skill))
    clip = load_motion_clip(path, expected_nq=89, expected_nv=88)
    assert clip.frequency_hz == 100.0
    assert clip.qpos.shape[0] == skill.end - skill.start + skill.pad


def test_detect_finds_alternating_punches() -> None:
    d = np.load(boxing_skills.CLIP_DIR / "punchboxing_push_poses.npz")
    punches = boxing_skills.detect_punches(
        d, boxing_skills.PUNCH_WINDOWS["punchboxing_push_poses"]
    )
    assert [p["hand"] for p in punches] == ["R", "L", "R"]
    for a, b in itertools.pairwise(punches):
        assert a["end"] <= b["start"]
