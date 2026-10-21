"""Stationary evasions preserve facing, planted feet and a fixed jump origin."""

import numpy as np
import pytest

from myosuite.integrations.musclemimic import (
    MIMIC_SITES,
    build_terrain_fullbody_spec,
    plan_waypoint_reference,
)

pytestmark = pytest.mark.tier1


def test_stationary_squat_and_jump() -> None:
    """A timed hold lowers the pelvis without walking or rotating the feet."""
    pytest.importorskip("musclemimic_models")
    model = build_terrain_fullbody_spec().compile()
    clip, info = plan_waypoint_reference(
        model,
        np.zeros((3, 2)),
        lambda xy: 0.0,
        model.qpos0,
        sites=MIMIC_SITES,
        jump_segments=(1,),
        segment_durations_s=(2.0, 1.0),
        heading_rad=0.0,
        planted_holds=True,
        squat_windows_s=((1.0, 2.0, 0.18),),
    )
    np.testing.assert_allclose(clip.qpos[:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        clip.qpos[:, 3:7], np.tile([1, 0, 0, 0], (clip.num_frames, 1))
    )
    assert clip.qpos[100, 2] - clip.qpos[150, 2] == pytest.approx(0.18)
    feet = info["foot_targets"][100:200]
    np.testing.assert_allclose(feet, np.broadcast_to(feet[0], feet.shape), atol=1e-12)
    assert info["jump_windows_s"][0][0] == pytest.approx(4.8)
    assert np.isfinite(clip.qvel).all()
