# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Native scene-object observation contracts across clip switches."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.integrations.musclemimic import fullbody_local_policy as fbl
from myosuite.integrations.musclemimic.activation_collector import (
    CollectionConfig,
    collect_activations,
)
from myosuite.tests.test_fullbody_obs_adapter import (
    _ACTUATED_JOINTS,
    _SITES,
    _XML,
    _write_clip,
)

pytestmark = pytest.mark.tier2


@pytest.fixture
def scene_clips(tmp_path: Path) -> tuple:
    actuators = "".join(
        f'<general joint="{name}" dyntype="filter" dynprm=".05" ctrlrange="0 1"/>'
        for name in _ACTUATED_JOINTS
    )
    xml = _XML.replace("__ACTUATORS__", actuators).replace(
        "</worldbody>",
        '<body name="prop" pos="2 0 1"><freejoint name="prop_root"/>'
        '<geom size=".1" mass="1"/></body></worldbody>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    clips = [
        _write_clip(model, tmp_path / f"clip_{frames}.npz", n_frames=frames)
        for frames in (8, 12)
    ]
    return model, clips


@pytest.mark.parametrize("exclude_scene", [True, False])
@pytest.mark.parametrize("concise", [True, False])
def test_with_clip_preserves_actor_joint_filter_and_observation_values(
    scene_clips, exclude_scene: bool, concise: bool
) -> None:
    model, clips = scene_clips
    params = {
        "sites_for_mimic": _SITES,
        "exclude_non_actor_joints": exclude_scene,
        "use_concise_lookahead": concise,
        "enable_muscle_force_observations": False,
        "n_step_lookahead": 3,
    }
    original = fbl.FullbodyObsAdapter(model, clips[0], params)
    data = mujoco.MjData(model)
    for clip in clips:
        switched = original.with_clip(clip)
        expected = fbl.FullbodyObsAdapter(model, clip, params)
        for frame in (0, 3, clip.num_frames - 1):
            data.qpos[:] = clip.qpos[frame]
            data.qvel[:] = clip.qvel[frame]
            data.qpos[-7:-4] += [0.3, -0.2, 0.1]
            mujoco.mj_forward(model, data)
            np.testing.assert_array_equal(
                switched.build(data, frame), expected.build(data, frame)
            )
            assert switched.goal_dim == original.goal_dim
            assert switched.build(data, frame).dtype == np.float32


def test_actual_multi_clip_collection_keeps_filtered_policy_layout(scene_clips) -> None:
    model, clips = scene_clips
    adapter = fbl.FullbodyObsAdapter(
        model, clips[0], {"sites_for_mimic": _SITES, "exclude_non_actor_joints": True}
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    obs_dim = adapter.build(data, 0).size
    artifacts = fbl.LocalPolicyArtifacts(
        params={
            "actor": {
                "Dense_0": {
                    "kernel": np.zeros((obs_dim, model.nu), dtype=np.float32),
                    "bias": np.full(model.nu, 0.4, dtype=np.float32),
                }
            }
        },
        obs_mean=np.zeros(obs_dim, dtype=np.float32),
        obs_var=np.ones(obs_dim, dtype=np.float32),
        obs_count=np.asarray(1.0, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=model.nu,
    )
    runner = fbl.LocalPolicyRunner(
        artifacts, False, 0, obs_adapter=adapter, update_normalizer=False
    )
    acts = collect_activations(
        runner,
        model,
        clips,
        CollectionConfig(
            n_episodes_per_clip=1, max_steps_per_episode=2, use_random_start=False
        ),
    )
    assert acts.shape == (4, model.na)
    assert acts.dtype == np.float32 and np.all(np.isfinite(acts))
    assert np.all((acts >= 0) & (acts <= 1))
