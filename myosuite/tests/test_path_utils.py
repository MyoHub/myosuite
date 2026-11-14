# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Terminal transitions bound the time arrays of MJRL-style rollout paths."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from myosuite.utils.path_utils import evaluate_success, truncate_paths
from myosuite.utils.tensor_utils import stack_tensor_dict_list

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize(
    ("done", "length", "terminated"),
    [
        ([], 0, False),
        ([False], 1, False),
        ([True], 1, True),
        ([False, False, False], 3, False),
        ([True, True, True], 1, True),
        ([True, False, False], 1, True),
        ([False, False, True], 3, True),
        ([False, False, True, True, True], 3, True),
        ([False, True, False, True, False], 2, True),
    ],
)
def test_first_terminal_transition_is_included_exactly_once(
    done: list[bool], length: int, terminated: bool
) -> None:
    path = {"done": np.array(done, dtype=bool), "rewards": np.arange(len(done))}
    paths = [path]
    assert truncate_paths(paths) is paths
    assert paths[0] is path
    np.testing.assert_array_equal(path["done"], done[:length])
    np.testing.assert_array_equal(path["rewards"], np.arange(length))
    assert path["terminated"] is terminated
    truncate_paths(paths)
    assert len(path["rewards"]) == length
    assert path["terminated"] is terminated


def test_empty_path_list_is_unchanged() -> None:
    paths: list[dict] = []
    assert truncate_paths(paths) is paths


def test_nested_mjrl_info_stays_aligned_without_slicing_metadata() -> None:
    steps = 5
    env_infos = stack_tensor_dict_list(
        [
            {
                "obs_dict": {"time": np.array([i]), "qpos": np.full(3, i)},
                "rwd_dict": {"dense": float(i), "solved": i > 2},
                "solved": i > 2,
            }
            for i in range(steps)
        ]
    )
    agent_infos = stack_tensor_dict_list(
        [{"mean": np.full(2, i), "log_std": np.zeros(2)} for i in range(steps)]
    )
    metadata = {
        "name": "recorded trial",
        "scale": np.array(2.0),
        "joint_names": [f"j{i}" for i in range(steps)],
        "calibration": np.array([2.0, 3.0]),
    }
    path = {
        "observations": np.arange(steps * 3).reshape(steps, 3),
        "actions": np.arange(steps * 2).reshape(steps, 2),
        "rewards": np.array([1, 1, 1, 1000, 1000]),
        "done": np.array([False, False, True, True, True]),
        "env_infos": env_infos,
        "agent_infos": agent_infos,
        "terminated": False,
        "metadata": metadata,
    }
    before = deepcopy(path)
    truncate_paths([path])
    for key in ("observations", "actions", "rewards", "done"):
        np.testing.assert_array_equal(path[key], before[key][:3])
    assert path["env_infos"] is env_infos
    assert path["agent_infos"] is agent_infos
    np.testing.assert_array_equal(
        env_infos["obs_dict"]["time"], before["env_infos"]["obs_dict"]["time"][:3]
    )
    np.testing.assert_array_equal(
        env_infos["obs_dict"]["qpos"], before["env_infos"]["obs_dict"]["qpos"][:3]
    )
    np.testing.assert_array_equal(
        env_infos["rwd_dict"]["dense"], before["env_infos"]["rwd_dict"]["dense"][:3]
    )
    np.testing.assert_array_equal(
        agent_infos["mean"], before["agent_infos"]["mean"][:3]
    )
    np.testing.assert_array_equal(
        agent_infos["log_std"], before["agent_infos"]["log_std"][:3]
    )
    assert path["metadata"] is metadata
    assert metadata["name"] == before["metadata"]["name"]
    assert metadata["joint_names"] == before["metadata"]["joint_names"]
    np.testing.assert_array_equal(metadata["scale"], before["metadata"]["scale"])
    np.testing.assert_array_equal(
        metadata["calibration"], before["metadata"]["calibration"]
    )
    assert path["rewards"].sum() == 3
    assert evaluate_success(None, [path], successful_steps=0) == 0
    truncate_paths([path])
    assert path["rewards"].sum() == 3
    assert len(env_infos["obs_dict"]["time"]) == 3


@pytest.mark.parametrize("done", [np.array([[False], [True]]), np.array([False])])
def test_invalid_done_layout_is_rejected(done: np.ndarray) -> None:
    with pytest.raises(ValueError, match="one-dimensional.*same length"):
        truncate_paths([{"rewards": np.zeros(2), "done": done}])
