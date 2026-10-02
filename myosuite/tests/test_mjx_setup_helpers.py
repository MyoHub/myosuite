# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Numpy-level tests for the MJX setup helpers (no JAX needed).

- ``utils.target_ranges``: target ranges resolved by joint/site name, in model
  order, whatever the mapping order (``ConfigDict`` iterates alphabetically).
- ``terms.mimic_obs.resolve_mimic_site_ids``: unknown site names raise.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401  # registers the CPU envs
from myosuite.terms.mimic_obs import resolve_mimic_site_ids
from myosuite.utils.target_ranges import (
    resolve_joint_target_ranges,
    resolve_site_target_ranges,
)

pytestmark = pytest.mark.tier1

_TINY_XML = """
<mujoco>
  <worldbody>
    <body name="b0">
      <joint name="h0" type="hinge"/>
      <geom size="0.1"/>
      <site name="tip"/>
      <body name="b1" pos="0 0 0.3">
        <joint name="h1" type="hinge" axis="1 0 0"/>
        <geom size="0.1"/>
        <site name="tip2"/>
        <site name="tip2_target"/>
        <body name="b2" pos="0 0 0.3">
          <joint name="ball" type="ball"/>
          <geom size="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _cpu_env(env_id: str):
    """Return the unwrapped CPU env and its registered kwargs."""
    env = gym.make(env_id).unwrapped
    return env, gym.spec(env_id).kwargs


def _orderings(mapping: dict) -> list:
    """Same mapping in declared, reversed and alphabetical (ConfigDict) order."""
    from ml_collections import config_dict

    return [
        mapping,
        dict(reversed(list(mapping.items()))),
        config_dict.ConfigDict(mapping),
    ]


@pytest.mark.parametrize(
    "env_id",
    [
        "myoElbowPose1D6MRandom-v0",
        "myoFingerPoseFixed-v0",
        "myoFingerPoseRandom-v0",
        "myoHandPoseRandom-v0",
    ],
)
def test_joint_ranges_match_cpu_and_ignore_mapping_order(env_id: str) -> None:
    """MJX target bounds equal what the CPU env compares with qpos[:n]."""
    pytest.importorskip("ml_collections")
    env, kwargs = _cpu_env(env_id)
    cpu = np.asarray(env._target_jnt_range)
    for ranges in _orderings(kwargs["target_jnt_range"]):
        lo, hi = resolve_joint_target_ranges(env.model, ranges)
        np.testing.assert_array_equal(lo, cpu[:, 0])
        np.testing.assert_array_equal(hi, cpu[:, 1])
    env.close()


def test_finger_pose_fixed_target_on_the_right_joints() -> None:
    """The review's example: [0, 0, 0.75, 0.75], not [0, 0.75, 0, 0.75]."""
    pytest.importorskip("ml_collections")
    env, kwargs = _cpu_env("myoFingerPoseFixed-v0")
    for ranges in _orderings(kwargs["target_jnt_range"]):
        lo, hi = resolve_joint_target_ranges(env.model, ranges)
        np.testing.assert_array_equal(lo, [0.0, 0.0, 0.75, 0.75])
        np.testing.assert_array_equal(hi, lo)
    env.close()


def test_joint_ranges_reject_bad_specs() -> None:
    model = mujoco.MjModel.from_xml_string(_TINY_XML)
    with pytest.raises(KeyError):  # unknown name (no silent id -1)
        resolve_joint_target_ranges(model, {"h0": (0, 1), "missing": (0, 1)})
    with pytest.raises(ValueError, match="qpos"):  # not the leading qpos entries
        resolve_joint_target_ranges(model, {"h1": (0, 1)})
    with pytest.raises(ValueError, match="hinge/slide"):
        resolve_joint_target_ranges(model, {"ball": (0, 1)})
    with pytest.raises(ValueError, match="pair"):
        resolve_joint_target_ranges(model, {"h0": (0, 1, 2)})
    with pytest.raises(ValueError, match="empty"):
        resolve_joint_target_ranges(model, {})


@pytest.mark.parametrize("env_id", ["myoHandReachFixed-v0", "myoHandReachRandom-v0"])
def test_site_ranges_keep_pairing_and_cpu_order(env_id: str) -> None:
    """Five distinct tips, each with its own box, in the CPU order."""
    pytest.importorskip("ml_collections")
    env, kwargs = _cpu_env(env_id)
    cpu_ranges = kwargs["target_reach_range"]
    for ranges in _orderings(cpu_ranges):
        targets = resolve_site_target_ranges(env.model, ranges)
        assert targets.names == tuple(cpu_ranges)  # TH, IF, MF, RF, LF
        np.testing.assert_array_equal(targets.tip_ids, env.tip_sids)
        np.testing.assert_array_equal(targets.target_ids, env.target_sids)
        for i, name in enumerate(targets.names):
            np.testing.assert_array_equal(targets.lo[i], cpu_ranges[name][0])
            np.testing.assert_array_equal(targets.hi[i], cpu_ranges[name][1])
    env.close()


def test_site_ranges_reject_unknown_names() -> None:
    env, kwargs = _cpu_env("myoHandReachFixed-v0")
    ranges = {
        name.removesuffix("_r"): span
        for name, span in kwargs["target_reach_range"].items()
    }
    with pytest.raises(KeyError):  # the old MJX names ("THtip", ...) used to give id -1
        resolve_site_target_ranges(env.model, ranges)
    env.close()
    model = mujoco.MjModel.from_xml_string(_TINY_XML)
    with pytest.raises(KeyError):  # tip without its "<name>_target" site
        resolve_site_target_ranges(model, {"tip": ((0, 0, 0), (1, 1, 1))})
    with pytest.raises(ValueError, match="lo_xyz"):
        resolve_site_target_ranges(model, {"tip2": ((0, 0), (1, 1))})


def test_mimic_site_ids_reject_unknown_names() -> None:
    model = mujoco.MjModel.from_xml_string(_TINY_XML)
    np.testing.assert_array_equal(
        resolve_mimic_site_ids(model, ("tip2", "tip")), [1, 0]
    )
    with pytest.raises(KeyError):
        resolve_mimic_site_ids(model, ("tip", "missing"))
