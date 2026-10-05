# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reward terms keep a leading env-batch axis; the torch muscle term stays on torch."""

from types import SimpleNamespace

import numpy as np
import pytest

from myosuite.terms.base_action import MuscleActionTerm, MuscleActionTermCfg
from myosuite.terms.base_reward import joint_penalty, reach_reward


class _Accessor:
    def __init__(self, tip_pos, qpos=None, ctrl_range=None, jnt_range=None):
        self._tip_pos, self._qpos, self._ctrl_range = tip_pos, qpos, ctrl_range
        self._jnt_range = jnt_range

    def array_module(self):
        return np

    def site_xpos(self, ids):
        return self._tip_pos

    def joint_pos(self):
        return self._qpos

    def ctrl_range(self):
        return self._ctrl_range

    def joint_range(self):
        # Every joint limited, in qpos order.
        return np.arange(len(self._jnt_range)), self._jnt_range


def test_reach_reward_scores_every_env() -> None:
    dist = np.array([0.0, 0.3, 0.6, 0.9])
    tip = np.zeros((4, 1, 3))
    tip[:, 0, 0] = dist
    out = reach_reward(
        _Accessor(tip), {"target_pos": np.zeros(3), "tip_site_ids": [0]}, reach_thd=0.1
    )
    np.testing.assert_allclose(out["reach"], -dist)
    assert out["solved"].tolist() == [True, False, False, False]


def test_reach_reward_single_env_stays_scalar() -> None:
    tip = np.array([[0.3, 0.0, 0.0], [0.5, 0.0, 0.0]])
    out = reach_reward(
        _Accessor(tip), {"target_pos": np.zeros(3), "tip_site_ids": [0, 1]}
    )
    assert np.ndim(out["reach"]) == 0
    assert out["reach"] == pytest.approx(-0.4)


def test_joint_penalty_is_per_env() -> None:
    jnt_range = np.array([[0.0, 1.0], [0.0, 1.0]])
    qpos = np.array([[0.5, 0.5], [1.0, 0.5], [1.0, 1.0]])
    out = joint_penalty(_Accessor(None, qpos, jnt_range=jnt_range), {}, weight=1.0)
    penalty = out["joint_penalty"]
    assert penalty.shape == (3,)
    assert penalty[0] == 0.0 and penalty[1] < 0.0 and penalty[2] < penalty[1]


def test_muscle_action_term_without_normalization_keeps_torch_tensors() -> None:
    torch = pytest.importorskip("torch")
    cfg = MuscleActionTermCfg(entity_name="robot", normalize=False)
    env = SimpleNamespace(scene={"robot": SimpleNamespace(num_actuators=4)})
    term = MuscleActionTerm(cfg, env)
    term.process_actions(torch.tensor([[-1.0, 0.5, 1.0, 2.0]]))
    assert isinstance(term._processed, torch.Tensor)
    assert term._processed.tolist() == [[0.0, 0.5, 1.0, 1.0]]
