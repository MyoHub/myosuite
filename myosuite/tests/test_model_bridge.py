# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for TensorDict-aware policy bridges."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite.integrations.musclemimic.model_bridge import (
    BridgedPredictPolicy,
    TensorDictPredictPolicyAdapter,
    to_muscle_activations,
)


def _single_hinge_model_xml(*, actuator_name: str = "muscle") -> str:
    return f"""
<mujoco model="unit_bridge">
  <worldbody>
    <body name="root">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="{actuator_name}" joint="hinge" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class _TensorDictPolicy:
    def __call__(self, obs_td):
        return obs_td["actor"] * 2.0


def test_tensor_dict_predict_policy_adapter_returns_flat_numpy_action() -> None:
    pytest.importorskip("tensordict")

    adapter = TensorDictPredictPolicyAdapter(_TensorDictPolicy(), device="cpu")

    action = adapter.predict(np.asarray([1.5], dtype=np.float32))

    assert isinstance(action, np.ndarray)
    np.testing.assert_allclose(action, np.asarray([3.0], dtype=np.float32))


def test_to_muscle_activations_maps_logits_into_unit_interval() -> None:
    mapped = to_muscle_activations(np.asarray([-1.0, 0.0, 1.0], dtype=np.float32))

    np.testing.assert_allclose(
        mapped,
        np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
    )


def test_bridged_predict_policy_supports_tensor_dict_adapter_and_bound_env() -> None:
    pytest.importorskip("tensordict")

    source_model = mujoco.MjModel.from_xml_string(_single_hinge_model_xml())
    target_model = mujoco.MjModel.from_xml_string(_single_hinge_model_xml())
    source_data = mujoco.MjData(source_model)
    source_data.qpos[0] = 0.25
    source_data.qvel[0] = -0.5
    source_data.ctrl[0] = 0.1
    mujoco.mj_forward(source_model, source_data)

    bridge_policy = BridgedPredictPolicy(
        source_model=source_model,
        target_model=target_model,
        obs_builder=lambda data, frame_idx: np.asarray(
            [data.qpos[0] + float(frame_idx)],
            dtype=np.float32,
        ),
        policy=TensorDictPredictPolicyAdapter(_TensorDictPolicy(), device="cpu"),
        clip_frame_count=8,
        ctrl_dt=0.01,
        source_action_transform=None,
        source_env=SimpleNamespace(model=source_model, data=source_data),
        output_device="cpu",
    )

    direct_action = bridge_policy.predict_from_source_data(source_data)
    call_action = bridge_policy(None)

    np.testing.assert_allclose(
        direct_action,
        np.asarray([0.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        call_action.detach().cpu().numpy(),
        np.asarray([[0.5]], dtype=np.float32),
    )
