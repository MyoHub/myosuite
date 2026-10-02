# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the MuscleMimic model bridge and TensorDict-aware policy bridges."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite.integrations.musclemimic.model_bridge import (
    FULLBODY_NAME_ALIASES,
    BridgedPredictPolicy,
    SharedModelStateBridge,
    TensorDictPredictPolicyAdapter,
    to_muscle_activations,
)
from myosuite.tests.support.optional_deps import require_musclemimic_models

_JOINT = mujoco.mjtObj.mjOBJ_JOINT
_ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR


def _single_hinge_model_xml(
    *,
    actuator_name: str = "muscle",
    joint_name: str = "hinge",
    extra_body: str = "",
) -> str:
    return f"""
<mujoco model="unit_bridge">
  <worldbody>
    <body name="root">
      <joint name="{joint_name}" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.01"/>
    </body>
    {extra_body}
  </worldbody>
  <actuator>
    <motor name="{actuator_name}" joint="{joint_name}" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def _hinge_model(**names: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_single_hinge_model_xml(**names))


def _single_muscle_model_xml(*, ctrlrange: str) -> str:
    return f"""
<mujoco model="unit_muscle">
  <worldbody>
    <body name="root">
      <joint name="hinge" type="hinge" axis="0 0 1" range="-1 1" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <muscle name="muscle" joint="hinge" ctrlrange="{ctrlrange}" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def _activation_trace(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    """Muscle activation over 200 substeps under a constant ``ctrl``."""
    data = mujoco.MjData(model)
    data.ctrl[:] = ctrl
    trace = []
    for _ in range(200):
        mujoco.mj_step(model, data)
        trace.append(float(data.act[0]))
    return np.asarray(trace)


class _TensorDictPolicy:
    def __call__(self, obs_td):
        return obs_td["actor"] * 2.0


def test_tensor_dict_predict_policy_adapter_returns_flat_numpy_action() -> None:
    pytest.importorskip("tensordict")

    adapter = TensorDictPredictPolicyAdapter(_TensorDictPolicy(), device="cpu")

    action = adapter.predict(np.asarray([1.5], dtype=np.float32))

    assert isinstance(action, np.ndarray)
    np.testing.assert_allclose(action, np.asarray([3.0], dtype=np.float32))


def test_to_muscle_activations_clips_actions_to_unit_interval() -> None:
    mapped = to_muscle_activations(
        np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    )

    assert mapped.dtype == np.float32
    np.testing.assert_allclose(mapped, [0.0, 0.0, 0.0, 0.5, 1.0])


@pytest.mark.parametrize("action", [-1.0, -0.4, 0.0, 0.3, 0.5, 1.0])
def test_to_muscle_activations_reproduces_musclemimic_excitation(
    action: float,
) -> None:
    """A [0, 1] source muscle driven by the bridge must follow the trained muscle.

    MuscleMimic models give muscles ctrlrange [-1, 1] and the policy runners
    write the action into ctrl unchanged; MuJoCo's muscle dynamics then clamp
    ctrl to [0, 1].
    """
    trained = mujoco.MjModel.from_xml_string(_single_muscle_model_xml(ctrlrange="-1 1"))
    source = mujoco.MjModel.from_xml_string(_single_muscle_model_xml(ctrlrange="0 1"))
    # Same float32 ctrl write as LocalPolicyRunner.step / OnnxPolicyRunner.step.
    low, high = trained.actuator_ctrlrange[0].astype(np.float32)
    trained_ctrl = np.clip(np.asarray([action], dtype=np.float32), low, high)

    expected = _activation_trace(trained, trained_ctrl)
    bridged = _activation_trace(source, to_muscle_activations([action]))

    np.testing.assert_allclose(bridged, expected, rtol=0.0, atol=1e-12)


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


_MYO_SIM_NAMES = {"joint_name": "elbow_flexion_r", "actuator_name": "BIClong_l"}
_MUSCLEMIMIC_NAMES = {"joint_name": "elbow_flex_r", "actuator_name": "BIClong_left"}


@pytest.mark.parametrize(
    ("source_names", "target_names"),
    [(_MYO_SIM_NAMES, _MUSCLEMIMIC_NAMES), (_MUSCLEMIMIC_NAMES, _MYO_SIM_NAMES)],
    ids=["myo_sim_to_musclemimic", "musclemimic_to_myo_sim"],
)
def test_bridge_maps_known_fullbody_name_aliases(
    source_names: dict[str, str], target_names: dict[str, str]
) -> None:
    source_model = _hinge_model(**source_names)
    target_model = _hinge_model(**target_names)
    bridge = SharedModelStateBridge(source_model, target_model)
    source_data = mujoco.MjData(source_model)
    source_data.qpos[0] = 0.4
    source_data.qvel[0] = -0.3
    target_data = mujoco.MjData(target_model)

    bridge.copy_source_into_target(source_data, target_data)

    np.testing.assert_allclose(target_data.qpos, [0.4])
    np.testing.assert_allclose(target_data.qvel, [-0.3])
    projected = bridge.project_target_action_to_source(
        np.asarray([0.7], dtype=np.float32), fill_value=-1.0
    )
    np.testing.assert_allclose(projected, [0.7])


def test_bridge_rejects_unmatched_names_and_lists_them() -> None:
    source_model = _hinge_model(joint_name="hinge", actuator_name="muscle")
    target_model = _hinge_model(joint_name="knee", actuator_name="other")

    with pytest.raises(ValueError, match="allow_partial") as excinfo:
        SharedModelStateBridge(source_model, target_model)

    message = str(excinfo.value)
    assert "target joints" in message and "'knee'" in message
    assert "target actuators" in message and "'other'" in message
    assert "source actuators" in message and "'muscle'" in message
    with pytest.raises(ValueError, match="'other'"):
        BridgedPredictPolicy(
            source_model,
            target_model,
            obs_builder=lambda data, frame_idx: np.zeros(1, dtype=np.float32),
            policy=SimpleNamespace(predict=lambda obs: np.zeros(1)),
            clip_frame_count=1,
            ctrl_dt=0.01,
        )


def test_bridge_ignores_source_only_joints() -> None:
    """Extra source DoFs (e.g. scene objects) are not part of the policy model."""
    object_body = """
    <body name="object" pos="1 0 0">
      <joint name="object_slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05"/>
    </body>"""
    source_model = _hinge_model(extra_body=object_body)
    target_model = _hinge_model()

    bridge = SharedModelStateBridge(source_model, target_model)

    assert bridge.shared_joint_names == ("hinge",)


def test_bridge_allow_partial_fills_unmatched_actuators_and_logs_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_model = _hinge_model(actuator_name="muscle")
    target_model = _hinge_model(actuator_name="other")

    with caplog.at_level(logging.WARNING, logger=SharedModelStateBridge.__module__):
        bridge = SharedModelStateBridge(source_model, target_model, allow_partial=True)

    assert "'other'" in caplog.text and "'muscle'" in caplog.text
    projected = bridge.project_target_action_to_source(
        np.asarray([0.7], dtype=np.float32), fill_value=-1.0
    )
    np.testing.assert_allclose(projected, [-1.0])


@pytest.fixture(scope="module")
def fullbody_models() -> tuple[mujoco.MjModel, mujoco.MjModel]:
    """``(myo_sim full body, musclemimic_models full body)`` with the mimic edits."""
    require_musclemimic_models()
    from myosuite.integrations.musclemimic.fullbody_model import (
        build_native_mimic_fullbody_spec,
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )

    config = default_mimic_fullbody_config()
    musclemimic_model, _, _ = compile_mimic_fullbody_mjmodel(config)
    return build_native_mimic_fullbody_spec(config).compile(), musclemimic_model


def test_fullbody_bridge_maps_every_joint_and_actuator(
    fullbody_models: tuple[mujoco.MjModel, mujoco.MjModel],
) -> None:
    source, target = fullbody_models
    bridge = SharedModelStateBridge(source, target)

    assert (target.njnt, target.nu) == (83, 354)
    assert len(bridge.shared_joint_names) == target.njnt
    assert len(bridge.shared_actuator_names) == target.nu == source.nu

    # The left elbow reaches the policy model instead of staying at the keyframe.
    source_data = mujoco.MjData(source)
    source_elbow = mujoco.mj_name2id(source, _JOINT, "elbow_flexion_l")
    source_data.qpos[source.jnt_qposadr[source_elbow]] = 1.2
    target_data = mujoco.MjData(target)
    bridge.copy_source_into_target(source_data, target_data)
    target_elbow = mujoco.mj_name2id(target, _JOINT, "elbow_flex_l")
    assert target_data.qpos[target.jnt_qposadr[target_elbow]] == pytest.approx(1.2)

    # A left-arm policy output reaches the source muscle instead of being dropped.
    action = np.full(target.nu, -1.0, dtype=np.float32)
    action[mujoco.mj_name2id(target, _ACTUATOR, "BIClong_left")] = 0.8
    projected = bridge.project_target_action_to_source(action, fill_value=-1.0)
    assert projected[mujoco.mj_name2id(source, _ACTUATOR, "BIClong_l")] == 0.8
    assert np.count_nonzero(projected != -1.0) == 1


def test_fullbody_name_aliases_pair_identical_joints_and_muscles(
    fullbody_models: tuple[mujoco.MjModel, mujoco.MjModel],
) -> None:
    """Each ``(myo_sim, musclemimic_models)`` pair is the same joint or muscle."""
    myo_sim_model, musclemimic_model = fullbody_models
    n_joints = n_muscles = 0
    for myo_sim_name, musclemimic_name in FULLBODY_NAME_ALIASES:
        src_joint = mujoco.mj_name2id(myo_sim_model, _JOINT, myo_sim_name)
        if src_joint >= 0:
            tgt_joint = mujoco.mj_name2id(musclemimic_model, _JOINT, musclemimic_name)
            assert tgt_joint >= 0, musclemimic_name
            for field in ("jnt_type", "jnt_axis", "jnt_range"):
                np.testing.assert_allclose(
                    getattr(myo_sim_model, field)[src_joint],
                    getattr(musclemimic_model, field)[tgt_joint],
                )
            src_body = myo_sim_model.jnt_bodyid[src_joint]
            tgt_body = musclemimic_model.jnt_bodyid[tgt_joint]
            for field in ("body_pos", "body_quat"):
                # The two MJCFs round body frames differently (1e-6).
                np.testing.assert_allclose(
                    getattr(myo_sim_model, field)[src_body],
                    getattr(musclemimic_model, field)[tgt_body],
                    atol=1e-5,
                )
            n_joints += 1
            continue
        src_act = mujoco.mj_name2id(myo_sim_model, _ACTUATOR, myo_sim_name)
        tgt_act = mujoco.mj_name2id(musclemimic_model, _ACTUATOR, musclemimic_name)
        assert src_act >= 0 and tgt_act >= 0, (myo_sim_name, musclemimic_name)
        for field in ("actuator_gainprm", "actuator_dynprm", "actuator_lengthrange"):
            np.testing.assert_allclose(
                getattr(myo_sim_model, field)[src_act],
                getattr(musclemimic_model, field)[tgt_act],
                rtol=1e-6,
            )
        n_muscles += 1
    assert (n_joints, n_muscles) == (2, 32)


def test_fullbody_bridge_from_full_myo_sim_body_needs_allow_partial(
    fullbody_models: tuple[mujoco.MjModel, mujoco.MjModel],
) -> None:
    """myo_sim's body keeps the finger muscles the policy model removes."""
    import myo_sim

    _, target = fullbody_models
    source = myo_sim.load_spec("myofullbody").compile()

    with pytest.raises(ValueError, match="62 source actuators"):
        SharedModelStateBridge(source, target)
    bridge = SharedModelStateBridge(source, target, allow_partial=True)

    assert len(bridge.shared_joint_names) == target.njnt
    assert len(bridge.shared_actuator_names) == target.nu
