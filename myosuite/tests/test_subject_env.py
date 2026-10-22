# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Registered subject task construction, reproducibility and policy admission."""

from __future__ import annotations

import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.core.subject_env import (
    SubjectEnvConfig,
    parse_subject_args,
    subject_metadata,
)
from myosuite.core.subject_model import (
    GroundAlignment,
    ScaleEstimate,
    SubjectModelConfig,
    align_root_to_ground,
)
from myosuite.utils.checkpoint_manifest import check_contract, write_manifest
from myosuite.utils.checkpoint_utils import load_policy

pytestmark = pytest.mark.tier1
ENV_ID = "myoLegWalk-v0"


def subject_config(**kwargs) -> SubjectEnvConfig:
    """A nonuniform subject with explicitly selected collision soles."""
    sole = tuple(
        f"{name}_{side}"
        for side in ("r", "l")
        for name in (
            "foot_col1",
            "foot_col3",
            "foot_col4",
            "bofoot_col1",
            "bofoot_col2",
        )
    )
    model = SubjectModelConfig(
        ScaleEstimate(
            {"femur_r": [1.02, 1.07, 0.98], "tibia_l": 1.04},
            mass_kg=70,
            source="synthetic",
        ),
        ground=GroundAlignment("root", sole, clearance=0.002),
        mass="keep",
    )
    return SubjectEnvConfig(model, min_height=0.8, contact_margin=0.001, **kwargs)


def test_subject_recipe_serializes_and_survives_worker_copy(tmp_path: Path) -> None:
    config = subject_config(joint_offsets={"hip_rotation_r": (-0.01, 0.01)})
    assert copy.deepcopy(config) is config
    restored = pickle.loads(pickle.dumps(config))
    assert restored.to_dict() == config.to_dict()
    path = tmp_path / "subject.json"
    path.write_text(json.dumps(config.to_dict()))
    assert SubjectEnvConfig.load(path).digest == config.digest
    parsed, args = parse_subject_args(["env", "--subject", str(path), "--seed", "3"])
    assert parsed.digest == config.digest
    assert args == ["env", "--seed", "3"]
    with pytest.raises(ValueError, match="once"):
        parse_subject_args(["--subject", str(path), "--subject", str(path)])


def test_registered_subject_preserves_contract_and_resets_deterministically() -> None:
    config = subject_config(joint_offsets={"hip_rotation_r": (-0.01, 0.01)})
    nominal, subject = make_env(ENV_ID), make_env(EnvConfig(ENV_ID, subject=config))
    try:
        assert nominal.observation_space == subject.observation_space
        assert nominal.action_space == subject.action_space
        assert subject.unwrapped.model.body_mass.sum() == pytest.approx(70)
        first, info = subject.reset(seed=5)
        assert info["subject_model"] == subject_metadata(subject)
        subject.step(np.full(subject.action_space.shape, 0.2, dtype=np.float32))
        second, _ = subject.reset(seed=5)
        np.testing.assert_array_equal(first, second)
        assert subject.unwrapped.min_height == config.min_height
        assert np.all(np.isfinite(first)) and first.dtype == np.float32
        assert len(subject.step(np.zeros(subject.action_space.shape))) == 5
        metadata = subject_metadata(subject)
        metadata["body_scales"].clear()
        assert subject_metadata(subject)["body_scales"]
    finally:
        nominal.close()
        subject.close()


def test_subject_reset_pose_is_aligned_and_nominal_registration_unchanged() -> None:
    config = subject_config()
    a = make_env(ENV_ID)
    b = make_env(EnvConfig(ENV_ID, subject=config))
    c = make_env(ENV_ID)
    try:
        b.reset(seed=0)
        _, delta = align_root_to_ground(
            b.unwrapped.model, config.model.ground, b.unwrapped.data.qpos
        )
        assert delta == pytest.approx(0, abs=1e-12)
        np.testing.assert_array_equal(
            a.unwrapped.model.body_pos, c.unwrapped.model.body_pos
        )
        assert subject_metadata(a) is None
        assert b.unwrapped.data.qpos[2] != a.unwrapped.model.key_qpos[2, 2]
        d = make_env(EnvConfig(ENV_ID, subject=replace(config, min_height=0.75)))
        try:
            assert (
                subject_metadata(b)["task_digest"] != subject_metadata(d)["task_digest"]
            )
        finally:
            d.close()
    finally:
        a.close()
        b.close()
        c.close()


def test_unsupported_subject_task_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="paired"):
        make_env(EnvConfig("myoElbowPose1D6MRandom-v0", subject=subject_config()))
    with pytest.raises(NotImplementedError, match="CPU and mjlab"):
        make_env(EnvConfig(ENV_ID, subject=subject_config()), backend="mjx")


def test_retargeted_reset_state_is_validated_and_part_of_identity() -> None:
    original = make_env(EnvConfig(ENV_ID, subject=subject_config()))
    try:
        original.reset(seed=0)
        model = original.unwrapped.model
        pose = original.unwrapped.data.qpos.copy()
        pose[model.joint("hip_flexion_r").qposadr[0]] += 0.1
        velocity = np.zeros(model.nv)
        velocity[0] = 0.2
        config = subject_config(reset_qpos=pose, reset_qvel=velocity)
        restored = pickle.loads(
            pickle.dumps(SubjectEnvConfig.from_dict(config.to_dict()))
        )
        env = make_env(EnvConfig(ENV_ID, subject=restored))
        try:
            env.reset(seed=0)
            aligned, _ = align_root_to_ground(model, config.model.ground, pose)
            np.testing.assert_allclose(env.unwrapped.data.qpos, aligned, atol=1e-12)
            np.testing.assert_array_equal(env.unwrapped.data.qvel, velocity)
            assert config.digest != subject_config().digest
        finally:
            env.close()
        with pytest.raises(ValueError, match="requires .* values"):
            make_env(EnvConfig(ENV_ID, subject=subject_config(reset_qpos=(0.0,))))
        pose[3:7] = 0
        with pytest.raises(ValueError, match="unit joint quaternions"):
            make_env(EnvConfig(ENV_ID, subject=subject_config(reset_qpos=pose)))
    finally:
        original.close()


def test_policy_entrypoint_checks_subject_automatically_before_loading(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"unloadable; admission must fail first")
    nominal = make_env(ENV_ID)
    a = make_env(EnvConfig(ENV_ID, subject=subject_config()))
    other = make_env(
        EnvConfig(ENV_ID, subject=replace(subject_config(), min_height=0.75))
    )
    config = subject_config()
    differently_scaled = make_env(
        EnvConfig(
            ENV_ID,
            subject=replace(
                config,
                model=replace(config.model, estimate=ScaleEstimate({"femur_r": 1.03})),
            ),
        )
    )
    try:
        with pytest.raises(ValueError, match="morphology"):
            load_policy(a, checkpoint, strict=True)
        write_manifest(tmp_path, ENV_ID, env=a)
        assert check_contract(ENV_ID, tmp_path, env=a, strict=True)
        for env in (nominal, other, differently_scaled):
            with pytest.raises(ValueError, match="morphology"):
                load_policy(env, checkpoint, strict=True)
    finally:
        nominal.close()
        a.close()
        other.close()
        differently_scaled.close()


def test_subject_strength_features_participate_in_checkpoint_admission(
    tmp_path: Path,
) -> None:
    from myosuite.envs.wrappers import SarcopeniaWrapper

    ordinary = make_env(EnvConfig(ENV_ID, subject=subject_config()))
    weak = make_env(
        EnvConfig(
            ENV_ID,
            subject=subject_config(),
            features=((SarcopeniaWrapper, {"force_scale": 0.5}),),
        )
    )
    weaker = make_env(
        EnvConfig(
            ENV_ID,
            subject=subject_config(),
            features=((SarcopeniaWrapper, {"force_scale": 0.4}),),
        )
    )
    try:
        write_manifest(tmp_path, ENV_ID, env=weak)
        assert check_contract(ENV_ID, tmp_path, env=weak, strict=True)
        for env in (ordinary, weaker):
            with pytest.raises(ValueError, match="morphology"):
                check_contract(ENV_ID, tmp_path, env=env, strict=True)
    finally:
        ordinary.close()
        weak.close()
        weaker.close()


def test_export_bundle_extract_and_onnx_loading_preserve_morphology(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from myosuite.utils.onnx_checkpoint import (
        bundle_onnx_with_checkpoint,
        embed_checkpoint_manifest,
        extract_checkpoint_from_onnx,
        read_onnx_manifest,
    )

    env = make_env(EnvConfig(ENV_ID, subject=subject_config()))
    nominal = make_env(ENV_ID)
    try:
        checkpoint = tmp_path / "model_0.pt"
        checkpoint.write_bytes(b"native state")
        write_manifest(tmp_path, ENV_ID, env=env)
        path = tmp_path / "policy.onnx"
        policy = torch.nn.Linear(
            env.observation_space.shape[0], env.action_space.shape[0]
        )
        torch.onnx.export(
            policy,
            torch.zeros(1, env.observation_space.shape[0]),
            str(path),
            input_names=["obs"],
            output_names=["actions"],
            dynamo=False,
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        )
        embed_checkpoint_manifest(path, checkpoint)
        assert read_onnx_manifest(path)["subject_model"] == subject_metadata(env)
        obs, _ = env.reset(seed=0)
        actions = load_policy(env, path, strict=True)(obs)
        assert actions.shape == env.action_space.shape
        with pytest.raises(ValueError, match="morphology"):
            load_policy(nominal, path, strict=True)
        bundle_onnx_with_checkpoint(path, checkpoint, framework="rslrl")
        restored, metadata, temporary = extract_checkpoint_from_onnx(path)
        try:
            assert restored.read_bytes() == checkpoint.read_bytes()
            assert check_contract(ENV_ID, restored.parent, env=env, strict=True)
            assert metadata["checkpoint_manifest"]["subject_model"] == subject_metadata(
                env
            )
        finally:
            temporary.cleanup()
    finally:
        env.close()
        nominal.close()
