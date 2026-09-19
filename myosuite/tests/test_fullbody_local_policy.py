# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for local full-body policy inference helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from myosuite.core.trajectory_io import MotionClip
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    LocalPolicyArtifacts,
    LocalPolicyRunner,
    ObservationHistoryBuffer,
    OnnxPolicyRunner,
    _actor_forward,
    fullbody_history_settings_from_metadata,
    fullbody_obs_adapter_params_from_metadata,
    has_local_policy_artifacts,
    running_mean_std_update,
)


def _toy_params(obs_dim: int, hidden_dim: int, action_dim: int) -> dict[str, object]:
    z_h = np.zeros((hidden_dim,), dtype=np.float32)
    z_oh = np.zeros((obs_dim, hidden_dim), dtype=np.float32)
    z_hh = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
    z_ha = np.zeros((hidden_dim, action_dim), dtype=np.float32)
    ones_h = np.ones((hidden_dim,), dtype=np.float32)
    return {
        "actor": {
            "block0_layer0_dense": {"kernel": z_oh, "bias": z_h},
            "block0_layer0_ln": {"scale": ones_h, "bias": z_h},
            "block0_layer1_dense": {"kernel": z_hh, "bias": z_h},
            "block0_layer1_ln": {"scale": ones_h, "bias": z_h},
            "block0_proj": {"kernel": z_oh, "bias": z_h},
            "block1_layer0_dense": {"kernel": z_hh, "bias": z_h},
            "block1_layer0_ln": {"scale": ones_h, "bias": z_h},
            "block1_layer1_dense": {"kernel": z_hh, "bias": z_h},
            "block1_layer1_ln": {"scale": ones_h, "bias": z_h},
            "tail_dense": {"kernel": z_hh, "bias": z_h},
            "tail_ln": {"scale": ones_h, "bias": z_h},
            "output": {
                "kernel": z_ha,
                "bias": np.full((action_dim,), 0.25, dtype=np.float32),
            },
            "res_gate_0": np.asarray([0.0], dtype=np.float32),
            "res_gate_1": np.asarray([0.0], dtype=np.float32),
        },
        "log_std": np.full((action_dim,), -4.0, dtype=np.float32),
    }


def _toy_variable_res_params(
    obs_dim: int,
    action_dim: int,
) -> dict[str, object]:
    """Actor with four residual blocks whose widths differ across layers."""
    actor: dict[str, object] = {}
    in_dim = obs_dim
    for idx, (mid_dim, out_dim) in enumerate(((7, 9), (9, 9), (5, 4), (6, 4))):
        actor[f"block{idx}_layer0_dense"] = {
            "kernel": np.zeros((in_dim, mid_dim), dtype=np.float32),
            "bias": np.zeros((mid_dim,), dtype=np.float32),
        }
        actor[f"block{idx}_layer0_ln"] = {
            "scale": np.ones((mid_dim,), dtype=np.float32),
            "bias": np.zeros((mid_dim,), dtype=np.float32),
        }
        actor[f"block{idx}_layer1_dense"] = {
            "kernel": np.zeros((mid_dim, out_dim), dtype=np.float32),
            "bias": np.zeros((out_dim,), dtype=np.float32),
        }
        actor[f"block{idx}_layer1_ln"] = {
            "scale": np.ones((out_dim,), dtype=np.float32),
            "bias": np.zeros((out_dim,), dtype=np.float32),
        }
        if in_dim != out_dim:
            actor[f"block{idx}_proj"] = {
                "kernel": np.zeros((in_dim, out_dim), dtype=np.float32),
                "bias": np.zeros((out_dim,), dtype=np.float32),
            }
        actor[f"res_gate_{idx}"] = np.asarray([0.0], dtype=np.float32)
        in_dim = out_dim

    actor["tail_dense"] = {
        "kernel": np.zeros((in_dim, 6), dtype=np.float32),
        "bias": np.zeros((6,), dtype=np.float32),
    }
    actor["tail_ln"] = {
        "scale": np.ones((6,), dtype=np.float32),
        "bias": np.zeros((6,), dtype=np.float32),
    }
    actor["output"] = {
        "kernel": np.zeros((6, action_dim), dtype=np.float32),
        "bias": np.full((action_dim,), 0.2, dtype=np.float32),
    }
    return {
        "actor": actor,
        "log_std": np.full((action_dim,), -4.0, dtype=np.float32),
    }


def _toy_mjx_mlp_params(
    obs_dim: int,
    hidden_dim: int,
    action_dim: int,
) -> dict[str, object]:
    z_h = np.zeros((hidden_dim,), dtype=np.float32)
    z_oh = np.zeros((obs_dim, hidden_dim), dtype=np.float32)
    z_hh = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
    z_ha = np.zeros((hidden_dim, action_dim), dtype=np.float32)
    ones_h = np.ones((hidden_dim,), dtype=np.float32)
    return {
        "actor": {
            "Dense_0": {"kernel": z_oh, "bias": z_h},
            "Dense_1": {"kernel": z_hh, "bias": z_h},
            "Dense_2": {
                "kernel": z_ha,
                "bias": np.full((action_dim,), 0.15, np.float32),
            },
            "LayerNorm_0": {"scale": ones_h, "bias": z_h},
            "LayerNorm_1": {"scale": ones_h, "bias": z_h},
        },
        "log_std": np.full((action_dim,), -4.0, dtype=np.float32),
    }


def _random_variable_res_params(
    obs_dim: int,
    action_dim: int,
    *,
    seed: int = 123,
) -> dict[str, object]:
    """Non-zero residual actor fixture for NumPy/Torch parity tests."""
    rng = np.random.default_rng(seed)
    actor: dict[str, object] = {}
    in_dim = obs_dim
    for idx, (mid_dim, out_dim) in enumerate(((5, 7), (6, 7), (4, 3))):
        actor[f"block{idx}_layer0_dense"] = {
            "kernel": rng.normal(0.0, 0.2, (in_dim, mid_dim)).astype(np.float32),
            "bias": rng.normal(0.0, 0.1, (mid_dim,)).astype(np.float32),
        }
        actor[f"block{idx}_layer0_ln"] = {
            "scale": rng.uniform(0.7, 1.3, (mid_dim,)).astype(np.float32),
            "bias": rng.normal(0.0, 0.05, (mid_dim,)).astype(np.float32),
        }
        actor[f"block{idx}_layer1_dense"] = {
            "kernel": rng.normal(0.0, 0.2, (mid_dim, out_dim)).astype(np.float32),
            "bias": rng.normal(0.0, 0.1, (out_dim,)).astype(np.float32),
        }
        actor[f"block{idx}_layer1_ln"] = {
            "scale": rng.uniform(0.7, 1.3, (out_dim,)).astype(np.float32),
            "bias": rng.normal(0.0, 0.05, (out_dim,)).astype(np.float32),
        }
        if in_dim != out_dim:
            actor[f"block{idx}_proj"] = {
                "kernel": rng.normal(0.0, 0.2, (in_dim, out_dim)).astype(np.float32),
                "bias": rng.normal(0.0, 0.1, (out_dim,)).astype(np.float32),
            }
        actor[f"res_gate_{idx}"] = rng.normal(0.0, 0.5, (1,)).astype(np.float32)
        in_dim = out_dim

    actor["output"] = {
        "kernel": rng.normal(0.0, 0.2, (in_dim, action_dim)).astype(np.float32),
        "bias": rng.normal(0.0, 0.1, (action_dim,)).astype(np.float32),
    }
    return {
        "actor": actor,
        "log_std": np.full((action_dim,), -4.0, dtype=np.float32),
    }


def test_has_local_policy_artifacts_flags_layout(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    assert not has_local_policy_artifacts(root)
    (root / "train_state").mkdir(parents=True)
    assert not has_local_policy_artifacts(root)
    (root / "config").mkdir(parents=True)
    (root / "config" / "metadata").write_text("{}", encoding="utf-8")
    assert has_local_policy_artifacts(root)


def test_fullbody_metadata_helpers_preserve_variant_observation_flags() -> None:
    metadata = {
        "experiment": {
            "len_obs_history": 4,
            "split_goal": True,
            "env_params": {
                "enable_muscle_length_observations": False,
                "enable_muscle_velocity_observations": True,
                "enable_touch_sensor_observations": False,
                "goal_params": {
                    "n_step_lookahead": 3,
                    "sites_for_mimic": ["pelvis_mimic", "head_mimic"],
                },
            },
        }
    }

    adapter_params = fullbody_obs_adapter_params_from_metadata(metadata)
    assert adapter_params["n_step_lookahead"] == 3
    assert adapter_params["sites_for_mimic"] == ["pelvis_mimic", "head_mimic"]
    assert adapter_params["enable_muscle_length_observations"] is False
    assert adapter_params["enable_muscle_velocity_observations"] is True
    assert adapter_params["enable_touch_sensor_observations"] is False

    history = fullbody_history_settings_from_metadata(metadata)
    assert history == {"len_obs_history": 4, "split_goal": True}


def test_running_mean_std_update_matches_upstream_formula() -> None:
    obs = np.asarray([[2.0, -1.0], [4.0, 3.0]], dtype=np.float32)
    mean = np.asarray([1.0, 2.0], dtype=np.float32)
    var = np.asarray([0.5, 4.0], dtype=np.float32)
    count = np.asarray(8.0, dtype=np.float32)

    norm, new_mean, new_var, new_count = running_mean_std_update(
        obs=obs,
        mean=mean,
        var=var,
        count=count,
    )

    batch_mean = obs.mean(axis=0)
    batch_var = obs.var(axis=0) + 1e-6
    expected_count = count + 2.0
    delta = batch_mean - mean
    expected_mean = mean + delta * 2.0 / expected_count
    expected_var = (
        var * count + batch_var * 2.0 + np.square(delta) * count * 2.0 / expected_count
    ) / expected_count
    expected_norm = (obs - expected_mean) / np.sqrt(expected_var + 1e-8)

    np.testing.assert_allclose(new_count, expected_count, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(new_mean, expected_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(new_var, expected_var, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(norm, expected_norm, rtol=1e-6, atol=1e-6)


def test_observation_history_buffer_matches_upstream_layout() -> None:
    buffer = ObservationHistoryBuffer(3)
    out0 = buffer.reset(np.asarray([1.0, 2.0], dtype=np.float32))
    out1 = buffer.step(np.asarray([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(
        out0,
        np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        out1,
        np.asarray([0.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )

    split = ObservationHistoryBuffer(
        3,
        split_goal=True,
        state_indices=np.asarray([0, 1]),
        goal_indices=np.asarray([2, 3]),
    )
    split.reset(np.asarray([1.0, 2.0, 10.0, 20.0], dtype=np.float32))
    out = split.step(np.asarray([3.0, 4.0, 30.0, 40.0], dtype=np.float32))
    np.testing.assert_allclose(
        out,
        np.asarray([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 30.0, 40.0], dtype=np.float32),
    )


def test_local_policy_runner_produces_action_vector() -> None:
    obs_dim = 8
    action_dim = 4
    artifacts = LocalPolicyArtifacts(
        params=_toy_params(
            obs_dim=obs_dim,
            hidden_dim=6,
            action_dim=action_dim,
        ),
        obs_mean=np.zeros((obs_dim,), dtype=np.float32),
        obs_var=np.ones((obs_dim,), dtype=np.float32),
        obs_count=np.asarray(1e-6, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    runner = LocalPolicyRunner(artifacts=artifacts, stochastic=False, seed=0)
    data = SimpleNamespace(
        qpos=np.asarray([0.0, 0.0], dtype=np.float32),
        qvel=np.asarray([0.0, 0.0], dtype=np.float32),
        act=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    clip = MotionClip(
        qpos=np.zeros((3, 2), dtype=np.float32),
        qvel=np.zeros((3, 2), dtype=np.float32),
        site_xpos=None,
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )
    action = runner.action_for(data=data, clip=clip, frame_idx=0)
    assert action.shape == (action_dim,)
    np.testing.assert_allclose(
        action,
        np.full((action_dim,), 0.25, dtype=np.float32),
    )


def test_local_policy_runner_action_trace_exposes_parity_tensors() -> None:
    obs_dim = 8
    action_dim = 4
    artifacts = LocalPolicyArtifacts(
        params=_toy_params(
            obs_dim=obs_dim,
            hidden_dim=6,
            action_dim=action_dim,
        ),
        obs_mean=np.linspace(-0.2, 0.3, obs_dim, dtype=np.float32),
        obs_var=np.linspace(0.8, 1.4, obs_dim, dtype=np.float32),
        obs_count=np.asarray(7.0, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    data = SimpleNamespace(
        qpos=np.asarray([0.25, -0.5], dtype=np.float32),
        qvel=np.asarray([0.1, -0.2], dtype=np.float32),
        act=np.asarray([0.05, 0.1, 0.15, 0.2], dtype=np.float32),
    )
    clip = MotionClip(
        qpos=np.zeros((3, 2), dtype=np.float32),
        qvel=np.zeros((3, 2), dtype=np.float32),
        site_xpos=None,
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )
    trace_runner = LocalPolicyRunner(artifacts=artifacts, stochastic=False, seed=0)
    action_runner = LocalPolicyRunner(artifacts=artifacts, stochastic=False, seed=0)

    trace = trace_runner.action_trace_for(data=data, clip=clip, frame_idx=0)
    action = action_runner.action_for(data=data, clip=clip, frame_idx=0)
    expected_norm, expected_mean, expected_var, expected_count = (
        running_mean_std_update(
            obs=trace.policy_obs,
            mean=trace.run_mean_before,
            var=trace.run_var_before,
            count=trace.run_count_before,
        )
    )

    assert trace.raw_obs.shape == (obs_dim,)
    assert trace.policy_obs.shape == (obs_dim,)
    assert trace.norm_obs.shape == (obs_dim,)
    assert trace.mean_action.shape == (action_dim,)
    np.testing.assert_allclose(trace.norm_obs, expected_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        trace.run_mean_after, expected_mean, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(trace.run_var_after, expected_var, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        trace.run_count_after, expected_count, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        trace.mean_action,
        _actor_forward(artifacts.params, trace.norm_obs),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(trace.action, action, rtol=1e-6, atol=1e-6)


def test_local_policy_runner_supports_mjx_dense_actor_layout() -> None:
    obs_dim = 8
    action_dim = 4
    artifacts = LocalPolicyArtifacts(
        params=_toy_mjx_mlp_params(
            obs_dim=obs_dim,
            hidden_dim=6,
            action_dim=action_dim,
        ),
        obs_mean=np.zeros((obs_dim,), dtype=np.float32),
        obs_var=np.ones((obs_dim,), dtype=np.float32),
        obs_count=np.asarray(1e-6, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    runner = LocalPolicyRunner(artifacts=artifacts, stochastic=False, seed=0)
    data = SimpleNamespace(
        qpos=np.asarray([0.0, 0.0], dtype=np.float32),
        qvel=np.asarray([0.0, 0.0], dtype=np.float32),
        act=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    clip = MotionClip(
        qpos=np.zeros((3, 2), dtype=np.float32),
        qvel=np.zeros((3, 2), dtype=np.float32),
        site_xpos=None,
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )
    action = runner.action_for(data=data, clip=clip, frame_idx=0)
    assert action.shape == (action_dim,)
    np.testing.assert_allclose(
        action,
        np.full((action_dim,), 0.15, dtype=np.float32),
    )


def test_local_policy_runner_applies_upstream_history_before_inference() -> None:
    raw_obs_dim = 4
    action_dim = 2
    artifacts = LocalPolicyArtifacts(
        params=_toy_mjx_mlp_params(
            obs_dim=raw_obs_dim * 3,
            hidden_dim=5,
            action_dim=action_dim,
        ),
        obs_mean=np.zeros((raw_obs_dim * 3,), dtype=np.float32),
        obs_var=np.ones((raw_obs_dim * 3,), dtype=np.float32),
        obs_count=np.asarray(1e-6, dtype=np.float32),
        obs_dim=raw_obs_dim * 3,
        action_dim=action_dim,
    )

    class _ObsAdapter:
        def __init__(self) -> None:
            self.raw_obs = [
                np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            ]

        def build(self, data: object, frame_idx: int) -> np.ndarray:
            del data
            return self.raw_obs[frame_idx]

    runner = LocalPolicyRunner(
        artifacts=artifacts,
        stochastic=False,
        seed=0,
        obs_adapter=_ObsAdapter(),  # type: ignore[arg-type]
        len_obs_history=3,
    )
    data = SimpleNamespace()
    clip = MotionClip(
        qpos=np.zeros((3, 1), dtype=np.float32),
        qvel=np.zeros((3, 1), dtype=np.float32),
        site_xpos=None,
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )

    runner.action_for(data=data, clip=clip, frame_idx=0)
    expected0 = np.asarray(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )
    np.testing.assert_allclose(runner._run_mean, expected0, rtol=1e-5, atol=1e-5)

    runner.action_for(data=data, clip=clip, frame_idx=1)
    expected1_batch = np.asarray(
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        dtype=np.float32,
    )
    expected_mean_after_second = (expected0 + expected1_batch) / 2.0
    np.testing.assert_allclose(
        runner._run_mean,
        expected_mean_after_second,
        rtol=1e-5,
        atol=1e-5,
    )


def test_onnx_policy_runner_applies_upstream_split_goal_history(monkeypatch) -> None:
    raw_obs_dim = 4
    goal_dim = 2
    len_history = 3
    policy_obs_dim = (raw_obs_dim - goal_dim) * len_history + goal_dim
    action_dim = 2

    class _FakeInput:
        name = "obs"
        shape = [None, policy_obs_dim]

    class _FakeOutput:
        name = "actions"
        shape = [None, action_dim]

    class _FakeSession:
        def __init__(self) -> None:
            self.seen: list[np.ndarray] = []

        def get_inputs(self) -> list[_FakeInput]:
            return [_FakeInput()]

        def get_outputs(self) -> list[_FakeOutput]:
            return [_FakeOutput()]

        def run(
            self, names: list[str], feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            assert names == ["actions"]
            obs = np.asarray(feed["obs"], dtype=np.float32)
            self.seen.append(obs.copy())
            return [np.zeros((obs.shape[0], action_dim), dtype=np.float32)]

    fake_session = _FakeSession()
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=lambda *args, **kwargs: fake_session),
    )

    class _SplitGoalObsAdapter:
        def __init__(self) -> None:
            self.goal_dim = goal_dim

        def goal_indices_for_obs_dim(self, obs_dim: int) -> np.ndarray:
            return np.arange(obs_dim - goal_dim, obs_dim, dtype=int)

        def state_indices_for_obs_dim(self, obs_dim: int) -> np.ndarray:
            mask = np.ones(obs_dim, dtype=bool)
            mask[self.goal_indices_for_obs_dim(obs_dim)] = False
            return np.arange(obs_dim, dtype=int)[mask]

        def build(self, data: object, frame_idx: int) -> np.ndarray:
            del data
            return np.asarray(
                [1.0 + 2.0 * frame_idx, 2.0 + 2.0 * frame_idx, 10.0, 20.0],
                dtype=np.float32,
            )

    runner = OnnxPolicyRunner(
        onnx_path="/tmp/fake.onnx",
        obs_dim=policy_obs_dim,
        action_dim=action_dim,
        obs_adapter=_SplitGoalObsAdapter(),  # type: ignore[arg-type]
        len_obs_history=len_history,
        split_goal=True,
    )
    clip = MotionClip(
        qpos=np.zeros((3, 1), dtype=np.float32),
        qvel=np.zeros((3, 1), dtype=np.float32),
        site_xpos=None,
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )

    runner.action_for(SimpleNamespace(), clip, frame_idx=0)
    runner.action_for(SimpleNamespace(), clip, frame_idx=1)

    np.testing.assert_allclose(
        fake_session.seen[0][0],
        np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 10.0, 20.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        fake_session.seen[1][0],
        np.asarray([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0], dtype=np.float32),
    )


def test_local_policy_runner_supports_variable_residual_blocks() -> None:
    obs_dim = 10
    action_dim = 3
    artifacts = LocalPolicyArtifacts(
        params=_toy_variable_res_params(obs_dim=obs_dim, action_dim=action_dim),
        obs_mean=np.zeros((obs_dim,), dtype=np.float32),
        obs_var=np.ones((obs_dim,), dtype=np.float32),
        obs_count=np.asarray(1e-6, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    runner = LocalPolicyRunner(artifacts=artifacts, stochastic=False, seed=0)
    data = SimpleNamespace(
        qpos=np.zeros((2,), dtype=np.float32),
        qvel=np.zeros((2,), dtype=np.float32),
        act=np.zeros((2,), dtype=np.float32),
    )
    clip = MotionClip(
        qpos=np.zeros((3, 2), dtype=np.float32),
        qvel=np.zeros((3, 2), dtype=np.float32),
        site_xpos=None,
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )
    action = runner.action_for(data=data, clip=clip, frame_idx=0)
    assert action.shape == (action_dim,)
    np.testing.assert_allclose(
        action,
        np.full((action_dim,), 0.2, dtype=np.float32),
    )


def test_torch_actor_supports_variable_residual_blocks() -> None:
    import torch

    from myosuite.integrations.musclemimic.actor_torch import MimicActorModule

    obs_dim = 10
    action_dim = 3
    artifacts = LocalPolicyArtifacts(
        params=_toy_variable_res_params(obs_dim=obs_dim, action_dim=action_dim),
        obs_mean=np.zeros((obs_dim,), dtype=np.float32),
        obs_var=np.ones((obs_dim,), dtype=np.float32),
        obs_count=np.asarray(1e-6, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    actor = MimicActorModule.from_artifacts(artifacts)
    action = actor(torch.zeros((2, obs_dim), dtype=torch.float32))
    assert tuple(action.shape) == (2, action_dim)
    torch.testing.assert_close(
        action,
        torch.full((2, action_dim), 0.2, dtype=torch.float32),
    )


def test_torch_actor_matches_numpy_actor_for_nonzero_residual_layout() -> None:
    import torch

    from myosuite.integrations.musclemimic.actor_torch import (
        make_actor_module,
    )

    obs_dim = 6
    action_dim = 3
    rng = np.random.default_rng(7)
    artifacts = LocalPolicyArtifacts(
        params=_random_variable_res_params(obs_dim=obs_dim, action_dim=action_dim),
        obs_mean=np.zeros((obs_dim,), dtype=np.float32),
        obs_var=np.ones((obs_dim,), dtype=np.float32),
        obs_count=np.asarray(1e-6, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    norm_obs = rng.normal(0.0, 1.0, (4, obs_dim)).astype(np.float32)

    expected = _actor_forward(artifacts.params, norm_obs)
    actor = make_actor_module(artifacts)
    actual = actor.forward_normalized(torch.as_tensor(norm_obs)).detach().numpy()

    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=5e-6)


def test_torch_actor_factory_supports_dense_layout() -> None:
    import torch

    from myosuite.integrations.musclemimic.actor_torch import (
        DenseActorModule,
        make_actor_module,
    )

    obs_dim = 8
    action_dim = 4
    artifacts = LocalPolicyArtifacts(
        params=_toy_mjx_mlp_params(
            obs_dim=obs_dim,
            hidden_dim=6,
            action_dim=action_dim,
        ),
        obs_mean=np.zeros((obs_dim,), dtype=np.float32),
        obs_var=np.ones((obs_dim,), dtype=np.float32),
        obs_count=np.asarray(1e-6, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    actor = make_actor_module(artifacts)
    assert isinstance(actor, DenseActorModule)
    assert actor.__class__.__name__ == "DenseActorModule"
    action = actor(torch.zeros((2, obs_dim), dtype=torch.float32))
    assert tuple(action.shape) == (2, action_dim)
    torch.testing.assert_close(
        action,
        torch.full((2, action_dim), 0.15, dtype=torch.float32),
    )


def test_musclemimic_citation_metadata_exported() -> None:
    from myosuite.integrations import (
        INTEGRATION_CITATIONS_BY_KEY,
        get_integration_citation,
    )
    from myosuite.integrations.musclemimic import (
        MUSCLEMIMIC_ARXIV_URL,
        MUSCLEMIMIC_CITATION,
        MUSCLEMIMIC_CITATION_BIBTEX,
        MUSCLEMIMIC_PROJECT_URL,
    )

    assert MUSCLEMIMIC_CITATION is get_integration_citation("musclemimic")
    assert INTEGRATION_CITATIONS_BY_KEY["musclemimic"] is MUSCLEMIMIC_CITATION
    assert MUSCLEMIMIC_PROJECT_URL == "https://github.com/amathislab/musclemimic"
    assert MUSCLEMIMIC_ARXIV_URL == "https://arxiv.org/abs/2603.25544"
    assert "Li2026MuscleMimic" in MUSCLEMIMIC_CITATION_BIBTEX


def test_orbax_mjlab_policy_bridge_returns_env_action_tensor() -> None:
    import mujoco
    import torch

    from myosuite.integrations.musclemimic.mjlab_onnx_policy import (
        FullbodyOrbaxMjlabPolicy,
    )

    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body name="body">
          <joint name="joint" type="hinge"/>
          <geom type="capsule" size="0.01 0.05"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor" joint="joint" ctrllimited="true" ctrlrange="0 1"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    obs_dim = 5
    artifacts = LocalPolicyArtifacts(
        params=_toy_params(obs_dim=obs_dim, hidden_dim=4, action_dim=model.nu),
        obs_mean=np.zeros((obs_dim,), dtype=np.float32),
        obs_var=np.ones((obs_dim,), dtype=np.float32),
        obs_count=np.asarray(1e-6, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=model.nu,
    )

    class _ObsAdapter:
        def __init__(self) -> None:
            self.frames: list[int] = []

        def build(self, data: object, frame_idx: int) -> np.ndarray:
            self.frames.append(frame_idx)
            return np.zeros((obs_dim,), dtype=np.float32)

    class _FakeEnv:
        def __init__(self) -> None:
            self.num_envs = 2
            self.device = torch.device("cpu")
            self.physics_dt = 0.01
            self.cfg = SimpleNamespace(decimation=2)
            self.scene = {}
            self.sim = SimpleNamespace(
                data=SimpleNamespace(
                    qpos=torch.zeros((2, model.nq), dtype=torch.float32),
                    qvel=torch.zeros((2, model.nv), dtype=torch.float32),
                    ctrl=torch.zeros((2, model.nu), dtype=torch.float32),
                    act=torch.zeros((2, model.na), dtype=torch.float32),
                    time=torch.zeros((2,), dtype=torch.float32),
                ),
            )

    clip = MotionClip(
        qpos=np.zeros((3, model.nq), dtype=np.float32),
        qvel=np.zeros((3, model.nv), dtype=np.float32),
        site_xpos=None,
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )
    adapter = _ObsAdapter()
    policy = FullbodyOrbaxMjlabPolicy(
        env=_FakeEnv(),
        cpu_model=model,
        obs_adapter=adapter,  # type: ignore[arg-type]
        clip=clip,
        artifacts=artifacts,
        output_ctrl=True,
    )

    action = policy(torch.zeros((2, 1), dtype=torch.float32))

    assert tuple(action.shape) == (2, model.nu)
    torch.testing.assert_close(
        action,
        torch.full((2, model.nu), 0.25, dtype=torch.float32),
    )
    assert adapter.frames == [0, 0]


def test_orbax_mjlab_policy_derives_split_goal_history_indices() -> None:
    import mujoco
    import torch

    from myosuite.integrations.musclemimic.mjlab_onnx_policy import (
        FullbodyOrbaxMjlabPolicy,
    )

    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body name="body">
          <joint name="joint" type="hinge"/>
          <geom type="capsule" size="0.01 0.05"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor" joint="joint" ctrllimited="true" ctrlrange="-1 1"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    raw_obs_dim = 6
    goal_dim = 2
    len_history = 3
    policy_obs_dim = (raw_obs_dim - goal_dim) * len_history + goal_dim
    artifacts = LocalPolicyArtifacts(
        params=_toy_mjx_mlp_params(
            obs_dim=policy_obs_dim,
            hidden_dim=5,
            action_dim=model.nu,
        ),
        obs_mean=np.zeros((policy_obs_dim,), dtype=np.float32),
        obs_var=np.ones((policy_obs_dim,), dtype=np.float32),
        obs_count=np.asarray(1e-6, dtype=np.float32),
        obs_dim=policy_obs_dim,
        action_dim=model.nu,
    )

    class _SplitGoalObsAdapter:
        def __init__(self) -> None:
            self.goal_dim = goal_dim
            self.frames: list[int] = []
            self.derived_dims: list[int] = []

        def goal_indices_for_obs_dim(self, obs_dim: int) -> np.ndarray:
            self.derived_dims.append(obs_dim)
            return np.arange(obs_dim - self.goal_dim, obs_dim, dtype=int)

        def state_indices_for_obs_dim(self, obs_dim: int) -> np.ndarray:
            mask = np.ones(obs_dim, dtype=bool)
            mask[self.goal_indices_for_obs_dim(obs_dim)] = False
            return np.arange(obs_dim, dtype=int)[mask]

        def build(self, data: object, frame_idx: int) -> np.ndarray:
            del data
            self.frames.append(frame_idx)
            return np.asarray([1.0, 2.0, 3.0, 4.0, 10.0, 20.0], dtype=np.float32)

    class _FakeSim:
        def __init__(self) -> None:
            self.data = SimpleNamespace(
                qpos=torch.zeros((1, model.nq), dtype=torch.float32),
                qvel=torch.zeros((1, model.nv), dtype=torch.float32),
                ctrl=torch.zeros((1, model.nu), dtype=torch.float32),
                act=torch.zeros((1, model.na), dtype=torch.float32),
                time=torch.zeros((1,), dtype=torch.float32),
            )

        def forward(self) -> None:
            pass

    class _FakeEnv:
        def __init__(self) -> None:
            self.num_envs = 1
            self.device = torch.device("cpu")
            self.physics_dt = None
            self.cfg = SimpleNamespace(decimation=1)
            self.sim = _FakeSim()
            self.scene = {
                "mimic_fullbody_robot": SimpleNamespace(
                    data=SimpleNamespace(data=self.sim.data)
                )
            }

    clip = MotionClip(
        qpos=np.zeros((4, model.nq), dtype=np.float32),
        qvel=np.zeros((4, model.nv), dtype=np.float32),
        site_xpos=None,
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )
    adapter = _SplitGoalObsAdapter()
    policy = FullbodyOrbaxMjlabPolicy(
        env=_FakeEnv(),
        cpu_model=model,
        obs_adapter=adapter,  # type: ignore[arg-type]
        clip=clip,
        artifacts=artifacts,
        output_ctrl=False,
        len_obs_history=len_history,
        split_goal=True,
    )

    action = policy(torch.zeros((1, 1), dtype=torch.float32))

    assert tuple(action.shape) == (1, model.nu)
    assert adapter.frames == [0]
    assert adapter.derived_dims == [raw_obs_dim, raw_obs_dim]
    assert policy._history is not None
    assert policy._history.split_goal is True


def test_orbax_mjlab_policy_reset_preserves_fallback_frame() -> None:
    import mujoco
    import torch

    from myosuite.integrations.musclemimic.mjlab_onnx_policy import (
        FullbodyOrbaxMjlabPolicy,
    )

    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body name="body">
          <joint name="joint" type="hinge"/>
          <geom type="capsule" size="0.01 0.05"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor" joint="joint" ctrllimited="true" ctrlrange="-1 1"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    obs_dim = 5
    artifacts = LocalPolicyArtifacts(
        params=_toy_params(obs_dim=obs_dim, hidden_dim=4, action_dim=model.nu),
        obs_mean=np.zeros((obs_dim,), dtype=np.float32),
        obs_var=np.ones((obs_dim,), dtype=np.float32),
        obs_count=np.asarray(1e-6, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=model.nu,
    )

    class _ObsAdapter:
        def __init__(self) -> None:
            self.frames: list[int] = []

        def build(self, data: object, frame_idx: int) -> np.ndarray:
            del data
            self.frames.append(frame_idx)
            return np.zeros((obs_dim,), dtype=np.float32)

    class _FakeSim:
        def __init__(self) -> None:
            self.data = SimpleNamespace(
                qpos=torch.zeros((1, model.nq), dtype=torch.float32),
                qvel=torch.zeros((1, model.nv), dtype=torch.float32),
                ctrl=torch.zeros((1, model.nu), dtype=torch.float32),
                act=torch.zeros((1, model.na), dtype=torch.float32),
                time=torch.zeros((1,), dtype=torch.float32),
            )

        def forward(self) -> None:
            pass

    class _FakeEnv:
        def __init__(self) -> None:
            self.num_envs = 1
            self.device = torch.device("cpu")
            self.physics_dt = None
            self.cfg = SimpleNamespace(decimation=1)
            self.sim = _FakeSim()
            self.scene = {
                "mimic_fullbody_robot": SimpleNamespace(
                    data=SimpleNamespace(data=self.sim.data)
                )
            }

    clip_qpos = np.zeros((5, model.nq), dtype=np.float32)
    clip_qvel = np.zeros((5, model.nv), dtype=np.float32)
    clip_qpos[2, 0] = 0.42
    clip_qvel[2, 0] = -0.17
    clip = MotionClip(
        qpos=clip_qpos,
        qvel=clip_qvel,
        site_xpos=None,
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )
    env = _FakeEnv()
    adapter = _ObsAdapter()
    policy = FullbodyOrbaxMjlabPolicy(
        env=env,
        cpu_model=model,
        obs_adapter=adapter,  # type: ignore[arg-type]
        clip=clip,
        artifacts=artifacts,
        output_ctrl=False,
    )

    frame = policy.reset_env_to_clip_frame(2)
    action = policy(torch.zeros((1, 1), dtype=torch.float32))

    assert frame == 2
    assert adapter.frames == [2]
    assert tuple(action.shape) == (1, model.nu)
    torch.testing.assert_close(env.sim.data.qpos[0], torch.as_tensor(clip_qpos[2]))
    torch.testing.assert_close(env.sim.data.qvel[0], torch.as_tensor(clip_qvel[2]))


def test_orbax_mjlab_policy_matches_local_cpu_running_inference() -> None:
    import mujoco
    import torch

    from myosuite.integrations.musclemimic.mjlab_onnx_policy import (
        FullbodyOrbaxMjlabPolicy,
    )

    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body name="body">
          <joint name="joint" type="hinge"/>
          <geom type="capsule" size="0.01 0.05"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor" joint="joint" ctrllimited="true" ctrlrange="-1 1"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    obs_dim = 6
    artifacts = LocalPolicyArtifacts(
        params=_random_variable_res_params(obs_dim=obs_dim, action_dim=model.nu),
        obs_mean=np.linspace(-0.2, 0.3, obs_dim, dtype=np.float32),
        obs_var=np.linspace(0.7, 1.4, obs_dim, dtype=np.float32),
        obs_count=np.asarray(32.0, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=model.nu,
    )
    obs_sequence = [
        np.linspace(-1.0, 1.0, obs_dim, dtype=np.float32),
        np.linspace(0.5, -0.5, obs_dim, dtype=np.float32),
    ]

    class _SequenceObsAdapter:
        def __init__(self) -> None:
            self.frames: list[int] = []

        def build(self, data: object, frame_idx: int) -> np.ndarray:
            del data
            self.frames.append(frame_idx)
            return obs_sequence[frame_idx]

    class _FakeEnv:
        def __init__(self) -> None:
            self.num_envs = 1
            self.device = torch.device("cpu")
            self.physics_dt = 0.01
            self.cfg = SimpleNamespace(decimation=1)
            self.scene = {}
            self.sim = SimpleNamespace(
                data=SimpleNamespace(
                    qpos=torch.zeros((1, model.nq), dtype=torch.float32),
                    qvel=torch.zeros((1, model.nv), dtype=torch.float32),
                    ctrl=torch.zeros((1, model.nu), dtype=torch.float32),
                    act=torch.zeros((1, model.na), dtype=torch.float32),
                    time=torch.zeros((1,), dtype=torch.float32),
                ),
            )

    clip = MotionClip(
        qpos=np.zeros((3, model.nq), dtype=np.float32),
        qvel=np.zeros((3, model.nv), dtype=np.float32),
        site_xpos=None,
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )
    local_adapter = _SequenceObsAdapter()
    mjlab_adapter = _SequenceObsAdapter()
    local = LocalPolicyRunner(
        artifacts=artifacts,
        stochastic=False,
        seed=0,
        obs_adapter=local_adapter,  # type: ignore[arg-type]
    )
    policy = FullbodyOrbaxMjlabPolicy(
        env=_FakeEnv(),
        cpu_model=model,
        obs_adapter=mjlab_adapter,  # type: ignore[arg-type]
        clip=clip,
        artifacts=artifacts,
        output_ctrl=False,
    )

    for frame_idx in range(2):
        expected = local.action_for(
            data=SimpleNamespace(),
            clip=clip,
            frame_idx=frame_idx,
        )
        actual = policy(torch.zeros((1, 1), dtype=torch.float32)).detach().numpy()[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    assert local_adapter.frames == [0, 1]
    assert mjlab_adapter.frames == [0, 1]
