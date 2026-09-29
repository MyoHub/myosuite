# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The SB3 ONNX export must return the same action as SB3's own predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import myosuite
from myosuite.utils import gym

pytestmark = pytest.mark.tier1

_ENV_ID = "myoElbowPose1D6MRandom-v0"


@pytest.mark.parametrize("algo_name", ["PPO", "SAC", "TD3"])
@pytest.mark.parametrize("action_low", [-1.0, 0.0], ids=["box-1to1", "box0to1"])
def test_sb3_onnx_export_matches_predict(
    algo_name: str, action_low: float, tmp_path: Path
) -> None:
    """ONNX actions equal model.predict(deterministic=True) on the env's Box.

    MyoSuite muscle envs expose Box(-1, 1); the [0, 1] case (an env
    without action normalisation) exercises the rescaling of squashed actions.
    """
    sb3 = pytest.importorskip("stable_baselines3")
    ort = pytest.importorskip("onnxruntime")
    from myosuite.utils.export_onnx import export_sb3_to_onnx

    myosuite.register_all_envs()
    env = gym.make(_ENV_ID)
    if action_low != -1.0:
        env = gym.wrappers.RescaleAction(
            env, min_action=np.float32(action_low), max_action=np.float32(1.0)
        )
    try:
        obs = np.stack([env.reset(seed=s)[0] for s in range(16)]).astype(np.float32)
        algo = getattr(sb3, algo_name)
        kwargs = {} if algo_name == "PPO" else {"buffer_size": 1}
        model = algo("MlpPolicy", env, seed=0, device="cpu", **kwargs)
    finally:
        env.close()

    # Spread the output bias so actions fall below, inside and above the bounds.
    head = model.policy.action_net if algo_name == "PPO" else model.policy.actor.mu
    out_layer = [m for m in head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        out_layer.bias.copy_(torch.linspace(-2.0, 2.0, out_layer.out_features))
    expected, _ = model.predict(obs, deterministic=True)
    assert (expected < 0).any() == (action_low < 0)

    model.save(tmp_path / "model.zip")
    onnx_path = tmp_path / "model.onnx"
    export_sb3_to_onnx(
        tmp_path / "model.zip", onnx_path, obs.shape[1], expected.shape[1]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = sess.run(None, {"obs": obs})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5)
