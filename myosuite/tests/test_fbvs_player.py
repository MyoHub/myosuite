# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A 5.4 walker exported as a ChaseTag FBVs player (tutorial 5.6)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myosuite.envs.myo.tasks.mimic.policy import ActorCritic  # noqa: E402
from myosuite.integrations.musclemimic.fbvs_player import (  # noqa: E402
    FBVS_OBS_DIM,
    FbvsDirectionalAdapter,
    export_fbvs_onnx,
    load_fbvs_player,
)

pytestmark = pytest.mark.tier2

_SHIPPED = (
    Path(__file__).resolve().parents[2]
    / "tutorials"
    / "files"
    / "5.6"
    / "policy_heading_table.pt"
)
_BC = _SHIPPED.with_name("policy_bc_best.pt")


def _fbvs_obs(data: mujoco.MjData, opponent_xy: np.ndarray, runner: bool) -> np.ndarray:
    """An FBVs player observation whose opponent block is relative to the root."""
    rel = np.zeros(7)
    rel[:2] = opponent_xy - data.qpos[:2]
    role = np.array([0.0, 1.0]) if runner else np.array([1.0, 0.0])
    obs = np.concatenate([data.qpos, data.qvel, data.act, np.zeros(6), rel, role])
    assert obs.shape == (FBVS_OBS_DIM,)
    return obs.astype(np.float32)


@pytest.mark.parametrize("runner", [False, True])
def test_adapter_matches_chasetag_heading_directional_obs(runner: bool) -> None:
    pytest.importorskip("musclemimic_models")
    from myosuite.envs.myo.tasks.mimic.chasetag_obs import (
        chasetag_heading_directional_obs,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_musclemimic_fullbody_mjmodel,
        default_musclemimic_fullbody_config,
    )

    model, _, _ = compile_musclemimic_fullbody_mjmodel(
        default_musclemimic_fullbody_config()
    )
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1)
    adapter = FbvsDirectionalAdapter()
    for _ in range(5):
        mujoco.mj_resetData(model, data)
        quat = rng.normal(size=4)
        data.qpos[:3] = [*rng.uniform(-3, 3, 2), 0.95]
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[7:] += rng.normal(scale=0.05, size=model.nq - 7)
        data.qvel[:] = rng.normal(scale=0.5, size=model.nv)
        data.act[:] = rng.uniform(0, 1, model.na)
        mujoco.mj_forward(model, data)
        opponent_xy = rng.uniform(-5, 5, 2)
        expected = chasetag_heading_directional_obs(
            model, data, opponent_xy, evade=runner
        )
        got = adapter(torch.from_numpy(_fbvs_obs(data, opponent_xy, runner))[None])[0]
        np.testing.assert_allclose(got.numpy(), expected, atol=1e-5)


def test_shipped_walker_exports_and_onnx_matches_torch(tmp_path: Path) -> None:
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    if not _SHIPPED.is_file():
        pytest.skip("needs a source checkout")
    player = load_fbvs_player(_SHIPPED)
    path = export_fbvs_onnx(_SHIPPED, tmp_path / "player.onnx")
    obs = np.random.default_rng(0).normal(size=(4, FBVS_OBS_DIM)).astype(np.float32)
    obs[:, 3:7] /= np.linalg.norm(obs[:, 3:7], axis=1, keepdims=True)
    obs[:, 544:546] = [[1, 0], [0, 1], [1, 0], [0, 1]]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"obs": obs})[0]
    with torch.no_grad():
        torch_out = player(torch.from_numpy(obs)).numpy()
    assert onnx_out.shape == (4, 354)
    assert ((onnx_out >= 0) & (onnx_out <= 1)).all()
    np.testing.assert_allclose(onnx_out, torch_out, atol=1e-5)


def test_fbvs_trained_checkpoint_is_used_as_is(tmp_path: Path) -> None:
    ckpt = tmp_path / "fbvs.pt"
    torch.save({"model_state_dict": ActorCritic(obs_dim=546).state_dict()}, ckpt)
    player = load_fbvs_player(ckpt)
    assert player.adapter is None
    assert player(torch.zeros(2, FBVS_OBS_DIM)).shape == (2, 354)


def test_policy_key_layout_is_accepted(tmp_path: Path) -> None:
    ckpt = tmp_path / "walker.pt"
    torch.save({"policy": ActorCritic(obs_dim=528).state_dict()}, ckpt)
    assert isinstance(load_fbvs_player(ckpt).adapter, FbvsDirectionalAdapter)


def test_colab_default_bc_walker_exports(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    if not _BC.is_file():
        pytest.skip("needs a source checkout")
    player = load_fbvs_player(_BC)
    assert isinstance(player.adapter, FbvsDirectionalAdapter)
    path = export_fbvs_onnx(_BC, tmp_path / "bc.onnx")
    assert path.stat().st_size > 1000


def test_raw_528_walker_gets_the_adapter(tmp_path: Path) -> None:
    ckpt = tmp_path / "walker.pt"
    torch.save(ActorCritic(obs_dim=528).state_dict(), ckpt)
    player = load_fbvs_player(ckpt)
    assert isinstance(player.adapter, FbvsDirectionalAdapter)
    assert player(torch.zeros(2, FBVS_OBS_DIM)).shape == (2, 354)


def test_unknown_obs_dim_is_rejected(tmp_path: Path) -> None:
    ckpt = tmp_path / "odd.pt"
    torch.save(ActorCritic(obs_dim=100).state_dict(), ckpt)
    with pytest.raises(ValueError, match="obs_dim"):
        load_fbvs_player(ckpt)


def test_non_actor_checkpoint_lists_its_keys(tmp_path: Path) -> None:
    ckpt = tmp_path / "other.pt"
    torch.save({"encoder.weight": torch.zeros(2, 2)}, ckpt)
    with pytest.raises(ValueError, match="encoder.weight"):
        load_fbvs_player(ckpt)
