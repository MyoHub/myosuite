# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shipped 1.0.6 heading-table walker and tutorial 5.6 ONNX wrap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from myosuite.envs.myo.tasks.mimic.policy import (
    HeadingTablePolicy,
    load_directional_policy,
)

pytestmark = pytest.mark.tier1

_REPO = Path(__file__).resolve().parents[2]
_TABLE_CKPT = _REPO / "tutorials" / "files" / "5.6" / "policy_heading_table.pt"
_BC_CKPT = _REPO / "tutorials" / "files" / "5.6" / "policy_bc_best.pt"


@pytest.fixture(scope="module")
def table_ckpt() -> Path:
    if not _TABLE_CKPT.is_file():
        pytest.skip(f"missing shipped walker {_TABLE_CKPT}")
    return _TABLE_CKPT


def test_heading_table_checkpoint_kind(table_ckpt: Path) -> None:
    ckpt = torch.load(table_ckpt, map_location="cpu", weights_only=True)
    assert ckpt["kind"] == "heading_table"
    assert tuple(ckpt["table"].shape) == (8, 40)
    assert "trunk.0.weight" in ckpt["actor"]
    assert ckpt["actor"]["trunk.0.weight"].shape[1] == 528


def test_load_directional_policy_heading_table(table_ckpt: Path) -> None:
    pol = load_directional_policy(table_ckpt)
    assert isinstance(pol, HeadingTablePolicy)
    obs = torch.zeros(2, 528)
    obs[0, 520:522] = torch.tensor([1.0, 0.0])
    obs[1, 520:522] = torch.tensor([-1.0, 0.0])
    act = pol(obs)
    assert act.shape == (2, 354)
    # East and West pick different table rows, so left-leg commands differ.
    assert not torch.allclose(act[0, 314:354], act[1, 314:354])


def test_load_directional_policy_raw_actor() -> None:
    if not _BC_CKPT.is_file():
        pytest.skip("no raw ActorCritic fallback checkpoint")
    ckpt = torch.load(_BC_CKPT, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and ckpt.get("kind") == "heading_table":
        pytest.skip("policy_bc_best.pt is also a heading-table package")
    pol = load_directional_policy(_BC_CKPT)
    assert not isinstance(pol, HeadingTablePolicy)
    act = pol.act(np.zeros(528, dtype=np.float32))
    assert act.shape == (354,)


def test_heading_table_onnx_fbvs_wrap(table_ckpt: Path, tmp_path: Path) -> None:
    pytest.importorskip("onnx")

    class FbvsWalker(torch.nn.Module):
        def __init__(self, walker: torch.nn.Module) -> None:
            super().__init__()
            self.walker = walker

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            qpos, qvel = obs[:, 0:89], obs[:, 89:177]
            act, opp_rel, role = obs[:, 177:531], obs[:, 537:544], obs[:, 544:546]
            w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            c, s = torch.cos(-yaw), torch.sin(-yaw)
            vx, vy = qvel[:, 0], qvel[:, 1]
            delta = opp_rel[:, :2]
            evade = (role[:, 1:2] > 0.5).to(obs.dtype)
            delta = torch.where(evade > 0.5, -delta, delta)
            n = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
            roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
            wx_w, wy_w = qvel[:, 3], qvel[:, 4]
            obs528 = torch.cat(
                [
                    qpos[:, 7:89],
                    qvel[:, 6:],
                    act,
                    torch.stack([c * vx - s * vy, s * vx + c * vy], dim=-1),
                    delta / n,
                    torch.stack(
                        [
                            roll,
                            pitch,
                            c * wx_w - s * wy_w,
                            s * wx_w + c * wy_w,
                            qvel[:, 5],
                            qvel[:, 2],
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )
            return torch.clamp(self.walker(obs528), 0.0, 1.0)

    walker = load_directional_policy(table_ckpt).eval()
    wrap = FbvsWalker(walker).eval()
    out = tmp_path / "chasetag_player.onnx"
    torch.onnx.export(
        wrap,
        torch.zeros(1, 546),
        str(out),
        opset_version=17,
        input_names=["obs"],
        output_names=["action"],
        dynamo=False,
    )
    assert out.is_file() and out.stat().st_size > 1000
    dummy = torch.zeros(1, 546)
    dummy[0, 3] = 1.0  # unit quat w
    dummy[0, 537] = 1.0  # opponent +x
    dummy[0, 544] = 1.0  # chase
    act = wrap(dummy)
    assert act.shape == (1, 354)
    assert torch.isfinite(act).all()
