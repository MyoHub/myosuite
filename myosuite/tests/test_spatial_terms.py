# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for spatial enclosure observation and reward terms."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from myosuite.core.protocols import PhysicsPath
from myosuite.terms.spatial import (
    boundary_proximity_obs,
    enclosure_clearance_reward,
    enclosure_contact_penalty,
)

pytestmark = pytest.mark.tier1


class _SpatialAccessor:
    def __init__(self, site_pos: Any, xp: Any, *, batched: bool) -> None:
        self._site_pos = site_pos
        self._xp = xp
        self._batched = batched

    @property
    def physics_path(self) -> PhysicsPath:
        return (
            PhysicsPath.MJLAB
            if getattr(self._xp, "__name__", "") == "torch"
            else PhysicsPath.CPU
        )

    def joint_pos(self) -> Any:
        if getattr(self._xp, "__name__", "") == "torch":
            n = self._site_pos.shape[0]
            return self._xp.zeros((n, 4), dtype=self._site_pos.dtype)
        return np.zeros((self._site_pos.shape[0], 4)) if self._batched else np.zeros(4)

    def joint_vel(self) -> Any:
        return self.joint_pos()

    def muscle_act(self) -> Any:
        return self.joint_pos()

    def site_xpos(self, site_ids: Any) -> Any:
        if isinstance(site_ids, int):
            return self._site_pos[..., site_ids, :]
        return self._site_pos[..., list(site_ids), :]

    def time(self) -> float:
        return 0.5

    def ctrl_range(self) -> Any:
        return self._xp.zeros((4, 2))

    def joint_limits(self) -> tuple[Any, Any, Any]:
        z = self._xp.zeros(4)
        return z, z, self._xp.ones(4, dtype=bool)

    def dt(self) -> float:
        return 0.02

    def array_module(self) -> Any:
        return self._xp

    def site_id(self, name: str) -> int:
        del name
        return 0


def test_boundary_proximity_obs_unbatched_and_signed() -> None:
    # Two sites in [-0.5, 0.5]^3: site 0 at center (clearance 0.5), site 1 outside +x at 0.6 (clearance -0.1)
    sites = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float64)
    acc = _SpatialAccessor(sites, np, batched=False)

    obs = boundary_proximity_obs(
        acc,
        site_ids=[0, 1],
        bounds_low=(-0.5, -0.5, -0.5),
        bounds_high=(0.5, 0.5, 0.5),
        soft_margin=0.2,
    )
    assert obs.dtype == np.float32
    assert obs.shape == (12,)
    # Site 0 is 0.5 m from every wall (> soft_margin=0.2) -> proximity 0.0
    np.testing.assert_allclose(obs[:6], np.zeros(6, dtype=np.float32))
    # Site 1 upper-x clearance is 0.5 - 0.6 = -0.1 -> proximity 1 - (-0.1)/0.2 = 1.5 (unclipped!)
    assert float(obs[9]) == pytest.approx(1.5)

    signed = boundary_proximity_obs(
        acc,
        site_ids=1,
        bounds_low=(-0.5, -0.5, -0.5),
        bounds_high=(0.5, 0.5, 0.5),
        signed=True,
        plane_normals=[[0.0, 0.0, -1.0]],
        plane_offsets=[-0.4],
    )
    assert signed.shape == (7,)
    assert float(signed[3]) == pytest.approx(-0.1)
    # Plane (0, 0, -1) . (0.6, 0, 0) - (-0.4) = +0.4
    assert float(signed[6]) == pytest.approx(0.4)

    with pytest.raises(ValueError, match="soft_margin"):
        boundary_proximity_obs(
            acc,
            site_ids=0,
            bounds_low=(-0.5, -0.5, -0.5),
            bounds_high=(0.5, 0.5, 0.5),
            soft_margin=0.0,
        )


def test_enclosure_clearance_reward_cpu_and_torch_batch_parity() -> None:
    # Batch of 3 envs, 2 sites:
    # Env 0: well inside -> solved=True, dense=0.0
    # Env 1: inside but within margin (clearance=0.02 < margin=0.05) -> solved=True, dense < 0
    # Env 2: outside boundary (clearance=-0.05) + excess contact force -> solved=False, done=True
    sites_np = np.array(
        [
            [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0]],
            [[0.48, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.55, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    forces_np = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [4.0, 3.0, 0.0]], dtype=np.float32
    )
    acc_np = _SpatialAccessor(sites_np, np, batched=True)
    state_np = {
        "enclosure_site_ids": [0, 1],
        "bounds_low": (-0.5, -0.5, -0.5),
        "bounds_high": (0.5, 0.5, 0.5),
        "enclosure_contact_force": forces_np,
    }
    res_np = enclosure_clearance_reward(
        acc_np,
        state_np,
        margin=0.05,
        violation_weight=4.0,
        contact_weight=2.0,
        contact_threshold=1.0,
        terminate_on_violation=True,
    )
    assert res_np["dense"].shape == (3,)
    assert float(res_np["dense"][0]) == pytest.approx(0.0)
    assert float(res_np["dense"][1]) < 0.0
    assert float(res_np["contact_penalty"][2]) == pytest.approx(-2.0 * (5.0 - 1.0))
    np.testing.assert_array_equal(res_np["solved"], [True, True, False])
    np.testing.assert_array_equal(res_np["done"], [False, False, True])
    assert enclosure_contact_penalty is enclosure_clearance_reward

    torch = pytest.importorskip("torch")
    acc_t = _SpatialAccessor(torch.tensor(sites_np), torch, batched=True)
    state_t = {
        "enclosure_site_ids": [0, 1],
        "bounds_low": (-0.5, -0.5, -0.5),
        "bounds_high": (0.5, 0.5, 0.5),
        "enclosure_contact_force": torch.tensor(forces_np),
    }
    obs_np = boundary_proximity_obs(
        acc_np,
        site_ids=[0, 1],
        bounds_low=(-0.5, -0.5, -0.5),
        bounds_high=(0.5, 0.5, 0.5),
    )
    obs_t = boundary_proximity_obs(
        acc_t,
        site_ids=[0, 1],
        bounds_low=(-0.5, -0.5, -0.5),
        bounds_high=(0.5, 0.5, 0.5),
    )
    np.testing.assert_allclose(obs_np, obs_t.numpy(), atol=1e-6)

    res_t = enclosure_clearance_reward(
        acc_t,
        state_t,
        margin=0.05,
        violation_weight=4.0,
        contact_weight=2.0,
        contact_threshold=1.0,
        terminate_on_violation=True,
    )
    np.testing.assert_allclose(res_np["dense"], res_t["dense"].numpy(), atol=1e-6)
    np.testing.assert_array_equal(res_np["solved"], res_t["solved"].numpy())
    np.testing.assert_array_equal(res_np["done"], res_t["done"].numpy())
