# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab TableTennis must simulate and score the CPU task of the same env id.

All runs use ``device="cpu"`` and one or two envs.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tier2

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

import myosuite  # noqa: E402, F401

_P0 = "myoChallengeTableTennisP0-v0"


def _make_env(env_id: str, num_envs: int) -> ManagerBasedRlEnv:
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415 (registers tasks)

    cfg = load_env_cfg(env_id)
    cfg.scene.num_envs = num_envs
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


@pytest.fixture(scope="module")
def p0() -> ManagerBasedRlEnv:
    env = _make_env(_P0, 1)
    yield env
    env.close()


@pytest.fixture(scope="module")
def cpu_p0() -> gym.Env:
    env = gym.make(_P0).unwrapped
    env.reset(seed=0)
    return env


def _geom_id(env: ManagerBasedRlEnv, short: str) -> int:
    model = env.sim.mj_model
    ids = [i for i in range(model.ngeom) if model.geom(i).name.split("/")[-1] == short]
    assert len(ids) == 1, short
    return ids[0]


def _touching_slice(cpu: gym.Env) -> slice:
    """Index range of ``touching_info`` in the observation vector."""
    start = 0
    for key in cpu.obs_keys:
        size = np.atleast_1d(cpu.obs_dict[key]).size
        if key == "touching_info":
            return slice(start, start + size)
        start += size
    raise KeyError("touching_info")


def _step(env: ManagerBasedRlEnv) -> tuple[float, bool, bool]:
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    _, rew, term, trunc, _ = env.step(action)
    return float(rew[0]), bool(term[0]), bool(trunc[0])


def test_stale_contact_rows_produce_no_labels(
    p0: ManagerBasedRlEnv, cpu_p0: gym.Env
) -> None:
    """Contact rows at or past ``nacon`` are left over from earlier passes."""
    p0.reset()
    _step(p0)
    data = p0.sim.data
    nacon = int(data.nacon[0])
    rows = data.contact.geom.shape[0]
    assert nacon + 8 < rows
    ball, own = _geom_id(p0, "pingpong"), _geom_id(p0, "coll_own_half")
    # A ball/own-half contact of world 0 in every row of the unused tail.
    data.contact.geom[nacon:] = torch.tensor([ball, own], dtype=data.contact.geom.dtype)
    data.contact.worldid[nacon:] = 0
    # compute_group: compute() would return the observations cached at the step.
    obs = p0.observation_manager.compute_group("policy")
    assert torch.count_nonzero(obs[0, _touching_slice(cpu_p0)]) == 0
