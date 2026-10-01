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

_P0, _P2 = "myoChallengeTableTennisP0-v0", "myoChallengeTableTennisP2-v0"


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


def _body_id(env: ManagerBasedRlEnv, short: str) -> int:
    model = env.sim.mj_model
    ids = [i for i in range(model.nbody) if model.body(i).name.split("/")[-1] == short]
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


def test_reset_state_matches_cpu(p0: ManagerBasedRlEnv, cpu_p0: gym.Env) -> None:
    """Athlete root, paddle grip and the whole first observation match CPU."""
    obs = p0.reset()[0]["policy"][0].numpy()
    cpu_obs, _ = cpu_p0.reset(seed=0)
    cm, cd = cpu_p0.model, cpu_p0.data
    data = p0.sim.data

    for body in ("Full Body", "pelvis", "paddle"):
        bid = _body_id(p0, body)
        np.testing.assert_allclose(
            data.xpos[0, bid].numpy(),
            cd.xpos[cm.body(body).id],
            atol=1e-5,
            err_msg=body,
        )
        np.testing.assert_allclose(
            data.xquat[0, bid].numpy(),
            cd.xquat[cm.body(body).id],
            atol=1e-5,
            err_msg=body,
        )
    grasp = [
        i
        for i in range(p0.sim.mj_model.nsite)
        if p0.sim.mj_model.site(i).name.endswith("/S_grasp")
    ][0]
    handle = _geom_id(p0, "handle")
    mj_grip = np.linalg.norm(
        data.site_xpos[0, grasp].numpy() - data.geom_xpos[0, handle].numpy()
    )
    cpu_grip = np.linalg.norm(
        cd.site_xpos[cm.site("S_grasp").id] - cd.geom_xpos[cm.geom("handle").id]
    )
    assert mj_grip == pytest.approx(cpu_grip, abs=1e-5)
    assert cpu_grip < 0.03  # the handle sits in the hand
    np.testing.assert_allclose(obs, cpu_obs, atol=1e-4)


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


def test_p2_randomizes_the_ball_and_paddle_per_env() -> None:
    """Ball friction DR hits the ball (not the floor); paddle mass differs per env."""
    env = _make_env(_P2, 2)
    try:
        default = env.sim.mj_model.geom_friction.copy()
        torch.manual_seed(0)
        env.reset()
        friction = env.sim.model.geom_friction.numpy()  # (num_envs, ngeom, 3)
        ball = _geom_id(env, "pingpong")
        low, high = [0.9, 0.004, 1e-5], [1.1, 0.006, 3e-5]
        others = np.delete(friction, ball, axis=1)
        np.testing.assert_allclose(
            others, np.broadcast_to(np.delete(default, ball, 0), others.shape)
        )
        assert np.all(friction[:, ball] >= np.array(low) - 1e-9)
        assert np.all(friction[:, ball] <= np.array(high) + 1e-9)
        assert not np.allclose(friction[0, ball], friction[1, ball])
        mass = env.sim.model.body_mass.numpy()[:, _body_id(env, "paddle")]
        assert np.all((mass >= 0.10) & (mass <= 0.15))
        assert mass[0] != mass[1]
    finally:
        env.close()
