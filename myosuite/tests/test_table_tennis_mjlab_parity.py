# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab TableTennis must simulate and score the CPU task of the same env id.

Covers the scene (athlete root, paddle in the hand), the ball-contact labels,
when the terminal bonus/penalty is paid, the P2 domain randomization targets
and the time limit. All runs use ``device="cpu"`` and one or two envs.
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
from myosuite.envs.myo.tasks.challenge.tabletennis import (  # noqa: E402
    ContactTrajIssue,
    PingpongContactLabels,
    evaluate_pingpong_trajectory,
)
from myosuite import make_env  # noqa: E402

_P0, _P2 = "myoChallengeTableTennisP0-v0", "myoChallengeTableTennisP2-v0"
# Reward of a step that pays ``done`` (-10) is clearly negative; every other
# per-step reward is positive (dense terms, ~2.5).
_DONE_PAID = -5.0
_SOLVED_PAID = 500.0  # ``solved`` is +1000


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
    env = make_env(_P0).unwrapped
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


def _place_ball(
    env: ManagerBasedRlEnv, pos: np.ndarray, vel: np.ndarray | None = None
) -> None:
    """Teleport the ball for the next step (world-frame linear velocity)."""
    state = torch.zeros(env.num_envs, 13)
    state[:, :3] = torch.as_tensor(pos, dtype=torch.float32)
    state[:, 3] = 1.0
    if vel is not None:
        state[:, 7:10] = torch.as_tensor(vel, dtype=torch.float32)
    env.scene["pingpong"].write_root_state_to_sim(state)


def _hit_face(
    env: ManagerBasedRlEnv, geom: str, normal: np.ndarray, gap: float
) -> None:
    """Send the ball at 1 m/s onto the face of *geom* with outward *normal*.

    The table contact is stiff (2 ms), so a ball placed at rest in it is pushed
    out before the step ends. From ``gap`` = 9 mm the ball flies freely at
    every substep start (9..1 mm) and is 1 mm deep at the post-step state; on
    the hand-held paddle, which moves during the step, it starts on the face.
    """
    gid = _geom_id(env, geom)
    centre = env.sim.data.geom_xpos[0, gid].numpy().copy()
    rot = env.sim.data.geom_xmat[0, gid].numpy().reshape(3, 3)
    size = env.sim.mj_model.geom_size[gid]
    if int(env.sim.mj_model.geom_type[gid]) == 5:  # cylinder: flat face along local z
        half = float(size[1])
    else:  # box: extent along the normal
        half = float(np.abs(rot.T @ normal) @ size)
    radius = float(env.sim.mj_model.geom_size[_geom_id(env, "pingpong"), 0])
    _place_ball(env, centre + normal * (half + radius + gap), -normal * 1.0)


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


def test_obs_and_reward_match_cpu_on_the_same_state(
    p0: ManagerBasedRlEnv, cpu_p0: gym.Env
) -> None:
    """Along a CPU rollout, mjlab scores each CPU state like CPU does.

    Open-loop rollouts diverge within a few steps (MuJoCo Warp resolves the
    hand/paddle contacts differently from C MuJoCo), so the CPU state is
    copied into mjlab before each comparison. The qpos/qvel layouts match.
    """
    cpu = cpu_p0
    cpu.reset(seed=0)
    p0.reset()
    data = p0.sim.data
    rng = np.random.default_rng(1)
    touching = _touching_slice(cpu)
    events = set()
    for t in range(1, 201):
        obs_cpu, _, _, _, info = cpu.step(rng.uniform(-1, 1, cpu.action_space.shape))
        for field in ("qpos", "qvel", "act"):
            getattr(data, field)[0] = torch.as_tensor(getattr(cpu.data, field))
        p0.episode_length_buf[:] = t
        p0.sim.forward()
        p0.termination_manager.compute()
        reward = float(p0.reward_manager.compute(dt=p0.step_dt)[0])
        obs = p0.observation_manager.compute_group("policy")[0].numpy()
        np.testing.assert_allclose(obs, obs_cpu, atol=2e-4, err_msg=f"step {t}")
        rwd = info["rwd_dict"]
        done = bool(np.squeeze(rwd["done"]))
        rally = p0.termination_manager.get_term_cfg("task_done").func
        assert bool(rally.done[0]) == done, t
        assert reward == pytest.approx(float(rwd["dense"]), abs=1e-3), t
        if obs[touching].any():
            events.add("touch")
        if done:
            events.add("done")
            break
    assert events == {"touch", "done"}  # the rollout reached a bounce and an end


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


def test_ball_drop_penalty_paid_once(p0: ManagerBasedRlEnv) -> None:
    """``done`` (-10) is paid on the terminating step only, like CPU."""
    p0.reset()
    _step(p0)
    _place_ball(p0, np.array([-2.5, 0.0, 0.2]))  # below the 0.3 m drop height
    rewards, terminated = [], []
    for _ in range(3):
        rew, term, _ = _step(p0)
        rewards.append(rew)
        terminated.append(term)
    paid = [r < _DONE_PAID for r in rewards]
    assert paid == [True, False, False], rewards
    assert terminated == [True, False, False]


def _win_rally(env: ManagerBasedRlEnv) -> tuple[list[float], list[bool]]:
    """Paddle hit, flight, opponent half (then flight until the episode ends)."""
    up = np.array([0.0, 0.0, 1.0])
    rewards, terminated = [], []
    for target in ("pad", "air", "coll_opponent_half", "air", "air"):
        if target == "pad":  # onto the pad face that points along its local z
            pad = _geom_id(env, "pad")
            normal = env.sim.data.geom_xmat[0, pad].numpy().reshape(3, 3)[:, 2].copy()
            _hit_face(env, "pad", normal, gap=0.0)
        elif target == "air":
            _place_ball(env, np.array([-0.5, 0.0, 1.8]))
        else:
            _hit_face(env, target, up, gap=0.009)
        rew, term, _ = _step(env)
        rewards.append(rew)
        terminated.append(term)
        if term or rew > _SOLVED_PAID:
            break
    return rewards, terminated


def test_solved_bonus_paid_once(p0: ManagerBasedRlEnv) -> None:
    """Paddle hit, flight, opponent half: +1000 (and done) once, on that step."""
    p0.reset()
    _step(p0)
    rewards, terminated = _win_rally(p0)
    if not terminated[-1]:  # the bonus came without the episode ending
        for _ in range(2):
            rew, term, _ = _step(p0)
            rewards.append(rew)
            terminated.append(term)
    solved = [r > _SOLVED_PAID for r in rewards]
    assert solved == [False, False, True] + [False] * (len(solved) - 3), rewards
    assert terminated[2], terminated


def test_second_rally_follows_a_solved_one() -> None:
    """With rally_count=2 a solved rally is paid, relaunches the ball and goes on."""
    import dataclasses  # noqa: PLC0415

    from myosuite.envs.myo.backends.mjlab.configs.table_tennis_cfg import (  # noqa: PLC0415
        TableTennisCfg,
    )
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tabletennis import (  # noqa: PLC0415
        make_table_tennis_mjlab_env_cfg,
    )

    cfg = make_table_tennis_mjlab_env_cfg(
        dataclasses.replace(TableTennisCfg.p0(), rally_count=2)
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        _step(env)
        ball = env.scene["pingpong"]
        launch = ball.data.default_root_state[0].numpy().copy()
        for rally in (1, 2):
            rewards, terminated = _win_rally(env)
            assert [r > _SOLVED_PAID for r in rewards] == [False, False, True]
            assert terminated == [False, False, rally == 2]
            if rally == 1:  # relaunched like at reset; the clock restarts
                np.testing.assert_allclose(
                    ball.data.root_link_pos_w[0].numpy(), launch[:3], atol=1e-6
                )
                np.testing.assert_allclose(
                    ball.data.root_link_lin_vel_w[0].numpy(), launch[7:10], atol=1e-5
                )
    finally:
        env.close()


def test_timeout_step_pays_no_done_penalty(p0: ManagerBasedRlEnv) -> None:
    """At the 300th step the task time (3 s) is up but not exceeded (CPU: no done)."""
    p0.reset()
    _step(p0)
    # State of step 299 of an episode: elapsed steps and float32 sim time.
    p0.episode_length_buf[:] = 299
    t = np.float32(0.0)
    for _ in range(299 * p0.cfg.decimation):
        t = np.float32(t + np.float32(p0.physics_dt))
    p0.sim.data.time[:] = float(t)
    _place_ball(p0, np.array([-0.5, 0.0, 1.8]))  # in free flight, nothing to hit
    rew, term, trunc = _step(p0)
    assert trunc and not term
    assert rew > 0.0, rew


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


def test_trajectory_state_matches_cpu_evaluation() -> None:
    """The per-step trajectory state gives ``evaluate_pingpong_trajectory`` of every prefix."""
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tabletennis import (  # noqa: PLC0415
        _PingpongTrajectory,
        _trajectory_outcome,
    )

    rng = np.random.default_rng(0)
    order = [
        PingpongContactLabels.PADDLE,
        PingpongContactLabels.OWN,
        PingpongContactLabels.OPPONENT,
        PingpongContactLabels.NET,
        PingpongContactLabels.GROUND,
        PingpongContactLabels.ENV,
    ]
    num_envs, steps = 512, 12
    # Sparse contacts so that long undecided prefixes occur.
    labels = rng.random((steps, num_envs, 6)) < np.array(
        [0.25, 0.3, 0.15, 0.05, 0.05, 0.05]
    )
    traj = _PingpongTrajectory(num_envs, "cpu")
    for t in range(steps):
        traj.update(torch.as_tensor(labels[t]))
        for e in range(num_envs):
            sets = [
                {order[k] for k in np.flatnonzero(labels[s, e])} for s in range(t + 1)
            ]
            want = evaluate_pingpong_trajectory(sets)
            assert _trajectory_outcome(int(traj.outcome[e])) == want, (t, e, sets)
    assert {int(o) for o in traj.outcome} >= {
        ContactTrajIssue.OWN_HALF.value,
        ContactTrajIssue.NO_PADDLE.value,
        ContactTrajIssue.DOUBLE_TOUCH.value,
        ContactTrajIssue.MISS.value,
    }


def test_contact_labels_use_only_live_rows() -> None:
    """Labels come from rows ``< nacon`` of the batched buffer, per world."""
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tabletennis import (  # noqa: PLC0415
        _ball_contact_labels,
    )

    # geoms: 0 ground, 1 own half, 2 pad, 3 ball (body 3), 4 other (body 4)
    geom_body = torch.tensor([0, 0, 2, 3, 4])
    label_of_geom = torch.tensor([4, 1, 0, 5, 5])
    geom = torch.tensor([[3, 1], [2, 3], [4, 0], [3, 4], [3, 1], [3, 0]])
    world = torch.tensor([0, 1, 1, 1, 0, 1])
    flags = _ball_contact_labels(
        torch.tensor([4]), geom, world, geom_body, 3, label_of_geom, num_envs=2
    )
    expected = torch.zeros(2, 6, dtype=torch.bool)
    expected[0, 1] = True  # ball / own half
    expected[1, 0] = True  # pad / ball
    expected[1, 5] = True  # ball / other body
    assert torch.equal(flags, expected)  # rows 4-5 (stale) ignored
