# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Steering a myoSuperMimic tracker on CPU hosts (``supermimic.steering``), on synthetic clips."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

pytestmark = pytest.mark.tier2

TINY = dict(
    wm_hidden=(64, 64, 64),
    enc_hidden=(64, 32, 16),
    dec_hidden=(64, 32, 16),
    critic_hidden=(32,),
    buffer_steps=40,
    warmup_steps=20,
    wm_batch=16,
    wm_batches=2,
    pi_batch=8,
    horizon=4,
    log_interval=1,
    save_interval=1000,
    compile=False,
)


def _make(clip_path: str, num_envs: int = 1) -> tuple[Any, Any]:
    from supermimic.config import fast_runner_cfg
    from supermimic.env import TaskEnvCfg, make_task_env
    from supermimic.runner import SuperMimicRunner

    torch.manual_seed(0)
    env = make_task_env(
        TaskEnvCfg(motion_file=clip_path, num_envs=num_envs, sampling_mode="start"),
        "cpu",
    )
    runner = SuperMimicRunner(env, asdict(fast_runner_cfg(**TINY)), device="cpu")
    return env, runner


@pytest.fixture(scope="module")
def walk(tmp_path_factory: Any, body: Any) -> Path:
    """A synthetic 8 s walk along +x at 1 m/s, every independent joint oscillating at 1 Hz."""
    from myosuite.envs.myo.tasks.mimic.tracking.motion import write_synthetic_clip

    d = tmp_path_factory.mktemp("steer")
    write_synthetic_clip(d / "walk.npz", body, 8.0, 1.0, 0.05, 0)
    return d / "walk.npz"


@pytest.fixture(scope="module")
def tracked(walk: Path, tmp_path_factory: Any) -> tuple[Any, Any, Any]:
    """The mjlab task on the walk (one env, a few random steps in), a checkpoint of its untrained runner, and the
    tracker built from that checkpoint."""
    from supermimic.steering import TrackerPolicy

    env, runner = _make(str(walk), num_envs=1)
    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        env.step(torch.rand(1, runner.cmd.info.num_muscles, generator=gen) * 0.3)
    ck = tmp_path_factory.mktemp("ck") / "model_000000.pt"
    runner.save(str(ck))
    return env, runner, TrackerPolicy(ck, walk)


def _cpu_copy(env: Any) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """The task's scene as a CPU MuJoCo model, with the simulator's state of env 0 (forwarded)."""
    u = env.unwrapped
    m = u.sim.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = u.sim.data.qpos[0].cpu().numpy()
    d.qvel[:] = u.sim.data.qvel[0].cpu().numpy()
    d.act[:] = u.sim.data.act[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    return m, d


def test_observation_matches_the_mjlab_task(tracked: Any) -> None:
    """The tracker's observation of the task's state equals the runner's analytic observation (and the mjlab task's
    up to the coupled joints' constraint violation), and the state read from a CPU copy of the scene equals the task's
    state (touch up to the two engines' contact solutions)."""
    from supermimic.heads import StepCtx
    from supermimic.steering import body_index, read_state

    env, runner, tracker = tracked
    cmd = runner.cmd
    obs = env.get_observations()
    S = cmd.read_state()
    a = env.unwrapped.action_manager.action
    o, g = tracker.observe(S, cmd.f, a)
    ctx = StepCtx(S, cmd.f, a)
    torch.testing.assert_close(
        o, runner.obs_policy.assemble(None, ctx), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(o, obs["policy"], atol=0.5, rtol=0.05)
    torch.testing.assert_close(g, obs["goal"], atol=1e-6, rtol=0)
    m, d = _cpu_copy(env)
    idx = body_index(m, tracker.info, "robot/")
    S_cpu = read_state(m, d, idx, tracker.kin.R_off.double().numpy())
    for k in range(7):
        torch.testing.assert_close(S_cpu[k], S[k], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(S_cpu[7], S[7], atol=5.0, rtol=0.05)
    assert float(S_cpu[7].sum()) > 0


def test_turned_state_gives_the_same_observation(tracked: Any) -> None:
    """A body turned by delta in the world, turned back by ``turn_state``, observes what the unturned body does."""
    from supermimic.steering import _rz, turn_state

    env, runner, tracker = tracked
    cmd = runner.cmd
    S, a = cmd.read_state(), env.unwrapped.action_manager.action
    delta = torch.tensor([2.1])
    Rz = _rz(delta)
    p, R, v, w = S[:4]
    turned = (
        (Rz @ p.unsqueeze(-1)).squeeze(-1),
        Rz @ R,
        (Rz @ v.unsqueeze(-1)).squeeze(-1),
        (Rz @ w.unsqueeze(-1)).squeeze(-1),
    ) + tuple(S[4:])
    o_turned, _ = tracker.observe(turned, cmd.f, a)
    o_back, g_back = tracker.observe(turn_state(turned, delta), cmd.f, a)
    o, g = tracker.observe(S, cmd.f, a)
    torch.testing.assert_close(o_back, o, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(g_back, g)
    assert not torch.allclose(o_turned, o, atol=1e-3)


def test_loop_clip_continues_a_periodic_walk(
    walk: Path, body: Any, tmp_path: Path
) -> None:
    from myosuite.envs.myo.tasks.mimic.tracking.motion import (
        build_motion_lib,
        load_clip,
    )
    from supermimic.kinematics import SkeletonKinematics
    from supermimic.steering import TRACKED_BODIES, loop_clip

    kin = SkeletonKinematics(TRACKED_BODIES, "cpu", body)
    R_off = kin.R_off.double().numpy()
    meta = loop_clip(walk, tmp_path / "loop.npz", kin.info, R_off, seconds=20.0)
    assert abs(meta["period_s"] - 1.0) < 0.011 and meta["cost"] < 1e-6
    assert abs(meta["turn_deg"]) < 0.1 and abs(meta["speed"] - 1.0) < 0.02
    assert meta["frames"] >= 20.0 * meta["fps"]
    _, p_src, _, q_src = load_clip(str(walk), kin.info, R_off)
    _, p, _, q = load_clip(str(tmp_path / "loop.npz"), kin.info, R_off)
    np.testing.assert_allclose(np.diff(p[:, 0]), 0.01, atol=1e-6)
    assert (
        np.abs(np.diff(q, axis=0)).max() <= 1.05 * np.abs(np.diff(q_src, axis=0)).max()
    )
    lib = build_motion_lib([str(tmp_path / "loop.npz")], body, TRACKED_BODIES, 0.02)
    assert int(lib.clip_len[0]) == 1 + int(
        (meta["frames"] - 1) / meta["fps"] / 0.02 + 1e-6
    )


def test_heading_steer_turns_at_the_bounded_rate(tracked: Any) -> None:
    from supermimic.steering import HeadingSteer

    tracker = tracked[2]
    steer = HeadingSteer(tracker.motion, max_rate=1.0)
    torch.testing.assert_close(
        steer.travel, torch.zeros_like(steer.travel), atol=1e-4, rtol=0
    )
    f, heading = torch.tensor([10]), torch.tensor([math.pi / 2])
    delta, n = torch.zeros(1), 0
    while abs(float(delta - steer.target(f, heading))) > 1e-6:
        delta, n = steer.step(delta, f, heading), n + 1
        assert float(delta) <= min(n * 0.02, math.pi / 2) + 1e-6
    assert n == math.ceil((math.pi / 2) / 0.02)
    d = steer.step(
        torch.tensor([math.radians(170)]), f, torch.tensor([math.radians(-170)])
    )
    assert abs(float(d) - math.radians(170) - 0.02) < 1e-5


def test_reference_state_round_trips(tracked: Any) -> None:
    """A body put on a reference frame reads back as that frame's state (the task's reset)."""
    from supermimic.steering import body_index, read_state, write_reference_state

    env, _, tracker = tracked
    m, d = _cpu_copy(env)
    idx = body_index(m, tracker.info, "robot/")
    f = torch.tensor([37])
    write_reference_state(m, d, idx, tracker, 37)
    S = read_state(m, d, idx, tracker.kin.R_off.double().numpy())
    fr = tracker.motion.frames
    cols = torch.as_tensor(tracker.info.indep_cols)
    Rb = fr["root_rot"][f]
    w = (Rb @ fr["root_qvel"][f, 3:6].unsqueeze(-1)).squeeze(-1)
    q = fr["joint_pos"][f][:, cols].clamp(tracker.kin.q_lo, tracker.kin.q_hi)
    qd = fr["joint_vel"][f][:, cols]
    want = (
        fr["root_pos"][f],
        tracker.heads.root_R[f],
        fr["root_qvel"][f, :3],
        w,
        q,
        qd,
    )
    for got, exp in zip(S[:6], want, strict=True):
        torch.testing.assert_close(got, exp, atol=1e-5, rtol=1e-5)
    assert float(S[6].abs().max()) == 0.0


def test_agent_drives_a_body_of_the_scene(tracked: Any) -> None:
    """On the task's scene as a CPU host: the phase match finds the gait phase the body was put on, the agent acts
    every hold steps on its frame, and the tracker body has no FK mismatch with itself."""
    from supermimic.steering import TrackerAgent, write_reference_state

    env, _, tracker = tracked
    m, d = _cpu_copy(env)
    agent = TrackerAgent(tracker, m, host_dt=0.01, prefix="robot/")
    assert agent.hold == 2
    k = 37
    write_reference_state(m, d, agent.idx, tracker, k)
    agent.reset(d)
    assert agent.reset_frame % 50 == k % 50 and abs(float(agent.delta)) < 1e-5
    assert agent.fk_mismatch_mm(d) < 1e-3
    for _ in range(6):
        d.ctrl[:] = agent.host_action(d, heading=0.0)
        for _ in range(5):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
    assert int(agent.f[0]) == agent.reset_frame + 2
    assert np.isfinite(d.qpos).all() and ((d.ctrl >= 0) & (d.ctrl <= 1)).all()


def test_tracking_check_scores_like_the_task(tracked: Any) -> None:
    """The CPU tracking check on the task's scene: the untrained tracker falls behind its clip, which ends the
    episode with a deviation termination, and the return is normalized."""
    from supermimic.steer_eval import evaluate_tracking

    env, _, tracker = tracked
    m, _ = _cpu_copy(env)
    r = evaluate_tracking(tracker, m, prefix="robot/")
    (row,) = r["per_clip"]
    clip_steps = int(tracker.motion.clip_len[0]) - 1
    assert not row["completed"] and 0 < row["steps"] < clip_steps
    assert 0.0 < row["return"] < row["steps"] / clip_steps


def test_body_index_resolves_the_chasetag_players(body: Any) -> None:
    """Both FBVs players map onto the tracker by name, in the order of each player's action vector."""
    from myosuite import make_env
    from myosuite.envs.myo.tasks.mimic.tracking.models import ModelInfo
    from supermimic.steering import body_index

    env = make_env("myoChallengeChaseTagFBVs-v0")
    raw = env.unwrapped
    info = ModelInfo(body)
    for agent, prefix in (("agent_0", "a0_"), ("agent_1", "a1_")):
        idx = body_index(raw.model, info, prefix)
        np.testing.assert_array_equal(
            idx.actuators, np.asarray(raw._meta.act_indices[agent])
        )
        assert len(idx.ground) >= 1 and len(set(idx.q_adr)) == len(info.indep_names)
    env.close()


def test_checkpoints_and_curve_rows(tmp_path: Path) -> None:
    from supermimic.steer_eval import checkpoints, curve_row

    for it in (0, 250, 500, 750, 1000):
        (tmp_path / f"model_{it:06d}.pt").touch()
    names = [Path(f).stem for f in checkpoints(tmp_path, every=3)]
    assert names == ["model_000000", "model_000750", "model_001000"]

    class _T:
        iteration, wall_time, sim_steps = 750, 90.0, 12

    d = {
        "summary": {
            "surv": 0.5,
            "mean_steps_alive": 150.0,
            "align": 0.9,
            "speed": 1.1,
            "turns": {90.0: {"surv": 0.25, "aligned": 0.25, "final_err_deg": 12.0}},
        }
    }
    c = {
        "summary": {
            "still/agent_0/CHASE": {
                "n": 4,
                "tags": 3,
                "chaser_falls": 1,
                "mean_length_s": 4.0,
            }
        }
    }
    row = curve_row(_T(), d, c)  # type: ignore[arg-type]
    assert (
        row["wall_time_min"] == 1.5
        and row["dir_surv"] == 0.5
        and row["turn+90_aligned"] == 0.25
    )
    assert (
        row["still_agent_0_CHASE_tag_rate"] == 0.75
        and row["still_agent_0_CHASE_chaser_fall_rate"] == 0.25
    )


def test_tracker_plays_a_chasetag_match(tracked: Any) -> None:
    """The tracker as the runner of a scored FBVs match against a chaser that does nothing: the chaser falls, which
    ends the match, and the tracker's actions reach its own actuators."""
    from supermimic.steer_eval import StillPlayer, TrackerPlayer, _fbvs_env, play_match

    tracker = tracked[2]
    env = _fbvs_env("CHASE")
    me = TrackerPlayer(tracker)
    r = play_match(env, {"agent_0": StillPlayer(), "agent_1": me}, seed=0)
    raw = env.unwrapped
    assert r["task"] == "CHASE" and r["chaser_fell"] and not r["tagged"]
    assert 0 < r["chaser_fell_at"] == r["length_s"] < 10.0
    (a1,) = (a for (m, a) in me.agents if m == id(raw.model))
    assert a1 == "agent_1"
    ctrl = raw.data.ctrl[np.asarray(raw._meta.act_indices["agent_1"])]
    assert (
        ctrl.max() > 0
        and raw.data.ctrl[np.asarray(raw._meta.act_indices["agent_0"])].max() == 0
    )
    env.close()
