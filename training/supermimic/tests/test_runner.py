# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The SuperMimic runner on the task built by ``make_env`` (CPU): it scores states like the env, learns with the
recipe and round-trips checkpoints; the graphed collector's wrapper against mjlab's ``env.step``; batched evaluation,
refinement, fine-tuning and the training entry point (train -> checkpoint -> resume -> evaluate). The CUDA graphs are
in ``test_supermimic_fast_cuda.py``. Ported in part from muscle-supermimic's tests (@ 8a6244b, Apache-2.0).
"""

from __future__ import annotations

import copy
import csv
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

pytestmark = pytest.mark.tier2


def _make(clip_path: str, device: str = "cpu", num_envs: int = 2, muscle_obs: bool = False, sampling: str = "start",
          env_edit: Any = None, **rl: Any) -> tuple[Any, Any]:  # fmt: skip
    """The task through ``make_env`` with the recipe's options, and a tiny runner on it."""
    from supermimic.config import fast_runner_cfg
    from supermimic.env import TaskEnvCfg, make_task_env
    from supermimic.runner import SuperMimicRunner

    torch.manual_seed(0)
    task = TaskEnvCfg(
        motion_file=clip_path,
        num_envs=num_envs,
        muscle_obs=muscle_obs,
        sampling_mode=sampling,
    )
    env = make_task_env(task, device)
    if env_edit is not None:
        env_edit(env.unwrapped)
    runner = SuperMimicRunner(
        env, asdict(fast_runner_cfg(**{**TINY, **rl})), device=device
    )
    return env, runner


@pytest.fixture(
    scope="module", params=[False, True], ids=["no_muscle_obs", "muscle_obs"]
)
def setup(request: Any, clips: Path) -> tuple[Any, Any]:
    return _make(str(clips / "clip0.npz"), muscle_obs=request.param)


TINY = dict(
    wm_hidden=(64, 64, 64), enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16), critic_hidden=(32,), buffer_steps=40,
    warmup_steps=20, wm_batch=16, wm_batches=2, pi_batch=8, horizon=4, log_interval=1, save_interval=1000,
    compile=False,
)  # fmt: skip


def test_clip_row_grad_bounds_each_row_over_the_group() -> None:
    from supermimic.runner import ClipRowGrad

    a = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(3, 2, 2, requires_grad=True)
    stats: list = []
    a1, b1 = ClipRowGrad.apply(1.0, 2.0, stats, a, b)
    w = torch.tensor([10.0, 0.01, 0.3])
    ((a1.sum(-1) + b1.flatten(1).sum(-1)) * w).sum().backward()
    norm = 2.0 * torch.sqrt(
        a.grad.square().sum(-1) + b.grad.flatten(1).square().sum(-1)
    )
    torch.testing.assert_close(norm, torch.tensor([1.0, 2 * 0.01 * 8**0.5, 1.0]))
    assert len(stats) == 1 and abs(float(stats[0]) - 2 / 3) < 1e-6


def test_runner_scores_like_the_env(setup: Any) -> None:
    """The runner's reward, failure terminations and both observation groups (the task's pure terms on the method's
    kinematics, what imagination evaluates) match the env's managers (the same terms on MuJoCo's state) on the states
    the simulator reaches. They differ by the coupled joints' constraint violation, which the kinematics projects
    away (sub-millimetre; velocities more)."""
    from myosuite.envs.myo.backends.mjlab.tasks.mdp import SYNC_TERM
    from supermimic.heads import StepCtx, analytic

    env, runner = setup
    u, cmd = env.unwrapped, runner.cmd
    u.reset()
    gen = torch.Generator().manual_seed(0)
    touched = 0.0
    for _ in range(12):
        S0, a_prev = cmd.read_state(), u.action_manager.action.clone()
        a = torch.rand(2, cmd.info.num_muscles, generator=gen) * 0.3
        obs, rew, dones, _ = env.step(a)
        assert not dones.any()
        S1 = cmd.read_state()
        torch.testing.assert_close(
            S1[6], cmd.muscle.step(S0[6], a)[0], atol=1e-6, rtol=0
        )
        r, _ = runner._reward_term(S1, cmd.f, a, a_prev)
        torch.testing.assert_close(r, rew, atol=2e-3, rtol=2e-2)
        tm = u.termination_manager
        for name in tm.active_terms:
            tc = tm.get_term_cfg(name)
            if not tc.time_out and name != SYNC_TERM:
                p = {k: v for k, v in tc.params.items() if k != "command_name"}
                assert torch.equal(
                    analytic(tc.func)(runner.heads, StepCtx(S1, cmd.f), **p),
                    tm.get_term(name),
                ), name
        ctx = cmd.obs_ctx()
        ours, env_obs = runner.obs_policy.assemble(None, ctx), obs["policy"]
        om = u.observation_manager
        start = 0
        for name, dim in zip(
            om.active_terms["policy"], om.group_obs_term_dim["policy"], strict=True
        ):
            sl = slice(start, start + dim[0])
            start += dim[0]
            # muscle state: MuJoCo's tendon geometry vs the method's analytic one
            loose = name in ("muscle_length", "muscle_velocity", "muscle_force")
            if loose:
                scale = env_obs[:, sl].abs().max()
                assert ((ours[:, sl] - env_obs[:, sl]).abs() <= 0.1 * scale).all(), name
            else:
                torch.testing.assert_close(
                    ours[:, sl], env_obs[:, sl], atol=0.5, rtol=0.05, msg=name
                )
        torch.testing.assert_close(
            runner.obs_goal.assemble(None, ctx), obs["goal"], atol=1e-6, rtol=0
        )
        touched += float(S1[7].sum())
    assert touched > 0  # the feet touch the ground


def test_imagined_step_matches_the_analytic_terms(setup: Any) -> None:
    """One imagined step: the observation the policy reads, the critic input and the rewards of the imagined state
    are the analytic terms of that state."""
    from supermimic.heads import StepCtx

    env, runner = setup
    cmd = runner.cmd
    S, f, a = (
        cmd.read_state(),
        cmd.f.clone(),
        env.unwrapped.action_manager.action.clone(),
    )
    hist = runner.obs_policy.init_history(StepCtx(S, f, a))
    S1, a1, hist1, f1, x = runner._imag_step(S, hist, f, a)
    assert not x.requires_grad and torch.equal(f1, f + 1)
    torch.testing.assert_close(x, runner._critic_input(S, f, a), atol=0, rtol=0)
    torch.testing.assert_close(a1, runner.pi(runner.obs_policy.assemble(None, StepCtx(S, f, a)),
                                             runner.obs_goal.assemble(None, StepCtx(S, f, a))))  # fmt: skip
    r, d = runner._reward_term(S1, f1, a1, a)
    ctx = StepCtx(S1, f1, a1, a)
    expect = sum(w * fn(runner.heads, ctx, **p) for fn, w, p in runner.reward_terms)
    torch.testing.assert_close(r, expect)
    assert ((d >= 0) & (d <= 1)).all()


def test_kin_unbind_is_bit_identical(setup: Any) -> None:
    """kin_unbind: the same body states and joint-coordinate gradients as a select per joint."""
    _, runner = setup
    kin = runner.cmd.kin
    p, R, v, w, q, qd = (x.detach().clone() for x in runner.cmd.read_state()[:6])
    out = []
    for unbind in (False, True):
        kin.unbind = unbind
        qq, qqd = q.clone().requires_grad_(), qd.clone().requires_grad_()
        P, _, V, _ = kin.forward(p, R, v, w, qq, qqd)
        (P.square().sum() + V.square().sum()).backward()
        out.append((P.detach(), V.detach(), qq.grad, qqd.grad))
    kin.unbind = runner.cfg.kin_unbind
    for a, b in zip(*out):
        assert torch.equal(a, b)


def test_muscle_geometry_only_when_observed(setup: Any, monkeypatch: Any) -> None:
    from supermimic.heads import StepCtx

    env, runner = setup
    u, cmd = env.unwrapped, runner.cmd
    observed = {"muscle_length", "muscle_velocity", "muscle_force"} & set(
        u.observation_manager.active_terms["policy"]
    )
    assert len(observed) in (0, 3) and cmd.observes_muscles == bool(observed)
    u.reset()
    calls: list = []
    side = "sim"
    lv = cmd.geo.length_velocity
    monkeypatch.setattr(
        cmd.geo,
        "length_velocity",
        lambda q, qd: calls.append((side, len(q))) or lv(q, qd),
    )
    env.step(torch.full((2, cmd.info.num_muscles), 0.1))
    side = "imagination"
    S, f, a = cmd.read_state(), cmd.f.clone(), u.action_manager.action.clone()
    runner._imag_step(S, runner.obs_policy.init_history(StepCtx(S, f, a)), f, a)
    # the env observes MuJoCo's muscle state; only imagination evaluates the geometry
    assert calls == ([("imagination", 2)] if observed else [])


def test_runner_with_the_recipe_learns_and_round_trips(
    clips: Path, tmp_path: Path
) -> None:
    env, runner = _make(str(clips / "clip0.npz"), pi_ema=0.9, wm_only_iters=0, pi_lr_mult=4.0, lr_decay_ret=2.0,
                        lr_decay_by=2, lr_decay_len=2)  # fmt: skip
    c = runner.cfg
    assert (
        not c.cuda_graph and not c.overlap_collect and c.collect_graph
    )  # CPU: the collector's wrapper runs eagerly
    assert (
        env.unwrapped.cfg.decimation == 10
        and "action_map" not in runner.pi.state_dict()
    )
    runner.log_dir = str(tmp_path)
    runner.learn(num_learning_iterations=4)
    assert runner.it == 4 and runner._ema_n == 4 and int(runner.nonfinite) == 0
    live, avg = runner.pi.state_dict(), runner.pi_avg.module.state_dict()
    assert any(not torch.equal(live[k], avg[k]) for k in live)
    obs = env.get_observations()
    torch.testing.assert_close(
        runner.get_inference_policy()(obs),
        runner.pi_avg.module(obs["policy"], obs["goal"]),
    )
    with open(tmp_path / "train_log.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert runner._decay_at == 2
    mults = [float(r["pi_mult"]) for r in rows]
    assert mults[:3] == [4.0, 4.0, 4.0] and mults[3] == 2.5
    s, h = runner.sample_valid_starts(c.pi_batch)
    m = runner.motion
    assert (
        s["f"].shape == (c.pi_batch,)
        and (s["f"] + c.horizon <= m.clip_end[m.clip_id[s["f"]]]).all()
    )
    ck = torch.load(tmp_path / "model_000004.pt", weights_only=False)
    for k in avg:
        torch.testing.assert_close(ck["pi"][k], avg[k], atol=0, rtol=0)
        torch.testing.assert_close(ck["pi_live"][k], live[k], atol=0, rtol=0)
    assert (
        ck["env"]["decimation"] == 10
        and ck["env"]["stride"] == 10
        and not ck["env"]["muscle_obs"]
    )
    assert ck["action_map"] == {"kind": "identity", "num_muscles": runner.nm}
    with torch.no_grad():
        for p in runner.pi_avg.module.parameters():
            p.add_(1.0)
    runner.load(str(tmp_path / "model_000004.pt"), load_cfg={"actor": True})
    for k, v in runner.pi_avg.module.state_dict().items():
        torch.testing.assert_close(v, avg[k], atol=0, rtol=0)
    runner._decay_at, runner._ema_n = None, 0
    runner.load(str(tmp_path / "model_000004.pt"))
    assert runner._decay_at == 2 and runner._ema_n == 4 and runner.it == 4
    runner.learn(num_learning_iterations=2)
    assert runner.it == 6 and runner._ema_n == 6
    # structured world model and mirror augmentation
    assert runner.mirror is not None and not torch.allclose(
        runner.wm.inc_norm.mean, runner.wm.out_norm.mean
    )
    w = runner.buffer.sample_windows(6, c.wm_window)
    wm_ = runner._mirror_windows(w, 3)
    assert torch.equal(wm_["mask"], w["mask"]) and torch.equal(wm_["s"][3:], w["s"][3:])
    back = runner._mirror_windows(wm_, 3)
    for k in ("s", "s1", "act", "a", "touch1"):
        torch.testing.assert_close(back[k], w[k], atol=1e-5, rtol=1e-5)
    # bootstrapped values never exceed the attainable maximum
    x = 1e3 * torch.randn(16, runner.critic.norm.mean.shape[0])
    assert (runner.value(x) <= runner.max_step_reward / (1 - c.gamma) + 1e-6).all()


def _short_episodes(u: Any) -> None:
    u.cfg.episode_length_s = 0.1  # time-outs after 5 steps
    # the rising clip's envs fail by then
    u.termination_manager.get_term_cfg("root_pos_dev").params["threshold"] = 0.015


def test_collector_matches_the_eager_env_step(clips: Path) -> None:
    """The graphed collection's wrapper (eager on a CPU) writes the transitions, episode ends, resets, bin and
    normalizer statistics and simulator state of mjlab's ``env.step`` + ``env.reset(ids)``."""
    from supermimic.collector import STATS

    pair = []
    for graph in (False, True):
        _, r = _make(str(clips / "clips.txt"), num_envs=6, env_edit=_short_episodes, buffer_steps=64, action_noise=0.0, action_noise_dist="gaussian",
                     collect_graph=graph)  # fmt: skip
        pair.append(r)
    eager, graphed = pair
    # a constant policy: the eager path acts on the env's observation (MuJoCo's state), the collector on the method's
    # (its kinematics), which differ slightly; with equal actions the transitions must agree
    with torch.no_grad():
        for w in eager.pi.parameters():
            w.zero_()
    graphed.pi.load_state_dict(eager.pi.state_dict())
    n, stats = 40, []
    for r in pair:
        r.init_collection()
        st: dict = {k: [] for k in STATS}
        r.collect(n, st)
        stats.append(st)
    be, bg = eager.buffer, graphed.buffer
    assert graphed.collector.calls == n and be.ptr == bg.ptr == int(bg.ptr_t)
    tol = {"atol": 1e-5, "rtol": 1e-5}
    for key in ("done", "f", "ep_step"):
        assert torch.equal(be.data[key][:n], bg.data[key][:n]), key
    for key in ("s", "act", "a", "s1"):
        torch.testing.assert_close(
            bg.data[key][:n].float(), be.data[key][:n].float(), **tol, msg=key
        )
    # foot touch: the collector reads the contacts before forward() (as the analytic env did), the env after its
    # sync_forward, one physics substep apart
    for key in ("touch", "touch1"):
        e, g = be.data[key][:n].float(), bg.data[key][:n].float()
        assert ((e - g).abs() <= 0.25 * e.abs().max() + 1.0).all(), key
    # rewards: the env scores MuJoCo's state, the collector the method's kinematics of it
    torch.testing.assert_close(bg.data["r"][:n], be.data["r"][:n], atol=2e-3, rtol=2e-2)
    ends = int(bg.data["done"][:n].sum())
    term = sum(float(v) for v in stats[0]["term"])
    assert 0 < term < ends  # both failures and time-outs
    for key in ("f", "start_bin"):
        assert torch.equal(getattr(eager.cmd, key), getattr(graphed.cmd, key)), key
    torch.testing.assert_close(graphed.cmd.bin_fail, eager.cmd.bin_fail, **tol)
    for key in ("bin_G", "ep_ret", "ep_disc_ret"):  # sums of the rewards
        torch.testing.assert_close(
            getattr(graphed.cmd, key),
            getattr(eager.cmd, key),
            atol=2e-3,
            rtol=2e-2,
            msg=key,
        )
    de, dg = eager.uenv.sim.data, graphed.uenv.sim.data
    for key in ("qpos", "qvel", "act", "ctrl"):
        torch.testing.assert_close(
            getattr(dg, key)._tensor, getattr(de, key)._tensor, **tol, msg=key
        )
    for ne, ng, t in (
        (
            eager.pi.obs_norm,
            graphed.pi.obs_norm,
            {"atol": 0.05, "rtol": 0.2},
        ),  # touch: see above
        (eager.wm.inc_norm, graphed.wm.inc_norm, tol),
    ):
        assert float(ne.count) == float(ng.count) == n * 6
        torch.testing.assert_close(ng.mean, ne.mean, **t)


def test_batched_evaluation_and_refinement(clips: Path, tmp_path: Path) -> None:
    from supermimic.evaluate import (
        _angle,
        _in_frame,
        _yaw,
        build,
        evaluate_many,
    )
    from supermimic.refine import Refiner

    env, runner = build(str(clips / "clips.txt"), "cpu", copies=2)
    assert env.unwrapped.num_envs == 6 and env.unwrapped.cfg.decimation == 10
    p0 = copy.deepcopy(runner.deployed).eval()
    p1 = copy.deepcopy(p0)
    with torch.no_grad():
        for w in p1.parameters():
            if w.dim() == 2:
                w.add_(0.3 * torch.randn_like(w) * w.std())
    r0, r1 = evaluate_many(env, runner, [p0, p1])
    assert r0["norm_return"] != r1["norm_return"] and 0 <= r0["n_failed"] <= 3
    for key in ("norm_return", "mpjpe_mm", "joint_rmse_deg"):
        assert math.isfinite(r0[key]), key

    # the error metrics' frame helpers
    def rz(a):
        return torch.tensor(
            [
                [math.cos(a), -math.sin(a), 0.0],
                [math.sin(a), math.cos(a), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )  # noqa: E731

    assert abs(_angle(torch.eye(3), rz(0.3)).item() - 0.3) < 1e-6
    torch.testing.assert_close(_yaw(rz(0.7)), rz(0.7), atol=1e-6, rtol=0)
    P, t = torch.randn(5, 3), torch.tensor([1.0, -2.0, 0.5])
    torch.testing.assert_close(
        _in_frame(rz(0.2), P),
        _in_frame(rz(1.3) @ rz(0.2), P @ rz(1.3).T + t),
        atol=1e-5,
        rtol=0,
    )
    # test-time refinement: an action in the box whose imagined return is at least the policy's own
    runner.log_dir = str(tmp_path)
    runner.save(str(tmp_path / "model.pt"))
    ck = runner.load(str(tmp_path / "model.pt"), load_cfg={"actor": True})
    refiner = Refiner(runner, ck, 2, 0.01, 4, True, compile_=False, graph=False)
    cmd = runner.cmd
    env.unwrapped.reset()
    ctx = cmd.obs_ctx()
    a_pi = runner.get_inference_policy()(env.get_observations())
    a, gain = refiner(ctx, a_pi)
    assert (
        a.shape == a_pi.shape
        and (a >= 0).all()
        and (a <= 1).all()
        and (gain >= 0).all()
    )


def test_finetune_from_a_baseline(clips: Path, tmp_path: Path) -> None:
    """A baseline trained on several clips, fine-tuned on one: its networks and normalizers carry over, the
    iteration count and clock restart, and the clip is evaluated before and after."""
    from supermimic.finetune import FinetuneCfg, finetune

    _, base = _make(str(clips / "clips.txt"), num_envs=4, wm_only_iters=1)
    base.log_dir = str(tmp_path / "base")
    base.learn(num_learning_iterations=3)
    ck = str(tmp_path / "base" / "model_000003.pt")
    res = finetune(FinetuneCfg(checkpoint=ck, motion_file=str(clips / "clip1.npz"), out=str(tmp_path / "ft"),
                               iterations=2, num_envs=2, device="cpu"))  # fmt: skip
    assert {"before", "after"} <= set(res) and res["final_checkpoint"].endswith(
        "model_000002.pt"
    )
    assert len(res["after"]["per_clip"]) == 1 and math.isfinite(
        res["after"]["norm_return"]
    )
    ft = torch.load(res["final_checkpoint"], weights_only=False)
    base_ck = torch.load(ck, weights_only=False)
    assert (
        ft["iteration"] == 2
        and ft["cfg"]["finetune"]
        and ft["cfg"]["wm_only_iters"] == 0
    )
    assert float(ft["wm"]["in_norm.count"]) > float(
        base_ck["wm"]["in_norm.count"]
    )  # the baseline's statistics went on
    assert (tmp_path / "ft" / "finetune.json").exists()


def test_a_checkpoint_with_another_action_map_is_rejected(
    clips: Path, tmp_path: Path
) -> None:
    _, runner = _make(str(clips / "clip0.npz"))
    runner.log_dir = str(tmp_path)
    runner.save(str(tmp_path / "model.pt"))
    ck = torch.load(tmp_path / "model.pt", weights_only=False)
    ck["action_map"] = {"kind": "groups", "num_muscles": runner.nm}
    torch.save(ck, tmp_path / "groups.pt")
    with pytest.raises(NotImplementedError, match="groups"):
        runner.load(str(tmp_path / "groups.pt"))
    del ck["action_map"]  # a checkpoint from before action maps: identity
    torch.save(ck, tmp_path / "old.pt")
    runner.load(str(tmp_path / "old.pt"))


def test_train_checkpoint_resume_evaluate(clips: Path, tmp_path: Path) -> None:
    """The training entry point on a CPU: a short run writes checkpoints, a second run resumes the first, and the
    evaluation reads the last checkpoint with the settings it stores."""
    from supermimic.config import fast_runner_cfg
    from supermimic.env import TaskEnvCfg
    from supermimic.evaluate import (
        build,
        checkpoint_settings,
        evaluate,
    )
    from supermimic.train import TrainArgs, train

    agent = fast_runner_cfg(
        **{**TINY, "max_iterations": 3, "save_interval": 2, "wm_only_iters": 1},
        run_name="a",
    )
    env = TaskEnvCfg(motion_file=str(clips / "clips.txt"), num_envs=3)
    first = train(TrainArgs(env=env, agent=agent, log_root=str(tmp_path), device="cpu"))
    assert {p.name for p in first.glob("model_*.pt")} >= {
        "model_000002.pt",
        "model_000003.pt",
    }
    assert (first / "params" / "env.yaml").exists() and (
        first / "params" / "agent.yaml"
    ).exists()
    resumed = copy.deepcopy(agent)
    resumed.resume, resumed.load_run, resumed.run_name, resumed.max_iterations = (
        True,
        first.name,
        "b",
        2,
    )
    second = train(
        TrainArgs(env=env, agent=resumed, log_root=str(tmp_path), device="cpu")
    )
    last = second / "model_000005.pt"
    assert last.exists() and torch.load(last, weights_only=False)["iteration"] == 5
    settings = checkpoint_settings(str(last))
    ev_env, runner = build(str(clips / "clips.txt"), "cpu", settings=settings)
    runner.load(str(last), load_cfg={"actor": True})
    res = evaluate(ev_env, runner)
    assert len(res["per_clip"]) == 3 and math.isfinite(res["norm_return"])


def test_a_checkpoint_with_another_task_contract_is_rejected(
    clips: Path, tmp_path: Path
) -> None:
    """Equal widths are not enough: a reordered observation (or another model version) is refused unless accepted."""
    _, runner = _make(str(clips / "clip0.npz"))
    runner.log_dir = str(tmp_path)
    runner.save(str(tmp_path / "model.pt"))
    runner.load(str(tmp_path / "model.pt"))
    ck = torch.load(tmp_path / "model.pt", weights_only=False)
    terms = ck["contract"]["obs"]["policy"]
    terms[0], terms[1] = terms[1], terms[0]  # same widths in total, another order
    torch.save(ck, tmp_path / "swapped.pt")
    with pytest.raises(ValueError, match="obs"):
        runner.load(str(tmp_path / "swapped.pt"))
    terms[0], terms[1] = (
        terms[1],
        terms[0],
    )  # back to the env's order: only the version differs now
    ck["contract"]["model_version"] = "0.0.0"
    torch.save(ck, tmp_path / "version.pt")
    with pytest.raises(ValueError, match="model_version") as err:
        runner.load(str(tmp_path / "version.pt"))
    assert "obs" not in str(err.value)
    runner.cfg.accept_contract_mismatch = True
    runner.load(str(tmp_path / "swapped.pt"))
    runner.cfg.accept_contract_mismatch = False
    pre = {
        k: v
        for k, v in runner.task_contract().items()
        if k not in ("root", "motors", "effectors")
    }
    torch.save(
        {**ck, "contract": pre}, tmp_path / "pre_submodels.pt"
    )  # a contract from before submodels
    runner.load(str(tmp_path / "pre_submodels.pt"))
    del ck[
        "contract"
    ]  # a checkpoint from before contracts: its stored widths still match
    torch.save(ck, tmp_path / "old.pt")
    runner.load(str(tmp_path / "old.pt"))
    ck["env"]["obs_dim"] += 1
    torch.save(ck, tmp_path / "old_wide.pt")
    with pytest.raises(ValueError, match="obs_dim"):
        runner.load(str(tmp_path / "old_wide.pt"))
