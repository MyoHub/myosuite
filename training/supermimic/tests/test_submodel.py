# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SuperMimic on a fixed-base submodel with muscles and motors (MyoSuite's ``myoelbow2dof`` plus a direct and a
first-order motor, ``myosuite.tests.mimic_toy``): the runner scores states like the env, the graphed collector's
wrapper matches ``env.step`` + ``env.reset(ids)``, and the recipe learns and evaluates (with the effector metric)."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

pytestmark = pytest.mark.tier2

TINY = dict(
    wm_hidden=(64, 64, 64), enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16), critic_hidden=(32,), buffer_steps=40,
    warmup_steps=20, wm_batch=16, wm_batches=2, pi_batch=8, horizon=4, log_interval=1, save_interval=1000,
    compile=False,
)  # fmt: skip


@pytest.fixture(scope="module", name="toy")
def toy_fixture(tmp_path_factory: Any) -> tuple[str, Path]:
    """The mixed toy body and three clips of it (the last one fast, so tight thresholds fail on it)."""
    from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_body
    from myosuite.envs.myo.tasks.mimic.tracking.motion import write_synthetic_clip
    from myosuite.tests.mimic_toy import register_mixed_model

    name = register_mixed_model().name
    d = tmp_path_factory.mktemp("toy_clips")
    for i, (sec, amp) in enumerate(((4.0, 0.2), (3.0, 0.4), (3.0, 0.9))):
        write_synthetic_clip(
            d / f"clip{i}.npz", compiled_body(name), sec, 0.0, amp, i, model_name=name
        )
    (d / "clips.txt").write_text("clip0.npz\nclip1.npz\nclip2.npz\n")
    return name, d


def _make(model: str, clip_path: str, num_envs: int = 2, env_edit: Any = None, sampling: str = "start",
          **rl: Any) -> tuple[Any, Any]:  # fmt: skip
    from supermimic.config import fast_runner_cfg
    from supermimic.env import TaskEnvCfg, make_task_env
    from supermimic.runner import SuperMimicRunner

    torch.manual_seed(0)
    env = make_task_env(
        TaskEnvCfg(
            model=model,
            motion_file=clip_path,
            num_envs=num_envs,
            sampling_mode=sampling,
        ),
        "cpu",
    )
    if env_edit is not None:
        env_edit(env.unwrapped)
    return env, SuperMimicRunner(
        env, asdict(fast_runner_cfg(**{**TINY, **rl})), device="cpu"
    )


def test_fixed_base_runner_scores_like_the_env(toy: tuple[str, Path]) -> None:
    """Rewards, failure terms, observations and the actuator drive (muscle activations, motor commands, filtered
    motor activations) of the method match the env on the states the simulator reaches."""
    from myosuite.envs.myo.backends.mjlab.tasks.mdp import SYNC_TERM
    from supermimic.heads import StepCtx, analytic

    model, d = toy
    env, runner = _make(model, str(d / "clip0.npz"))
    u, cmd = env.unwrapped, runner.cmd
    assert (
        cmd.fixed_root
        and cmd.has_motors
        and runner.mirror is None
        and runner.num_touch == 0
    )
    assert cmd.num_actuators == 8 and cmd.info.num_muscles == 6
    names = set(u.reward_manager._term_names)
    assert "mimic_root_pos" not in names and "mimic_root_vel" not in names
    assert {t for t in u.termination_manager.active_terms if t.endswith("_dev")} == {
        "rel_site_dev"
    }
    u.reset()
    gen = torch.Generator().manual_seed(0)
    for _ in range(10):
        S0, a_prev = cmd.read_state(), u.action_manager.action.clone()
        torch.testing.assert_close(S0[0], torch.zeros_like(S0[0]))
        a = torch.rand(2, cmd.num_actuators, generator=gen)
        obs, rew, dones, _ = env.step(a)
        assert not dones.any()
        S1 = cmd.read_state()
        torch.testing.assert_close(
            S1[6], cmd.muscle.step(S0[6], a)[0], atol=1e-5, rtol=0
        )
        r, _ = runner._reward_term(S1, cmd.f, a, a_prev)
        torch.testing.assert_close(r, rew, atol=1e-4, rtol=1e-3)
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
        torch.testing.assert_close(
            runner.obs_policy.assemble(None, ctx), obs["policy"], atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            runner.obs_goal.assemble(None, ctx), obs["goal"], atol=1e-6, rtol=0
        )
    # imagination keeps the fixed root and evolves the joints
    S = cmd.read_state()
    hist = runner.obs_policy.init_history(StepCtx(S, cmd.f.clone(), a))
    S1, *_ = runner._imag_step(S, hist, cmd.f.clone(), a)
    for i in range(4):
        assert torch.equal(S1[i], S[i])
    assert S1[6].shape == (2, 8) and S1[7].shape == (2, 0)


def _short_episodes(u: Any) -> None:
    u.cfg.episode_length_s = 0.1  # time-outs after 5 steps
    u.termination_manager.get_term_cfg("rel_site_dev").params["threshold"] = 0.02


def test_fixed_base_collector_matches_the_eager_env_step(toy: tuple[str, Path]) -> None:
    from supermimic.collector import STATS

    model, d = toy
    pair = []
    for graph in (False, True):
        _, r = _make(model, str(d / "clips.txt"), num_envs=6, env_edit=_short_episodes,
                     buffer_steps=64, action_noise=0.0, action_noise_dist="gaussian", collect_graph=graph)  # fmt: skip
        pair.append(r)
    eager, graphed = pair
    with torch.no_grad():  # a constant policy (motors off their zero command)
        for w in eager.pi.parameters():
            w.zero_()
        eager.pi.dec[-1].bias.fill_(0.7)
    graphed.pi.load_state_dict(eager.pi.state_dict())
    n, stats = 40, []
    for r in pair:
        r.init_collection()
        st: dict = {k: [] for k in STATS}
        r.collect(n, st)
        stats.append(st)
    be, bg = eager.buffer, graphed.buffer
    tol = {"atol": 1e-5, "rtol": 1e-5}
    for key in ("done", "f", "ep_step"):
        assert torch.equal(be.data[key][:n], bg.data[key][:n]), key
    for key in ("s", "act", "a", "s1"):
        torch.testing.assert_close(
            bg.data[key][:n].float(), be.data[key][:n].float(), **tol, msg=key
        )
    torch.testing.assert_close(bg.data["r"][:n], be.data["r"][:n], atol=1e-4, rtol=1e-3)
    ends = int(bg.data["done"][:n].sum())
    term = sum(float(v) for v in stats[0]["term"])
    assert 0 < term < ends  # both failures and time-outs
    for key in ("f", "start_bin"):
        assert torch.equal(getattr(eager.cmd, key), getattr(graphed.cmd, key)), key
    de, dg = eager.uenv.sim.data, graphed.uenv.sim.data
    for key in ("qpos", "qvel", "act", "ctrl"):
        torch.testing.assert_close(
            getattr(dg, key)._tensor, getattr(de, key)._tensor, **tol, msg=key
        )


def test_fixed_base_learns_and_evaluates(toy: tuple[str, Path], tmp_path: Path) -> None:
    from supermimic.evaluate import build, checkpoint_settings, evaluate

    model, d = toy
    env, runner = _make(model, str(d / "clips.txt"), num_envs=3, wm_only_iters=1)
    runner.log_dir = str(tmp_path)
    runner.learn(num_learning_iterations=4)
    assert runner.it == 4 and int(runner.nonfinite) == 0
    runner.save(str(tmp_path / "model.pt"))
    settings = checkpoint_settings(str(tmp_path / "model.pt"))
    assert settings["model"] == model and math.isclose(settings["ctrl_dt"], 0.02)
    ev_env, ev = build(str(d / "clips.txt"), "cpu", settings=settings)
    ev.load(str(tmp_path / "model.pt"), load_cfg={"actor": True})
    res = evaluate(ev_env, ev)
    assert (
        len(res["per_clip"]) == 3
        and math.isfinite(res["tip_err_cm"])
        and math.isfinite(res["mpjpe_mm"])
    )


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "collector"])
def test_noise_free_envs_collect_the_deterministic_policy(
    toy: tuple[str, Path], graph: bool
) -> None:
    """noise_free_frac: the first envs act with the policy's own action, the others explore; their episodes are logged
    as det_ret / det_len."""
    from supermimic.collector import STATS

    model, d = toy
    _, r = _make(model, str(d / "clips.txt"), num_envs=8, env_edit=_short_episodes, noise_free_frac=0.25,
                 action_noise=0.2, collect_graph=graph)  # fmt: skip
    assert r.det_mask.tolist() == [True] * 2 + [False] * 6 and r.stat_keys == STATS + (
        "det_ret",
        "det_len",
    )
    with torch.no_grad():  # a constant policy
        for w in r.pi.parameters():
            if w.dim() == 2:
                w.zero_()
    r.init_collection()
    st: dict = {k: [] for k in r.stat_keys}
    r.collect(20, st)
    a = r.buffer.data["a"][:20].float()
    # a zero policy (constant action): the deterministic envs execute it exactly, the others explore around it
    const = torch.sigmoid(r.pi.dec[-1].bias).detach()
    torch.testing.assert_close(
        a[:, :2], const.expand(20, 2, -1), atol=2e-3, rtol=0
    )  # float16 buffer
    assert (a[:, 2:] - const).abs().mean() > 0.02
    det_rows = [float(x) for x in st["det_ret"] if x == x]
    assert det_rows and all(0.0 <= x <= 1.5 for x in det_rows)


def test_noise_free_actions_are_the_policy_mean() -> None:
    """explore_actions: the masked rows are the clamped action, the others drawn around it."""
    from types import SimpleNamespace

    from supermimic.runner import SuperMimicRunner

    r = SimpleNamespace(cfg=SimpleNamespace(action_noise=0.2, action_noise_dist="beta"),
                        noise_t=torch.tensor(0.2), det_mask=torch.tensor([True, False, False]))  # fmt: skip
    a = torch.full((3, 50), 0.3)
    out = SuperMimicRunner.explore_actions(r, a)
    assert (
        torch.equal(out[0], a[0])
        and not torch.equal(out[1], a[1])
        and (out >= 0).all()
        and (out <= 1).all()
    )
    r.det_mask = None
    assert not torch.equal(SuperMimicRunner.explore_actions(r, a)[0], a[0])


def test_budget_schedules(toy: tuple[str, Path], tmp_path: Path) -> None:
    """budget_schedules: the learning rate and the action noise follow a cosine over the time budget (3 % and
    action_noise_final at its end); without it the iteration schedule is unchanged."""
    import csv

    model, d = toy
    _, plain = _make(model, str(d / "clip0.npz"))
    assert plain.budget_progress(10.0) is None and plain.noise_t is None
    assert plain.lr_at(0) == plain.cfg.lr
    _, r = _make(model, str(d / "clip0.npz"), budget_schedules=True, max_hours=1.0, action_noise=0.1,
                 action_noise_final=0.02, wm_only_iters=1)  # fmt: skip
    c = r.cfg
    assert (
        r.budget_progress(0.0) == 0.0
        and r.budget_progress(1800.0) == 0.5
        and r.budget_progress(7200.0) == 1.0
    )
    assert math.isclose(r.lr_at(5, 0.0), c.lr) and math.isclose(
        r.lr_at(5, 1.0), 0.03 * c.lr
    )
    assert math.isclose(r.lr_at(5, 0.5), c.lr * (0.03 + 0.97 * 0.5))
    assert math.isclose(r.noise_at(0.0), 0.1) and math.isclose(r.noise_at(1.0), 0.02)
    r.log_dir = str(tmp_path)
    r.learn(num_learning_iterations=3)
    rows = list(csv.DictReader(open(tmp_path / "train_log.csv")))
    assert rows and all(0.02 <= float(row["action_noise"]) <= 0.1 for row in rows)
    assert all(float(row["lr"]) <= c.lr for row in rows)
    assert math.isclose(
        float(r.noise_t), r.noise_at(r.budget_progress(0.0) or 0.0), rel_tol=0.05
    )
