# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``myoMimic-v0``: pure tracking terms, motion library, CPU env, mjlab twin, metrics.

``data/mimic_track_golden.npz`` holds the outputs of the mjlab-only task this one
replaces (``myoMimicFullbodyMulticlip-v0``, PR #573 at deb2ef49: its torch
``MotionLib`` and ``heads``) on a synthetic clip (:func:`write_synthetic_clip`, 1.5 s,
seed 0) at a 0.02 s control step, and on 16 perturbed states of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.envs.myo.tasks.mimic.tracking import MIMIC_ENV_ID
from myosuite.envs.myo.tasks.mimic.tracking.config import (
    PRESETS,
    MimicTaskCfg,
    RewardCfg,
    RewardTermCfg,
    resolve_task,
)
from myosuite.envs.myo.tasks.mimic.tracking.metrics import METRICS, evaluate, summarize
from myosuite.envs.myo.tasks.mimic.tracking.models import (
    compiled_body,
    model_cfg,
    root_offset,
)
from myosuite.envs.myo.tasks.mimic.tracking.motion import (
    build_motion_lib,
    write_synthetic_clip,
)
from myosuite.envs.myo.tasks.mimic.tracking.sampling import ResetSampler
from myosuite.terms import mimic_track as mt

pytestmark = pytest.mark.tier2

GOLDEN = Path(__file__).parent / "data" / "mimic_track_golden.npz"
KERNELS = {"qpos": 10, "qvel": 2, "root_pos": 10, "root_vel": 10, "rpos": 100,
           "rquat": 10, "rvel_rot": 0.1, "rvel_lin": 0.1}  # fmt: skip
POLICY_DIM = 1079
GOAL_DIM = 277


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("clips") / "synthetic.npz"
    write_synthetic_clip(path, compiled_body("myofullbody"), 1.5, 1.0, 0.3, 0)
    return str(path)


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    return dict(np.load(GOLDEN))


@pytest.fixture(scope="module")
def lib(clip: str):
    mc = model_cfg("myofullbody")
    return build_motion_lib(
        [clip], compiled_body("myofullbody"), mc.tracked_bodies, 0.02
    )


def _env(clip: str, **overrides):
    return make_env(
        EnvConfig(
            MIMIC_ENV_ID, task_kwargs={"overrides": {"motion.clips": clip, **overrides}}
        )
    )


# --------------------------------------------------------------------------- terms


def test_motion_lib_matches_573(lib, golden) -> None:
    """MuJoCo FK (float64) reproduces #573's torch FK (float32)."""
    for k in (
        "clip_len",
        "clip_start",
        "clip_end",
        "bin_start",
        "bin_clip",
        "frame_bin",
    ):
        np.testing.assert_array_equal(getattr(lib, k), golden[k], err_msg=k)
    R_off = root_offset(compiled_body("myofullbody"))
    fr = lib.frames
    pairs = {
        "root_pos": "root_p", "joint_pos": "ref_q", "root_qvel": "ref_root_qvel",
        "root_vel_local": "ref_root_vel", "site_rpos": "ref_rpos",
        "site_rangles": "ref_rangles", "body_pos": "body_p",
    }  # fmt: skip
    for ours, theirs in pairs.items():
        np.testing.assert_allclose(fr[ours], golden[theirs], atol=5e-6, err_msg=ours)
    np.testing.assert_allclose(fr["root_rot"], golden["root_R"] @ R_off.T, atol=5e-6)
    for ours, theirs in (("joint_vel", "ref_qd"), ("site_rvel", "ref_rvel"),
                         ("body_lin_vel", "body_v")):  # fmt: skip
        np.testing.assert_allclose(fr[ours], golden[theirs], atol=1e-4, err_msg=ours)


def _golden_state(g: dict[str, np.ndarray]) -> mt.TrackState:
    n = len(g["S_f"])
    return mt.TrackState(
        root_pos=g["S_p"], root_quat=mt.mat_to_quat(np, g["S_Rp"]),
        root_qvel=g["S_root_qvel"], joint_pos=g["S_qf"], joint_vel=g["S_qdf"],
        body_pos=g["S_P"], body_rot=g["S_R"], body_lin_vel=g["S_V"], body_ang_vel=g["S_W"],
        act=np.zeros((n, 354)), excitation=np.zeros((n, 354)), touch=np.zeros((n, 4)),
    )  # fmt: skip


def test_pure_terms_match_573(golden) -> None:
    """Rewards, failure rule and observations equal #573's heads on the same states."""
    g = golden
    f = g["S_f"]
    s = _golden_state(g)
    feats = mt.features(np, s)
    R_off = root_offset(compiled_body("myofullbody"))
    r = mt.TrackRef(
        root_pos=g["root_p"][f], root_rot=g["root_R"][f] @ R_off.T,
        root_qvel=g["ref_root_qvel"][f], root_vel_local=g["ref_root_vel"][f],
        joint_pos=g["ref_q"][f], joint_vel=g["ref_qd"][f], site_rpos=g["ref_rpos"][f],
        site_rangles=g["ref_rangles"][f], site_rvel=g["ref_rvel"][f],
    )  # fmt: skip
    for name, k in KERNELS.items():
        np.testing.assert_allclose(
            mt.reward(np, s, feats, r, name, k), g[f"R_{name}"], atol=2e-6, err_msg=name
        )
    for ours, theirs in (("rel_site", "bad_body_pos_dev"), ("root_pos", "bad_root_pos_dev"),
                         ("root_ori", "bad_root_ori_dev")):  # fmt: skip
        np.testing.assert_array_equal(
            mt.exceeds(np, s, feats, r, ours, 0.3), g[f"T_{theirs}"]
        )
    for name in ("root_pos", "site_rpos", "site_rangles", "site_rvel"):
        np.testing.assert_allclose(
            mt.observation(np, s, feats, name), g[f"O_{name}"], atol=2e-6, err_msg=name
        )


def test_goal_matches_573(lib, golden) -> None:
    f = golden["S_f"]
    fut = np.minimum(
        f[:, None] + 10 * np.arange(1, 5), lib.clip_end[lib.clip_id[f]][:, None]
    )
    goal = mt.goal_observation(np, lib.ref(f), lib.ref(fut), lib.phase(f))
    np.testing.assert_allclose(goal, golden["goal"], atol=5e-5)


def test_pure_terms_torch_differentiable(golden) -> None:
    """Every reward and deviation has finite gradients in the body state (torch)."""
    torch = pytest.importorskip("torch")
    g = golden
    s = _golden_state(g)
    s = mt.TrackState(**{k: torch.tensor(np.asarray(v), dtype=torch.float64,
                                         requires_grad=k in ("body_pos", "joint_pos", "root_quat"))
                         for k, v in vars(s).items() if v is not None})  # fmt: skip
    f = g["S_f"]
    ref = mt.TrackRef(
        root_pos=g["root_p"][f], root_rot=g["root_R"][f], root_qvel=g["ref_root_qvel"][f],
        root_vel_local=g["ref_root_vel"][f], joint_pos=g["ref_q"][f], joint_vel=g["ref_qd"][f],
        site_rpos=g["ref_rpos"][f], site_rangles=g["ref_rangles"][f], site_rvel=g["ref_rvel"][f],
    )  # fmt: skip
    ref = mt.TrackRef(
        **{
            k: torch.tensor(np.asarray(v), dtype=torch.float64)
            for k, v in vars(ref).items()
        }
    )
    feats = mt.features(torch, s)
    total = sum(mt.reward(torch, s, feats, ref, n, k).sum() for n, k in KERNELS.items())
    total = total + sum(
        mt.deviation(torch, s, feats, ref, n).sum() for n in mt.DEVIATIONS
    )
    grads = torch.autograd.grad(total, (s.body_pos, s.joint_pos, s.root_quat))
    assert all(torch.isfinite(x).all() and x.abs().sum() > 0 for x in grads)


# --------------------------------------------------------------------------- config


def test_presets_and_overrides() -> None:
    cfg = resolve_task("supermimic", {"motion.clips": "a.npz", "reset.mode": "uniform"})
    assert cfg.motion.clips == "a.npz" and cfg.reset.mode == "uniform"
    assert cfg.fingerprint() != PRESETS["supermimic"].fingerprint()
    assert isinstance(resolve_task(), MimicTaskCfg)
    assert "muscle_length" not in PRESETS["supermimic"].obs.policy
    with pytest.raises(KeyError):
        resolve_task("supermimic", {"motion.nope": 1})
    with pytest.raises(KeyError):
        resolve_task("nope")


def test_model_keyword() -> None:
    with pytest.raises(ValueError, match="choose from"):
        make_env(MIMIC_ENV_ID, model="musclemimic")
    with pytest.raises(ValueError, match="fixed model"):
        make_env("myoElbowPose1D6MRandom-v0", model="myofullbody")


# --------------------------------------------------------------------------- CPU env


def test_cpu_env_contract(clip: str) -> None:
    env = _env(clip)
    u = env.unwrapped
    assert u.observation_space.shape == (POLICY_DIM + GOAL_DIM,)
    assert u.action_space.shape == (354,) and u._ctrl_dt == pytest.approx(0.02)
    obs, _ = env.reset(seed=3)
    assert obs.dtype == np.float32 and np.isfinite(obs).all()
    obs2, _ = _env(clip).reset(seed=3)
    np.testing.assert_array_equal(obs, obs2)
    out = env.step(np.full(354, 0.1))
    assert len(out) == 5 and {f"r_{k}" for k in KERNELS} <= set(out[4])
    assert 0 < out[1] <= u.max_step_reward


def test_cpu_env_reference_start_and_clip_end(clip: str) -> None:
    """Started on the reference, the first transition scores near the maximum; the
    episode truncates at the clip end."""
    env = _env(
        clip,
        **{"failure.rel_site": 1e9, "failure.root_pos": 1e9, "failure.root_ori": 1e9},
    )
    u = env.unwrapped
    env.reset(seed=0, options={"clip": 0})
    s = u.track_state()
    r = mt.reward(np, s, mt.features(np, s), u.motion.ref(u.frame), "rpos", 100.0)
    assert r > 0.999
    steps, truncated = 0, False
    while not truncated:
        _, _, terminated, truncated, info = env.step(np.zeros(354))
        assert not terminated
        steps += 1
    assert steps == u.motion.clip_len[0] - 1 and info["clip_end"]


def test_ctrl_dt_keeps_return_per_second(clip: str) -> None:
    cfg = EnvConfig(MIMIC_ENV_ID, task_kwargs={"overrides": {"motion.clips": clip}})
    fast = make_env(cfg, ctrl_dt=0.01)
    slow = _env(clip)
    assert fast.unwrapped.frame_skip == 5 and slow.unwrapped.frame_skip == 10
    assert fast.unwrapped.max_step_reward == pytest.approx(
        slow.unwrapped.max_step_reward / 2
    )


def test_reset_sampler(lib) -> None:
    """Return-based resets favour poorly tracked bins; fail resets the bin before a failure."""
    import dataclasses  # noqa: PLC0415

    cfg = dataclasses.replace(PRESETS["supermimic"].reset, fail_reset=0.0)
    sampler = ResetSampler(lib, cfg, max_step_reward=0.24)
    p = sampler.probs()
    assert p.sum() == pytest.approx(1.0) and len(p) == lib.num_bins
    sampler.finish_episode(0, 10, 10, 20.0, 2.4, terminated=False)
    assert sampler.probs()[1] > sampler.probs()[0]
    sampler = ResetSampler(
        lib, dataclasses.replace(cfg, fail_reset=0.3), max_step_reward=0.24
    )
    sampler.finish_episode(1, lib.bin_start[1] + 5, 5, 0.1, 0.1, terminated=True)
    p = sampler.probs()
    assert p[0] > p[1]
    frame, b = sampler.sample(np.random.default_rng(0))
    assert lib.bin_start[b] <= frame <= lib.clip_end[lib.bin_clip[b]]


def test_metrics_report(clip: str) -> None:
    env = _env(clip, episode_length_s=None)
    rep = evaluate(lambda obs: np.full(354, 0.05), env)
    assert rep["n_clips"] == 1 and 0 <= rep["norm_return"] <= 1
    assert (
        rep["tracked_to_end"] in (0.0, 1.0)
        and rep["tracked_to_end_05m"] <= rep["tracked_to_end"]
    )
    for k in ("mpjpe_mm", "g_mpjpe_mm", "joint_rmse_deg", "root_ori_rad"):
        assert np.isfinite(rep[k])


def test_metrics_report_without_flagged_clips() -> None:
    keys = {k for k, _ in METRICS.values()}
    sums = {k: np.array([1.0, 1.0, 9.0]) for k in keys}
    args = (np.ones(3), np.array([1.0, 1.0, 0.0]), np.ones(3),
            np.array([False, False, True]), np.array([False, False, True]))  # fmt: skip
    rep = summarize(["a", "b", "c"], sums, *args, flagged=[False, False, True])
    assert rep["flagged"] == ["c"] and rep["tracked_to_end"] == pytest.approx(2 / 3)
    sub = rep["unflagged"]
    assert sub["n_clips"] == 2 and sub["tracked_to_end"] == 1.0
    assert sub["mpjpe_mm"] == pytest.approx(1000.0) and "per_clip" not in sub
    assert "unflagged" not in summarize(["a", "b", "c"], sums, *args)


# --------------------------------------------------------------------------- mjlab twin


def test_mjlab_twin_parity(clip: str) -> None:
    """From the same reference start, the twin observes and scores like the CPU env.

    Open loop for a few steps (float32 Warp vs float64 MuJoCo); the foot contact
    forces get a looser bound (N).
    """
    pytest.importorskip("mjlab")
    torch = pytest.importorskip("torch")

    tk = {"overrides": {"motion.clips": clip, "reset.mode": "start"}}
    mj = make_env(
        EnvConfig(MIMIC_ENV_ID, backend="mjlab", num_envs=1, task_kwargs=tk),
        device="cpu",
    )
    cpu = make_env(EnvConfig(MIMIC_ENV_ID, task_kwargs=tk)).unwrapped
    obs_mj, _ = mj.reset()
    obs_cpu, _ = cpu.reset(seed=0, options={"clip": 0})
    assert (
        obs_mj["policy"].shape[-1] == POLICY_DIM
        and obs_mj["goal"].shape[-1] == GOAL_DIM
    )
    np.testing.assert_allclose(
        torch.cat((obs_mj["policy"][0], obs_mj["goal"][0])).numpy(), obs_cpu, atol=1e-5
    )
    tol = {"touch": 0.5, "joint_vel": 0.1, "root_vel": 0.1, "site_rvel": 0.1}
    keys = np.concatenate([[k] * n for k, n in cpu.obs_dims.items()])
    a = np.full(354, 0.1, np.float32)
    for _ in range(5):
        obs_cpu, r_cpu, term_cpu, _, _ = cpu.step(a)
        obs_mj, r_mj, term_mj, _, _ = mj.step(torch.tensor(a)[None])
        o_mj = torch.cat((obs_mj["policy"][0], obs_mj["goal"][0])).numpy()
        assert float(r_mj[0]) == pytest.approx(r_cpu, rel=1e-4)
        assert bool(term_mj[0]) == term_cpu
        for k in cpu.obs_dims:
            d = np.abs(o_mj - obs_cpu)[keys == k].max()
            assert d <= tol.get(k, 2e-3), (k, d)
    pure = [
        t.func
        for group in (mj.cfg.observations["policy"].terms, mj.cfg.rewards)
        for t in group.values()
    ] + [t.func for k, t in mj.cfg.terminations.items() if k.endswith("_dev")]
    assert all(hasattr(fn, "pure") for fn in pure)


@pytest.mark.parametrize("backend", ["cpu", "mjlab"])
def test_features_run(clip: str, backend: str) -> None:
    """Muscle fatigue and motor noise run on both halves."""
    from myosuite.envs.wrappers import FatigueWrapper, MotorNoiseWrapper  # noqa: PLC0415

    if backend == "mjlab":
        pytest.importorskip("mjlab")
    feats = (
        FatigueWrapper,
        (MotorNoiseWrapper, {"motor_noise": {"constant_std": 0.05}}),
    )
    cfg = EnvConfig(
        MIMIC_ENV_ID, backend=backend, features=feats,
        task_kwargs={"overrides": {"motion.clips": clip}},
        **({"num_envs": 2} if backend == "mjlab" else {}),
    )  # fmt: skip
    env = make_env(cfg, **({"device": "cpu"} if backend == "mjlab" else {}))
    env.reset()
    if backend == "cpu":
        env.step(np.full(354, 0.2))
    else:
        import torch  # noqa: PLC0415

        env.step(torch.full((2, 354), 0.2))


def _root_reward(state: mt.TrackState, ref: mt.TrackRef, ctrl_dt: float):
    """A method's own reward: root tracking per second (pure, NumPy or torch)."""
    xp = mt.array_module(state)
    return xp.exp(-10.0 * ((state.root_pos - ref.root_pos) ** 2).sum(-1)) * ctrl_dt


def test_custom_reward_cpu(clip: str) -> None:
    """``reward_fn`` replaces the step reward; ``task_reward`` stays the proposed one."""
    base = _env(clip)
    custom = _env(clip, reward_fn=_root_reward)
    whole = make_env(  # a whole replacement RewardCfg from library components
        EnvConfig(MIMIC_ENV_ID, task_kwargs={"overrides": {
            "motion.clips": clip, "reward": RewardCfg(terms=(RewardTermCfg("rpos", 1.0, 100.0),), scale=1.0)}})
    )  # fmt: skip
    for env in (base, custom, whole):
        env.reset(seed=0, options={"clip": 0})
    for _ in range(3):
        a = np.full(354, 0.1)
        _, r0, *_, i0 = base.step(a)
        _, r1, *_, i1 = custom.step(a)
        _, r2, *_, i2 = whole.step(a)
        assert (
            r0 == pytest.approx(i0["task_reward"]) == pytest.approx(i1["task_reward"])
        )
        u = custom.unwrapped
        s = u.track_state()
        expect = _root_reward(
            s, u.motion.ref(min(u.frame, u.motion.clip_end[0])), u._ctrl_dt
        )
        assert r1 == pytest.approx(float(expect), rel=1e-6) and r1 != pytest.approx(r0)
        assert r2 == pytest.approx(i2["r_rpos"] * 0.02 / 0.01)


def test_custom_reward_mjlab_matches_cpu(clip: str) -> None:
    """The twin uses the same ``reward_fn`` and reports the proposed reward as ``task_reward``."""
    pytest.importorskip("mjlab")
    torch = pytest.importorskip("torch")

    def make(backend: str, **ov):
        tk = {"overrides": {"motion.clips": clip, "reset.mode": "start", **ov}}
        kw = {"device": "cpu"} if backend == "mjlab" else {}
        return make_env(EnvConfig(MIMIC_ENV_ID, backend=backend, task_kwargs=tk,
                                  **({"num_envs": 1} if backend == "mjlab" else {})), **kw)  # fmt: skip

    mj_base, mj = make("mjlab"), make("mjlab", reward_fn=_root_reward)
    cpu = make("cpu", reward_fn=_root_reward)
    for e in (mj_base, mj):
        e.reset()
    cpu.reset(seed=0, options={"clip": 0})
    a = np.full(354, 0.1, np.float32)
    for _ in range(3):
        _, r_cpu, *_, info = cpu.step(a)
        _, r_base, *_ = mj_base.step(torch.tensor(a)[None])
        _, r_mj, *_ = mj.step(torch.tensor(a)[None])
        assert float(r_mj[0]) == pytest.approx(r_cpu, rel=1e-4)
        task_reward = mj.unwrapped.command_manager.get_term("motion").task_reward
        assert float(task_reward[0]) == pytest.approx(float(r_base[0]), rel=1e-5)
        assert float(task_reward[0]) == pytest.approx(info["task_reward"], rel=1e-4)
    assert "task_reward" in mj.unwrapped.command_manager.get_term("motion").metrics
