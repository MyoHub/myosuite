# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``myoMimicFullbodyMulticlip-v0``: its analytic contracts against MuJoCo, and the env built by ``make_env``.

Torch kinematics, activation dynamics and muscle geometry of myo_sim's ``myofullbody`` against MuJoCo, with
gradients; the managers' observation, reward and termination against the analytic terms they wrap, on the states the
simulator reaches; the reset statistics; the ``make_env`` options. Synthetic clips stand in for MuscleMimic's KIT
clips (a gated dataset). Ported in part from muscle-supermimic's tests (github.com/kaibohex/muscle-supermimic @
8a6244b, Apache-2.0).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.tests.support.optional_deps import require_mjlab, require_mujoco_warp

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tier2

PKG = "myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip"


def _model() -> Any:
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.body import (
        compiled_body,
    )

    return compiled_body()


@pytest.fixture(scope="module")
def clips(tmp_path_factory: Any) -> Path:
    """Three synthetic clips of 8, 7 and 6 s, and a list of them. The last one rises at 1 m/s: the body, reset onto
    it, falls behind it under gravity. The second names its elbows as MuscleMimic's body does."""
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.motion import (
        write_synthetic_clip,
    )

    m = _model()
    d = tmp_path_factory.mktemp("clips")
    for i, (sec, speed, climb) in enumerate(
        ((8.0, 0.0, 0.0), (7.0, 0.5, 0.0), (6.0, 1.0, 1.0))
    ):
        write_synthetic_clip(d / f"clip{i}.npz", m, sec, speed, 0.05, i, climb)
    data = dict(np.load(d / "clip1.npz"))
    data["joint_names"] = np.array(
        [str(n).replace("elbow_flexion", "elbow_flex") for n in data["joint_names"]]
    )
    np.savez(d / "clip1.npz", **data)
    (d / "clips.txt").write_text("# synthetic\nclip0.npz\nclip1.npz\nclip2.npz\n")
    return d


def make_task_env(clip_path: str, num_envs: int = 2, **task_kwargs: Any) -> Any:
    """The task through ``make_env`` on the CPU."""
    require_mjlab()
    require_mujoco_warp()
    from myosuite import make_env
    from myosuite.core.config import EnvConfig
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip import ENV_ID

    torch.manual_seed(0)
    cfg = EnvConfig(
        ENV_ID,
        backend="mjlab",
        num_envs=num_envs,
        task_kwargs={"motion_file": clip_path, "sampling_mode": "start", **task_kwargs},
    )
    return make_env(cfg, device="cpu", seed=0)


@pytest.fixture(
    scope="module", params=[False, True], ids=["no_muscle_obs", "muscle_obs"]
)
def env(request: Any, clips: Path) -> Any:
    return make_task_env(str(clips / "clip0.npz"), muscle_obs=request.param)


# ---------------------------------------------------------------- the body, kinematics, muscles


def test_body_is_myo_sims_myofullbody() -> None:
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.model_info import (
        ModelInfo,
        root_offset,
    )

    m = _model()
    info = ModelInfo(m)
    # fingers disabled: 83 joints (root included), 354 muscles, 51 equalities (2 locks), 31 independent DoF
    assert (m.njnt, m.nu, m.neq, len(info.coupled), len(info.indep_ids)) == (
        83,
        354,
        51,
        51,
        31,
    )
    assert len(info.couplings.locked_qpos) == 2 and info.couplings.level_ends == (49,)
    assert m.joint("elbow_flexion_r").id > 0  # myo_sim's names
    assert (
        m.nkey == 0 and m.nlight == 0 and not any(m.geom_bodyid == 0)
    )  # no scene left
    assert (m.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH).all() and m.nsensor == 4
    assert m.npair == 75 and not any(
        m.site(i).name.endswith("_mimic") for i in range(m.nsite)
    )
    np.testing.assert_allclose(
        root_offset(m), [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-6
    )


def _full_state(
    m: Any, info: Any, q: np.ndarray, qd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Independent coordinates -> every non-root joint (MuJoCo order), coupled ones by MyoSuite's projection."""
    from myosuite.physics.joint_coupling import project_joint_couplings

    qpos, qvel = np.tile(m.qpos0, (len(q), 1)), np.zeros((len(q), m.nv))
    qpos[:, m.jnt_qposadr[info.indep_ids]] = q
    qvel[:, m.jnt_dofadr[info.indep_ids]] = qd
    project_joint_couplings(info.couplings, qpos, qvel)
    return qpos[:, 7:], qvel[:, 6:]


def test_fk_matches_mujoco_and_is_differentiable() -> None:
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.body import (
        TRACKED_BODIES,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.kinematics import (
        SkeletonKinematics,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.model_info import (
        ModelInfo,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.rotations import so3_exp

    m = _model()
    info = ModelInfo(m)
    kin = SkeletonKinematics(TRACKED_BODIES, "cpu", m, dtype=torch.float64)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    n = 8
    q = rng.uniform(info.q_lo, info.q_hi, size=(n, len(info.indep_ids)))
    qd = rng.normal(0, 2.0, size=q.shape)
    p, v, w = (
        rng.normal(0, 1.0, (n, 3)),
        rng.normal(0, 1.0, (n, 3)),
        rng.normal(0, 2.0, (n, 3)),
    )
    R = so3_exp(torch.tensor(rng.normal(0, 1.5, size=(n, 3)))).numpy()
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    bp, bR, bv, bw = kin.forward(t(p), t(R), t(v), t(w), t(q), t(qd))
    qf, qdf = _full_state(m, info, q, qd)
    jq, jqd = kin.full_joint_state(t(q), t(qd))
    np.testing.assert_allclose(jq.numpy(), qf, atol=1e-12)
    np.testing.assert_allclose(jqd.numpy(), qdf, atol=1e-12)
    Rb = (t(R) @ kin.R_off.T).numpy()
    for i in range(n):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, Rb[i].reshape(-1))
        d.qpos[:7] = np.concatenate((p[i], quat))
        d.qvel[:3], d.qvel[3:6] = v[i], Rb[i].T @ w[i]
        d.qpos[7:], d.qvel[6:] = qf[i], qdf[i]
        mujoco.mj_forward(m, d)
        for k, name in enumerate(TRACKED_BODIES):
            b = m.body(name).id
            np.testing.assert_allclose(
                bp[i, k].numpy(), d.xpos[b], atol=1e-9, err_msg=name
            )
            np.testing.assert_allclose(
                bR[i, k].numpy(), d.xmat[b].reshape(3, 3), atol=1e-9, err_msg=name
            )
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)
            np.testing.assert_allclose(
                bw[i, k].numpy(), vel[:3], atol=1e-8, err_msg=name
            )
            np.testing.assert_allclose(
                bv[i, k].numpy(), vel[3:], atol=1e-8, err_msg=name
            )
    # the coupled joints satisfy the model's equalities (positions and velocities)
    d.qpos[:], d.qvel[:] = m.qpos0, 0.0
    d.qpos[7:], d.qvel[6:] = qf[0], qdf[0]
    mujoco.mj_forward(m, d)
    rows = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    assert rows.sum() == len(info.coupled)
    np.testing.assert_allclose(d.efc_pos[rows], 0.0, atol=1e-12)
    np.testing.assert_allclose(d.efc_vel[rows], 0.0, atol=1e-10)
    # the default pose is the canonical frame: upright, facing +x
    R0 = kin.canonical_root(torch.tensor(m.qpos0[3:7], dtype=torch.float64))
    torch.testing.assert_close(R0, torch.eye(3, dtype=torch.float64), atol=1e-6, rtol=0)
    # gradients through the couplings and the chain of bodies match finite differences
    qg, qdg = t(q[:2]).requires_grad_(), t(qd[:2]).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a, b: kin.forward(t(p[:2]), t(R[:2]), t(v[:2]), t(w[:2]), a, b)[0].sum(
            -1
        ),
        (qg, qdg),
    )


def test_activation_matches_mujoco() -> None:
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.activation import (
        MuscleActivation,
    )

    m = copy.copy(_model())
    m.opt.timestep = 0.002
    substeps = 10
    d = mujoco.MjData(m)
    d.qpos[2] = 1.5  # in the air
    rng = np.random.default_rng(0)
    mus = MuscleActivation(m, substeps, m.opt.timestep, "cpu")
    act = torch.tensor(rng.uniform(0, 1, m.na))
    d.act[:] = act.numpy()
    for _ in range(3):
        ctrl = rng.uniform(0, 1, m.nu)
        d.ctrl[:] = ctrl
        mean_mj = np.zeros(m.na)
        for _ in range(substeps):
            mean_mj += d.act / substeps
            mujoco.mj_step(m, d)
        act, mean = mus.step(act.float(), torch.tensor(ctrl, dtype=torch.float32))
        np.testing.assert_allclose(act.numpy(), d.act, atol=1e-6)
        np.testing.assert_allclose(mean.numpy(), mean_mj, atol=1e-6)
        act = torch.tensor(d.act.copy())


def _grazing(m: Any, d: Any, tol: float = 1e-5) -> np.ndarray:
    """Muscles with a wrap whose straight segment passes within ``tol`` of the wrap surface (the wrap switches there,
    and MuJoCo and MuJoCo-Warp, which the torch code follows, differ by microns)."""
    out = np.zeros(m.nu, dtype=bool)
    for a in range(m.nu):
        t = m.actuator_trnid[a, 0]
        for k in range(m.tendon_adr[t] + 1, m.tendon_adr[t] + m.tendon_num[t] - 1):
            if m.wrap_type[k] == mujoco.mjtWrap.mjWRAP_SITE:
                continue
            g = m.wrap_objid[k]
            R, c = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            p0, p1 = (R.T @ (d.site_xpos[m.wrap_objid[j]] - c) for j in (k - 1, k + 1))
            if m.wrap_type[k] == mujoco.mjtWrap.mjWRAP_CYLINDER:
                p0, p1 = p0[:2], p1[:2]
            u = p1 - p0
            closest = p0 + np.clip(-p0 @ u / (u @ u), 0, 1) * u
            out[a] |= abs(np.linalg.norm(closest) - m.geom_size[g, 0]) < tol
    return out


def test_muscle_geometry_matches_mujoco() -> None:
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.kinematics import (
        SkeletonKinematics,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.model_info import (
        ModelInfo,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.muscle_geometry import (
        MuscleGeometry,
    )

    m = _model()
    info = ModelInfo(m)
    geo = MuscleGeometry(m, "cpu", torch.float64)
    kin = SkeletonKinematics(("pelvis",), "cpu", m, torch.float64)
    rng = np.random.default_rng(0)
    d = mujoco.MjData(m)
    n, wrapped, checked = 8, 0, 0
    q = torch.tensor(rng.uniform(info.q_lo, info.q_hi, (n, len(info.indep_ids))))
    qd = torch.tensor(rng.normal(0, 2.0, q.shape))
    act = torch.tensor(rng.uniform(0, 1, (n, m.nu)))
    L, Ld = geo.length_velocity(q, qd)
    F = geo.force(L, Ld, act)
    qf, qdf = kin.full_joint_state(q, qd)
    for i in range(n):
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[7:] = qf[i].numpy()
        d.qvel[:] = 0
        d.qvel[6:] = qdf[i].numpy()
        d.act[:] = act[i].numpy()
        mujoco.mj_forward(m, d)
        ok = ~_grazing(m, d)
        checked += int(ok.sum())
        np.testing.assert_allclose(
            L[i].numpy()[ok], d.actuator_length[ok], atol=1e-9, rtol=0
        )
        np.testing.assert_allclose(
            Ld[i].numpy()[ok], d.actuator_velocity[ok], atol=1e-8, rtol=0
        )
        np.testing.assert_allclose(
            F[i].numpy()[ok], d.actuator_force[ok], atol=1e-6, rtol=1e-8
        )
        wrapped += int((d.wrap_xpos.reshape(-1, 6)[:, 3:].any(-1)).sum())
    assert wrapped > 0 and checked > 0.95 * n * m.nu
    qg = q.clone().requires_grad_()
    Lg, Ldg = geo.length_velocity(qg, qd)
    (geo.force(Lg, Ldg, act).sum() + Lg.sum()).backward()
    assert torch.isfinite(qg.grad).all() and qg.grad.abs().sum() > 0


def test_motion_files_aliases_and_the_kit_split(clips: Path, tmp_path: Path) -> None:
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip import kit_split
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.body import (
        TRACKED_BODIES,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.kinematics import (
        SkeletonKinematics,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.motion import (
        MotionLib,
        load_clip,
        motion_files,
    )

    for rel in ("a/1.npz", "a/b/2.npz", "3.npz"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    root = str(tmp_path)
    assert motion_files(root) == [
        str(tmp_path / r) for r in ("3.npz", "a/1.npz", "a/b/2.npz")
    ]
    (tmp_path / "a" / "clips.txt").write_text(
        "# two clips\nb/2.npz\n\n1.npz  # the first\n"
    )
    assert motion_files(str(tmp_path / "a" / "clips.txt")) == [
        str(tmp_path / "a" / r) for r in ("b/2.npz", "1.npz")
    ]
    (tmp_path / "list.txt").write_text("a/1.npz\n3.npz\n")
    assert motion_files(str(tmp_path / "list.txt"), root=root) == [
        str(tmp_path / r) for r in ("a/1.npz", "3.npz")
    ]
    (tmp_path / "bad.txt").write_text("3.npz\nnope.npz\n")
    with pytest.raises(FileNotFoundError, match="1 of 2 listed clips missing"):
        motion_files(str(tmp_path / "bad.txt"))

    def entries(split: str) -> list[str]:
        lines = kit_split(split).read_text().splitlines()
        return [x.strip() for x in lines if x.strip() and not x.startswith("#")]

    train, test = entries("train"), entries("test")
    assert (len(train), len(set(train)), len(test), len(set(test))) == (
        972,
        972,
        108,
        108,
    )
    assert not set(train) & set(test)
    assert all(
        e.startswith("MyoFullBody/gmr/KIT/") and e.endswith(".npz")
        for e in train + test
    )
    # MuscleMimic's elbow_flex_* columns land on myo_sim's elbow_flexion_*
    m = _model()
    kin = SkeletonKinematics(TRACKED_BODIES, "cpu", m)
    raw = np.load(clips / "clip1.npz")
    _, _, _, q = load_clip(
        str(clips / "clip1.npz"), kin.info, kin.R_off.double().numpy()
    )
    elbow = kin.info.indep_names.index("elbow_flexion_r")
    np.testing.assert_allclose(
        q[:, elbow], raw["qpos"][:, m.jnt_qposadr[m.joint("elbow_flexion_r").id]]
    )
    lib = MotionLib(motion_files(str(clips / "clips.txt")), kin, 0.01, "cpu")
    assert lib.num_frames == 801 + 701 + 601 and lib.clip_len.tolist() == [
        801,
        701,
        601,
    ]


def test_bin_ema_averages_the_episodes_of_a_bin() -> None:
    require_mjlab()
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.commands import bin_ema

    out = bin_ema(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0, 0, 2, 2, 2]),
                  torch.tensor([3.0, 5.0, 0.0, 6.0, 100.0]), torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]), 0.8)  # fmt: skip
    torch.testing.assert_close(
        out, torch.tensor([0.8 * 1 + 0.2 * 4, 2.0, 0.8 * 3 + 0.2 * 3])
    )


# ---------------------------------------------------------------- the env (make_env, CPU)


def _cmd(env: Any) -> Any:
    return env.command_manager.get_term("motion")


def _analytic_obs(env: Any, group: str, ctx: Any) -> Any:
    om, cmd = env.observation_manager, _cmd(env)
    parts = []
    for name, tc in zip(
        om.active_terms[group], om._group_obs_term_cfgs[group], strict=True
    ):
        del name
        p = {k: v for k, v in tc.params.items() if k != "command_name"}
        parts.append(tc.func.analytic(cmd.heads, ctx, **p).reshape(ctx.f.shape[0], -1))
    return torch.cat(parts, -1)


def test_make_env_options() -> None:
    """Registered defaults are MuscleMimic's; the options reach the config."""
    require_mjlab()
    from mjlab.tasks.registry import load_env_cfg

    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip import ENV_ID
    from myosuite.envs.myo.backends.mjlab.tasks.registration import build_task_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401  (registers the task)

    reg = load_env_cfg(ENV_ID)
    cmd = reg.commands["motion"]
    assert (
        reg.decimation,
        cmd.stride,
        cmd.sampling_mode,
        cmd.bin_fix,
        cmd.fail_reset,
    ) == (5, 20, "uniform", False, 0.0)
    assert "muscle_length" in reg.observations["policy"].terms and reg.auto_reset
    cfg = build_task_cfg(ENV_ID, {"motion_file": "x.txt", "muscle_obs": False, "sampling_mode": "return",
                                  "bin_fix": True, "fail_reset": 0.3, "time_limit": False}, ctrl_dt=0.02)  # fmt: skip
    cmd = cfg.commands["motion"]
    assert (
        cfg.decimation,
        cmd.stride,
        cmd.sampling_mode,
        cmd.bin_fix,
        cmd.fail_reset,
    ) == (10, 10, "return", True, 0.3)
    assert (
        "muscle_length" not in cfg.observations["policy"].terms
        and "time_out" not in cfg.terminations
    )
    # the reward weights are the 100 Hz task's at every control rate
    assert {k: r.weight for k, r in cfg.rewards.items()} == {
        k: r.weight for k, r in reg.rewards.items()
    }
    with pytest.raises(ValueError, match="whole multiple"):
        build_task_cfg(ENV_ID, {}, ctrl_dt=0.003)


def test_reset_writes_reference_state(env: Any) -> None:
    cmd = _cmd(env)
    env.reset()
    S = cmd.read_state()
    p, R, _, w, q, qd = cmd.motion.ref_state(cmd.f)
    torch.testing.assert_close(S[0], p, atol=1e-6, rtol=0)
    torch.testing.assert_close(S[1], R, atol=1e-6, rtol=0)
    torch.testing.assert_close(S[3], w, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        S[4], q.clamp(cmd.kin.q_lo, cmd.kin.q_hi), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(S[5], qd, atol=1e-6, rtol=0)
    assert S[6].abs().max() == cmd.cfg.init_activation
    assert S[7].shape == (2, 4) and (S[7] >= 0).all()


def test_sim_matches_analytic_terms(env: Any) -> None:
    """Activation, reward, failure terminations and both observation groups: the managers equal the analytic terms
    they wrap, on the states the simulator reaches."""
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.heads import StepCtx

    cmd = _cmd(env)
    env.reset()
    gen = torch.Generator().manual_seed(0)
    touched = 0.0
    rm, tm = env.reward_manager, env.termination_manager
    for _ in range(12):
        S0, a_prev = cmd.read_state(), env.action_manager.action.clone()
        a = torch.rand(2, cmd.info.num_muscles, generator=gen) * 0.3
        obs, rew, term, trunc, _ = env.step(a)
        assert not (term | trunc).any()
        S1 = cmd.read_state()
        torch.testing.assert_close(
            S1[6], cmd.muscle.step(S0[6], a)[0], atol=1e-6, rtol=0
        )
        # the reward and failure terms score the post-step state against frame f + 1, the observation frame f
        ctx = StepCtx(S1, cmd.f, a, a_prev)
        expect = sum(
            tc.weight * env.step_dt * tc.func.analytic(cmd.heads, ctx, **{k: v for k, v in tc.params.items() if k != "command_name"})
            for tc in rm._term_cfgs
        )  # fmt: skip
        torch.testing.assert_close(expect, rew, atol=1e-6, rtol=1e-5)
        for name in tm.active_terms:
            tc = tm.get_term_cfg(name)
            if not tc.time_out:
                p = {
                    k: v
                    for k, v in tc.params.items()
                    if k not in ("command_name", "soft_width")
                }
                assert torch.equal(
                    tc.func.analytic(cmd.heads, StepCtx(S1, cmd.f), **p),
                    tm.get_term(name),
                ), name
        octx = cmd.obs_ctx()
        torch.testing.assert_close(
            _analytic_obs(env, "policy", octx), obs["policy"], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            _analytic_obs(env, "goal", octx), obs["goal"], atol=1e-6, rtol=0
        )
        touched += float(S1[7].sum())
    assert touched > 0  # the feet touch the ground


def test_observation_follows_resets(env: Any) -> None:
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.heads import StepCtx

    cmd = _cmd(env)
    env.reset()
    for _ in range(3):
        env.step(torch.full((2, cmd.info.num_muscles), 0.1))
    stale = _analytic_obs(env, "policy", cmd.obs_ctx())
    obs, _ = env.reset(env_ids=torch.tensor([0]))
    am = env.action_manager
    fresh = _analytic_obs(
        env,
        "policy",
        StepCtx(cmd.read_state(), cmd.f.clone(), am.action, am.prev_action),
    )
    torch.testing.assert_close(obs["policy"], fresh, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(obs["policy"][0], stale[0])


def test_deviation_terminations_and_reward_on_reference(env: Any) -> None:
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip import heads as H
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.heads import StepCtx

    cmd = _cmd(env)
    h = cmd.heads
    f = torch.arange(0, 600, 60)
    p, R, v, w, q, qd = cmd.motion.ref_state(f)
    act, touch = torch.zeros(len(f), cmd.info.num_muscles), torch.zeros(len(f), 4)
    terms = (H.bad_body_pos_dev, H.bad_root_pos_dev, H.bad_root_ori_dev)

    def fired(p: Any, R: Any, threshold: float) -> list:
        ctx = StepCtx((p, R, v, w, q, qd, act, touch), f)
        return [fn(h, ctx, threshold=threshold).float().mean().item() for fn in terms]

    assert fired(p, R, 1e-3) == [0.0, 0.0, 0.0]
    assert fired(p + torch.tensor([0.0, 1.2, 0.0]), R, 1.0) == [0.0, 1.0, 0.0]
    c, s = math.cos(1.2), math.sin(1.2)
    Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert fired(p, Rz @ R, 1.0)[1:] == [0.0, 1.0]
    soft = H.bad_root_pos_dev(
        h,
        StepCtx((p + torch.tensor([0.0, 1.0, 0.0]), R, v, w, q, qd, act, touch), f),
        1.0,
        0.05,
    )
    torch.testing.assert_close(soft, torch.full_like(soft, 0.5), atol=1e-4, rtol=0)
    # every MimicReward component is at its maximum on the reference, the critic's reference errors vanish
    ctx = StepCtx((p, R, v, w, q, qd, act, touch), f, act)
    for tc in env.reward_manager._term_cfgs:
        prm = {k: v for k, v in tc.params.items() if k != "command_name"}
        torch.testing.assert_close(
            tc.func.analytic(h, ctx, **prm), torch.ones(len(f)), atol=1e-4, rtol=0
        )
    extras = h.critic_extras(ctx)
    torch.testing.assert_close(extras[:, 4], torch.ones(len(f)), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        extras[:, 5:], torch.zeros_like(extras[:, 5:]), atol=1e-5, rtol=0
    )


def test_reset_probs_move_fail_reset_before_the_failure_bins(clips: Path) -> None:
    env = make_task_env(
        str(clips / "clips.txt"), sampling_mode="return", bin_fix=True, fail_reset=0.3
    )
    cmd = _cmd(env)
    nb = cmd.motion.num_bins
    cmd.bin_G.zero_()
    cmd.bin_fail.zero_()
    b = nb - 1
    cmd.bin_fail[b] = 0.5
    expect = torch.full((nb,), 0.7 / nb)
    expect[cmd.bin_prev[b]] += 0.3
    torch.testing.assert_close(cmd.reset_probs(), expect, atol=1e-6, rtol=0)
    cmd.bin_fail.zero_()
    torch.testing.assert_close(
        cmd.reset_probs(), torch.full((nb,), 1.0 / nb), atol=1e-6, rtol=0
    )
    # the masked draw (fixed-size, for graphed collection) follows the same distribution
    cmd.bin_G.copy_(torch.linspace(0, cmd.G_max, nb))
    p = cmd.reset_probs()
    counts = torch.zeros(nb)
    m = cmd.motion
    for _ in range(200):
        frames, bins = cmd.sample_frames_masked(torch.tensor(False))
        assert (
            (frames >= m.bin_start[bins]) & (frames < m.bin_start[bins] + m.bin_frames)
        ).all()
        counts += torch.bincount(bins, minlength=nb).float()
    torch.testing.assert_close(counts / counts.sum(), p, atol=0.04, rtol=0)
