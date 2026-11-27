# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Co-contraction cost (wasted muscle torque): analytic checks, CPU envs and the mjlab twin."""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.terms.cocontraction import (
    MomentArmIndex,
    WastedTorqueCfg,
    muscle_moment_arms,
    wasted_torque_reward,
)
from myosuite.tests.support.optional_deps import (
    require_mjlab,
    require_mujoco_warp,
    require_musclemimic_models,
)

# Hinge j1 with a flexor (+R) and an extensor (-R); a biarticular muscle crosses j1 and j2.
# At qpos 0 every muscle is shorter than its optimal length, so there is no passive force.
_R = 0.03
_ANTAGONIST_XML = f"""
<mujoco>
  <worldbody>
    <body name="upper">
      <joint name="j1" type="hinge" axis="0 1 0" range="-1 1" limited="true"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3" mass="2"/>
      <body name="lower" pos="0 0 -0.3">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1 1" limited="true"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3" mass="1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="flexor"><joint joint="j1" coef="{_R}"/></fixed>
    <fixed name="extensor"><joint joint="j1" coef="-{_R}"/></fixed>
    <fixed name="biarticular"><joint joint="j1" coef="0.01"/><joint joint="j2" coef="0.02"/></fixed>
  </tendon>
  <actuator>
    <muscle name="flexor" tendon="flexor" force="300"/>
    <muscle name="extensor" tendon="extensor" force="200"/>
    <muscle name="biarticular" tendon="biarticular" force="100"/>
  </actuator>
</mujoco>
"""


class _Accessor:
    """Minimal accessor over sparse moment arrays (numpy or torch)."""

    def __init__(
        self, xp: Any, force: Any, moment: Any, rowadr: Any, qpos: Any = None
    ) -> None:
        self._xp, self._force, self._moment, self._rowadr = xp, force, moment, rowadr
        self._qpos = qpos

    def array_module(self) -> Any:
        return self._xp

    def muscle_force(self) -> Any:
        return self._force

    def actuator_moment_sparse(self) -> tuple[Any, Any]:
        return self._moment, self._rowadr

    def joint_pos(self) -> Any:
        return self._qpos


def _antagonist_state(act: tuple[float, float, float]) -> tuple[Any, Any]:
    model = mujoco.MjModel.from_xml_string(_ANTAGONIST_XML)
    data = mujoco.MjData(model)
    data.act[:] = act
    mujoco.mj_forward(model, data)
    return model, data


def _cpu_wasted(model: Any, data: Any, joints: tuple[str, ...]) -> float:
    index = MomentArmIndex.from_model(model, joints)
    acc = _Accessor(np, data.actuator_force, *_sparse(data), data.qpos)
    return float(wasted_torque_reward(acc, {}, moment_index=index)["wasted_torque"])


def _sparse(data: Any) -> tuple[np.ndarray, np.ndarray]:
    return data.actuator_moment, data.moment_rowadr


@pytest.mark.tier1
def test_co_contraction_wastes_the_smaller_torque() -> None:
    """Flexor and extensor both active: the cancelled torque is twice the smaller one."""
    model, data = _antagonist_state((1.0, 1.0, 0.0))
    f = data.actuator_force
    assert f[0] < 0 and f[1] < 0 and f[2] == 0
    tau = np.array([_R * f[0], -_R * f[1]])
    expected = (np.abs(tau).sum() - abs(tau.sum())) / mujoco.mj_getTotalmass(model)
    wasted = _cpu_wasted(model, data, ("j1",))
    assert wasted > 0
    assert wasted == pytest.approx(expected, rel=1e-12)
    assert wasted == pytest.approx(2 * np.abs(tau).min() / 3.0, rel=1e-12)


@pytest.mark.tier1
def test_a_single_agonist_wastes_nothing() -> None:
    """Only the flexor (and the biarticular muscle, pulling the same way) active: no waste."""
    model, data = _antagonist_state((1.0, 0.0, 1.0))
    assert data.actuator_force[1] == 0
    assert _cpu_wasted(model, data, ("j1", "j2")) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.tier1
@pytest.mark.parametrize("weight", [-0.005, float("nan"), float("inf"), -float("inf")])
def test_penalty_magnitude_must_be_nonnegative_and_finite(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        WastedTorqueCfg(weight=weight)


@pytest.mark.tier1
def test_dofs_outside_the_set_are_ignored() -> None:
    """Muscles that do not cross the scored DOFs are left out of the index."""
    model, data = _antagonist_state((1.0, 1.0, 1.0))
    index = MomentArmIndex.from_model(model, ("j2",))
    assert index.actuator_ids.tolist() == [2]
    assert _cpu_wasted(model, data, ("j2",)) == 0.0
    with pytest.raises(ValueError, match="not found"):
        MomentArmIndex.from_model(model, ("knee",))


@pytest.mark.tier1
def test_batched_torch_rows_in_any_order() -> None:
    """mujoco-warp allocates moment rows per world in any order; the gather follows rowadr."""
    torch = pytest.importorskip("torch")
    model, data = _antagonist_state((0.7, 0.4, 0.9))
    index = MomentArmIndex.from_model(model, ("j1", "j2"))
    expected = _cpu_wasted(model, data, ("j1", "j2"))
    rownnz, rowadr = data.moment_rownnz, data.moment_rowadr
    worlds = []
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        moment = np.zeros_like(data.actuator_moment)
        new_adr = np.zeros_like(rowadr)
        pos = 0
        for a in order:
            moment[pos : pos + rownnz[a]] = data.actuator_moment[
                rowadr[a] : rowadr[a] + rownnz[a]
            ]
            new_adr[a] = pos
            pos += rownnz[a]
        worlds.append((moment, new_adr))
    acc = _Accessor(
        torch,
        torch.as_tensor(np.tile(data.actuator_force, (3, 1))),
        torch.as_tensor(np.stack([w[0] for w in worlds])),
        torch.as_tensor(np.stack([w[1] for w in worlds]).astype(np.int32)),
    )
    out = wasted_torque_reward(acc, {}, moment_index=index.to_backend(torch.as_tensor))[
        "wasted_torque"
    ]
    assert out.shape == (3,)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-12)


def _dense_reference(env: Any, cost: WastedTorqueCfg) -> float:
    """Wasted torque from the dense moment matrix (``mju_sparse2dense``)."""
    m, d = env.model, env.data
    dense = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(
        dense, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind
    )
    tangent = _finite_difference_tangent(m, d.qpos, cost.joints)
    tau = (dense @ tangent) * d.actuator_force[:, None]
    return float(
        (np.abs(tau).sum(0) - np.abs(tau.sum(0))).sum() / mujoco.mj_getTotalmass(m)
    )


def _finite_difference_tangent(
    model: mujoco.MjModel, qpos: np.ndarray, joints: tuple[str, ...]
) -> np.ndarray:
    couplings = JointCouplings.from_model(model)
    tangent = np.empty((model.nv, len(joints)))
    for col, name in enumerate(joints):
        adr = model.jnt_qposadr[model.joint(name).id]
        minus, plus = qpos.copy(), qpos.copy()
        minus[adr] -= 1e-6
        plus[adr] += 1e-6
        project_joint_couplings(couplings, minus)
        project_joint_couplings(couplings, plus)
        velocity = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, velocity, 2e-6, minus, plus)
        tangent[:, col] = velocity
    return tangent


@pytest.mark.tier1
def test_coupled_moments_match_length_finite_differences() -> None:
    """A muscle crossing only a chained dependent still acts on its master."""
    from myosuite.tests.test_joint_coupling import _XML

    names = ("master", "dep", "chained", "locked", "loose")
    tendons = "".join(
        f'<fixed name="t{i}"><joint joint="{name}" coef="0.03"/></fixed>'
        for i, name in enumerate(names)
    )
    actuators = "".join(
        f'<muscle name="m{i}" tendon="t{i}" lengthrange="0.1 1" force="100"/>'
        for i in range(len(names))
    )
    xml = _XML.replace(
        "</mujoco>",
        f"<tendon>{tendons}</tendon><actuator>{actuators}</actuator></mujoco>",
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] += 0.2
    project_joint_couplings(JointCouplings.from_model(model), data.qpos)
    mujoco.mj_forward(model, data)
    index = MomentArmIndex.from_model(model, ("master", "loose"))
    assert index.actuator_ids.tolist() == [0, 1, 2, 4]
    arms = muscle_moment_arms(np, *_sparse(data), index, data.qpos)
    tangent = _finite_difference_tangent(model, data.qpos, ("master", "loose"))
    np.testing.assert_allclose(arms, 0.03 * tangent[index.actuator_ids], atol=1e-10)
    assert abs(arms[2, 0]) > 0.001
    with pytest.raises(ValueError, match="independent"):
        MomentArmIndex.from_model(model, ("dep",))
    torch = pytest.importorskip("torch")
    index_t = index.to_backend(torch.as_tensor)
    actual = muscle_moment_arms(
        torch,
        torch.tensor(np.tile(data.actuator_moment, (2, 1))),
        torch.tensor(np.tile(data.moment_rowadr, (2, 1))),
        index_t,
        torch.tensor(np.tile(data.qpos, (2, 1))),
    )
    np.testing.assert_allclose(
        actual.numpy(), np.broadcast_to(arms, actual.shape), atol=1e-12
    )


@pytest.mark.tier1
def test_fullbody_knee_moments_match_tendon_length_differences() -> None:
    """The effective knee arms include tibia/patella motion, not just knee rotation."""
    require_musclemimic_models()
    env = make_env("myoMimicFullbody-v0").unwrapped
    model, data = env.model, env.data
    names = ("knee_angle_r", "knee_angle_l")
    for name in names:
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = 0.5
    couplings = JointCouplings.from_model(model)
    project_joint_couplings(couplings, data.qpos)
    mujoco.mj_forward(model, data)
    index = MomentArmIndex.from_model(model, names)
    actual = muscle_moment_arms(np, *_sparse(data), index, data.qpos)
    expected = np.empty_like(actual)
    for col, name in enumerate(names):
        lengths = []
        for delta in (-1e-6, 1e-6):
            perturbed = mujoco.MjData(model)
            perturbed.qpos[:] = data.qpos
            perturbed.qpos[model.jnt_qposadr[model.joint(name).id]] += delta
            project_joint_couplings(couplings, perturbed.qpos)
            mujoco.mj_forward(model, perturbed)
            lengths.append(perturbed.actuator_length[index.actuator_ids].copy())
        expected[:, col] = (lengths[1] - lengths[0]) / 2e-6
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-4)
    partial = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(
        partial,
        data.actuator_moment,
        data.moment_rownnz,
        data.moment_rowadr,
        data.moment_colind,
    )
    assert (
        np.max(np.abs(actual - partial[index.actuator_ids][:, index.dof_ids])) > 0.001
    )
    env.close()


@pytest.mark.tier1
def test_fullbody_weight_zero_keeps_the_reward_and_logs_the_metric() -> None:
    """``myoMimicFullbody-v0``: weight 0 changes nothing; a weight subtracts weight x metric."""
    require_musclemimic_models()
    base = make_env("myoMimicFullbody-v0")
    logged = make_env(
        EnvConfig(
            "myoMimicFullbody-v0", task_kwargs={"wasted_torque": WastedTorqueCfg()}
        )
    )
    weighted = make_env(
        EnvConfig(
            "myoMimicFullbody-v0",
            task_kwargs={"wasted_torque": WastedTorqueCfg(weight=0.005)},
        )
    )
    for env in (base, logged, weighted):
        env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(4):
        action = rng.uniform(-2.5, 2.5, base.action_space.shape).astype(np.float32)
        obs0, rew0, *_, info0 = base.step(action)
        obs1, rew1, *_, info1 = logged.step(action)
        obs2, rew2, *_, info2 = weighted.step(action)
        np.testing.assert_array_equal(obs0, obs1)
        np.testing.assert_array_equal(obs0, obs2)
        assert rew1 == rew0
        assert "wasted_torque" not in info0
        wasted = info1["wasted_torque"]
        assert wasted > 0
        assert wasted == pytest.approx(
            _dense_reference(logged.unwrapped, WastedTorqueCfg()), rel=1e-10
        )
        assert info2["wasted_torque"] == wasted
        assert rew2 == pytest.approx(rew0 - 0.005 * wasted, abs=1e-12)
    for env in (base, logged, weighted):
        env.close()


@pytest.mark.tier1
def test_directional_weight_zero_keeps_the_reward() -> None:
    """``myoFullBodyDirectional-v0`` takes the same opt-in cost."""
    require_musclemimic_models()
    pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import HfHubHTTPError

    try:
        base = make_env("myoFullBodyDirectional-v0")
        weighted = make_env(
            EnvConfig(
                "myoFullBodyDirectional-v0",
                task_kwargs={"wasted_torque": WastedTorqueCfg(weight=0.005)},
            )
        )
    except HfHubHTTPError as exc:
        pytest.skip(f"gated HF dataset unavailable ({exc})")
    base.reset(seed=1)
    weighted.reset(seed=1)
    action = np.full(base.action_space.shape, 0.3, dtype=np.float32)
    for _ in range(3):
        _, rew0, *_ = base.step(action)
        _, rew1, *_, info = weighted.step(action)
        assert info["wasted_torque"] > 0
        assert rew1 == pytest.approx(rew0 - 0.005 * info["wasted_torque"], abs=1e-12)
    base.close()
    weighted.close()


_ENTITY = "mimic_fullbody_robot"
_COST = WastedTorqueCfg(weight=0.005)


@pytest.mark.tier2
@pytest.mark.parametrize("ctrl_dt", [0.01, 0.02])
def test_directional_sar_penalty_is_a_reward_rate(ctrl_dt: float) -> None:
    """Equal physical durations earn equal penalties at either control rate."""
    require_mjlab()
    torch = pytest.importorskip("torch")
    from mjlab.managers.reward_manager import RewardManager

    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    cfg = mimic._make_directional_sar_env_cfg(
        _task_id="sar-test",
        entity_name="robot",
        spec_fn=mujoco.MjSpec,
        muscle_actuators=("m",),
        tendon_targets=("t",),
        sar_transform=None,
        sim_dt=0.002,
        ctrl_dt=ctrl_dt,
        max_episode_steps=100,
        wasted_torque=_COST,
    )
    term = cfg.rewards["wasted_torque"]
    assert cfg.scale_rewards_by_dt
    assert "wasted_torque" in cfg.metrics
    env = SimpleNamespace(num_envs=2, device="cpu")
    term.func = lambda _env, **_params: torch.full((2,), 12.0)
    manager = RewardManager({"wasted_torque": term}, env)
    torch.testing.assert_close(
        manager.compute(ctrl_dt), torch.full((2,), -0.005 * 12 * ctrl_dt)
    )
    total = torch.zeros(2)
    for _ in range(round(1.0 / ctrl_dt)):
        total += manager.compute(ctrl_dt)
    torch.testing.assert_close(
        total, torch.full((2,), -0.005 * 12), atol=1e-7, rtol=1e-5
    )
    mimic_term, _ = mimic._wasted_torque_terms("robot", ctrl_dt, _COST)
    mimic_term.func = term.func
    mimic_manager = RewardManager({"wasted_torque": mimic_term}, env)
    torch.testing.assert_close(
        mimic_manager.compute(ctrl_dt), torch.full((2,), -0.005 * 12)
    )


@pytest.mark.tier2
@pytest.mark.parametrize("weight", [0.005, float("nan"), float("inf"), -float("inf")])
def test_mjlab_rejects_a_reward_that_pays_for_cocontraction(weight: float) -> None:
    """A CLI-overridden RewardTermCfg must fail while the manager is configured."""
    require_mjlab()
    from mjlab.managers.reward_manager import RewardManager, RewardTermCfg

    from myosuite.envs.myo.backends.mjlab.tasks.mdp.rewards import wasted_torque

    cfg = RewardTermCfg(
        func=wasted_torque,
        weight=weight,
        params={"entity_name": "robot", "cfg": _COST},
    )
    with pytest.raises(ValueError, match="finite and non-positive"):
        RewardManager({"wasted_torque": cfg}, SimpleNamespace(num_envs=2, device="cpu"))


@functools.cache
def _mjlab_play_cfgs(cost: WastedTorqueCfg | None) -> dict[str, Any]:
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    cfgs: dict[str, Any] = {}
    mimic.register_mimic_mjlab_tasks(
        register_mjlab_task=lambda **task: cfgs.__setitem__(
            task["task_id"], task["play_env_cfg"]
        ),
        rl_cfg_fn=mimic.default_mimic_clip_on_policy_runner_cfg,
        wasted_torque=cost,
    )
    return cfgs


@pytest.mark.tier2
def test_mjlab_matches_cpu_on_the_same_state() -> None:
    """Same state: same metric; same action: same reward with the cost enabled."""
    require_mjlab()
    require_mujoco_warp()
    require_musclemimic_models()
    torch = pytest.importorskip("torch")
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic
    from myosuite.envs.myo.backends.mjlab.tasks.mdp import (
        wasted_torque,
        write_cpu_state,
    )
    from myosuite.tests.support.host_sync import HostSyncCounter

    default_cfg = _mjlab_play_cfgs(None)["myoMimicFullbody-v0"]
    assert default_cfg.rewards["wasted_torque"].weight == 0.0
    assert "wasted_torque" in default_cfg.metrics

    cpu = make_env(
        EnvConfig("myoMimicFullbody-v0", task_kwargs={"wasted_torque": _COST})
    ).unwrapped
    cpu.reset(seed=0)
    mj = ManagerBasedRlEnv(
        cfg=_mjlab_play_cfgs(_COST)["myoMimicFullbody-v0"], device="cpu"
    )
    mj.reset()
    assert mj.reward_manager.get_term_cfg("wasted_torque").weight == pytest.approx(
        -_COST.weight / float(mj.step_dt)
    )
    term = wasted_torque(
        SimpleNamespace(params={"entity_name": _ENTITY, "cfg": _COST}), mj
    )

    def _t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x)[None], dtype=torch.float32)

    cache = mimic._resolve_mimic_mjlab_ids(mj, _ENTITY, "fullbody")
    env0 = torch.zeros(1, dtype=torch.long)
    rng = np.random.default_rng(0)
    term(mj, _ENTITY)  # mjlab wraps (and caches) each Warp array on first access
    for _ in range(4):
        sites = cpu.data.site_xpos[cpu._site_ids]
        cpu._target_site_pos = (sites + rng.normal(0.0, 0.01, sites.shape)).astype(
            np.float32
        )
        write_cpu_state(mj, _ENTITY, env0, _t(cpu.data.qpos), _t(cpu.data.qvel))
        mj.sim.data.act[:] = _t(cpu.data.act)
        cache["target_torch"] = _t(cpu._target_site_pos)
        mj.sim.forward()
        cpu_wasted = cpu._add_wasted_torque({"dense": 0.0})["wasted_torque"]
        with HostSyncCounter(package_only=True) as syncs:
            value = term(mj, _ENTITY)
        assert syncs.total == 0, syncs.report()
        assert float(value[0]) == pytest.approx(cpu_wasted, rel=1e-4)

        action = rng.uniform(-2.5, 2.5, cpu.action_space.shape).astype(np.float32)
        _, cpu_rew, _, _, info = cpu.step(action)
        _, mj_rew, _, _, _ = mj.step(torch.as_tensor(action[None]))
        assert info["wasted_torque"] > 0
        # float32 Warp vs float64 MuJoCo one step after a synced state (test_mimic_cpu_mjlab_twin).
        np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=5e-3)
        np.testing.assert_allclose(
            float(term(mj, _ENTITY)[0]), info["wasted_torque"], rtol=2e-2
        )
    mj.close()
    cpu.close()
