# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Motor noise on the mjlab twin: registration wiring, statistics, CPU agreement.

A CPU registration with a ``MotorNoiseWrapper`` must configure the twin's ``MyoAction``
through ``cpu_reference.action_cfg``; the twin draws independent noise per env
and per muscle. Tolerances are five standard errors (see ``test_motor_noise``).
"""

from __future__ import annotations

from collections.abc import Iterator

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import WrapperSpec
from scipy import stats

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tier2

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

import myosuite  # noqa: E402, F401
from myosuite.core import registry  # noqa: E402
from myosuite.envs.wrappers import MotorNoiseWrapper  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import cpu_task_spec  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.pose.config.elbow.env_cfgs import (  # noqa: E402
    elbow_pose_env_cfg,
)
from myosuite.terms.base_action import MotorNoiseCfg  # noqa: E402
from myosuite import make_env  # noqa: E402

_BASE = "myoElbowPose1D6MRandom-v0"
_NOISY = "myoElbowPose1D6MMotorNoiseTest-v0"
_NOISE = MotorNoiseCfg(signal_dependent_std=0.1, constant_std=0.03)
_NUM_ENVS, _N = 4, 2500
_N_SE = 5.0


@pytest.fixture(scope="module")
def noisy_id() -> Iterator[str]:
    """A CPU registration of the elbow pose task with motor noise."""
    spec = gym.spec(_BASE)
    registry.register_env(
        env_id=_NOISY,
        entry_point=spec.entry_point,
        max_episode_steps=spec.max_episode_steps,
        kwargs=spec.kwargs,
        additional_wrappers=(
            WrapperSpec(
                name="MotorNoiseWrapper",
                entry_point="myosuite.envs.wrappers:MotorNoiseWrapper",
                kwargs={
                    "motor_noise": {"signal_dependent_std": 0.1, "constant_std": 0.03}
                },
            ),
        ),
    )
    yield _NOISY
    gym.registry.pop(_NOISY, None)


@pytest.fixture(scope="module")
def twin(noisy_id: str) -> Iterator[ManagerBasedRlEnv]:
    cfg = elbow_pose_env_cfg(noisy_id)
    cfg.scene.num_envs = _NUM_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    yield env
    env.close()


def _samples(env: ManagerBasedRlEnv, action: float, n: int = _N) -> np.ndarray:
    """``(n, num_envs, muscles)`` excitations the term writes for a constant action."""
    term = env.action_manager.get_term("muscles")
    actions = torch.full((env.num_envs, term.action_dim), action)
    out = []
    for _ in range(n):
        term.process_actions(actions)
        out.append(term.processed_action.clone())
    return torch.stack(out).numpy()


def test_cpu_registration_configures_twin(noisy_id: str) -> None:
    assert cpu_task_spec(noisy_id).motor_noise == _NOISE
    assert elbow_pose_env_cfg(noisy_id).actions["muscles"].motor_noise == _NOISE
    assert not elbow_pose_env_cfg(_BASE).actions["muscles"].motor_noise.enabled


def test_cpu_task_spec_rejects_wrappers_the_cpu_env_does_not_run() -> None:
    """A twin cannot pick up muscle wrappers from a CPU env class that would ignore them."""
    base, env_id = gym.spec("myoMimicFullbody-v0"), "myoMimicFullbodyNoiseTest-v0"
    registry.register_env(
        env_id=env_id,
        entry_point=base.entry_point,
        max_episode_steps=base.max_episode_steps,
        kwargs=base.kwargs,
        additional_wrappers=(
            WrapperSpec(
                name="MotorNoiseWrapper",
                entry_point="myosuite.envs.wrappers:MotorNoiseWrapper",
                kwargs={"motor_noise": {"constant_std": 0.1}},
            ),
        ),
    )
    try:
        with pytest.raises(
            ValueError, match="MuscleMimicFullbodyEnv.*does not run them"
        ):
            cpu_task_spec(env_id)
    finally:
        gym.registry.pop(env_id, None)


def test_twin_noise_statistics_and_independence(twin: ManagerBasedRlEnv) -> None:
    torch.manual_seed(0)
    resid = _samples(twin, 0.5) - 0.5  # sigmoid(0.5) = 0.5
    sigma = float(np.hypot(_NOISE.signal_dependent_std * 0.5, _NOISE.constant_std))
    assert abs(resid.std(ddof=1) - sigma) < _N_SE * sigma / np.sqrt(2 * resid.size)
    corr_tol = _N_SE / np.sqrt(_N)
    for e in range(_NUM_ENVS):  # across muscles, within an env
        corr = np.corrcoef(resid[:, e].T)
        assert np.abs(corr[~np.eye(len(corr), dtype=bool)]).max() < corr_tol
    for m in range(resid.shape[2]):  # across envs, same muscle
        corr = np.corrcoef(resid[:, :, m].T)
        assert np.abs(corr[~np.eye(len(corr), dtype=bool)]).max() < corr_tol


def test_twin_noise_is_seeded_by_torch(twin: ManagerBasedRlEnv) -> None:
    torch.manual_seed(7)
    first = _samples(twin, 0.3, n=3)
    torch.manual_seed(7)
    np.testing.assert_array_equal(_samples(twin, 0.3, n=3), first)


def test_twin_noise_off_is_exact_and_draws_nothing(twin: ManagerBasedRlEnv) -> None:
    term = twin.action_manager.get_term("muscles")
    term.cfg.motor_noise = MotorNoiseCfg()
    try:
        state = torch.get_rng_state()
        out = _samples(twin, 0.3, n=2)
        assert torch.equal(torch.get_rng_state(), state)
    finally:
        term.cfg.motor_noise = _NOISE
    expected = torch.sigmoid(torch.tensor(5.0 * (0.3 - 0.5))).item()
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    "action", [0.5, 0.0]
)  # excitation 0.5 and 0.076 (clipped at 0)
def test_twin_matches_cpu_distribution(twin: ManagerBasedRlEnv, action: float) -> None:
    """CPU and mjlab excitations agree in distribution (van Beers levels, KS test)."""
    vb = MotorNoiseCfg.van_beers_2004()
    term = twin.action_manager.get_term("muscles")
    term.cfg.motor_noise = vb
    try:
        torch.manual_seed(1)
        gpu = _samples(twin, action, n=1000).ravel()
    finally:
        term.cfg.motor_noise = _NOISE
    cpu_env = MotorNoiseWrapper(make_env(_BASE), vb)
    cpu_env.reset(seed=1)
    base = cpu_env.unwrapped
    a = np.full(cpu_env.action_space.shape, action, np.float32)
    cpu = []
    for _ in range(4000):
        base._apply_action(a)
        cpu.append(base.data.ctrl[base._muscle_act_ind].copy())
    cpu_env.close()
    assert min(np.std(cpu), np.std(gpu)) > 0.05  # both noisy
    assert stats.ks_2samp(np.ravel(cpu), gpu).pvalue > 1e-3


# ── portable custom stages ───────────────────────────────────────────────────

_STAGED = "myoFatiElbowPose1D6MStageTest-v0"


@pytest.fixture(scope="module")
def staged_id() -> Iterator[str]:
    """A fatigue registration with a low-pass stage (no explicit order: after fatigue)."""
    import functools

    from myosuite.envs.muscle_stages import LowPassStage

    spec = gym.spec("myoFatiElbowPose1D6MRandom-v0")
    registry.register_env(
        env_id=_STAGED,
        entry_point=spec.entry_point,
        max_episode_steps=spec.max_episode_steps,
        kwargs=spec.kwargs,
        additional_wrappers=(
            *spec.additional_wrappers[:-1],
            WrapperSpec(
                name="ExcitationStageWrapper",
                entry_point="myosuite.envs.wrappers:ExcitationStageWrapper",
                kwargs={"make_stage": functools.partial(LowPassStage, 0.4)},
            ),
            spec.additional_wrappers[-1],
        ),
    )
    yield _STAGED
    gym.registry.pop(_STAGED, None)


def test_registration_configures_the_twin_with_the_stage(staged_id: str) -> None:
    cfg = elbow_pose_env_cfg(staged_id)
    factories = cfg.actions["muscles"].excitation_stages
    assert len(factories) == 1 and factories[0]().name == "lowpass"
    assert not elbow_pose_env_cfg(_BASE).actions["muscles"].excitation_stages


def test_twin_matches_cpu_with_a_stateful_stage(staged_id: str) -> None:
    """Low-pass after fatigue (default order): CPU env and twin give the same ctrl."""
    cfg = elbow_pose_env_cfg(staged_id)
    cfg.scene.num_envs = 3
    twin = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    twin.reset()
    term = twin.action_manager.get_term("muscles")
    cpu = make_env(staged_id)
    cpu.reset(seed=0)
    base = cpu.unwrapped
    rng = np.random.default_rng(0)
    for step in range(6):
        a = rng.uniform(-1.0, 1.0, cpu.action_space.shape).astype(np.float32)
        term.process_actions(torch.as_tensor(np.tile(a, (3, 1))))
        base._apply_action(a)
        np.testing.assert_allclose(
            term.processed_action.numpy(),
            np.tile(base.data.ctrl, (3, 1)),
            atol=1e-5,
            err_msg=f"step {step}",
        )
    # a per-env reset clears the filter of that env only
    term.reset(torch.tensor([1]))
    stage = next(st for st in term._stages if st.name == "lowpass").stage
    assert bool(stage._fresh[1]) and not bool(stage._fresh[0])
    assert term.stage_names == ("noise", "fatigue", "lowpass")
    twin.close()


def test_same_order_warns_on_the_twin() -> None:
    import functools

    from myosuite.envs.muscle_stages import LowPassStage, StageOrderWarning

    cfg = elbow_pose_env_cfg(_BASE)
    cfg.scene.num_envs = 2
    cfg.actions["muscles"].excitation_stages = (
        functools.partial(LowPassStage, 0.5, "a", 25),
        functools.partial(LowPassStage, 0.5, "b", 25),
    )
    with pytest.warns(StageOrderWarning, match="STAGE ORDER CLASH.*'a', 'b'"):
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.close()


@pytest.mark.parametrize("order", [10, 5, 100, 150])
def test_out_of_range_order_raises_on_the_twin(order: float) -> None:
    import functools

    from myosuite.envs.muscle_stages import LowPassStage

    cfg = elbow_pose_env_cfg(_BASE)
    cfg.scene.num_envs = 2
    cfg.actions["muscles"].excitation_stages = (
        functools.partial(LowPassStage, 0.5, "x", order),
    )
    with pytest.raises(ValueError, match="between 10 and 100"):
        ManagerBasedRlEnv(cfg=cfg, device="cpu").close()


# ── combined conditions ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "conditions", [("sarcopenia", "fatigue"), ("fatigue", "reafferentation")]
)
def test_twin_takes_every_registered_condition(conditions: tuple[str, ...]) -> None:
    """The condition wrappers compose on the CPU env, so each one configures the twin."""
    import dataclasses

    from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import action_cfg
    from myosuite.envs.wrappers import condition_wrapper_specs

    wrappers = tuple(s for c in conditions for s in condition_wrapper_specs(c))
    task = dataclasses.replace(cpu_task_spec("myoHandPoseRandom-v0"), wrappers=wrappers)
    cfg = action_cfg(task, "robot")
    assert task.muscle_conditions == conditions
    assert cfg.muscle_fatigue == ("fatigue" in conditions)
    assert (cfg.reroute == ("EIP_r", "EPL_r")) == ("reafferentation" in conditions)


def test_twin_matches_cpu_with_sarcopenia_and_fatigue() -> None:
    """A sarcopenia + fatigue registration: the twin fatigues like the CPU env."""
    from myosuite.envs.wrappers import condition_wrapper_specs

    env_id, spec = "myoSarcFatiElbowPose1D6MTest-v0", gym.spec(_BASE)
    registry.register_env(
        env_id=env_id,
        entry_point=spec.entry_point,
        max_episode_steps=spec.max_episode_steps,
        kwargs=spec.kwargs,
        additional_wrappers=(
            *condition_wrapper_specs("sarcopenia"),
            *condition_wrapper_specs("fatigue"),
        ),
    )
    try:
        cfg = elbow_pose_env_cfg(env_id)
        cfg.scene.num_envs = 2
        twin = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        twin.reset()
        term = twin.action_manager.get_term("muscles")
        assert term.stage_names == ("noise", "fatigue")
        cpu = make_env(env_id)
        cpu.reset(seed=0)
        base = cpu.unwrapped
        assert base.ctrl_stages == ("fatigue",) and base._sarcopenia_applied
        a = np.ones(cpu.action_space.shape, np.float32)  # sustained effort fatigues
        for step in range(20):
            term.process_actions(torch.as_tensor(np.tile(a, (2, 1))))
            base._apply_action(a)
            np.testing.assert_allclose(
                term.processed_action.numpy(),
                np.tile(base.data.ctrl, (2, 1)),
                atol=1e-5,
                err_msg=f"step {step}",
            )
        assert base.data.ctrl.max() < float(torch.sigmoid(torch.tensor(2.5)))
        twin.close()
    finally:
        gym.registry.pop(env_id, None)


def test_twin_custom_stage_order_defaults_to_after_the_builtins() -> None:
    import functools

    from myosuite.envs.muscle_stages import LowPassStage

    cfg = elbow_pose_env_cfg(_BASE)
    cfg.scene.num_envs = 2
    cfg.actions["muscles"].excitation_stages = (
        functools.partial(LowPassStage, 0.5, "a"),
        functools.partial(LowPassStage, 0.5, "b"),
        functools.partial(LowPassStage, 0.5, "early", 15),
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        names = env.action_manager.get_term("muscles").stage_names
        assert names == ("early", "noise", "a", "b")
    finally:
        env.close()
