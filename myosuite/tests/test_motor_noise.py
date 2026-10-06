# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Motor noise on muscle excitations: term statistics and the ``MotorNoiseWrapper``.

Statistical checks use fixed seeds and a tolerance of five standard errors
(SE of a sample SD: ``sigma / sqrt(2 (n - 1))``; of a correlation: ``1 / sqrt(n)``).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import load_env_creator

import myosuite  # noqa: F401
from myosuite.envs.wrappers import (
    FatigueWrapper,
    MotorNoiseWrapper,
    ReafferentationWrapper,
)
from myosuite.terms.base_action import (
    MotorNoiseCfg,
    motor_noise,
    sample_motor_noise,
    sigmoid_muscle_activation,
)
from myosuite import make_env

pytestmark = pytest.mark.tier1

_N_SE = 5.0


def _expected_sd(u: float, sd: float, c: float) -> float:
    return float(np.hypot(sd * u, c))


def _sd_tolerance(sigma: float, n: int) -> float:
    return _N_SE * sigma / np.sqrt(2.0 * (n - 1))


# ── Term ─────────────────────────────────────────────────────────────────────


def test_term_formula_and_clip() -> None:
    u = np.array([0.0, 0.2, 0.5, 0.9, 0.95])
    n1 = np.array([1.0, -1.0, 2.0, 0.0, 3.0])
    n2 = np.array([-1.0, 0.5, 0.0, 1.0, 0.0])
    out = motor_noise(u, n1, n2, 0.1, 0.2, np)
    expected = np.clip(u + 0.1 * u * n1 + 0.2 * n2, 0.0, 1.0)
    np.testing.assert_array_equal(out, expected)
    assert out[0] == 0.0 and out[3] == 1.0  # clipped to [0, 1]


@pytest.mark.parametrize("u", [0.2, 0.5, 0.7])
def test_term_statistics(u: float) -> None:
    """SD of the applied excitation is sqrt((sd u)^2 + c^2) in the unclipped interior."""
    sd, c, n = 0.1, 0.03, 200_000
    rng = np.random.default_rng(0)
    out = sample_motor_noise(
        np.full(n, u), MotorNoiseCfg(sd, c), rng.standard_normal, np
    )
    sigma = _expected_sd(u, sd, c)
    assert abs(out.std(ddof=1) - sigma) < _sd_tolerance(sigma, n)
    assert abs(out.mean() - u) < _N_SE * sigma / np.sqrt(n)


def test_term_numpy_torch_parity() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, (64, 39))
    n1, n2 = rng.standard_normal((2, 64, 39))
    ref = motor_noise(u, n1, n2, 0.103, 0.185, np)
    out = motor_noise(*(torch.as_tensor(x) for x in (u, n1, n2)), 0.103, 0.185, torch)
    assert np.abs(out.numpy() - ref).max() <= 1e-7


def test_disabled_config_draws_nothing() -> None:
    def _fail(shape: tuple[int, ...]) -> np.ndarray:
        raise AssertionError("drew normals with noise disabled")

    u = np.full(6, 0.4)
    assert sample_motor_noise(u, MotorNoiseCfg(), _fail, np) is u


def test_cfg_coercion_and_validation() -> None:
    assert not MotorNoiseCfg().enabled
    assert MotorNoiseCfg.from_value(None) == MotorNoiseCfg()
    vb = MotorNoiseCfg.van_beers_2004()
    assert (vb.signal_dependent_std, vb.constant_std) == (0.103, 0.185)
    assert MotorNoiseCfg.from_value(vb) is vb
    assert MotorNoiseCfg.from_value({"constant_std": 0.1}) == MotorNoiseCfg(0.0, 0.1)
    with pytest.raises(ValueError):
        MotorNoiseCfg(signal_dependent_std=-0.1)
    with pytest.raises(ValueError):
        MotorNoiseCfg(constant_std=float("nan"))
    with pytest.raises(TypeError):
        MotorNoiseCfg.from_value(0.1)  # type: ignore[arg-type]


# ── CPU envs ─────────────────────────────────────────────────────────────────

_ELBOW = "myoElbowPose1D6MRandom-v0"


def _noisy(env_id: str, noise: object, **kwargs: object) -> gym.Env:
    return MotorNoiseWrapper(make_env(env_id, **kwargs), noise)


def _excitations(env: gym.Env, action: np.ndarray, n: int) -> np.ndarray:
    """``n`` applied muscle excitations (rows) for a constant action."""
    base = env.unwrapped
    rows = []
    for _ in range(n):
        base._apply_action(action)
        rows.append(base.data.ctrl[base._muscle_act_ind].copy())
    return np.array(rows)


def test_cpu_noise_is_independent_per_muscle() -> None:
    """Each muscle gets its own draw (a shared draw would give correlation 1)."""
    c, n = 0.05, 3000
    env = _noisy(_ELBOW, {"constant_std": c})
    env.reset(seed=0)
    action = np.full(env.action_space.shape, 0.5, np.float32)  # excitation 0.5
    resid = _excitations(env, action, n) - 0.5
    assert resid.shape[1] == 6
    np.testing.assert_array_less(
        np.abs(resid.std(axis=0, ddof=1) - c), _sd_tolerance(c, n)
    )
    corr = np.corrcoef(resid.T)
    off_diag = corr[~np.eye(len(corr), dtype=bool)]
    assert np.abs(off_diag).max() < _N_SE / np.sqrt(n)
    env.close()


def test_cpu_signal_dependent_sd_scales_with_excitation() -> None:
    sd, n = 0.2, 3000
    env = _noisy(_ELBOW, {"signal_dependent_std": sd})
    env.reset(seed=0)
    for u in (0.25, 0.5):
        action = np.full(env.action_space.shape, 0.5 + np.log(u / (1 - u)) / 5.0)
        action = action.astype(np.float32)
        u_exact = float(sigmoid_muscle_activation(action[0], np))
        resid = _excitations(env, action, n) - u_exact
        sigma = sd * u_exact
        assert abs(resid.std(ddof=1) - sigma) < _sd_tolerance(sigma, resid.size)
    env.close()


def _rollout(env: gym.Env, seed: int, steps: int = 15) -> np.ndarray:
    obs, _ = env.reset(seed=seed)
    actions = np.random.default_rng(123).uniform(
        -1.0, 1.0, (steps, *env.action_space.shape)
    )
    out = [obs]
    for a in actions:
        out.append(env.step(a.astype(np.float32))[0])
    return np.array(out)


def test_cpu_seeded_noisy_rollouts_reproduce() -> None:
    """Same seed, same noise; another seed, other noise (this reset ignores the seed)."""
    kwargs = {"reset_type": "init"}  # fixed target, fixed initial pose
    env_id, noise = "myoElbowPose1D6MFixed-v0", MotorNoiseCfg.van_beers_2004()
    clean = make_env(env_id, **kwargs)
    np.testing.assert_array_equal(_rollout(clean, 3), _rollout(clean, 4))
    a = _noisy(env_id, noise, **kwargs)
    b = _noisy(env_id, noise, **kwargs)
    np.testing.assert_array_equal(_rollout(a, 3), _rollout(b, 3))
    assert not np.array_equal(_rollout(a, 3), _rollout(a, 4))


@pytest.mark.parametrize(
    "env_id",
    [_ELBOW, "myoFingerReachRandom-v0", "myoChallengeDieReorientP1-v0"],
)
def test_cpu_noise_off_by_default_and_leaves_rng_untouched(env_id: str) -> None:
    """No wrapper, or a disabled config, rolls out identically without drawing."""
    ref_env = make_env(env_id)
    assert "noise" not in ref_env.unwrapped.ctrl_stages
    ref = _rollout(ref_env, 0)
    for off in (None, {}, MotorNoiseCfg()):
        np.testing.assert_array_equal(_rollout(_noisy(env_id, off), 0), ref)

    def _rng_moves(env: gym.Env) -> bool:
        env.reset(seed=0)
        before = env.unwrapped.np_random.bit_generator.state
        env.step(np.zeros(env.action_space.shape, np.float32))
        return env.unwrapped.np_random.bit_generator.state != before

    assert not _rng_moves(ref_env)
    assert _rng_moves(_noisy(env_id, {"constant_std": 0.1}))


def test_cpu_noise_precedes_fatigue_and_reafferentation(monkeypatch) -> None:
    """Fatigue receives the noisy excitation; reafferentation reroutes the noisy EIP command."""
    fati = _noisy("myoFatiElbowPose1D6MRandom-v0", {"constant_std": 0.05})
    fati.reset(seed=0)
    seen: list[np.ndarray] = []
    fatigue = fati.muscle_fatigue
    original = fatigue.compute_act

    def _record(excitation: np.ndarray, *args, **kwargs):
        seen.append(np.array(excitation, copy=True))
        return original(excitation, *args, **kwargs)

    monkeypatch.setattr(fatigue, "compute_act", _record)
    fati.step(np.full(fati.action_space.shape, 0.5, np.float32))
    assert np.abs(seen[0] - 0.5).max() > 1e-3  # noisy, not sigmoid(0.5) = 0.5

    reaf = _noisy("myoReafHandPoseRandom-v0", {"constant_std": 0.05})
    reaf.reset(seed=0)
    reaf.step(np.full(reaf.action_space.shape, 0.5, np.float32))
    base = reaf.unwrapped
    epl, eip = base.model.actuator("EPL_r").id, base.model.actuator("EIP_r").id
    assert base.data.ctrl[eip] == 0.0
    assert abs(base.data.ctrl[epl] - 0.5) > 1e-3  # EIP's noisy command


def test_stage_order_does_not_depend_on_the_wrapping_order() -> None:
    """Noise, fatigue and reafferentation run in their fixed order, however wrapped."""
    cfg = {"constant_std": 0.05}
    a = ReafferentationWrapper(
        FatigueWrapper(MotorNoiseWrapper(make_env("myoHandPoseRandom-v0"), cfg))
    )
    b = MotorNoiseWrapper(
        FatigueWrapper(
            ReafferentationWrapper(make_env("myoHandPoseRandom-v0")),
        ),
        cfg,
    )
    assert (
        a.unwrapped.ctrl_stages
        == b.unwrapped.ctrl_stages
        == (
            "noise",
            "fatigue",
            "reroute",
        )
    )
    np.testing.assert_array_equal(_rollout(a, 0), _rollout(b, 0))


def test_a_stage_is_installed_once() -> None:
    """A second wrapper of the same kind, or one on a registered variant, is rejected."""
    env = _noisy(_ELBOW, {"constant_std": 0.05})
    with pytest.raises(ValueError, match="already installed"):
        MotorNoiseWrapper(env, {"constant_std": 0.1})
    with pytest.raises(ValueError, match="already installed"):
        FatigueWrapper(make_env("myoFatiElbowPose1D6MRandom-v0"))


def test_sarcopenia_is_applied_once() -> None:
    from myosuite.envs.wrappers import SarcopeniaWrapper

    healthy = make_env(_ELBOW).unwrapped.model.actuator_gainprm[:, 2].copy()
    env = SarcopeniaWrapper(make_env(_ELBOW))
    np.testing.assert_allclose(
        env.unwrapped.model.actuator_gainprm[:, 2], 0.5 * healthy
    )
    with pytest.raises(ValueError, match="already applied"):
        SarcopeniaWrapper(env)
    with pytest.raises(ValueError, match="already applied"):
        SarcopeniaWrapper(make_env("myoSarcElbowPose1D6MRandom-v0"))
    np.testing.assert_allclose(
        env.unwrapped.model.actuator_gainprm[:, 2], 0.5 * healthy
    )


@pytest.mark.parametrize(
    "env_id",
    [
        _ELBOW,
        "myoChallengeDieReorientP1-v0",
        "myoHandPenTwirlRandom-v0",  # subclass
        "myoChallengeBaodingP1-v1",
    ],
)
def test_wrapper_applies_to_the_muscle_envs(env_id: str) -> None:
    env = _noisy(env_id, MotorNoiseCfg.van_beers_2004())
    assert env.motor_noise == MotorNoiseCfg.van_beers_2004()
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape, np.float32))
    env.close()


def test_wrapper_rejects_envs_without_stages() -> None:
    """An env whose pipeline does not run stages cannot be wrapped (it would be ignored)."""
    with pytest.raises(TypeError, match="does not"):
        MotorNoiseWrapper(gym.make("CartPole-v1"), {"constant_std": 0.1})


@pytest.mark.parametrize(
    "env_id, kwarg",
    [
        (_ELBOW, "motor_noise"),
        (_ELBOW, "muscle_condition"),
        ("myoChallengeSoccerP1-v0", "fatigue_reset_random"),
    ],
)
def test_removed_constructor_kwargs_raise(env_id: str, kwarg: str) -> None:
    with pytest.raises(TypeError, match=f"no longer takes '{kwarg}'"):
        make_env(
            env_id, **{kwarg: {"constant_std": 0.1} if kwarg == "motor_noise" else "x"}
        )


def test_registered_condition_wrappers_sit_on_envs_with_stages() -> None:
    """Every id registered with a muscle wrapper has an env class that runs the stages."""
    names = {"SarcopeniaWrapper", "FatigueWrapper", "ReafferentationWrapper"}
    checked = 0
    for spec in gym.registry.values():
        if not any(w.name in names for w in spec.additional_wrappers or ()):
            continue
        entry = spec.entry_point
        cls = load_env_creator(entry) if isinstance(entry, str) else entry
        assert getattr(cls, "supports_ctrl_stages", False), spec.id
        checked += 1
    assert checked > 100


def test_cpu_motor_actuators_are_not_noised() -> None:
    noisy = _noisy("motorFingerPoseFixed-v0", MotorNoiseCfg.van_beers_2004())
    clean = make_env("motorFingerPoseFixed-v0")
    np.testing.assert_array_equal(_rollout(noisy, 0), _rollout(clean, 0))


def test_wrapped_env_survives_pickle_and_deepcopy() -> None:
    """Restoring a wrapped env re-installs its stages (the env is rebuilt from its constructor args)."""
    import copy
    import pickle

    env = FatigueWrapper(
        MotorNoiseWrapper(make_env(_ELBOW), {"constant_std": 0.05}),
        fatigue_reset_random=True,
    )
    for clone in (pickle.loads(pickle.dumps(env)), copy.deepcopy(env)):
        assert clone.unwrapped.ctrl_stages == ("noise", "fatigue")
        assert clone.fatigue_reset_random
        assert clone.muscle_fatigue is clone.unwrapped.muscle_fatigue
        assert clone.motor_noise == MotorNoiseCfg(constant_std=0.05)
        np.testing.assert_array_equal(_rollout(clone, 0), _rollout(env, 0))


def test_set_motor_noise_reaches_the_wrapper_under_others() -> None:
    """``set_motor_noise`` is forwarded; assigning ``env.motor_noise`` on an outer wrapper is not."""
    env = FatigueWrapper(MotorNoiseWrapper(make_env(_ELBOW)))
    quiet = _rollout(env, 0)
    env.set_motor_noise({"constant_std": 0.05})
    assert env.env.motor_noise == MotorNoiseCfg(constant_std=0.05)
    assert not np.array_equal(_rollout(env, 0), quiet)
    env.set_motor_noise(None)
    np.testing.assert_array_equal(_rollout(env, 0), quiet)
    # a plain gymnasium wrapper on top: reach the method with get_wrapper_attr
    stats = gym.wrappers.RecordEpisodeStatistics(env)
    stats.get_wrapper_attr("set_motor_noise")({"constant_std": 0.05})
    assert env.env.motor_noise.enabled


# ── custom stages ────────────────────────────────────────────────────────────

_CALLS: list[str] = []


def _record(label: str):
    def apply(env, ctrl):
        _CALLS.append(label)
        return ctrl

    return apply


def _cap(env, ctrl):
    """Module-level, so that the wrapped env can be pickled."""
    idx = env._stage_muscle_index()
    ctrl[idx] = np.minimum(ctrl[idx], 0.3)
    return ctrl


def _count_resets(env) -> None:
    _CALLS.append("reset")


def test_custom_stages_run_by_priority_between_the_built_in_ones() -> None:
    from myosuite.envs.wrappers import CtrlStageWrapper

    _CALLS.clear()
    env = make_env("myoFatiElbowPose1D6MRandom-v0")
    # wrapped in the "wrong" order on purpose
    for name, order in (("late", 50), ("early", 15), ("mid", 25)):
        env = CtrlStageWrapper(env, _record(name), name=name, order=order)
    env = MotorNoiseWrapper(env, {"constant_std": 0.01})
    assert env.unwrapped.ctrl_stages == ("early", "noise", "mid", "fatigue", "late")
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape, np.float32))
    assert _CALLS == ["early", "mid", "late"]


def test_custom_stage_changes_the_control_and_resets_with_the_env() -> None:
    from myosuite.envs.wrappers import CtrlStageWrapper

    _CALLS.clear()
    env = CtrlStageWrapper(
        make_env(_ELBOW), _cap, name="cap", order=25, reset=_count_resets
    )
    env.reset(seed=0)
    env.step(np.ones(env.action_space.shape, np.float32))  # sigmoid(1) = 0.92
    base = env.unwrapped
    assert base.data.ctrl[base._muscle_act_ind].max() <= 0.3 + 1e-6
    assert _CALLS == ["reset"]


def test_custom_stage_validation() -> None:
    from myosuite.envs.wrappers import CtrlStageWrapper

    env = make_env(_ELBOW)
    for bad in (10, 100, 5, 500):
        with pytest.raises(ValueError, match="between 10 and 100"):
            CtrlStageWrapper(env, _record("x"), name="x", order=bad)
    with pytest.raises(ValueError, match="built-in"):
        CtrlStageWrapper(env, _record("x"), name="noise", order=25)
    env = CtrlStageWrapper(env, _record("x"), name="x", order=25)
    with pytest.raises(ValueError, match="already installed"):
        CtrlStageWrapper(env, _record("x"), name="x", order=26)
    with pytest.raises(ValueError, match="fixed order"):
        env.unwrapped.add_ctrl_stage("fatigue", _record("f"), order=99)


def test_custom_stages_default_to_after_the_builtins_in_installation_order() -> None:
    import warnings

    from myosuite.envs.wrappers import CtrlStageWrapper

    _CALLS.clear()
    env = make_env(_ELBOW)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no StageOrderWarning without explicit orders
        env = CtrlStageWrapper(env, _record("second"), name="second")
        env = CtrlStageWrapper(env, _record("first"), name="first")
        env = CtrlStageWrapper(env, _record("early"), name="early", order=15)
        env = FatigueWrapper(MotorNoiseWrapper(env, {"constant_std": 0.01}))
    # built-ins in their fixed order, custom stages after them in installation order
    assert env.unwrapped.ctrl_stages == (
        "early",
        "noise",
        "fatigue",
        "second",
        "first",
    )
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape, np.float32))
    assert _CALLS == ["early", "second", "first"]


def test_same_order_warns_prominently_and_runs_in_installation_order() -> None:
    from myosuite.envs.muscle_stages import StageOrderWarning
    from myosuite.envs.wrappers import CtrlStageWrapper

    _CALLS.clear()
    env = CtrlStageWrapper(make_env(_ELBOW), _record("b"), name="b", order=25)
    with pytest.warns(StageOrderWarning, match="STAGE ORDER CLASH.*'b', 'a'.*order 25"):
        env = CtrlStageWrapper(env, _record("a"), name="a", order=25)
    assert env.unwrapped.ctrl_stages == ("b", "a")
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape, np.float32))
    assert _CALLS == ["b", "a"]
    # the clash with a built-in stage warns as well
    with pytest.warns(StageOrderWarning, match="noise"):
        MotorNoiseWrapper(
            CtrlStageWrapper(make_env(_ELBOW), _record("c"), name="c", order=20),
            {"constant_std": 0.01},
        )


def test_custom_stage_survives_pickle() -> None:
    import pickle

    from myosuite.envs.wrappers import CtrlStageWrapper

    env = CtrlStageWrapper(make_env(_ELBOW), _cap, name="cap", order=25)
    clone = pickle.loads(pickle.dumps(env))
    assert clone.unwrapped.ctrl_stages == ("cap",)
    np.testing.assert_array_equal(_rollout(clone, 0), _rollout(env, 0))


# ── portable excitation stages (CPU) ─────────────────────────────────────────


def test_low_pass_stage_on_the_cpu_env() -> None:
    import functools

    from myosuite.envs.muscle_stages import LowPassStage
    from myosuite.envs.wrappers import ExcitationStageWrapper

    env = ExcitationStageWrapper(make_env(_ELBOW), functools.partial(LowPassStage, 0.5))
    assert env.unwrapped.ctrl_stages == ("lowpass",)
    env.reset(seed=0)
    base, idx = env.unwrapped, env.unwrapped._muscle_act_ind
    hi, lo = (
        np.ones(env.action_space.shape, np.float32),
        -np.ones(env.action_space.shape, np.float32),
    )
    env.step(hi)  # the first step after a reset passes through
    first = base.data.ctrl[idx].copy()
    env.step(lo)  # then y = y + 0.5 (u - y)
    u_lo = 1.0 / (1.0 + np.exp(5.0 * 1.5))
    np.testing.assert_allclose(
        base.data.ctrl[idx], first + 0.5 * (u_lo - first), rtol=1e-5
    )
    env.reset(seed=0)
    env.step(lo)  # the filter state was reset
    np.testing.assert_allclose(base.data.ctrl[idx], u_lo, rtol=1e-5)


def test_low_pass_stage_keeps_its_state_on_the_input_device() -> None:
    """The filter state lives on the excitations' device (CUDA on the GPU twin).

    ``meta`` stands in for CUDA where no GPU is present: a CPU-side flag fails on it the
    same way.
    """
    torch = pytest.importorskip("torch")
    from myosuite.envs.muscle_stages import LowPassStage

    for device in ("meta", *(("cuda",) if torch.cuda.is_available() else ())):
        stage = LowPassStage(0.5)
        u = torch.rand(3, 6, device=device)
        stage(u, torch)
        stage.reset(torch.tensor([1], device=device))
        stage(u, torch)
        assert stage._y.device == stage._fresh.device == u.device


def test_excitation_stage_name_and_pickle() -> None:
    import functools
    import pickle

    from myosuite.envs.muscle_stages import LowPassStage
    from myosuite.envs.wrappers import ExcitationStageWrapper

    with pytest.raises(ValueError, match="built-in"):
        ExcitationStageWrapper(
            make_env(_ELBOW), functools.partial(LowPassStage, 0.5, "noise")
        )
    env = ExcitationStageWrapper(make_env(_ELBOW), functools.partial(LowPassStage, 0.5))
    clone = pickle.loads(pickle.dumps(env))
    assert clone.unwrapped.ctrl_stages == ("lowpass",)
    np.testing.assert_array_equal(_rollout(clone, 0), _rollout(env, 0))
