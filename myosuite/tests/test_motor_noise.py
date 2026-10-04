# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Motor noise on muscle excitations: term statistics and CPU env wiring.

Statistical checks use fixed seeds and a tolerance of five standard errors
(SE of a sample SD: ``sigma / sqrt(2 (n - 1))``; of a correlation: ``1 / sqrt(n)``).
"""

from __future__ import annotations

import inspect

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import load_env_creator

import myosuite  # noqa: F401
from myosuite.envs.gymnasium_env import MOTOR_NOISE_ENV_CLASSES
from myosuite.terms.base_action import (
    MotorNoiseCfg,
    motor_noise,
    sample_motor_noise,
    sigmoid_muscle_activation,
)

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
    env = gym.make(_ELBOW, motor_noise={"constant_std": c})
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
    env = gym.make(_ELBOW, motor_noise={"signal_dependent_std": sd})
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
    clean = gym.make(env_id, **kwargs)
    np.testing.assert_array_equal(_rollout(clean, 3), _rollout(clean, 4))
    a = gym.make(env_id, motor_noise=noise, **kwargs)
    b = gym.make(env_id, motor_noise=noise, **kwargs)
    np.testing.assert_array_equal(_rollout(a, 3), _rollout(b, 3))
    assert not np.array_equal(_rollout(a, 3), _rollout(a, 4))


@pytest.mark.parametrize(
    "env_id",
    [_ELBOW, "myoFingerReachRandom-v0", "myoChallengeDieReorientP1-v0"],
)
def test_cpu_noise_off_by_default_and_leaves_rng_untouched(env_id: str) -> None:
    """Default and explicitly disabled configs roll out identically without drawing."""
    ref_env = gym.make(env_id)
    assert not ref_env.unwrapped.motor_noise.enabled
    ref = _rollout(ref_env, 0)
    for off in (None, {}, MotorNoiseCfg()):
        np.testing.assert_array_equal(
            _rollout(gym.make(env_id, motor_noise=off), 0), ref
        )

    def _rng_moves(env: gym.Env) -> bool:
        env.reset(seed=0)
        before = env.unwrapped.np_random.bit_generator.state
        env.step(np.zeros(env.action_space.shape, np.float32))
        return env.unwrapped.np_random.bit_generator.state != before

    assert not _rng_moves(ref_env)
    assert _rng_moves(gym.make(env_id, motor_noise={"constant_std": 0.1}))


def test_cpu_noise_precedes_fatigue_and_reafferentation(monkeypatch) -> None:
    """Fatigue receives the noisy excitation; reafferentation reroutes the noisy EIP command."""
    fati = gym.make("myoFatiElbowPose1D6MRandom-v0", motor_noise={"constant_std": 0.05})
    fati.reset(seed=0)
    seen: list[np.ndarray] = []
    fatigue = fati.unwrapped.muscle_fatigue
    original = fatigue.compute_act

    def _record(excitation: np.ndarray, *args, **kwargs):
        seen.append(np.array(excitation, copy=True))
        return original(excitation, *args, **kwargs)

    monkeypatch.setattr(fatigue, "compute_act", _record)
    fati.step(np.full(fati.action_space.shape, 0.5, np.float32))
    assert np.abs(seen[0] - 0.5).max() > 1e-3  # noisy, not sigmoid(0.5) = 0.5

    reaf = gym.make("myoReafHandPoseRandom-v0", motor_noise={"constant_std": 0.05})
    reaf.reset(seed=0)
    reaf.step(np.full(reaf.action_space.shape, 0.5, np.float32))
    base = reaf.unwrapped
    ctrl = base.data.ctrl
    assert ctrl[base.EIPpos] == 0.0
    assert abs(ctrl[base.EPLpos] - 0.5) > 1e-3  # EIP's noisy command


@pytest.mark.parametrize(
    "env_id",
    [
        _ELBOW,
        "myoChallengeDieReorientP1-v0",
        "myoHandPenTwirlRandom-v0",
    ],  # last: subclass
)
def test_supported_env_accepts_enabled_noise(env_id: str) -> None:
    env = gym.make(env_id, motor_noise=MotorNoiseCfg.van_beers_2004())
    assert env.unwrapped.motor_noise == MotorNoiseCfg.van_beers_2004()
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape, np.float32))
    env.close()


@pytest.mark.parametrize(
    "env_id, cls_name",
    [
        ("myoChallengeBaodingP1-v1", "BaodingEnv"),
        ("myoElbowPoseTaskFixed-v0", "ModularTaskEnv"),
    ],
)
def test_unsupported_env_rejects_enabled_noise(env_id: str, cls_name: str) -> None:
    """An env that would ignore the kwarg raises; None or a disabled cfg is accepted."""
    with pytest.raises(
        ValueError, match=rf"{cls_name} does not apply motor_noise.*PoseEnvV0"
    ):
        gym.make(env_id, motor_noise={"constant_std": 0.1})
    for off in (None, {}, MotorNoiseCfg()):
        gym.make(env_id, motor_noise=off).close()


def test_motor_noise_env_classes_match_registry() -> None:
    """The documented class list is exactly the registered classes that set the flag."""
    defining = set()
    for spec in gym.registry.values():
        entry = spec.entry_point
        if not (isinstance(entry, str) and entry.startswith("myosuite.")):
            continue
        cls = load_env_creator(entry)
        if getattr(cls, "supports_motor_noise", False):
            owner = next(k for k in cls.__mro__ if "supports_motor_noise" in vars(k))
            assert "motor_noise" in inspect.signature(owner.__init__).parameters, owner
            defining.add(owner.__name__)
    assert defining == set(MOTOR_NOISE_ENV_CLASSES)


def test_cpu_motor_actuators_are_not_noised() -> None:
    noisy = gym.make(
        "motorFingerPoseFixed-v0", motor_noise=MotorNoiseCfg.van_beers_2004()
    )
    clean = gym.make("motorFingerPoseFixed-v0")
    np.testing.assert_array_equal(_rollout(noisy, 0), _rollout(clean, 0))
