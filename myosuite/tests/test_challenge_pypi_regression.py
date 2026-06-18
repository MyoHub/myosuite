# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests: current challenge env implementations vs public PyPI baselines.

What is tested
--------------
Step-by-step bit-exact replay is *not* appropriate after a full native rewrite
because different implementations consume the internal RNG in different orders
even when seeded identically.  Instead this test suite checks properties that
*must* be consistent between versions:

1. **obs_shape** — observation vector length must match exactly.
2. **action_shape** — action vector length must match exactly.
3. **obs_dict_keys** — the named observation keys returned by ``_get_obs_dict()``
   must be a superset of the keys present in the baseline (new envs may add
   keys but must not drop any).  Only checked when the baseline recorded keys
   (old BaseV0 envs with ``obs_dict`` attribute).
4. **Reward finiteness** — every step reward over 200 steps must be finite.
5. **Termination contract** — ``step()`` must return a 5-tuple with ``bool``
   flags; ``reset()`` must return a 2-tuple.
6. **Reward distribution** — mean reward over 200 random steps must have the
   same sign as the PyPI baseline and be within 80 % relative tolerance.
   Envs with high-variance / terrain-dependent rewards use a looser tolerance.

Baselines are generated from the latest public PyPI ``myosuite`` with::

    python scripts/generate_pypi_challenge_baselines.py

If the baseline directory is empty or missing, **all parametrized tests are
skipped** rather than failing.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

import myosuite
from myosuite.utils import gym

myosuite.register_all_envs()

pytestmark = pytest.mark.tier2

_BASELINE_DIR = Path(__file__).parent / "pypi_challenge_baselines"
_N_STEPS = 200
_SEED = 42

# Per-env override for reward mean relative tolerance.
# Terrain-based envs can produce very different mean rewards across RNG streams.
_REWARD_MEAN_RTOL: dict[str, float] = {
    # Terrain-based envs: heightfield sampling differs across RNG streams.
    "myoChallengeOslRunFixed-v0": 2.0,
    "myoChallengeOslRunRandom-v0": 2.0,
    "myoFatiChallengeOslRunFixed-v0": 2.0,
    "myoFatiChallengeOslRunRandom-v0": 2.0,
    "myoSarcChallengeOslRunFixed-v0": 2.0,
    "myoSarcChallengeOslRunRandom-v0": 2.0,
}
_DEFAULT_REWARD_MEAN_RTOL = 0.80

# Per-env expected obs-vector shrinkage relative to the PyPI baseline, for
# *intentional* model simplifications made after the baseline was recorded.
# Each entry must point to the code comment that documents why the model
# changed. Do NOT regenerate the baseline pkl to paper over this -- the
# baseline is captured from the *public PyPI release* (see module docstring
# and scripts/generate_pypi_challenge_baselines.py), so it is the one fixed
# point this test relies on to catch unintended drift. Silently regenerating
# it would erase that signal instead of explaining the divergence.
_OBS_SHAPE_DELTA: dict[str, int] = {
    # myo_sim pip's leg model has a simplified hinge knee, lacking 3 passive
    # coupling DOFs (e.g. "knee_angle_l_beta_rotation1") that
    # SoccerEnv.MYO_JOINTS filters out at runtime -- see the comment next to
    # "self.MYO_JOINTS = [j for j in self.MYO_JOINTS if j in _model_joints]"
    # in SoccerEnv.__init__ (myosuite/envs/myo/tasks/challenge/soccer.py).
    "myoChallengeSoccerP1-v0": -3,
    "myoChallengeSoccerP2-v0": -3,
}


def _collect_env_ids() -> list[str]:
    if not _BASELINE_DIR.is_dir():
        return []
    return [p.stem for p in sorted(_BASELINE_DIR.glob("*.pkl"))]


def _load_baseline(env_id: str) -> dict:
    path = _BASELINE_DIR / f"{env_id}.pkl"
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    # Support old list-of-dicts format (pre-restructure).
    if isinstance(data, list):
        steps = data
        return {
            "steps": steps,
            "obs_dict_keys": [],
            "reward_mean": float(np.mean([s["rwd"] for s in steps])),
            "reward_std": float(np.std([s["rwd"] for s in steps])),
            "obs_shape": list(steps[0]["obs"].shape),
            "action_shape": list(steps[0]["action"].shape),
        }
    return data


def _make_env(env_id: str):
    try:
        return gym.make(env_id)
    except Exception as exc:
        pytest.skip(f"Cannot instantiate {env_id}: {exc}")


@pytest.mark.parametrize("env_id", _collect_env_ids())
def test_obs_shape_matches_pypi(env_id: str) -> None:
    """Observation vector length must match the PyPI baseline (± documented deltas).

    See ``_OBS_SHAPE_DELTA`` for envs with an accepted, intentional shape
    change since the baseline was recorded.
    """
    baseline = _load_baseline(env_id)
    ref_len = baseline["obs_shape"][0] + _OBS_SHAPE_DELTA.get(env_id, 0)
    ref_shape = (ref_len,)

    env = _make_env(env_id)
    try:
        obs, _ = env.reset(seed=_SEED)
    finally:
        env.close()

    assert obs.shape == ref_shape, (
        f"{env_id}: obs shape {obs.shape} != expected {ref_shape} "
        f"(PyPI baseline {tuple(baseline['obs_shape'])}, "
        f"documented delta {_OBS_SHAPE_DELTA.get(env_id, 0)})"
    )


@pytest.mark.parametrize("env_id", _collect_env_ids())
def test_action_shape_matches_pypi(env_id: str) -> None:
    """Action space shape must be identical to the PyPI baseline."""
    baseline = _load_baseline(env_id)
    ref_shape = tuple(baseline["action_shape"])

    env = _make_env(env_id)
    try:
        current_shape = env.action_space.shape
    finally:
        env.close()

    assert (
        current_shape == ref_shape
    ), f"{env_id}: action shape {current_shape} != PyPI baseline {ref_shape}"


@pytest.mark.parametrize("env_id", _collect_env_ids())
def test_obs_dict_keys_superset_of_pypi(env_id: str) -> None:
    """Current env must expose all obs dict keys that the PyPI env exposed.

    Only checked when the baseline recorded obs_dict_keys (i.e. the PyPI env
    had a BaseV0 ``obs_dict`` attribute).
    """
    baseline = _load_baseline(env_id)
    ref_keys = set(baseline.get("obs_dict_keys", []))
    if not ref_keys:
        pytest.skip(f"{env_id}: baseline has no obs_dict_keys — skipping key check")

    env = _make_env(env_id)
    try:
        env.reset(seed=_SEED)
        # Access obs_dict through the new pattern.
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "_get_obs_dict"):
            accessor = getattr(unwrapped, "_accessor", None)
            if accessor is not None:
                obs_dict = unwrapped._get_obs_dict(accessor)
            else:
                obs_dict = {}
        elif hasattr(unwrapped, "obs_dict"):
            obs_dict = unwrapped.obs_dict
        else:
            obs_dict = {}
        current_keys = set(obs_dict.keys())
    finally:
        env.close()

    if not current_keys:
        pytest.skip(
            f"{env_id}: cannot read obs_dict from current env — skipping key check"
        )

    missing = ref_keys - current_keys
    assert (
        not missing
    ), f"{env_id}: current env is missing obs keys present in PyPI: {sorted(missing)}"


@pytest.mark.parametrize("env_id", _collect_env_ids())
def test_reward_finiteness(env_id: str) -> None:
    """All step rewards must be finite over a 200-step rollout."""
    env = _make_env(env_id)
    try:
        env.reset(seed=_SEED)
        for i in range(_N_STEPS):
            _, rwd, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(rwd), f"{env_id} step {i}: non-finite reward {rwd}"
            if terminated or truncated:
                env.reset()
    finally:
        env.close()


@pytest.mark.parametrize("env_id", _collect_env_ids())
def test_step_returns_5tuple(env_id: str) -> None:
    """step() must return a 5-tuple with correct types."""
    env = _make_env(env_id)
    try:
        env.reset(seed=_SEED)
        result = env.step(env.action_space.sample())
    finally:
        env.close()

    assert (
        len(result) == 5
    ), f"{env_id}: step() returned {len(result)}-tuple, expected 5"
    _, rwd, terminated, truncated, info = result
    assert isinstance(rwd, float), f"{env_id}: reward must be float, got {type(rwd)}"
    assert isinstance(terminated, bool), f"{env_id}: terminated must be bool"
    assert isinstance(truncated, bool), f"{env_id}: truncated must be bool"
    assert isinstance(info, dict), f"{env_id}: info must be dict"


@pytest.mark.parametrize("env_id", _collect_env_ids())
def test_reset_returns_2tuple(env_id: str) -> None:
    """reset() must return a 2-tuple (obs, info)."""
    env = _make_env(env_id)
    try:
        result = env.reset(seed=_SEED)
    finally:
        env.close()

    assert (
        len(result) == 2
    ), f"{env_id}: reset() returned {len(result)}-tuple, expected 2"
    obs, info = result
    assert obs.ndim == 1, f"{env_id}: obs should be 1-D, got shape {obs.shape}"
    assert np.all(np.isfinite(obs)), f"{env_id}: reset obs contains non-finite values"
    assert isinstance(info, dict)


@pytest.mark.parametrize("env_id", _collect_env_ids())
def test_reward_mean_sign_vs_pypi(env_id: str) -> None:
    """Mean reward sign over 200 steps must match the PyPI baseline.

    Catches inverted reward implementations while tolerating numerical drift
    from different RNG streams (terrain generation, task sampling).
    """
    baseline = _load_baseline(env_id)
    ref_mean = baseline["reward_mean"]
    rtol = _REWARD_MEAN_RTOL.get(env_id, _DEFAULT_REWARD_MEAN_RTOL)

    env = _make_env(env_id)
    action_rng = np.random.default_rng(_SEED)
    try:
        env.reset(seed=_SEED)
        rewards = []
        for _ in range(_N_STEPS):
            # Fixed action stream (independent of env RNG) so mid-episode
            # ``reset()`` without seed does not change which actions are applied.
            action = action_rng.uniform(
                env.action_space.low, env.action_space.high, size=env.action_space.shape
            )
            _, rwd, terminated, truncated, _ = env.step(action)
            rewards.append(float(rwd))
            if terminated or truncated:
                env.reset()
    finally:
        env.close()

    cur_mean = float(np.mean(rewards))

    if abs(ref_mean) > 0.05:
        # Sign must agree for non-trivial reference means.
        assert np.sign(cur_mean) == np.sign(ref_mean), (
            f"{env_id}: reward mean sign flipped — "
            f"current={cur_mean:.4f}, PyPI={ref_mean:.4f}"
        )

    # Magnitude within tolerance.
    scale = max(abs(ref_mean), 0.1)
    assert abs(cur_mean - ref_mean) <= rtol * scale, (
        f"{env_id}: reward mean too far from PyPI baseline — "
        f"current={cur_mean:.4f}, PyPI={ref_mean:.4f} "
        f"(|diff|={abs(cur_mean - ref_mean):.4f} > tol={rtol * scale:.4f}, rtol={rtol})"
    )


def test_baselines_present() -> None:
    """Fail clearly (not skip) if the baseline directory exists but is empty."""
    if not _BASELINE_DIR.exists():
        pytest.skip(
            "Baseline directory not yet created — "
            "run: python scripts/generate_pypi_challenge_baselines.py"
        )
    pkls = list(_BASELINE_DIR.glob("*.pkl"))
    if not pkls:
        pytest.fail(
            f"{_BASELINE_DIR} exists but contains no .pkl files. "
            "Run: python scripts/generate_pypi_challenge_baselines.py"
        )
