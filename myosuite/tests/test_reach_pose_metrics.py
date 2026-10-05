"""``get_metrics`` of the CPU reach and pose envs on ``examine_policy`` traces.

Scripted (non-learned) constant policies; the metrics are checked against the
env's own ``solved`` flags and observations, and the reset state must open each
trajectory.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the envs)
from myosuite.utils.movement_metrics import point_to_point_metrics
from myosuite.utils.path_utils import path_obs_series

pytestmark = pytest.mark.tier1

_KEYS = {
    "success",
    "final_error",
    "time_to_target",
    "time_to_acquire",
    "target_entries",
    "movement_time",
    "peak_speed",
    "time_to_peak_ratio",
    "speed_peaks",
    "ldlj",
    "sparc",
    "straightness",
    "effort",
}


class _ConstantPolicy:
    """``examine_policy`` policy that always returns the same action."""

    def __init__(self, action: np.ndarray) -> None:
        self.action = action.astype(np.float32)

    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, dict]:
        return self.action, {"evaluation": self.action}


def _rollouts(env_id: str, horizon: int, episodes: int, seed: int = 3):
    env = gym.make(env_id).unwrapped
    env.reset(seed=seed)
    action = np.linspace(-1.0, 1.0, env.action_space.shape[0])
    trace = env.examine_policy(
        _ConstantPolicy(action),
        horizon=horizon,
        num_episodes=episodes,
        mode="evaluation",
    )
    return env, trace


def test_reach_get_metrics_matches_env_signals():
    env, trace = _rollouts("myoFingerReachRandom-v0", horizon=40, episodes=3)
    metrics = env.get_metrics(trace)
    assert set(metrics) == _KEYS
    paths = list(trace)
    finals = [p["env_infos"]["obs_dict"]["reach_err"][-1] for p in paths]
    solved = [bool(p["env_infos"]["rwd_dict"]["solved"][-1]) for p in paths]
    assert metrics["success"] == pytest.approx(np.mean(solved))
    assert metrics["final_error"] == pytest.approx(
        np.mean(np.linalg.norm(finals, axis=-1))
    )
    # Effort is per trial (all samples from the reset), then averaged over trials,
    # so episodes that terminate early weigh the same as full ones.
    per_trial = [np.mean(path_obs_series(p, ("act",))["act"] ** 2) for p in paths]
    assert 0.0 < metrics["effort"] <= 1.0
    assert metrics["effort"] == pytest.approx(np.mean(per_trial))
    assert metrics["peak_speed"] > 0.0 and metrics["movement_time"] > 0.0
    assert math.isfinite(metrics["ldlj"]) and metrics["straightness"] >= 1.0


def test_reach_metrics_time_to_target_uses_reset_time_base():
    # Fixed target: the rollout uses no RNG, so it is the same on every NumPy version
    # (a random target may lead the finger away and never into the radius below).
    env, trace = _rollouts("myoFingerReachFixed-v0", horizon=40, episodes=1)
    path = trace[0]
    obs = path_obs_series(path, ("tip_pos", "reach_err"))
    # examine_policy records are time-aligned: record t is state s_t, s_0 the reset.
    dist = np.linalg.norm(path["env_infos"]["obs_dict"]["reach_err"], axis=-1)
    times = np.asarray(path["time"], dtype=float)
    assert len(obs["tip_pos"]) == len(dist) == len(times) and times[0] == 0.0
    # A radius the finger certainly enters: the mean of its start and closest distance.
    radius = 0.5 * (dist[0] + dist.min())
    m = point_to_point_metrics(
        obs["tip_pos"], obs["tip_pos"] + obs["reach_err"], env.dt, radius
    )
    first = int(np.flatnonzero(dist < radius)[0])
    assert first > 0
    assert m["time_to_target"] == pytest.approx(times[first])


def test_path_obs_series_realigns_the_old_post_step_layout():
    # States s_0..s_4 of two obs keys; the pre-#471 examine_policy logged the
    # post-step obs dict (s_1..s_4) and repeated the last record.
    states = np.arange(5.0)[:, None] * np.array([1.0, 10.0])
    aligned = {"a": states[:, :1], "b": states[:, 1:]}
    post_step = {k: np.concatenate([v[1:], v[-1:]]) for k, v in aligned.items()}
    for obs_dict in (aligned, post_step):
        path = {"observations": states, "env_infos": {"obs_dict": obs_dict}}
        series = path_obs_series(path, ("a", "b"))
        for key, expected in aligned.items():
            np.testing.assert_array_equal(series[key], expected)


def test_pose_get_metrics_matches_env_signals():
    env, trace = _rollouts("myoElbowPose1D6MRandom-v0", horizon=50, episodes=3)
    metrics = env.get_metrics(trace)
    assert set(metrics) == _KEYS
    paths = list(trace)
    solved = [bool(p["env_infos"]["rwd_dict"]["solved"][-1]) for p in paths]
    errors = [np.linalg.norm(p["env_infos"]["obs_dict"]["pose_err"][-1]) for p in paths]
    assert metrics["success"] == pytest.approx(np.mean(solved))
    assert metrics["final_error"] == pytest.approx(np.mean(errors))
    assert metrics["peak_speed"] > 0.0


def test_get_metrics_requires_trajectory_keys():
    env, trace = _rollouts("myoFingerReachFixed-v0", horizon=5, episodes=1)
    path = trace[0]
    del path["env_infos"]["obs_dict"]["tip_pos"]
    with pytest.raises(KeyError, match="tip_pos"):
        env.get_metrics([path])
