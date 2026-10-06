"""The batched MuscleMimic mjlab bridge keeps history and normalizer statistics per env."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myosuite.integrations.musclemimic.mjlab_onnx_policy import (  # noqa: E402
    _BatchedObservationHistoryBuffer,
    _FullbodyMjlabPolicyBridge,
)
from myosuite.integrations.musclemimic.running_stats import (  # noqa: E402
    torch_running_mean_std_update,
    torch_running_mean_std_update_per_env,
)

pytestmark = pytest.mark.tier1

_N, _D = 3, 5


def test_per_env_running_update_equals_independent_single_env_runs() -> None:
    gen = torch.Generator().manual_seed(0)
    mean0, var0, count0 = torch.randn(_D), torch.rand(_D) + 0.5, torch.tensor(100.0)
    mean = mean0.expand(_N, -1).clone()
    var = var0.expand(_N, -1).clone()
    count = count0.expand(_N).clone()
    singles = [(mean0.clone(), var0.clone(), count0.clone()) for _ in range(_N)]
    for _ in range(12):
        obs = torch.randn(_N, _D, generator=gen)
        norm, mean, var, count = torch_running_mean_std_update_per_env(
            obs, mean, var, count
        )
        for i in range(_N):
            ref, *singles[i] = torch_running_mean_std_update(
                obs[i : i + 1], *singles[i]
            )
            torch.testing.assert_close(norm[i : i + 1], ref)
            torch.testing.assert_close(mean[i], singles[i][0])
            torch.testing.assert_close(var[i], singles[i][1])
            assert count[i] == singles[i][2]


def test_history_restarts_only_the_flagged_envs() -> None:
    hist = _BatchedObservationHistoryBuffer(3)
    first = np.arange(_N * 2, dtype=np.float32).reshape(_N, 2)
    hist.reset(first)
    hist.step(first + 10)
    out = hist.step(first + 20, np.array([False, True, False]))
    frames = out.reshape(_N, 3, 2)
    np.testing.assert_array_equal(frames[1], [[0, 0], [0, 0], first[1] + 20])
    for i in (0, 2):
        np.testing.assert_array_equal(
            frames[i], [first[i], first[i] + 10, first[i] + 20]
        )


def _bridge(steps: list[int]) -> _FullbodyMjlabPolicyBridge:
    bridge = object.__new__(_FullbodyMjlabPolicyBridge)
    bridge._env_indices = tuple(range(len(steps)))
    bridge._unwrapped = SimpleNamespace(episode_length_buf=torch.tensor(steps))
    bridge._last_steps = None
    return bridge


def test_episode_start_mask_follows_each_envs_step_counter() -> None:
    bridge = _bridge([4, 4, 4])
    assert bridge._episode_start_mask().all()  # first call: every env starts
    bridge._unwrapped.episode_length_buf = torch.tensor([5, 5, 5])
    assert not bridge._episode_start_mask().any()
    bridge._unwrapped.episode_length_buf = torch.tensor([6, 0, 6])  # env 1 reset
    assert bridge._episode_start_mask().tolist() == [False, True, False]
    bridge._unwrapped.episode_length_buf = torch.tensor([7, 1, 2])  # env 2 reset
    assert bridge._episode_start_mask().tolist() == [False, False, True]
    bridge._unwrapped.episode_length_buf = torch.tensor([8, 2, 3])
    assert not bridge._episode_start_mask().any()
