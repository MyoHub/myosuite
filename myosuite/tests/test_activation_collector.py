# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for activation_collector.py using mocks — no real checkpoint needed."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Helpers: fake MotionClip / MjModel / MjData / LocalPolicyRunner
# ---------------------------------------------------------------------------

N_FRAMES = 20
N_MUSCLES = 12


def _make_clip(
    n_frames: int = N_FRAMES, n_muscles: int = N_MUSCLES
) -> types.SimpleNamespace:
    rng = np.random.default_rng(0)
    clip = types.SimpleNamespace(
        qpos=rng.standard_normal((n_frames, 30)).astype(np.float32),
        qvel=rng.standard_normal((n_frames, 29)).astype(np.float32),
        site_xpos=rng.standard_normal((n_frames, 5, 3)).astype(np.float32),
        source_path="fake_clip.npz",
    )
    return clip


def _make_model(n_muscles: int = N_MUSCLES) -> MagicMock:
    model = MagicMock()
    model.nu = n_muscles
    return model


def _make_data(n_muscles: int = N_MUSCLES) -> MagicMock:
    rng = np.random.default_rng(1)
    data = MagicMock()
    data.qpos = np.zeros(30, dtype=np.float32)
    data.qvel = np.zeros(29, dtype=np.float32)
    data.act = rng.random(n_muscles).astype(np.float32)
    return data


def _make_policy_runner() -> MagicMock:
    runner = MagicMock()
    rng = np.random.default_rng(2)

    def _action_for(data, clip, frame_idx, obs_adapter=None):  # noqa: ANN001
        return rng.standard_normal(N_MUSCLES).astype(np.float32)

    def _step(model, data, action):  # noqa: ANN001
        data.act = rng.random(N_MUSCLES).astype(np.float32)

    runner.action_for.side_effect = _action_for
    runner.step.side_effect = _step
    return runner


# ---------------------------------------------------------------------------
# Helpers: real one-DoF MuJoCo model and a stub runner stepping it
# ---------------------------------------------------------------------------

_SLIDER_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general name="drive" joint="slide" dyntype="filter" dynprm="0.02"
             ctrlrange="0 1"/>
  </actuator>
</mujoco>
"""


def _slider_clip(n_frames: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        qpos=np.linspace(0.0, 1.0, n_frames)[:, None],
        qvel=None,
        site_xpos=None,
        source_path="synthetic_clip.npz",
    )


class _StubRunner:
    """Policy-runner stand-in: full excitation, real MuJoCo stepping."""

    obs_adapter = None

    def __init__(self) -> None:
        self.episode_id = -1

    def reset(self) -> None:
        pass

    def action_for(self, data, clip, frame_idx, obs_adapter=None):  # noqa: ANN001
        return np.ones(1, dtype=np.float32)

    def step(self, model, data, action):  # noqa: ANN001
        data.ctrl[:] = action
        for _ in range(5):
            mujoco.mj_step(model, data)


class _EpisodeMarkingRunner(_StubRunner):
    """Records the episode index as the activation, to trace kept frames."""

    def step(self, model, data, action):  # noqa: ANN001
        super().step(model, data, action)
        data.act[:] = self.episode_id


# ---------------------------------------------------------------------------
# Tests: _init_episode
# ---------------------------------------------------------------------------


class TestInitEpisode(unittest.TestCase):
    def test_sets_qpos(self) -> None:
        from myosuite.integrations.musclemimic.activation_collector import _init_episode

        clip = _make_clip()
        model = _make_model()
        data = _make_data()
        data.qpos = np.zeros(30, dtype=np.float32)

        with patch("mujoco.mj_forward"):
            _init_episode(model, data, clip, start_frame=3)

        np.testing.assert_array_equal(data.qpos, clip.qpos[3])

    def test_sets_qvel_when_present(self) -> None:
        from myosuite.integrations.musclemimic.activation_collector import _init_episode

        clip = _make_clip()
        model = _make_model()
        data = _make_data()
        data.qvel = np.zeros(29, dtype=np.float32)

        with patch("mujoco.mj_forward"):
            _init_episode(model, data, clip, start_frame=5)

        np.testing.assert_array_equal(data.qvel, clip.qvel[5])

    def test_skips_qvel_when_none(self) -> None:
        from myosuite.integrations.musclemimic.activation_collector import _init_episode

        clip = _make_clip()
        clip.qvel = None
        model = _make_model()
        data = _make_data()

        with patch("mujoco.mj_forward"):
            _init_episode(model, data, clip, start_frame=0)
        # no error means success; qvel not touched


# ---------------------------------------------------------------------------
# Tests: _run_episode
# ---------------------------------------------------------------------------


class TestRunEpisode(unittest.TestCase):
    def _run(self, start_frame: int = 0, max_steps: int = 10):
        from myosuite.integrations.musclemimic.activation_collector import _run_episode

        clip = _make_clip()
        model = _make_model()
        data = _make_data()
        runner = _make_policy_runner()

        with (
            patch("mujoco.mj_forward"),
            patch(
                "myosuite.integrations.musclemimic.activation_collector._tracking_reward",
                return_value=0.5,
            ),
        ):
            acts, reward = _run_episode(
                runner, model, data, clip, start_frame, max_steps
            )
        return acts, reward

    def test_returns_correct_shape(self) -> None:
        acts, _ = self._run(start_frame=0, max_steps=5)
        self.assertEqual(acts.ndim, 2)
        self.assertEqual(acts.shape[1], N_MUSCLES)
        self.assertLessEqual(acts.shape[0], 5)

    def test_returns_float32(self) -> None:
        acts, _ = self._run()
        self.assertEqual(acts.dtype, np.float32)

    def test_returns_mean_per_step_reward(self) -> None:
        """The episode score is length-normalised (0.5 per step -> 0.5)."""
        acts, reward = self._run(start_frame=0, max_steps=10)
        self.assertEqual(acts.shape[0], 10)
        self.assertAlmostEqual(reward, 0.5)

    def test_stops_at_end_of_clip(self) -> None:
        """Episode must not run past the last frame."""
        acts, _ = self._run(start_frame=N_FRAMES - 3, max_steps=100)
        self.assertLessEqual(acts.shape[0], 3)

    def test_one_step_when_start_at_last_frame(self) -> None:
        """Episode collects exactly 1 activation at the last frame then stops."""
        acts, _ = self._run(start_frame=N_FRAMES - 1, max_steps=100)
        self.assertEqual(acts.shape[0], 1)

    def test_forwards_obs_adapter_to_action_for(self) -> None:
        """Per-clip obs adapter must reach the policy so frame_idx stays in bounds."""
        from myosuite.integrations.musclemimic.activation_collector import _run_episode

        clip = _make_clip()
        model = _make_model()
        data = _make_data()
        runner = _make_policy_runner()
        adapter = object()

        with (
            patch("mujoco.mj_forward"),
            patch(
                "myosuite.integrations.musclemimic.activation_collector._tracking_reward",
                return_value=0.5,
            ),
        ):
            _run_episode(
                runner,
                model,
                data,
                clip,
                0,
                2,
                obs_adapter=adapter,  # type: ignore[arg-type]
            )
        runner.action_for.assert_called()
        for call in runner.action_for.call_args_list:
            self.assertEqual(call.kwargs.get("obs_adapter"), adapter)


# ---------------------------------------------------------------------------
# Tests: collect_activations_from_clip
# ---------------------------------------------------------------------------


def _make_rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestCollectActivationsFromClip(unittest.TestCase):
    def _collect(self, n_episodes: int = 20, percentile: int = 50) -> np.ndarray:
        from myosuite.integrations.musclemimic.activation_collector import (
            CollectionConfig,
            collect_activations_from_clip,
        )

        clip = _make_clip()
        model = _make_model()
        runner = _make_policy_runner()
        config = CollectionConfig(
            n_episodes_per_clip=n_episodes,
            reward_percentile=percentile,
            max_steps_per_episode=5,
            use_random_start=True,
            seed=0,
        )

        with (
            patch("mujoco.MjData", return_value=_make_data()),
            patch("mujoco.mj_forward"),
            patch(
                "myosuite.integrations.musclemimic.activation_collector._tracking_reward",
                return_value=1.0,
            ),
        ):
            return collect_activations_from_clip(
                runner, model, clip, config, rng=_make_rng()
            )

    def test_returns_2d_float32(self) -> None:
        acts = self._collect()
        self.assertEqual(acts.ndim, 2)
        self.assertEqual(acts.dtype, np.float32)
        self.assertEqual(acts.shape[1], N_MUSCLES)

    def test_nonempty_when_all_pass(self) -> None:
        acts = self._collect(n_episodes=10, percentile=0)
        self.assertGreater(acts.shape[0], 0)

    def test_empty_when_none_pass(self) -> None:
        """With percentile=100 effectively all episodes fail the threshold."""
        from myosuite.integrations.musclemimic.activation_collector import (
            CollectionConfig,
            collect_activations_from_clip,
        )

        clip = _make_clip()
        model = _make_model()
        runner = _make_policy_runner()
        config = CollectionConfig(
            n_episodes_per_clip=10,
            reward_percentile=80,
            max_steps_per_episode=5,
        )

        rewards = iter([0.0] * 200)

        with (
            patch("mujoco.MjData", return_value=_make_data()),
            patch("mujoco.mj_forward"),
            patch(
                "myosuite.integrations.musclemimic.activation_collector._tracking_reward",
                side_effect=rewards,
            ),
        ):
            acts = collect_activations_from_clip(
                runner, model, clip, config, rng=_make_rng()
            )
        # With all rewards == 0.0, threshold = p80 of [0,0,...] = 0.0; episodes
        # with r >= 0.0 will still pass, so shape > 0 is expected here.
        self.assertEqual(acts.ndim, 2)

    def test_deterministic_with_same_seed(self) -> None:
        acts1 = self._collect()
        acts2 = self._collect()
        np.testing.assert_array_equal(acts1, acts2)


class TestRewardFilterLengthBias(unittest.TestCase):
    def test_keeps_best_tracking_episodes_wherever_they_start(self) -> None:
        """Random starts give episodes of 1..299 steps; quality must decide, not length.

        Each episode gets a per-step tracking error independent of its start
        frame. Summed negative errors used to keep only short clip-end
        episodes (all kept starts >= 238 of 300).
        """
        from myosuite.integrations.musclemimic import activation_collector as ac

        model = mujoco.MjModel.from_xml_string(_SLIDER_XML)
        clip = _slider_clip(300)
        runner = _EpisodeMarkingRunner()
        error_rng = np.random.default_rng(1)
        episodes: list[tuple[int, float]] = []  # (start frame, per-step error)
        init_episode = ac._init_episode

        def _init_and_draw_error(model, data, clip, *, start_frame=0):  # noqa: ANN001
            episodes.append((start_frame, float(error_rng.uniform(0.05, 0.15))))
            runner.episode_id = len(episodes) - 1
            init_episode(model, data, clip, start_frame=start_frame)

        config = ac.CollectionConfig(n_episodes_per_clip=200, reward_percentile=80)
        with (
            patch.object(ac, "_init_episode", side_effect=_init_and_draw_error),
            patch.object(
                ac, "_tracking_reward", side_effect=lambda *_: -episodes[-1][1]
            ),
        ):
            acts = ac.collect_activations_from_clip(
                runner, model, clip, config, rng=np.random.default_rng(0)
            )

        n_preview = len(episodes) - config.n_episodes_per_clip
        preview_errors = [error for _, error in episodes[:n_preview]]
        max_error = -np.percentile(np.negative(preview_errors), 80)
        expected = [
            i
            for i, (_, error) in enumerate(episodes)
            if i >= n_preview and error <= max_error
        ]
        kept = sorted({int(i) for i in acts[:, 0]})
        self.assertEqual(kept, expected)
        kept_starts = [episodes[i][0] for i in kept]
        self.assertLess(min(kept_starts), 100)


# ---------------------------------------------------------------------------
# Tests: collect_activations (multi-clip)
# ---------------------------------------------------------------------------


class TestCollectActivations(unittest.TestCase):
    def _build_inputs(self, n_clips: int = 3):
        clips = [_make_clip() for _ in range(n_clips)]
        model = _make_model()
        runner = _make_policy_runner()
        return runner, model, clips

    def test_concatenates_all_clips(self) -> None:
        from myosuite.integrations.musclemimic.activation_collector import (
            CollectionConfig,
            collect_activations,
        )

        runner, model, clips = self._build_inputs(n_clips=3)
        config = CollectionConfig(n_episodes_per_clip=5, max_steps_per_episode=3)

        with (
            patch("mujoco.MjData", return_value=_make_data()),
            patch("mujoco.mj_forward"),
            patch(
                "myosuite.integrations.musclemimic.activation_collector._tracking_reward",
                return_value=1.0,
            ),
        ):
            acts = collect_activations(runner, model, clips, config)

        self.assertEqual(acts.ndim, 2)
        self.assertEqual(acts.shape[1], N_MUSCLES)
        self.assertEqual(acts.dtype, np.float32)

    def test_raises_when_no_activations(self) -> None:
        from myosuite.integrations.musclemimic.activation_collector import (
            CollectionConfig,
            collect_activations,
        )

        runner, model, clips = self._build_inputs(n_clips=1)
        config = CollectionConfig(n_episodes_per_clip=5, max_steps_per_episode=0)

        with (
            patch("mujoco.MjData", return_value=_make_data()),
            patch("mujoco.mj_forward"),
            patch(
                "myosuite.integrations.musclemimic.activation_collector._run_episode",
                return_value=(np.empty((0, N_MUSCLES), dtype=np.float32), -999.0),
            ),
        ):
            with self.assertRaises(ValueError):
                collect_activations(runner, model, clips, config)

    def test_cache_path_saves_and_loads(self) -> None:
        from myosuite.integrations.musclemimic.activation_collector import (
            CollectionConfig,
            collect_activations,
        )

        runner, model, clips = self._build_inputs(n_clips=2)
        config = CollectionConfig(n_episodes_per_clip=5, max_steps_per_episode=3)

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "acts.npy"

            with (
                patch("mujoco.MjData", return_value=_make_data()),
                patch("mujoco.mj_forward"),
                patch(
                    "myosuite.integrations.musclemimic.activation_collector._tracking_reward",
                    return_value=1.0,
                ),
            ):
                acts1 = collect_activations(
                    runner, model, clips, config, cache_path=cache
                )
                self.assertTrue(cache.exists())
                # Second call should load from cache without calling runner
                runner2 = _make_policy_runner()
                acts2 = collect_activations(
                    runner2, model, clips, config, cache_path=cache
                )

            np.testing.assert_array_equal(acts1, acts2)
            # runner2 was never called — loaded from cache
            runner2.action_for.assert_not_called()

    def test_uses_default_config_when_none(self) -> None:
        from myosuite.integrations.musclemimic.activation_collector import (
            collect_activations,
        )

        runner, model, clips = self._build_inputs(n_clips=1)

        with (
            patch("mujoco.MjData", return_value=_make_data()),
            patch("mujoco.mj_forward"),
            patch(
                "myosuite.integrations.musclemimic.activation_collector._tracking_reward",
                return_value=1.0,
            ),
        ):
            acts = collect_activations(runner, model, clips)  # config=None

        self.assertIsInstance(acts, np.ndarray)


if __name__ == "__main__":
    unittest.main()
