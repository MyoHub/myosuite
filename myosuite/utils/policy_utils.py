# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Policy rollout utilities for MyoSuite environments."""

from __future__ import annotations

import time as timer
from pathlib import Path
from sys import platform
from typing import Any

import numpy as np

from myosuite.utils.prompt_utils import Prompt, prompt


def examine_policy(
    env: Any,
    policy: Any,
    horizon: int = 1000,
    num_episodes: int = 1,
    mode: str = "exploration",
    render: str | None = None,
    camera_name: str | None = None,
    frame_size: tuple[int, int] = (640, 480),
    output_dir: str = "/tmp/",
    filename: str = "newvid",
    device_id: int = 0,
) -> Any:
    """Roll out a policy and return a :class:`~myosuite.logger.grouped_datasets.Trace`.

    Args:
        env: A :class:`~myosuite.envs.gymnasium_env.MyoGymnasiumEnv` instance.
        policy: Policy object exposing ``get_action(obs)``.
        horizon: Maximum steps per episode.
        num_episodes: Number of episodes to collect.
        mode: ``"exploration"`` uses stochastic actions; ``"evaluation"`` uses
            deterministic (mean) actions.
        render: ``"onscreen"`` / ``"offscreen"`` / ``None``.
        camera_name: Camera name for offscreen rendering.
        frame_size: ``(width, height)`` for offscreen frames.
        output_dir: Directory for saving MP4 files.
        filename: Base filename for saved videos.
        device_id: Unused; kept for signature compatibility.

    Returns:
        A :class:`~myosuite.logger.grouped_datasets.Trace` containing all
        rollout data grouped by episode (``Trial0``, ``Trial1``, …).
    """
    from myosuite.logger.grouped_datasets import Trace
    from myosuite.utils.video_io import write_video

    trace = Trace(env.id + "_rollouts")
    exp_t0 = timer.time()

    frames: np.ndarray | None = None
    if render == "onscreen":
        env.render_mode = "human"
    elif render == "offscreen":
        frames = np.zeros((horizon, frame_size[1], frame_size[0], 3), dtype=np.uint8)

    for ep in range(num_episodes):
        ep_t0 = timer.time()
        group_key = "Trial" + str(ep)
        trace.create_group(group_key)
        prompt(f"Episode {ep}", end=":> ", type=Prompt.INFO)

        obs, _ = env.reset()
        done = False
        t = 0
        ep_rwd = 0.0
        rwd = 0.0
        env_info: dict = {}

        while t < horizon and not done:
            act = (
                policy.get_action(obs)[0]
                if mode == "exploration"
                else policy.get_action(obs)[1]["evaluation"]
            )

            if render == "offscreen" and frames is not None:
                frame = env.render()
                if frame is not None:
                    frames[t] = frame
                prompt(str(t), end=", ", flush=True, type=Prompt.INFO)
            elif render == "onscreen":
                env.render()

            obs_next, rwd, terminated, truncated, env_info = env.step(act)
            done = terminated or truncated

            trace.append_datums(
                group_key=group_key,
                dataset_key_val=dict(
                    time=t * env.dt if hasattr(env, "dt") else t,
                    observations=obs,
                    actions=act.copy(),
                    rewards=rwd,
                    env_infos=env_info,
                    done=done,
                ),
            )

            obs = obs_next
            t += 1
            ep_rwd += rwd

        # Record terminal observation
        trace.append_datums(
            group_key=group_key,
            dataset_key_val=dict(
                time=t * env.dt if hasattr(env, "dt") else t,
                observations=obs,
                actions=np.full(env.action_space.shape, np.nan),
                rewards=rwd,
                env_infos=env_info,
                done=done,
            ),
        )

        prompt(
            f"Episode {ep}:> Finished in {(timer.time() - ep_t0):0.4f} sec. "
            f"Total rewards {ep_rwd}",
            type=Prompt.INFO,
        )

        if render == "offscreen" and frames is not None:
            file_name = str(Path(output_dir) / (filename + str(ep) + ".mp4"))
            if platform == "darwin":
                write_video(
                    file_name,
                    np.asarray(frames[:t]),
                    outputdict={"-pix_fmt": "yuv420p"},
                )
            else:
                write_video(file_name, np.asarray(frames[:t]))
            prompt("saved: " + file_name, type=Prompt.ALWAYS)

    prompt("Total time taken = %f" % (timer.time() - exp_t0), type=Prompt.INFO)
    trace.stack()
    return trace
