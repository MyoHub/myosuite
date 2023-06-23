""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Cameron Berg (cameronberg@fb.com), Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

"""
This is a job script for running SB3 on myosuite tasks.
extends https://github.com/facebookresearch/TCDM/blob/main/tcdm/rl/trainers/util.py
"""

import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from collections import deque as dq
from stable_baselines3.common.monitor import Monitor
import numpy as np
import time

class InfoCallback(BaseCallback):
    def _on_rollout_end(self) -> None:
        all_keys = {}
        for info in self.model.ep_info_buffer:
            for k, v in info.items():
                if k in ('r', 't', 'l'):
                    continue
                elif k not in all_keys:
                    all_keys[k] = []
                all_keys[k].append(v)

        for k, v in all_keys.items():
            self.model.logger.record(f'env/{k}', np.mean(v))

    def _on_step(self) -> bool:
        return True


class FallbackCheckpoint(BaseCallback):
    def __init__(self, checkpoint_freq=1, verbose=0):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq == 0 or self.n_calls <= 1:
            self.model.save('restore_checkpoint')
        return True

class SaveSuccesses(BaseCallback):
    """
    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, timesteps: int, check_freq: int, log_dir: str, env_name: str, verbose: int = 1):
        super(SaveSuccesses, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = log_dir
        self.success_buffer = dq(maxlen=200)
        self.success_results = np.zeros(timesteps)
        self.env_name = env_name
        
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.success_buffer.append(self.locals['infos'][0]['solved'])
        if len(self.success_buffer) > 0:
            self.success_results[self.n_calls-1] = np.mean(self.success_buffer)
            success = f'./successes_{self.env_name}'
            if os.path.isfile(success):
                os.remove(success)
            np.save(success, self.success_results)
        return True


class EvalCallback(BaseCallback):
    def __init__(self, eval_freq, eval_env, verbose=0, n_eval_episodes=25):
        super().__init__(verbose)
        self._vid_log_dir = os.path.join(os.getcwd(), 'eval_videos/')
        if not os.path.exists(self._vid_log_dir):
            os.makedirs(self._vid_log_dir)
        self._eval_freq = eval_freq
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes

    def _info_callback(self, locals, _):
        if locals['done']:
            for k, v in locals['info'].items():
                if isinstance(v, (float, int)):
                    if k not in self._info_tracker:
                        self._info_tracker[k] = []
                    self._info_tracker[k].append(v)

    def _on_step(self, fps=25) -> bool:
        if self.n_calls % self._eval_freq == 0 or self.n_calls <= 1:
            self._info_tracker = dict(rollout_video=[])
            start_time = time.time()
            episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self._eval_env,
                    n_eval_episodes=self._n_eval_episodes,
                    render=False,
                    deterministic=False,
                    return_episode_rewards=True,
                    warn=True,
                    callback=self._info_callback,
                )
            end_time = time.time()

            mean_reward, mean_length = np.mean(episode_rewards), np.mean(episode_lengths)
            self.logger.record('eval/time', end_time - start_time)
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/mean_length', mean_length)
            for k, v in self._info_tracker.items():
                if k == 'rollout_video':
                    pass
                else:
                    self.logger.record('eval/mean_{}'.format(k), np.mean(v))
            self.logger.dump(self.num_timesteps)
        return True
