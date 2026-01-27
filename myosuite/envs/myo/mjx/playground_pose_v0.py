import logging
from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from mujoco_playground._src import mjx_env  # Several helper functions are only visible under _src
from myosuite.envs.myo.mjx.fatigue_jax import CumulativeFatigue
import numpy as np
from myosuite.envs.myo.mjx.mjx_base_env import MjxMyoBase


class MjxPoseEnvV0(MjxMyoBase):

    def generate_target_pose(self, rng: jp.ndarray) -> Dict[str, jp.ndarray]:
        targets = []
        for span in self._config.target_jnt_range.values():
            targets.append(jax.random.uniform(
                rng, (span[0].size,),
                minval=span[0],
                maxval=span[1]
            ))
        return jp.hstack(targets)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        qpos = jax.random.uniform(
            rng1, (self.mjx_model.nq,),
            minval=self.mjx_model.jnt_range[:, 0],
            maxval=self.mjx_model.jnt_range[:, 1]
        )
        # TODO: Velocity initialization
        qvel = jp.zeros(self.mjx_model.nv)

        target_angles = self.generate_target_pose(rng2)

        # We store the target angles in the info, can't store it as an instance variable,
        # as it has to be determined in a parallelized manner
        info = {'rng': rng,
                'target_angles': target_angles,
                'step_count': jp.array(0, dtype=jp.int32)}

        data = self._get_data(qpos, qvel)

        obs = self._get_obs(data, info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'pose_reward': zero,
            'act_reg_reward': zero,
            'bonus_reward': zero,
            'penalty_reward': zero,
            'solved_frac': zero
        }
        return State(data, obs, reward, done, metrics, info)

    def _get_obs(
            self, data: mjx.Data, info) -> jp.ndarray:
        """Observe qpos, qvel, act and qpos_err."""
        obs= jp.concatenate([
            data.qpos,
            data.qvel * self.mjx_model.opt.timestep,
            data.act,
            info['target_angles'] - data.qpos
        ])
        return {"base_obs": obs}

    def _get_rewards(self, data: mjx.Data, info: dict) -> dict:
        """We are counting on CSE to simplify the two calls to this into one"""
        pose_dist = self._pose_dist(data, info)
        act_mag = jp.linalg.norm(data.act, axis=-1)

        pose = pose_dist * -self._config.reward_config.angle_reward_weight
        act_reg = act_mag * -self._config.reward_config.ctrl_cost_weight
        bonus = (jp.where(pose_dist < self._config.reward_config.pose_thd, 1., 0.)
                 + jp.where(pose_dist < self._config.reward_config.pose_thd * 1.5, 1.,
                            0.)) * self._config.reward_config.bonus_weight
        penalty = -1. * (pose_dist > self._config.reward_config.far_th)
        return {"pose": pose, "act_reg": act_reg, "bonus": bonus, "penalty": penalty}

    def _get_reward(self, data: mjx.Data, info: dict) -> float:
        """Return a scalar value."""
        return sum(jax.tree_util.tree_leaves(self._get_rewards(data, info)))

    def _get_done(self, state: State) -> float:
        pose_dist = self._pose_dist(state.data, state.info)
        return jp.where(pose_dist > self._config.reward_config.far_th, 1., 0.)

    def _pose_dist(self, data, info):
        # TODO: confirm this gets Common Subexpression Eliminated
        pose_err = info['target_angles'] - data.qpos
        return jp.linalg.norm(pose_err, axis=-1)

    def _get_info(self, state) -> dict:
        truncation = jp.where(state.info['step_count'] >= self._config.max_episode_steps, 1. - state.done, jp.array(0.))
        step_count = jp.where(jp.logical_or(state.done, truncation), jp.array(0, dtype=jp.int32),
                              state.info['step_count'])

        # reset target angles if done or truncation
        rng, rng1 = jax.random.split(state.info['rng'])
        target_angles = jp.where(jp.logical_or(state.done, truncation), self.generate_target_pose(rng1),
                                 state.info['target_angles'])
        state.info['rng'] = rng
        state.info['target_angles'] = target_angles
        state.info['step_count'] = step_count
        return state.info

    def _get_metrics(self, state: State) -> dict:
        pose_dist = self._pose_dist(state.data, state.info)
        solved = 1. * (pose_dist < self._config.reward_config.pose_thd)
        rewards = self._get_rewards(state.data, state.info)

        return {
            "pose_reward": rewards["pose"],
            "act_reg_reward": rewards["act_reg"],
            "bonus_reward": rewards["bonus"],
            "penalty_reward": rewards["penalty"],
            "solved_frac": solved / self._config.max_episode_steps
        }
