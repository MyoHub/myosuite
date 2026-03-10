from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from myosuite.envs.myo.mjx.mjx_base_env import MjxMyoBase


class MjxReachEnvV0(MjxMyoBase):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._tip_sids = []
        self._target_sids = []
        for site in self._config.target_reach_range.keys():
            self._tip_sids.append(
                mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site)
            )
            self._target_sids.append(
                mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + "_target"
                )
            )
        self._tip_sids = jp.array(self._tip_sids)
        self._target_sids = self._target_sids

    def generate_target_pose(self, rng: jp.ndarray) -> Dict[str, jp.ndarray]:
        targets = []
        for span in self._config.target_reach_range.values():
            targets.append(
                jax.random.uniform(rng, (span[0].size,), minval=span[0], maxval=span[1])
            )
        return jp.stack(targets)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        qpos = jp.array(self._mj_model.qpos0)
        qvel = jp.zeros(self.mjx_model.nv)

        targets = self.generate_target_pose(rng2)
        self.n_targets = len(targets)
        self.near_th = self.n_targets * .0125
        
        # We store the targets in the info, can't store it as an instance variable,
        # as it has to be determined in a parallelized manner
        info = {'rng': rng,
                'targets': targets,
                'step_count': jp.array(0, dtype=jp.int32)}

        data = self._get_data(qpos, qvel)
        obs = self._get_obs(data, info)
        
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reach_reward": zero,
            "bonus_reward": zero,
            "penalty_reward": zero,
            "solved_frac": zero,
        }
        return State(data, obs, reward, done, metrics, info)

    def _get_rewards(self, data, info):
        reach_err = self._reach_err(data, info)
        reach_dist = jp.linalg.norm(reach_err, axis=-1)
        
        far_th = jp.where(
            data.time > 2.0 * self.mjx_model.opt.timestep,
            self._config.far_th * self.n_targets,
            jp.inf,
        )

        reach = -1.0 * reach_dist * self._config.reward_config.reach_weight
        bonus = (
            1.0 * (reach_dist < 2 * self.near_th) + 1.0 * (reach_dist < self.near_th)
        ) * self._config.reward_config.bonus_scale
        penalty = -1.0 * (reach_dist > far_th) * self._config.reward_config.penalty_scale
        
        return {"reach": reach, "bonus": bonus, "penalty": penalty}
    
    def _get_done(self, state: State) -> float:
        reach_err = self._reach_err(state.data, state.info)
        reach_dist = jp.linalg.norm(reach_err, axis=-1)
        far_th = jp.where(
            state.data.time > 2.0 * self.mjx_model.opt.timestep,
            self._config.far_th * self.n_targets,
            jp.inf,
        )
        done = 1.0 * (reach_dist > far_th)
        
        return done
    
    def _get_metrics(self, state: State) -> dict:
        reach_err = self._reach_err(state.data, state.info)
        reach_dist = jp.linalg.norm(reach_err, axis=-1)
        solved = 1.0 * (reach_dist < self.near_th)
        rewards = self._get_rewards(state.data, state.info)

        return {
            "reach_reward": rewards["reach"],
            "bonus_reward": rewards["bonus"],
            "penalty_reward": rewards["penalty"],
            "solved_frac": solved / self._config.max_episode_steps
        }
    
    def _get_info(self, state: State) -> dict:
        done = state.done

        # reset step counter if done or truncation
        truncation = jp.where(
            state.info["step_count"] >= self._config.max_episode_steps,
            1.0 - done,
            jp.array(0.0),
        )
        step_count = jp.where(
            jp.logical_or(done, truncation),
            jp.array(0, dtype=jp.int32),
            state.info["step_count"],
        )

        # reset targets if done or truncation
        rng, rng1 = jax.random.split(state.info["rng"])
        targets = jp.where(
            jp.logical_or(done, truncation),
            self.generate_target_pose(rng1),
            state.info["targets"],
        )
        
        info={
                **state.info,
                "rng": rng,
                "step_count": step_count,
                "targets": targets,
            }
        
        return info
    
    def _reach_err(self, data, info):
        tip_pos = data.site_xpos[self._tip_sids]
        reach_err = (info['targets'] - tip_pos).ravel()
        return reach_err
      
    def _get_obs(self, data: mjx.Data, info: Dict) -> jp.ndarray:
        """Observe qpos, qvel, act, tip_pos and reach_err."""
        tip_pos = data.site_xpos[self._tip_sids]
        reach_err = (info["targets"] - tip_pos).ravel()
        obs = jp.concatenate(
            [
                data.qpos,
                data.qvel * self.mjx_model.opt.timestep,
                data.act,
                tip_pos.ravel(),
                reach_err,
            ]
        )
        return {"state": obs}
