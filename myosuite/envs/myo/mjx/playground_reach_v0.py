from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from mujoco_playground._src import mjx_env  # Several helper functions are only visible under _src
import numpy as np

class MjxReachEnvV0(mjx_env.MjxEnv):
    def __init__(
            self,
            config: config_dict.ConfigDict,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        spec = mujoco.MjSpec.from_file(config.model_path.as_posix())
        spec = self.preprocess_spec(spec)
        self._mj_model = spec.compile()

        self._mj_model.geom_margin = np.zeros(self._mj_model.geom_margin.shape)
        print(f"All margins set to 0")

        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.disableflags = self._mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = config.model_path.as_posix()

        self._tip_sids = []
        for site in self._config.target_reach_range.keys():
            self._tip_sids.append(mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site))
        self._tip_sids = jp.array(self._tip_sids)

    def preprocess_spec(self, spec:mujoco.MjSpec):
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.conaffinity = 0
                geom.contype = 0
                print(f"Disabled contacts for cylinder geom named \"{geom.name}\"")
        return spec
    
    def generate_target_pose(self, rng: jp.ndarray) -> Dict[str, jp.ndarray]:
        targets = []
        for span in self._config.target_reach_range.values():
            targets.append(jax.random.uniform(
                rng, (span[0].size,),
                minval=span[0],
                maxval=span[1]
            ))
        return jp.stack(targets)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        qpos = jax.random.uniform(
            rng1, (self.mjx_model.nq,),
            minval=self.mjx_model.jnt_range[:,0],
            maxval=self.mjx_model.jnt_range[:,1]
        )
        # TODO: Velocity initialization
        qvel = jp.zeros(self.mjx_model.nv)

        targets = self.generate_target_pose(rng2)
        self.n_targets = len(targets)
        self.near_th = self.n_targets*.0125

        # We store the targets in the info, can't store it as an instance variable,
        # as it has to be determined in a parallelized manner
        info = {'rng': rng, 'targets': targets}

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=jp.zeros((self.mjx_model.nu,)))

        obs, _ = self._get_obs(data, info)

        reward, done, zero = jp.zeros(3)

        metrics = {
            'reach_reward': zero,
            'bonus_reward': zero,
            'penalty_reward': zero,
        }
        return State(data, {"state": obs}, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        
        norm_action = 1.0/(1.0+jp.exp(-5.0*(action-0.5))) 

        data = mjx_env.step(self.mjx_model, state.data, norm_action)

        obs, reach_err = self._get_obs(data, state.info)
                
        reach_dist = jp.linalg.norm(reach_err, axis=-1)

        far_th = jp.where(data.time>2.*self.mjx_model.opt.timestep, self._config.far_th*self.n_targets, jp.inf)
        
        reach = -1.*reach_dist * self._config.reward_weights.reach
        bonus = (1.*(reach_dist<2*self.near_th) + 1.*(reach_dist<self.near_th)) * self._config.reward_weights.bonus
        penalty = -1.*(reach_dist>far_th) * self._config.reward_weights.penalty 

        reward = reach + bonus + penalty
        done = -1.*(reach_dist > far_th)

        state.metrics.update(
            reach_reward=reach,
            bonus_reward=bonus,
            penalty_reward=penalty,
        )

        return state.replace(
            data=data, obs={"state": obs}, reward=reward, done=done
        )

    def _get_obs(
            self, data: mjx.Data, info: Dict
    ) -> jp.ndarray:
        """Observe qpos, qvel, act, tip_pos and reach_err."""
        tip_pos = data.site_xpos[self._tip_sids]
        reach_err = (info['targets']-tip_pos).ravel()
        obs = jp.concatenate([
            data.qpos,
            data.qvel*self.mjx_model.opt.timestep,
            data.act,
            tip_pos.ravel(),
            reach_err
        ])
        return obs, reach_err

    # Accessors.
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
