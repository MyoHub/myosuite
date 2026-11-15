# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Roll out TERRA's own MyoFullBody env (TerraGoal, TERRA-4B settings) on a reference.

Runs in the upstream TERRA / MuscleMimic environment, not in MyoSuite's.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
from loco_mujoco.trajectory.dataclasses import (
    Trajectory,
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
)
from omegaconf import OmegaConf
from terra.rl import register_components

STEPS = 15


def make_env():
    register_components()
    from terra.rl.environment import MyoFullBody

    import terra

    config = Path(terra.__file__).parent / "rl" / "configs" / "ppo_multi_motion.yaml"
    params = OmegaConf.to_container(
        OmegaConf.load(config).experiment.env_params, resolve=False
    )
    for key in [k for k in params if k.startswith("mjx_")]:
        params.pop(key)
    for key in (
        "num_envs",
        "nconmax",
        "njmax",
        "env_name",
        "headless",
        "domain_randomization_type",
        "domain_randomization_params",
    ):
        params.pop(key, None)
    params["init_state_type"], params["init_state_params"] = (
        "TrajInitialStateHandler",
        {},
    )
    # Native-MuJoCo evaluation physics (scripts/terra/play_policy.py).
    params["model_option_conf"] = {
        "iterations": 4,
        "ls_iterations": 8,
        "disableflags": int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
    }
    params["th_params"] = {"random_start": False, "fixed_start_conf": [0, 0]}
    for key in (
        "mean_site_deviation_threshold",
        "core_upper_body_mean_site_deviation_threshold",
        "curriculum_initial_global_threshold",
        "core_upper_body_curriculum_initial_threshold",
        "root_orientation_threshold",
    ):
        params["terminal_state_params"][key] = 10.0  # never terminate the short check
    return MyoFullBody(headless=True, **params)


def load_reference(env, path: str) -> np.ndarray:
    m = env._model
    ref = np.load(path)
    assert [m.joint(i).name for i in range(m.njnt)] == list(ref["joint_names"])
    assert [m.actuator(i).name for i in range(m.nu)] == list(ref["actuator_names"])
    qpos = ref["ref_qpos"]
    sites = list(env.sites_for_mimic)
    ids = np.array([m.site(s).id for s in sites])
    data, xpos, xmat = mujoco.MjData(m), [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(m, data)
        xpos.append(data.site_xpos[ids].copy())
        xmat.append(data.site_xmat[ids].copy())
    model = TrajectoryModel(
        njnt=m.njnt,
        jnt_type=jnp.array(m.jnt_type),
        nbody=m.nbody,
        body_rootid=jnp.array(m.body_rootid),
        body_weldid=jnp.array(m.body_weldid),
        body_mocapid=jnp.array(m.body_mocapid),
        body_pos=jnp.array(m.body_pos),
        body_quat=jnp.array(m.body_quat),
        body_ipos=jnp.array(m.body_ipos),
        body_iquat=jnp.array(m.body_iquat),
        nsite=len(ids),
        site_bodyid=jnp.array(m.site_bodyid)[ids],
        site_pos=jnp.array(m.site_pos)[ids],
        site_quat=jnp.array(m.site_quat)[ids],
    )
    info = TrajectoryInfo(
        list(ref["joint_names"]), model=model, frequency=100.0, site_names=sites
    )
    traj = TrajectoryData(
        jnp.array(qpos),
        jnp.zeros((len(qpos), m.nv)),
        site_xpos=jnp.array(np.array(xpos)),
        site_xmat=jnp.array(np.array(xmat)),
        split_points=jnp.array([0, len(qpos)]),
    )
    env.load_trajectory(Trajectory(info=info, data=traj), warn=False)
    return qpos


def main(ref_path: str, out: str) -> None:
    env = make_env()
    ref_qpos = load_reference(env, ref_path)
    actions = np.random.default_rng(0).uniform(-1, 1, (STEPS, env._model.nu))
    obs, states = [env.reset()], []
    for action in actions:
        states.append(env._data.qpos.copy())
        obs.append(env.step(action)[0])
    np.savez_compressed(
        out,
        ref_qpos=ref_qpos,
        actions=actions.astype(np.float32),
        obs=np.asarray(obs[:-1], dtype=np.float32),
        qpos=np.asarray(states),
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
