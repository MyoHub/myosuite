#!/usr/bin/env python3
DESC = """
Vizualize model in a viewer\n
    - render forward kinematics if `qpos` is provided\n
    - simulate dynamcis if `ctrl` is provided\n
Example:\n
    - python utils/examine_sim.py --sim_path envs/arms/franka/assets/franka_reach_v0.xml --qpos "0, 0, -1, -1, 0, 0, 0, 0, 0"\n
    - python utils/examine_sim.py --sim_path envs/arms/franka/assets/franka_reach_v0.xml --ctrl "0, 0, -1, -1, 0, 0, 0, 0, 0"\n
"""

from mujoco import MjModel, MjData, mj_step, mj_forward, viewer
import click
import numpy as np

@click.command(help=DESC)
@click.option('-s', '--sim_path', type=str, help='environment to load', required= True)
@click.option('-q', '--qpos', type=str, help='joint position', default=None)
@click.option('-c', '--ctrl', type=str, help='actuator position', default=None)
@click.option('-h', '--horizon', type=int, help='time (s) to simulate', default=5)

def main(sim_path, qpos, ctrl, horizon):
    model = MjModel.from_xml_path(sim_path)
    data = MjData(model)

    viewer.launch(model, data)

    while data.time<horizon:
        if qpos is not None:
            data.qpos[:] = np.array(qpos.split(','), dtype=np.float)
            mj_forward(model, data)
            data.time += model.opt.timestep
        elif ctrl is not None:
            data.ctrl[:] = np.array(ctrl.split(','), dtype=np.float)
            mj_step(model, data)

if __name__ == '__main__':
    main()
