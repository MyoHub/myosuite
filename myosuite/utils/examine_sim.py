# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3


import click
from mujoco import MjData, MjModel, viewer
from myosuite.core.registry import make_env

DESC = """
Vizualize model in the MuJoCo "Simulate" viewer. No policy or environment logic included.\n
    - Loads an MJCF if provided in --sim_path\n
    - Otherwise you can specify a MyoSuite env via --env_name\n
Example:\n
    - python -m myosuite.utils.examine_sim --sim_path path/to/model.xml\n
    - python -m myosuite.utils.examine_sim --env_name myoElbowPose1D6MRandom-v0\n
"""


@click.command(help=DESC)
@click.option(
    "-e", "--env_name", type=str, help="MyoSuite env id to load", required=False
)
@click.option(
    "-s", "--sim_path", type=str, help="path to an MJCF model", required=False
)
def main(sim_path, env_name):
    if sim_path is None and env_name is None:
        raise click.UsageError("Provide --sim_path or --env_name")
    if sim_path is not None:
        model = MjModel.from_xml_path(sim_path)
    else:
        model = make_env(env_name).model
    data = MjData(model)

    viewer.launch(model, data)


if __name__ == "__main__":
    main()
