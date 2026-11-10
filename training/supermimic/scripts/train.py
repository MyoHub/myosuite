# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train SuperMimic on ``myoMimic-v0`` with the fast recipe (the KIT training clips by default).

    python training/supermimic/scripts/train.py --agent.max-hours 0.5 --agent.seed 0

Every ``--env.*`` field of ``TaskEnvCfg`` (the task's ``make_env`` options) and every ``--agent.*`` field of
``SuperMimicRunnerCfg`` can be overridden (booleans take a value, tuples Python syntax:
``--agent.wm-hidden "(1024,1024,1024)"``). Runs go to ``logs/rsl_rl/supermimic_fullbody/<date>_<run name>/``;
``--agent.resume True --agent.load-run <run>`` continues one.
"""

import tyro

from supermimic.train import TrainArgs, train

if __name__ == "__main__":
    print(train(tyro.cli(TrainArgs)))
