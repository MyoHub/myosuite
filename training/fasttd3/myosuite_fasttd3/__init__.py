# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""FastTD3 trainer for MyoSuite. Depends on younggyoseo/FastTD3 (MIT)."""

from myosuite_fasttd3.env import FastTD3GymEnv, FastTD3MjlabEnv, make_fast_td3_env
from myosuite_fasttd3.train import TrainArgs, train

__all__ = [
    "FastTD3GymEnv",
    "FastTD3MjlabEnv",
    "TrainArgs",
    "make_fast_td3_env",
    "train",
]
