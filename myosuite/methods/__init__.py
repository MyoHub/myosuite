# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Learning methods: the third layer, after tasks (MyoSuite env ids) and models (myo_sim).

A method takes a task id, a model and a control step and builds its env through
``make_env(EnvConfig(env_id, ctrl_dt=...), backend=...)``. Each method imports its heavy
dependencies (torch, mjlab) only when its own package is imported.

- ``biomech``: biomechanics tools for the tasks' data (AMASS motions onto a model).
"""
