# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Muscle synergies: opt-in action maps ``a_muscles = M(z, state)`` for suitable methods.

A synergy method changes the action space: the policy commands ``z`` and an
:class:`~.base.ActionMap` turns it into one action per muscle. The map is a method
option, not a task or model property; checkpoints record it, and plain loaders
(:func:`myosuite.utils.rslrl_policy.load_rslrl_policy`) refuse them.

- :mod:`.base`: the :class:`~.base.ActionMap` protocol, the CPU wrapper
  (:class:`~.base.ActionMapWrapper`) and the mjlab action term (needs ``myosuite[mjlab]``).
- :mod:`.dynsyn`: DynSyn (He et al., ICML 2024, arXiv:2407.11472), one action per muscle
  group plus scheduled corrections. Supported by PPO through
  ``scripts/train_mjlab.py --dynsyn-groups`` and ``scripts/eval_mjlab_policy.py``.
- :mod:`.checkpoint`: the map a checkpoint stores, and :func:`~.checkpoint.load_policy`
  (needs torch).
- Planned: ``sar.py`` for SAR (Berg et al., RSS 2023), from the extraction in
  :mod:`myosuite.integrations.musclemimic.sar_extraction`.
"""

from myosuite.methods.synergies.base import (
    ActionMap,
    ActionMapWrapper,
    TrainedActionMap,
    action_map_from_manifest,
    register_action_map,
)
from myosuite.methods.synergies.dynsyn import DynSyn, DynSynGroups

__all__ = [
    "ActionMap",
    "ActionMapWrapper",
    "DynSyn",
    "DynSynGroups",
    "TrainedActionMap",
    "action_map_from_manifest",
    "register_action_map",
]
