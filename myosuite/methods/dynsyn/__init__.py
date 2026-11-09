# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""DynSyn muscle synergies (He et al., ICML 2024, arXiv:2407.11472) as an opt-in method option.

The policy commands one action per muscle group (plus optional bounded per-muscle
corrections) instead of one action per muscle. Supported by PPO through
``scripts/train_mjlab.py --dynsyn-groups`` and ``scripts/eval_mjlab_policy.py``.

- :mod:`.groups`: group computation and the action mapping (numpy / torch).
- :mod:`.cpu_wrapper`: :class:`DynSynWrapper` for CPU envs.
- :mod:`.mjlab_action`: the mjlab action term (needs ``myosuite[mjlab]``).
- :mod:`.checkpoint`: the setup stored in a training checkpoint (needs torch).
"""

from myosuite.methods.dynsyn.cpu_wrapper import DynSynWrapper
from myosuite.methods.dynsyn.groups import (
    CHECKPOINT_KEY,
    DynSynGroups,
    TrainedDynSyn,
    expand_actions,
    synergy_groups,
    weight_amplitude,
)

__all__ = [
    "CHECKPOINT_KEY",
    "DynSynGroups",
    "DynSynWrapper",
    "TrainedDynSyn",
    "expand_actions",
    "synergy_groups",
    "weight_amplitude",
]
