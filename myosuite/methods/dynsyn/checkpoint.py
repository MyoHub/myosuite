# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The DynSyn setup a training checkpoint stores (needs torch)."""

from __future__ import annotations

from pathlib import Path

import yaml

try:
    import torch
except ImportError as e:
    raise ImportError(
        "DynSyn checkpoints need torch: pip install 'myosuite[mjlab]'"
    ) from e

from myosuite.methods.dynsyn.groups import CHECKPOINT_KEY, DynSynGroups, TrainedDynSyn
from myosuite.utils.rslrl_policy import CfgLoader


def trained_dynsyn(checkpoint: str | Path) -> TrainedDynSyn | None:
    """DynSyn setup of the run that wrote a checkpoint.

    ``scripts/train_mjlab.py`` stores the setup in the checkpoint's ``infos``, so a
    checkpoint read on its own (a W&B download, a moved file) still has it. Older
    checkpoints fall back to the run's ``params/env.yaml``, which holds the action
    term, and to the env's step counter in the checkpoint.

    Args:
        checkpoint: ``model_*.pt`` of an mjlab training run.

    Returns:
        The setup, or ``None`` if the run did not use DynSyn (or is an older run
        without its ``params/env.yaml``).
    """
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    infos = state.get("infos") or {}
    counter = infos.get("env_state", {}).get("common_step_counter")
    if CHECKPOINT_KEY in infos:
        return TrainedDynSyn.from_infos(infos[CHECKPOINT_KEY], counter or 0)
    path = checkpoint.parent / "params" / "env.yaml"
    if not path.is_file() or "DynSynActionCfg" not in path.read_text(errors="replace"):
        return None
    with path.open() as f:
        env = yaml.load(f, Loader=CfgLoader)
    term = next(
        t for t in env["actions"].values() if isinstance(t, dict) and "start_steps" in t
    )
    if counter is None:
        print(f"{checkpoint} stores no step count: using the full DynSyn bound.")
    steps = (
        float("inf")
        if counter is None
        else counter * env["scene"]["num_envs"] + term.get("step_offset", 0.0)
    )
    return TrainedDynSyn(
        groups=DynSynGroups(**term["groups"]),
        start_steps=term["start_steps"],
        ramp_steps=term["ramp_steps"],
        max_amplitude=term["max_amplitude"],
        env_steps=steps,
        saved_counter=counter or 0,
    )
