# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The action map a training checkpoint stores, and loading its policy (needs torch)."""

from __future__ import annotations

from pathlib import Path
import yaml

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Action-map checkpoints need torch: pip install 'myosuite[mjlab]'"
    ) from e

from myosuite.methods.synergies.base import (
    TrainedActionMap,
    action_map_from_manifest,
)
from myosuite.methods.synergies.dynsyn import DynSyn
from myosuite.utils.checkpoint_manifest import ACTION_MAP_FIELD, CfgLoader
from myosuite.utils.rslrl_policy import RslRlPolicy, load_rslrl_policy

# Checkpoint ``infos`` key of DynSyn runs written before ACTION_MAP_FIELD.
_LEGACY_DYNSYN_KEY = "dynsyn"


def _from_env_yaml(checkpoint: Path, counter: int | None) -> TrainedActionMap | None:
    """The action map in the run's ``params/env.yaml`` (runs without it in ``infos``)."""
    path = checkpoint.parent / "params" / "env.yaml"
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    legacy = "DynSynActionCfg" in text
    if not legacy and "ActionMapActionCfg" not in text:
        return None
    env = yaml.load(text, Loader=CfgLoader)
    key = "start_steps" if legacy else ACTION_MAP_FIELD
    term = next(t for t in env["actions"].values() if isinstance(t, dict) and key in t)
    action_map = (
        DynSyn.from_manifest(term) if legacy else action_map_from_manifest(term[key])
    )
    if counter is None:
        print(f"{checkpoint} stores no step count: using the full schedule.")
    steps = (
        float("inf")
        if counter is None
        else counter * env["scene"]["num_envs"] + term.get("step_offset", 0.0)
    )
    return TrainedActionMap(action_map, env_steps=steps, saved_counter=counter or 0)


def trained_action_map(checkpoint: str | Path) -> TrainedActionMap | None:
    """Action map of the run that wrote a checkpoint.

    ``scripts/train_mjlab.py`` stores the map in the checkpoint's ``infos``, so a
    checkpoint read on its own (a W&B download, a moved file) still has it. Older
    checkpoints fall back to the run's ``params/env.yaml``, which holds the action
    term, and to the env's step counter in the checkpoint. DynSyn checkpoints from
    before the generic record are read too.

    Args:
        checkpoint: ``model_*.pt`` of an mjlab training run.

    Returns:
        The map, or ``None`` if the run used none (or is an older run without its
        ``params/env.yaml``).
    """
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    infos = state.get("infos") or {}
    counter = infos.get("env_state", {}).get("common_step_counter")
    if ACTION_MAP_FIELD in infos:
        return TrainedActionMap.from_infos(infos[ACTION_MAP_FIELD], counter or 0)
    if _LEGACY_DYNSYN_KEY in infos:
        legacy = infos[_LEGACY_DYNSYN_KEY]
        return TrainedActionMap(
            DynSyn.from_manifest(legacy),
            env_steps=legacy["env_steps"],
            saved_counter=counter or 0,
        )
    return _from_env_yaml(checkpoint, counter)


def load_policy(
    checkpoint: str | Path, action_dim: int
) -> tuple[RslRlPolicy, TrainedActionMap | None]:
    """The deterministic actor of a checkpoint and the action map it acts through.

    Args:
        checkpoint: ``model_*.pt`` of an mjlab training run.
        action_dim: Action dimension of the env (with the map applied) it will drive.

    Returns:
        The policy (see :func:`~myosuite.utils.rslrl_policy.load_rslrl_policy`) and the
        run's map, or ``None`` for one action per muscle. Apply the map with
        :class:`~myosuite.methods.synergies.base.ActionMapWrapper` or
        :func:`~myosuite.methods.synergies.base.apply_trained_action_map`.
    """
    trained = trained_action_map(checkpoint)
    policy = load_rslrl_policy(
        checkpoint, action_dim, action_map_applied=trained is not None
    )
    return policy, trained
