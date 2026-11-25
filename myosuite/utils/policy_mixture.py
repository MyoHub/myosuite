# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A uniform mixture of policy snapshots: each episode is run by one snapshot, drawn at
its start.

A reward that depends on the policy's own distribution of a feature (estimated online
from its samples) makes the policy and that estimate follow each other, so the last
policy is one point of an oscillation. The mixture of the policies of training is the
readout that matches the target (historical averaging; Lee et al. 2019,
arXiv:1906.05274). This module takes the snapshots from the last part of training.

``scripts/eval_mjlab_policy.py --mixture`` evaluates RSL-RL snapshots. SB3 snapshots
(``train_sb3.py --snapshot-freq``) are loaded with
:func:`myosuite.utils.checkpoint_utils.load_policy` and run as
``PolicyMixture(policies)(obs)``, with ``mixture.redraw()`` at the end of each episode,
which draws the snapshot of the next one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from myosuite.utils.checkpoint_utils import checkpoint_step

_NAMES = {
    ".pt": re.compile(r"model_\d+"),
    ".zip": re.compile(r".+_(\d+)_steps"),
}  # RSL-RL, SB3


def snapshot_step(path: Path) -> int | None:
    """Training step in a snapshot's name: ``model_400.pt`` (RSL-RL) or
    ``<name>_4000_steps.zip`` (SB3)."""
    name = _NAMES.get(path.suffix)
    match = name.fullmatch(path.stem) if name is not None else None
    if match is None:
        return None
    return int(match[1]) if path.suffix == ".zip" else checkpoint_step(path)


def select_snapshots(
    run_dir: str | Path, count: int, last: float = 0.5, until: int | None = None
) -> list[Path]:
    """*count* snapshots spread evenly over the last part of a run, the newest included.

    Args:
        run_dir: Directory of ``model_<iteration>.pt`` or ``<name>_<steps>_steps.zip``
            files.
        count: Number of snapshots.
        last: Share of training they are taken from, in ``(0, 1]``; 1 includes step 0.
        until: The newest step taken (a given checkpoint's); ``None`` takes the run's
            newest.

    Returns:
        The snapshots, oldest first.

    Raises:
        ValueError: *last* is outside ``(0, 1]``, or that part of the run holds fewer
            than *count* snapshots.
    """
    if not 0.0 < last <= 1.0:
        raise ValueError(f"last must lie in (0, 1], got {last}")
    steps = {p: snapshot_step(p) for p in Path(run_dir).iterdir() if p.is_file()}
    found = sorted(
        (s, p)
        for p, s in steps.items()
        if s is not None and (until is None or s <= until)
    )
    if not found:
        raise ValueError(f"No snapshots in {run_dir}")
    first = (1.0 - last) * found[-1][0]
    late = [p for s, p in found if s > first or last == 1.0]
    if len(late) < count:
        raise ValueError(
            f"{run_dir} has {len(late)} snapshots in the last {last:.0%} of training, "
            f"{count} were asked for; "
            "save more often or lower the count"
        )
    picks = np.linspace(0, len(late) - 1, count).round().astype(int)
    return [late[i] for i in picks]


class PolicyMixture:
    """One of several policies per env, drawn uniformly and redrawn when the env's
    episode ends.

    Args:
        policies: The snapshots.
        n_envs: Number of parallel envs.
        seed: Seed of the draws.
    """

    def __init__(self, policies: Sequence[Any], n_envs: int = 1, seed: int = 0) -> None:
        if not policies:
            raise ValueError("a mixture needs at least one policy")
        self.policies = list(policies)
        self.rng = np.random.default_rng(seed)
        self.index = self.rng.integers(len(self.policies), size=n_envs)

    def redraw(self, env_ids: Sequence[int] | None = None) -> None:
        """Draw a new policy for the envs *env_ids* (all by default); call it when their
        episodes end."""
        ids = (
            np.arange(len(self.index))
            if env_ids is None
            else np.asarray(env_ids, dtype=int)
        )
        self.index[ids] = self.rng.integers(len(self.policies), size=len(ids))

    def apply(self, call: Callable[[Any, Any], Any], obs: Any) -> Any:
        """Actions of all envs: ``call(policy, obs[rows])`` for each policy and the envs
        that hold it.

        Args:
            call: ``(policy, observations) -> actions``.
            obs: Observations, ``(n_envs, ...)``, a numpy array or a torch tensor.

        Returns:
            Actions, ``(n_envs, ...)``, in the array type *call* returns.
        """
        held = np.unique(self.index)
        if len(held) == 1:  # one policy holds every env: no gather and scatter
            return call(self.policies[held[0]], obs)
        out = None
        for k in held:
            rows = self.index == k
            if not isinstance(obs, np.ndarray):
                import torch

                rows = torch.as_tensor(rows, device=obs.device)
            part = call(self.policies[k], obs[rows])
            if out is None:
                shape = (len(self.index), *part.shape[1:])
                out = (
                    np.empty(shape, part.dtype)
                    if isinstance(part, np.ndarray)
                    else part.new_empty(shape)
                )
            if isinstance(part, np.ndarray) and not isinstance(rows, np.ndarray):
                rows = self.index == k
            out[rows] = part
        return out

    def __call__(self, obs: Any) -> Any:
        """Action of a single env's current policy, for policies that are
        ``act(obs) -> action``.

        The policy changes only at :meth:`redraw`: call it when the episode ends.

        Raises:
            ValueError: The mixture serves several envs; use :meth:`apply`.
        """
        if len(self.index) != 1:
            raise ValueError(
                f"the mixture serves {len(self.index)} envs: "
                "call apply(), which runs each env's own policy"
            )
        return self.policies[self.index[0]](obs)
