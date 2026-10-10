# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TERRA-4B policy inference in NumPy (no TERRA, MuscleMimic or JAX runtime).

The checkpoint is a MuscleMimic PPO checkpoint: a gated residual SiLU MLP with
layer norm behind a frozen running mean/std normalizer, and a diagonal Gaussian
with a fixed ``log_std``. Actions are muscle controls in ``[-1, 1]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.integrations.musclemimic.fullbody_local_policy import _actor_forward
from myosuite.integrations.terra.observation import TerraObsCfg, TerraObservation
from myosuite.integrations.terra.reference import ReferenceMotion

TERRA_REPO = "merc-s/TERRA-4B"
# Tested release revision of the TERRA-4B model card (training seed 0, update 24416).
TERRA_REVISION = "b89a604d0687549f2678bb1b6a436dea058cd8b9"
TERRA_CHECKPOINT = "checkpoint_24416"


def download_terra_checkpoint(cache_dir: str | Path | None = None) -> Path:
    """Download TERRA-4B at the pinned revision; returns the ``checkpoint_*`` folder."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    root = Path(
        snapshot_download(TERRA_REPO, revision=TERRA_REVISION, cache_dir=cache_dir)
    )
    return root / TERRA_CHECKPOINT if (root / TERRA_CHECKPOINT).is_dir() else root


@dataclass(frozen=True)
class TerraPolicy:
    """Frozen TERRA actor.

    Attributes:
        params: Actor parameters (``params["actor"]``, ``params["log_std"]``).
        obs_mean: Normalizer mean.
        obs_var: Normalizer variance.
        obs_cfg: Observation settings saved with the checkpoint.
    """

    params: dict[str, Any]
    obs_mean: np.ndarray
    obs_var: np.ndarray
    obs_cfg: TerraObsCfg

    @property
    def obs_dim(self) -> int:
        """Observation size the network expects."""
        return int(self.obs_mean.shape[-1])

    @property
    def action_dim(self) -> int:
        """Number of muscle controls."""
        return int(np.asarray(self.params["log_std"]).shape[-1])

    @classmethod
    def load(cls, checkpoint: str | Path, seed: int = 0) -> TerraPolicy:
        """Load a ``checkpoint_*`` folder (``train_state`` Orbax item, ``config`` JSON).

        Needs ``orbax-checkpoint`` (and therefore JAX) to read the Orbax item.
        """
        import orbax.checkpoint as ocp  # noqa: PLC0415

        root = Path(checkpoint)
        config = json.loads((root / "config" / "metadata").read_text(encoding="utf-8"))
        state = ocp.StandardCheckpointer().restore(
            str((root / "train_state").absolute())
        )
        params = _to_numpy(state["params"])
        stats = _to_numpy(state["run_stats"]["RunningMeanStd_0"])
        mean, var = stats["mean"], stats["var"]
        if mean.ndim == 2:  # leading training-seed axis
            params = _index_tree(params, seed)
            mean, var = mean[seed], var[seed]
        return cls(params, mean, var, TerraObsCfg.from_config(config))

    def act(
        self, obs: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Muscle controls for *obs*: the mean, or a sample when *rng* is given."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"TERRA expects {self.obs_dim} observations, got {obs.shape}"
            )
        norm = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        action = _actor_forward(self.params, norm.astype(np.float32))
        if rng is not None:
            std = np.exp(np.asarray(self.params["log_std"], dtype=np.float32))
            action = action + std * rng.standard_normal(action.shape).astype(np.float32)
        return np.clip(action, -1.0, 1.0)


def _to_numpy(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _to_numpy(v) for k, v in tree.items()}
    return np.asarray(tree, dtype=np.float32)


def _index_tree(tree: Any, index: int) -> Any:
    if isinstance(tree, dict):
        return {k: _index_tree(v, index) for k, v in tree.items()}
    return tree[index]


class TerraController:
    """TERRA acting in an env that steps ``frame_skip`` substeps per action.

    TERRA observed the data right after ``mj_step`` (contacts, touch and site
    positions one substep behind ``qpos``), while MyoSuite envs refresh them with
    ``mj_forward``. The controller replays each step on a shadow ``MjData`` to give
    the policy those exact inputs.

    Args:
        policy: The TERRA actor.
        model: The env's model (TERRA actor plus scene).
        reference: Reference motion at the control rate.
        frame_skip: Physics substeps per control step of the env.
        rng: Sample actions with this generator (``None``: mean actions).
    """

    def __init__(
        self,
        policy: TerraPolicy,
        model: mujoco.MjModel,
        reference: ReferenceMotion,
        frame_skip: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.policy, self.reference, self.rng = policy, reference, rng
        self.observe = TerraObservation(model, policy.obs_cfg)
        self._model, self._frame_skip = model, frame_skip
        self._shadow = mujoco.MjData(model)
        self._frame = 0

    def reset(self) -> None:
        """Start a new episode at reference frame 0."""
        self._frame = 0

    def __call__(self, data: mujoco.MjData) -> np.ndarray:
        """Muscle controls for the env state *data*; call once per env step."""
        if self._frame == 0:
            # TERRA's reset: forward at qpos0, then at the initial state (solver warm start).
            source = self._shadow
            mujoco.mj_resetData(self._model, source)
            mujoco.mj_forward(self._model, source)
            source.qpos[:], source.qvel[:], source.act[:] = (
                data.qpos,
                data.qvel,
                data.act,
            )
            mujoco.mj_forward(self._model, source)
        else:
            mujoco.mj_step(self._model, self._shadow, self._frame_skip)
            source = self._shadow
        action = self.policy.act(
            self.observe(source, self.reference, self._frame), self.rng
        )
        mujoco.mj_copyData(self._shadow, self._model, data)
        lo, hi = self._model.actuator_ctrlrange.T
        self._shadow.ctrl[:] = np.clip(action, lo, hi)
        self._frame += 1
        return action
