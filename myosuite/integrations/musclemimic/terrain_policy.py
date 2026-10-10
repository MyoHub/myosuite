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

from pathlib import Path

import mujoco
import numpy as np

from myosuite.integrations.musclemimic.fullbody_local_policy import (
    LocalPolicyRunner,
    load_local_policy_artifacts,
    read_checkpoint_config_metadata,
)
from myosuite.integrations.musclemimic.terrain_observation import (
    TerrainObsCfg,
    TerrainObservation,
)
from myosuite.core.trajectory_io import MotionClip

TERRA_REPO = "merc-s/TERRA-4B"
# Tested release revision of the TERRA-4B model card (training seed 0, update 24416).
TERRA_REVISION = "b89a604d0687549f2678bb1b6a436dea058cd8b9"
TERRA_CHECKPOINT = "checkpoint_24416"


def download_terrain_checkpoint(cache_dir: str | Path | None = None) -> Path:
    """Download TERRA-4B at the pinned revision; returns the ``checkpoint_*`` folder."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    root = Path(
        snapshot_download(
            TERRA_REPO, revision=TERRA_REVISION, cache_dir=cache_dir, token=False
        )
    )
    return root / TERRA_CHECKPOINT if (root / TERRA_CHECKPOINT).is_dir() else root


def load_terrain_policy(
    checkpoint: str | Path,
    model: mujoco.MjModel,
    reference: MotionClip,
    *,
    seed: int = 0,
    stochastic: bool = False,
) -> LocalPolicyRunner:
    """Load TERRA through the MuscleMimic runner with its saved observation layout."""
    root = Path(checkpoint)
    cfg = TerrainObsCfg.from_config(read_checkpoint_config_metadata(root))
    return LocalPolicyRunner(
        load_local_policy_artifacts(root),
        stochastic=stochastic,
        seed=seed,
        obs_adapter=TerrainObservation(model, cfg, reference),
        update_normalizer=False,
    )


class TerrainController:
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
    """

    def __init__(
        self,
        policy: LocalPolicyRunner,
        model: mujoco.MjModel,
        reference: MotionClip,
        frame_skip: int,
    ) -> None:
        self.policy, self.reference = policy, reference
        self.observe = policy.obs_adapter
        self._model, self._frame_skip = model, frame_skip
        self._shadow = mujoco.MjData(model)
        self._frame = 0

    def reset(self) -> None:
        """Start a new episode at reference frame 0."""
        self._frame = 0
        self.policy.reset()

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
        if not np.allclose(
            source.qpos, data.qpos, atol=1e-7, rtol=0
        ) or not np.allclose(source.qvel, data.qvel, atol=1e-7, rtol=0):
            mujoco.mj_copyData(source, self._model, data)
        action = self.policy.action_for(source, self.reference, self._frame)
        mujoco.mj_copyData(self._shadow, self._model, data)
        lo, hi = self._model.actuator_ctrlrange.T
        self._shadow.ctrl[:] = np.clip(action, lo, hi)
        self._frame += 1
        return action
