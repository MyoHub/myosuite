# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Stable-Baselines3 training callbacks."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from myosuite.utils.export_onnx import export_sb3_to_onnx
from myosuite.utils.onnx_checkpoint import (
    _FATIGUE_STATE_KEY,
    bundle_onnx_with_checkpoint,
    get_env_fatigue_state,
)


class OnnxCheckpointCallback(BaseCallback):
    """Save SB3 checkpoints as ONNX bundles at a fixed timestep interval.

    Args:
        checkpoint_dir: Directory to write ``.onnx`` bundle files into.
        task_id: Environment ID stored in the bundle metadata.
        obs_dim: Observation vector dimension.
        act_dim: Action vector dimension.
        save_freq: Interval in total environment timesteps between saves.
    """

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        task_id: str,
        obs_dim: int,
        act_dim: int,
        save_freq: int,
        env: Any | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.task_id = task_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.save_freq = max(1, save_freq)
        self._fatigue_env = env
        self._last_save_timestep = 0

    def _save_bundle(self, stem: str) -> None:
        assert self.model is not None
        with tempfile.TemporaryDirectory(prefix="saber-cpu-onnx-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            native_ckpt = tmp_root / f"{stem}.zip"
            onnx_path = tmp_root / f"{stem}.onnx"
            self.model.save(native_ckpt)
            export_sb3_to_onnx(
                checkpoint=native_ckpt,
                output=onnx_path,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
            )
            metadata: dict[str, Any] = {
                "task_id": self.task_id,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "num_timesteps": int(self.model.num_timesteps),
            }
            env = self._fatigue_env or getattr(self.model, "env", None)
            fatigue_state = get_env_fatigue_state(env) if env is not None else None
            if fatigue_state is not None:
                metadata[_FATIGUE_STATE_KEY] = fatigue_state
            bundle_onnx_with_checkpoint(
                onnx_path=onnx_path,
                checkpoint_path=native_ckpt,
                framework="sb3-ppo",
                metadata=metadata,
                output_path=self.checkpoint_dir / f"{stem}.onnx",
            )

    def _on_step(self) -> bool:
        # Compare against num_timesteps (not n_calls) so save_freq is in terms
        # of total environment timesteps regardless of n_envs.
        if self.model.num_timesteps - self._last_save_timestep >= self.save_freq:
            self._save_bundle(f"model_{self.model.num_timesteps}")
            self._last_save_timestep = self.model.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self._save_bundle("model_final")
