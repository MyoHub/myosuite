from typing import Callable, Any

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg

from mjlab_myosuite.trajectory_io import MotionClip


_DEFAULT_ASSET_CFG = SceneEntityCfg("human")


def body_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the asset's root height is below the minimum height."""
  asset: Entity = env.scene[asset_cfg.name]
  return (asset.data.data.xpos[:, asset.data.indexing.body_ids[asset_cfg.body_ids], 2] < minimum_height).any(dim=1)