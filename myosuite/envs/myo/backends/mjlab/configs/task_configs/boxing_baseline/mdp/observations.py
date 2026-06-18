import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("human")


def raw_activation(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.data.act[:, :]


def mean_activation(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(asset.data.data.act[:, :], dim=1)


def body_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.data.xpos[:, asset.data.indexing.body_ids[asset_cfg.body_ids], 2]
