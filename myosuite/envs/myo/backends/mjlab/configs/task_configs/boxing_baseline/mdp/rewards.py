import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("human")


def activation(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize muscle activations on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(asset.data.data.act[:, :], dim=1)


def activation_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize muscle activations on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.data.act[:, :]), dim=1)
