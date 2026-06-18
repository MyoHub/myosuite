import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
_DEFAULT_ASSET_CFG = SceneEntityCfg("human")


def activation(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(asset.data.data.act[:, :], dim=1)


def activation_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(torch.square(asset.data.data.act[:, :]), dim=1)


def excitation_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(torch.square(asset.data.data.ctrl[:, :]), dim=1)


def exponential_body_height(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.exp(-(asset.data.data.xpos[:, asset.data.indexing.body_ids[asset_cfg.body_ids][0], 2]-target_height).abs()/std**2)


def quadratic_action_bounds(
    env: ManagerBasedRlEnv,
    lower_limit: float = 0.0,
    upper_limit: float = 1.0,
) -> torch.Tensor:
    actions = env.action_manager.action

    clamped_actions = torch.clamp(actions, min=lower_limit, max=upper_limit)
    squared_violation = torch.mean(torch.square(actions - clamped_actions), dim=-1)

    return squared_violation
