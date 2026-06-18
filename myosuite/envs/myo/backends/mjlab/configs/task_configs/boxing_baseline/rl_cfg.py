import os

import numpy as np
import torch
import wandb
from torch import nn
from torch.distributions import Beta, Normal
from rsl_rl.modules import Distribution

from mjlab.rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class BetaDistribution(Distribution):
  """Beta distribution module for bounded action spaces.

  This distribution parameterizes stochastic outputs using a Beta distribution, which naturally constrains samples
  to [0, 1]. Samples are linearly rescaled to ``action_range``, which defaults to ``(-1.0, 1.0)``.

  The MLP must output a tensor of shape ``[..., 2, output_dim]``, where the first slice along the second-to-last
  dimension contains the raw alpha parameters and the second contains the raw beta parameters. Both are passed
  through ``Softplus + 1`` to ensure they are strictly greater than 1, which guarantees a unimodal distribution.
  """

  def __init__(
      self,
      output_dim: int,
      action_range: tuple[float, float] = (-1.0, 1.0),
  ) -> None:
    """Initialize the Beta distribution module.

    Args:
        output_dim: Dimension of the action/output space.
        action_range: Interval ``(min, max)`` to which Beta samples in ``[0, 1]`` are linearly rescaled.
            Defaults to ``(-1.0, 1.0)``.
    """
    super().__init__(output_dim)

    # Compute scaling and offset for rescaling samples
    self.action_range = action_range
    self._range_scale = action_range[1] - action_range[0]
    self._range_offset = action_range[0]
    self._log_range_scale = np.log(self._range_scale)

    self._distribution: Beta | None = None
    self._alpha: torch.Tensor | None = None
    self._beta: torch.Tensor | None = None

    # Disable args validation for speedup
    Beta.set_default_validate_args(False)

  def update(self, mlp_output: torch.Tensor) -> None:
    """Update the Beta distribution from MLP output."""
    alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
    self._alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
    self._beta = torch.nn.functional.softplus(beta_raw) + 1.0
    self._distribution = Beta(self._alpha, self._beta)

  def sample(self) -> torch.Tensor:
    """Sample from the Beta distribution and rescale to ``action_range``."""
    return self._distribution.sample() * self._range_scale + self._range_offset  # type: ignore

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    """Extract the mean from the MLP output and rescale to ``action_range``."""
    alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
    alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
    beta = torch.nn.functional.softplus(beta_raw) + 1.0
    return (alpha / (alpha + beta)) * self._range_scale + self._range_offset

  def as_deterministic_output_module(self) -> nn.Module:
    """Return export-friendly module that computes the mean from the MLP output."""
    return _BetaDeterministicOutput(self._range_scale, self._range_offset)

  @property
  def input_dim(self) -> list[int]:
    """Return the input dimension required by the distribution.

    The MLP must output a tensor of shape ``[..., 2, output_dim]`` where the first slice along the second-to-last
    dimension is the raw alpha parameter and the second is the raw beta parameter.
    """
    return [2, self.output_dim]

  @property
  def mean(self) -> torch.Tensor:
    """Return the mean of the Beta distribution, rescaled to ``action_range``."""
    return (self._alpha / (self._alpha + self._beta)) * self._range_scale + self._range_offset  # type: ignore

  @property
  def std(self) -> torch.Tensor:
    """Return the standard deviation of the Beta distribution, rescaled to ``action_range``."""
    return self._distribution.stddev * self._range_scale  # type: ignore

  @property
  def entropy(self) -> torch.Tensor:
    """Return the entropy of the Beta distribution, summed over the last dimension."""
    return self._distribution.entropy().sum(dim=-1)  # type: ignore

  @property
  def params(self) -> tuple[torch.Tensor, ...]:
    """Return ``(alpha, beta)`` of the current Beta distribution."""
    return (self._alpha, self._beta)  # type: ignore

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    """Compute the log probability under the Beta distribution, summed over the last dimension.

    Outputs are unscaled from ``action_range`` back to ``[0, 1]`` before computing the log probability.
    The Jacobian correction for the linear rescaling is included.
    """
    unscaled = (outputs - self._range_offset) / self._range_scale
    unscaled = unscaled.clamp(1e-6, 1.0 - 1e-6)
    # Jacobian correction: log p(y) = log p(x) - log(scale), where y = x * scale + offset
    return (self._distribution.log_prob(unscaled) - self._log_range_scale).sum(dim=-1)  # type: ignore

  def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Compute KL(old || new) between two Beta distributions."""
    old_alpha, old_beta = old_params
    new_alpha, new_beta = new_params
    return torch.distributions.kl_divergence(Beta(old_alpha, old_beta), Beta(new_alpha, new_beta)).sum(dim=-1)

  def init_mlp_weights(self, mlp: nn.Module) -> None:
    """Initialize the beta-parameter head weights to zero for a near-uniform initial distribution."""
    torch.nn.init.zeros_(mlp[-2].weight[self.output_dim:])  # type: ignore
    torch.nn.init.zeros_(mlp[-2].bias[self.output_dim:])  # type: ignore


class _BetaDeterministicOutput(nn.Module):
    """Exportable module that computes the mean of the Beta distribution from the MLP output."""

    def __init__(self, range_scale: float, range_offset: float) -> None:
      super().__init__()
      self.range_scale = range_scale
      self.range_offset = range_offset

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
      alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
      alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
      beta = torch.nn.functional.softplus(beta_raw) + 1.0
      return (alpha / (alpha + beta)) * self.range_scale + self.range_offset


class OnnxSavingOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = "policy.onnx"
        self.export_policy_to_onnx(policy_path, filename)
        run_name: str = (
            wandb.run.name
            if self.logger.logger_type == "wandb" and wandb.run
            else "local"
        )  # type: ignore[assignment]
        onnx_path = os.path.join(policy_path, filename)
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)
        if self.logger.logger_type in ["wandb"]:
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


def torque_body_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for the standing task."""
    return RslRlOnPolicyRunnerCfg(
      actor=RslRlModelCfg(
        hidden_dims=(512, 256, 256),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
          "class_name": "GaussianDistribution",
          "init_std": 0.8,
          "std_range": (1e-5, 0.8)
        },
      ),
      critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
      ),
      algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.98,
        lam=0.96,
        desired_kl=0.01,
        max_grad_norm=1.0,
      ),
      experiment_name="muscle_body",
      save_interval=200,
      num_steps_per_env=64,
      max_iterations=10001,
  )
