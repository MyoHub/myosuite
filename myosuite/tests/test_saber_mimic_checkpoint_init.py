# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for saber mimic-checkpoint initialization helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from myosuite.integrations.musclemimic.actor_torch import load_mimic_checkpoint_policy
from tutorials.mc26 import mc26_train_saber_mjlab_mimic_init as saber_mimic_init


def _load_duplicate_script_module():
    module_path = Path(saber_mimic_init.__file__).resolve()
    spec = importlib.util.spec_from_file_location(
        "duplicate_saber_mimic_init_test_module",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to build import spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_obs(obs_dim: int) -> TensorDict:
    zeros = torch.zeros(2, obs_dim, dtype=torch.float32)
    return TensorDict({"actor": zeros, "critic": zeros.clone()}, batch_size=[2])


def _make_layer_tree(
    obs_dim: int, hidden_dims: tuple[int, ...], output_dim: int
) -> dict[str, dict]:
    rng = np.random.default_rng(7)
    layer_tree: dict[str, dict] = {}
    in_dim = obs_dim
    for layer_idx, hidden_dim in enumerate(hidden_dims):
        layer_tree[f"Dense_{layer_idx}"] = {
            "kernel": rng.standard_normal((in_dim, hidden_dim)).astype(np.float32)
            * 0.1,
            "bias": rng.standard_normal(hidden_dim).astype(np.float32) * 0.1,
        }
        layer_tree[f"LayerNorm_{layer_idx}"] = {
            "scale": rng.standard_normal(hidden_dim).astype(np.float32) * 0.1 + 1.0,
            "bias": rng.standard_normal(hidden_dim).astype(np.float32) * 0.1,
        }
        in_dim = hidden_dim
    layer_tree[f"Dense_{len(hidden_dims)}"] = {
        "kernel": rng.standard_normal((in_dim, output_dim)).astype(np.float32) * 0.1,
        "bias": rng.standard_normal(output_dim).astype(np.float32) * 0.1,
    }
    return layer_tree


def test_checkpoint_loader_accepts_duplicate_module_model() -> None:
    duplicate_module = _load_duplicate_script_module()
    obs_dim = 5
    hidden_dims = (7, 6)
    output_dim = 3
    model = duplicate_module.LayerNormSiLUMLPModel(
        obs=_make_obs(obs_dim),
        obs_groups={"actor": ["actor"], "critic": ["critic"]},
        obs_set="actor",
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        distribution_cfg=None,
    )
    layer_tree = _make_layer_tree(obs_dim, hidden_dims, output_dim)
    obs_mean = np.linspace(-1.0, 1.0, obs_dim, dtype=np.float32)
    obs_var = np.linspace(0.5, 1.5, obs_dim, dtype=np.float32)
    obs_count = np.array(123.0, dtype=np.float32)

    saber_mimic_init._load_dense_checkpoint_into_model(
        model,
        layer_tree,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_count=obs_count,
    )

    np.testing.assert_allclose(
        model.obs_normalizer._mean.detach().cpu().numpy(),
        obs_mean[None, :],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        model.obs_normalizer._var.detach().cpu().numpy(),
        obs_var[None, :],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        model.obs_normalizer._std.detach().cpu().numpy(),
        np.sqrt(obs_var + 1e-8)[None, :],
        rtol=1e-6,
        atol=1e-6,
    )
    assert float(model.obs_normalizer.count.item()) == float(obs_count)

    linear_layers = [
        module for module in model.mlp if isinstance(module, torch.nn.Linear)
    ]
    norm_layers = [
        module for module in model.mlp if isinstance(module, torch.nn.LayerNorm)
    ]
    for layer_idx, linear in enumerate(linear_layers[:-1]):
        np.testing.assert_allclose(
            linear.weight.detach().cpu().numpy(),
            layer_tree[f"Dense_{layer_idx}"]["kernel"].T,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            linear.bias.detach().cpu().numpy(),
            layer_tree[f"Dense_{layer_idx}"]["bias"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            norm_layers[layer_idx].weight.detach().cpu().numpy(),
            layer_tree[f"LayerNorm_{layer_idx}"]["scale"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            norm_layers[layer_idx].bias.detach().cpu().numpy(),
            layer_tree[f"LayerNorm_{layer_idx}"]["bias"],
            rtol=0.0,
            atol=0.0,
        )

    np.testing.assert_allclose(
        linear_layers[-1].weight.detach().cpu().numpy(),
        layer_tree[f"Dense_{len(hidden_dims)}"]["kernel"].T,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        linear_layers[-1].bias.detach().cpu().numpy(),
        layer_tree[f"Dense_{len(hidden_dims)}"]["bias"],
        rtol=0.0,
        atol=0.0,
    )


def test_load_mimic_checkpoint_policy_builds_actor(monkeypatch) -> None:
    obs_dim = 5
    hidden_dims = (7, 6)
    output_dim = 3
    layer_tree = _make_layer_tree(obs_dim, hidden_dims, output_dim)
    obs_mean = np.linspace(-1.0, 1.0, obs_dim, dtype=np.float32)
    obs_var = np.linspace(0.5, 1.5, obs_dim, dtype=np.float32)
    obs_count = np.array(123.0, dtype=np.float32)
    artifacts = SimpleNamespace(
        params={"actor": layer_tree},
        obs_dim=obs_dim,
        action_dim=output_dim,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_count=obs_count,
    )

    monkeypatch.setattr(
        saber_mimic_init,
        "resolve_checkpoint_ref",
        lambda checkpoint_root: SimpleNamespace(local_path=Path(checkpoint_root)),
    )
    monkeypatch.setattr(
        saber_mimic_init,
        "load_local_policy_artifacts",
        lambda checkpoint_root: artifacts,
    )

    policy = load_mimic_checkpoint_policy(Path("/tmp/fake"), device="cpu")
    output = policy(_make_obs(obs_dim))

    assert output.shape == (2, output_dim)
    np.testing.assert_allclose(
        policy.obs_normalizer._mean.detach().cpu().numpy(),
        obs_mean[None, :],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        policy.obs_normalizer._var.detach().cpu().numpy(),
        obs_var[None, :],
        rtol=0.0,
        atol=0.0,
    )


def test_prepare_config_uses_conservative_resume_defaults() -> None:
    args = saber_mimic_init._parse_args([])

    cfg = saber_mimic_init._prepare_config(args)

    assert cfg.agent.actor.distribution_cfg["init_std"] == 0.05
    assert cfg.agent.algorithm.learning_rate == 3e-5
    assert cfg.agent.algorithm.num_learning_epochs == 1
    assert cfg.agent.algorithm.clip_param == 0.05
    assert cfg.agent.algorithm.entropy_coef == 0.0
    assert cfg.agent.algorithm.desired_kl == 0.001


def test_freeze_actor_std_disables_distribution_gradients() -> None:
    std_param = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    log_std_param = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            actor=SimpleNamespace(
                distribution=SimpleNamespace(
                    std_param=std_param,
                    log_std_param=log_std_param,
                )
            )
        )
    )

    saber_mimic_init._maybe_freeze_actor_std(runner, learn_actor_std=False)

    assert not std_param.requires_grad
    assert not log_std_param.requires_grad
