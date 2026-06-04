# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Round-trip parity tests for MimicActorModule vs the original numpy forward.

All tests use synthetic param trees — no real checkpoint required.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers: synthetic param trees
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _rand(*shape: int) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float32) * 0.1


def make_fake_params(
    obs_dim: int = 16,
    hidden_dim: int = 32,
    act_dim: int = 8,
    with_proj: bool = True,
) -> dict[str, Any]:
    """Build a minimal param tree matching the expected checkpoint structure."""
    actor: dict[str, Any] = {}
    in_dim = obs_dim
    for block_idx in range(2):
        actor[f"block{block_idx}_layer0_dense"] = {
            "kernel": _rand(in_dim, hidden_dim),
            "bias": _rand(hidden_dim),
        }
        actor[f"block{block_idx}_layer0_ln"] = {
            "scale": np.ones(hidden_dim, dtype=np.float32),
            "bias": np.zeros(hidden_dim, dtype=np.float32),
        }
        actor[f"block{block_idx}_layer1_dense"] = {
            "kernel": _rand(hidden_dim, hidden_dim),
            "bias": _rand(hidden_dim),
        }
        actor[f"block{block_idx}_layer1_ln"] = {
            "scale": np.ones(hidden_dim, dtype=np.float32),
            "bias": np.zeros(hidden_dim, dtype=np.float32),
        }
        if with_proj and in_dim != hidden_dim:
            actor[f"block{block_idx}_proj"] = {
                "kernel": _rand(in_dim, hidden_dim),
                "bias": _rand(hidden_dim),
            }
        actor[f"res_gate_{block_idx}"] = np.array(0.5, dtype=np.float32)
        in_dim = hidden_dim

    actor["tail_dense"] = {
        "kernel": _rand(hidden_dim, hidden_dim),
        "bias": _rand(hidden_dim),
    }
    actor["tail_ln"] = {
        "scale": np.ones(hidden_dim, dtype=np.float32),
        "bias": np.zeros(hidden_dim, dtype=np.float32),
    }
    actor["output"] = {
        "kernel": _rand(hidden_dim, act_dim),
        "bias": _rand(act_dim),
    }
    return {"actor": actor}


def make_fake_artifacts(
    obs_dim: int = 16,
    hidden_dim: int = 32,
    act_dim: int = 8,
) -> Any:
    """Return a namespace mimicking LocalPolicyArtifacts."""
    import types

    params = make_fake_params(obs_dim, hidden_dim, act_dim)
    return types.SimpleNamespace(
        params=params,
        obs_mean=np.zeros(obs_dim, dtype=np.float32),
        obs_var=np.ones(obs_dim, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=act_dim,
    )


# ---------------------------------------------------------------------------
# Reference numpy forward (copy of fullbody_local_policy._actor_forward)
# ---------------------------------------------------------------------------


def _layer_norm_np(x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + 1e-5)) * scale + bias


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _dense_np(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.matmul(x, kernel) + bias


def numpy_actor_forward(params: dict[str, Any], obs: np.ndarray) -> np.ndarray:
    """Pure-numpy reference matching fullbody_local_policy._actor_forward."""
    actor = params["actor"]
    h = obs
    for block_idx in (0, 1):
        d0 = actor[f"block{block_idx}_layer0_dense"]
        ln0 = actor[f"block{block_idx}_layer0_ln"]
        y = _dense_np(h, d0["kernel"], d0["bias"])
        y = _silu_np(_layer_norm_np(y, ln0["scale"], ln0["bias"]))

        d1 = actor[f"block{block_idx}_layer1_dense"]
        ln1 = actor[f"block{block_idx}_layer1_ln"]
        y = _dense_np(y, d1["kernel"], d1["bias"])
        y = _layer_norm_np(y, ln1["scale"], ln1["bias"])

        proj = actor.get(f"block{block_idx}_proj")
        shortcut = h if proj is None else _dense_np(h, proj["kernel"], proj["bias"])
        gate_raw = float(np.asarray(actor[f"res_gate_{block_idx}"], dtype=np.float32))
        gate = 1.0 / (1.0 + np.exp(-gate_raw))
        h = _silu_np(shortcut + gate * y)

    tail = actor["tail_dense"]
    tail_ln = actor["tail_ln"]
    h = _dense_np(h, tail["kernel"], tail["bias"])
    h = _silu_np(_layer_norm_np(h, tail_ln["scale"], tail_ln["bias"]))
    out = actor["output"]
    return _dense_np(h, out["kernel"], out["bias"]).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: MimicActorModule
# ---------------------------------------------------------------------------


class TestMimicActorModuleForward(unittest.TestCase):
    def setUp(self) -> None:
        from myosuite.integrations.musclemimic.actor_torch import MimicActorModule

        self.obs_dim = 16
        self.hidden_dim = 32
        self.act_dim = 8
        self.params = make_fake_params(self.obs_dim, self.hidden_dim, self.act_dim)
        self.obs_mean = np.zeros(self.obs_dim, dtype=np.float32)
        self.obs_var = np.ones(self.obs_dim, dtype=np.float32)
        self.module = MimicActorModule(self.params, self.obs_mean, self.obs_var)
        self.module.eval()

    def test_output_shape_single(self) -> None:
        obs = torch.zeros(1, self.obs_dim)
        out = self.module(obs)
        self.assertEqual(out.shape, (1, self.act_dim))

    def test_output_shape_batch(self) -> None:
        obs = torch.zeros(64, self.obs_dim)
        out = self.module(obs)
        self.assertEqual(out.shape, (64, self.act_dim))

    def test_output_dtype_float32(self) -> None:
        obs = torch.zeros(4, self.obs_dim)
        out = self.module(obs)
        self.assertEqual(out.dtype, torch.float32)

    def test_parity_with_numpy_single(self) -> None:
        """forward_normalized must match numpy reference on a single obs."""
        rng = np.random.default_rng(7)
        obs_np = rng.standard_normal((self.obs_dim,)).astype(np.float32)

        expected = numpy_actor_forward(self.params, obs_np)
        obs_t = torch.as_tensor(obs_np).unsqueeze(0)
        got = self.module.forward_normalized(obs_t).detach().numpy().squeeze(0)

        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)

    def test_parity_with_numpy_batch(self) -> None:
        """Batched forward must match row-wise numpy reference."""
        rng = np.random.default_rng(13)
        obs_np = rng.standard_normal((8, self.obs_dim)).astype(np.float32)

        expected = np.stack(
            [numpy_actor_forward(self.params, obs_np[i]) for i in range(8)]
        )
        obs_t = torch.as_tensor(obs_np)
        got = self.module.forward_normalized(obs_t).detach().numpy()

        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)

    def test_normalization_applied(self) -> None:
        """forward(obs) must equal forward_normalized((obs - mean) / std)."""
        from myosuite.integrations.musclemimic.actor_torch import MimicActorModule

        obs_mean = np.full(self.obs_dim, 2.0, dtype=np.float32)
        obs_var = np.full(self.obs_dim, 4.0, dtype=np.float32)
        module = MimicActorModule(self.params, obs_mean, obs_var)
        module.eval()

        rng = np.random.default_rng(99)
        obs_np = rng.standard_normal((4, self.obs_dim)).astype(np.float32)
        obs_t = torch.as_tensor(obs_np)

        norm_obs = (obs_np - obs_mean) / np.sqrt(obs_var + 1e-8)
        expected = module.forward_normalized(torch.as_tensor(norm_obs))
        got = module(obs_t)

        np.testing.assert_allclose(
            got.detach().numpy(), expected.detach().numpy(), rtol=1e-5, atol=1e-6
        )

    def test_from_artifacts(self) -> None:
        from myosuite.integrations.musclemimic.actor_torch import MimicActorModule

        artifacts = make_fake_artifacts(self.obs_dim, self.hidden_dim, self.act_dim)
        module = MimicActorModule.from_artifacts(artifacts)
        self.assertIsInstance(module, MimicActorModule)
        self.assertEqual(module.obs_dim, self.obs_dim)
        self.assertEqual(module.act_dim, self.act_dim)

    def test_dimensions_inferred(self) -> None:
        self.assertEqual(self.module.obs_dim, self.obs_dim)
        self.assertEqual(self.module.act_dim, self.act_dim)

    def test_no_proj_block(self) -> None:
        """When hidden_dim == obs_dim, no projection is needed."""
        from myosuite.integrations.musclemimic.actor_torch import MimicActorModule

        dim = 32
        params = make_fake_params(
            obs_dim=dim, hidden_dim=dim, act_dim=8, with_proj=False
        )
        module = MimicActorModule(
            params, np.zeros(dim, np.float32), np.ones(dim, np.float32)
        )
        module.eval()

        obs = torch.zeros(2, dim)
        out = module(obs)
        self.assertEqual(out.shape, (2, 8))

    def test_device_move(self) -> None:
        """Module must move cleanly to cpu (and back) without errors."""
        self.module.cpu()
        obs = torch.zeros(2, self.obs_dim)
        out = self.module(obs)
        self.assertEqual(out.device.type, "cpu")


class TestMimicActorModuleNormalize(unittest.TestCase):
    def test_normalize_zero_mean_unit_var(self) -> None:
        from myosuite.integrations.musclemimic.actor_torch import MimicActorModule

        obs_dim = 8
        params = make_fake_params(obs_dim, 16, 4, with_proj=False)
        module = MimicActorModule(
            params,
            np.zeros(obs_dim, dtype=np.float32),
            np.ones(obs_dim, dtype=np.float32),
        )
        obs = torch.tensor([[1.0, 2.0, 3.0, 0.0, -1.0, 0.5, -0.5, 1.5]])
        norm = module.normalize(obs)
        # var=1 → std=sqrt(1+1e-8)≈1; mean=0 → norm≈obs
        np.testing.assert_allclose(
            norm.detach().numpy(), obs.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_normalize_shifts_and_scales(self) -> None:
        from myosuite.integrations.musclemimic.actor_torch import MimicActorModule

        obs_dim = 4
        params = make_fake_params(obs_dim, 8, 2, with_proj=False)
        mean = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        var = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
        module = MimicActorModule(params, mean, var)

        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # equal to mean → norm ≈ 0
        norm = module.normalize(obs)
        np.testing.assert_allclose(
            norm.detach().numpy(), np.zeros((1, obs_dim)), atol=1e-5
        )


def test_actor_module_backward_compatible_aliases() -> None:
    from myosuite.integrations.musclemimic.actor_torch import (
        DenseActorModule,
        MimicActorModule,
        MuscleMimicActorModule,
        MuscleMimicDenseActorModule,
        make_actor_module,
        make_musclemimic_actor_module,
    )

    assert MuscleMimicActorModule is MimicActorModule
    assert MuscleMimicDenseActorModule is DenseActorModule
    assert make_musclemimic_actor_module is make_actor_module


# ---------------------------------------------------------------------------
# Tests: MjlabPolicyRunner (torch backend, no ONNX needed)
# ---------------------------------------------------------------------------


class TestMjlabPolicyRunnerTorch(unittest.TestCase):
    def setUp(self) -> None:
        from myosuite.integrations.musclemimic.actor_torch import MimicActorModule
        from myosuite.integrations.musclemimic.mjlab_policy_runner import (
            MjlabPolicyRunner,
        )

        self.obs_dim = 16
        self.act_dim = 8
        artifacts = make_fake_artifacts(self.obs_dim, 32, self.act_dim)
        module = MimicActorModule.from_artifacts(artifacts)
        self.runner = MjlabPolicyRunner.from_torch(module)

    def test_act_shape(self) -> None:
        obs = torch.zeros(1024, self.obs_dim)
        actions = self.runner.act(obs)
        self.assertEqual(actions.shape, (1024, self.act_dim))

    def test_act_clamped(self) -> None:
        obs = torch.randn(64, self.obs_dim) * 100.0  # extreme inputs
        actions = self.runner.act(obs)
        self.assertTrue((actions >= -1.0).all())
        self.assertTrue((actions <= 1.0).all())

    def test_act_dtype_float32(self) -> None:
        obs = torch.zeros(4, self.obs_dim, dtype=torch.float64)
        actions = self.runner.act(obs)
        self.assertEqual(actions.dtype, torch.float32)

    def test_act_raises_wrong_obs_dim(self) -> None:
        obs = torch.zeros(4, self.obs_dim + 1)
        with self.assertRaises(ValueError):
            self.runner.act(obs)

    def test_act_stochastic_shape(self) -> None:
        obs = torch.zeros(16, self.obs_dim)
        log_std = np.zeros(self.act_dim, dtype=np.float32)
        actions = self.runner.act_stochastic(obs, log_std)
        self.assertEqual(actions.shape, (16, self.act_dim))

    def test_act_stochastic_clamped(self) -> None:
        obs = torch.randn(64, self.obs_dim)
        log_std = np.full(self.act_dim, 5.0, dtype=np.float32)  # large std
        actions = self.runner.act_stochastic(obs, log_std)
        self.assertTrue((actions >= -1.0).all())
        self.assertTrue((actions <= 1.0).all())

    def test_act_deterministic_reproducible(self) -> None:
        obs = torch.zeros(4, self.obs_dim)
        a1 = self.runner.act(obs)
        a2 = self.runner.act(obs)
        np.testing.assert_array_equal(a1.detach().numpy(), a2.detach().numpy())


if __name__ == "__main__":
    unittest.main()
