# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ONNX export and round-trip parity.

Requires ``onnxruntime``; tests are skipped when it is not installed.
"""

from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch

from myosuite.tests.test_actor_torch import make_fake_artifacts


def _onnx_stack_available() -> bool:
    return (
        importlib.util.find_spec("onnx") is not None
        and importlib.util.find_spec("onnxruntime") is not None
    )


@unittest.skipUnless(_onnx_stack_available(), "onnx/onnxruntime not installed")
class TestOnnxExport(unittest.TestCase):
    def setUp(self) -> None:
        from myosuite.integrations.musclemimic.actor_torch import MimicActorModule

        self.obs_dim = 16
        self.act_dim = 8
        artifacts = make_fake_artifacts(self.obs_dim, 32, self.act_dim)
        self.module = MimicActorModule.from_artifacts(artifacts)

    def test_export_creates_file(self) -> None:
        from myosuite.integrations.musclemimic.actor_onnx import export_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            path = export_to_onnx(self.module, Path(tmp) / "actor.onnx")
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_onnx_matches_torch_single(self) -> None:
        from myosuite.integrations.musclemimic.actor_onnx import (
            actor_to_onnx_bytes,
            load_onnx_session_from_bytes,
        )

        data = actor_to_onnx_bytes(self.module)
        session = load_onnx_session_from_bytes(data)

        rng = np.random.default_rng(0)
        obs_np = rng.standard_normal((1, self.obs_dim)).astype(np.float32)

        torch_out = self.module(torch.as_tensor(obs_np)).detach().numpy()
        ort_out = session.act(obs_np)

        np.testing.assert_allclose(torch_out, ort_out, rtol=1e-4, atol=1e-5)

    def test_onnx_matches_torch_batch(self) -> None:
        from myosuite.integrations.musclemimic.actor_onnx import (
            actor_to_onnx_bytes,
            load_onnx_session_from_bytes,
        )

        data = actor_to_onnx_bytes(self.module)
        session = load_onnx_session_from_bytes(data)

        rng = np.random.default_rng(1)
        for batch_size in (1, 8, 64, 256):
            obs_np = rng.standard_normal((batch_size, self.obs_dim)).astype(np.float32)
            torch_out = self.module(torch.as_tensor(obs_np)).detach().numpy()
            ort_out = session.act(obs_np)
            np.testing.assert_allclose(
                torch_out,
                ort_out,
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Mismatch at batch_size={batch_size}",
            )

    def test_session_dimensions(self) -> None:
        from myosuite.integrations.musclemimic.actor_onnx import (
            actor_to_onnx_bytes,
            load_onnx_session_from_bytes,
        )

        data = actor_to_onnx_bytes(self.module)
        session = load_onnx_session_from_bytes(data)
        self.assertEqual(session.obs_dim, self.obs_dim)
        self.assertEqual(session.act_dim, self.act_dim)

    def test_act_returns_float32(self) -> None:
        from myosuite.integrations.musclemimic.actor_onnx import (
            actor_to_onnx_bytes,
            load_onnx_session_from_bytes,
        )

        data = actor_to_onnx_bytes(self.module)
        session = load_onnx_session_from_bytes(data)
        obs = np.zeros((4, self.obs_dim), dtype=np.float32)
        out = session.act(obs)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (4, self.act_dim))

    def test_act_torch_wrapper(self) -> None:
        from myosuite.integrations.musclemimic.actor_onnx import (
            actor_to_onnx_bytes,
            load_onnx_session_from_bytes,
        )

        data = actor_to_onnx_bytes(self.module)
        session = load_onnx_session_from_bytes(data)
        obs = torch.zeros(8, self.obs_dim)
        out = session.act_torch(obs)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (8, self.act_dim))

    def test_load_onnx_missing_file_raises(self) -> None:
        from myosuite.integrations.musclemimic.actor_onnx import load_onnx_session

        with self.assertRaises(FileNotFoundError):
            load_onnx_session("/nonexistent/path/actor.onnx")


@unittest.skipUnless(_onnx_stack_available(), "onnx/onnxruntime not installed")
class TestMjlabPolicyRunnerOnnx(unittest.TestCase):
    def setUp(self) -> None:
        from myosuite.integrations.musclemimic.actor_onnx import (
            actor_to_onnx_bytes,
            load_onnx_session_from_bytes,
        )
        from myosuite.integrations.musclemimic.actor_torch import MimicActorModule
        from myosuite.integrations.musclemimic.mjlab_policy_runner import (
            MjlabPolicyRunner,
        )

        self.obs_dim = 16
        self.act_dim = 8
        artifacts = make_fake_artifacts(self.obs_dim, 32, self.act_dim)
        module = MimicActorModule.from_artifacts(artifacts)
        session = load_onnx_session_from_bytes(actor_to_onnx_bytes(module))
        self.runner = MjlabPolicyRunner.from_onnx(session)

    def test_act_shape(self) -> None:
        obs = torch.zeros(64, self.obs_dim)
        actions = self.runner.act(obs)
        self.assertEqual(actions.shape, (64, self.act_dim))

    def test_act_clamped(self) -> None:
        obs = torch.randn(32, self.obs_dim) * 100.0
        actions = self.runner.act(obs)
        self.assertTrue((actions >= -1.0).all())
        self.assertTrue((actions <= 1.0).all())

    def test_onnx_matches_torch_runner(self) -> None:
        """OnnxActorSession and MimicActorModule must agree on the same obs."""
        from myosuite.integrations.musclemimic.actor_torch import MimicActorModule
        from myosuite.integrations.musclemimic.mjlab_policy_runner import (
            MjlabPolicyRunner,
        )

        artifacts = make_fake_artifacts(self.obs_dim, 32, self.act_dim)
        torch_runner = MjlabPolicyRunner.from_torch(
            MimicActorModule.from_artifacts(artifacts)
        )
        from myosuite.integrations.musclemimic.actor_onnx import (
            actor_to_onnx_bytes,
            load_onnx_session_from_bytes,
        )
        from myosuite.integrations.musclemimic.actor_torch import (
            MimicActorModule as M,
        )

        session = load_onnx_session_from_bytes(
            actor_to_onnx_bytes(M.from_artifacts(artifacts))
        )
        onnx_runner = MjlabPolicyRunner.from_onnx(session)

        rng = np.random.default_rng(55)
        obs = torch.as_tensor(
            rng.standard_normal((16, self.obs_dim)).astype(np.float32)
        )
        a_torch = torch_runner.act(obs)
        a_onnx = onnx_runner.act(obs)

        np.testing.assert_allclose(
            a_torch.numpy(), a_onnx.numpy(), rtol=1e-4, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
