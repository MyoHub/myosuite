# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for SARMuscleActivationActionCfg / SARMuscleActivationAction
and register_mimic_mjlab_tasks_with_sar.

No mjlab or musclemimic_models installation required — SAR transform
mechanics are tested directly via the SARTorchTransform unit.
"""

from __future__ import annotations

import tempfile
import types
import unittest

import numpy as np


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        import joblib  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SYN = 6
N_MUSCLES = 20


def _make_sar_transform(n_syn: int = N_SYN, n_muscles: int = N_MUSCLES) -> object:
    """Build a SARTorchTransform from synthetic numpy arrays (no sklearn needed).

    Constructs lightweight namespace objects that expose only the attributes
    consumed by SARTorchTransform: ``ica.mixing_``, ``ica.mean_``,
    ``pca.components_``, ``pca.mean_``, ``normalizer.min_``,
    ``normalizer.scale_``.
    """
    from benchmarks.sar_backends.sar_torch_transform import SARTorchTransform

    rng = np.random.default_rng(0)
    # PCA: projects n_muscles → n_syn
    pca = types.SimpleNamespace(
        components_=rng.standard_normal((n_syn, n_muscles)).astype(np.float64),
        mean_=rng.random(n_muscles).astype(np.float64),
    )
    # ICA: square transform in PCA space
    ica = types.SimpleNamespace(
        mixing_=rng.standard_normal((n_syn, n_syn)).astype(np.float64),
        mean_=rng.random(n_syn).astype(np.float64),
    )
    # MinMaxScaler: scale and min per synergy component
    normalizer = types.SimpleNamespace(
        scale_=np.ones(n_syn, dtype=np.float64),
        min_=np.zeros(n_syn, dtype=np.float64),
    )
    return SARTorchTransform(ica, pca, normalizer, device="cpu")


def _make_fake_env(
    n_envs: int = 4, n_muscles: int = N_MUSCLES
) -> types.SimpleNamespace:
    """Minimal mock of a mjlab ManagerBasedRlEnv."""
    import torch

    ctrl = torch.zeros(n_envs, n_muscles)
    ctrl_ids = torch.arange(n_muscles, dtype=torch.long)

    indexing = types.SimpleNamespace(ctrl_ids=ctrl_ids)
    entity_data_inner = types.SimpleNamespace(ctrl=ctrl)
    entity_data = types.SimpleNamespace(data=entity_data_inner, indexing=indexing)

    def _find_actuators(names: tuple[str, ...]):
        ids = list(range(len(names)))
        return ids, list(names)

    def _write_ctrl_to_sim(ctrl_values, ctrl_ids=None, env_ids=None):
        """Mock of the mjlab Entity write API used by apply_actions()."""
        row_idx = slice(None) if env_ids is None else env_ids
        col_idx = slice(None) if ctrl_ids is None else ctrl_ids
        entity_data_inner.ctrl[row_idx, col_idx] = ctrl_values

    entity = types.SimpleNamespace(
        data=entity_data,
        find_actuators=_find_actuators,
        write_ctrl_to_sim=_write_ctrl_to_sim,
    )
    env = types.SimpleNamespace(
        num_envs=n_envs,
        device="cpu",
        scene={"test_entity": entity},
    )
    return env


# ---------------------------------------------------------------------------
# Tests: SARMuscleActivationActionCfg / SARMuscleActivationAction
# ---------------------------------------------------------------------------


@unittest.skipUnless(_torch_available(), "torch not available")
class TestSARMuscleActivationAction(unittest.TestCase):
    def setUp(self) -> None:
        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            SARMuscleActivationActionCfg,
        )

        self.sar_transform = _make_sar_transform(N_SYN, N_MUSCLES)
        actuator_names = tuple(f"muscle_{i}" for i in range(N_MUSCLES))
        self.cfg = SARMuscleActivationActionCfg(
            entity_name="test_entity",
            actuator_names=actuator_names,
            sar_transform=self.sar_transform,
        )
        self.env = _make_fake_env(n_envs=4, n_muscles=N_MUSCLES)
        self.action = self.cfg.build(self.env)

    def test_action_dim_equals_n_syn(self) -> None:
        self.assertEqual(self.action.action_dim, N_SYN)

    def test_process_actions_shape(self) -> None:
        import torch

        actions = torch.zeros(4, N_SYN)
        self.action.process_actions(actions)
        self.assertEqual(self.action._processed_actions.shape, (4, N_MUSCLES))

    def test_processed_actions_in_0_1(self) -> None:
        import torch

        actions = torch.randn(4, N_SYN)
        self.action.process_actions(actions)
        acts = self.action._processed_actions
        self.assertTrue((acts >= 0.0).all(), "activations below 0")
        self.assertTrue((acts <= 1.0).all(), "activations above 1")

    def test_apply_actions_writes_to_ctrl(self) -> None:
        import torch

        actions = torch.randn(4, N_SYN)
        self.action.process_actions(actions)
        self.action.apply_actions()
        ctrl = self.env.scene["test_entity"].data.data.ctrl
        # ctrl must have been updated (not all zeros)
        self.assertFalse((ctrl == 0).all().item())

    def test_raw_action_shape(self) -> None:
        self.assertEqual(self.action.raw_action.shape, (4, N_SYN))

    def test_wrong_actuator_count_raises(self) -> None:
        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            SARMuscleActivationActionCfg,
        )

        cfg = SARMuscleActivationActionCfg(
            entity_name="test_entity",
            actuator_names=tuple(f"m_{i}" for i in range(N_MUSCLES + 5)),
            sar_transform=self.sar_transform,
        )
        with self.assertRaises(ValueError):
            cfg.build(self.env)

    def test_build_returns_action_instance(self) -> None:
        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            SARMuscleActivationAction,
        )

        self.assertIsInstance(self.action, SARMuscleActivationAction)


# ---------------------------------------------------------------------------
# Tests: SARTorchTransform forward (inverse pipeline correctness)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_torch_available(), "torch not available")
class TestSARTorchTransformForward(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = _make_sar_transform(N_SYN, N_MUSCLES)

    def test_output_shape(self) -> None:
        import torch

        syn = torch.zeros(8, N_SYN)
        out = self.transform(syn)
        self.assertEqual(out.shape, (8, N_MUSCLES))

    def test_output_clamped_0_1(self) -> None:
        import torch

        syn = torch.randn(64, N_SYN) * 10.0
        out = self.transform(syn)
        self.assertTrue((out >= 0.0).all())
        self.assertTrue((out <= 1.0).all())

    def test_n_syn_property(self) -> None:
        self.assertEqual(self.transform.n_syn, N_SYN)

    def test_n_muscles_property(self) -> None:
        self.assertEqual(self.transform.n_muscles, N_MUSCLES)

    def test_deterministic(self) -> None:
        import torch

        syn = torch.randn(4, N_SYN)
        out1 = self.transform(syn)
        out2 = self.transform(syn)
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())


# ---------------------------------------------------------------------------
# Tests: register_mimic_mjlab_tasks_with_sar (smoke, no mjlab needed)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _torch_available() and _sklearn_available(), "torch/sklearn not available"
)
class TestRegisterMimicMjlabTasksWithSar(unittest.TestCase):
    def test_missing_sar_dir_raises(self) -> None:
        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            register_mimic_mjlab_tasks_with_sar,
        )

        with self.assertRaises(FileNotFoundError):
            register_mimic_mjlab_tasks_with_sar(
                register_mjlab_task=lambda **_: None,
                rl_cfg_fn=lambda: None,
                sar_dir="/nonexistent/path",
            )

    def test_valid_sar_dir_does_not_raise(self) -> None:
        """With a real SAR dir, registration proceeds without error even
        when musclemimic_models is absent (silently skips registration)."""
        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            register_mimic_mjlab_tasks_with_sar,
        )
        from myosuite.integrations.musclemimic.sar_extraction import (
            extract_synergies,
            save_synergy_model,
        )

        rng = np.random.default_rng(7)
        acts = rng.random((200, N_MUSCLES)).astype(np.float32)
        model = extract_synergies(acts, n_synergies=N_SYN)

        with tempfile.TemporaryDirectory() as tmp:
            save_synergy_model(model, tmp)
            registered: list[str] = []

            def _reg(**kwargs: object) -> None:
                registered.append(str(kwargs.get("task_id", "")))

            # Should not raise regardless of musclemimic_models availability.
            try:
                register_mimic_mjlab_tasks_with_sar(
                    register_mjlab_task=_reg,
                    rl_cfg_fn=lambda: None,
                    sar_dir=tmp,
                )
            except Exception as exc:
                # Only fail if the exception is NOT about missing musclemimic_models / mjlab.
                msg = str(exc).lower()
                if (
                    "musclemimic_models" not in msg
                    and "mjlab" not in msg
                    and "no module" not in msg
                ):
                    raise


if __name__ == "__main__":
    unittest.main()
