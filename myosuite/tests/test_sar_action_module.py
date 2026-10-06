# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for SARMuscleActivationActionCfg / SARMuscleActivationAction
and register_mimic_mjlab_tasks_with_sar.

No mjlab or musclemimic_models installation required — SAR transform
mechanics are tested directly via the SARTorchTransform unit. The one
real-mjlab test is skipped when mjlab is not installed.
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


def _mjlab_available() -> bool:
    try:
        import mjlab  # noqa: F401

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
    from myosuite.integrations.musclemimic.sar_torch_transform import SARTorchTransform

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


def _muscle_tendon(i: int, n_muscles: int = N_MUSCLES) -> int:
    """Tendon pulled by muscle ``i`` in :func:`_make_fake_env`.

    Tendon 0 is passive and the muscles pull the others in reverse order, so
    actuator and tendon indices differ.
    """
    return n_muscles - i


def _make_fake_env(
    n_envs: int = 4, n_muscles: int = N_MUSCLES
) -> types.SimpleNamespace:
    """Minimal mock of a mjlab env whose muscles are XmlActuatorCfg-wrapped."""
    import mujoco
    import torch

    n_tendons = n_muscles + 1
    muscle_tendons = [_muscle_tendon(i, n_muscles) for i in range(n_muscles)]
    trnid = np.full((n_muscles, 2), -1)
    trnid[:, 0] = muscle_tendons
    mj_model = types.SimpleNamespace(
        actuator_trntype=np.full(n_muscles, mujoco.mjtTrn.mjTRN_TENDON),
        actuator_trnid=trnid,
    )
    indexing = types.SimpleNamespace(
        ctrl_ids=torch.arange(n_muscles, dtype=torch.long),
        tendon_ids=torch.arange(n_tendons, dtype=torch.long),
    )
    entity_data = types.SimpleNamespace(
        indexing=indexing,
        ctrl=torch.zeros(n_envs, n_muscles),
        tendon_effort_target=torch.zeros(n_envs, n_tendons),
    )

    def _find_actuators(names: tuple[str, ...]):
        ids = list(range(len(names)))
        return ids, list(names)

    def _write_ctrl_to_sim(ctrl_values, ctrl_ids=None, env_ids=None):
        row_idx = slice(None) if env_ids is None else env_ids
        col_idx = slice(None) if ctrl_ids is None else ctrl_ids
        entity_data.ctrl[row_idx, col_idx] = ctrl_values

    def _set_tendon_effort_target(effort, tendon_ids=None, env_ids=None):
        row_idx = slice(None) if env_ids is None else env_ids
        col_idx = slice(None) if tendon_ids is None else tendon_ids
        entity_data.tendon_effort_target[row_idx, col_idx] = effort

    def _write_data_to_sim():
        """What XmlActuatorCfg does before every physics step."""
        entity_data.ctrl[:] = entity_data.tendon_effort_target[:, muscle_tendons]

    entity = types.SimpleNamespace(
        data=entity_data,
        indexing=indexing,
        find_actuators=_find_actuators,
        write_ctrl_to_sim=_write_ctrl_to_sim,
        set_tendon_effort_target=_set_tendon_effort_target,
        write_data_to_sim=_write_data_to_sim,
    )
    env = types.SimpleNamespace(
        num_envs=n_envs,
        device="cpu",
        scene={"test_entity": entity},
        sim=types.SimpleNamespace(mj_model=mj_model),
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

    def test_apply_actions_survives_the_actuator_write(self) -> None:
        import torch

        entity = self.env.scene["test_entity"]
        self.action.process_actions(torch.randn(4, N_SYN))
        self.action.apply_actions()
        entity.write_data_to_sim()
        torch.testing.assert_close(entity.data.ctrl, self.action._processed_actions)

    def test_reset_zeroes_only_the_reset_envs(self) -> None:
        import torch

        self.action.process_actions(torch.randn(4, N_SYN))
        self.action.reset(env_ids=torch.tensor([1, 3]))
        for buf in (self.action.raw_action, self.action._processed_actions):
            self.assertTrue((buf[[1, 3]] == 0).all().item())
            self.assertFalse((buf[[0, 2]] == 0).all().item())
        self.action.reset()
        self.assertTrue((self.action.raw_action == 0).all().item())

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


@unittest.skipUnless(
    _torch_available() and _mjlab_available(), "torch/mjlab not available"
)
class TestSARMuscleActivationActionMjlab(unittest.TestCase):
    def test_activations_reach_sim_ctrl(self) -> None:
        """Reset and step a real mjlab env whose muscles are XmlActuatorCfg-wrapped.

        Same composition as ``_make_mimic_sar_env_cfg`` (which needs
        musclemimic_models), rebuilt on the packaged elbow model.
        """
        import torch
        from mjlab.envs import ManagerBasedRlEnv

        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            SARMuscleActivationActionCfg,
        )
        from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
            _ELBOW_ENTITY_NAME,
            _elbow_tendon_names,
            _make_elbow_env_cfg,
        )

        muscles = tuple(n.replace("_tendon", "") for n in _elbow_tendon_names())
        cfg = _make_elbow_env_cfg()
        cfg.actions = {
            "muscles": SARMuscleActivationActionCfg(
                entity_name=_ELBOW_ENTITY_NAME,
                actuator_names=muscles,
                sar_transform=_make_sar_transform(3, len(muscles)),
            )
        }
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        try:
            env.reset()
            term = env.action_manager.get_term("muscles")
            for _ in range(3):
                env.step(torch.full((env.num_envs, term.action_dim), 0.5))
            activations = term._processed_actions
            self.assertGreater(float(activations.max()), 0.1)
            ctrl_ids = env.scene[_ELBOW_ENTITY_NAME].indexing.ctrl_ids
            torch.testing.assert_close(env.sim.data.ctrl[:, ctrl_ids], activations)
        finally:
            env.close()


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


def _synergistic_acts(n_samples: int = 400, n_muscles: int = N_MUSCLES) -> np.ndarray:
    """Low-rank activations (5 non-negative synergies plus noise) in [0, 1]."""
    rng = np.random.default_rng(0)
    weights = rng.uniform(0.0, 1.0, (5, n_muscles))
    drive = rng.uniform(0.0, 1.0, (n_samples, 5)) ** 2
    noise = 0.02 * rng.standard_normal((n_samples, n_muscles))
    return np.clip(drive @ weights / 5.0 + noise, 0.0, 1.0).astype(np.float32)


@unittest.skipUnless(
    _torch_available() and _sklearn_available(), "torch/sklearn not available"
)
class TestSARTorchTransformMatchesSklearn(unittest.TestCase):
    """SARTorchTransform must reproduce the sklearn inverse chain it mirrors,
    ``pca.inverse_transform(ica.inverse_transform(scaler.inverse_transform(s)))``,
    for real fitted sklearn objects (the fakes above have no ``whiten``)."""

    def _assert_matches_sklearn(self, pca, ica, scaler, syn: np.ndarray) -> None:
        import torch

        from myosuite.integrations.musclemimic.sar_torch_transform import (
            SARTorchTransform,
        )

        expected = np.clip(
            pca.inverse_transform(ica.inverse_transform(scaler.inverse_transform(syn))),
            0.0,
            1.0,
        )
        transform = SARTorchTransform(ica, pca, scaler, device="cpu")
        out = transform(torch.as_tensor(syn, dtype=torch.float32)).numpy()
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-5)

    def test_whitened_pca_from_extract_synergies(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import (
            encode_activations,
            extract_synergies,
        )

        acts = _synergistic_acts()
        model = extract_synergies(acts, n_synergies=N_SYN)
        self.assertTrue(model.pca.whiten)
        syn = encode_activations(model, acts[:100])
        self._assert_matches_sklearn(model.pca, model.ica, model.scaler, syn)

    def test_unwhitened_pca(self) -> None:
        from sklearn.decomposition import PCA, FastICA
        from sklearn.preprocessing import MinMaxScaler

        acts = _synergistic_acts().astype(np.float64)
        pca = PCA(n_components=N_SYN).fit(acts)
        ica = FastICA(n_components=N_SYN, max_iter=1000, random_state=0)
        x_ica = ica.fit_transform(pca.transform(acts))
        scaler = MinMaxScaler().fit(x_ica)
        syn = np.clip(scaler.transform(x_ica[:100]), 0.0, 1.0)
        self._assert_matches_sklearn(pca, ica, scaler, syn)


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
