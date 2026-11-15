# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Unit and round-trip integration tests for ``training/eden_rl``."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch import nn

if importlib.util.find_spec("gymnasium") is None:
    _repo = Path(__file__).resolve().parents[3]
    for _pkg, _rel in (
        ("myosuite", "myosuite"),
        ("myosuite.physics", "myosuite/physics"),
        ("myosuite.terms", "myosuite/terms"),
        ("myosuite.utils", "myosuite/utils"),
        ("myosuite.core", "myosuite/core"),
    ):
        if _pkg not in sys.modules:
            _mod = types.ModuleType(_pkg)
            _mod.__path__ = [str(_repo / _rel)]  # type: ignore[attr-defined]
            sys.modules[_pkg] = _mod
    if "myosuite.core.registry" not in sys.modules:
        _reg = types.ModuleType("myosuite.core.registry")
        _reg.make_env = lambda *a, **kw: None  # type: ignore[attr-defined]
        sys.modules["myosuite.core.registry"] = _reg

from eden_rl import (
    AmpDiscriminator,
    AmpMotionSampler,
    AmpReplayBuffer,
    FrameNormalizer,
    GenesisMuscleBridge,
    HIMActor,
    HIMEstimator,
    MuonAdamW,
    SimBa,
    build_simba_trunk,
    export_simba_to_onnx,
    him_symmetry_losses,
    load_simba_policy,
    rebuild_simba_trunk,
    register_simba_rslrl_loader,
    sinkhorn,
)
from eden_rl.train import configure_eden_rsl_runner
from eden_rl.train import main as train_main
from myosuite.utils.rslrl_policy import _build_normalizer
from myosuite.physics.hill_muscle import (
    MuscleParams,
    muscle_activation_dot,
    muscle_actuator_force,
    step_muscle_activation,
)


def _build_hill_test_module() -> types.ModuleType:
    """Reference Millard/Hill test stub used only when ``myo-sim==0.2.5`` lacks ``#200``."""
    mod = types.ModuleType("myo_sim.physics.hill_muscle")
    mj_min = 1e-15

    def _is_pt(x: Any) -> bool:
        return isinstance(x, torch.Tensor)

    def _bump(L: Any, a: Any, mid: Any, b: Any) -> Any:
        lm, rm = 0.5 * (a + mid), 0.5 * (mid + b)
        if _is_pt(L):
            out = torch.zeros_like(L)
            t1 = (L - a) / (lm - a).clamp(min=mj_min)
            t2 = (mid - L) / (mid - lm).clamp(min=mj_min)
            t3 = (L - mid) / (rm - mid).clamp(min=mj_min)
            t4 = (b - L) / (b - rm).clamp(min=mj_min)
            out = torch.where((L > a) & (L <= lm), 0.5 * t1 * t1, out)
            out = torch.where((L > lm) & (L <= mid), 1.0 - 0.5 * t2 * t2, out)
            out = torch.where((L > mid) & (L <= rm), 1.0 - 0.5 * t3 * t3, out)
            return torch.where((L > rm) & (L < b), 0.5 * t4 * t4, out)
        out_np = np.zeros_like(L, dtype=np.float64)
        t1 = (L - a) / np.maximum(mj_min, lm - a)
        t2 = (mid - L) / np.maximum(mj_min, mid - lm)
        t3 = (L - mid) / np.maximum(mj_min, rm - mid)
        t4 = (b - L) / np.maximum(mj_min, b - rm)
        out_np = np.where((L > a) & (L <= lm), 0.5 * t1 * t1, out_np)
        out_np = np.where((L > lm) & (L <= mid), 1.0 - 0.5 * t2 * t2, out_np)
        out_np = np.where((L > mid) & (L <= rm), 1.0 - 0.5 * t3 * t3, out_np)
        return np.where((L > rm) & (L < b), 0.5 * t4 * t4, out_np)

    def _act_dot(ctrl: Any, act: Any, params: Any) -> Any:
        if _is_pt(act):
            dev, dt = act.device, act.dtype
            u = torch.as_tensor(ctrl, device=dev, dtype=dt).clamp(0.0, 1.0)
            ac = act.clamp(0.0, 1.0)
            ta = torch.as_tensor(params.tau_act, device=dev, dtype=dt) * (
                0.5 + 1.5 * ac
            )
            td = torch.as_tensor(params.tau_deact, device=dev, dtype=dt) / (
                0.5 + 1.5 * ac
            )
            sm = torch.as_tensor(params.tausmooth, device=dev, dtype=dt)
            dc = u - act
            x = (dc / sm.clamp(min=mj_min) + 0.5).clamp(0.0, 1.0)
            sig = x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)
            tau = torch.where(
                sm < mj_min, torch.where(dc > 0.0, ta, td), td + (ta - td) * sig
            )
            return dc / tau.clamp(min=mj_min)
        u_np = np.clip(np.asarray(ctrl, dtype=np.float64), 0.0, 1.0)
        a_np = np.asarray(act, dtype=np.float64)
        ac_np = np.clip(a_np, 0.0, 1.0)
        ta_np = params.tau_act * (0.5 + 1.5 * ac_np)
        td_np = params.tau_deact / (0.5 + 1.5 * ac_np)
        dc_np = u_np - a_np
        x_np = np.clip(dc_np / np.maximum(mj_min, params.tausmooth) + 0.5, 0.0, 1.0)
        sig_np = x_np * x_np * x_np * (3.0 * x_np * (2.0 * x_np - 5.0) + 10.0)
        tau_np = np.where(
            params.tausmooth < mj_min,
            np.where(dc_np > 0.0, ta_np, td_np),
            td_np + (ta_np - td_np) * sig_np,
        )
        return dc_np / np.maximum(mj_min, tau_np)

    def _step_act(
        act: Any, ctrl: Any, params: Any, dt: float, substeps: int = 1
    ) -> tuple[Any, Any]:
        if _is_pt(act):
            dev, dtype = act.device, act.dtype
            lo = torch.as_tensor(params.act_range[:, 0], device=dev, dtype=dtype)
            hi = torch.as_tensor(params.act_range[:, 1], device=dev, dtype=dtype)
            cur = act
            tot = torch.zeros_like(cur)
            for _ in range(substeps):
                tot = tot + cur
                cur = cur + dt * _act_dot(ctrl, cur, params)
                if bool(params.act_limited.any()):
                    cur = torch.maximum(torch.minimum(cur, hi), lo)
            return cur, tot / substeps
        cur_np = np.asarray(act, dtype=np.float64).copy()
        tot_np = np.zeros_like(cur_np)
        for _ in range(substeps):
            tot_np = tot_np + cur_np
            cur_np = cur_np + dt * _act_dot(ctrl, cur_np, params)
            if bool(params.act_limited.any()):
                cur_np = np.clip(cur_np, params.act_range[:, 0], params.act_range[:, 1])
        return cur_np, tot_np / substeps

    def _gain(length: Any, velocity: Any, params: Any) -> Any:
        prm, lr = params.gain_prm, params.length_range
        L0_np = (lr[:, 1] - lr[:, 0]) / np.maximum(1e-10, prm[:, 1] - prm[:, 0])
        F0_np = np.where(
            prm[:, 2] < 0.0, prm[:, 3] / np.maximum(1e-10, params.acc0), prm[:, 2]
        )
        if _is_pt(length):
            dev, dt = length.device, length.dtype
            L0 = torch.as_tensor(L0_np, device=dev, dtype=dt)
            F0 = torch.as_tensor(F0_np, device=dev, dtype=dt)
            L = torch.as_tensor(prm[:, 0], device=dev, dtype=dt) + (
                length - torch.as_tensor(lr[:, 0], device=dev, dtype=dt)
            ) / L0.clamp(min=1e-10)
            V = velocity / (
                L0 * torch.as_tensor(prm[:, 6], device=dev, dtype=dt)
            ).clamp(min=1e-10)
            lmin = torch.as_tensor(prm[:, 4], device=dev, dtype=dt)
            lmax = torch.as_tensor(prm[:, 5], device=dev, dtype=dt)
            fvmax = torch.as_tensor(prm[:, 8], device=dev, dtype=dt)
            fl = _bump(L, lmin, torch.ones_like(lmin), lmax)
            y = fvmax - 1.0
            fv = torch.where(
                V <= -1.0,
                torch.zeros_like(V),
                torch.where(
                    V <= 0.0,
                    (V + 1.0) ** 2,
                    torch.where(
                        V <= y, fvmax - (y - V) ** 2 / y.clamp(min=mj_min), fvmax
                    ),
                ),
            )
            return -F0 * fl * fv
        L_np = prm[:, 0] + (
            np.asarray(length, dtype=np.float64) - lr[:, 0]
        ) / np.maximum(1e-10, L0_np)
        V_np = np.asarray(velocity, dtype=np.float64) / np.maximum(
            1e-10, L0_np * prm[:, 6]
        )
        fl_np = _bump(L_np, prm[:, 4], 1.0, prm[:, 5])
        y_np = prm[:, 8] - 1.0
        fv_np = np.where(
            V_np <= -1.0,
            0.0,
            np.where(
                V_np <= 0.0,
                (V_np + 1.0) ** 2,
                np.where(
                    V_np <= y_np,
                    prm[:, 8] - (y_np - V_np) ** 2 / np.maximum(mj_min, y_np),
                    prm[:, 8],
                ),
            ),
        )
        return -F0_np * fl_np * fv_np

    def _bias(length: Any, params: Any) -> Any:
        prm, lr = params.bias_prm, params.length_range
        L0_np = (lr[:, 1] - lr[:, 0]) / np.maximum(1e-10, prm[:, 1] - prm[:, 0])
        F0_np = np.where(
            prm[:, 2] < 0.0, prm[:, 3] / np.maximum(1e-10, params.acc0), prm[:, 2]
        )
        b_np = 0.5 * (1.0 + prm[:, 5])
        d_np = np.maximum(mj_min, b_np - 1.0)
        if _is_pt(length):
            dev, dt = length.device, length.dtype
            L0 = torch.as_tensor(L0_np, device=dev, dtype=dt)
            F0 = torch.as_tensor(F0_np, device=dev, dtype=dt)
            b = torch.as_tensor(b_np, device=dev, dtype=dt)
            d = torch.as_tensor(d_np, device=dev, dtype=dt)
            fpmax = torch.as_tensor(prm[:, 7], device=dev, dtype=dt)
            L = torch.as_tensor(prm[:, 0], device=dev, dtype=dt) + (
                length - torch.as_tensor(lr[:, 0], device=dev, dtype=dt)
            ) / L0.clamp(min=1e-10)
            fp = fpmax * torch.where(
                L <= 1.0,
                torch.zeros_like(L),
                torch.where(L <= b, 0.5 * ((L - 1.0) / d) ** 2, 0.5 + (L - b) / d),
            )
            return -F0 * fp
        L_np = prm[:, 0] + (
            np.asarray(length, dtype=np.float64) - lr[:, 0]
        ) / np.maximum(1e-10, L0_np)
        fp_np = prm[:, 7] * np.where(
            L_np <= 1.0,
            0.0,
            np.where(
                L_np <= b_np,
                0.5 * ((L_np - 1.0) / d_np) ** 2,
                0.5 + (L_np - b_np) / d_np,
            ),
        )
        return -F0_np * fp_np

    def _force(length: Any, velocity: Any, act: Any, params: Any) -> Any:
        return _gain(length, velocity, params) * act + _bias(length, params)

    class _HillModel:
        def __init__(
            self,
            model: Any = None,
            *,
            params: Any = None,
            nv: int | None = None,
            dt: float = 0.002,
        ) -> None:
            assert params is not None and nv is not None
            self.params = params
            self.nu = len(params.tau_act)
            self.nv = int(nv)
            self.dt = float(dt)

        def generalized_forces(
            self, length: Any, velocity: Any, act: Any, moment_arm: Any
        ) -> Any:
            f = _force(length, velocity, act, self.params)
            if _is_pt(f):
                J = torch.as_tensor(moment_arm, device=f.device, dtype=f.dtype)
                return torch.einsum("...uv,...u->...v", J, f)
            return np.einsum(
                "...uv,...u->...v", np.asarray(moment_arm, dtype=np.float64), f
            )

    mod.muscle_activation_dot = _act_dot  # type: ignore[attr-defined]
    mod.step_muscle_activation = _step_act  # type: ignore[attr-defined]
    mod.muscle_gain = _gain  # type: ignore[attr-defined]
    mod.muscle_bias = _bias  # type: ignore[attr-defined]
    mod.muscle_actuator_force = _force  # type: ignore[attr-defined]
    mod.HillMuscleModel = _HillModel  # type: ignore[attr-defined]
    return mod


class TestSimBaAndRslRlRoundTrip(unittest.TestCase):
    def test_hypersphere_invariants_and_gradients(self) -> None:
        torch.manual_seed(0)
        trunk = build_simba_trunk(
            [14, 22],
            output_dim=39,
            hidden_dim=64,
            num_blocks=2,
            latent_dim=32,
            default_encoder_hidden_dims=(48,),
        )
        x = torch.randn(8, 36, requires_grad=True)
        simba: SimBa = trunk[1]  # type: ignore[index]
        enc_out = trunk[0](x)  # type: ignore[index]
        h0 = simba.embedder(enc_out)
        h1 = simba.blocks(h0)
        torch.testing.assert_close(
            torch.linalg.vector_norm(h0, dim=-1),
            torch.ones(8),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(h1, dim=-1),
            torch.ones(8),
            atol=1e-5,
            rtol=1e-5,
        )
        out = trunk(x)
        self.assertEqual(tuple(out.shape), (8, 39))
        out.pow(2).mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_simba_checkpoint_loads_registers_hook_and_exports_onnx(self) -> None:
        torch.manual_seed(42)
        obs_dim, act_dim = 30, 12
        trunk = build_simba_trunk(
            [10, 20],
            output_dim=act_dim,
            hidden_dim=64,
            num_blocks=2,
            latent_dim=16,
            default_encoder_hidden_dims=(32,),
            activation="swish",
        ).eval()
        norm = _build_normalizer(obs_dim).eval()
        norm._mean.copy_(torch.randn(1, obs_dim))
        norm._var.copy_(torch.rand(1, obs_dim) + 0.25)
        norm._std.copy_(norm._var.sqrt())

        # Non-SimBa state dicts return None from rebuild_simba_trunk.
        self.assertIsNone(rebuild_simba_trunk({"0.weight": torch.ones(4, 4)}))

        # Standalone SimBa (without GroupEncoder) rebuilds cleanly.
        plain_simba = SimBa(
            input_dim=14, output_dim=5, hidden_dim=32, num_blocks=2
        ).eval()
        rebuilt_plain = rebuild_simba_trunk(plain_simba.state_dict())
        self.assertIsNotNone(rebuilt_plain)
        assert rebuilt_plain is not None
        plain_obs = torch.randn(3, 14)
        with torch.no_grad():
            torch.testing.assert_close(
                rebuilt_plain[0].eval()(plain_obs), plain_simba(plain_obs)
            )

        actor_sd: dict[str, torch.Tensor] = {
            f"mlp.{k}": v for k, v in trunk.state_dict().items()
        }
        for k, v in norm.state_dict().items():
            actor_sd[f"obs_normalizer.{k}"] = v
        actor_sd["distribution.std_param"] = torch.full((act_dim,), 0.2)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            (params_dir / "agent.yaml").write_text(
                yaml.safe_dump(
                    {
                        "actor": {
                            "class_name": "eden_rl.simba:SimBaModel",
                            "activation": "swish",
                            "obs_normalization": True,
                            "c_shift": 3.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            ckpt_v5 = run_dir / "model_100.pt"
            torch.save({"actor_state_dict": actor_sd}, ckpt_v5)

            loaded = load_simba_policy(ckpt_v5, act_dim)
            test_obs = torch.randn(5, obs_dim)
            with torch.no_grad():
                want = trunk(norm(test_obs))
                got = loaded(test_obs)
            torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)

            onnx_path = run_dir / "simba_policy.onnx"
            export_simba_to_onnx(
                ckpt_v5,
                onnx_path,
                obs_dim=obs_dim,
                act_dim=act_dim,
            )
            self.assertTrue(onnx_path.is_file())
            import onnxruntime as ort

            sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            ort_out = sess.run(["action"], {"obs": test_obs.numpy()})[0]
            np.testing.assert_allclose(ort_out, want.numpy(), atol=1e-5, rtol=1e-5)

            # Verify rsl-rl v3 model_state_dict translation layout.
            v3_sd: dict[str, torch.Tensor] = {"std": torch.ones(act_dim)}
            for k, v in trunk.state_dict().items():
                v3_sd[f"actor.{k}"] = v
            for k, v in norm.state_dict().items():
                v3_sd[f"actor_obs_normalizer.{k}"] = v
            ckpt_v3 = run_dir / "model_v3.pt"
            torch.save({"model_state_dict": v3_sd}, ckpt_v3)
            loaded_v3 = load_simba_policy(ckpt_v3, act_dim)
            with torch.no_grad():
                torch.testing.assert_close(
                    loaded_v3(test_obs), want, atol=1e-6, rtol=1e-6
                )

        # Verify register_simba_rslrl_loader registers with rslrl_policy hook API.
        registered: dict[str, Any] = {}
        fake_rslrl = types.ModuleType("myosuite.utils.rslrl_policy")
        fake_rslrl.register_actor_trunk_builder = (  # type: ignore[attr-defined]
            lambda name, fn: registered.setdefault(name, fn)
        )
        with patch.dict(sys.modules, {"myosuite.utils.rslrl_policy": fake_rslrl}):
            import myosuite.utils as _utils

            with patch.object(_utils, "rslrl_policy", fake_rslrl, create=True):
                self.assertTrue(register_simba_rslrl_loader())
        self.assertIs(registered.get("eden_rl.simba"), rebuild_simba_trunk)


class TestMuonAdamW(unittest.TestCase):
    def test_partition_step_lr_sync_and_state_dict(self) -> None:
        torch.manual_seed(1)
        model = SimBa(input_dim=16, output_dim=8, hidden_dim=32, num_blocks=2)
        opt = MuonAdamW([model], lr=1e-3)
        self.assertGreater(len(opt.muon_params), 0)
        self.assertGreater(len(opt.adamw_params), 0)
        self.assertTrue(all(p.dim() == 2 for p in opt.muon_params))
        self.assertTrue(all(p.dim() != 2 for p in opt.adamw_params))

        x = torch.randn(32, 16)
        target = torch.randn(32, 8)
        initial_loss = float(nn.functional.mse_loss(model(x), target).item())
        for _ in range(12):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            opt.step()
        final_loss = float(nn.functional.mse_loss(model(x), target).item())
        self.assertLess(final_loss, initial_loss)

        for g in opt.param_groups:
            g["lr"] = 2.5e-4
        for sub in opt.optimizers:
            for g in sub.param_groups:
                self.assertAlmostEqual(g["lr"], 2.5e-4)

        sd = opt.state_dict()
        opt2 = MuonAdamW([model], lr=1e-2)
        opt2.load_state_dict(sd)
        self.assertTrue(all(abs(g["lr"] - 2.5e-4) < 1e-12 for g in opt2.param_groups))


class TestHIMEstimatorAndPhysiologyAdaptation(unittest.TestCase):
    def test_sinkhorn_and_fatigue_latent_regression(self) -> None:
        torch.manual_seed(2)
        scores = torch.randn(16, 8)
        q = sinkhorn(scores, eps=0.1, iters=5)
        self.assertEqual(tuple(q.shape), (16, 8))
        torch.testing.assert_close(q.sum(dim=-1), torch.ones(16), atol=1e-2, rtol=1e-2)

        est = HIMEstimator(
            temporal_steps=5,
            num_one_step_obs=12,
            target_dim=10,
            latent_dim=16,
            num_prototype=16,
            learning_rate=3e-3,
        )
        hist = torch.randn(64, 60)
        true_fatigue = torch.sigmoid(hist[:, :10] * 1.5)
        next_critic = torch.cat([torch.randn(64, 12), true_fatigue], dim=-1)

        loss0, _ = est.update(hist, next_critic)
        loss_last = loss0
        for _ in range(25):
            loss_last, _ = est.update(hist, next_critic)
        self.assertLess(loss_last, loss0 * 0.4)

        actor = HIMActor(est, act_dim=10, hidden_dims=(64, 64)).eval()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_out = Path(tmp) / "him_actor.onnx"
            sample = torch.randn(4, 60)
            torch.onnx.export(
                actor,
                (sample,),
                str(onnx_out),
                input_names=["obs"],
                output_names=["action"],
                dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
            self.assertTrue(onnx_out.is_file())
            import onnxruntime as ort

            sess = ort.InferenceSession(
                str(onnx_out), providers=["CPUExecutionProvider"]
            )
            ort_out = sess.run(["action"], {"obs": sample.numpy()})[0]
            with torch.no_grad():
                torch_out = actor(sample).numpy()
            np.testing.assert_allclose(ort_out, torch_out, atol=1e-4, rtol=1e-4)

    def test_symmetry_losses_zero_under_symmetric_linear_policy(self) -> None:
        obs = torch.randn(8, 6)
        c_obs = torch.randn(8, 8)
        actor_sym, critic_sym = him_symmetry_losses(
            actor_fn=lambda x: x * 0.5,
            critic_fn=lambda x: (x**2).sum(dim=-1, keepdim=True),
            obs_batch=obs,
            critic_obs_batch=c_obs,
            mirror_obs_fn=lambda x: -x,
            mirror_critic_obs_fn=lambda x: -x,
            mirror_act_fn=lambda a: -a,
        )
        self.assertAlmostEqual(float(actor_sym.item()), 0.0, places=6)
        self.assertAlmostEqual(float(critic_sym.item()), 0.0, places=6)


class TestAmpComponents(unittest.TestCase):
    def test_myomimic_qpos_clip_sampler_normalizer_and_discriminator(self) -> None:
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "walk_ref.npz"
            qpos = np.random.default_rng(0).standard_normal((40, 21)).astype(np.float32)
            np.savez(clip_path, qpos=qpos, fps=np.float32(50.0))

            sampler = AmpMotionSampler(
                clip_path, num_frames=4, qpos_slice=slice(7, None)
            )
            self.assertEqual(sampler.frame_dim, 14)
            expert_win = sampler.sample(24)
            self.assertEqual(tuple(expert_win.shape), (24, 4, 14))

            norm = FrameNormalizer(frame_dim=14)
            raw_frames = torch.randn(200, 14) * 3.0 + 2.0
            norm.update(raw_frames[:100])
            norm.update(raw_frames[100:])
            torch.testing.assert_close(
                norm.mean, raw_frames.mean(dim=0), atol=1e-2, rtol=1e-2
            )

            buf = AmpReplayBuffer(window_dim=56, capacity=20)
            buf.insert(torch.randn(15, 56))
            buf.insert(torch.randn(15, 56))
            self.assertEqual(len(buf), 20)
            pol_flat = buf.sample(24)
            self.assertEqual(tuple(pol_flat.shape), (24, 56))

            disc = AmpDiscriminator(window_dim=56, hidden_dims=(64, 64))
            opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
            metrics = disc.lsgan_step(
                opt, pol_flat, expert_win.flatten(1), grad_pen_scale=5.0
            )
            self.assertIn("amp", metrics)
            self.assertGreaterEqual(metrics["amp_grad_pen"], 0.0)
            rew = disc.style_reward(expert_win.flatten(1), coef=5.0)
            self.assertEqual(tuple(rew.shape), (24,))
            self.assertTrue((rew >= 0.0).all())


class TestGenesisMuscleBridgeAndTorchHillParity(unittest.TestCase):
    def test_numpy_vs_torch_hill_curves_and_mocked_genesis_entity(self) -> None:
        has_native_hill = (
            importlib.util.find_spec("myo_sim.physics.hill_muscle") is not None
        )
        ctx = (
            patch.dict(sys.modules, {})
            if has_native_hill
            else patch.dict(
                sys.modules, {"myo_sim.physics.hill_muscle": _build_hill_test_module()}
            )
        )
        with ctx:
            rng = np.random.default_rng(7)
            nu, nv, batch = 8, 5, 4
            params = MuscleParams(
                tau_act=np.full(nu, 0.01),
                tau_deact=np.full(nu, 0.04),
                tausmooth=np.full(nu, 0.2),
                act_limited=np.ones(nu, dtype=bool),
                act_range=np.tile([0.0, 1.0], (nu, 1)),
                length_range=np.tile([0.12, 0.24], (nu, 1)),
                acc0=np.full(nu, 0.02),
                gain_prm=np.tile(
                    [0.75, 1.05, 120.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2], (nu, 1)
                ),
                bias_prm=np.tile(
                    [0.75, 1.05, 120.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2], (nu, 1)
                ),
            )
            length_np = rng.uniform(0.11, 0.26, size=(batch, nu))
            vel_np = rng.uniform(-0.2, 0.2, size=(batch, nu))
            ctrl_np = rng.uniform(0.0, 1.0, size=(batch, nu))
            act_np = rng.uniform(0.05, 0.85, size=(batch, nu))

            dot_np = muscle_activation_dot(ctrl_np, act_np, params)
            dot_pt = muscle_activation_dot(
                torch.as_tensor(ctrl_np), torch.as_tensor(act_np), params
            ).numpy()
            np.testing.assert_allclose(dot_pt, dot_np, atol=1e-12, rtol=1e-12)

            next_np, _ = step_muscle_activation(
                act_np, ctrl_np, params, dt=0.002, substeps=5
            )
            next_pt, _ = step_muscle_activation(
                torch.as_tensor(act_np),
                torch.as_tensor(ctrl_np),
                params,
                dt=0.002,
                substeps=5,
            )
            np.testing.assert_allclose(next_pt.numpy(), next_np, atol=1e-12, rtol=1e-12)

            f_np = muscle_actuator_force(length_np, vel_np, act_np, params)
            f_pt = muscle_actuator_force(
                torch.as_tensor(length_np),
                torch.as_tensor(vel_np),
                torch.as_tensor(act_np),
                params,
            ).numpy()
            np.testing.assert_allclose(f_pt, f_np, atol=1e-12, rtol=1e-12)

            moment_t = torch.randn(batch, nu, nv, dtype=torch.float32) * 0.03

            class _MockGenesisEntity:
                def __init__(self) -> None:
                    self.last_force: torch.Tensor | None = None

                def get_dofs_position(self, dofs_idx_local: Any = None) -> torch.Tensor:
                    return torch.zeros(batch, nv)

                def get_dofs_velocity(self, dofs_idx_local: Any = None) -> torch.Tensor:
                    return torch.zeros(batch, nv)

                def control_dofs_force(
                    self, force: torch.Tensor, dofs_idx_local: Any = None
                ) -> None:
                    self.last_force = force

            def _geom_fn(
                q: torch.Tensor, qd: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return (
                    torch.full((q.shape[0], nu), 0.18),
                    torch.zeros((q.shape[0], nu)),
                    moment_t,
                )

            bridge = GenesisMuscleBridge(
                params=params, nv=nv, num_envs=batch, geometry_fn=_geom_fn
            )
            entity = _MockGenesisEntity()
            tau = bridge.step_actuation(entity, torch.zeros(batch, nu), dt=0.002)
            self.assertEqual(tuple(tau.shape), (batch, nv))
            self.assertIsNotNone(entity.last_force)
            assert entity.last_force is not None
            torch.testing.assert_close(tau, entity.last_force)


class TestRunnerConfigCli(unittest.TestCase):
    def test_configure_runner_and_dry_run_cli(self) -> None:
        cfg = configure_eden_rsl_runner(
            {"actor": {"hidden_dims": [256, 256]}, "critic": {}, "algorithm": {}},
            backbone="simba",
            algo="muon_ppo",
            hidden_dim=256,
        )
        self.assertEqual(cfg["actor"]["class_name"], "eden_rl.simba:SimBaModel")
        self.assertEqual(cfg["actor"]["hidden_dims"], [])
        self.assertEqual(cfg["algorithm"]["class_name"], "eden_rl.muon_ppo:MuonPPO")
        rc = train_main(["--dry-run", "--env", "myoHandReachFixed-v0"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
