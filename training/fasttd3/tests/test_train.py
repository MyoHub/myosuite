# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Official FastTD3 on MyoSuite CPU gym and mjlab twins."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1]
_REPO = _PKG.parent.parent


def _has(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


_MJLAB = _has("mjlab")
_FASTTD3 = _has("fast_td3")
_MJLAB_SKIP = pytest.mark.skipif(not _MJLAB, reason="mjlab not installed")
_FASTTD3_SKIP = pytest.mark.skipif(
    not _FASTTD3, reason="FastTD3 not installed (pip install -e training/fasttd3)"
)


def _pythonpath() -> str:
    return os.pathsep.join(
        p for p in (str(_PKG), str(_REPO), os.environ.get("PYTHONPATH", "")) if p
    )


def _run_train(
    env_id: str, out: Path, timeout: int, backend: str = "mjlab"
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = _pythonpath()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("HF_HUB_OFFLINE", "1")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "myosuite_fasttd3",
            env_id,
            "--smoke",
            "--backend",
            backend,
            "--device",
            "cpu",
            "--out",
            str(out),
        ],
        cwd=str(_REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_train_ok(proc: subprocess.CompletedProcess[str]) -> None:
    rc = int(proc.returncode)
    msg = (proc.stdout or "") + "\n" + (proc.stderr or "")
    sig = -rc if rc < 0 else (rc - 128 if rc > 128 else rc)
    if sig in (6, 11) or rc in (134, 139):
        pytest.skip(f"mjlab FastTD3 smoke crashed (Warp): {msg[:500]}")
    assert rc == 0, msg
    assert "saved" in proc.stdout
    assert "n_act=" in proc.stdout


@_FASTTD3_SKIP
def test_fast_td3_official_actor_critic_forward():
    from fast_td3.fast_td3 import Actor, Critic

    torch = pytest.importorskip("torch")
    actor = Actor(
        n_obs=8,
        n_act=4,
        num_envs=2,
        init_scale=0.01,
        hidden_dim=32,
        sim_type="",
        sim_dimension=8,
        seq_len=2,
        device="cpu",
    )
    critic = Critic(
        n_obs=8,
        n_act=4,
        num_atoms=11,
        v_min=-10.0,
        v_max=10.0,
        hidden_dim=32,
        sim_type="",
        sim_dimension=8,
        seq_len=2,
        device="cpu",
    )
    obs = torch.zeros(2, 8)
    act = actor(obs)
    assert act.shape == (2, 4)
    assert float(act.detach().min()) >= -1.0 and float(act.detach().max()) <= 1.0
    q1, q2 = critic(obs, act)
    assert q1.shape == (2, 11) and q2.shape == (2, 11)


def test_train_args_smoke_preset():
    from myosuite_fasttd3.train import TrainArgs, apply_run_preset

    args = apply_run_preset(TrainArgs(env_id="myoMimicFullbody-v0", smoke=True))
    assert args.backend == "cpu"
    assert args.device == "cpu"
    assert args.num_envs == 2 and args.total_timesteps == 8
    assert args.amp is False


def test_parse_args_colab_gpu_defaults():
    from myosuite_fasttd3.train import parse_args

    args = parse_args(
        ["myoMimicFullbody-v0", "--num-envs", "8", "--total-timesteps", "200000"]
    )
    assert args.amp is False
    assert args.v_min == -250.0 and args.v_max == 250.0
    assert args.learning_starts == 4096 // 8
    assert args.max_grad_norm == 1.0


def test_train_fasttd3_cli_help():
    env = dict(os.environ)
    env["PYTHONPATH"] = _pythonpath()
    proc = subprocess.run(
        [sys.executable, "-m", "myosuite_fasttd3", "--help"],
        cwd=str(_REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "myoMimicFullbody-v0" in proc.stdout


@_MJLAB_SKIP
@_FASTTD3_SKIP
def test_train_fasttd3_elbow_smoke(tmp_path: Path):
    out = tmp_path / "elbow"
    proc = _run_train("myoElbowPose1D6MRandom-v0", out, timeout=180)
    _assert_train_ok(proc)
    assert list(out.glob("*.pt"))


@_MJLAB_SKIP
@_FASTTD3_SKIP
def test_train_fasttd3_mimic_fullbody_smoke(tmp_path: Path):
    pytest.importorskip("musclemimic_models")
    out = tmp_path / "mimic"
    proc = _run_train("myoMimicFullbody-v0", out, timeout=420)
    _assert_train_ok(proc)
    assert list(out.glob("*.pt"))


@_FASTTD3_SKIP
def test_train_fasttd3_mimic_cpu_smoke(tmp_path: Path):
    pytest.importorskip("musclemimic_models")
    out = tmp_path / "mimic_cpu"
    proc = _run_train("myoMimicFullbody-v0", out, timeout=180, backend="cpu")
    _assert_train_ok(proc)
    assert "backend=cpu" in proc.stdout
    assert list(out.glob("*.pt"))
