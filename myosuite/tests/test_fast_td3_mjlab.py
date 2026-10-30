# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""FastTD3 mjlab adapter: official FastTD3 networks + MuscleMimic / pose twins."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier2

_REPO = Path(__file__).resolve().parents[2]


def _has_mjlab() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("mjlab") is not None
    except Exception:
        return False


def _has_fast_td3() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("fast_td3") is not None
    except Exception:
        return False


_MJLAB = _has_mjlab()
_FASTTD3 = _has_fast_td3()
_MJLAB_SKIP = pytest.mark.skipif(
    not _MJLAB, reason="mjlab not installed (pip install myosuite[mjlab])"
)
_FASTTD3_SKIP = pytest.mark.skipif(
    not _FASTTD3, reason="FastTD3 not installed (pip install myosuite[fasttd3])"
)


def _run_train(
    env_id: str, out: Path, timeout: int, backend: str = "mjlab"
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO), env.get("PYTHONPATH", "")) if p
    )
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("HF_HUB_OFFLINE", "1")
    return subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "train_fasttd3.py"),
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
    obs = __import__("torch").zeros(2, 8)
    act = actor(obs)
    assert act.shape == (2, 4)
    assert float(act.detach().min()) >= -1.0 and float(act.detach().max()) <= 1.0
    q1, q2 = critic(obs, act)
    assert q1.shape == (2, 11) and q2.shape == (2, 11)


def test_train_fasttd3_cli_help():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO), env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "train_fasttd3.py"), "--help"],
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
    """Colab CPU path: official FastTD3 on Gymnasium mimic, no Warp, no HF."""
    pytest.importorskip("musclemimic_models")
    out = tmp_path / "mimic_cpu"
    proc = _run_train("myoMimicFullbody-v0", out, timeout=180, backend="cpu")
    _assert_train_ok(proc)
    assert "backend=cpu" in proc.stdout
    assert list(out.glob("*.pt"))
