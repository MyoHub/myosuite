from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import myosuite.utils.onnx_checkpoint as onnx_checkpoint
from myosuite.utils.onnx_checkpoint import (
    bundle_onnx_with_checkpoint,
    extract_checkpoint_from_onnx,
    get_env_fatigue_state,
    get_wandb_onnx_checkpoint_path,
    normalize_onnx_checkpoint_name,
    read_onnx_checkpoint_metadata,
    set_env_fatigue_state,
)
from myosuite import make_env


def test_onnx_checkpoint_bundle_round_trip() -> None:
    pytest.importorskip("onnx")

    class _Tiny(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return obs + 1.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        onnx_path = tmp / "policy.onnx"
        ckpt_path = tmp / "policy_state.bin"
        ckpt_path.write_bytes(b"resume-me")
        torch.onnx.export(
            _Tiny(),
            torch.zeros(1, 3, dtype=torch.float32),
            str(onnx_path),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )

        bundle_onnx_with_checkpoint(
            onnx_path=onnx_path,
            checkpoint_path=ckpt_path,
            framework="unit-test",
            metadata={"step": 123},
        )

        meta = read_onnx_checkpoint_metadata(onnx_path)
        assert meta["framework"] == "unit-test"
        assert meta["metadata"]["step"] == 123

        extracted, extracted_meta, temp_dir = extract_checkpoint_from_onnx(onnx_path)
        try:
            assert extracted.read_bytes() == b"resume-me"
            assert extracted_meta["checkpoint_name"] == "policy_state.bin"
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


def test_normalize_onnx_checkpoint_name_accepts_pt_aliases() -> None:
    assert normalize_onnx_checkpoint_name("model_42.pt") == "model_42.onnx"
    assert normalize_onnx_checkpoint_name("model_final.pt") == "model_final.onnx"
    assert normalize_onnx_checkpoint_name("custom.onnx") == "custom.onnx"


def test_get_wandb_onnx_checkpoint_path_prefers_final_and_normalizes_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("wandb")

    class _FakeFile:
        def __init__(self, name: str) -> None:
            self.name = name

        def download(self, root: str, replace: bool = True) -> None:
            del replace
            target = Path(root) / self.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.name.encode("utf-8"))

    class _FakeRun:
        def __init__(self, names: list[str]) -> None:
            self._files = [_FakeFile(name) for name in names]

        def files(self) -> list[_FakeFile]:
            return list(self._files)

        def file(self, name: str) -> _FakeFile:
            for file in self._files:
                if file.name == name:
                    return file
            raise KeyError(name)

    class _FakeApi:
        def __init__(self, run: _FakeRun) -> None:
            self._run = run

        def run(self, path: str) -> _FakeRun:
            assert path == "org/project/run-id"
            return self._run

    fake_run = _FakeRun(["model_10.onnx", "model_20.onnx", "model_final.onnx"])
    monkeypatch.setattr(onnx_checkpoint.wandb, "Api", lambda: _FakeApi(fake_run))

    latest_path, was_cached = get_wandb_onnx_checkpoint_path(
        tmp_path, Path("org/project/run-id")
    )
    assert latest_path.name == "model_final.onnx"
    assert was_cached is False
    assert latest_path.read_bytes() == b"model_final.onnx"

    aliased_path, was_cached = get_wandb_onnx_checkpoint_path(
        tmp_path,
        Path("org/project/run-id"),
        checkpoint_name="model_20.pt",
    )
    assert aliased_path.name == "model_20.onnx"
    assert was_cached is False
    assert aliased_path.read_bytes() == b"model_20.onnx"


_FATIGUE_ENV_ID = "myoFatiElbowPose1D6MRandom-v0"


def _fatigued_env(n_steps: int):
    """CPU fatigue env (state in ``muscle_fatigue``) after ``n_steps`` of full excitation."""

    env = make_env(_FATIGUE_ENV_ID)
    env.reset(seed=0)
    for _ in range(n_steps):
        env.step(env.action_space.high)
    return env


def test_cpu_fatigue_state_round_trips_through_gym_wrappers() -> None:
    """MyoGymnasiumEnv fatigue used to be invisible (only mjlab / ModularTaskEnv)."""
    state = get_env_fatigue_state(_fatigued_env(20))
    assert state is not None and set(state) == {"cpu"}
    assert max(state["cpu"]["MF"]) > 0.0
    fresh = _fatigued_env(0)
    assert get_env_fatigue_state(fresh) != state
    set_env_fatigue_state(fresh, state)
    assert get_env_fatigue_state(fresh) == state


def test_cpu_fatigue_state_through_sb3_vec_envs() -> None:
    """OnnxCheckpointCallback falls back to model.env: a (VecNormalize-wrapped) VecEnv."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec = VecNormalize(
        DummyVecEnv([lambda: _fatigued_env(10), lambda: _fatigued_env(30)])
    )
    state = get_env_fatigue_state(vec)
    assert state is not None and len(state["cpu"]) == 2
    assert not np.allclose(state["cpu"][0]["MF"], state["cpu"][1]["MF"])

    fresh = VecNormalize(DummyVecEnv([lambda: _fatigued_env(0)] * 2))
    set_env_fatigue_state(fresh, state)
    assert get_env_fatigue_state(fresh) == state
