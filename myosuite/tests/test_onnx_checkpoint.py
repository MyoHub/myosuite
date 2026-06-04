from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import myosuite.utils.onnx_checkpoint as onnx_checkpoint
from myosuite.utils.onnx_checkpoint import (
    bundle_onnx_with_checkpoint,
    extract_checkpoint_from_onnx,
    get_wandb_onnx_checkpoint_path,
    normalize_onnx_checkpoint_name,
    read_onnx_checkpoint_metadata,
)


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


def _assert_nested_equal(left: object, right: object, path: str = "root") -> None:
    if type(left) is not type(right):
        raise AssertionError(
            f"type mismatch at {path}: {type(left).__name__} != {type(right).__name__}"
        )
    if torch.is_tensor(left):
        assert torch.is_tensor(right)
        assert left.shape == right.shape, f"shape mismatch at {path}"
        assert left.dtype == right.dtype, f"dtype mismatch at {path}"
        assert torch.equal(
            left.detach().cpu(), right.detach().cpu()
        ), f"tensor mismatch at {path}"
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left.keys()) == set(right.keys()), f"key mismatch at {path}"
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right), f"length mismatch at {path}"
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_equal(left_item, right_item, f"{path}[{idx}]")
        return
    assert left == right, f"value mismatch at {path}: {left!r} != {right!r}"


def test_mjlab_onnx_runner_restore_matches_pt_restore() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("mjlab")
    pytest.importorskip("tensordict")

    import myosuite
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from tensordict import TensorDict

    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        bootstrap_myosuite_mjlab_registry,
    )
    from tutorials.mc26.mc26_train_saber_mjlab import (
        OnnxCheckpointingMjlabRunner,
        _prepare_config,
    )

    myosuite.register_all_envs()
    bootstrap_myosuite_mjlab_registry()
    configure_torch_backends()

    cfg = _prepare_config(
        argparse.Namespace(
            task_id="myoChallengeSaberP0-v0",
            num_envs=2,
            num_steps_per_env=4,
            max_iterations=1,
            save_interval=1,
            seed=7,
            device="cpu",
            run_name="",
            resume_from=None,
            log_root=Path("logs") / "rsl_rl" / "saber_mjlab",
        )
    )

    def make_runner() -> (
        tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper, OnnxCheckpointingMjlabRunner]
    ):
        env = ManagerBasedRlEnv(cfg=cfg.env, device="cpu")
        vec_env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
        runner = OnnxCheckpointingMjlabRunner(
            vec_env,
            asdict(cfg.agent),
            None,
            "cpu",
            task_id="myoChallengeSaberP0-v0",
        )
        return env, vec_env, runner

    obs_sample = torch.linspace(-1.0, 1.0, steps=632, dtype=torch.float32).reshape(
        1, -1
    )
    obs_td = TensorDict({"actor": obs_sample}, batch_size=[1])

    with tempfile.TemporaryDirectory(prefix="onnx-restore-check-") as tmp_dir:
        tmp = Path(tmp_dir)
        env0, vec0, runner0 = make_runner()
        try:
            runner0.current_learning_iteration = 17
            vec0.unwrapped.common_step_counter = 1234
            actor_before = runner0.alg.actor(obs_td).detach().cpu()
            saved_dict = runner0.alg.save()
            saved_dict["iter"] = runner0.current_learning_iteration
            saved_dict["infos"] = {
                "env_state": {"common_step_counter": vec0.unwrapped.common_step_counter}
            }
            pt_path = tmp / "control.pt"
            torch.save(saved_dict, pt_path)
            runner0.save(str(tmp / "bundle.pt"))
            onnx_path = tmp / "bundle.onnx"
        finally:
            env0.close()

        env_pt, vec_pt, runner_pt = make_runner()
        try:
            infos_pt = runner_pt.load(str(pt_path), map_location="cpu")
            actor_pt = runner_pt.alg.actor(obs_td).detach().cpu()
            state_pt = runner_pt.alg.save()
        finally:
            env_pt.close()

        env_onnx, vec_onnx, runner_onnx = make_runner()
        try:
            infos_onnx = runner_onnx.load_onnx(onnx_path)
            actor_onnx = runner_onnx.alg.actor(obs_td).detach().cpu()
            state_onnx = runner_onnx.alg.save()
        finally:
            env_onnx.close()

    _assert_nested_equal(saved_dict["actor_state_dict"], state_pt["actor_state_dict"])
    _assert_nested_equal(saved_dict["critic_state_dict"], state_pt["critic_state_dict"])
    _assert_nested_equal(
        saved_dict["optimizer_state_dict"], state_pt["optimizer_state_dict"]
    )
    _assert_nested_equal(saved_dict["actor_state_dict"], state_onnx["actor_state_dict"])
    _assert_nested_equal(
        saved_dict["critic_state_dict"], state_onnx["critic_state_dict"]
    )
    _assert_nested_equal(
        saved_dict["optimizer_state_dict"], state_onnx["optimizer_state_dict"]
    )
    assert runner_pt.current_learning_iteration == 17
    assert runner_onnx.current_learning_iteration == 17
    assert int(vec_pt.unwrapped.common_step_counter) == 1234
    assert int(vec_onnx.unwrapped.common_step_counter) == 1234
    assert torch.equal(actor_before, actor_pt)
    assert torch.equal(actor_before, actor_onnx)
    assert infos_pt == infos_onnx == {"env_state": {"common_step_counter": 1234}}


def test_mjlab_onnx_export_matches_runner_actor_output() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("mjlab")
    pytest.importorskip("tensordict")

    import myosuite
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from tensordict import TensorDict

    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        bootstrap_myosuite_mjlab_registry,
    )
    from myosuite.integrations.musclemimic.actor_onnx import load_onnx_session
    from tutorials.mc26.mc26_train_saber_mjlab import (
        OnnxCheckpointingMjlabRunner,
        _prepare_config,
    )

    myosuite.register_all_envs()
    bootstrap_myosuite_mjlab_registry()
    configure_torch_backends()

    cfg = _prepare_config(
        argparse.Namespace(
            task_id="myoChallengeSaberP0-v0",
            num_envs=2,
            num_steps_per_env=4,
            max_iterations=1,
            save_interval=1,
            seed=7,
            device="cpu",
            run_name="",
            resume_from=None,
            log_root=Path("logs") / "rsl_rl" / "saber_mjlab",
        )
    )

    env = ManagerBasedRlEnv(cfg=cfg.env, device="cpu")
    vec_env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner = OnnxCheckpointingMjlabRunner(
        vec_env,
        asdict(cfg.agent),
        None,
        "cpu",
        task_id="myoChallengeSaberP0-v0",
    )
    try:
        obs_sample = torch.linspace(-1.0, 1.0, steps=632, dtype=torch.float32).reshape(
            1, -1
        )
        obs_td = TensorDict({"actor": obs_sample}, batch_size=[1])
        torch_actor_out = runner.alg.actor(obs_td).detach().cpu()

        with tempfile.TemporaryDirectory(prefix="onnx-export-check-") as tmp_dir:
            onnx_path = Path(tmp_dir) / "actor.onnx"
            runner._export_actor_onnx(onnx_path)  # noqa: SLF001
            session = load_onnx_session(onnx_path)
            onnx_out = session.act_torch(obs_sample).detach().cpu()

        assert torch.allclose(torch_actor_out, onnx_out, rtol=1e-4, atol=1e-5)
    finally:
        env.close()
