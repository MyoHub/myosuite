"""Checkpoint load guards and legacy manifest compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myosuite import make_env
from myosuite.utils import checkpoint_manifest as cm
from myosuite.utils.deployment_manifest import (
    env_deployment_manifest,
    model_identity,
    validate_policy_manifest,
    write_deployment_manifest,
)

pytestmark = pytest.mark.tier1


def test_new_checkpoint_guard_precedes_deserialization_and_legacy_stays_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from myosuite.utils import checkpoint_utils

    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"test fixture")
    with make_env("myoElbowPose1D6MFixed-v0") as env:
        # Schema 1 remains usable without new metadata.
        (tmp_path / cm.MANIFEST_NAME).write_text(
            json.dumps({"schema": 1, "contract": {}})
        )
        assert validate_policy_manifest(checkpoint, env) == []
        manifest = env_deployment_manifest(env, artifacts={checkpoint.name: checkpoint})
        write_deployment_manifest(checkpoint, manifest)
        assert validate_policy_manifest(checkpoint, env) == []
        called = []
        monkeypatch.setattr(
            checkpoint_utils, "load_sb3_model", lambda *a: called.append(a)
        )
        checkpoint.write_bytes(b"different checkpoint with same shape")
        with pytest.raises(ValueError, match="artifact policy.zip"):
            checkpoint_utils.load_policy(env, checkpoint)
        assert not called
        checkpoint.write_bytes(b"test fixture")
        env.unwrapped.model.body_mass[1] *= 2
        with pytest.raises(ValueError, match="model"):
            validate_policy_manifest(checkpoint, env)


def test_writer_records_actual_configured_model_and_control_step(
    tmp_path: Path,
) -> None:
    from myosuite.core.config import EnvConfig

    (tmp_path / "model_1.pt").write_bytes(b"fixture")
    config = tmp_path / "params" / "agent.yaml"
    config.parent.mkdir()
    config.write_text("actor:\n  activation: elu\n")
    with make_env(EnvConfig("myoElbowPose1D6MFixed-v0", ctrl_dt=0.01)) as env:
        env.unwrapped.model.body_mass[1] *= 2
        cm.write_manifest(tmp_path, "myoElbowPose1D6MFixed-v0", env=env)
        manifest = cm.read_manifest(tmp_path)
        assert manifest["schema"] == 2
        assert manifest["contract"]["ctrl_dt"] == 0.01
        assert manifest["deployment"]["model"] == model_identity(env.unwrapped.model)
        assert validate_policy_manifest(tmp_path / "model_1.pt", env) == []
        config.write_text("actor:\n  activation: relu\n")
        with pytest.raises(ValueError, match="artifact params/agent.yaml"):
            validate_policy_manifest(tmp_path / "model_1.pt", env)
