# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the checkpoint manifest and the env contract hash."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myosuite.utils import checkpoint_manifest as cm

pytestmark = pytest.mark.tier1

CONTRACT = {"obs_shape": [9], "action_shape": [6], "ctrl_dt": 0.02}


def test_contract_hash_is_stable_and_sensitive() -> None:
    """The hash ignores key order and changes with every field of the contract."""
    assert cm.contract_hash(CONTRACT) == cm.contract_hash(
        dict(reversed(CONTRACT.items()))
    )
    assert len(cm.contract_hash(CONTRACT)) == 12
    for key, value in (("obs_shape", [10]), ("action_shape", [7]), ("ctrl_dt", 0.01)):
        assert cm.contract_hash({**CONTRACT, key: value}) != cm.contract_hash(CONTRACT)


def test_env_contract_of_a_registered_env() -> None:
    """The contract of the elbow pose env records its observation, action and control step."""
    contract = cm.env_contract("myoElbowPose1D6MFixed-v0")
    assert contract["obs_shape"] == [9]
    assert contract["action_shape"] == [6]
    assert contract["ctrl_dt"] == pytest.approx(0.02)
    assert cm.contract_hash(contract) == cm.contract_hash(
        cm.env_contract("myoElbowPose1D6MFixed-v0")
    )


def test_manifest_roundtrip(tmp_path: Path) -> None:
    """write_manifest records the contract, the version and the files."""
    (tmp_path / "model_200.pt").write_bytes(b"")
    path = cm.write_manifest(
        tmp_path, "myoElbowPose1D6MRandom-v0", contract=CONTRACT, success=100.0
    )
    assert path == tmp_path / "manifest.json"
    manifest = cm.read_manifest(tmp_path)
    assert manifest["contract_hash"] == cm.contract_hash(CONTRACT)
    assert manifest["files"] == ["model_200.pt"]
    assert manifest["success_percent"] == 100.0
    assert json.loads(path.read_text())["env_id"] == "myoElbowPose1D6MRandom-v0"
    assert cm.read_manifest(tmp_path / "missing") is None


def test_check_contract_warns_on_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed env contract warns; an unchanged one or a missing manifest does not."""
    assert (
        cm.check_contract("myoElbowPose1D6MFixed-v0", tmp_path) is True
    )  # no manifest
    cm.write_manifest(tmp_path, "myoElbowPose1D6MFixed-v0", contract=CONTRACT)
    monkeypatch.setattr(cm, "env_contract", lambda env_id: CONTRACT)
    assert cm.check_contract("myoElbowPose1D6MFixed-v0", tmp_path) is True
    monkeypatch.setattr(
        cm, "env_contract", lambda env_id: {**CONTRACT, "obs_shape": [10]}
    )
    with pytest.warns(UserWarning, match="may not work"):
        assert cm.check_contract("myoElbowPose1D6MFixed-v0", tmp_path) is False
