# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MuscleMimic checkpoints restore whatever CPU device name they were saved with."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from myosuite.integrations.musclemimic.fullbody_local_policy import _restore_on_cpu

pytestmark = pytest.mark.tier2


def _save_with_foreign_device(path: Path) -> dict[str, np.ndarray]:
    """Save a small checkpoint, then rename its device as another JAX would."""
    jnp = pytest.importorskip("jax.numpy")
    ocp = pytest.importorskip("orbax.checkpoint")
    tree = {"params": {"w": np.arange(6.0).reshape(2, 3)}, "step": np.asarray(3)}
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        path,
        {"params": {"w": jnp.asarray(tree["params"]["w"])}, "step": jnp.asarray(3)},
    )
    checkpointer.wait_until_finished()
    sharding = path / "_sharding"
    entries = json.loads(sharding.read_text())
    for key, value in entries.items():
        meta = json.loads(value)
        meta["device_str"] = "NOT_A_DEVICE_7"
        entries[key] = json.dumps(meta)
    sharding.write_text(json.dumps(entries))
    return tree


def test_plain_restore_fails_on_a_foreign_device_name(tmp_path: Path) -> None:
    import orbax.checkpoint as ocp

    _save_with_foreign_device(tmp_path / "ckpt")
    with pytest.raises(ValueError, match="NOT_A_DEVICE_7"):
        ocp.StandardCheckpointer().restore(str(tmp_path / "ckpt"))


def test_restore_on_cpu_ignores_the_saved_device_name(tmp_path: Path) -> None:
    import orbax.checkpoint as ocp

    tree = _save_with_foreign_device(tmp_path / "ckpt")
    restored = _restore_on_cpu(ocp.StandardCheckpointer(), str(tmp_path / "ckpt"))
    np.testing.assert_array_equal(
        np.asarray(restored["params"]["w"]), tree["params"]["w"]
    )
    assert int(np.asarray(restored["step"])) == 3
