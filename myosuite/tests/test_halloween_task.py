# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke tests for MyoHalloween (MyoFullBody + CC0 Kenney pumpkin)."""

from __future__ import annotations

import mujoco
import pytest

from myosuite.envs.myo.tasks.halloween.halloween_model import (
    ASSET_DIR,
    PUMPKIN_BODY,
    PUMPKIN_JOINT,
    PUMPKIN_OBJ,
    compile_halloween_model,
    park_pumpkin,
    pumpkin_body_contact_frames,
    throw_pumpkin,
)

pytestmark = pytest.mark.tier2


@pytest.fixture(scope="module")
def halloween_model():
    model, spec, label = compile_halloween_model()
    assert "halloween" in label
    return model, spec


def test_kenney_pumpkin_asset_present():
    assert PUMPKIN_OBJ.is_file(), PUMPKIN_OBJ
    license_file = ASSET_DIR / "LICENSE_Kenney_FoodKit_CC0.txt"
    assert license_file.is_file()
    text = license_file.read_text(encoding="utf-8")
    assert "CC0" in text or "Creative Commons Zero" in text
    assert (ASSET_DIR / "Textures" / "colormap.png").is_file()


def test_halloween_model_dimensions(halloween_model):
    model, _spec = halloween_model
    # Reduced MyoFullBody (89/88) + free pumpkin (7/6).
    assert (model.nq, model.nv) == (96, 94)
    assert model.nu == 354
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PUMPKIN_BODY) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PUMPKIN_JOINT) >= 0
    # Robot free joint remains at qpos address 0.
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root < 0:
        # Prefixed when compiled through mjlab attach; bare when compiled alone.
        root = next(
            i
            for i in range(model.njnt)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").endswith(
                "root"
            )
            or i == 0
        )
    assert int(model.jnt_qposadr[0]) == 0


def test_pumpkin_throw_produces_body_contacts(halloween_model):
    model, _spec = halloween_model
    data = mujoco.MjData(model)
    stats = pumpkin_body_contact_frames(model, data, steps=500, side=-1.0, speed=1.0)
    assert stats["contact_frames"] >= 5, stats
    assert stats["body_contact_frames"] >= 1, stats


def test_park_and_throw_helpers(halloween_model):
    model, _spec = halloween_model
    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    park_pumpkin(model, data)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PUMPKIN_JOINT)
    adr = int(model.jnt_qposadr[jid])
    assert float(data.qpos[adr]) > 5.0
    throw_pumpkin(model, data, side=1.0, speed=1.2)
    assert float(data.qpos[adr + 1]) > float(data.qpos[1])  # from the right


def test_register_myohalloween_mjlab_task():
    pytest.importorskip("mjlab")
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    from myosuite.envs.myo.backends.mjlab.register_mjlab_halloween import (
        register_halloween_mjlab_task,
    )

    register_halloween_mjlab_task()
    assert "myoHalloween-v0" in list_tasks()
    cfg = load_env_cfg("myoHalloween-v0", play=True)
    assert cfg.decimation >= 1
    assert "muscles" in cfg.actions
    assert "pumpkin_rel" in cfg.observations["policy"].terms
    assert "throw_pumpkin" in cfg.events
