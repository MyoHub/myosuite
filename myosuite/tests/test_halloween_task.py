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

pytestmark = pytest.mark.tier1


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
    assert cfg.actions["muscles"].action_mode == "excitation"
    assert "pumpkin_rel" in cfg.observations["policy"].terms
    assert "throw_pumpkin" in cfg.events


def test_cpu_registration_seed_and_contract():
    import numpy as np

    from myosuite import make_env

    env = make_env("myoHalloween-v0")
    try:
        first, _ = env.reset(seed=123)
        deadline = env.unwrapped._next_throw
        second, _ = env.reset(seed=123)
        np.testing.assert_array_equal(first, second)
        assert deadline == env.unwrapped._next_throw
        assert first.shape == (535,) and first.dtype == np.float32
        assert env.action_space.shape == (354,)
        assert env.spec.max_episode_steps == 600
        result = env.step(-np.ones(354, dtype=np.float32))
        assert np.all(env.unwrapped.data.ctrl == 0)
        assert len(result) == 5 and np.isfinite(result[0]).all()
        assert result[1] > 0 and not result[2]
        assert env.unwrapped.data.time == pytest.approx(0.01)
    finally:
        env.close()


def test_proximity_does_not_count_as_hit():
    import numpy as np

    from myosuite import make_env
    from myosuite.envs.myo.tasks.halloween.halloween_model import pumpkin_qposadr

    env = make_env("myoHalloween-v0")
    try:
        env.reset(seed=1)
        raw = env.unwrapped
        # Close to a link origin, but outside the physical body geometry.
        qa = pumpkin_qposadr(raw.model)
        raw.data.qpos[qa : qa + 3] = raw.data.xpos[raw._head] + [0, 0.19, 0]
        mujoco.mj_forward(raw.model, raw.data)
        raw._hit = False
        for c in raw.data.contact:
            assert not (
                (
                    c.geom1 == raw._gid
                    and raw.model.geom_bodyid[c.geom2] in raw._robot_bodies
                )
                or (
                    c.geom2 == raw._gid
                    and raw.model.geom_bodyid[c.geom1] in raw._robot_bodies
                )
            )
        _, _, _, _, stats = env.step(np.zeros(354))
        assert not stats["hit"]
        # The hit signal must come from a real contact in step(), not link distance.
        assert (
            np.linalg.norm(raw.data.xpos[raw._head] - raw.data.qpos[qa : qa + 3]) < 0.20
        )
    finally:
        env.close()


def test_cpu_physical_contact_terminates():
    import numpy as np

    from myosuite import make_env
    from myosuite.envs.myo.tasks.halloween.halloween_model import pumpkin_qposadr

    env = make_env("myoHalloween-v0")
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        qa = pumpkin_qposadr(raw.model)
        raw.data.qpos[qa : qa + 3] = raw.data.geom("hat_skull_coll").xpos
        mujoco.mj_forward(raw.model, raw.data)
        _, _, terminated, _, info = env.step(np.zeros(354))
        assert terminated and info["hit"] and info["body_contacts"] > 0
    finally:
        env.close()


def test_mjlab_matches_cpu_physics_and_contact_sensor():
    pytest.importorskip("mjlab")
    from myosuite.envs.myo.backends.mjlab.register_mjlab_halloween import (
        make_halloween_env_cfg,
    )

    cfg = make_halloween_env_cfg()
    assert cfg.sim.mujoco.integrator == "euler"
    assert cfg.actions["muscles"].action_mode == "excitation"
    assert cfg.decimation == 5 and cfg.episode_length_s == 6.0
    (sensor,) = cfg.scene.sensors
    assert sensor.secondary.entity == "halloween_fullbody"
    assert sensor.history_length == cfg.decimation
    assert sensor.fields == ("found", "force")


def test_mjlab_hit_uses_substep_history_not_proximity():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    from types import SimpleNamespace

    from myosuite.envs.myo.backends.mjlab.register_mjlab_halloween import _term_hit

    forces = torch.zeros(2, 1, 5, 3)
    forces[0, 0, 2, 0] = 1.0
    env = SimpleNamespace(
        scene={
            "pumpkin_body_contact": SimpleNamespace(
                data=SimpleNamespace(force_history=forces)
            )
        }
    )
    assert _term_hit(env).tolist() == [True, False]


def test_mjlab_scene_compiles_with_filtered_contact_sensors():
    pytest.importorskip("mjlab")
    from mjlab.scene import Scene

    from myosuite.envs.myo.backends.mjlab.register_mjlab_halloween import (
        make_halloween_env_cfg,
    )

    cfg = make_halloween_env_cfg()
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    assert (model.nq, model.nv, model.nu) == (96, 94, 354)
    assert model.sensor("pumpkin_body_contact_pumpkin_collision_force").dim[0] == 3
    assert model.npair > 0
