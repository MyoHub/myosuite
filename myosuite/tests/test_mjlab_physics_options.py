# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab configs must simulate with the CPU model's physics options.

mjlab drops the ``<option>`` of every entity spec when it attaches it to the
scene and applies its own ``MujocoCfg`` instead (default integrator
``implicitfast``), so each mjlab config has to carry the options of the CPU
model of the same env id (cross-backend contract, invariant 4).
"""

from __future__ import annotations

import importlib.util
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import pytest

pytest.importorskip("mjlab")

pytestmark = pytest.mark.tier2

from mjlab.managers.observation_manager import ObservationGroupCfg  # noqa: E402
from mjlab.scene import Scene  # noqa: E402
from mjlab.tasks.registry import list_tasks, load_env_cfg  # noqa: E402

import myosuite  # noqa: E402, F401
from myosuite.core.config import TaskConfig  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import (  # noqa: E402
    mujoco_cfg_from_model,
)
from myosuite import make_env  # noqa: E402

# Not a twin yet: obs, control step and terrain differ from CPU (fixed separately).
_NOT_TWINS = {"myoChallengeChaseTagFBP2-v0"}

Opt = dict[str, np.ndarray]


def _opt_values(model: mujoco.MjModel) -> Opt:
    """Copy of every ``model.opt`` field (integrator, solver, flags, ...)."""
    opt = model.opt
    return {
        name: np.array(getattr(opt, name))
        for name in dir(opt)
        if not name.startswith("_") and not callable(getattr(opt, name))
    }


def _applied_opt(cfg: Any) -> Opt:
    """``model.opt`` mjlab simulates with: MuJoCo defaults + ``MujocoCfg.apply``."""
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    cfg.sim.mujoco.apply(model)
    return _opt_values(model)


def _opt_diff(cpu: Opt, sim: Opt) -> dict[str, tuple]:
    """Every option that differs, as ``{name: (cpu, mjlab)}``."""
    return {
        name: (cpu[name].tolist(), sim[name].tolist())
        for name in cpu
        if not np.array_equal(cpu[name], sim[name])
    }


def _cpu_opt(env_id: str) -> Opt:
    env = make_env(env_id)
    try:
        return _opt_values(env.unwrapped.model)
    finally:
        env.close()


def _mimic_cfgs() -> dict[str, tuple[Any, Any]]:
    """Mimic twins (registered on demand, so absent from the default registry)."""
    if importlib.util.find_spec("musclemimic_models") is None:
        return {}
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (  # noqa: PLC0415
        register_mimic_mjlab_tasks,
    )

    cfgs: dict[str, tuple[Any, Any]] = {}

    def _register(*, task_id: str, env_cfg: Any, play_env_cfg: Any, **_: Any) -> None:
        cfgs[task_id] = (env_cfg, play_env_cfg)

    register_mimic_mjlab_tasks(_register, rl_cfg_fn=lambda: None)
    assert {"myoMimicBimanual-v0", "myoMimicFullbody-v0"} <= set(cfgs)
    return cfgs


def test_every_twin_uses_cpu_physics_options() -> None:
    """Each id registered on both backends simulates with the CPU ``model.opt``."""
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415 (registers twins)

    cfgs = {
        env_id: (load_env_cfg(env_id), load_env_cfg(env_id, play=True))
        for env_id in list_tasks()
        if env_id in gym.registry and env_id not in _NOT_TWINS
    }
    cfgs.update(_mimic_cfgs())
    # Guard against a silently partial registry.
    assert len(cfgs) > 100
    assert {
        "myoElbowPose1D6MFixed-v0",
        "myoLegWalk-v0",
        "myoChallengeTableTennisP0-v0",
        "myoChallengeTableTennisP2-v0",
    } <= set(cfgs)

    mismatches: dict[str, dict] = {}
    for env_id, (train, play) in sorted(cfgs.items()):
        cpu = _cpu_opt(env_id)
        diff = _opt_diff(cpu, _applied_opt(train))
        diff.update(
            {f"play.{k}": v for k, v in _opt_diff(cpu, _applied_opt(play)).items()}
        )
        if diff:
            mismatches[env_id] = diff
    assert not mismatches, "mjlab physics options differ from CPU (cpu, mjlab):\n" + (
        "\n".join(f"{k}: {v}" for k, v in mismatches.items())
    )


def test_scene_options_come_only_from_mujoco_cfg() -> None:
    """The compiled scene ignores entity ``<option>``s (what ``_applied_opt`` assumes)."""
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415 (registers twins)

    cfg = load_env_cfg("myoChallengeTableTennisP0-v0")
    model = Scene(cfg.scene, device="cpu").compile()
    cfg.sim.mujoco.apply(model)
    assert not _opt_diff(_applied_opt(cfg), _opt_values(model))


@pytest.mark.parametrize("variant", ["bimanual", "fullbody"])
def test_sar_mimic_cfgs_use_cpu_physics_options(variant: str) -> None:
    """The synergy-action mimic configs simulate the CPU MuscleMimic physics."""
    pytest.importorskip("musclemimic_models")
    from ml_collections import config_dict  # noqa: PLC0415

    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mm  # noqa: PLC0415
    from myosuite.integrations.musclemimic import (  # noqa: PLC0415
        bimanual_model,
        fullbody_model,
    )

    if variant == "bimanual":
        build = bimanual_model.build_mimic_bimanual_spec
        cfg = bimanual_model.default_mimic_config()
    else:
        build = fullbody_model.build_mimic_fullbody_spec
        cfg = fullbody_model.default_mimic_fullbody_config()

    def spec_fn() -> mujoco.MjSpec:
        return build(config_dict.create(**dict(cfg)))[0]

    model = spec_fn().compile()
    common = dict(
        entity_name="robot",
        spec_fn=spec_fn,
        muscle_actuators=mm._muscle_actuator_names(model),
        tendon_targets=mm._muscle_tendon_names(model),
        sar_transform=None,
        sim_dt=float(cfg.sim_dt),
        ctrl_dt=float(cfg.ctrl_dt),
        max_episode_steps=10,
    )
    env_cfgs = [mm._make_mimic_sar_env_cfg(_task_id="sar", variant=variant, **common)]
    if variant == "fullbody":
        env_cfgs.append(mm._make_directional_sar_env_cfg(_task_id="walk", **common))
    cpu = _cpu_opt(f"myoMimic{variant.capitalize()}-v0")
    for env_cfg in env_cfgs:
        assert not _opt_diff(cpu, _applied_opt(env_cfg))


def test_elbow_test_cfg_uses_cpu_physics_options() -> None:
    """The hand-built elbow config used by the builder tests matches CPU too."""
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (  # noqa: PLC0415
        _make_elbow_env_cfg,
    )

    cpu = _cpu_opt("myoElbowPose1D6MFixed-v0")
    assert not _opt_diff(cpu, _applied_opt(_make_elbow_env_cfg()))


def test_task_builder_default_carries_model_options() -> None:
    """Without ``sim_cfg`` the factory copies the model's ``<option>``."""
    from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (  # noqa: PLC0415
        mjlab_env_cfg_from_task_config,
    )
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (  # noqa: PLC0415
        _elbow_spec_fn,
    )

    def spec_fn() -> mujoco.MjSpec:
        spec = _elbow_spec_fn()
        spec.option.timestep = 0.001
        spec.option.iterations = 7
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        return spec

    cfg = mjlab_env_cfg_from_task_config(
        cfg=TaskConfig(max_episode_steps=10),
        spec_fn=spec_fn,
        entity_name="elbow",
        actuators=(),
        observations={"policy": ObservationGroupCfg(terms={})},
        actions={},
        decimation=5,
    )
    assert not _opt_diff(_opt_values(spec_fn().compile()), _applied_opt(cfg))
    assert cfg.episode_length_s == pytest.approx(10 * 5 * 0.001)


def test_mujoco_cfg_from_model_rejects_a_different_timestep() -> None:
    """A config whose decimation assumes another timestep is an error, not a drift."""
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002"/></mujoco>'
    )
    assert mujoco_cfg_from_model(model, timestep=0.002).timestep == 0.002
    with pytest.raises(ValueError, match="timestep"):
        mujoco_cfg_from_model(model, timestep=0.001)
