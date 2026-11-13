"""The optional browser adapter refuses changes to the trained action contract."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

pytest.importorskip("mjswan")
pytest.importorskip("mjlab")
pytestmark = pytest.mark.tier1

from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import MyoActionCfg  # noqa: E402
from scripts.build_browser_walk import browser_actions  # noqa: E402


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        actions={
            "muscles": MyoActionCfg(
                entity_name="robot", action_range=(0, 1), muscle_sigmoid=False
            )
        }
    )


def test_walk_adapter_uses_native_excitation_and_order() -> None:
    model = SimpleNamespace(actuator_dyntype=np.full(3, mujoco.mjtDyn.mjDYN_MUSCLE))
    adapted = browser_actions(_config(), model)["muscles"].to_dict()
    assert adapted == {
        "type": "muscle_activation",
        "mode": "excitation",
        "actuator_names": ["robot/.*"],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("muscle_sigmoid", True),
        ("normalize_act", False),
        ("action_range", (-1, 1)),
        ("muscle_fatigue", True),
        ("reroute", (0, 1)),
        ("excitation_stages", (lambda: None,)),
        ("clip", {".*": (0.2, 0.8)}),
    ],
)
def test_walk_adapter_rejects_changed_pipeline(field: str, value: object) -> None:
    cfg = _config()
    setattr(cfg.actions["muscles"], field, value)
    with pytest.raises(ValueError, match="plain excitation"):
        browser_actions(
            cfg,
            SimpleNamespace(actuator_dyntype=np.array([mujoco.mjtDyn.mjDYN_MUSCLE])),
        )


def test_walk_adapter_rejects_noise_and_mixed_actuators() -> None:
    cfg = _config()
    cfg.actions["muscles"].motor_noise = replace(
        cfg.actions["muscles"].motor_noise, constant_std=0.1
    )
    with pytest.raises(ValueError, match="plain excitation"):
        browser_actions(
            cfg,
            SimpleNamespace(actuator_dyntype=np.array([mujoco.mjtDyn.mjDYN_MUSCLE])),
        )
    with pytest.raises(ValueError, match="plain excitation"):
        browser_actions(
            _config(),
            SimpleNamespace(actuator_dyntype=np.array([mujoco.mjtDyn.mjDYN_NONE])),
        )
