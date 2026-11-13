"""Finger removal and left-arm stripping on myo_sim hands of different revisions.

The released hand names right-side muscles without a suffix and has no
tendon-tied joints; the revised hand (myo_sim#147) suffixes them (``FDS2_r``),
adds intrinsic muscles and ties lumbrical/junction node joints to the finger
tendons. The ``#147`` tests skip when the installed myo_sim has no such nodes.
"""

from __future__ import annotations

import mujoco
from ml_collections import config_dict
import numpy as np
import pytest

from myosuite.integrations.musclemimic.bimanual_model import (
    FINGER_JOINT_TOKENS,
    FINGER_MUSCLE_TOKENS,
    _translate_finger_token,
    native_finger_names,
)

pytestmark = pytest.mark.tier2


def _has_lumbrical_nodes(spec: mujoco.MjSpec) -> bool:
    return any(j.name.startswith("node_LUM") for j in spec.joints)


def test_translate_finger_token_suffix_variants() -> None:
    """Unsuffixed, ``_r`` and ``_left`` tokens resolve against either naming."""
    assert _translate_finger_token("FDS2", frozenset({"FDS2"})) == "FDS2"
    assert _translate_finger_token("FDS2", frozenset({"FDS2_r"})) == "FDS2_r"
    assert _translate_finger_token("cmc_flexion_r", frozenset({"cmc_flexion"})) == (
        "cmc_flexion"
    )
    assert _translate_finger_token("FDS2_left", frozenset({"FDS2_l"})) == "FDS2_l"
    assert _translate_finger_token("RI2", frozenset({"LUM1_r"})) is None


def test_native_finger_names_match_tokens_on_released_hand() -> None:
    """On a hand without tied nodes the names are exactly the translated tokens."""
    import myo_sim

    spec = myo_sim.load_spec("myofullbody")
    if _has_lumbrical_nodes(spec):
        pytest.skip("installed myo_sim hand is the revised one")
    joints, muscles = native_finger_names(spec)
    jpool = frozenset(j.name for j in spec.joints)
    apool = frozenset(a.name for a in spec.actuators)
    assert set(joints) == {
        _translate_finger_token(t, jpool) for t in FINGER_JOINT_TOKENS
    }
    assert muscles == tuple(
        _translate_finger_token(t, apool) for t in FINGER_MUSCLE_TOKENS
    )


def test_native_fullbody_disables_fingers_on_lumbrical_hand() -> None:
    """The native full body drops every hand joint and muscle on the revised hand (myo_sim#147)."""
    import myo_sim

    from myosuite.integrations.musclemimic.fullbody_model import (
        build_native_mimic_fullbody_spec,
    )

    full = myo_sim.load_spec("myofullbody")
    if not _has_lumbrical_nodes(full):
        pytest.skip("installed myo_sim hand has no lumbrical junction nodes")
    joints, muscles = native_finger_names(full)
    assert {"cmc4_flexion_r", "node_LUM1_r", "node_JT5_l"} <= set(joints)
    assert {"FDS2_r", "FDS2_l", "LUM1_r", "LUM4_l"} <= set(muscles)

    model = build_native_mimic_fullbody_spec(
        config_dict.create(disable_fingers=True, model_name="myofullbody")
    ).compile()
    names = {model.joint(i).name for i in range(model.njnt)}
    actuators = {model.actuator(i).name for i in range(model.nu)}
    assert not names & set(joints)
    assert not actuators & set(muscles)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=10)
    assert np.isfinite(data.qpos).all()


def test_tabletennis_builds_on_lumbrical_hand() -> None:
    """The table-tennis body strips the left arm with its tied nodes (myo_sim#147)."""
    import myo_sim

    from myosuite.core.model_builder import build_from_recipe

    if not _has_lumbrical_nodes(myo_sim.load_spec("myoarms")):
        pytest.skip("installed myo_sim hand has no lumbrical junction nodes")
    model, _ = build_from_recipe("challenge_tabletennis")
    names = [model.joint(i).name for i in range(model.njnt)]
    assert not [n for n in names if n.endswith("_l") and "node" in n]
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=10)
    assert np.isfinite(data.qpos).all()
