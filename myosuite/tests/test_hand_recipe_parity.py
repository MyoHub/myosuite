# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Numerical parity: hand task recipes vs myo_sim's ``myohand`` model.

Verifies that the hand_* model recipes produce models that are structurally and
dynamically equivalent to ``myo_sim.load_spec("myohand")`` (the canonical
reference distributed on PyPI). myo-sim is a mandatory dependency, so a missing
reference fails instead of skipping.
"""

from __future__ import annotations

import numpy as np
import pytest
from myosuite import make_env

pytestmark = pytest.mark.tier1

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_model():
    """mujoco.MjModel of myo_sim's public ``myohand`` model."""
    import myo_sim

    return myo_sim.load_spec("myohand").compile()


@pytest.fixture(scope="module")
def recipe_hand_model():
    """MjModel built via the hand_standard recipe."""
    import myosuite.core.model_recipes  # noqa: F401 — registers recipes
    from myosuite.core.model_builder import build_from_recipe

    m, _ = build_from_recipe("hand_standard")
    return m


@pytest.fixture(scope="module")
def recipe_hand_pose_model():
    """MjModel built via hand_pose recipe (adds fingertip target sites)."""
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe

    m, _ = build_from_recipe("hand_pose")
    return m


# ---------------------------------------------------------------------------
# Structural parity: hand_standard vs myo_sim myohand
# ---------------------------------------------------------------------------


def test_hand_standard_nq_nv_nu_na(ref_model, recipe_hand_model):
    """Recipe model has same degrees of freedom and actuator counts."""
    assert recipe_hand_model.nq == ref_model.nq
    assert recipe_hand_model.nv == ref_model.nv
    assert recipe_hand_model.nu == ref_model.nu
    assert recipe_hand_model.na == ref_model.na


def test_hand_standard_joint_names_and_order(ref_model, recipe_hand_model):
    """Joint names and ordering match the myo_sim reference exactly."""
    ref_names = [ref_model.joint(i).name for i in range(ref_model.njnt)]
    recipe_names = [
        recipe_hand_model.joint(i).name for i in range(recipe_hand_model.njnt)
    ]
    assert (
        recipe_names == ref_names
    ), f"Joint name/order mismatch.\n  ref:    {ref_names}\n  recipe: {recipe_names}"


def test_hand_standard_actuator_names(ref_model, recipe_hand_model):
    """Actuator names match the myo_sim reference exactly."""
    import mujoco

    ref_names = [
        mujoco.mj_id2name(ref_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(ref_model.nu)
    ]
    recipe_names = [
        mujoco.mj_id2name(recipe_hand_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(recipe_hand_model.nu)
    ]
    assert recipe_names == ref_names


def test_hand_standard_qpos0(ref_model, recipe_hand_model):
    """Default qpos matches the myo_sim reference (same neutral pose)."""
    np.testing.assert_array_equal(recipe_hand_model.qpos0, ref_model.qpos0)


def test_hand_standard_joint_ranges(ref_model, recipe_hand_model):
    """Joint ranges (limits) are identical to the myo_sim reference."""
    np.testing.assert_array_equal(recipe_hand_model.jnt_range, ref_model.jnt_range)


def test_hand_standard_dynamics_zero_action(ref_model, recipe_hand_model):
    """Zero-action rollout matches the reference with gravity off.

    The recipe re-roots the bare hand in the legacy world frame, and the
    reference hangs it on a passive torso, so only gravity-free dynamics compare.
    """
    import copy

    import mujoco

    N = 50
    qpos = []
    for model in (ref_model, recipe_hand_model):
        model = copy.copy(model)
        model.opt.gravity[:] = 0.0
        data = mujoco.MjData(model)
        for _ in range(N):
            mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())

    np.testing.assert_allclose(
        qpos[1],
        qpos[0],
        atol=1e-10,
        err_msg="qpos diverged after zero-action rollout",
    )


# ---------------------------------------------------------------------------
# Structural parity: task recipes add the right scene elements
# ---------------------------------------------------------------------------


def test_hand_pose_adds_target_sites(recipe_hand_model, recipe_hand_pose_model):
    """hand_pose recipe adds 5 fingertip target sites over the base hand."""
    expected_extra = {
        "THtip_r_target",
        "IFtip_r_target",
        "MFtip_r_target",
        "RFtip_r_target",
        "LFtip_r_target",
    }
    recipe_sites = {
        recipe_hand_pose_model.site(i).name for i in range(recipe_hand_pose_model.nsite)
    }
    assert expected_extra.issubset(
        recipe_sites
    ), f"Missing target sites: {expected_extra - recipe_sites}"
    assert recipe_hand_pose_model.nsite == recipe_hand_model.nsite + len(expected_extra)


def test_hand_keyturn_adds_key(ref_model):
    """hand_keyturn recipe adds a key body and keyhead site."""
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe

    m, _ = build_from_recipe("hand_keyturn")
    site_names = {m.site(i).name for i in range(m.nsite)}
    assert "keyhead" in site_names
    assert m.njnt == ref_model.njnt + 1  # key hinge joint


def test_hand_hold_adds_object_and_goal(ref_model):
    """hand_hold recipe adds freejoint object and goal site."""
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe

    m, _ = build_from_recipe("hand_hold")
    site_names = {m.site(i).name for i in range(m.nsite)}
    assert "object" in site_names
    assert "goal" in site_names
    assert m.njnt == ref_model.njnt + 1  # freejoint (7 DOF counted as 1 joint)


def test_hand_pen_adds_pen_object(ref_model):
    """hand_pen recipe adds 6-DOF pen object with top/bottom sites."""
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe

    m, _ = build_from_recipe("hand_pen")
    site_names = {m.site(i).name for i in range(m.nsite)}
    assert {"object_top", "object_bottom", "eps_ball", "target_ball"}.issubset(
        site_names
    )
    assert m.njnt == ref_model.njnt + 6  # 3 slide + 3 hinge joints for pen


def test_hand_pen_success_indicator_on_target_pen():
    """Pen twirl shows success on the target pen and hides the position goal (issue 136)."""
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe

    m, _ = build_from_recipe("hand_pen")
    assert m.site_bodyid[m.site("target_ball").id] == m.body("target").id
    assert m.site_group[m.site("eps_ball").id] == 4


def test_hand_sar_adds_sar_object(ref_model):
    """hand_sar recipe adds 6-DOF SAR object with success site."""
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe

    m, _ = build_from_recipe("hand_sar")
    site_names = {m.site(i).name for i in range(m.nsite)}
    assert {"success", "eps_ball"}.issubset(site_names)
    assert m.njnt == ref_model.njnt + 6


# ---------------------------------------------------------------------------
# Gym env smoke-test: recipe-based envs step without error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_id",
    [
        "myoHandPoseFixed-v0",
        "myoHandPoseRandom-v0",
        "myoHandReachFixed-v0",
        "myoHandReachRandom-v0",
        "myoHandKeyTurnFixed-v0",
        "myoHandObjHoldFixed-v0",
        "myoHandPenTwirlFixed-v0",
        "myoHandReorient8-v0",
    ],
)
def test_hand_env_reset_and_step(env_id):
    """Hand env resets and steps without exception."""
    import myosuite  # noqa: F401 — registers envs

    env = make_env(env_id)
    obs, _ = env.reset(seed=0)
    assert obs.ndim == 1
    obs2, rwd, term, trunc, info = env.step(env.action_space.sample())
    assert obs2.shape == obs.shape
    assert np.isfinite(rwd)
    env.close()


@pytest.mark.parametrize(
    ("env_id", "recipe", "expected_counts"),
    [
        ("myoHandKeyTurnFixed-v0", "hand_keyturn", (40, 101, 330)),
        ("myoHandObjHoldFixed-v0", "hand_hold", (40, 99, 330)),
        ("myoHandPenTwirlFixed-v0", "hand_pen", (41, 106, 334)),
    ],
)
def test_legacy_hand_task_recipe_compatibility(env_id, recipe, expected_counts):
    """Public contact-hand IDs retain release-era recipe semantics.

    The composed ``hand_*`` recipes swap ``cmc_flexion`` / ``cmc_abduction``
    relative to the release-era hand. Public NPG baselines require the legacy
    order, while task furniture is built through MjSpec transforms.
    """
    import myosuite  # noqa: F401 — registers envs
    from myosuite.core.model_builder import build_from_recipe

    recipe_model, _ = build_from_recipe(recipe)
    env = make_env(env_id)
    try:
        names = [
            env.unwrapped.model.joint(i).name for i in range(env.unwrapped.model.njnt)
        ]
        cmc = [n for n in names if "cmc" in n]
        assert cmc[:2] == [
            "cmc_abduction",
            "cmc_flexion",
        ], f"{env_id}: unexpected CMC order {cmc[:2]}"
        assert not any(n.endswith("_r") for n in cmc[:2])
        assert (recipe_model.nbody, recipe_model.ngeom, recipe_model.nsite) == (
            expected_counts
        )
    finally:
        env.close()
