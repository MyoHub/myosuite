# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test for MuscleMimic full-body MJCF compilation."""

from __future__ import annotations

import pytest

from myosuite.tests.support.optional_deps import require_musclemimic_models


def test_compile_fullbody_mjmodel_dimensions() -> None:
    """Compiled full-body model should have consistent DoF counts."""
    require_musclemimic_models()
    from ml_collections import config_dict

    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )

    cfg = default_mimic_fullbody_config()
    mj, _, path = compile_mimic_fullbody_mjmodel(config_dict.create(**dict(cfg)))
    assert mj.nq > 0 and mj.nu > 0
    assert "myofullbody" in path.replace("\\", "/").lower() or path.endswith(".xml")


@pytest.mark.tier1
def test_fullbody_arena_fits_every_contact_and_limit_active() -> None:
    """The explicit arena replaces the 1.3 GB legacy one and fits the worst case.

    The worst case buries the body 2 m under the floor with 5 m contact margins
    and 10 rad joint margins, so every floor geom, explicit pair and joint limit
    is active at once (~1.5 MiB used).
    """
    require_musclemimic_models()
    import mujoco
    import numpy as np

    from myosuite.integrations.musclemimic.fullbody_model import (
        MIMIC_FULLBODY_ARENA_BYTES,
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )

    model, spec, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
    assert model.narena == MIMIC_FULLBODY_ARENA_BYTES
    # Legacy sizes stay on the model for the MJX Warp path.
    assert (model.nconmax, model.njmax) == (spec.nconmax, spec.njmax)

    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.margin = 5.0
    for pair in spec.pairs:
        pair.margin = 5.0
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_FREE:
            joint.margin = 10.0
    worst = spec.compile()
    assert worst.narena == MIMIC_FULLBODY_ARENA_BYTES
    body_geoms = (worst.geom_contype | worst.geom_conaffinity) & (worst.geom_bodyid > 0)
    n_candidates = int(np.count_nonzero(body_geoms)) + worst.npair
    data = mujoco.MjData(worst)
    rng = np.random.default_rng(0)
    for _ in range(5):
        mujoco.mj_resetData(worst, data)
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[2] = -2.0
        # mj_forward: mj_step would auto-reset (and zero maxuse) on a bad qacc.
        mujoco.mj_forward(worst, data)
        assert data.ncon >= n_candidates
        for warning in (
            mujoco.mjtWarning.mjWARN_CONTACTFULL,
            mujoco.mjtWarning.mjWARN_CNSTRFULL,
        ):
            assert data.warning[warning].number == 0
        assert 4 * data.maxuse_arena < worst.narena


def test_fullbody_mimic_site_count() -> None:
    """Mimic site map matches MuscleMimic MyoFullBody (17 sites)."""
    from myosuite.integrations.musclemimic.fullbody_model import (
        FULLBODY_BODY2SITES_FOR_MIMIC,
    )

    assert len(FULLBODY_BODY2SITES_FOR_MIMIC) == 17
