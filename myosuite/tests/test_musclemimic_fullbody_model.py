# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test for MuscleMimic full-body MJCF compilation."""

from __future__ import annotations


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


def test_fullbody_mimic_site_count() -> None:
    """Mimic site map matches MuscleMimic MyoFullBody (17 sites)."""
    from myosuite.integrations.musclemimic.fullbody_model import (
        FULLBODY_BODY2SITES_FOR_MIMIC,
    )

    assert len(FULLBODY_BODY2SITES_FOR_MIMIC) == 17
