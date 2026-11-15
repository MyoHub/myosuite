# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Left-right mirror maps on myo_sim's full body: symmetric involutions, validated through the task's kinematics and
muscle geometry. Ported from muscle-supermimic's tests (@ 8a6244b, Apache-2.0)."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

pytestmark = pytest.mark.tier1


def test_mirror_maps_are_symmetric_involutions(body: Any) -> None:
    from myosuite.envs.myo.tasks.mimic.tracking.models import model_cfg
    from supermimic.kinematics import (
        SkeletonKinematics,
    )
    from supermimic.muscle_geometry import (
        MuscleGeometry,
    )
    from supermimic.rotations import (
        quat_to_mat,
    )
    from supermimic.mirror import Mirror, partner_name

    mc = model_cfg("myofullbody")
    kin, geo = (
        SkeletonKinematics(mc.tracked_bodies, "cpu", body),
        MuscleGeometry(body, "cpu"),
    )
    mr = Mirror(kin, geo, mc.foot_bodies, "cpu")
    r = mr.residuals
    assert (
        r["positions"] < 3e-3 and r["rotations"] < 1e-2 and r["default_pose"] < 1e-3
    ), r
    assert (
        float(mr.muscle_residual.median()) < 1e-5
        and float(mr.muscle_residual.max()) < 5e-2
    )
    names = kin.info.indep_names
    for k, name in enumerate(names):
        assert names[int(mr.j_src[k])] == partner_name(name), (
            name,
            names[int(mr.j_src[k])],
        )
    muscles, perm = kin.info.muscle_names, mr.muscle_perm.tolist()
    assert [perm[k] for k in perm] == list(range(len(perm)))  # an involution
    pair = {muscles[k]: muscles[perm[k]] for k in range(len(perm))}
    # myo_sim names the right arm's muscles bare and the left arm's *_l
    assert (
        pair["DELT1"] == "DELT1_l"
        and pair["BIClong_l"] == "BIClong"
        and pair["soleus_r"] == "soleus_l"
    )
    assert mr.touch_perm.tolist() == [2, 3, 0, 1]
    gen = torch.Generator().manual_seed(0)
    rn = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
    S = (
        rn(4, 3),
        quat_to_mat(rn(4, 4)),
        rn(4, 3),
        rn(4, 3),
        0.5 * rn(4, kin.num_q),
        rn(4, kin.num_q),
    )
    for a, b in zip(mr.body(mr.body(S)), S, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    x = rn(4, geo.num_muscles)
    torch.testing.assert_close(mr.muscles(mr.muscles(x)), x, atol=0, rtol=0)
    q = mr.q0.clone()
    q[names.index("hip_flexion_l")] += 0.3
    j = names.index("hip_flexion_r")
    torch.testing.assert_close(mr.joints(q)[j] - mr.q0[j], torch.tensor(0.3))
