# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Registration of myoedits environments (arm-reach with procedural model edits)."""

import mujoco

import myosuite.core.registry as _registry
from myosuite.utils.spec_processing import recursive_immobilize
import numpy as np

# Arm Reaching ==============================
def edit_fn_arm_reaching(spec: mujoco.MjSpec, remove_wrist=False, min_moment=1e-10) -> None:
    # Get the positions of each body of each digit. Names carry myo_sim's
    # "_r" (right-side) suffix — this model is built via the "full_arm"
    # recipe (myo_sim.load_spec("myoarm_r")), not the legacy bare-named XML.
    root_list = ["firstmc_r", "secondmc_r", "thirdmc_r", "fourthmc_r", "fifthmc_r"]

    if remove_wrist:
        root_list = ["lunate_r"]
        spec.delete(spec.joint("pro_sup_r"))

    for root in root_list:
        recursive_immobilize(spec, spec.copy().compile(), spec.body(root), remove_sites=False)


    _m = spec.compile()
    _d = mujoco.MjData(_m)
    mujoco.mj_step(_m, _d, nstep=100)

    # Now that we immobilized fingers, som muscles have no effect and can be pruned. We check through the tendon
    # moment arms to find them.
    J_tendon = np.empty((_m.ntendon, _m.nv))
    mujoco.mju_sparse2dense(J_tendon, _d.ten_J, _m.ten_J_rownnz, _m.ten_J_rowadr, _m.ten_J_colind)

    for t in spec.tendons:
        if np.sum(np.abs(J_tendon[_m.tendon(t.name).id, :])) < min_moment:
            for a in spec.actuators:

                if a.target == t.name:
                    spec.delete(a)
            spec.delete(t)

    spec.body("distph2_r").add_site(name="IFtip")

    # Add a reach target
    spec.body("world").add_site(
        name="IFtip_target",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0.02, 0.02],
        pos=[-0.2, -0.2, 1.2],
        rgba=[0.0, 0.0, 1.0, 0.3],
    )


_EP = "myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0"

_fixed_kwargs = {
    "model_recipe": "full_arm",
    "target_reach_range": {
        "IFtip": ((-0.175, -0.245, 1.405), (-0.175, -0.245, 1.405)),
    },
    "normalize_act": True,
    "far_th": 1.0,
    "edit_fn": edit_fn_arm_reaching,
}
_random_kwargs = {
    "model_recipe": "full_arm",
    "target_reach_range": {
        "IFtip": (
            (-0.175 - 0.175, -0.245 - 0.175, 1.405 - 0.425),
            (-0.175 + 0.175, -0.245 + 0.175, 1.405 + 0.425),
        ),
    },
    "normalize_act": True,
    "far_th": 1.0,
    "edit_fn": edit_fn_arm_reaching,
}

for _env_id, _kw in [
    ("myoArmReachFixed-v0", _fixed_kwargs),
    ("myoArmReachRandom-v0", _random_kwargs),
]:
    _registry.register_env(
        env_id=_env_id, entry_point=_EP, max_episode_steps=150, kwargs=_kw
    )
    _registry.register_env(
        env_id="myoSarc" + _env_id[3:],
        entry_point=_EP,
        max_episode_steps=150,
        kwargs={**_kw, "muscle_condition": "sarcopenia"},
    )
    _registry.register_env(
        env_id="myoFati" + _env_id[3:],
        entry_point=_EP,
        max_episode_steps=150,
        kwargs={**_kw, "muscle_condition": "fatigue"},
    )
