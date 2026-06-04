# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Registration of myoedits environments (arm-reach with procedural model edits)."""

import pathlib

import mujoco

import myosuite.core.registry as _registry

_SIMHIVE_ROOT = pathlib.Path(__file__).parents[3] / "simhive" / "myo_sim"


# Arm Reaching ==============================
def edit_fn_arm_reaching(spec: mujoco.MjSpec) -> None:
    # Get the positions of each body of each digit
    root_list = ["firstmc", "secondmc", "thirdmc", "fourthmc", "fifthmc"]
    body_positions = {}
    IFtip_site = {}

    for root in root_list:
        body_positions[root] = []
        body = spec.body(root)
        child_body = body.first_body()
        while child_body is not None:
            mesh_names = [
                geom.name
                for geom in child_body.geoms
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH
            ]
            body_positions[root].append(
                (child_body.name, child_body.pos.copy(), mesh_names)
            )
            child_body = child_body.first_body()

    # Get the properties of the IFtip site
    site = spec.site("IFtip")
    IFtip_site = {
        attr: (
            getattr(site, attr).copy()
            if hasattr(getattr(site, attr), "copy")
            else getattr(site, attr)
        )
        for attr in ["name", "size", "pos", "rgba"]
    }

    # Remove the digits
    for root in root_list:
        root_body = spec.body(root)
        child_body = root_body.first_body()
        if child_body is not None:
            spec.delete(child_body)

    # Add back simplified digits using mesh names as body names.
    # The original body names (e.g. "proxph2") are replaced by the
    # corresponding mesh names (e.g. "2proxph") to match the simplified
    # arm-reaching model convention expected by the arm-reaching tasks.
    for root in root_list:
        body = spec.body(root)
        for orig_body_name, pos, mesh_names in body_positions[root]:
            new_name = mesh_names[0] if mesh_names else orig_body_name
            body.add_body(name=new_name, pos=pos)
            for mesh_name in mesh_names:
                spec.body(new_name).add_geom(
                    meshname=mesh_name, name=new_name, type=mujoco.mjtGeom.mjGEOM_MESH
                )
            if orig_body_name == "distph2":
                spec.body(new_name).add_site(
                    name=IFtip_site["name"],
                    size=IFtip_site["size"] * 2,
                    pos=IFtip_site["pos"],
                    rgba=IFtip_site["rgba"],
                )
            body = spec.body(new_name)

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
    "model_path": str(_SIMHIVE_ROOT / "arm" / "myoarm.xml"),
    "target_reach_range": {
        "IFtip": ((-0.175, -0.245, 1.405), (-0.175, -0.245, 1.405)),
    },
    "normalize_act": True,
    "far_th": 1.0,
    "edit_fn": edit_fn_arm_reaching,
}
_random_kwargs = {
    "model_path": str(_SIMHIVE_ROOT / "arm" / "myoarm.xml"),
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
