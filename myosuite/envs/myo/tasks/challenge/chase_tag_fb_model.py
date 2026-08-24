# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Full-body + scripted-mocap-opponent ChaseTag model builder.

Reuses the existing full-body spec builder
(:func:`myosuite.integrations.musclemimic.fullbody_model.build_mimic_fullbody_spec`)
and grafts on the mocap "opponent" body plus a heightfield terrain, matching
the leg-only ``myolegs_chasetag.xml`` scene contract that :class:`ChaseTagEnv`
expects (a "terrain" hfield geom, an "opponent" mocap body, and an
"opponent_indicator" site).

The opponent geometry mirrors ``myosuite/envs/myo/assets/leg/myolegs_chasetag.xml``
(lines ~27-34) but is added programmatically via the ``MjSpec`` Python API
rather than an XML ``<include>``, because that source file has unrelated,
actively-evolving uncommitted changes from a concurrent session and must not
be edited or depended on as an include target.
"""

from __future__ import annotations

import mujoco
import numpy as np
from ml_collections import config_dict

from myosuite.integrations.musclemimic.fullbody_model import (
    build_mimic_fullbody_spec,
    default_mimic_fullbody_config,
)
from myosuite.utils.asset_path_resolver import get_sim_asset_root

# A single real mid-gait frame captured from ``myoFullBodyDirectional-v0``'s
# own reset (seed=0) -- i.e. the actual state distribution
# ``bc_directional_v2`` was trained on (that env samples from real motion-
# capture circular-walk clips at every reset, never from a static standing
# pose; see MuscleMimicFullbodyDirectionalEnv.reset_task in
# myosuite/envs/myo/tasks/mimic/cpu.py). Without this, the base full-body
# model's own default keyframe (an approximately-neutral, identity-root-
# orientation rest pose) is used instead -- confirmed by a zero-action
# rollout to be physically unstable on its own (the pelvis visibly topples
# from gravity alone within ~15 steps), and far outside anything the
# warm-started policy has ever seen, which was the actual root cause of the
# "falls almost immediately" behavior observed in early FBP2 renders, not a
# policy-quality issue. Baking this single representative frame in as the
# model's keyframe is a static, offline-computed fix (matching how
# myolegs_chasetag.xml's own keyframes are also fixed baked arrays) --
# no runtime clip-loading dependency, so env construction stays fast and
# offline-safe for tests.
_STANDING_KEYFRAME_QPOS = np.array(
    [
        -0.637794,
        -0.368976,
        0.931779,
        -0.937175,
        0.026269,
        -0.022932,
        0.347113,
        -0.102543,
        -0.000350,
        0.082689,
        0.000000,
        0.000000,
        -0.065927,
        -0.018969,
        -0.000066,
        0.003121,
        -0.020933,
        -0.000082,
        0.003121,
        -0.023715,
        -0.000076,
        0.002572,
        -0.026185,
        -0.000049,
        0.002391,
        -0.057532,
        0.024266,
        -0.024368,
        0.057475,
        -0.011632,
        0.094003,
        0.042260,
        -0.042298,
        -0.094099,
        0.011646,
        -0.608851,
        0.237522,
        0.608847,
        -0.080915,
        0.449644,
        -0.007012,
        0.049920,
        0.028681,
        -0.056961,
        0.024121,
        -0.024130,
        0.056958,
        -0.011533,
        0.093199,
        0.041895,
        -0.041897,
        -0.093215,
        0.011533,
        0.908226,
        0.235366,
        -0.908229,
        0.463036,
        0.650477,
        0.051369,
        0.105866,
        0.108197,
        0.313840,
        -0.111675,
        -0.119994,
        0.002414,
        0.000900,
        0.457362,
        0.027382,
        0.135915,
        0.108682,
        -0.113686,
        0.091456,
        -0.030095,
        0.039667,
        -0.190999,
        -0.266952,
        0.100041,
        0.281748,
        0.000041,
        0.000420,
        -0.000077,
        -0.000008,
        0.000110,
        0.189898,
        0.034397,
        -0.048875,
        -0.010824,
        0.052420,
        0.010503,
    ]
)
_STANDING_KEYFRAME_QVEL = np.array(
    [
        -0.585961,
        -0.634855,
        -0.085714,
        -0.165934,
        -0.300404,
        1.793627,
        -0.111712,
        -0.162702,
        -0.190435,
        0.000000,
        0.000000,
        -0.071817,
        -0.020675,
        -0.029481,
        -0.007206,
        -0.022793,
        -0.039898,
        -0.007207,
        -0.025802,
        -0.040652,
        -0.005930,
        -0.028475,
        -0.030528,
        -0.005496,
        -0.008086,
        0.003423,
        -0.003452,
        0.008143,
        -0.001649,
        0.013375,
        0.006055,
        -0.005913,
        -0.013238,
        0.001643,
        -0.647605,
        0.033589,
        0.647626,
        -0.014855,
        -0.193477,
        0.100358,
        -0.066512,
        -0.086718,
        -0.032784,
        0.013919,
        -0.013840,
        0.032753,
        -0.006654,
        0.053641,
        0.024068,
        -0.024137,
        -0.053601,
        0.006616,
        -0.363445,
        0.135416,
        0.363434,
        0.211170,
        0.743576,
        0.032062,
        0.328730,
        -0.062085,
        0.039950,
        0.160365,
        0.494512,
        -0.022584,
        -0.009541,
        -4.566703,
        -0.172059,
        -1.057431,
        -2.616331,
        1.641603,
        -0.109738,
        0.152477,
        0.172656,
        3.520837,
        -1.261072,
        -0.316222,
        0.878331,
        0.000213,
        0.002070,
        -0.004448,
        -0.000425,
        0.001945,
        0.799916,
        1.234739,
        -0.149091,
        0.000218,
        0.000065,
        -0.000180,
    ]
)


def _add_standing_keyframe(spec: mujoco.MjSpec) -> None:
    """Add the gait-phase-consistent keyframe as the spec's (only) key 0.

    ``spec.keys`` is empty pre-compile even though the compiled model ends
    up with ``nkey=1`` (the base model's own default keyframe is injected
    later in the attach/compile chain, not visible on this spec object) --
    confirmed empirically. Adding our own key here means it becomes the
    compiled model's ``key_qpos[0]``, which is what ``ChaseTagEnv``'s
    ``reset_type="none"`` (this env's setting) actually reads.
    """
    spec.add_key(
        name="standing",
        qpos=_STANDING_KEYFRAME_QPOS.tolist(),
        qvel=_STANDING_KEYFRAME_QVEL.tolist(),
    )


# Matches the hfield/geom used by myolegs_chasetag.xml + ChaseTagField.
_TERRAIN_SIZE = (6.0, 6.0, 1.0, 0.001)
_TERRAIN_NROW = 100
_TERRAIN_NCOL = 100


def _add_terrain(spec: mujoco.MjSpec) -> None:
    """Add the "terrain" hfield asset + geom that :class:`ChaseTagField` expects.

    Args:
        spec: Full-body MjSpec to edit in place. Must already have a
            "matfloor" material (present on the Mimic full-body scene).
    """
    hfield = spec.add_hfield(
        name="terrain",
        size=list(_TERRAIN_SIZE),
        nrow=_TERRAIN_NROW,
        ncol=_TERRAIN_NCOL,
    )
    # Flat (all-zero) elevation data at build time; ChaseTagField mutates
    # ``model.hfield_data`` in place at runtime via ``.sample()``.
    hfield.userdata = np.zeros(_TERRAIN_NROW * _TERRAIN_NCOL, dtype=np.float64)
    material_names = {m.name for m in spec.materials}
    terrain_material = "matfloor" if "matfloor" in material_names else ""
    spec.worldbody.add_geom(
        name="terrain",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname="terrain",
        pos=[0, 0, -0.005],
        material=terrain_material,
        conaffinity=1,
        contype=1,
        rgba=[0.9, 0.9, 0.9, 1.0],
    )


def _add_opponent(spec: mujoco.MjSpec) -> None:
    """Add the scripted mocap "opponent" body (base + logo + indicator site).

    Geometry mirrors the inline opponent block in ``myolegs_chasetag.xml``.

    Args:
        spec: Full-body MjSpec to edit in place.
    """
    icon_path = get_sim_asset_root("myo_sim") / "scene" / "myosuite_icon.png"
    spec.add_texture(
        name="texmyo",
        type=mujoco.mjtTexture.mjTEXTURE_CUBE,
        file=icon_path.as_posix(),
    )
    textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "texmyo"
    spec.add_material(name="matmyo", textures=textures, rgba=[1, 1, 1, 1])

    opponent = spec.worldbody.add_body(
        name="opponent", pos=[0, 0, 0], zaxis=[0, 0, 1], mocap=True
    )
    opponent.add_light(
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        diffuse=[0.25, 0.25, 0.25],
        specular=[0.25, 0.25, 0.25],
        pos=[0, -3, 3],
        dir=[0, 1, -1],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
    )
    opponent.add_camera(
        name="opponent_view",
        pos=[4, 0, 2.75],
        xyaxes=[0, 1, 0, -1, 0, 2],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
    )
    opponent.add_geom(
        name="base",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        pos=[0, 0, 0.15],
        size=[0.25, 0.15, 0],
        rgba=[0.11, 0.1, 0.1, 1],
        group=2,
        contype=0,
        conaffinity=0,
    )
    opponent.add_geom(
        name="base_bar",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0.8],
        size=[0.078, 0.28, 0],
        rgba=[0.7, 0.7, 0.7, 1],
        group=2,
        contype=0,
        conaffinity=0,
    )
    opponent.add_geom(
        name="logo",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0, 0, 1.20],
        euler=[1.57, 0, 0],
        size=[0.25, 0.0201, 0],
        material="matmyo",
        group=2,
        contype=0,
        conaffinity=0,
    )
    opponent.add_site(
        name="opponent_indicator", size=[0.3, 0, 0], pos=[0, 0, 1.2], rgba=[0, 0, 0, 0]
    )


def _add_head_site(spec: mujoco.MjSpec) -> None:
    """Add a "head" site so :class:`ChaseTagEnv`'s non-flat fall detection works.

    ``ChaseTagEnv._get_fallen_condition``/reward shaping read
    ``data.site("head")`` on non-flat terrain (mirroring the "head" site
    ``myolegs_chasetag.xml`` defines on its ``root`` body). The full-body
    model has no such site (only the mimic-tracking "head_mimic" site added
    by :func:`build_mimic_fullbody_spec`), so this adds one at the "head"
    body's origin.

    Args:
        spec: Full-body MjSpec to edit in place. Must have a "head" body.
    """
    spec.body("head").add_site(name="head", group=4, size=[0.02, 0, 0], pos=[0, 0, 0])


def build_fullbody_chasetag_spec(
    config: config_dict.ConfigDict | None = None,
) -> mujoco.MjSpec:
    """Build the full-body + mocap-opponent + terrain ``MjSpec`` (uncompiled).

    Reuses :func:`build_mimic_fullbody_spec` for the base 354-muscle
    full-body model, then grafts on the ChaseTag terrain hfield and mocap
    opponent body so the result satisfies the same scene contract
    :class:`~myosuite.envs.myo.tasks.challenge.chasetag.ChaseTagEnv` expects
    from ``myolegs_chasetag.xml``.

    Args:
        config: Full-body model config (see
            :func:`default_mimic_fullbody_config`). Defaults are used when
            omitted.

    Returns:
        Uncompiled ``MjSpec`` ready for ``.compile()``.
    """
    cfg = config if config is not None else default_mimic_fullbody_config()
    spec, _xml_path = build_mimic_fullbody_spec(cfg)
    # build_mimic_fullbody_spec returns the *uncompiled* spec, which keeps
    # the raw base MJCF asset's own default timestep (0.001s). cfg.sim_dt
    # (0.002s -- the actual value the directional policy was trained under,
    # and what MuscleMimicFullbodyDirectionalEnv/FBVs both compile with) is
    # normally only applied post-compile, in
    # compile_mimic_fullbody_mjmodel's `mj_model.opt.timestep = ...` line --
    # which this spec-only path never calls. Confirmed empirically: without
    # this line, FBP2 compiled at timestep=0.001 while every other env the
    # policy was trained/evaluated on used 0.002 -- a genuine physics-level
    # train/test mismatch (half the intended integration step, and thus a
    # different effective control frequency for the same frame_skip),
    # independent of and in addition to the reset-pose and action-space
    # bugs already fixed.
    spec.option.timestep = float(cfg.sim_dt)
    _add_terrain(spec)
    _add_opponent(spec)
    _add_head_site(spec)
    _add_standing_keyframe(spec)
    return spec


def compile_fullbody_chasetag_model(
    config: config_dict.ConfigDict | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjSpec, str]:
    """Compile the full-body + mocap-opponent ChaseTag model.

    Args:
        config: Full-body model config, see :func:`build_fullbody_chasetag_spec`.

    Returns:
        Tuple of compiled ``MjModel``, the edited ``MjSpec``, and a label
        string (no standalone XML file backs this in-memory spec).
    """
    spec = build_fullbody_chasetag_spec(config)
    return spec.compile(), spec, "fullbody_chasetag:in_memory"


def build_default_fullbody_chasetag_spec() -> mujoco.MjSpec:
    """Zero-arg wrapper for use as a picklable ``model_spec_fn`` env kwarg.

    Returns:
        Uncompiled ``MjSpec`` built with default full-body config.
    """
    return build_fullbody_chasetag_spec()


__all__ = [
    "build_fullbody_chasetag_spec",
    "compile_fullbody_chasetag_model",
    "build_default_fullbody_chasetag_spec",
]
