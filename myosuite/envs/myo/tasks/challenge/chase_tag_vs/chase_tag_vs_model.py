# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Model builders for the two-agent competitive chase-tag scene.

Both supported host models -- the myoLeg-only ``myolegs_with_torso.xml``
(the same host used by ``LegWalkEnvV0`` / the directional locomotion tasks)
and the MuscleMimic full-body model (legs + torso + arms, 354 muscles each,
which lets the chase-tag policy reuse a pretrained MuscleMimic locomotion
checkpoint as a competent walking substrate) -- are combined into a two-agent
scene by the *same* parameterized builder below. The only thing that varies
per host model is how a single agent's ``MjSpec`` (+ its standing keyframe
qpos) is constructed; that piece is supplied as the ``agent_spec_builder``
callback to ``_build_two_agent_spec`` / ``_build_model_and_meta``.

The combined model exposes the same per-agent site/sensor naming convention
(``{prefix}pelvis_site`` / ``{prefix}pelvis_site_vel``) for every host model,
so every term function in ``myosuite/terms/multiplayer/chase_tag_vs_obs.py``
and ``chase_tag_vs_reward.py`` is reused unchanged across host models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from myosuite.envs.myo.assets._resolve import resolve_leg_xml
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.arena import add_arena
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config import (
    ChaseTagVsConfig,
)
from myosuite.envs.myo.tasks.challenge.combat_model_meta import CombatModelMetaBase
from myosuite.integrations.musclemimic.fullbody_model import (
    build_mimic_fullbody_spec,
    default_mimic_fullbody_config,
)
from myosuite.integrations.musclemimic.two_agent_scene import (
    build_combined_spec,
    default_root_offsets,
    extract_prefix_indices,
    extract_sensor_addrs,
    extract_site_ids,
    mj_name2id_strict,
    two_agent_standing_qpos,
)
from myosuite.utils.asset_path_resolver import resolve_model_xml_path

logger = logging.getLogger(__name__)

AGENTS: tuple[str, str] = ("agent_0", "agent_1")
_AGENT_PREFIXES: dict[str, str] = {"agent_0": "a0_", "agent_1": "a1_"}
_AGENT_COLORS: dict[int, list[float]] = {
    0: [0.84, 0.12, 0.12, 1.0],  # chaser — red
    1: [0.12, 0.12, 0.84, 1.0],  # runner — blue
}
_INVISIBLE_RGBA: list[float] = [0.0, 0.0, 0.0, 0.0]

# A single-agent spec builder returns the standalone (pre-attach) MjSpec
# plus its source model's keyframe-0 qpos (or None if it ships no keyframe).
AgentSpecBuilder = Callable[
    [ChaseTagVsConfig], tuple[mujoco.MjSpec, "np.ndarray | None"]
]


@dataclass
class ChaseTagVsModelMeta(CombatModelMetaBase):
    """Pre-computed indices into MjModel/MjData arrays for both chase-tag agents.

    Shared across every host model (myoLeg-only, MuscleMimic full-body) --
    the set of cached indices is identical regardless of which agent model
    the combined scene was built from.
    """

    pelvis_body_id: dict[str, int] = field(default_factory=dict)


def _add_agent_pelvis_site(
    spec: mujoco.MjSpec, prefix: str, rgba: list[float] | None = None
) -> None:
    """Add a pelvis tracking site for one agent."""
    spec.body(f"{prefix}pelvis").add_site(
        name=f"{prefix}pelvis_site",
        pos=[0.0, 0.0, 0.0],
        size=[0.03, 0.0, 0.0],
        rgba=rgba if rgba is not None else _INVISIBLE_RGBA,
    )


def _add_agent_sensor(spec: mujoco.MjSpec, prefix: str) -> None:
    spec.add_sensor(
        name=f"{prefix}pelvis_site_vel",
        type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=f"{prefix}pelvis_site",
    )


def _leg_standing_key_qpos(leg_xml: str) -> np.ndarray | None:
    """Return keyframe-0 ``qpos`` from the standalone (pre-attach) leg model.

    ``myolegs_with_torso.xml`` ships a real standing-pose keyframe (pelvis
    ~0.92 m, nonzero hip/knee angles). It must be captured here, before the
    model is attached into the combined two-agent scene, because
    ``MjSpec.attach`` does not carry keyframes across (they would collide on
    name after prefixing) -- see ``two_agent_standing_qpos`` for how this is
    reconstructed into the combined model's per-agent qpos block.
    """
    m = mujoco.MjModel.from_xml_path(leg_xml)
    return m.key_qpos[0].copy() if m.nkey > 0 else None


def _build_leg_agent_spec(
    config: ChaseTagVsConfig,
) -> tuple[mujoco.MjSpec, np.ndarray | None]:
    """Build one standalone myoLeg-only agent spec (+ its standing keyframe)."""
    leg_xml = str(resolve_model_xml_path(resolve_leg_xml("myolegs_with_torso.xml")))
    standing_key_qpos = _leg_standing_key_qpos(leg_xml)
    spec = mujoco.MjSpec.from_file(leg_xml)
    # myolegs_with_torso.xml ships unnamed standing-pose keyframes; after
    # agent-prefix attachment they would all collide on the same generated
    # name. The standing pose is captured above (before this deletion) and
    # reconstructed into the combined model's per-agent qpos block by
    # two_agent_standing_qpos(); drop the raw per-agent keys here rather than
    # carry the name clash into the compiled model.
    for key in list(spec.keys):
        spec.delete(key)
    return spec, standing_key_qpos


def _build_fullbody_agent_spec(
    config: ChaseTagVsConfig,
) -> tuple[mujoco.MjSpec, np.ndarray | None]:
    """Build one standalone MuscleMimic full-body agent spec (+ its standing keyframe)."""
    fullbody_cfg = default_mimic_fullbody_config()
    fullbody_cfg.sim_dt = config.sim_dt
    spec, _ = build_mimic_fullbody_spec(fullbody_cfg)
    standalone_model = spec.compile()
    standing_key_qpos = (
        standalone_model.key_qpos[0].copy() if standalone_model.nkey > 0 else None
    )
    return spec, standing_key_qpos


def _build_two_agent_spec(
    config: ChaseTagVsConfig,
    agent_spec_builder: AgentSpecBuilder,
    *,
    model_name: str,
    floor_rgba: list[float],
    site_rgba_by_idx: dict[int, list[float]] | None = None,
) -> tuple[mujoco.MjSpec, np.ndarray | None]:
    """Build the combined two-agent chase-tag ``MjSpec`` for any host model.

    Args:
        config: Task configuration (timing, placement, tag hyperparameters).
        agent_spec_builder: Builds one standalone agent ``MjSpec`` (+ its
            standing keyframe qpos). Called once per agent; both agents are
            built from the same host model, so the two returned standing
            keyframes are identical -- only the first is kept.
        model_name: Value assigned to the combined spec's ``modelname``.
        floor_rgba: RGBA colour of the combined scene's floor plane.
        site_rgba_by_idx: Optional ``{0: rgba, 1: rgba}`` colours for each
            agent's pelvis tracking site marker. Defaults to fully
            transparent (myoLeg's variant colours its markers red/blue for
            chaser/runner visual debugging; full-body leaves them invisible).

    Returns:
        Tuple of ``(combined spec, standing keyframe-0 qpos)``. The standing
        qpos is ``None`` if the source host model ships no keyframe.
    """
    agent0_spec, standing_key_qpos = agent_spec_builder(config)
    agent1_spec, _ = agent_spec_builder(config)

    spec = build_combined_spec(
        agent0_spec,
        agent1_spec,
        separation_m=config.agent_separation_m,
        sim_dt=config.sim_dt,
        model_name=model_name,
        floor_rgba=floor_rgba,
        floor_collision=True,
    )

    for idx, prefix in enumerate(("a0_", "a1_")):
        rgba = (site_rgba_by_idx or {}).get(idx)
        _add_agent_pelvis_site(spec, prefix, rgba)
        _add_agent_sensor(spec, prefix)

    add_arena(spec)

    return spec, standing_key_qpos


def _extract_meta(
    model: mujoco.MjModel,
    standing_key_qpos: np.ndarray | None,
    separation_m: float,
) -> ChaseTagVsModelMeta:
    meta = ChaseTagVsModelMeta(n_act=model.nu)
    for agent_id, prefix in _AGENT_PREFIXES.items():
        act_idx, jnt_ids, _ = extract_prefix_indices(model, prefix)
        meta.act_indices[agent_id] = act_idx
        meta.jnt_ids[agent_id] = jnt_ids
        meta.pelvis_body_id[agent_id] = mj_name2id_strict(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}pelvis"
        )
        meta.site_ids[agent_id] = extract_site_ids(model, prefix, ("pelvis_site",))
        meta.sensor_adr[agent_id] = extract_sensor_addrs(
            model, prefix, ("pelvis_site",)
        )
    meta.standing_qpos = two_agent_standing_qpos(
        model,
        meta.jnt_ids,
        {agent_id: standing_key_qpos for agent_id in _AGENT_PREFIXES},
        default_root_offsets(separation_m),
    )
    return meta


def _build_model_and_meta(
    config: ChaseTagVsConfig,
    agent_spec_builder: AgentSpecBuilder,
    *,
    model_name: str,
    floor_rgba: list[float],
    site_rgba_by_idx: dict[int, list[float]] | None = None,
    hide_geoms: tuple[str, ...] = (),
) -> tuple[mujoco.MjModel, mujoco.MjData, ChaseTagVsModelMeta]:
    """Compile a combined two-agent chase-tag scene and extract its metadata.

    Args:
        config: Task configuration (timing, placement, tag hyperparameters).
        agent_spec_builder: See ``_build_two_agent_spec``.
        model_name: Value assigned to the combined spec's ``modelname``.
        floor_rgba: RGBA colour of the combined scene's floor plane.
        site_rgba_by_idx: Optional per-agent pelvis-site marker colours.
        hide_geoms: Names of geoms to disable collision on and make fully
            transparent after compilation (e.g. per-agent floor planes that
            duplicate the combined scene's world floor).

    Returns:
        Tuple of (MjModel, MjData, ChaseTagVsModelMeta).
    """
    spec, standing_key_qpos = _build_two_agent_spec(
        config,
        agent_spec_builder,
        model_name=model_name,
        floor_rgba=floor_rgba,
        site_rgba_by_idx=site_rgba_by_idx,
    )
    model = spec.compile()
    model.opt.timestep = config.sim_dt

    for geom_name in hide_geoms:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid >= 0:
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0
            model.geom_rgba[gid, 3] = 0.0  # fully transparent

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    logger.info(
        "Chase-tag-vs (%s) model compiled: nq=%d nv=%d nu=%d",
        model_name,
        model.nq,
        model.nv,
        model.nu,
    )

    meta = _extract_meta(model, standing_key_qpos, config.agent_separation_m)
    logger.info(
        "Agent 0 (chaser): %d actuators | Agent 1 (runner): %d actuators",
        len(meta.act_indices["agent_0"]),
        len(meta.act_indices["agent_1"]),
    )
    return model, data, meta


# ---------------------------------------------------------------------------
# myoLeg-only variant
# ---------------------------------------------------------------------------

_LEG_FLOOR_RGBA: list[float] = [0.6, 0.75, 0.6, 1.0]


def _build_chase_tag_vs_spec(config: ChaseTagVsConfig) -> mujoco.MjSpec:
    """Build and return the combined two-agent myoLeg-only chase-tag MjSpec."""
    spec, _ = _build_two_agent_spec(
        config,
        _build_leg_agent_spec,
        model_name="chase_tag_vs",
        floor_rgba=_LEG_FLOOR_RGBA,
        site_rgba_by_idx=_AGENT_COLORS,
    )
    return spec


def build_chase_tag_vs_model(
    config: ChaseTagVsConfig,
) -> tuple[mujoco.MjModel, mujoco.MjData, ChaseTagVsModelMeta]:
    """Build the combined two-agent myoLeg-only chase-tag MuJoCo model.

    Args:
        config: Task configuration (timing, placement, tag hyperparameters).

    Returns:
        Tuple of (MjModel, MjData, ChaseTagVsModelMeta).
    """
    return _build_model_and_meta(
        config,
        _build_leg_agent_spec,
        model_name="chase_tag_vs",
        floor_rgba=_LEG_FLOOR_RGBA,
        site_rgba_by_idx=_AGENT_COLORS,
    )


# ---------------------------------------------------------------------------
# MuscleMimic full-body variant
# ---------------------------------------------------------------------------

_FULLBODY_FLOOR_RGBA: list[float] = [1.0, 1.0, 1.0, 1.0]
_FULLBODY_HIDE_GEOMS: tuple[str, ...] = ("a0_floor", "a1_floor")


def _build_chase_tag_vs_fullbody_spec(
    config: ChaseTagVsConfig,
) -> mujoco.MjSpec:
    """Build and return the combined two-agent full-body chase-tag MjSpec."""
    spec, _ = _build_two_agent_spec(
        config,
        _build_fullbody_agent_spec,
        model_name="chase_tag_vs_fullbody",
        floor_rgba=_FULLBODY_FLOOR_RGBA,
    )
    return spec


def build_chase_tag_vs_fullbody_model(
    config: ChaseTagVsConfig,
) -> tuple[mujoco.MjModel, mujoco.MjData, ChaseTagVsModelMeta]:
    """Build the combined two-agent full-body chase-tag MuJoCo model.

    Args:
        config: Task configuration (timing, placement, tag hyperparameters).

    Returns:
        Tuple of (MjModel, MjData, ChaseTagVsModelMeta).
    """
    return _build_model_and_meta(
        config,
        _build_fullbody_agent_spec,
        model_name="chase_tag_vs_fullbody",
        floor_rgba=_FULLBODY_FLOOR_RGBA,
        # Per-agent floor planes duplicate the world floor -- disable
        # collision and hide them.
        hide_geoms=_FULLBODY_HIDE_GEOMS,
    )
