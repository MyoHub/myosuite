# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reward v1 terms for the two-agent boxing environment (capstone 2026).

* Contact-based hit events: one event per glove impact (rising edge of the
  glove/opponent contact + cooldown), classified as head / body / block, with
  damage scaled by the approach speed along the contact normal measured on the
  previous control step (so it does not depend on contact softness).
* Attributed falls: the opponent's fall only pays if this agent hit it within
  a time window; a double fall is penalised for both.
* Shaping (alive, reach progress, facing) multiplied by an annealing factor.
* Effort on the clipped activation command, averaged over muscles.
* Per-term breakdowns of both the legacy and the v1 reward, for logging.

Everything here is CPU MuJoCo and side-effect free except
:class:`HitTracker`, which owns the per-episode detection state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_model import (
    ZONE_BLOCK,
    ZONE_BODY,
    ZONE_HEAD,
    ZONE_NAMES,
    ZONE_NONE,
)

AGENT_INDEX: dict[str, int] = {"agent_0": 0, "agent_1": 1}

LEGACY_TERMS: tuple[str, ...] = (
    "damage_dealt",
    "damage_received",
    "survival",
    "fall_bonus",
    "ko_bonus",
    "self_fall",
    "effort",
)
LEGACY_ATTRIBUTED_TERMS: tuple[str, ...] = (
    "damage_dealt",
    "damage_received",
    "survival",
    "ko_bonus",
    "fall",
    "effort",
)
V1_TERMS: tuple[str, ...] = (
    "hit_dealt",
    "hit_received",
    "alive",
    "reach",
    "facing",
    "effort",
    "fall",
    "ko",
)


def reward_term_names(config: Any) -> tuple[str, ...]:
    """Names of the logged reward terms for *config*'s reward mode."""
    if config.reward_mode == "v1":
        return V1_TERMS
    if config.fall_reward_mode == "attributed":
        return LEGACY_ATTRIBUTED_TERMS
    return LEGACY_TERMS


# ---------------------------------------------------------------------------
# Hit detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GloveContact:
    """One contact between a glove sphere and an opponent geom."""

    attacker: str
    side: str  # "r" | "l"
    glove_geom: int
    other_geom: int
    zone: int  # ZONE_* code of the opponent geom
    normal: np.ndarray  # unit vector from the glove towards the other geom
    pos: np.ndarray  # contact point (world)


@dataclass(frozen=True)
class HitEvent:
    """A glove impact that counts: head/body hit or blocked punch."""

    attacker: str
    side: str
    zone: str  # "head" | "body" | "block"
    speed: float  # approach speed along the contact normal (m/s)
    damage: float  # 0 for blocks


@dataclass
class Kinematics:
    """Snapshot of what is needed to compute point velocities later."""

    cvel: np.ndarray
    subtree_com: np.ndarray
    glove_pos: dict[int, np.ndarray]


def snapshot_kinematics(data: mujoco.MjData, glove_geoms: Iterable[int]) -> Kinematics:
    """Copy body velocities, subtree COMs and glove centres from *data*."""
    return Kinematics(
        cvel=np.array(data.cvel, copy=True),
        subtree_com=np.array(data.subtree_com, copy=True),
        glove_pos={g: np.array(data.geom_xpos[g], copy=True) for g in glove_geoms},
    )


def point_velocity(
    kin: Kinematics, body_rootid: np.ndarray, body: int, point: np.ndarray
) -> np.ndarray:
    """World linear velocity of a point rigidly attached to *body*.

    ``cvel`` is the com-based spatial velocity ``[ang; lin]`` expressed at the
    subtree COM of the body's root, so ``v(p) = lin + ang x (p - c)``.
    """
    c = kin.subtree_com[int(body_rootid[body])]
    ang = kin.cvel[body, :3]
    lin = kin.cvel[body, 3:]
    return lin - np.cross(np.asarray(point) - c, ang)


def glove_owner_table(meta: Any) -> dict[int, tuple[str, str]]:
    """``{glove_geom_id: (agent_id, side)}`` from the model metadata."""
    return {
        gid: (agent, side)
        for agent, gloves in meta.glove_geom_ids.items()
        for side, gid in gloves.items()
    }


def collect_glove_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, meta: Any
) -> list[GloveContact]:
    """All current contacts between a glove and a geom of the other agent.

    A glove-glove contact is reported twice (once from each glove's side).
    """
    owner = glove_owner_table(meta)
    out: list[GloveContact] = []
    if not owner:
        return out
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in owner and g2 not in owner:
            continue
        normal = np.array(c.frame[:3], dtype=np.float64)  # geom1 -> geom2
        pos = np.array(c.pos, dtype=np.float64)
        for glove, other, sign in ((g1, g2, 1.0), (g2, g1, -1.0)):
            if glove not in owner:
                continue
            attacker, side = owner[glove]
            if int(meta.geom_agent[other]) != 1 - AGENT_INDEX[attacker]:
                continue
            out.append(
                GloveContact(
                    attacker=attacker,
                    side=side,
                    glove_geom=glove,
                    other_geom=other,
                    zone=int(meta.geom_zone[other]),
                    normal=sign * normal,
                    pos=pos,
                )
            )
    return out


def impact_speed(
    contact: GloveContact, prev: Kinematics | None, model: mujoco.MjModel
) -> float:
    """Approach speed of the glove relative to the touched body (m/s, >= 0).

    Uses the velocities of the previous control step, i.e. just before the
    impact, projected on the contact normal.
    """
    if prev is None or contact.glove_geom not in prev.glove_pos:
        return 0.0
    rootid = model.body_rootid
    v_glove = point_velocity(
        prev,
        rootid,
        int(model.geom_bodyid[contact.glove_geom]),
        prev.glove_pos[contact.glove_geom],
    )
    v_other = point_velocity(
        prev, rootid, int(model.geom_bodyid[contact.other_geom]), contact.pos
    )
    return max(0.0, float(np.dot(v_glove - v_other, contact.normal)))


def _own_glove_speed(
    glove: int, normal: np.ndarray, prev: Kinematics | None, model: mujoco.MjModel
) -> float:
    if prev is None or glove not in prev.glove_pos:
        return 0.0
    v = point_velocity(
        prev, model.body_rootid, int(model.geom_bodyid[glove]), prev.glove_pos[glove]
    )
    return float(np.dot(v, normal))


def speed_scale(speed: float, min_speed: float, full_speed: float) -> float:
    """0 below *min_speed*, else ``min(speed, full_speed) / full_speed``."""
    if speed < min_speed:
        return 0.0
    return min(speed, full_speed) / max(full_speed, 1e-9)


@dataclass
class HitTracker:
    """Per-episode state for glove hit events.

    ``touching[key]`` is True while glove ``key = (agent, side)`` touches the
    opponent; an event can only start on a False -> True transition and at
    least ``cooldown_s`` after the previous event of the same glove.
    """

    touching: dict[tuple[str, str], bool] = field(default_factory=dict)
    last_event_step: dict[tuple[str, str], int] = field(default_factory=dict)
    prev: Kinematics | None = None

    def update(
        self,
        contacts: list[GloveContact],
        model: mujoco.MjModel,
        step: int,
        config: Any,
    ) -> list[HitEvent]:
        """Turn this step's glove contacts into hit events (updates state)."""
        # A glove-glove contact is listed from both sides, so the touched
        # glove also shows up as a ``glove_geom``.
        glove_geoms = {c.glove_geom for c in contacts}
        by_glove: dict[tuple[str, str], list[GloveContact]] = {}
        for c in contacts:
            by_glove.setdefault((c.attacker, c.side), []).append(c)

        events: list[HitEvent] = []
        now_touching: dict[tuple[str, str], bool] = {}
        for key, cs in by_glove.items():
            now_touching[key] = True
            if self.touching.get(key, False):
                continue  # still the same contact (leaning / clinch)
            last = self.last_event_step.get(key)
            if (
                last is not None
                and (step - last) * config.ctrl_dt < config.hit_cooldown_s
            ):
                continue
            scoring = [c for c in cs if c.zone != ZONE_NONE]
            if not scoring:
                continue
            best = min(scoring, key=lambda c: c.zone)  # head > body > block
            speed = impact_speed(best, self.prev, model)
            if best.zone == ZONE_BLOCK and best.other_geom in glove_geoms:
                # Glove-on-glove: only the glove moving faster towards the
                # other one is the puncher.
                mine = _own_glove_speed(best.glove_geom, best.normal, self.prev, model)
                theirs = _own_glove_speed(
                    best.other_geom, -best.normal, self.prev, model
                )
                if mine < theirs:
                    continue
            scale = speed_scale(
                speed, config.hit_min_speed_mps, config.hit_full_speed_mps
            )
            if scale <= 0.0:
                continue  # a touch, not a punch
            if best.zone == ZONE_HEAD:
                damage = config.w_head * scale
            elif best.zone == ZONE_BODY:
                damage = config.w_body * scale
            else:
                damage = 0.0
            events.append(
                HitEvent(
                    attacker=key[0],
                    side=key[1],
                    zone=ZONE_NAMES[best.zone],
                    speed=speed,
                    damage=float(damage),
                )
            )
            self.last_event_step[key] = step
        self.touching = now_touching
        return events


# ---------------------------------------------------------------------------
# Falls
# ---------------------------------------------------------------------------


def attributed_fall_term(
    *,
    self_fell: bool,
    opponent_fell: bool,
    hit_opponent_recently: bool,
    r_fall_bonus: float,
    r_self_fall_penalty: float,
    r_double_fall_penalty: float,
) -> float:
    """Fall reward where only a fall caused by our hit pays.

    * both fall in the same step       -> ``r_double_fall_penalty`` (each)
    * I fall                            -> ``r_self_fall_penalty``
    * opponent falls after my hit       -> ``r_fall_bonus``
    * opponent falls on its own (slip)  -> 0
    """
    if self_fell and opponent_fell:
        return float(r_double_fall_penalty)
    if self_fell:
        return float(r_self_fall_penalty)
    if opponent_fell and hit_opponent_recently:
        return float(r_fall_bonus)
    return 0.0


# ---------------------------------------------------------------------------
# Shaping signals
# ---------------------------------------------------------------------------


def reach_potential(data: mujoco.MjData, meta: Any, agent_id: str) -> float:
    """``-min`` distance from this agent's fists to the opponent's zones."""
    opp = "agent_1" if agent_id == "agent_0" else "agent_0"
    own, other = meta.site_ids[agent_id], meta.site_ids[opp]
    best = np.inf
    for fist in ("fist_r", "fist_l"):
        p = data.site_xpos[own[fist]]
        for zone in ("head_zone", "body_zone"):
            best = min(best, float(np.linalg.norm(p - data.site_xpos[other[zone]])))
    return -best


def facing_cosine(data: mujoco.MjData, meta: Any, agent_id: str) -> float:
    """Cosine between the pelvis heading and the direction to the opponent
    (horizontal plane).  The pelvis local +X axis is the forward direction."""
    opp = "agent_1" if agent_id == "agent_0" else "agent_0"
    pel = meta.pelvis_body_id[agent_id]
    fwd = np.array(data.xmat[pel]).reshape(3, 3)[:, 0][:2]
    to_opp = (data.xpos[meta.pelvis_body_id[opp]] - data.xpos[pel])[:2]
    nf, no = np.linalg.norm(fwd), np.linalg.norm(to_opp)
    if nf < 1e-9 or no < 1e-9:
        return 0.0
    return float(np.dot(fwd, to_opp) / (nf * no))


# ---------------------------------------------------------------------------
# Reward assembly
# ---------------------------------------------------------------------------


def effort_mean_sq(actions: np.ndarray) -> float:
    """Mean squared activation command, clipped to the physical range [0, 1]."""
    a = np.clip(np.asarray(actions, dtype=np.float64), 0.0, 1.0)
    return float(np.mean(a**2)) if a.size else 0.0


def legacy_reward_terms(
    *,
    delivered_damage: float,
    received_damage: float,
    actions: np.ndarray,
    self_fell: bool,
    opponent_fell: bool,
    ko_happened: bool,
    config: Any,
) -> dict[str, float]:
    """Per-term breakdown of ``combine_combat_reward`` (for logging only)."""
    return {
        "damage_dealt": delivered_damage * config.r_damage_delivered,
        "damage_received": received_damage * config.r_damage_received,
        "survival": config.r_survival_per_step,
        "fall_bonus": config.r_fall_bonus if opponent_fell else 0.0,
        "ko_bonus": config.r_ko_bonus if ko_happened else 0.0,
        "self_fall": config.r_self_fall_penalty if self_fell else 0.0,
        "effort": config.r_act_reg * float(np.sum(np.asarray(actions) ** 2)),
    }


def v1_reward_terms(
    *,
    dealt: float,
    received: float,
    self_fell: bool,
    opponent_fell: bool,
    opponent_ko: bool,
    hit_opponent_recently: bool,
    reach_progress: float,
    facing: float,
    actions: np.ndarray,
    config: Any,
) -> dict[str, float]:
    """Reward v1, term by term (the reward is the sum of the values)."""
    alpha = float(config.shaping_alpha)
    return {
        "hit_dealt": config.r_hit * dealt,
        "hit_received": -config.r_hit * received,
        "alive": 0.0 if self_fell else alpha * config.r_alive,
        "reach": alpha * config.r_reach * reach_progress,
        "facing": alpha * config.r_facing * facing,
        "effort": config.r_effort * effort_mean_sq(actions),
        "fall": attributed_fall_term(
            self_fell=self_fell,
            opponent_fell=opponent_fell,
            hit_opponent_recently=hit_opponent_recently,
            r_fall_bonus=config.r_fall_bonus,
            r_self_fall_penalty=config.r_self_fall_penalty,
            r_double_fall_penalty=config.r_double_fall_penalty,
        ),
        "ko": config.r_ko_bonus if opponent_ko else 0.0,
    }


def event_counts(events: Iterable[HitEvent], agent_id: str) -> Mapping[str, float]:
    """Per-step hit statistics of *agent_id* (as attacker) for ``info``."""
    mine = [e for e in events if e.attacker == agent_id]
    hits = [e for e in mine if e.zone in ("head", "body")]
    return {
        "hits_head": float(sum(e.zone == "head" for e in mine)),
        "hits_body": float(sum(e.zone == "body" for e in mine)),
        "blocked": float(sum(e.zone == "block" for e in mine)),
        "hits_r": float(sum(e.side == "r" for e in hits)),
        "hits_l": float(sum(e.side == "l" for e in hits)),
        "impact_speed_max": float(max((e.speed for e in mine), default=0.0)),
    }
