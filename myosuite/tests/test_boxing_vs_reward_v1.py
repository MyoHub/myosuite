# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the BoxingVs reward-v1 options (capstone 2026).

Covers: legacy defaults unchanged, cross-agent collision masks, massless visual
gear, contact-based hit events (real impacts and synthetic contacts),
attributed falls, shaping annealing, reset pose/noise and the held dummy.
No training and no pretrained policy needed.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.envs.myo.tasks.challenge.boxing_vs import (
    BoxingVsConfig,
    BoxingVsEnv,
    boxing_vs_level1_config,
)
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_model import (
    ZONE_BLOCK,
    ZONE_BODY,
    ZONE_HEAD,
)
from myosuite.terms.multiplayer.boxing_vs_reward_v1 import (
    GloveContact,
    HitTracker,
    attributed_fall_term,
    effort_mean_sq,
    snapshot_kinematics,
    speed_scale,
    v1_reward_terms,
)
from myosuite.terms.multiplayer.common import combine_combat_reward

pytestmark = pytest.mark.tier2

AGENTS = ("agent_0", "agent_1")
N_ACT = 354
PUNCH_FRAME = 215  # push clip: right fist ~0.68 m forward, ~0.57 m above pelvis


def _zeros():
    return {a: np.zeros(N_ACT, dtype=np.float32) for a in AGENTS}


def _task_config(env: BoxingVsEnv):
    return env._base_env._config  # pylint: disable=protected-access


def _agent_of_geom(model, g):
    name = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
    )
    return name[:2] if name[:3] in ("a0_", "a1_") else "world"


def _can_collide(model, g1, g2):
    return bool(
        (model.geom_contype[g1] & model.geom_conaffinity[g2])
        or (model.geom_contype[g2] & model.geom_conaffinity[g1])
    )


# ---------------------------------------------------------------------------
# Fixtures (env builds are the slow part: share them)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def legacy_env():
    env = BoxingVsEnv()
    yield env
    env.close()


@pytest.fixture(scope="module")
def contact_env():
    """Contact hits + dummy, default start pose, no noise (deterministic)."""
    env = BoxingVsEnv(
        boxing_vs_level1_config(
            reset_pose="default",
            reset_noise_separation_m=0.0,
            reset_noise_lateral_m=0.0,
            reset_noise_yaw_deg=0.0,
            reset_noise_joint_rad=0.0,
            agent_separation_m=0.6,
        )
    )
    yield env
    env.close()


@pytest.fixture(scope="module")
def level1_env():
    env = BoxingVsEnv(boxing_vs_level1_config())
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Legacy defaults unchanged
# ---------------------------------------------------------------------------


def test_defaults_have_no_cross_agent_collisions(legacy_env):
    m = legacy_env.model
    g0 = [g for g in range(m.ngeom) if _agent_of_geom(m, g) == "a0"]
    g1 = [g for g in range(m.ngeom) if _agent_of_geom(m, g) == "a1"]
    assert not any(_can_collide(m, a, b) for a in g0 for b in g1)
    assert legacy_env.observation_space["agent_0"].shape == (603,)
    assert not legacy_env._meta.glove_geom_ids  # pylint: disable=protected-access


def test_default_reward_matches_legacy_formula(legacy_env):
    cfg = _task_config(legacy_env)
    legacy_env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        acts = {a: rng.uniform(0, 1, N_ACT).astype(np.float32) for a in AGENTS}
        _, rew, term, _, info = legacy_env.step(acts)
        for a, o in (("agent_0", "agent_1"), ("agent_1", "agent_0")):
            expected = combine_combat_reward(
                delivered_damage=info[f"damage_delivered/{a}"],
                received_damage=info[f"damage_delivered/{o}"],
                actions=acts[a],
                r_damage_delivered=cfg.r_damage_delivered,
                r_damage_received=cfg.r_damage_received,
                r_survival_per_step=cfg.r_survival_per_step,
                r_act_reg=cfg.r_act_reg,
                self_fell=info[f"fell/{a}"],
                opponent_fell=info[f"fell/{o}"],
                ko_happened=info[f"ko/{o}"],
                r_fall_bonus=cfg.r_fall_bonus,
                r_ko_bonus=cfg.r_ko_bonus,
                r_self_fall_penalty=cfg.r_self_fall_penalty,
            )
            assert rew[a] == expected
            breakdown = sum(
                v for k, v in info.items() if k.startswith("reward/") and k.endswith(a)
            )
            assert breakdown == pytest.approx(rew[a], abs=1e-9)
        if any(term.values()):
            break


def test_legacy_double_fall_is_nearly_free(legacy_env):
    """Documents the legacy flaw: +5 - 5 + 0.01 when both fall together."""
    legacy_env.reset(seed=0)
    for _ in range(200):
        _, rew, term, _, info = legacy_env.step(_zeros())
        if term["agent_0"]:
            break
    assert info["fell/agent_0"] and info["fell/agent_1"]
    assert rew["agent_0"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Model: collisions and visual gear
# ---------------------------------------------------------------------------


def test_collision_masks(contact_env):
    m = contact_env.model
    meta = contact_env._meta  # pylint: disable=protected-access
    a0 = [
        g
        for g in range(m.ngeom)
        if _agent_of_geom(m, g) == "a0" and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]
    a1 = [
        g
        for g in range(m.ngeom)
        if _agent_of_geom(m, g) == "a1" and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    # never with itself (explicit <pair> contacts are separate and unchanged)
    assert not any(_can_collide(m, x, y) for x in a0 for y in a0 if x < y)
    assert not any(_can_collide(m, x, y) for x in a1 for y in a1 if x < y)
    # every collision geom still touches the floor
    assert all(_can_collide(m, g, floor) for g in a0 + a1)
    # gloves hit the opponent's head; fingers only touch the floor
    glove = meta.glove_geom_ids["agent_0"]["r"]
    head1 = [g for g in a1 if meta.geom_zone[g] == ZONE_HEAD]
    assert head1 and all(_can_collide(m, glove, h) for h in head1)
    finger = next(
        g
        for g in a0
        if "3proxph_r"
        in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    )
    assert not any(_can_collide(m, finger, g) for g in a1)
    # zones are where we expect
    zones = {int(meta.geom_zone[g]) for g in a1}
    assert {ZONE_HEAD, ZONE_BODY, ZONE_BLOCK} <= zones


def test_massless_visual_gear_restores_body_masses(legacy_env, contact_env):
    def mass(env, body):
        m = env.model
        return m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)]

    # legacy: the visual meshes add ~2 kg per hand and ~5.8 kg to the head
    assert mass(legacy_env, "a0_lunate_r") > 2.0
    assert mass(legacy_env, "a0_head") > 7.0
    assert mass(contact_env, "a0_lunate_r") == pytest.approx(0.164, abs=0.01)
    assert mass(contact_env, "a0_head") == pytest.approx(2.126, abs=0.01)


def test_contact_detection_requires_collisions():
    with pytest.raises(ValueError):
        BoxingVsEnv(BoxingVsConfig(hit_detection="contact"))


# ---------------------------------------------------------------------------
# Hit events with real impacts
# ---------------------------------------------------------------------------


def _shoot(env, speed: float, gap: float, target_body: str):
    """Put agent_0's right arm in a punch pose, place its glove *gap* metres
    in front of agent_1's *target_body*, launch agent_0 at *speed* towards
    agent_1 and return the per-step info of 15 control steps."""
    cfg = _task_config(env)
    m, d, meta = env.model, env.data, env._meta  # pylint: disable=protected-access
    env.reset(seed=0)
    old_frame = cfg.reset_clip_frame
    cfg.reset_clip_frame = PUNCH_FRAME
    cfg._rv1_cache.pop("clip", None)  # pylint: disable=protected-access
    frame, _, maps = cfg._clip_pose_maps(m, meta)  # pylint: disable=protected-access
    cfg.reset_clip_frame = old_frame
    cfg._rv1_cache.pop("clip", None)  # pylint: disable=protected-access
    dst, src = maps["agent_0"]
    d.qpos[dst] = frame[src]
    mujoco.mj_forward(m, d)

    glove = meta.glove_geom_ids["agent_0"]["r"]
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target_body)
    targets = [
        g for g in range(m.ngeom) if m.geom_bodyid[g] == tb and m.geom_contype[g]
    ]
    target = max(targets, key=lambda g: d.geom_xpos[g][1])  # agent_1 faces +Y
    root = cfg._root_qpos_adr(m, meta, "agent_0")  # pylint: disable=protected-access
    delta = d.geom_xpos[target] - d.geom_xpos[glove]
    d.qpos[root] += delta[0]
    d.qpos[root + 2] += delta[2]
    mujoco.mj_forward(m, d)
    fromto = np.zeros(6)
    dist = mujoco.mj_geomDistance(m, d, glove, target, 2.0, fromto)
    d.qpos[root + 1] -= dist - gap
    vadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "a0_root")]
    d.qvel[vadr + 1] = -speed
    mujoco.mj_forward(m, d)
    cfg._rv1["tracker"].prev = snapshot_kinematics(d, cfg._glove_geoms(meta))  # pylint: disable=protected-access
    return [env.step(_zeros())[4] for _ in range(15)]


def _total(infos, key, agent="agent_0"):
    return sum(i[f"{key}/{agent}"] for i in infos)


def test_head_punch_counts_once_with_speed_scaled_damage(contact_env):
    infos = _shoot(contact_env, speed=3.0, gap=0.03, target_body="a1_head")
    assert _total(infos, "hits_head") == 1
    assert _total(infos, "hits_body") == 0
    assert _total(infos, "hits_r") == 1
    speed = max(i["impact_speed_max/agent_0"] for i in infos)
    assert 2.0 < speed < 3.5
    cfg = _task_config(contact_env)
    assert _total(infos, "reward/hit_dealt") == pytest.approx(
        cfg.r_hit * cfg.w_head * min(speed, 4.5) / 4.5, rel=1e-6
    )
    # the dummy cannot hit back
    assert _total(infos, "reward/hit_received") == 0.0


def test_body_punch(contact_env):
    infos = _shoot(contact_env, speed=3.0, gap=0.03, target_body="a1_thorax")
    assert _total(infos, "hits_body") == 1
    assert _total(infos, "hits_head") == 0


@pytest.mark.parametrize("speed,gap", [(0.3, 0.003), (0.0, -0.002)])
def test_slow_touch_or_leaning_does_not_score(contact_env, speed, gap):
    infos = _shoot(contact_env, speed=speed, gap=gap, target_body="a1_head")
    assert _total(infos, "hits_head") + _total(infos, "hits_body") == 0
    assert _total(infos, "reward/hit_dealt") == 0.0


# ---------------------------------------------------------------------------
# Hit events with synthetic contacts (block, glove-on-glove, cooldown)
# ---------------------------------------------------------------------------


def _kin_with_velocity(env, moving_geom: int, velocity):
    m, d, meta = env.model, env.data, env._meta  # pylint: disable=protected-access
    env.reset(seed=0)
    kin = snapshot_kinematics(d, _task_config(env)._glove_geoms(meta))  # pylint: disable=protected-access
    kin.cvel[:] = 0.0
    kin.cvel[m.geom_bodyid[moving_geom], 3:] = velocity
    return kin


def _forearm_geom(env, agent_prefix="a1_"):
    m, meta = env.model, env._meta  # pylint: disable=protected-access
    return next(
        g
        for g in range(m.ngeom)
        if meta.geom_zone[g] == ZONE_BLOCK
        and (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        ).startswith(agent_prefix + "radius")
    )


def test_glove_on_forearm_is_blocked(contact_env):
    m, meta = contact_env.model, contact_env._meta  # pylint: disable=protected-access
    glove = meta.glove_geom_ids["agent_0"]["r"]
    other = _forearm_geom(contact_env)
    tracker = HitTracker(prev=_kin_with_velocity(contact_env, glove, [0.0, -3.0, 0.0]))
    contact = GloveContact(
        "agent_0",
        "r",
        glove,
        other,
        ZONE_BLOCK,
        np.array([0.0, -1.0, 0.0]),
        np.zeros(3),
    )
    events = tracker.update([contact], m, step=1, config=_task_config(contact_env))
    assert [(e.zone, e.damage) for e in events] == [("block", 0.0)]
    assert events[0].speed == pytest.approx(3.0)


def test_glove_on_glove_only_credits_the_puncher(contact_env):
    m, meta = contact_env.model, contact_env._meta  # pylint: disable=protected-access
    g0 = meta.glove_geom_ids["agent_0"]["r"]
    g1 = meta.glove_geom_ids["agent_1"]["l"]
    n = np.array([0.0, -1.0, 0.0])  # from agent_0's glove towards agent_1's
    tracker = HitTracker(prev=_kin_with_velocity(contact_env, g0, [0.0, -3.0, 0.0]))
    contacts = [
        GloveContact("agent_0", "r", g0, g1, ZONE_BLOCK, n, np.zeros(3)),
        GloveContact("agent_1", "l", g1, g0, ZONE_BLOCK, -n, np.zeros(3)),
    ]
    events = tracker.update(contacts, m, step=1, config=_task_config(contact_env))
    assert [(e.attacker, e.zone) for e in events] == [("agent_0", "block")]


def test_cooldown_and_rearm(contact_env):
    m, meta = contact_env.model, contact_env._meta  # pylint: disable=protected-access
    cfg = _task_config(contact_env)
    glove = meta.glove_geom_ids["agent_0"]["r"]
    head = next(
        g
        for g in range(m.ngeom)
        if meta.geom_zone[g] == ZONE_HEAD and meta.geom_agent[g] == 1
    )
    kin = _kin_with_velocity(contact_env, glove, [0.0, -3.0, 0.0])
    c = GloveContact(
        "agent_0", "r", glove, head, ZONE_HEAD, np.array([0.0, -1.0, 0.0]), np.zeros(3)
    )
    tracker = HitTracker(prev=kin)
    n_events = []
    # contact for 3 steps, release, re-contact after 0.05 s (< cooldown), release,
    # re-contact after the cooldown
    schedule = [1, 1, 1, 0, 0, 1, 0] + [0] * 15 + [1]
    for step, touching in enumerate(schedule, start=1):
        n_events.append(len(tracker.update([c] if touching else [], m, step, cfg)))
    assert n_events[0] == 1
    assert sum(n_events[1:-1]) == 0
    assert n_events[-1] == 1


def test_speed_scale():
    assert speed_scale(1.0, 1.5, 4.5) == 0.0
    assert speed_scale(2.25, 1.5, 4.5) == pytest.approx(0.5)
    assert speed_scale(9.0, 1.5, 4.5) == 1.0


# ---------------------------------------------------------------------------
# Falls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "self_fell,opp_fell,hit_recently,expected",
    [
        (False, True, False, 0.0),  # opponent slipped: no free bonus
        (False, True, True, 5.0),  # knockdown after my hit
        (True, False, False, -5.0),  # I fell
        (True, True, True, -5.0),  # double fall: penalised for both
        (False, False, True, 0.0),
    ],
)
def test_attributed_fall_term(self_fell, opp_fell, hit_recently, expected):
    assert (
        attributed_fall_term(
            self_fell=self_fell,
            opponent_fell=opp_fell,
            hit_opponent_recently=hit_recently,
            r_fall_bonus=5.0,
            r_self_fall_penalty=-5.0,
            r_double_fall_penalty=-5.0,
        )
        == expected
    )


def test_attributed_double_fall_in_env():
    env = BoxingVsEnv(BoxingVsConfig(fall_reward_mode="attributed"))
    env.reset(seed=0)
    for _ in range(200):
        _, rew, term, _, info = env.step(_zeros())
        if term["agent_0"]:
            break
    env.close()
    assert info["fell/agent_0"] and info["fell/agent_1"]
    assert info["reward/fall/agent_0"] == -5.0
    assert rew["agent_0"] == pytest.approx(-5.0 + 0.01)


# ---------------------------------------------------------------------------
# Shaping, effort
# ---------------------------------------------------------------------------


def test_v1_terms_and_annealing():
    cfg = BoxingVsConfig(reward_mode="v1")
    kwargs = {
        "dealt": 2.0,
        "received": 0.5,
        "self_fell": False,
        "opponent_fell": False,
        "opponent_ko": False,
        "hit_opponent_recently": False,
        "reach_progress": 0.05,
        "facing": 1.0,
        "actions": np.full(N_ACT, 0.5),
    }
    terms = v1_reward_terms(config=cfg, **kwargs)
    assert terms["hit_dealt"] == 2.0 and terms["hit_received"] == -0.5
    assert terms["alive"] == 0.01 and terms["reach"] == pytest.approx(0.05)
    assert terms["effort"] == pytest.approx(-0.1 * 0.25)
    cfg.shaping_alpha = 0.0
    terms = v1_reward_terms(config=cfg, **kwargs)
    assert terms["alive"] == terms["reach"] == terms["facing"] == 0.0
    assert terms["hit_dealt"] == 2.0  # hits are never annealed


def test_set_shaping_alpha(level1_env):
    level1_env.set_shaping_alpha(0.0)
    try:
        level1_env.reset(seed=0)
        _, _, _, _, info = level1_env.step(_zeros())
        assert info["reward/alive/agent_0"] == 0.0
        assert info["reward/facing/agent_0"] == 0.0
    finally:
        level1_env.set_shaping_alpha(1.0)


def test_effort_uses_clipped_actions():
    assert effort_mean_sq(np.array([-3.0, 0.5, 2.0])) == pytest.approx(
        (0 + 0.25 + 1) / 3
    )


# ---------------------------------------------------------------------------
# Reset pose, noise, dummy
# ---------------------------------------------------------------------------


def test_level1_reset_is_seeded_and_randomised(level1_env):
    level1_env.reset(seed=1)
    q1 = level1_env.data.qpos.copy()
    level1_env.reset(seed=1)
    assert np.array_equal(q1, level1_env.data.qpos)
    level1_env.reset(seed=2)
    assert not np.array_equal(q1, level1_env.data.qpos)


def test_clip_pose_sets_root_height(level1_env):
    level1_env.reset(seed=0)
    d, meta = level1_env.data, level1_env._meta  # pylint: disable=protected-access
    root = _task_config(level1_env)._root_qpos_adr(level1_env.model, meta, "agent_0")  # pylint: disable=protected-access
    assert d.qpos[root + 2] == pytest.approx(0.933, abs=0.01)  # clip frame 0, not 1.0


def test_dummy_is_held_and_never_falls(level1_env):
    level1_env.reset(seed=0)
    d, meta = level1_env.data, level1_env._meta  # pylint: disable=protected-access
    pel = meta.site_ids["agent_1"]["pelvis_site"]
    z0 = d.site_xpos[pel].copy()
    rng = np.random.default_rng(0)
    for _ in range(60):
        acts = {a: rng.uniform(0, 1, N_ACT).astype(np.float32) for a in AGENTS}
        _, _, term, _, info = level1_env.step(acts)
        assert not info["fell/agent_1"]
        if term["agent_0"]:
            break
    assert np.linalg.norm(d.site_xpos[pel] - z0) < 0.01


def test_level1_registered_single_agent_env():
    from myosuite import make_env

    env = make_env("myoChallengeBoxingVsLevel1-v0", agent_separation_m=0.5)
    try:
        assert env.observation_space.shape == (603,)
        assert env.action_space.shape == (N_ACT,)
        assert env.unwrapped._config.agent_separation_m == 0.5  # pylint: disable=protected-access
        obs, _ = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (603,) and isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "reward/hit_dealt/agent_0" in info
    finally:
        env.close()


def test_level1_info_has_all_terms(level1_env):
    _, info = level1_env.reset(seed=0)
    for name in (
        "hit_dealt",
        "hit_received",
        "alive",
        "reach",
        "facing",
        "effort",
        "fall",
        "ko",
    ):
        assert f"reward/{name}/agent_0" in info
    _, rew, _, _, info = level1_env.step(_zeros())
    total = sum(
        v for k, v in info.items() if k.startswith("reward/") and k.endswith("agent_0")
    )
    assert total == pytest.approx(rew["agent_0"])
    assert info["reward/facing/agent_0"] > 0.0  # starts roughly facing the dummy
