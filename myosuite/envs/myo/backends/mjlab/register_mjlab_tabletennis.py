# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Register MyoChallenge TableTennis with mjlab (ManagerBasedRlEnvCfg + task registry)."""

from __future__ import annotations

import functools
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.actuator import XmlActuatorCfg as _XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from scipy.spatial.transform import Rotation as R

from myosuite.core.model_builder import build_from_recipe, cached_spec
from myosuite.core.model_recipes import (
    _add_tabletennis_furniture,
    _tabletennis_body_spec,
)
from myosuite.envs.myo.backends.mjlab.mjlab_env_base import normalize_mjlab_env_ids
from myosuite.envs.myo.backends.mjlab.tasks.mdp import SYNC_TERM, sync_forward
from myosuite.envs.myo.backends.mjlab.configs.table_tennis_cfg import TableTennisCfg
from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import mujoco_cfg_from_model
from myosuite.envs.myo.tasks.challenge.tabletennis import (
    ContactTrajIssue,
)
from myosuite.terms.base_action import sigmoid_muscle_activation

logger = logging.getLogger(__name__)

_TT_ENTITY_NAME = "table_tennis_robot"
_TT_PADDLE_ENTITY_NAME = "paddle"
_TT_BALL_ENTITY_NAME = "pingpong"
_MAX_TIME = 3.0
_TT_RWD_WEIGHTS: dict[str, float] = {
    "reach_dist": 1.0,
    "palm_dist": 1.0,
    "paddle_quat": 2.0,
    "act_reg": 0.5,
    "torso_up": 2.0,
    "sparse": 100.0,
    "solved": 1000.0,
    "done": -10.0,
}

# Columns of the CPU ``touching_info`` observation (``_ball_label_to_obs``).
_PADDLE, _OWN, _OPPONENT, _NET, _GROUND, _ENV = range(6)
_NUM_LABELS = 6

# ``evaluate_pingpong_trajectory`` results as int codes: the ContactTrajIssue
# values, MISS while nothing is decided yet, and _SOLVED for its ``None``.
_UNDECIDED = ContactTrajIssue.MISS.value
_SOLVED = -1
_FAILED = (
    ContactTrajIssue.OWN_HALF.value,
    ContactTrajIssue.NO_PADDLE.value,
    ContactTrajIssue.DOUBLE_TOUCH.value,
)

_BALL_DROP_Z = 0.3  # CPU ``_get_done``: a ball below this height ends the rally
_BALL_LAUNCH_VEL = (5.6, 1.6, 0.1)  # CPU ``start_vel`` (P0/P1)
# CPU ``cal_ball_qvel``: a sampled launch lands between these table points.
_TABLE_UPPER = (1.35, 0.70, 0.785)
_TABLE_LOWER = (0.5, -0.60, 0.785)
_GRAVITY = 9.81


@functools.cache
def _reference_model() -> mujoco.MjModel:
    """CPU model of ``myoChallengeTableTennisP*-v0`` (same recipe); read-only."""
    model, _ = build_from_recipe("challenge_tabletennis")
    return model


@dataclass(frozen=True)
class _TTReference:
    """CPU keyframe and actuator data the mjlab scene reproduces."""

    arm_joint_pos: dict[str, float]
    paddle_pose: tuple[float, ...]
    ball_pose: tuple[float, ...]
    muscle_mask: np.ndarray
    ctrlrange: np.ndarray
    init_paddle_quat: tuple[float, ...]


@functools.cache
def _tt_reference() -> _TTReference:
    model = _reference_model()
    key = model.key_qpos[0]

    def pose(joint: str) -> tuple[float, ...]:
        adr = int(model.joint(joint).qposadr[0])
        return tuple(float(q) for q in key[adr : adr + 7])

    # Anchored: mjlab matches init_state joint patterns as regex prefixes.
    arm_joint_pos = {
        re.escape(model.joint(j).name) + "$": float(key[model.jnt_qposadr[j]])
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    }
    # Intrinsic "XYZ" matches the keyframe paddle orientation (as on the CPU env).
    init_paddle_quat = R.from_euler(
        "XYZ", np.array([-0.3, 1.57, 0]), degrees=False
    ).as_quat()[[3, 0, 1, 2]]
    return _TTReference(
        arm_joint_pos=arm_joint_pos,
        paddle_pose=pose("paddle_freejoint"),
        ball_pose=pose("pingpong_freejoint"),
        muscle_mask=np.asarray(
            model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE, dtype=bool
        ),
        ctrlrange=np.asarray(model.actuator_ctrlrange, dtype=np.float64),
        init_paddle_quat=tuple(float(q) for q in init_paddle_quat),
    )


@functools.cache
def _tt_actuator_xml_groups() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (muscle tendon names, position actuator names) for mjlab Xml wrapping."""
    m = _reference_model()
    tendons: list[str] = []
    positions: list[str] = []
    for i in range(m.nu):
        aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if m.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
            tid = int(m.actuator_trnid[i, 0])
            tname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, tid)
            if tname is not None:
                tendons.append(tname)
        else:
            if aname is not None:
                positions.append(aname)
    return tuple(tendons), tuple(positions)


@cached_spec()
def _full_spec_in(cwd: str) -> mujoco.MjSpec:
    """Build the same torso+arms+legs+furniture spec as the CPU recipe.

    *cwd* only keys the cache: the furniture mesh paths are written relative to
    the working directory.

    Reuses ``model_recipes._tabletennis_body_spec``/``_add_tabletennis_furniture``
    directly (not :func:`build_from_recipe`) so mjlab and CPU always compose
    the identical un-keyframed spec — the mjlab entity/scene split below then
    strips/re-adds pieces from this one shared spec instead of maintaining a
    second, independently-written composition.
    """
    return _add_tabletennis_furniture(_tabletennis_body_spec())


def _table_tennis_full_spec() -> mujoco.MjSpec:
    """Private copy of the composed full spec (the mjlab entities split it three times).

    Reuses ``model_recipes._tabletennis_body_spec``/``_add_tabletennis_furniture``
    through :func:`_full_spec_in`, kept per working directory.
    """
    return _full_spec_in(os.getcwd())


def _table_tennis_spec_fn() -> mujoco.MjSpec:
    """Athlete and table; the paddle and the ball are entities of their own.

    mjlab resets a floating body only through the root of its entity, so each
    free body of the CPU model gets its own entity. Without a free joint mjlab
    mounts this entity on a mocap body at the origin, which keeps the recipe's
    calibrated root pose (the CPU world pose).
    """
    spec = _table_tennis_full_spec()
    # The velocimeters move with their bodies (see _free_body_spec).
    for s in list(spec.sensors):
        if s.name in ("pingpong_vel_sensor", "paddle_vel_sensor"):
            spec.delete(s)
    for name in ("paddle", "pingpong"):
        spec.delete(next(b for b in spec.bodies if b.name == name))
    return spec


def _free_body_spec(body_name: str) -> mujoco.MjSpec:
    """Spec holding one free body of the full spec, with its velocimeter."""
    full = _table_tennis_full_spec()
    # The attached body is compiled with the compiler options of its source spec;
    # its boundinertia (1e-4) would raise the 7.2e-7 inertia of the ball.
    full.compiler.boundinertia = 0.0
    # ``full.body(name)`` returns None for this composed/attached spec
    # (name lookup isn't populated pre-compile for attached subtrees) —
    # iterate instead.
    body = next(b for b in full.bodies if b.name == body_name)
    spec = mujoco.MjSpec()
    # attach_body also carries the sensor that reads the body's site.
    spec.worldbody.add_frame().attach_body(body, "", "")
    return spec


def _paddle_spec_fn() -> mujoco.MjSpec:
    """Minimal spec containing only the paddle (freejoint + site + geoms)."""
    return _free_body_spec("paddle")


def _pingpong_spec_fn() -> mujoco.MjSpec:
    """Minimal spec containing only the pingpong ball (freejoint + site + geom)."""
    return _free_body_spec("pingpong")


def _free_body_init_state(
    pose: tuple[float, ...], lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> EntityCfg.InitialStateCfg:
    """Root state of a free-body entity from a free-joint qpos (pos + wxyz quat)."""
    return EntityCfg.InitialStateCfg(
        pos=(pose[0], pose[1], pose[2]),
        rot=(pose[3], pose[4], pose[5], pose[6]),
        lin_vel=lin_vel,
    )


def _ball_contact_labels(
    nacon: torch.Tensor,
    geom: torch.Tensor,
    worldid: torch.Tensor,
    geom_body: torch.Tensor,
    ball_body: int,
    label_of_geom: torch.Tensor,
    num_envs: int,
) -> torch.Tensor:
    """Labels touching the ball, per env (CPU ``get_ball_contact_labels``).

    Args:
        nacon: Number of live contacts over all worlds, shape ``(1,)``. Rows at
            or past it keep contacts of earlier collision passes.
        geom: Geom pair of each contact row, ``(naconmax, 2)``.
        worldid: World of each contact row, ``(naconmax,)``.
        geom_body: Body id of each geom.
        ball_body: Body id of the ball.
        label_of_geom: ``touching_info`` column of each geom.
        num_envs: Number of worlds.

    Returns:
        ``(num_envs, 6)`` bool, one flag per ``touching_info`` column (the CPU
        label set). Computed on device, without host syncs.
    """
    live = torch.arange(geom.shape[0], device=geom.device) < nacon.reshape(1)
    pair = geom.long().clamp(0, geom_body.shape[0] - 1)
    body = geom_body[pair]
    ball_first = body[:, 0] == ball_body
    hit = live & (ball_first | (body[:, 1] == ball_body))
    other = torch.where(ball_first, pair[:, 1], pair[:, 0])
    world = worldid.long().clamp(0, num_envs - 1)
    dropped = num_envs * _NUM_LABELS  # slot for rows that are not ball contacts
    slot = torch.where(hit, world * _NUM_LABELS + label_of_geom[other], dropped)
    flags = torch.zeros(dropped + 1, dtype=torch.bool, device=geom.device)
    flags[slot] = True
    return flags[:dropped].view(num_envs, _NUM_LABELS)


class _BallContacts:
    """:func:`_ball_contact_labels` bound to the ids of one scene."""

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        model = env.sim.mj_model
        label_of_geom = torch.full(
            (model.ngeom,), _ENV, dtype=torch.long, device=env.device
        )
        for column, geom in (
            (_PADDLE, f"{_TT_PADDLE_ENTITY_NAME}/pad"),
            (_OWN, f"{_TT_ENTITY_NAME}/coll_own_half"),
            (_OPPONENT, f"{_TT_ENTITY_NAME}/coll_opponent_half"),
            (_NET, f"{_TT_ENTITY_NAME}/coll_net"),
            (_GROUND, f"{_TT_ENTITY_NAME}/ground"),
        ):
            label_of_geom[model.geom(geom).id] = column
        self._label_of_geom = label_of_geom
        self._geom_body = torch.as_tensor(
            model.geom_bodyid, dtype=torch.long, device=env.device
        )
        self._ball_body = env.scene[_TT_BALL_ENTITY_NAME].indexing.root_body_id
        self._num_envs = env.num_envs

    def __call__(self, data: Any) -> torch.Tensor:
        return _ball_contact_labels(
            data.nacon,
            data.contact.geom,
            data.contact.worldid,
            self._geom_body,
            self._ball_body,
            self._label_of_geom,
            self._num_envs,
        )


def _trajectory_outcome(code: int) -> ContactTrajIssue | None:
    """``evaluate_pingpong_trajectory`` result of an outcome code."""
    return None if code == _SOLVED else ContactTrajIssue(code)


class _PingpongTrajectory:
    """Incremental, vectorized ``evaluate_pingpong_trajectory``.

    The CPU function re-scans the whole contact trajectory and returns at its
    first decisive contact set, so feeding one set per step and keeping the
    first decision gives the CPU result after every step.
    """

    def __init__(self, num_envs: int, device: Any) -> None:
        def zeros(dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(num_envs, dtype=dtype, device=device)

        self.hit_paddle = zeros(torch.bool)  # has_hit_paddle
        self.left_paddle = zeros(torch.bool)  # has_bounced_from_paddle
        self.bounced = zeros(torch.bool)  # has_bounced_from_table
        self.bounce_over = zeros(torch.bool)  # own_contact_phase_done
        self.own_count = zeros(torch.int32)
        self.outcome = torch.full(
            (num_envs,), _UNDECIDED, dtype=torch.int32, device=device
        )

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        """Empty the trajectory of *env_ids*."""
        for flag in (self.hit_paddle, self.left_paddle, self.bounced, self.bounce_over):
            flag[env_ids] = False
        self.own_count[env_ids] = 0
        self.outcome[env_ids] = _UNDECIDED

    def restart(self, mask: torch.Tensor) -> None:
        """Empty the trajectory where the bool *mask* is set (no host sync)."""
        for flag in (self.hit_paddle, self.left_paddle, self.bounced, self.bounce_over):
            flag.masked_fill_(mask, False)
        self.own_count.masked_fill_(mask, 0)
        self.outcome.masked_fill_(mask, _UNDECIDED)

    def update(self, labels: torch.Tensor) -> None:
        """Append one contact set per env (``(num_envs, 6)`` bool)."""
        paddle = labels[:, _PADDLE]
        own = labels[:, _OWN]
        opponent = labels[:, _OPPONENT]
        left_paddle = self.left_paddle | (~paddle & self.hit_paddle)
        double_touch = paddle & left_paddle
        hit_paddle = self.hit_paddle | paddle
        first_own = own & ~self.bounced
        more_own = own & self.bounced & ~self.bounce_over
        own_count = torch.where(
            first_own,
            torch.ones_like(self.own_count),
            self.own_count + more_own.to(self.own_count.dtype),
        )
        long_bounce = more_own & (own_count > 2)
        own_half = long_bounce | (own & self.bounced & self.bounce_over)
        # The first matching check of the CPU loop decides the step.
        step = torch.full_like(self.outcome, _UNDECIDED)
        step = torch.where(
            opponent & ~hit_paddle, ContactTrajIssue.NO_PADDLE.value, step
        )
        step = torch.where(opponent & hit_paddle, _SOLVED, step)
        step = torch.where(own_half, ContactTrajIssue.OWN_HALF.value, step)
        step = torch.where(double_touch, ContactTrajIssue.DOUBLE_TOUCH.value, step)
        # A decided trajectory keeps its outcome (its flags no longer matter).
        self.outcome = torch.where(self.outcome == _UNDECIDED, step, self.outcome)
        self.left_paddle = left_paddle
        self.hit_paddle = hit_paddle
        self.bounce_over = self.bounce_over | long_bounce | (~own & self.bounced)
        self.bounced = self.bounced | own
        self.own_count = own_count


@dataclass(frozen=True)
class _BallLaunch:
    """Launch ranges of the ball on the env device, uploaded once per term.

    Attributes:
        xyz_low: Lower corner of the sampled launch position; ``None``: keyframe.
        xyz_high: Upper corner of the sampled launch position.
        sample_vel: Whether the launch velocity is sampled (``ball_qvel``).
        table_upper: :data:`_TABLE_UPPER`.
        table_lower: :data:`_TABLE_LOWER`.
    """

    xyz_low: torch.Tensor | None
    xyz_high: torch.Tensor | None
    sample_vel: bool
    table_upper: torch.Tensor
    table_lower: torch.Tensor


def _ball_launch(tt_cfg: TableTennisCfg, device: Any) -> _BallLaunch:
    """The :class:`_BallLaunch` of *tt_cfg* on *device*."""
    xyz = tt_cfg.ball_xyz_range
    return _BallLaunch(
        xyz_low=None if xyz is None else torch.tensor(xyz["low"], device=device),
        xyz_high=None if xyz is None else torch.tensor(xyz["high"], device=device),
        sample_vel=bool(tt_cfg.ball_qvel),
        table_upper=torch.tensor(_TABLE_UPPER, dtype=torch.float32, device=device),
        table_lower=torch.tensor(_TABLE_LOWER, dtype=torch.float32, device=device),
    )


def _sample_ball_lin_vel(pos: torch.Tensor, launch: _BallLaunch) -> torch.Tensor:
    """CPU ``cal_ball_qvel`` + uniform draw, for every row of *pos*."""
    n, device = pos.shape[0], pos.device
    upper, lower = launch.table_upper, launch.table_lower
    v_z = torch.rand(n, dtype=pos.dtype, device=device) * 0.2 - 0.1
    a = -0.5 * _GRAVITY
    c = pos[:, 2] - upper[2]
    t = (-v_z - torch.sqrt(torch.clamp(v_z * v_z - 4 * a * c, min=0.0))) / (2 * a)
    v_low = (lower[:2] - pos[:, :2]) / t[:, None]
    v_high = (upper[:2] - pos[:, :2]) / t[:, None]
    v_xy = v_low + torch.rand(n, 2, dtype=pos.dtype, device=device) * (v_high - v_low)
    return torch.cat([v_xy, v_z[:, None]], dim=-1)


def _ball_launch_state(
    ball: Any, env_ids: torch.Tensor, launch: _BallLaunch
) -> torch.Tensor:
    """Root state of a (re)launched ball: CPU keyframe or sampled pos/vel."""
    state = ball.data.default_root_state[env_ids].clone()
    n, device = state.shape[0], state.device
    if launch.xyz_low is not None and launch.xyz_high is not None:
        low, high = launch.xyz_low, launch.xyz_high
        state[:, :3] = low + torch.rand(n, 3, device=device) * (high - low)
        if launch.sample_vel:
            state[:, 7:10] = _sample_ball_lin_vel(state[:, :3], launch)
    return state


@dataclass(kw_only=True)
class TableTennisMixedCtrlActionCfg(ActionTermCfg):
    """Config for mixed muscle + position actuator control (TableTennis)."""

    def build(self, env: ManagerBasedRlEnv) -> TableTennisMixedCtrlAction:
        return TableTennisMixedCtrlAction(self, env)


class TableTennisMixedCtrlAction(ActionTerm):
    """Map policy actions in [-1, 1] to MuJoCo ctrl (muscles + pelvis position)."""

    cfg: TableTennisMixedCtrlActionCfg

    def __init__(
        self, cfg: TableTennisMixedCtrlActionCfg, env: ManagerBasedRlEnv
    ) -> None:
        super().__init__(cfg=cfg, env=env)
        ref = _tt_reference()
        self._nu = int(ref.muscle_mask.shape[0])
        self._muscle_mask = torch.as_tensor(ref.muscle_mask, device=self.device)
        lo = torch.as_tensor(
            ref.ctrlrange[:, 0], dtype=torch.float32, device=self.device
        )
        hi = torch.as_tensor(
            ref.ctrlrange[:, 1], dtype=torch.float32, device=self.device
        )
        self._mid = 0.5 * (lo + hi)
        self._half = 0.5 * (hi - lo)
        self._raw_actions = torch.zeros((self.num_envs, self._nu), device=self.device)
        self._processed = torch.zeros_like(self._raw_actions)
        # The XmlActuatorCfg groups rewrite ctrl from the actuator targets before
        # every physics step, so write targets: tendon effort for the muscles,
        # joint position for the pelvis position actuators.
        tendon_names, joint_names = _tt_actuator_xml_groups()
        tendon_ids, _ = self._entity.find_tendons(tendon_names, preserve_order=True)
        joint_ids, _ = self._entity.find_joints(joint_names, preserve_order=True)
        self._muscle_cols = torch.nonzero(self._muscle_mask).squeeze(-1)
        self._position_cols = torch.nonzero(~self._muscle_mask).squeeze(-1)
        self._tendon_ids = torch.tensor(
            tendon_ids, dtype=torch.long, device=self.device
        )
        self._joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._nu

    @property
    def raw_action(self) -> Any:
        return self._raw_actions

    def process_actions(self, actions: Any) -> None:
        a = torch.clamp(actions.to(self.device), -1.0, 1.0)
        self._raw_actions[:] = a
        m = self._muscle_mask[None, :].expand_as(a)
        ctrl = torch.where(
            m, sigmoid_muscle_activation(a, torch), self._mid + a * self._half
        )
        self._processed[:] = ctrl

    def apply_actions(self) -> None:
        self._entity.set_tendon_effort_target(
            self._processed[:, self._muscle_cols], tendon_ids=self._tendon_ids
        )
        self._entity.set_joint_position_target(
            self._processed[:, self._position_cols], joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Any | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0


class TableTennisObservation(ManagerTermBase):
    """CPU ``TableTennisEnv`` observation vector (417-d): same order and units."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._arm = env.scene[_TT_ENTITY_NAME]
        self._paddle = env.scene[_TT_PADDLE_ENTITY_NAME]
        self._ball = env.scene[_TT_BALL_ENTITY_NAME]
        self._pelvis = self._arm.find_sites(["pelvis"])[0][0]
        self._paddle_site = self._paddle.find_sites(["paddle"])[0][0]
        self._ball_site = self._ball.find_sites(["pingpong"])[0][0]
        self._ball_vel = env.scene[f"{_TT_BALL_ENTITY_NAME}/pingpong_vel_sensor"]
        self._paddle_vel = env.scene[f"{_TT_PADDLE_ENTITY_NAME}/paddle_vel_sensor"]
        self._contacts = _BallContacts(env)
        self._has_act = int(env.sim.mj_model.na) > 0

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        ball_pos = self._ball.data.site_pos_w[:, self._ball_site]
        paddle_pos = self._paddle.data.site_pos_w[:, self._paddle_site]
        parts = [
            self._arm.data.site_pos_w[:, self._pelvis],
            self._arm.data.joint_pos,
            self._arm.data.joint_vel,
            ball_pos,
            self._ball_vel.data,
            paddle_pos,
            self._paddle_vel.data,
            self._paddle.data.root_link_quat_w,
            paddle_pos - ball_pos,
            # Contacts are fresh here: mjlab runs forward() before observations.
            self._contacts(env.sim.data).float(),
        ]
        if self._has_act:
            # accepted: no entity.data API for muscle activation
            parts.append(self._arm.data.data.act)
        return torch.cat([p.to(dtype=torch.float32) for p in parts], dim=-1)


def _rally_done(
    timed_out: torch.Tensor,
    ball_z: torch.Tensor,
    solved: torch.Tensor,
    outcome: torch.Tensor,
) -> torch.Tensor:
    """CPU ``_get_done``: time up, ball dropped, solved, or a failed rally."""
    failed = (outcome == _FAILED[0]) | (outcome == _FAILED[1]) | (outcome == _FAILED[2])
    return timed_out | (ball_z < _BALL_DROP_Z) | solved | failed


class TableTennisRally(ManagerTermBase):
    """Contacts, rally outcome and ``done`` of the step (the ``task_done`` term).

    CPU ``step()`` runs ``mj_forward`` and derives the contact set, the reward
    and ``done`` from that one post-step state. The ``sync_forward`` term
    refreshes the same state for mjlab before this term, which appends the
    step's ball contacts to the trajectory and keeps the step's flags for
    :class:`TableTennisReward`. The bonus and the penalty are thus paid once,
    on the step that ends the rally, as on CPU.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        tt_cfg: TableTennisCfg = cfg.params["tt_cfg"]
        n, device = env.num_envs, env.device
        self._contacts = _BallContacts(env)
        self._trajectory = _PingpongTrajectory(n, device)
        self._ball = env.scene[_TT_BALL_ENTITY_NAME]
        self._ball_site = self._ball.find_sites(["pingpong"])[0][0]
        self._rally_count = int(tt_cfg.rally_count)
        # CPU compares the float64 sim time with 3 s (false at 300 steps of
        # 10 ms); counting steps avoids float32 time drift.
        self._max_steps = round(_MAX_TIME / env.step_dt)
        self.labels = torch.zeros(n, _NUM_LABELS, dtype=torch.bool, device=device)
        self.solved = torch.zeros(n, dtype=torch.bool, device=device)
        self.done = torch.zeros_like(self.solved)
        self.relaunch = torch.zeros_like(self.solved)
        self._rallies = torch.zeros(n, dtype=torch.long, device=device)
        self._rally_start = torch.zeros(n, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._trajectory.reset(ids)
        for flag in (self.labels, self.solved, self.done, self.relaunch):
            flag[ids] = False
        self._rallies[ids] = 0
        self._rally_start[ids] = 0

    def __call__(self, env: ManagerBasedRlEnv, tt_cfg: TableTennisCfg) -> torch.Tensor:
        del tt_cfg  # read at construction
        self.labels = self._contacts(env.sim.data)
        self._trajectory.update(self.labels)
        outcome = self._trajectory.outcome
        self.solved = outcome == _SOLVED
        self.done = _rally_done(
            env.episode_length_buf - self._rally_start > self._max_steps,
            self._ball.data.site_pos_w[:, self._ball_site, 2],
            self.solved,
            outcome,
        )
        # CPU rally bookkeeping: a solved rally short of rally_count goes on
        # with a fresh trajectory and clock (the ball is relaunched by an event).
        self._rallies += self.solved.long()
        self.relaunch = self.solved & (self._rallies < self._rally_count)
        self._rally_start = torch.where(
            self.relaunch, env.episode_length_buf, self._rally_start
        )
        self._trajectory.restart(self.relaunch)
        return self.done & ~self.relaunch


def _rally_term(env: ManagerBasedRlEnv) -> TableTennisRally:
    return env.termination_manager.get_term_cfg("task_done").func


class TableTennisReward(ManagerTermBase):
    """CPU ``get_reward_dict()["dense"]`` with ``_TT_RWD_WEIGHTS``, for all envs."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._arm = env.scene[_TT_ENTITY_NAME]
        self._paddle = env.scene[_TT_PADDLE_ENTITY_NAME]
        self._ball = env.scene[_TT_BALL_ENTITY_NAME]
        self._grasp = self._arm.find_sites(["S_grasp"])[0][0]
        self._paddle_site = self._paddle.find_sites(["paddle"])[0][0]
        self._ball_site = self._ball.find_sites(["pingpong"])[0][0]
        self._flex = self._arm.find_joints(["flex_extension"])[0][0]
        self._init_quat = torch.tensor(
            _tt_reference().init_paddle_quat, dtype=torch.float32, device=env.device
        )
        self._na = int(env.sim.mj_model.na)

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        rally = _rally_term(env)
        w = _TT_RWD_WEIGHTS
        paddle_pos = self._paddle.data.site_pos_w[:, self._paddle_site]
        ball_pos = self._ball.data.site_pos_w[:, self._ball_site]
        palm_pos = self._arm.data.site_pos_w[:, self._grasp]
        reach_dist = torch.linalg.norm(paddle_pos - ball_pos, dim=-1)
        palm_dist = torch.linalg.norm(palm_pos - paddle_pos, dim=-1)
        quat_err = torch.linalg.norm(
            self._paddle.data.root_link_quat_w - self._init_quat, dim=-1
        )
        torso_err = torch.abs(self._arm.data.joint_pos[:, self._flex])
        if self._na:
            # accepted: no entity.data API for muscle activation
            act_mag = torch.linalg.norm(self._arm.data.data.act, dim=-1) / self._na
        else:
            act_mag = torch.zeros_like(reach_dist)
        return (
            w["reach_dist"] * torch.exp(-1.0 * reach_dist)
            + w["palm_dist"] * torch.exp(-5.0 * palm_dist)
            + w["paddle_quat"] * torch.exp(-5.0 * quat_err)
            + w["torso_up"] * torch.exp(-5.0 * torso_err)
            - w["act_reg"] * act_mag
            + w["sparse"] * rally.labels[:, _PADDLE]
            + w["solved"] * rally.solved
            + w["done"] * rally.done
        )


class TableTennisReset(ManagerTermBase):
    """CPU ``reset()``: keyframe pose (+ optional joint noise), paddle back in
    the hand and the ball (re)launched from the keyframe or a sampled state."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        tt_cfg: TableTennisCfg = cfg.params["tt_cfg"]
        self._arm = env.scene[_TT_ENTITY_NAME]
        self._paddle = env.scene[_TT_PADDLE_ENTITY_NAME]
        self._ball = env.scene[_TT_BALL_ENTITY_NAME]
        self.launch = _ball_launch(tt_cfg, env.device)
        self._noise: tuple[torch.Tensor, ...] | None = None
        if tt_cfg.qpos_noise_range is not None:
            # CPU: uniform fraction of each joint range, clipped to the raw
            # jnt_range (which is [0, 0] for unlimited joints).
            joint_ids = self._arm.indexing.joint_ids.cpu().numpy()
            jnt_range = env.sim.mj_model.jnt_range[joint_ids]
            n = len(joint_ids)
            qr = tt_cfg.qpos_noise_range

            def bound(key: str, default: float) -> np.ndarray:
                v = np.asarray(qr.get(key, default), dtype=np.float64).ravel()
                return np.full(n, v[0]) if v.size == 1 else v[:n]

            self._noise = tuple(
                torch.as_tensor(v, dtype=torch.float32, device=env.device)
                for v in (
                    bound("low", 0.0),
                    bound("high", 1.0),
                    jnt_range[:, 0],
                    jnt_range[:, 1],
                )
            )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        tt_cfg: TableTennisCfg,
    ) -> None:
        env_ids = normalize_mjlab_env_ids(env, env_ids)
        pos = self._arm.data.default_joint_pos[env_ids].clone()
        if self._noise is not None:
            low, high, j_lo, j_hi = self._noise
            frac = low + torch.rand_like(pos) * (high - low)
            pos = torch.clamp(pos + frac * (j_hi - j_lo), j_lo, j_hi)
        self._arm.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=env_ids)
        self._paddle.write_root_state_to_sim(
            self._paddle.data.default_root_state[env_ids], env_ids=env_ids
        )
        self._ball.write_root_state_to_sim(
            _ball_launch_state(self._ball, env_ids, self.launch), env_ids=env_ids
        )


def _tt_relaunch_ball(
    env: ManagerBasedRlEnv, env_ids: None, tt_cfg: TableTennisCfg
) -> None:
    """Step event: relaunch the ball of envs that go on to their next rally."""
    del env_ids, tt_cfg  # step events cover all envs; launch ranges of tt_reset
    # Host sync; only registered when rally_count > 1.
    ids = _rally_term(env).relaunch.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel():
        ball = env.scene[_TT_BALL_ENTITY_NAME]
        launch = env.event_manager.get_term_cfg("tt_reset").func.launch
        ball.write_root_state_to_sim(_ball_launch_state(ball, ids, launch), env_ids=ids)


def _table_tennis_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    # Match register_mjlab_tasks: env observation group is ``policy`` (flat 417-d).
    _policy_obs_groups: dict[str, tuple[str, ...]] = {
        "actor": ("policy",),
        "critic": ("policy",),
    }
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=None,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="myo_table_tennis",
        save_interval=100,
        num_steps_per_env=48,
        max_iterations=500,
        obs_groups=dict(_policy_obs_groups),
    )


def _tt_dr_events(tt_cfg: TableTennisCfg) -> dict[str, EventTermCfg]:
    """Per-env P2 domain randomization of the paddle mass and the ball friction."""
    events: dict[str, EventTermCfg] = {}
    if tt_cfg.paddle_mass_range is not None:
        # Mass only, inertia kept, as on CPU (hence dr.body_mass's warning).
        events["paddle_mass"] = EventTermCfg(
            func=dr.body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    _TT_PADDLE_ENTITY_NAME, body_names=("paddle",)
                ),
                "ranges": tuple(tt_cfg.paddle_mass_range),
                "operation": "abs",
            },
        )
    if tt_cfg.ball_friction_range is not None:
        low = tt_cfg.ball_friction_range["low"]
        high = tt_cfg.ball_friction_range["high"]
        events["ball_friction"] = EventTermCfg(
            func=dr.geom_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    _TT_BALL_ENTITY_NAME, geom_names=("pingpong",)
                ),
                "ranges": {axis: (low[axis], high[axis]) for axis in range(3)},
                "axes": [0, 1, 2],
                "operation": "abs",
            },
        )
    return events


def make_table_tennis_mjlab_env_cfg(tt_cfg: TableTennisCfg) -> ManagerBasedRlEnvCfg:
    """Build mjlab ``ManagerBasedRlEnvCfg`` for TableTennis (vectorised)."""
    ref = _tt_reference()
    tendon_names, pos_names = _tt_actuator_xml_groups()
    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlActuatorCfg(
                target_names_expr=tendon_names,
                transmission_type=TransmissionType.TENDON,
            ),
            _XmlActuatorCfg(
                target_names_expr=pos_names,
                transmission_type=TransmissionType.JOINT,
            ),
        ),
    )

    # The CPU keyframe: arm joints, paddle in the hand, ball launch pose.
    entity_cfg = EntityCfg(
        spec_fn=_table_tennis_spec_fn,
        articulation=articulation,
        init_state=EntityCfg.InitialStateCfg(joint_pos=dict(ref.arm_joint_pos)),
    )
    paddle_entity_cfg = EntityCfg(
        spec_fn=_paddle_spec_fn, init_state=_free_body_init_state(ref.paddle_pose)
    )
    ball_entity_cfg = EntityCfg(
        spec_fn=_pingpong_spec_fn,
        init_state=_free_body_init_state(ref.ball_pose, _BALL_LAUNCH_VEL),
    )
    scene_cfg = SceneCfg(
        num_envs=int(tt_cfg.num_envs),
        # Same order as the CPU qpos: athlete, paddle, ball.
        entities={
            _TT_ENTITY_NAME: entity_cfg,
            _TT_PADDLE_ENTITY_NAME: paddle_entity_cfg,
            _TT_BALL_ENTITY_NAME: ball_entity_cfg,
        },
    )

    decimation = max(1, int(round(tt_cfg.ctrl_dt / tt_cfg.sim_dt)))
    episode_length_s = float(tt_cfg.max_episode_steps) * float(tt_cfg.ctrl_dt)

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "table_tennis_vec": ObservationTermCfg(func=TableTennisObservation),
            },
        ),
    }
    actions = {"ctrl": TableTennisMixedCtrlActionCfg(entity_name=_TT_ENTITY_NAME)}
    terminations = {
        # First: contacts and positions of the post-step state, as on CPU.
        SYNC_TERM: TerminationTermCfg(func=sync_forward),
        "time_out": TerminationTermCfg(
            func=mdp_terminations.time_out,
            time_out=True,
        ),
        "task_done": TerminationTermCfg(
            func=TableTennisRally,
            params={"tt_cfg": tt_cfg},
            time_out=False,
        ),
    }
    rewards = {
        "dense": RewardTermCfg(func=TableTennisReward, weight=1.0),
    }
    events = {
        "tt_reset": EventTermCfg(
            func=TableTennisReset,
            mode="reset",
            params={"tt_cfg": tt_cfg},
        ),
        **_tt_dr_events(tt_cfg),
    }
    if tt_cfg.rally_count > 1:
        events["tt_relaunch"] = EventTermCfg(
            func=_tt_relaunch_ball, mode="step", params={"tt_cfg": tt_cfg}
        )

    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        decimation=decimation,
        episode_length_s=episode_length_s,
        scale_rewards_by_dt=False,  # CPU rewards are per step, not per second
        observations=observations,
        actions=actions,
        terminations=terminations,
        rewards=rewards,
        events=events,
        sim=SimulationCfg(
            # Physics options of the CPU model (same recipe).
            mujoco=mujoco_cfg_from_model(
                _reference_model(), timestep=float(tt_cfg.sim_dt)
            ),
            # The myo_sim-native torso+both-arms+legs body has more
            # equality-constraint rows (nefc) than the legacy single-arm
            # chain — 512 overflowed at nefc=1071; use 1536 for headroom.
            njmax=1536,
            nconmax=1024,
        ),
    )


def register_table_tennis_mjlab_tasks() -> None:
    """Register ``myoChallengeTableTennisP{0,1,2}-v0`` with mjlab (idempotent)."""
    try:
        _reference_model()
    except Exception as exc:
        logger = logging.getLogger(__name__)
        if isinstance(exc, FileNotFoundError) or "Error opening file" in str(exc):
            logger.info("mjlab: skipping optional table tennis registration: %s", exc)
        else:
            logger.warning(
                "mjlab: skipping optional table tennis registration: %s", exc
            )
        return

    rl_cfg = _table_tennis_ppo_runner_cfg()
    tasks: tuple[tuple[str, TableTennisCfg], ...] = (
        ("myoChallengeTableTennisP0-v0", TableTennisCfg.p0()),
        ("myoChallengeTableTennisP1-v0", TableTennisCfg.p1()),
        ("myoChallengeTableTennisP2-v0", TableTennisCfg.p2()),
    )
    for task_id, cfg in tasks:
        try:
            env_cfg = make_table_tennis_mjlab_env_cfg(cfg)
            play_cfg = make_table_tennis_mjlab_env_cfg(cfg)
            register_mjlab_task(
                task_id=task_id,
                env_cfg=env_cfg,
                play_env_cfg=play_cfg,
                rl_cfg=rl_cfg,
                runner_cls=None,
            )
        except ValueError:
            pass
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Table tennis mjlab: skip %s: %s", task_id, exc
            )
