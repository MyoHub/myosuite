# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MDP terms of the ``myoChallengeChaseTagFBVsTrain-v0`` twin (CPU ``ChaseTagVsTrainEnv``).

Player 0 (the learner, CPU ``agent_0``) and player 1 (the opponent, ``agent_1``)
are two entities. Every observation term takes ``player``: the ``policy`` group
is the learner's FBVs observation, the ``opponent`` group the opponent's, which
the action term feeds to the opponent policy. The per-env role lives on the
reset event :class:`Match`. Rewards and terminations read ``qpos`` / ``qvel`` /
``act`` only (the pelvis site sits at the free-joint origin and its velocity
sensor is the free-joint linear velocity), so they need no extra ``forward()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import torch
from mjlab.managers.manager_base import ManagerTermBase

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import normalize_mjlab_env_ids
from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import MyoAction, MyoActionCfg
from myosuite.envs.myo.backends.mjlab.tasks.mdp.self_play import (
    OpponentPool,
    SelfPlayCfg,
)
from myosuite.terms.multiplayer.chase_tag_vs_reward import (
    chase_tag_dense_reward,
    steering_reward,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg

PLAYERS = ("learner", "opponent")  # entity names of CPU agent_0 / agent_1
MATCH_TERM = "match"  # reset event holding the per-env role


# ── Match state ──────────────────────────────────────────────────────────────


class Match(ManagerTermBase):
    """Reset event: draws who chases (CPU ``on_reset``: CHASE or EVADE, 50/50).

    ``chase[i]`` is ``True`` when the learner chases in env ``i`` (CPU
    ``current_task == "CHASE"``). Also holds each player's ``act`` columns
    (actuator order), uploaded once.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self.chase = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        model = env.sim.mj_model
        self.act_cols = tuple(
            torch.as_tensor(
                model.actuator_actadr[env.scene[name].indexing.ctrl_ids.cpu().numpy()],
                dtype=torch.long,
                device=env.device,
            )
            for name in PLAYERS
        )

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
        ids = normalize_mjlab_env_ids(env, env_ids)
        self.chase[ids] = torch.rand(ids.numel(), device=env.device) < 0.5


def match(env: ManagerBasedRlEnv) -> Match:
    """The match state of *env*."""
    return env.event_manager.get_term_cfg(MATCH_TERM).func


def is_chaser(env: ManagerBasedRlEnv, player: int) -> torch.Tensor:
    """Bool (N,): *player* chases this episode."""
    chase = match(env).chase
    return chase if player == 0 else ~chase


# ── State of one player (CPU layout) ─────────────────────────────────────────


def root(env: ManagerBasedRlEnv, player: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Free-joint ``qpos[:7]`` (position relative to the env origin) / ``qvel[:6]``.

    Read from ``qpos`` / ``qvel``, which are current in terminations and
    rewards (mjlab's ``root_link_*`` derive from ``xpos`` / ``cvel``, which lag
    one substep until ``forward()``).
    """
    index = env.scene[PLAYERS[player]].indexing
    qpos = env.sim.data.qpos[:, index.free_joint_q_adr.long()]
    qvel = env.sim.data.qvel[:, index.free_joint_v_adr.long()]
    return torch.cat([qpos[:, :3] - env.scene.env_origins, qpos[:, 3:]], dim=1), qvel


def pelvis_pos(env: ManagerBasedRlEnv, player: int) -> torch.Tensor:
    """World pelvis position (CPU ``pelvis_site``), (N, 3)."""
    return root(env, player)[0][:, :3]


def pelvis_vel(env: ManagerBasedRlEnv, player: int) -> torch.Tensor:
    """World pelvis linear velocity (CPU ``pelvis_site_vel`` sensor), (N, 3)."""
    return root(env, player)[1][:, :3]


def act(env: ManagerBasedRlEnv, player: int) -> torch.Tensor:
    """Muscle activations of *player*, (N, 354)."""
    # accepted: no entity.data API for muscle activation (act of the whole scene).
    return env.sim.data.act[:, match(env).act_cols[player]]


def fallen(env: ManagerBasedRlEnv, player: int, threshold: float) -> torch.Tensor:
    """Bool (N,): *player*'s pelvis below *threshold* (CPU ``is_fallen``)."""
    return pelvis_pos(env, player)[:, 2] < threshold


def pelvis_distance(env: ManagerBasedRlEnv) -> torch.Tensor:
    """3-D distance between the pelvises (CPU ``_pelvis_distance``), (N,)."""
    return torch.linalg.norm(pelvis_pos(env, 1) - pelvis_pos(env, 0), dim=1)


def tagged(env: ManagerBasedRlEnv, radius: float) -> torch.Tensor:
    """Bool (N,): the chaser is within *radius* of the runner (CPU ``is_tagged``)."""
    return pelvis_distance(env) <= radius


# ── Observation terms: FBVs get_agent_obs blocks ─────────────────────────────


def obs_qpos(env: ManagerBasedRlEnv, player: int) -> torch.Tensor:
    """Root ``qpos`` (7) + joint angles (82), (N, 89)."""
    return torch.cat(
        [root(env, player)[0], env.scene[PLAYERS[player]].data.joint_pos], dim=1
    )


def obs_qvel(env: ManagerBasedRlEnv, player: int) -> torch.Tensor:
    """Root ``qvel`` (6, angular in the body frame) + joint velocities (82), (N, 88)."""
    return torch.cat(
        [root(env, player)[1], env.scene[PLAYERS[player]].data.joint_vel], dim=1
    )


def obs_act(env: ManagerBasedRlEnv, player: int) -> torch.Tensor:
    """Muscle activations, (N, 354)."""
    return act(env, player)


def obs_pelvis(env: ManagerBasedRlEnv, player: int) -> torch.Tensor:
    """Own pelvis position and linear velocity (CPU ``own_pelvis_obs``), (N, 6)."""
    return torch.cat([pelvis_pos(env, player), pelvis_vel(env, player)], dim=1)


def obs_opponent_relative(env: ManagerBasedRlEnv, player: int) -> torch.Tensor:
    """``[rel_pos, rel_vel, dist]`` of the other pelvis (CPU ``relative_pose_obs``), (N, 7)."""
    other = 1 - player
    rel_pos = pelvis_pos(env, other) - pelvis_pos(env, player)
    rel_vel = pelvis_vel(env, other) - pelvis_vel(env, player)
    dist = torch.linalg.norm(rel_pos, dim=1, keepdim=True)
    return torch.cat([rel_pos, rel_vel, dist], dim=1)


def obs_role(env: ManagerBasedRlEnv, player: int) -> torch.Tensor:
    """``[1, 0]`` chaser / ``[0, 1]`` runner (CPU ``role_obs``), (N, 2)."""
    chaser = is_chaser(env, player).float()[:, None]
    return torch.cat([chaser, 1.0 - chaser], dim=1)


# ── Rewards (learner), terminations and metrics ──────────────────────────────


def _learner_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The learner's unclipped action this step (CPU ``actions[agent_0]``)."""
    return env.action_manager.get_term("muscles").raw_action


class ChaseTagReward(ManagerTermBase):
    """FBVs ``compute_agent_reward`` of the learner, (N,).

    Params are the reward fields of ``ChaseTagVsConfig`` (``r_*``) plus
    ``tag_radius_m`` and ``fall_pelvis_z_threshold``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config import (  # noqa: PLC0415
            ChaseTagVsConfig,
        )

        self._config = ChaseTagVsConfig(**cfg.params)

    def __call__(self, env: ManagerBasedRlEnv, **params: float) -> torch.Tensor:
        del params
        config = self._config
        threshold = config.fall_pelvis_z_threshold
        return chase_tag_dense_reward(
            torch,
            is_chaser=is_chaser(env, 0),
            tagged=tagged(env, config.tag_radius_m).float(),
            self_fell=fallen(env, 0, threshold).float(),
            opponent_fell=fallen(env, 1, threshold).float(),
            act_sq_sum=(_learner_action(env) ** 2).sum(dim=1),
            dist=pelvis_distance(env),
            config=config,
        )


class GaitSteeringReward(ManagerTermBase):
    """FBVs ``compute_gait_steering_reward`` of the learner (gait clip on the device).

    The clip frame is the episode step (CPU ``round(data.time / ctrl_dt)``).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (  # noqa: PLC0415
            gait_clip_local,
        )

        qpos, qvel = gait_clip_local(cfg.params["clip_repo"], cfg.params["clip_file"])
        f32 = {"dtype": torch.float32, "device": env.device}
        self._qpos = torch.as_tensor(qpos.copy(), **f32)
        self._qvel = torch.as_tensor(qvel.copy(), **f32)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        clip_repo: str,
        clip_file: str,
        target_speed: float,
        w_jpos: float,
        w_jvel: float,
        w_heading: float,
    ) -> torch.Tensor:
        del clip_repo, clip_file
        frame = env.episode_length_buf % self._qpos.shape[0]
        entity = env.scene[PLAYERS[0]]
        rel_xy = (pelvis_pos(env, 1) - pelvis_pos(env, 0))[:, :2]
        dense, _ = steering_reward(
            torch,
            qpos_local=entity.data.joint_pos,
            qvel_local=entity.data.joint_vel,
            ref_qpos_local=self._qpos[frame],
            ref_qvel_local=self._qvel[frame],
            rel_xy=rel_xy,
            dist_xy=torch.linalg.norm(rel_xy, dim=1, keepdim=True),
            root_lin_vel=pelvis_vel(env, 0),
            pursue=is_chaser(env, 0)[:, None],
            target_speed=target_speed,
            w_jpos=w_jpos,
            w_jvel=w_jvel,
            w_heading=w_heading,
        )
        return dense


def elapsed_s(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Episode time in seconds (CPU ``step_count * ctrl_dt``), (N,)."""
    return env.episode_length_buf * env.step_dt


def learner_won(env: ManagerBasedRlEnv, radius: float, max_time: float) -> torch.Tensor:
    """Bool (N,): the learner tagged as chaser, or survived the episode as runner."""
    tag = tagged(env, radius)
    chase = is_chaser(env, 0)
    survived = ~tag & (elapsed_s(env) >= max_time - 1e-6)
    return (tag & chase) | (survived & ~chase)


def score(env: ManagerBasedRlEnv, radius: float, max_time: float) -> torch.Tensor:
    """FBVs ``info["score"]`` of the learner (CPU ``get_info``), (N,).

    Non-zero once the episode is won by either side: ``1 - t / T`` as chaser,
    ``t / T`` as runner.
    """
    tag = tagged(env, radius)
    t = torch.round(elapsed_s(env) * 1e4) / 1e4
    over = tag | (~tag & (elapsed_s(env) >= max_time - 1e-6))
    value = torch.where(is_chaser(env, 0), 1.0 - t / max_time, t / max_time)
    return torch.where(over, value, torch.zeros_like(value))


# ── Action: learner muscles, opponent muscles from the opponent policy ───────


@dataclass(kw_only=True)
class ChaseTagVsActionCfg(MyoActionCfg):
    """Learner muscles (``[0, 1]``, as FBVs) plus the opponent's, driven in the env.

    Attributes:
        opponent: ``"self_play"`` (frozen snapshots of the learner, filled by
            ``install_self_play`` during training), ``"policy"`` (a fixed
            RSL-RL checkpoint, :attr:`opponent_checkpoint`) or ``"idle"``.
        opponent_checkpoint: ``model_*.pt`` of a ``"policy"`` opponent.
        self_play: Snapshot pool settings of a ``"self_play"`` opponent.
    """

    entity_name: str = PLAYERS[0]
    normalize_act: bool = True
    action_range: tuple[float, float] = (0.0, 1.0)
    muscle_sigmoid: bool = False
    opponent: str = "self_play"
    opponent_checkpoint: str | None = None
    self_play: SelfPlayCfg = field(default_factory=SelfPlayCfg)

    def build(self, env: ManagerBasedRlEnv) -> ChaseTagVsAction:
        return ChaseTagVsAction(self, env)


class ChaseTagVsAction(MyoAction):
    """The learner's muscles (``MyoAction``), then the opponent's from its policy.

    The opponent policy reads the ``opponent`` observation group, computed at
    the end of the previous step (CPU: ``agent_1``'s last observation). Its
    excitations are clipped to ``[0, 1]``; an empty pool or ``"idle"`` leaves
    them at zero.
    """

    cfg: ChaseTagVsActionCfg

    def __init__(self, cfg: ChaseTagVsActionCfg, env: ManagerBasedRlEnv) -> None:
        if cfg.opponent not in ("self_play", "policy", "idle"):
            raise ValueError(f"Unknown opponent {cfg.opponent!r}.")
        super().__init__(cfg, env)
        self._opponent = env.scene[PLAYERS[1]]
        model = env.sim.mj_model
        tendons = self._opponent.indexing.tendon_ids.cpu().tolist()
        targets = []
        for act_id in self._opponent.indexing.ctrl_ids.cpu().tolist():
            if int(model.actuator_trntype[act_id]) != mujoco.mjtTrn.mjTRN_TENDON:
                raise ValueError("The opponent's actuators must all be muscles.")
            targets.append(tendons.index(int(model.actuator_trnid[act_id, 0])))
        self._opp_targets = torch.tensor(targets, dtype=torch.long, device=self.device)
        self._opp_ctrl = torch.zeros(self.num_envs, len(targets), device=self.device)
        self.opponent_pool: OpponentPool | None = None
        self._opp_policy: Any = None
        if cfg.opponent == "self_play":
            self.opponent_pool = OpponentPool(cfg.self_play, self.num_envs, self.device)
        elif cfg.opponent == "policy":
            if cfg.opponent_checkpoint is None:
                raise ValueError('opponent="policy" needs opponent_checkpoint.')
            from myosuite.utils.rslrl_policy import load_rslrl_policy  # noqa: PLC0415

            policy = load_rslrl_policy(cfg.opponent_checkpoint, len(targets))
            self._opp_policy = policy.to(self.device).eval().requires_grad_(False)

    @property
    def opponent_ctrl(self) -> torch.Tensor:
        """The opponent's excitations this step, (N, 354)."""
        return self._opp_ctrl

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        obs = self._env.obs_buf.get("opponent")
        opp = None
        if obs is not None and self.opponent_pool is not None:
            opp = self.opponent_pool.act(obs)
        elif obs is not None and self._opp_policy is not None:
            with torch.no_grad():
                opp = self._opp_policy(obs)
        if opp is None:
            self._opp_ctrl.zero_()
        else:
            self._opp_ctrl[:] = torch.clamp(opp, 0.0, 1.0)

    def apply_actions(self) -> None:
        super().apply_actions()
        self._opponent.set_tendon_effort_target(
            self._opp_ctrl, tendon_ids=self._opp_targets
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._opp_ctrl[ids] = 0.0
        if self.opponent_pool is not None:
            self.opponent_pool.reset(ids)
