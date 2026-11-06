# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Build CPU TaskConfig / mjlab ManagerBasedRlEnvCfg from AthleticBundle."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from myosuite.core.config import (
    ActuatorGroupSpec,
    BackendConfig,
    GoalSpec,
    ObsSpec,
    RewardSpec,
    TaskConfig,
)
from myosuite.envs.modular_env import ModularTaskEnv
from myosuite.envs.myo.athletic.specs import AthleticBundle, ensure_tendons_applied
from myosuite.envs.myo.athletic.terms import default_lane_reward_terms
from myosuite.envs.myo.tasks.basic.full_body.athletic_contract import AthleticSpec

_LOG = logging.getLogger(__name__)


def task_config_from_bundle(bundle: AthleticBundle) -> TaskConfig:
    """CPU ModularTaskEnv TaskConfig for an athletic bundle."""
    spec = bundle.athletic
    weights = default_lane_reward_terms(spec)
    # ModularTaskEnv uses a single "sprint" term that reads λs from extra.
    extra = {
        "heading_dir": spec.heading_dir,
        "lambda_vel": spec.lambda_vel,
        "lambda_mid_lane": spec.lambda_mid_lane,
        "lambda_limit": spec.lambda_limit,
        "fall_height": spec.fall_height,
        "alive_bonus": spec.alive_bonus,
        "mid_lane_weight": spec.mid_lane_weight,
        "lane_width": spec.lane_width,
        "facing_angle_deg": spec.facing_angle_deg,
        "task_key": bundle.task.key,
        **dict(bundle.task.extra),
        "model": bundle.plant.key,
        "tendons": bundle.tendons,
    }
    reward_term = "sprint"
    if bundle.task.kind == "locomotion":
        reward_term = "heading"
    elif bundle.task.kind == "imitate":
        reward_term = "pose"
    elif bundle.task.kind == "vertical":
        reward_term = "sprint"
    elif bundle.task.kind == "curve":
        reward_term = "sprint"

    @dataclass
    class _AthleticTask(TaskConfig):
        model: str = bundle.plant.model_recipe
        max_episode_steps: int = int(bundle.task.episode_length_s / spec.ctrl_dt)
        backend: BackendConfig = field(
            default_factory=lambda: BackendConfig(
                n_substeps=spec.n_substeps,
                ctrl_dt=spec.ctrl_dt,
                sim_dt=spec.sim_dt,
            )
        )
        obs: ObsSpec = field(
            default_factory=lambda: ObsSpec(
                keys=list(bundle.task.obs_keys),
                extra={"heading_dir": spec.heading_dir},
            )
        )
        goal: GoalSpec = field(default_factory=lambda: GoalSpec(target_type="joint_angles"))
        reward: RewardSpec = field(
            default_factory=lambda: RewardSpec(
                terms=[reward_term],
                weights={reward_term: 1.0},
                extra=extra,
            )
        )
        actuators: list[ActuatorGroupSpec] = field(
            default_factory=lambda: [
                ActuatorGroupSpec(name="muscles", normalize_actions=True)
            ]
        )

    return _AthleticTask()


def sprint_ppo_runner_cfg(bundle: AthleticBundle | None = None) -> Any:
    """Default PPO runner for athletic mjlab tasks."""
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    name = "myo_athletic"
    if bundle is not None:
        name = f"myo_{bundle.plant.key}_{bundle.task.key}"
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            init_noise_std=1.0,
            noise_std_type="scalar",
            stochastic=True,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=name,
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=2000,
        obs_groups={"actor": ("policy",), "critic": ("policy",)},
    )


def mjlab_env_cfg_from_bundle(
    bundle: AthleticBundle,
    *,
    num_envs: int = 64,
    visual: bool = False,
    **kwargs: Any,
) -> Any:
    """Build ManagerBasedRlEnvCfg for an athletic bundle.

    **Bit-match freeze:** ``myosprinter`` × ``sprint`` with rigid tendons
    delegates to :func:`make_sprint_env_cfg` so existing TD3 policies and
    Phase 5/6 gates stay identical. Any other series element is rejected
    before that delegation.
    """
    ensure_tendons_applied(bundle)
    if (
        bundle.plant.use_golden_sprint_cfg
        and bundle.task.key == "sprint"
    ):
        from myosuite.envs.myo.backends.mjlab.tasks.myosprint.env_cfgs import (
            make_sprint_env_cfg,
        )

        return make_sprint_env_cfg(num_envs=num_envs, visual=visual, **kwargs)

    return _make_generic_athletic_env_cfg(bundle, num_envs=num_envs, visual=visual, **kwargs)


def _make_generic_athletic_env_cfg(
    bundle: AthleticBundle,
    *,
    num_envs: int = 64,
    visual: bool = False,
    **kwargs: Any,
) -> Any:
    """Manager-based cfg for non-golden athletic cells (fullbody_simple, lane variants)."""
    del visual  # reserved for future bone-mesh attachment
    import torch
    from mjlab.entity import EntityCfg
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp.events import reset_scene_to_default
    from mjlab.envs.mdp.terminations import time_out
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg

    from myosuite.envs.myo.backends.mjlab.actuators.bolt_muscle_action import (
        BoltMuscleActionCfg,
    )

    spec = bundle.athletic
    plant = bundle.plant
    heading = bundle.heading_dir
    entity = plant.entity_name

    def _spec_fn():
        if plant.load_spec is not None:
            return plant.load_spec()
        import myo_sim

        return myo_sim.load_spec(plant.key)

    def _init_state() -> EntityCfg.InitialStateCfg:
        # Do not compile the MjSpec here — registration must stay cheap.
        if plant.up_axis == "y":
            return EntityCfg.InitialStateCfg(pos=(0.0, 1.05, 0.0))
        return EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.95))

    def _obs_qpos(env: Any) -> torch.Tensor:
        return env.scene[entity].data.data.qpos

    def _obs_qvel(env: Any) -> torch.Tensor:
        return env.scene[entity].data.data.qvel

    hx, hy = float(heading[0]), float(heading[1])

    def _rew_vel(env: Any) -> torch.Tensor:
        qvel = env.scene[entity].data.data.qvel
        # freejoint: linvel is qvel[..., 0:3]; planar depends on up axis
        if plant.up_axis == "y":
            planar = qvel[..., [0, 2]]
        else:
            planar = qvel[..., :2]
        direction = torch.tensor([hx, hy], dtype=torch.float32, device=planar.device)
        return (planar * direction).sum(dim=-1)

    def _rew_mid_lane(env: Any) -> torch.Tensor:
        qpos = env.scene[entity].data.data.qpos
        if plant.up_axis == "y":
            # lateral is world-z for +X heading
            lateral = qpos[..., 2] * hx - qpos[..., 0] * 0.0
            # For Y-up Bolt frame with heading (hx, hy) in (x,z): lateral = x*hy - z*hx
            lateral = qpos[..., 0] * hy - qpos[..., 2] * hx
        else:
            lateral = qpos[..., 0] * hy - qpos[..., 1] * hx
        return torch.exp(-float(spec.mid_lane_weight) * lateral * lateral)

    def _rew_limit(env: Any) -> torch.Tensor:
        import mujoco

        m = env.sim.mj_model
        qpos = env.scene[entity].data.data.qpos
        pain = torch.zeros(qpos.shape[0], device=qpos.device)
        for j in range(int(m.njnt)):
            jtype = int(m.jnt_type[j])
            if jtype not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
                continue
            lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
            if not (hi > lo) or abs(hi - lo) > 1e6:
                continue
            adr = int(m.jnt_qposadr[j])
            q = qpos[..., adr]
            pain = pain + torch.clamp(lo - q, min=0.0) + torch.clamp(q - hi, min=0.0)
        return pain

    def _fall(env: Any) -> torch.Tensor:
        qpos = env.scene[entity].data.data.qpos
        if plant.up_axis == "y":
            height = qpos[..., 1]
        else:
            height = qpos[..., 2]
        return height < float(spec.fall_height)

    def _not_facing(env: Any) -> torch.Tensor:
        # Approximate: torso/root planar yaw vs heading via freejoint quat — soft gate.
        # Full facing uses body x-axis; here use planar vel direction when moving.
        qvel = env.scene[entity].data.data.qvel
        if plant.up_axis == "y":
            planar = qvel[..., [0, 2]]
        else:
            planar = qvel[..., :2]
        speed = torch.linalg.norm(planar, dim=-1)
        direction = torch.tensor([hx, hy], dtype=torch.float32, device=planar.device)
        # When nearly still, do not terminate on facing.
        cos_ang = (planar * direction).sum(dim=-1) / (speed + 1e-6)
        thr = float(torch.cos(torch.tensor(spec.facing_angle_deg * 3.14159265 / 180.0)))
        return (speed > 0.3) & (cos_ang < thr)

    entity_cfg = EntityCfg(spec_fn=_spec_fn, init_state=_init_state())
    scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=3.0, entities={entity: entity_cfg})

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos": ObservationTermCfg(func=_obs_qpos),
                "qvel": ObservationTermCfg(func=_obs_qvel),
            },
        ),
    }
    if bundle.action_kind == "bolt_muscle":
        actions = {
            "bolt_muscles": BoltMuscleActionCfg(
                entity_name=entity,
                default_activation=kwargs.get("default_activation"),
                tendons=bundle.tendons,
            ),
        }
    else:
        from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
            MyoMuscleActivationActionCfg,
        )

        actions = {
            "muscles": MyoMuscleActivationActionCfg(entity_name=entity),
        }

    rewards = {
        "rew_vel": RewardTermCfg(func=_rew_vel, weight=float(spec.lambda_vel)),
        "rew_mid_lane": RewardTermCfg(func=_rew_mid_lane, weight=float(spec.lambda_mid_lane)),
        "rew_limit": RewardTermCfg(func=_rew_limit, weight=float(spec.lambda_limit)),
    }
    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "fallen": TerminationTermCfg(func=_fall, time_out=False),
    }
    if "not_facing" in bundle.task.terminations:
        terminations["not_facing"] = TerminationTermCfg(func=_not_facing, time_out=False)

    gravity = (0.0, -9.80665, 0.0) if plant.up_axis == "y" else (0.0, 0.0, -9.80665)

    return ManagerBasedRlEnvCfg(
        decimation=int(spec.n_substeps),
        scene=scene_cfg,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events={
            "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
        },
        episode_length_s=float(bundle.task.episode_length_s),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=float(spec.sim_dt),
                gravity=gravity,
                ccd_iterations=500,
            ),
            njmax=768,
            nconmax=384,
        ),
    )


class AthleticModularTaskEnv(ModularTaskEnv):
    """CPU athletic env.

    The registered id fixes the skeleton. ``gym.make(id)`` is that skeleton
    with rigid tendons. ``gym.make(id, tendons="elastic")`` requests the
    series-elastic plugin. ``model=`` must match the id.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        render_mode: str | None = None,
        *,
        tendons: str = "rigid",
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        extra = task_config.reward.extra
        baked_model = str(extra.get("model", task_config.model))
        if model is not None and model != baked_model:
            raise ValueError(
                f"{baked_model!r} is fixed by this env id; "
                f"use make_athletic(..., model={model!r}, tendons={tendons!r}) "
                "or the myoFullbody / myoFullbodySimple id for that skeleton"
            )
        if tendons not in ("rigid", "elastic"):
            raise ValueError(f"unknown tendons {tendons!r}; expected 'rigid' or 'elastic'")
        if tendons != "rigid":
            raise NotImplementedError(
                "tendons='elastic' selects the series-elastic plugin "
                f"(model={baked_model!r}) but that element is not applied yet. "
                "Use tendons='rigid'."
            )
        super().__init__(task_config, render_mode=render_mode, **kwargs)
        self.tendons = tendons
        self.model_name = baked_model


def make_athletic_env(
    bundle: AthleticBundle,
    *,
    backend: str = "mjlab",
    num_envs: int = 1,
    **overrides: Any,
) -> Any:
    """Instantiate an athletic environment for ``backend`` in {cpu, mjlab}."""
    ensure_tendons_applied(bundle)
    if backend == "mjlab":
        from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

        device = overrides.pop("device", "cpu")
        # device is an env ctor arg, not a cfg field (make_sprint_env_cfg rejects it).
        cfg = mjlab_env_cfg_from_bundle(bundle, num_envs=num_envs, **overrides)
        return ManagerBasedRlEnv(cfg=cfg, device=device)
    if backend == "cpu":
        from myosuite.core.registry import make_env

        return make_env(bundle.env_id(), backend="cpu", **overrides)
    raise ValueError(f"Unknown athletic backend: {backend!r}")
