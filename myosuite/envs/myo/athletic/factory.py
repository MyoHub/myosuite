# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Build CPU TaskConfig / mjlab ManagerBasedRlEnvCfg from AthleticBundle."""

from __future__ import annotations

import functools
import logging

import numpy as np
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

_LOG = logging.getLogger(__name__)


def task_config_from_bundle(bundle: AthleticBundle) -> TaskConfig:
    """CPU ModularTaskEnv TaskConfig for an athletic bundle."""
    spec = bundle.athletic
    default_lane_reward_terms(spec)
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
        goal: GoalSpec = field(
            default_factory=lambda: GoalSpec(target_type="joint_angles")
        )
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

    task_cfg = _AthleticTask()
    if bundle.task.scene == "hurdles":
        # the hurdles as static colliders (World Athletics men's 110 m layout) and their marks for the
        # nearest_hurdle observation
        from myosuite.envs.myo.athletic.events import HurdleLayout, hurdle_scene

        layout = HurdleLayout()
        task_cfg.scene = hurdle_scene(layout, spec.heading_dir)
        task_cfg.obs.extra.update(
            hurdle_positions=tuple(float(x) for x in layout.positions()),
            hurdle_height=layout.height_m,
        )
    return task_cfg


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
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
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
    if bundle.plant.use_golden_sprint_cfg and bundle.task.key == "sprint":
        from myosuite.envs.myo.backends.mjlab.tasks.myosprint.env_cfgs import (
            make_sprint_env_cfg,
        )

        return make_sprint_env_cfg(num_envs=num_envs, visual=visual, **kwargs)

    return _make_generic_athletic_env_cfg(
        bundle, num_envs=num_envs, visual=visual, **kwargs
    )


@functools.cache
def _plant_muscle_targets(plant_key: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Muscle actuator names and their tendons, in actuator order (one compile per plant)."""
    from myosuite.envs.myo.athletic.catalog import PLANTS
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _muscle_actuator_names,
        _muscle_tendon_names,
    )

    plant = PLANTS[plant_key]
    if plant.load_spec is not None:
        spec = plant.load_spec()
    else:
        import myo_sim

        spec = myo_sim.load_spec(plant_key)
    model = spec.compile()
    return _muscle_actuator_names(model), _muscle_tendon_names(model)


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
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
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
            if jtype not in (
                int(mujoco.mjtJoint.mjJNT_SLIDE),
                int(mujoco.mjtJoint.mjJNT_HINGE),
            ):
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

    if bundle.action_kind == "bolt_muscle":
        entity_cfg = EntityCfg(spec_fn=_spec_fn, init_state=_init_state())
    else:
        from mjlab.actuator import XmlActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg
        from mjlab.actuator.actuator import TransmissionType

        muscle_names, tendon_names = _plant_muscle_targets(plant.key)
        entity_cfg = EntityCfg(
            spec_fn=_spec_fn,
            init_state=_init_state(),
            articulation=EntityArticulationInfoCfg(
                actuators=(
                    XmlActuatorCfg(
                        command_field="effort",
                        target_names_expr=tendon_names,
                        transmission_type=TransmissionType.TENDON,
                    ),
                )
            ),
        )
    scene_cfg = SceneCfg(
        num_envs=num_envs, env_spacing=3.0, entities={entity: entity_cfg}
    )

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
            "muscles": MyoMuscleActivationActionCfg(
                entity_name=entity,
                actuator_names=muscle_names,
                tendon_names=tendon_names,
            ),
        }

    rewards = {
        "rew_vel": RewardTermCfg(func=_rew_vel, weight=float(spec.lambda_vel)),
        "rew_mid_lane": RewardTermCfg(
            func=_rew_mid_lane, weight=float(spec.lambda_mid_lane)
        ),
        "rew_limit": RewardTermCfg(func=_rew_limit, weight=float(spec.lambda_limit)),
    }
    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "fallen": TerminationTermCfg(func=_fall, time_out=False),
    }
    if "not_facing" in bundle.task.terminations:
        terminations["not_facing"] = TerminationTermCfg(
            func=_not_facing, time_out=False
        )

    gravity = (0.0, -9.80665, 0.0) if plant.up_axis == "y" else (0.0, 0.0, -9.80665)

    return ManagerBasedRlEnvCfg(
        decimation=int(spec.n_substeps),
        scene=scene_cfg,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events={
            "reset_scene_to_default": EventTermCfg(
                func=reset_scene_to_default, mode="reset"
            ),
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

    The registered id fixes the skeleton and the tendon variant. Build through
    :func:`~myosuite.envs.myo.athletic.catalog.make_athletic` /
    :func:`~myosuite.core.registry.make_env`. ``model`` / ``tendons`` kwargs
    may only restate what the id already baked in.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        render_mode: str | None = None,
        *,
        tendons: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        extra = task_config.reward.extra
        baked_model = str(extra.get("model", task_config.model))
        baked_tendons = str(extra.get("tendons", "rigid"))
        if model is not None and model != baked_model:
            raise ValueError(
                f"{baked_model!r} is fixed by this env id; "
                f"use make_athletic(..., model={model!r}, tendons={baked_tendons!r})"
            )
        if tendons is not None and tendons != baked_tendons:
            raise ValueError(
                f"tendons={baked_tendons!r} is fixed by this env id "
                f"({task_config.model}); use make_athletic(..., tendons={tendons!r}) "
                "for the Elastic variant id"
            )
        tendons = baked_tendons
        if tendons not in ("rigid", "elastic"):
            raise ValueError(
                f"unknown tendons {tendons!r}; expected 'rigid' or 'elastic'"
            )
        if tendons != "rigid":
            raise NotImplementedError(
                "tendons='elastic' selects the series-elastic plugin "
                f"(model={baked_model!r}) but that element is not applied yet. "
                "Use tendons='rigid'."
            )
        super().__init__(task_config, render_mode=render_mode, **kwargs)
        self.tendons = tendons
        self.model_name = baked_model
        # n_substeps in the spec is for the spec's sim_dt (Bolt: 1e-4 s); the compiled MuJoCo model keeps its own
        # timestep. Keep that timestep (rounded so a whole number of substeps spans ctrl_dt), so one step is
        # ctrl_dt of simulated time rather than n_substeps * the model's timestep.
        self._command = (
            task_config.reward.extra
            if task_config.reward.extra.get("task_key") == "locomotion"
            else None
        )
        ctrl_dt = float(task_config.backend.ctrl_dt)
        if abs(self.frame_skip * self.model.opt.timestep - ctrl_dt) > 1e-9:
            n = max(1, round(ctrl_dt / float(self.model.opt.timestep)))
            self.frame_skip = n
            self.model.opt.timestep = ctrl_dt / n

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Locomotion: sample the episode's velocity command (speed, heading) into ``task_state``."""
        task_state = super().reset_task(np_random)
        ex = self._command
        if ex is not None:
            hx, hy = ex["heading_dir"]
            half = np.radians(float(ex.get("heading_range_deg", 0.0))) / 2.0
            turn = float(np_random.uniform(-half, half))
            c, s = np.cos(turn), np.sin(turn)
            task_state["heading_dir"] = (float(c * hx - s * hy), float(s * hx + c * hy))
            stand = float(np_random.uniform()) < float(ex.get("stand_prob", 0.0))
            task_state["target_speed"] = (
                0.0
                if stand
                else float(
                    np_random.uniform(
                        ex.get("speed_min", 0.0), ex.get("speed_max", 1.0)
                    )
                )
            )
        return task_state


def make_athletic_env(
    bundle: AthleticBundle,
    *,
    backend: str = "mjlab",
    num_envs: int = 1,
    **overrides: Any,
) -> Any:
    """Instantiate an athletic environment via :func:`~myosuite.core.registry.make_env`."""
    ensure_tendons_applied(bundle)
    if bundle.task.kind in ("long_jump", "high_jump"):
        raise NotImplementedError(
            f"{bundle.task.key}: jumps.JumpJudge (rules, rewards, endings) runs on the fullbody C-plugin env "
            "(CFullbodySimpleTD3Env(task='long_jump' | 'high_jump_hXXX')); the ModularTaskEnv and mjlab "
            "terms do not apply it yet."
        )
    if backend not in ("cpu", "mjlab"):
        raise ValueError(f"Unknown athletic backend: {backend!r}")
    # Lazy import: catalog imports this module at load time.
    from myosuite.core.registry import make_env
    from myosuite.envs.myo.athletic.catalog import register_athletic_matrix

    register_athletic_matrix(
        tasks=(bundle.task.key,),
        plants=(bundle.plant.key,),
        tendons=(bundle.tendons,),
        backends={backend},
    )
    if backend == "mjlab" and "num_envs" not in overrides:
        overrides = {**overrides, "num_envs": num_envs}
    return make_env(bundle.env_id(), backend=backend, **overrides)
