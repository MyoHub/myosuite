# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Experimental PyTorch DLPack accessor and muscle-stage pipeline for Newton.

.. warning::
   **Experimental validation probe only.** CPU ``MyoGymnasiumEnv`` and GPU
   ``mjlab`` are the only two supported MyoSuite backends. Like the experimental
   MJX route, this module exists solely for solver compatibility diagnostics and
   should not be used as a foundation for new MyoSuite tasks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import torch

from myosuite.core.muscle_conditions import TorchFatigueState
from myosuite.core.protocols import EnvAccessor, PhysicsPath
from myosuite.envs.muscle_stages import (
    LATE_ORDER,
    STAGE_ORDER,
    ExcitationStage,
    check_custom_order,
    warn_order_clash,
)
from myosuite.integrations.newton.accessor import (
    _entry_point_defaults,
    _NewtonSimContext,
    _require_newton,
)
from myosuite.terms.base_action import (
    MotorNoiseCfg,
    sample_motor_noise,
    sigmoid_muscle_activation,
)


def _warp_to_torch_device(wp_device: Any) -> tuple[torch.device, bool]:
    dev_str = str(wp_device)
    if dev_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(dev_str), True
        return torch.device("cpu"), False
    return torch.device("cpu"), True


class TorchNewtonAccessor(EnvAccessor):
    """Zero-copy PyTorch ``EnvAccessor`` presenting Newton state in authored MuJoCo order."""

    def __init__(self, sim: _NewtonSimContext) -> None:
        _, wp = _require_newton()
        self._wp = wp
        self._sim = sim
        self.device, self._use_dlpack = _warp_to_torch_device(
            sim.solver.mjw_data.qpos.device
        )
        m = sim.mapping
        self._qpos_idx = torch.as_tensor(
            m.src_to_conv_qpos, dtype=torch.long, device=self.device
        )
        self._qvel_idx = torch.as_tensor(
            m.src_to_conv_qvel, dtype=torch.long, device=self.device
        )
        self._act_idx = torch.as_tensor(
            m.src_to_conv_act, dtype=torch.long, device=self.device
        )
        self._ctrl_idx = torch.as_tensor(
            m.src_to_conv_ctrl, dtype=torch.long, device=self.device
        )
        self._site_idx = torch.as_tensor(
            m.src_to_conv_site, dtype=torch.long, device=self.device
        )
        self._body_idx = torch.as_tensor(
            m.src_to_conv_body, dtype=torch.long, device=self.device
        )
        self._ctrl_range = torch.as_tensor(
            sim.reference_model.actuator_ctrlrange,
            dtype=torch.float32,
            device=self.device,
        )

    def _as_tensor(self, wp_arr: Any) -> torch.Tensor:
        if self._use_dlpack:
            return self._wp.to_torch(wp_arr)
        return torch.from_numpy(wp_arr.numpy())

    @property
    def physics_path(self) -> PhysicsPath:
        return PhysicsPath.MJLAB

    def joint_pos(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.qpos)[:, self._qpos_idx]

    def joint_vel(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.qvel)[:, self._qvel_idx]

    def muscle_act(self) -> torch.Tensor:
        if self._sim.reference_model.na == 0:
            return torch.zeros(
                (self._sim.num_envs, 0), dtype=torch.float32, device=self.device
            )
        return self._as_tensor(self._sim.solver.mjw_data.act)[:, self._act_idx]

    def site_xpos(self, site_ids: Any) -> torch.Tensor:
        xpos = self._as_tensor(self._sim.solver.mjw_data.site_xpos)
        if isinstance(site_ids, int | np.integer):
            conv = self._site_idx[int(site_ids)]
            return xpos[:, conv]
        ids_t = torch.as_tensor(site_ids, dtype=torch.long, device=self.device)
        return xpos[:, self._site_idx[ids_t]]

    def site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(
            self._sim.reference_model, mujoco.mjtObj.mjOBJ_SITE, name
        )
        if sid == -1:
            raise ValueError(f"Site {name!r} not found in model")
        return int(sid)

    def time(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.time).clone()

    def ctrl_range(self) -> torch.Tensor:
        return self._ctrl_range

    def dt(self) -> float:
        return self._sim.ctrl_dt

    def array_module(self) -> Any:
        return torch

    def muscle_length(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.actuator_length)[
            :, self._ctrl_idx
        ]

    def muscle_velocity(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.actuator_velocity)[
            :, self._ctrl_idx
        ]

    def muscle_force(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.actuator_force)[
            :, self._ctrl_idx
        ]

    def body_xpos(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.xpos)[:, self._body_idx]

    def body_xquat(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.xquat)[:, self._body_idx]

    def body_xipos(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.xipos)[:, self._body_idx]

    def body_cvel(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.cvel)[:, self._body_idx]

    def subtree_com(self) -> torch.Tensor:
        return self._as_tensor(self._sim.solver.mjw_data.subtree_com)[:, self._body_idx]

    def contact_counts(self) -> torch.Tensor:
        data = self._sim.solver.mjw_data
        nacon = self._as_tensor(data.nacon)[0]
        worldid = self._as_tensor(data.contact.worldid)
        idx = torch.arange(worldid.shape[0], device=self.device)
        active = (idx < nacon) & (worldid >= 0) & (worldid < self._sim.num_envs)
        counts = torch.zeros(self._sim.num_envs, dtype=torch.long, device=self.device)
        valid_worlds = torch.where(active, worldid.to(dtype=torch.long), 0)
        counts.scatter_add_(0, valid_worlds, active.to(dtype=torch.long))
        return counts


@dataclass(kw_only=True)
class NewtonActionCfg:
    """Configuration reproducing CPU MyoSuite normalization and muscle stages on PyTorch.

    Args:
        normalize_act: Map actions from ``action_range`` to actuator controls.
        action_range: Normalized policy action bounds.
        muscle_sigmoid: Map muscle actions via MyoSuite sigmoid when ``normalize_act=True``.
        muscle_fatigue: Enable 3CC-r cumulative muscle fatigue.
        fatigue_reset_vec: Optional initial fatigued fraction ``MF`` per muscle on reset.
        fatigue_reset_random: Draw random fatigue states on reset.
        fatigue_muscles: Optional subset of muscle names or authored indices to fatigue.
        reroute: Optional ``(src_actuator, dst_actuator)`` pair for reafferentation.
        motor_noise: Signal-dependent and constant Gaussian motor noise config.
        excitation_stages: Factories producing portable :class:`ExcitationStage` instances.
    """

    normalize_act: bool = True
    action_range: tuple[float, float] = (-1.0, 1.0)
    muscle_sigmoid: bool = True
    muscle_fatigue: bool = False
    fatigue_reset_vec: tuple[float, ...] | None = None
    fatigue_reset_random: bool = False
    fatigue_muscles: tuple[str | int, ...] | None = None
    reroute: tuple[str, str] | None = None
    sarcopenia_force_scale: float | None = None
    motor_noise: MotorNoiseCfg = field(default_factory=MotorNoiseCfg)
    excitation_stages: tuple[Callable[[], ExcitationStage], ...] = ()

    @classmethod
    def from_env_id(cls, env_id: str) -> NewtonActionCfg:
        """Build a :class:`NewtonActionCfg` from a registered CPU MyoSuite ``env_id``."""
        import myosuite  # noqa: F401

        spec = gym.spec(env_id)
        kwargs = dict(spec.kwargs)
        defaults = _entry_point_defaults(spec.entry_point)
        normalize_act = bool(
            kwargs.get("normalize_act", defaults.get("normalize_act", True))
        )
        entry_str = str(spec.entry_point)
        is_walk = "LegWalkEnv" in entry_str or "walk" in entry_str.lower()
        action_range = (0.0, 1.0) if (is_walk and normalize_act) else (-1.0, 1.0)
        muscle_sigmoid = not is_walk

        wrappers = tuple(spec.additional_wrappers or ())
        by_name: dict[str, list[dict[str, Any]]] = {}
        for w in wrappers:
            by_name.setdefault(w.name, []).append(dict(w.kwargs or {}))

        uses_recipe = kwargs.get("model_recipe") is not None
        reroute = None
        if "ReafferentationWrapper" in by_name:
            sfx = "_r" if uses_recipe else ""
            reroute = (f"EIP{sfx}", f"EPL{sfx}")

        fatigue = "FatigueWrapper" in by_name
        f_kwargs = by_name["FatigueWrapper"][0] if fatigue else {}
        reset_vec = f_kwargs.get("fatigue_reset_vec")
        muscles = f_kwargs.get("muscles")
        noise_kwargs = (
            by_name["MotorNoiseWrapper"][0] if "MotorNoiseWrapper" in by_name else {}
        )
        sarcopenia_scale = (
            float(by_name["SarcopeniaWrapper"][0].get("force_scale", 0.5))
            if "SarcopeniaWrapper" in by_name
            else None
        )
        stage_factories = tuple(
            s["make_stage"] for s in by_name.get("ExcitationStageWrapper", ())
        )

        return cls(
            normalize_act=normalize_act,
            action_range=action_range,
            muscle_sigmoid=muscle_sigmoid,
            muscle_fatigue=fatigue,
            fatigue_reset_vec=(
                None if reset_vec is None else tuple(float(v) for v in reset_vec)
            ),
            fatigue_reset_random=bool(f_kwargs.get("fatigue_reset_random", False)),
            fatigue_muscles=(
                None
                if not fatigue or muscles is None
                else tuple(m if isinstance(m, str) else int(m) for m in muscles)
            ),
            reroute=reroute,
            sarcopenia_force_scale=sarcopenia_scale,
            motor_noise=MotorNoiseCfg.from_value(noise_kwargs.get("motor_noise")),
            excitation_stages=stage_factories,
        )


class _PipelineStage:
    name: str = ""
    order: float = 0.0

    def active(self) -> bool:
        return True

    def __call__(self, ctrl: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reset(
        self, env_ids: torch.Tensor | slice, generator: torch.Generator | None = None
    ) -> None:
        pass


class _NoisePipelineStage(_PipelineStage):
    name, order = "noise", STAGE_ORDER["noise"]

    def __init__(self, pipeline: NewtonActionPipeline) -> None:
        self._p = pipeline

    def active(self) -> bool:
        return self._p.cfg.motor_noise.enabled

    def __call__(self, ctrl: torch.Tensor) -> torch.Tensor:
        if not self.active() or self._p.muscle_cols.numel() == 0:
            return ctrl
        cols = self._p.muscle_cols
        ctrl[:, cols] = sample_motor_noise(
            ctrl[:, cols], self._p.cfg.motor_noise, self._p._randn, torch
        )
        return ctrl


class _FatiguePipelineStage(_PipelineStage):
    name, order = "fatigue", STAGE_ORDER["fatigue"]

    def __init__(
        self,
        pipeline: NewtonActionPipeline,
        state: TorchFatigueState,
        cols: torch.Tensor,
    ) -> None:
        self._p, self.state, self._cols = pipeline, state, cols

    def __call__(self, ctrl: torch.Tensor) -> torch.Tensor:
        ctrl[:, self._cols] = self.state.step(ctrl[:, self._cols], self._p.ctrl_dt)
        return ctrl

    def reset(
        self, env_ids: torch.Tensor | slice, generator: torch.Generator | None = None
    ) -> None:
        idx = None if isinstance(env_ids, slice) else env_ids
        self.state.reset(
            idx,
            fatigue_reset_vec=self._p.cfg.fatigue_reset_vec,
            fatigue_reset_random=self._p.cfg.fatigue_reset_random,
            generator=generator,
        )


class _ReroutePipelineStage(_PipelineStage):
    name, order = "reroute", STAGE_ORDER["reroute"]

    def __init__(self, src: int, dst: int) -> None:
        self._src, self._dst = src, dst

    def __call__(self, ctrl: torch.Tensor) -> torch.Tensor:
        ctrl[:, self._dst] = ctrl[:, self._src].clone()
        ctrl[:, self._src] = 0.0
        return ctrl


class _CustomPipelineStage(_PipelineStage):
    def __init__(self, pipeline: NewtonActionPipeline, stage: ExcitationStage) -> None:
        self._p, self.stage = pipeline, stage
        self.name = stage.name
        self.explicit_order = stage.order
        self.order = LATE_ORDER if stage.order is None else stage.order

    def __call__(self, ctrl: torch.Tensor) -> torch.Tensor:
        if self._p.muscle_cols.numel() == 0:
            return ctrl
        cols = self._p.muscle_cols
        ctrl[:, cols] = self.stage(ctrl[:, cols], torch)
        return ctrl

    def reset(
        self, env_ids: torch.Tensor | slice, generator: torch.Generator | None = None
    ) -> None:
        self.stage.reset(None if isinstance(env_ids, slice) else env_ids)


class NewtonActionPipeline:
    """Batched PyTorch muscle-command pipeline mirroring CPU and mjlab ``MyoAction``."""

    def __init__(
        self,
        reference_model: mujoco.MjModel,
        cfg: NewtonActionCfg,
        *,
        num_envs: int,
        ctrl_dt: float,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.nu = int(reference_model.nu)
        self.ctrl_dt = float(ctrl_dt)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

        muscle_ids = [
            i
            for i in range(self.nu)
            if int(reference_model.actuator_dyntype[i])
            == int(mujoco.mjtDyn.mjDYN_MUSCLE)
        ]
        self.muscle_cols = torch.tensor(
            muscle_ids, dtype=torch.long, device=self.device
        )
        self._has_activation = int(reference_model.na) > 0
        ctrl_range = torch.as_tensor(
            reference_model.actuator_ctrlrange,
            dtype=torch.float32,
            device=self.device,
        )
        self._ctrl_lo = ctrl_range[:, 0]
        self._ctrl_hi = ctrl_range[:, 1]
        self._ctrl_limited = torch.as_tensor(
            reference_model.actuator_ctrllimited.astype(bool),
            dtype=torch.bool,
            device=self.device,
        )

        act_names = [reference_model.actuator(i).name for i in range(self.nu)]
        self._stages: list[_PipelineStage] = [_NoisePipelineStage(self)]
        if cfg.muscle_fatigue and muscle_ids:
            if cfg.fatigue_muscles is None:
                f_cols_list = muscle_ids
            else:
                f_cols_list = [
                    act_names.index(m) if isinstance(m, str) else int(m)
                    for m in cfg.fatigue_muscles
                ]
            f_cols = torch.tensor(f_cols_list, dtype=torch.long, device=self.device)
            f_state = TorchFatigueState.from_mj_model(
                reference_model,
                num_envs=self.num_envs,
                device=str(self.device),
                muscles=f_cols_list,
            )
            self._stages.append(_FatiguePipelineStage(self, f_state, f_cols))

        if cfg.reroute is not None:
            src_idx = act_names.index(cfg.reroute[0])
            dst_idx = act_names.index(cfg.reroute[1])
            self._stages.append(_ReroutePipelineStage(src_idx, dst_idx))

        for make_stage in cfg.excitation_stages:
            custom = _CustomPipelineStage(self, make_stage())
            if custom.name in {s.name for s in self._stages} | set(STAGE_ORDER):
                raise ValueError(
                    f"The stage name {custom.name!r} is taken (built-in or twice)."
                )
            check_custom_order(custom.name, custom.explicit_order)
            if custom.explicit_order is not None:
                active_orders = {s.name: s.order for s in self._stages if s.active()}
                warn_order_clash(custom.name, custom.order, active_orders, 3)
            self._stages.append(custom)

        self._stages = sorted(self._stages, key=lambda s: s.order)

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self._stages)

    @property
    def fatigue_state(self) -> TorchFatigueState | None:
        return next(
            (s.state for s in self._stages if isinstance(s, _FatiguePipelineStage)),
            None,
        )

    def _randn(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(
            shape, generator=self.generator, device=self.device, dtype=torch.float32
        )

    def process(self, actions: torch.Tensor) -> torch.Tensor:
        """Transform policy ``actions`` into authored-order actuator controls."""
        act_t = actions.to(device=self.device, dtype=torch.float32)
        if act_t.shape == (self.nu,) and self.num_envs == 1:
            act_t = act_t.unsqueeze(0)
        if act_t.shape != (self.num_envs, self.nu):
            raise ValueError(
                f"Expected actions of shape ({self.num_envs}, {self.nu}), got {tuple(act_t.shape)}."
            )

        a_lo, a_hi = self.cfg.action_range
        if self.cfg.normalize_act:
            ctrl = torch.clamp(act_t, a_lo, a_hi)
            if self._has_activation:
                if self.cfg.muscle_sigmoid and self.muscle_cols.numel() > 0:
                    ctrl = ctrl.clone()
                    ctrl[:, self.muscle_cols] = sigmoid_muscle_activation(
                        ctrl[:, self.muscle_cols], torch
                    )
            else:
                ctrl = self._ctrl_lo + (ctrl - a_lo) / (a_hi - a_lo) * (
                    self._ctrl_hi - self._ctrl_lo
                )
        else:
            ctrl = act_t.clone()
            if torch.any(self._ctrl_limited):
                ctrl[:, self._ctrl_limited] = torch.clamp(
                    ctrl[:, self._ctrl_limited],
                    self._ctrl_lo[self._ctrl_limited],
                    self._ctrl_hi[self._ctrl_limited],
                )

        for stage in self._stages:
            ctrl = stage(ctrl)

        if torch.any(self._ctrl_limited):
            ctrl[:, self._ctrl_limited] = torch.clamp(
                ctrl[:, self._ctrl_limited],
                self._ctrl_lo[self._ctrl_limited],
                self._ctrl_hi[self._ctrl_limited],
            )
        return ctrl

    def write_to_control(
        self,
        actions: torch.Tensor,
        control: Any,
        conv_to_src_ctrl: torch.Tensor,
    ) -> torch.Tensor:
        """Process ``actions`` and write into ``control.mujoco.ctrl`` (zero-copy when DLPack-compatible)."""
        _, wp = _require_newton()
        processed = self.process(actions)
        reordered = processed[:, conv_to_src_ctrl]
        ctrl_wp = control.mujoco.ctrl
        if str(ctrl_wp.device).startswith("cuda") and not torch.cuda.is_available():
            ctrl_wp.assign(reordered.detach().cpu().numpy().reshape(-1))
        else:
            dst = wp.to_torch(ctrl_wp).view(self.num_envs, self.nu)
            dst.copy_(reordered)
        return processed

    def seed(self, seed: int) -> None:
        self.generator.manual_seed(int(seed))

    def reset(
        self, env_ids: torch.Tensor | Sequence[int] | np.ndarray | slice | None = None
    ) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            target: torch.Tensor | slice = slice(None)
        elif isinstance(env_ids, torch.Tensor):
            target = env_ids.to(device=self.device, dtype=torch.long)
        else:
            target = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for stage in self._stages:
            stage.reset(target, generator=self.generator)
