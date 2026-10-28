"""CPU counterpart of the raw 354-muscle MJLab Halloween task.

The default benchmark and opt-in TERRA showcase share the muscle-action API.
"""

from typing import Any, ClassVar

import mujoco
import numpy as np
from gymnasium import spaces

from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.myo.tasks.halloween.config import HALLOWEEN, AvoidanceConfig
from myosuite.envs.myo.tasks.halloween.avoidance import AvoidanceTask
from myosuite.envs.myo.tasks.halloween.halloween_model import (
    PUMPKIN_BODY,
    PUMPKIN_COLLISION_GEOM,
    compile_halloween_model,
    park_pumpkin,
    throw_pumpkin,
)
from myosuite.terms.halloween import halloween_scores


class HalloweenEnv(MyoGymnasiumEnv):
    """Full-body muscle actions with seeded ballistic pumpkins and real hit scoring."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        *,
        frame_skip: int = 5,
        throw_speed: float = 1.0,
        curriculum: bool = False,
        render_mode: str | None = None,
        profile: str = "benchmark",
        avoidance_config: AvoidanceConfig | None = None,
    ) -> None:
        super().__init__(frame_skip=frame_skip, render_mode=render_mode)
        if frame_skip != 5:
            raise ValueError("Halloween uses 5 x 2 ms substeps on both backends")
        if not np.isfinite(throw_speed) or not 0.5 <= throw_speed <= 2.0:
            raise ValueError("throw_speed must be finite and between 0.5 and 2.0")
        if profile == "benchmark" and avoidance_config is not None:
            raise ValueError("avoidance_config requires profile='terra'")
        if profile == "terra" and (throw_speed != 1.0 or curriculum):
            raise ValueError(
                "The terra profile uses avoidance_config for its throw curriculum"
            )
        self.profile = profile
        self.model, self._mj_spec, self._xml_path = compile_halloween_model(
            profile=profile
        )
        self.data = mujoco.MjData(self.model)
        self.avoidance = (
            AvoidanceTask(self.model, self.data, avoidance_config or AvoidanceConfig())
            if profile == "terra"
            else None
        )
        self._ctrl_dt = HALLOWEEN.ctrl_dt
        self.throw_speed, self.curriculum = throw_speed, curriculum
        self._root = int(self.model.jnt_bodyid[0])
        self._head = self.model.body("head").id
        self._pumpkin = self.model.body(PUMPKIN_BODY).id
        self._gid = self.model.geom(PUMPKIN_COLLISION_GEOM).id
        self._robot_bodies = np.flatnonzero(
            (np.arange(self.model.nbody) != 0)
            & (np.arange(self.model.nbody) != self._pumpkin)
        )
        self._actor_nq = self.model.nq - 7
        self._actor_nv = self.model.nv - 6
        self.action_space = spaces.Box(0.0, 1.0, (self.model.nu,), np.float32)
        size = 11 + self._actor_nq - 7 + self._actor_nv - 6 + self.model.na + 6
        self.observation_space = self._unbounded_obs_space(size)
        self._hit = False
        self._contact_count = 0
        self._common_steps = 0

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Restore the actor and sample the first interval from the episode RNG."""
        if self.model.nkey:
            key = (
                self.model.key("halloween_ready").id
                if self.avoidance is not None
                else 0
            )
            self.data.qpos[:] = self.model.key_qpos[key]
            self.data.qvel[:] = self.model.key_qvel[key]
            self.data.act[:] = self.model.key_act[key]
        if self.avoidance is None:
            park_pumpkin(self.model, self.data)
        self._hit = False
        self._contact_count = 0
        if self.avoidance is not None:
            self.avoidance.reset(np_random)
        else:
            self._next_throw = float(np_random.uniform(*HALLOWEEN.throw_interval_s))
        return {}

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """Reset physics and expose the seeded showcase state to external controllers."""
        obs, info = super().reset(seed=seed, options=options)
        if self.avoidance is not None:
            info["halloween"] = self.avoidance.info()
        return obs, info

    @property
    def dt(self) -> float:
        """Seconds of physical time advanced by each step()."""
        return self._ctrl_dt

    def render(self, **kwargs: Any) -> np.ndarray | None:
        """Render the fixed showcase camera, or the benchmark's standard view."""
        if self.avoidance is not None and not kwargs:
            kwargs = dict(
                width=640,
                height=480,
                camera_id=-1,
                max_geom=10000,
                default_cam_config=dict(
                    lookat=[0, -0.6, 1.35], distance=4.8, azimuth=115, elevation=-10
                ),
            )
        return super().render(**kwargs)

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        d = self.data
        rotation = d.xmat[self._root].reshape(3, 3)
        root = np.r_[
            d.xpos[self._root, 2],
            d.xquat[self._root],
            d.qvel[:3] @ rotation,
            d.qvel[3:6],
        ]
        return {
            "root": root,
            "joint_pos": d.qpos[7 : self._actor_nq].copy(),
            "joint_vel": d.qvel[6 : self._actor_nv].copy(),
            "act": d.act.copy(),
            "pumpkin_rel": np.r_[
                d.xpos[self._pumpkin] - d.xpos[self._root],
                d.qvel[self._actor_nv : self._actor_nv + 3],
            ],
        }

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        d = self.data
        root, head = d.xpos[self._root], d.xpos[self._head]
        pumpkin = d.xpos[self._pumpkin]
        clearance = np.linalg.norm(d.xpos[self._robot_bodies] - pumpkin, axis=-1).min()
        scores = halloween_scores(np, root, head, clearance)
        if np.linalg.norm(pumpkin[:2] - root[:2]) >= 1.0:
            scores["dodge"] = 1.0
        fallen = root[2] < HALLOWEEN.fall_pelvis_z or head[2] < HALLOWEEN.fall_head_z
        reward = {
            **scores,
            "dense": self._ctrl_dt
            * (1.0 + 2.0 * scores["upright"] + scores["posture"] + scores["dodge"]),
            "done": fallen
            or self._hit
            or (self.avoidance is not None and self.avoidance.complete),
            "hit": self._hit,
            "body_contacts": self._contact_count,
            "fallen": bool(fallen),
        }
        if self.avoidance is not None:
            reward["solved"] = self.avoidance.complete and all(
                event["body_contacts"] == 0 and event["recovered"]
                for event in self.avoidance.events
            )
        return reward

    def step(self, action: np.ndarray, **kwargs: Any) -> tuple:
        """Apply muscle excitation and retain contacts across every physics substep."""
        if self.avoidance is None and self.data.time >= self._next_throw:
            easy = (
                1.0
                + (HALLOWEEN.easy_start - 1.0)
                * max(0.0, 1.0 - self._common_steps / HALLOWEEN.ramp_steps)
                if self.curriculum
                else 1.0
            )
            throw_pumpkin(
                self.model,
                self.data,
                side=float(self.np_random.choice([-1, 1])),
                speed=self.throw_speed,
                x_offset=float(self.np_random.uniform(-0.08, 0.08)),
                height=float(self.np_random.uniform(0.25, 0.55)),
                flight=float(self.np_random.uniform(0.55, 0.75)) * easy,
            )
            self._next_throw += float(
                self.np_random.uniform(*HALLOWEEN.throw_interval_s)
            )
        self.data.ctrl[:] = np.clip(action, 0.0, 1.0)
        self._hit = False
        self._contact_count = 0
        for _ in range(self.frame_skip):
            self._step_physics(nstep=1)
            if self.avoidance is not None:
                count = self.avoidance.count_contacts()
                self._contact_count += count
                self._hit |= count > 0
                continue
            for contact_index, contact in enumerate(self.data.contact):
                other = (
                    int(contact.geom2)
                    if contact.geom1 == self._gid
                    else (int(contact.geom1) if contact.geom2 == self._gid else -1)
                )
                if other >= 0 and self.model.geom_bodyid[other] in self._robot_bodies:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                    if np.linalg.norm(force[:3]) > 0:
                        self._contact_count += 1
                        self._hit = True
        if self.avoidance is not None:
            self.avoidance.after_step()
        self._common_steps += 1
        obs = self._get_obs_dict(CpuEnvAccessor(self.model, self.data, self._ctrl_dt))
        observation, reward, terminated, truncated, info = self._finalize_step(
            obs, self.get_reward_dict(obs)
        )
        if self.avoidance is not None:
            info["halloween"] = self.avoidance.info()
        return observation, reward, terminated, truncated, info
