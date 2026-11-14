# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The myoSuperMimic tracking policy driving one ChaseTag FBVs player on CPU MuJoCo.

The tracker's decoder observation is computed from the player's CPU state with the task's own analytic pieces
(``SkeletonKinematics``, ``rel_sites``), so it matches the mjlab task term for term; the goal comes from a
:class:`~chasetag_teacher.reference.SteeredReference`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from supermimic.heads import rel_sites
from supermimic.kinematics import SkeletonKinematics
from supermimic.rotations import mat_to_quat, quat_to_mat

from myosuite.envs.myo.tasks.mimic.tracking.models import (
    ModelInfo,
    compiled_body,
    model_cfg,
)

BODY = model_cfg("myofullbody")
TRACKED_BODIES, FOOT_BODIES = BODY.tracked_bodies, BODY.foot_bodies

TRACKER_DT = 0.02
"""Control step of the tracker checkpoints (50 Hz)."""


@functools.lru_cache(maxsize=4)
def load_policy(path: str) -> tuple[torch.nn.Module, dict]:
    """The deployed (EMA) policy of a myoSuperMimic checkpoint, on CPU, and its env settings (cached: stateless)."""
    from supermimic.networks import ActionMap, Policy  # noqa: PLC0415

    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg, env = ck["cfg"], ck["env"]
    nm = ck["action_map"]["num_muscles"]
    pi = Policy(env["obs_dim"], env["goal_dim"], nm, tuple(cfg["enc_hidden"]), tuple(cfg["dec_hidden"]),
                cfg["latent_dim"], cfg["init_excitation"], action_map=ActionMap.from_spec(ck["action_map"], nm))  # fmt: skip
    pi.load_state_dict(ck["pi"])
    pi.eval().requires_grad_(False)
    return pi, env


@dataclass
class PlayerIndex:
    """Addresses of one player's state in the two-player model."""

    free_q: int
    free_v: int
    q_adr: np.ndarray
    v_adr: np.ndarray
    act_adr: np.ndarray
    ctrl_ids: np.ndarray
    foot_bodies: np.ndarray
    floor_geoms: np.ndarray

    @staticmethod
    def build(model: mujoco.MjModel, prefix: str, info: ModelInfo) -> PlayerIndex:
        """The addresses of the player with name prefix ``prefix`` in ``model``."""
        jid = lambda n: model.joint(prefix + n).id  # noqa: E731
        root = jid(info.root_joint_name)
        ctrl = np.array([model.actuator(prefix + n).id for n in info.muscle_names])
        # ground planes (world planes in the FBVs scene, the ``terrain`` body's plane in myoMimic-v0)
        floors = [
            g
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
        ]
        return PlayerIndex(
            free_q=int(model.jnt_qposadr[root]),
            free_v=int(model.jnt_dofadr[root]),
            q_adr=np.array([model.jnt_qposadr[jid(n)] for n in info.indep_names]),
            v_adr=np.array([model.jnt_dofadr[jid(n)] for n in info.indep_names]),
            act_adr=model.actuator_actadr[ctrl].astype(int),
            ctrl_ids=ctrl,
            foot_bodies=np.array([model.body(prefix + b).id for b in FOOT_BODIES]),
            floor_geoms=np.array(floors),
        )


class CpuTracker:
    """Observation, goal and action of the tracking policy for one player of a CPU MuJoCo scene.

    Args:
        checkpoint: A myoSuperMimic checkpoint (``model_*.pt``); ``None`` builds the observation only.
        model: The scene (the FBVs two-player model, or the single-body ``myoMimic-v0`` model).
        prefix: Name prefix of the player's joints and actuators (``"a0_"``, ``"a1_"``; ``""`` for one body).
    """

    def __init__(self, checkpoint: str | None, model: mujoco.MjModel, prefix: str):
        self.pi, self.settings = load_policy(checkpoint) if checkpoint else (None, {})
        body = compiled_body("myofullbody")
        self.info = ModelInfo(body)
        self.kin = SkeletonKinematics(TRACKED_BODIES, "cpu", body)
        self.idx = PlayerIndex.build(model, prefix, self.info)
        self.model = model
        self.prev_action = np.zeros(self.info.num_muscles, np.float32)
        self._force = np.zeros(6)

    def reset(self) -> None:
        """Clear the previous action (the env's reset state)."""
        self.prev_action[:] = 0.0

    def touch(self, data: mujoco.MjData) -> np.ndarray:
        """Normal force of the floor on each foot body (|world z| of the summed contact forces)."""
        out = np.zeros(len(self.idx.foot_bodies))
        gb = self.model.geom_bodyid
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.idx.floor_geoms:
                other = gb[g2]
            elif g2 in self.idx.floor_geoms:
                other = gb[g1]
            else:
                continue
            k = np.nonzero(self.idx.foot_bodies == other)[0]
            if len(k) == 0:
                continue
            mujoco.mj_contactForce(self.model, data, i, self._force)
            out[k[0]] += abs((c.frame.reshape(3, 3).T @ self._force[:3])[2])
        return out

    def state(self, data: mujoco.MjData) -> tuple[torch.Tensor, ...]:
        """(p, R, v, w, q, qd, act, touch) as torch tensors with a batch of 1, as ``MotionCommand.read_state``."""
        ix = self.idx
        rq, rv = (
            data.qpos[ix.free_q : ix.free_q + 7],
            data.qvel[ix.free_v : ix.free_v + 6],
        )
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)[None]  # noqa: E731
        Rb = quat_to_mat(t(rq[3:7]))
        w = (Rb @ t(rv[3:6]).unsqueeze(-1)).squeeze(-1)
        return (t(rq[:3]), Rb @ self.kin.R_off, t(rv[:3]), w, t(data.qpos[ix.q_adr]), t(data.qvel[ix.v_adr]),
                t(data.act[ix.act_adr]), t(self.touch(data)))  # fmt: skip

    def observation(self, S: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """The decoder observation (no muscle terms): root height + quat, joints, root qvel, joint velocities,
        excitation, activation, touch and the relative site quantities."""
        p, R, v, w, q, qd, act, touch = S
        Rp = R @ self.kin.R_off.T
        qf, qdf = self.kin.full_joint_state(q, qd)
        w_body = (Rp.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
        rpos, rang, rvel = rel_sites(*self.kin.forward(p, R, v, w, q, qd))
        a = torch.as_tensor(self.prev_action)[None]
        return torch.cat((p[..., 2:3], mat_to_quat(Rp), qf, v, w_body, qdf, a, act, touch,
                          rpos.flatten(-2), rang.flatten(-2), rvel.flatten(-2)), -1)  # fmt: skip

    @torch.no_grad()
    def act(self, data: mujoco.MjData, goal: torch.Tensor) -> np.ndarray:
        """Excitations in [0, 1] for the current state and goal (1, goal_dim)."""
        obs = self.observation(self.state(data))
        a = self.pi(obs, goal)[0].numpy().astype(np.float32)
        self.prev_action = a
        return a


def check_same_body(
    fbvs: mujoco.MjModel,
    body: mujoco.MjModel,
    prefixes: tuple[str, ...] = ("a0_", "a1_"),
) -> list[str]:
    """Check that every FBVs player has the tracker body's joints and muscles, in its order and with its parameters.

    Args:
        fbvs: The two-player FBVs model.
        body: The tracker's body (or the single-body ``myoMimic-v0`` scene).
        prefixes: The players' name prefixes.

    Returns:
        The body's joint names.

    Raises:
        AssertionError: On a difference in names, order, muscle parameters, joint ranges or body masses.
    """

    def names(m: mujoco.MjModel, kind: str, n: int, pre: str) -> list[str]:
        return [
            getattr(m, kind)(i).name[len(pre) :]
            for i in range(n)
            if getattr(m, kind)(i).name.startswith(pre)
        ]

    tj = [body.joint(i).name for i in range(body.njnt)]
    ta = [body.actuator(i).name for i in range(body.nu)]
    for pre in prefixes:
        assert names(fbvs, "joint", fbvs.njnt, pre) == tj, f"{pre} joints differ"
        assert names(fbvs, "actuator", fbvs.nu, pre) == ta, f"{pre} actuators differ"
        aid = np.array([fbvs.actuator(pre + n).id for n in ta])
        jid = np.array([fbvs.joint(pre + n).id for n in tj])
        for f in (
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_dynprm",
            "actuator_lengthrange",
        ):
            d = np.abs(getattr(fbvs, f)[aid] - getattr(body, f)).max()
            assert d < 1e-9, (pre, f, d)
        d = np.abs(fbvs.jnt_range[jid] - body.jnt_range).max()
        assert d < 1e-9, (pre, "jnt_range", d)
        fbb = {fbvs.body(i).name for i in range(fbvs.nbody)}
        tb = [
            i for i in range(1, body.nbody) if pre + body.body(i).name in fbb
        ]  # a scene's terrain is no body there
        bid = [fbvs.body(pre + body.body(i).name).id for i in tb]
        d = np.abs(fbvs.body_mass[bid] - body.body_mass[tb]).max()
        assert d < 1e-9, (pre, "body_mass", d)
    return tj
