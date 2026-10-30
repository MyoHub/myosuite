# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Analytic observation, reward and termination heads of the Mimic tasks.

Each head is a differentiable function of the world-model state and the clip
frame, and reproduces the mjlab ``myoMimic*`` term functions (same layout and the
same shared reward / termination terms), so imagined rollouts are scored exactly
like the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from myosuite.learning.supermimic.kinematics import TorchKinematics
from myosuite.learning.supermimic.state import rotvec_from_quat
from myosuite.physics.quat_math_torch import mul_quat, neg_quat
from myosuite.terms.mimic_obs import mimic_termination_mask
from myosuite.terms.mimic_reward import (
    DEFAULT_SCALES,
    DEFAULT_WEIGHTS,
    mimic_composite_reward,
)


@dataclass
class MimicClip:
    """Device tensors of one motion clip at the control rate."""

    sites: torch.Tensor  # (T, n_sites, 3) tracked-site targets
    qpos: torch.Tensor  # (T, nq)
    qvel: torch.Tensor  # (T, nv)
    qpos_cols: torch.Tensor  # model qpos columns the clip covers
    qvel_cols: torch.Tensor
    root_pos: bool  # clip covers root position / quaternion / linear velocity
    root_quat: bool
    root_vel: bool

    @property
    def n_frames(self) -> int:
        """Number of clip frames."""
        return int(self.sites.shape[0])


class MimicHeads(torch.nn.Module):
    """Observations, reward and termination of a free-root Mimic task.

    Args:
        kinematics: FK of the tracked sites.
        clip: Clip tensors.
        ctrl_dt: Control timestep (the observation scales ``qvel`` by it).
        lookahead_k: Number of lookahead frames.
        lookahead_stride: Frame stride of the lookahead.
        site_err_threshold: Termination threshold of the mean site error (m).
        root_err_threshold: Termination threshold of the root error (m).
        soft_termination_width: Width (m) of the sigmoid around each threshold
            that :meth:`termination_prob` uses.
        trunk_body: Body whose orientation :meth:`posture_errors` compares (must be
            computed by ``kinematics``); ``-1`` disables the trunk terms.
        trunk_err_threshold: Trunk orientation error (rad) at which
            :meth:`termination_prob` terminates; ``0`` disables it (the env has no such rule).
    """

    def __init__(
        self,
        kinematics: TorchKinematics,
        clip: MimicClip,
        ctrl_dt: float,
        lookahead_k: int = 5,
        lookahead_stride: int = 20,
        site_err_threshold: float = 1.0,
        root_err_threshold: float = 0.3,
        soft_termination_width: float = 0.02,
        trunk_body: int = -1,
        trunk_err_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kin = kinematics
        self.clip = clip
        self.ctrl_dt = ctrl_dt
        self.k = lookahead_k
        self.stride = lookahead_stride
        self.site_thr = site_err_threshold
        self.root_thr = root_err_threshold
        self.soft_w = soft_termination_width
        self.trunk = trunk_body
        self.trunk_thr = trunk_err_threshold
        self._last = max(clip.n_frames - 1, 1)

    def frame_index(self, frame: torch.Tensor) -> torch.Tensor:
        """Clip frame held at the last frame past the end."""
        return frame.clamp(max=self.clip.n_frames - 1)

    def clip_end(self, frame: torch.Tensor) -> torch.Tensor:
        """The episode has played past the clip's last frame."""
        return frame >= self.clip.n_frames

    def observation(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        act: torch.Tensor,
        frame: torch.Tensor,
        sites: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Policy observation (the env's ``actor`` group layout)."""
        c = self.clip
        f = self.frame_index(frame)
        sites = self.kin.site_pos(qpos) if sites is None else sites
        tgt = c.sites[f]
        batch = qpos.shape[:-1]
        flat = lambda x: x.reshape(batch + (-1,))  # noqa: E731
        root = qpos[..., None, :3]
        fut = [(f + i * self.stride) % c.n_frames for i in range(1, self.k + 1)]
        look = []
        for fi in fut:
            look.append(flat(c.sites[fi] - root))
            if c.root_pos:
                look.append(c.qpos[fi][..., :3] - qpos[..., :3])
            if c.root_vel:
                look.append(c.qvel[fi][..., :3])
            look.append((fi.float() / self._last)[..., None])
        return torch.cat(
            [
                qpos,
                qvel * self.ctrl_dt,
                act,
                flat(sites),
                flat(tgt),
                flat(tgt - sites),
                c.qpos[f],
                c.qvel[f],
                (f.float() / c.n_frames)[..., None],
                *look,
            ],
            dim=-1,
        )

    def reward_terms(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        frame: torch.Tensor,
        sites: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """``mimic_composite_reward`` terms; the root orientation term uses a smooth angle."""
        c = self.clip
        f = self.frame_index(frame)
        sites = self.kin.site_pos(qpos) if sites is None else sites
        ref_q, ref_v = c.qpos[f], c.qvel[f]
        # The site and root-position terms take a plain sqrt of the distance, whose gradient
        # is NaN at zero error: pass the reference for them here and add smoothed versions below.
        out = mimic_composite_reward(
            torch,
            c.sites[f],
            c.sites[f],
            qpos.index_select(-1, c.qpos_cols),
            ref_q.index_select(-1, c.qpos_cols),
            qvel.index_select(-1, c.qvel_cols),
            ref_v.index_select(-1, c.qvel_cols),
            None,
            None,
            qvel[..., :3] if c.root_vel else None,
            ref_v[..., :3] if c.root_vel else None,
            None,
            None,
        )
        # Same values as mimic_site_tracking_reward / mimic_root_pos_reward up to a 0.1 mm
        # smoothing of the distances.
        d = sites - c.sites[f]
        site = torch.exp(
            -DEFAULT_SCALES.site * torch.sqrt((d * d).sum(-1) + 1e-8).mean(-1)
        )
        out["dense"] = out["dense"] + DEFAULT_WEIGHTS.site * (site - out["site"])
        out["site"] = site
        for key in (
            "mean_site_dist",
            "solved",
        ):  # computed on the reference above: meaningless here
            out.pop(key, None)
        if c.root_pos:
            r = qpos[..., :3] - ref_q[..., :3]
            out["root_pos"] = torch.exp(
                -DEFAULT_SCALES.root_pos * torch.sqrt((r * r).sum(-1) + 1e-8)
            )
            out["dense"] = out["dense"] + DEFAULT_WEIGHTS.root_pos * out["root_pos"]
        if c.root_quat:
            # Same value as mimic_root_orient_reward (2 acos|dot|), but finite gradient at 0.
            angle = rotvec_from_quat(
                mul_quat(neg_quat(ref_q[..., 3:7]), qpos[..., 3:7])
            ).norm(dim=-1)
            out["root_orient"] = torch.exp(-DEFAULT_SCALES.root_orient * angle)
            out["dense"] = (
                out["dense"] + DEFAULT_WEIGHTS.root_orient * out["root_orient"]
            )
        return out

    def reward(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        frame: torch.Tensor,
        sites: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense tracking reward in ``[0, 1]`` (before the env's reward weight)."""
        return self.reward_terms(qpos, qvel, frame, sites)["dense"]

    def termination(
        self, qpos: torch.Tensor, frame: torch.Tensor, sites: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Hard early termination of the env (``mimic_deviation``)."""
        c = self.clip
        f = self.frame_index(frame)
        sites = self.kin.site_pos(qpos) if sites is None else sites
        dead = mimic_termination_mask(
            torch,
            sites,
            c.sites[f],
            qpos[..., :3] if c.root_pos else None,
            c.qpos[f][..., :3] if c.root_pos else None,
            self.site_thr,
            self.root_thr,
        )
        return dead & ~self.clip_end(frame)

    def posture_errors(
        self, qpos: torch.Tensor, frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pelvis (root) and trunk orientation errors against the clip, in rad."""
        f = self.frame_index(frame)
        ref = self.clip.qpos[f]
        pelvis = rotvec_from_quat(
            mul_quat(neg_quat(ref[..., 3:7]), qpos[..., 3:7])
        ).norm(dim=-1)
        if self.trunk < 0:
            return pelvis, torch.zeros_like(pelvis)
        _, qa = self.kin.body_poses(qpos)
        _, qr = self.kin.body_poses(ref)
        trunk = rotvec_from_quat(
            mul_quat(neg_quat(qr[..., self.trunk, :]), qa[..., self.trunk, :])
        ).norm(dim=-1)
        return pelvis, trunk

    def termination_prob(
        self, qpos: torch.Tensor, frame: torch.Tensor, sites: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Smooth version of :meth:`termination` for imagined survival weights."""
        c = self.clip
        f = self.frame_index(frame)
        sites = self.kin.site_pos(qpos) if sites is None else sites
        d = sites - c.sites[f]
        site_err = torch.sqrt((d * d).sum(-1) + 1e-12).mean(-1)
        alive = 1 - torch.sigmoid((site_err - self.site_thr) / self.soft_w)
        if c.root_pos:
            r = qpos[..., :3] - c.qpos[f][..., :3]
            root_err = torch.sqrt((r * r).sum(-1) + 1e-12)
            alive = alive * (
                1 - torch.sigmoid((root_err - self.root_thr) / self.soft_w)
            )
        if self.trunk_thr > 0 and self.trunk >= 0:
            _, trunk = self.posture_errors(qpos, frame)
            alive = alive * (
                1 - torch.sigmoid((trunk - self.trunk_thr) / (5 * self.soft_w))
            )
        return (1 - alive) * (~self.clip_end(frame)).float()

    @classmethod
    def from_env(
        cls, env: Any, entity_name: str, variant: str = "fullbody", **kwargs: Any
    ) -> MimicHeads:
        """Heads of a constructed mjlab Mimic env in trajectory mode."""
        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            _resolve_mimic_mjlab_ids,
        )
        from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import (
            _musclemimic_cpu_model,
        )

        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant)
        source = cache["clip_source"]
        if (
            source is None
            or not hasattr(source, "n_frames")
            or hasattr(source, "clips")
        ):
            raise ValueError("a single-clip trajectory-mode Mimic env is required")
        clip = cache["clip"]
        dev = env.device
        frames = torch.arange(source.n_frames, device=dev)
        source._ensure_device(torch.device(dev), env.num_envs)
        qcols = np.asarray(clip.qpos_model_indices)
        vcols = np.asarray(clip.qvel_model_indices)
        has = lambda cols, r: bool(np.isin(np.arange(*r), cols).all())  # noqa: E731
        mclip = MimicClip(
            sites=source.site_targets_at_frames(frames).float(),
            qpos=source.ref_qpos_at_frames(frames).float(),
            qvel=source.ref_qvel_at_frames(frames).float(),
            qpos_cols=torch.as_tensor(qcols, device=dev, dtype=torch.long),
            qvel_cols=torch.as_tensor(vcols, device=dev, dtype=torch.long),
            root_pos=has(qcols, (0, 3)),
            root_quat=has(qcols, (3, 7)),
            root_vel=has(vcols, (0, 3)),
        )
        model = _musclemimic_cpu_model(variant)
        trunk = int(model.body("torso").id) if variant == "fullbody" else -1
        extra = (trunk,) if trunk >= 0 else ()
        kin = TorchKinematics(
            model, cache["site_ids"], prune=True, extra_bodies=extra
        ).to(dev)
        ctrl_dt = env.physics_dt * env.cfg.decimation
        kwargs.setdefault("trunk_body", trunk)
        return cls(kin, mclip, ctrl_dt, **kwargs)
