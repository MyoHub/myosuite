"""Pretrained command selection and references for the Halloween TERRA profile."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any
from urllib.request import urlretrieve

import joblib
import mujoco
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from myosuite.core.trajectory_io import MotionClip, motion_clip_from_states
from myosuite.integrations.musclemimic import (
    MIMIC_SITES,
    TerrainController,
    build_terrain_fullbody_spec,
    load_terrain_policy,
)

if TYPE_CHECKING:
    from myosuite.envs.myo.tasks.halloween.cpu import HalloweenEnv

COMMANDS = ("idle", "left", "right", "squat")
ASSET_REVISION = "8842a86052d46be13b489a1e755bd265bcbc78ae"
ASSET_URL = f"https://raw.githubusercontent.com/MyoHub/myosuite/{ASSET_REVISION}/tutorials/files/halloween/"
HASHES = {
    "motion_decoder.npz": "b3f9a27ed5a9b3dd64f0efb8671eb5954fe1c0ce1ed681e3e5ba339d2468b78c",
    "selector.joblib": "c748be2fa397cd8aefb70bf7fa6eda20eb5e5ecefb49963245531cb41097db8e",
}


@dataclass(frozen=True)
class HalloweenAssets:
    """Paths to the two verified public learned-data files."""

    motion: Path
    selector: Path


def download_halloween_assets(cache_dir: str | Path | None = None) -> HalloweenAssets:
    """Download hash-pinned decoder/selector weights, reusing a verified cache.

    Args:
        cache_dir: Download directory; defaults to ~/.cache/myosuite/halloween.

    Returns:
        Paths to the public NPZ and trusted joblib files. Downloads are atomic.
    """
    root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.home() / ".cache/myosuite/halloween"
    )
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        destination = root / name
        if (
            destination.is_file()
            and hashlib.sha256(destination.read_bytes()).hexdigest() == expected
        ):
            continue
        with NamedTemporaryFile(dir=root, delete=False) as temporary:
            staging = Path(temporary.name)
        try:
            urlretrieve(ASSET_URL + name, staging)
            if hashlib.sha256(staging.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Halloween asset checksum mismatch: {name}")
            staging.replace(destination)
        finally:
            staging.unlink(missing_ok=True)
    return HalloweenAssets(root / "motion_decoder.npz", root / "selector.joblib")


class HalloweenController:
    """Learned command values + references, executed by the existing TERRA runner.

    Args:
        env: Unwrapped Halloween environment with profile='terra'.
        checkpoint: Folder returned by download_terrain_checkpoint().
        assets: Paths returned by download_halloween_assets().
        command_override: Optional fixed command for the tutorial's experiments.
        return_after_s: Scripted recovery delay after each lateral launch response.

    Call reset(info) after every env.reset(), then controller(data, info) once per
    env.step(). Only the environment advances the live actor's physical state.
    """

    def __init__(
        self,
        env: HalloweenEnv,
        checkpoint: str | Path,
        assets: HalloweenAssets,
        *,
        command_override: str | None = None,
        return_after_s: float = 3.7,
    ) -> None:
        if env.profile != "terra":
            raise ValueError("HalloweenController requires profile='terra'")
        if command_override is not None and command_override not in COMMANDS:
            raise ValueError(f"command_override must be one of {COMMANDS}")
        if not np.isfinite(return_after_s) or return_after_s <= 0:
            raise ValueError("return_after_s must be positive and finite")
        for name, path in (
            ("motion_decoder.npz", assets.motion),
            ("selector.joblib", assets.selector),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
                raise ValueError(f"Unverified Halloween asset: {path}")
        with np.load(assets.motion, allow_pickle=False) as weights:
            self.weights = {key: weights[key].copy() for key in weights.files}
        self.selector = joblib.load(assets.selector)
        self.actor = build_terrain_fullbody_spec().compile()
        self.actor_data = mujoco.MjData(self.actor)
        self.checkpoint, self.frame_skip = Path(checkpoint), env.frame_skip
        self.dt = env.frame_skip * env.model.opt.timestep
        self.return_step = round(return_after_s / self.dt)
        self.command_override = command_override
        self._cache: dict[str, MotionClip] = {}
        self._ready = False

    def reference_for(self, command: str, anchor: np.ndarray) -> MotionClip:
        """Decode a command into site/joint targets translated to the given XY anchor."""
        if command not in COMMANDS:
            raise ValueError(f"Unknown command {command!r}")
        if command not in self._cache:
            frames = int(self.weights[command + "_frames"])
            phase = np.linspace(0, 1, frames)[:, None]
            basis = rbf_kernel(
                phase,
                self.weights[command + "_centers"],
                gamma=float(self.weights["gamma"]),
            )
            poses = (
                basis @ self.weights[command + "_coef"]
                + self.weights[command + "_base"]
            )
            self._cache[command] = motion_clip_from_states(
                self.actor, poses, self.dt, MIMIC_SITES
            )
        clip = self._cache[command]
        shift = np.r_[anchor, 0.0]
        positions = clip.qpos.copy()
        positions[:, :3] += shift
        return replace(clip, qpos=positions, site_xpos=clip.site_xpos + shift)

    def reset(self, info: dict[str, Any]) -> None:
        """Load/reset frozen TERRA using the environment's reproducible policy seed."""
        context = info["halloween"]
        self.anchor = np.array(context["home"], copy=True)
        self.command, self.selected, self.launch_id = "idle", "idle", 0
        self.decisions: list[dict[str, Any]] = []
        self.reference = self.reference_for("idle", self.anchor)
        self.policy = load_terrain_policy(
            self.checkpoint,
            self.actor,
            self.reference,
            stochastic=True,
            seed=context["policy_seed"],
        )
        self._track(self.reference)
        self._ready = True

    def _track(self, reference: MotionClip) -> None:
        self.reference = reference
        self.policy.obs_adapter.reference = reference
        self.tracker = TerrainController(
            self.policy, self.actor, reference, self.frame_skip
        )
        self.tracker.reset()
        self.frame = 0

    def _choose(self, data: mujoco.MjData) -> str:
        relative = (
            data.qpos[self.actor.nq : self.actor.nq + 3] - data.qpos[:3]
        ).astype(np.float32)
        velocity = data.qvel[self.actor.nv : self.actor.nv + 3].astype(np.float32)
        intercept_time = -relative[1] / velocity[1]
        features = np.array(
            [
                relative[0] + velocity[0] * intercept_time,
                relative[2]
                + np.float32(data.qpos[2])
                + velocity[2] * intercept_time
                - 0.5 * 9.81 * intercept_time**2,
                intercept_time,
                1.0,
            ],
            dtype=np.float32,
        )
        values = self.selector.predict(np.c_[np.tile(features, (4, 1)), np.eye(4)])
        selected = self.command_override or COMMANDS[int(np.argmax(values))]
        self.decisions.append(
            dict(throw=self.launch_id, command=selected, values=values.tolist())
        )
        return selected

    def __call__(self, data: mujoco.MjData, info: dict[str, Any]) -> np.ndarray:
        """Return 354 muscle controls from the current observed state and launch context."""
        if not self._ready:
            raise RuntimeError("Call controller.reset(info) after env.reset()")
        context = info["halloween"]
        next_command = None
        if context["launch_id"] != self.launch_id:
            self.launch_id = context["launch_id"]
            self.selected = next_command = self._choose(data)
        elif context["steps_since_launch"] == self.return_step and self.selected in (
            "left",
            "right",
        ):
            next_command = "right" if self.selected == "left" else "left"
        if next_command is not None and next_command != "idle":
            if self.command != "idle":
                raise RuntimeError(
                    "The previous command has not finished; increase throw spacing"
                )
            self.command = next_command
            self._track(self.reference_for(next_command, self.anchor))
        self.actor_data.qpos[:] = data.qpos[: self.actor.nq]
        self.actor_data.qvel[:] = data.qvel[: self.actor.nv]
        self.actor_data.act[:] = data.act
        self.actor_data.time = data.time
        mujoco.mj_forward(self.actor, self.actor_data)
        controls = self.tracker(self.actor_data)
        self.frame += 1
        if self.frame >= self.reference.num_frames - 1:
            clip = self.reference
            self.anchor = clip.qpos[-1, :2].copy()
            self.command = "idle"
            self._track(
                replace(
                    clip,
                    qpos=np.repeat(clip.qpos[-1:], 1101, axis=0),
                    qvel=np.zeros((1101, self.actor.nv)),
                    site_xpos=np.repeat(clip.site_xpos[-1:], 1101, axis=0),
                    site_xmat=np.repeat(clip.site_xmat[-1:], 1101, axis=0),
                    site_velocity=np.zeros((1101, *clip.site_velocity.shape[1:])),
                )
            )
        return controls
