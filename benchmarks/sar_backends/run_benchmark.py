#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SAR Backend Benchmark: Gymnasium (CPU) vs MJX (XLA) vs MJX (Warp) vs mjlab.

Trains SB3 SAC with SynergyWrapper on the myoLegWalk task on four physics
backends and logs per-step and per-episode diagnostics so that numerical
differences between backends can be identified and debugged.

All backends share:
  - The same SAR transform (sklearn PCA/ICA loaded from pretrained pkl files)
  - Identical observation/reward definitions where possible

Algorithm per backend:
  - CPU/MJX XLA/MJX Warp: SB3 SAC (off-policy, single-env, CPU-side)
  - mjlab: RSL-RL PPO (on-policy, GPU-native, vectorised)
    The mjlab training loop keeps all tensors on GPU (no CPU round-trips):
    policy forward pass, SAR inverse transform (torch matmul), env step,
    rollout buffer, and PPO update are all torch operations on the GPU.
    The SAR inverse transform is implemented as a fixed ``SARTorchTransform``
    (see ``sar_torch_transform.py``) rather than sklearn on CPU.

What differs is only the physics backend:
  - CPU: MuJoCo C++ float64 via ``myoLegWalk-v0``
  - MJX XLA: JAX/XLA float32 via ``MjxLegWalk-v0`` (impl=None)
  - MJX Warp: MuJoCo Warp float32 via ``MjxLegWalk-v0`` (impl="warp")
  - mjlab: MuJoCo Warp + Isaac Lab manager env, fully on GPU via RSL-RL PPO

Usage::

    # Quick 10k-step diagnostic run (all backends, uses pretrained SAR pkl):
    python benchmarks/sar_backends/run_benchmark.py --steps 10000

    # Longer 50k-step training comparison:
    python benchmarks/sar_backends/run_benchmark.py --steps 50000 --output-dir ./benchmark_results

    # CPU only (skip MJX/mjlab if not installed):
    python benchmarks/sar_backends/run_benchmark.py --backends cpu

    # GPU run with Warp + mjlab backends:
    MYOSUITE_GPU_TESTS=1 python benchmarks/sar_backends/run_benchmark.py --steps 50000

Output files (in ``--output-dir``)::

    cpu/
        training_log.csv       per-episode metrics (timestep, ep_reward, vel_reward, …)
        steps_log.csv          per-step diagnostics sampled every LOG_STEP_INTERVAL steps
    mjx_xla/
        training_log.csv
        steps_log.csv
    mjx_warp/
        training_log.csv       (skipped if warp not available)
        steps_log.csv
    mjlab/
        training_log.csv       (skipped if mjlab not installed)
        steps_log.csv
    comparison_reward.png      reward convergence curves for all backends
    comparison_components.png  per-component reward divergence over time
    comparison_physics.png     qpos/height/com_vel statistics per backend
    divergence_report.txt      summary of max/mean divergence between backends

Running on CI / without a GPU::

    The script auto-detects GPU availability.  On CPU-only machines, MJX runs
    via JAX/XLA (slower but numerically equivalent to GPU XLA).  The "warp"
    backend is skipped if ``import warp`` fails.

Diagnosing convergence failures::

    If one backend fails to converge, examine:
    1. ``comparison_components.png`` — which reward component diverges first?
    2. ``steps_log.csv`` — at which step do obs or reward terms start to differ?
    3. ``divergence_report.txt`` — which obs dimensions show highest variance?

    Common sources of divergence:
    - Float32 vs float64: height/velocity threshold crossing near min_height=0.8
    - cvel sign convention differences in MJX vs CPU
    - actuator_force scaling (÷1000) numerical precision in float32
    - Contact model differences (MJX approximates contacts differently)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import pathlib
import sys
import time
from dataclasses import asdict
from typing import Any

import numpy as np

# Ensure repo root on sys.path
_REPO_ROOT = pathlib.Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pretrained SAR pkl files (from myosuite/agents/SAR_pretrained/locomotion/)
_SAR_DIR = _REPO_ROOT / "myosuite" / "agents" / "SAR_pretrained" / "locomotion"

# Step interval for per-step logging (to avoid huge files)
LOG_STEP_INTERVAL = 50

# Video recording: render a short eval rollout every N training steps (0 = disabled).
VIDEO_INTERVAL_STEPS = 2500
VIDEO_N_EVAL_STEPS = 300  # frames to capture per video

# SB3 SAC hyperparameters — identical across all backends
_SAC_KWARGS = dict(
    learning_rate=1e-3,
    buffer_size=int(3e5),
    learning_starts=1000,
    batch_size=256,
    tau=0.02,
    gamma=0.98,
    train_freq=(1, "episode"),
    gradient_steps=-1,
    policy_kwargs=dict(net_arch=dict(pi=[400, 300], qf=[400, 300])),
    verbose=0,
)

# ---------------------------------------------------------------------------
# SAR transform helpers
# ---------------------------------------------------------------------------


def load_sar(sar_dir: pathlib.Path):
    """Load pretrained SAR pkl files (ica, pca, normalizer).

    Args:
        sar_dir: Directory containing ``ica.pkl``, ``pca.pkl``,
                 ``normalizer.pkl``.

    Returns:
        Tuple ``(ica, pca, normalizer)`` sklearn objects.

    Raises:
        FileNotFoundError: If any pkl is missing.
    """
    import warnings

    import joblib

    for fname in ("ica.pkl", "pca.pkl", "normalizer.pkl"):
        p = sar_dir / fname
        if not p.exists():
            raise FileNotFoundError(
                f"SAR pkl not found: {p}\n"
                "Run the SAR play-phase training first or provide a custom --sar-dir."
            )
    with warnings.catch_warnings():
        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except ImportError:
            pass
        ica = joblib.load(sar_dir / "ica.pkl")
        pca = joblib.load(sar_dir / "pca.pkl")
        normalizer = joblib.load(sar_dir / "normalizer.pkl")
    return ica, pca, normalizer


def synergy_inverse_transform(
    syn_act: np.ndarray,
    ica,
    pca,
    normalizer,
) -> np.ndarray:
    """Map synergy-space action → muscle activations (CPU sklearn transform).

    Args:
        syn_act: Shape ``(n_syn,)`` synergy actions in ``[-1, 1]``.
        ica: Fitted FastICA.
        pca: Fitted PCA.
        normalizer: Fitted MinMaxScaler.

    Returns:
        Muscle activations, shape ``(n_muscles,)``, dtype float32.
    """
    muscle = pca.inverse_transform(
        ica.inverse_transform(normalizer.inverse_transform([syn_act]))
    )
    return muscle[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Diagnostic logging
# ---------------------------------------------------------------------------


class TrainingLogger:
    """Per-episode CSV logger for reward components and physics stats.

    Logs one row per episode with columns:
        timestep, ep_len, ep_reward, vel_reward, cyclic_hip, ref_rot,
        joint_angle_rew, solved_rate, mean_height, mean_com_vel_y,
        done_rate, wall_time_s

    Args:
        log_path: Path to the output CSV file.
        backend_name: Human-readable backend label (for log messages).
    """

    def __init__(self, log_path: pathlib.Path, backend_name: str) -> None:
        self._path = log_path
        self._backend = backend_name
        self._file = open(log_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "timestep",
                "episode",
                "ep_len",
                "ep_reward",
                "vel_reward",
                "cyclic_hip",
                "ref_rot",
                "joint_angle_rew",
                "solved_rate",
                "mean_height",
                "mean_com_vel_y",
                "done_count",
                "wall_time_s",
            ],
        )
        self._writer.writeheader()
        self._episode = 0
        self._t0 = time.time()

    def log_episode(self, row: dict) -> None:
        """Write one episode row.

        Args:
            row: Dict with the fields listed in the CSV header.
        """
        row["episode"] = self._episode
        row["wall_time_s"] = round(time.time() - self._t0, 2)
        self._writer.writerow(row)
        self._file.flush()
        self._episode += 1

        if self._episode % 20 == 0:
            log.info(
                "[%s] ep=%d  ts=%d  ep_rwd=%.2f  vel_rwd=%.3f",
                self._backend,
                self._episode,
                row["timestep"],
                row["ep_reward"],
                row.get("vel_reward", float("nan")),
            )

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._file.close()


class StepLogger:
    """Per-step CSV logger (sampled every ``LOG_STEP_INTERVAL`` steps).

    Logs one row per sampled step with columns:
        timestep, step_reward, vel_reward, cyclic_hip, ref_rot,
        joint_angle_rew, height, com_vel_y, qpos_2 (z-root),
        qpos_3..6 (torso quat), act_mean, act_std, muscle_force_mean

    Args:
        log_path: Path to the output CSV file.
    """

    _QPOS_FIELDS = [f"qpos_{i}" for i in range(7)]  # x, y, z, qw, qx, qy, qz

    def __init__(self, log_path: pathlib.Path) -> None:
        self._path = log_path
        self._file = open(log_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "timestep",
                "step_reward",
                "vel_reward",
                "cyclic_hip",
                "ref_rot",
                "joint_angle_rew",
                "height",
                "com_vel_y",
                "act_mean",
                "act_std",
                "muscle_force_mean",
            ]
            + self._QPOS_FIELDS,
        )
        self._writer.writeheader()

    def log_step(self, timestep: int, reward: float, info: dict) -> None:
        """Write one sampled step row.

        Args:
            timestep: Global environment step count.
            reward: Step reward (dense).
            info: Info dict from the env step (should contain reward components
                  and physics state from ``MJXGymnasiumAdapter`` or
                  the WalkEnvV0 ``rwd_dict``).
        """
        row: dict[str, Any] = {
            "timestep": timestep,
            "step_reward": round(reward, 6),
            "vel_reward": round(float(info.get("vel_reward", float("nan"))), 6),
            "cyclic_hip": round(float(info.get("cyclic_hip", float("nan"))), 6),
            "ref_rot": round(float(info.get("ref_rot", float("nan"))), 6),
            "joint_angle_rew": round(
                float(info.get("joint_angle_rew", float("nan"))), 6
            ),
            "height": round(float(info.get("height", float("nan"))), 6),
            "com_vel_y": round(float(info.get("com_vel_y", float("nan"))), 6),
            "act_mean": float("nan"),
            "act_std": float("nan"),
            "muscle_force_mean": float("nan"),
        }
        if "act" in info and info["act"] is not None:
            act = np.asarray(info["act"])
            row["act_mean"] = round(float(np.mean(act)), 6)
            row["act_std"] = round(float(np.std(act)), 6)
        if "muscle_force" in info and info["muscle_force"] is not None:
            row["muscle_force_mean"] = round(
                float(np.mean(np.abs(np.asarray(info["muscle_force"])))), 6
            )
        qpos = np.asarray(info.get("qpos", np.full(7, float("nan"))))
        for i, fname in enumerate(self._QPOS_FIELDS):
            row[fname] = round(float(qpos[i]) if i < len(qpos) else float("nan"), 6)

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._file.close()


# ---------------------------------------------------------------------------
# Backend-specific SB3 callback
# ---------------------------------------------------------------------------


def _record_sb3_video(
    model,
    norm_env,
    cpu_env_factory,
    run_dir: pathlib.Path,
    timestep: int,
    wandb_run: Any | None,
    wandb_backend: str | None,
    n_eval_steps: int = VIDEO_N_EVAL_STEPS,
) -> None:
    """Render a short evaluation rollout with the current policy and save to mp4.

    Called periodically during SB3 (CPU and MJX) training.  A fresh CPU
    gymnasium env is used for rendering because MJX and mjlab run headless.
    Obs normalisation stats are synced from the live training ``norm_env`` so
    the policy sees the same normalised observations it was trained on.

    Args:
        model: SB3 SAC model (policy in synergy action space).
        norm_env: Live VecNormalize training env (used to copy obs_rms).
        cpu_env_factory: ``() -> gymnasium.Env`` with ``render_mode="rgb_array"``.
        run_dir: Directory to save video files.
        timestep: Current training timestep (used in filename).
        wandb_run: Active wandb run or ``None``.
        wandb_backend: Backend prefix for wandb metric key (e.g. ``"cpu"``).
        n_eval_steps: Number of steps to capture.
    """
    import copy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    try:
        render_base = cpu_env_factory()
        render_vec = DummyVecEnv([lambda: Monitor(render_base)])
        render_norm = VecNormalize(
            render_vec,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )
        # Sync observation running stats from the live training env so the
        # policy sees correctly normalised observations during eval.
        render_norm.obs_rms = copy.deepcopy(norm_env.obs_rms)
        render_norm.ret_rms = copy.deepcopy(norm_env.ret_rms)

        frames: list[np.ndarray] = []
        obs = render_norm.reset()
        for _ in range(n_eval_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = render_norm.step(action)
            frame = render_base.render()  # rgb_array from inner gymnasium env
            if frame is not None:
                frames.append(frame)
            if dones[0]:
                obs = render_norm.reset()

        render_norm.close()

        if not frames:
            log.debug(
                "No frames captured for %s video at step %d", wandb_backend, timestep
            )
            return

        video_path = run_dir / f"video_{timestep:07d}.mp4"
        _save_frames_to_video(frames, video_path)

        if video_path.exists() and wandb_run is not None and wandb_backend is not None:
            try:
                import wandb

                wandb_run.log(
                    {
                        f"{wandb_backend}/eval_video": wandb.Video(
                            str(video_path), fps=50, format="mp4"
                        )
                    },
                    step=timestep,
                )
            except Exception as exc:
                log.debug("W&B video upload failed for %s: %s", wandb_backend, exc)

    except Exception as exc:
        log.warning(
            "Video render failed for %s at step %d (non-fatal): %s",
            wandb_backend,
            timestep,
            exc,
        )


def _save_frames_to_video(frames: list, path: pathlib.Path) -> None:
    """Write a list of uint8 RGB frames to an mp4 file (imageio, fps=50)."""
    try:
        import imageio.v3 as iio

        iio.imwrite(str(path), frames, fps=50, codec="libx264")
        log.info("Saved video → %s (%d frames)", path, len(frames))
    except Exception:
        try:
            import imageio

            with imageio.get_writer(str(path), fps=50) as w:
                for f in frames:
                    w.append_data(f)
            log.info("Saved video → %s (%d frames)", path, len(frames))
        except Exception as exc:
            fallback = path.with_suffix(".npz")
            np.savez_compressed(str(fallback), frames=np.stack(frames))
            log.warning(
                "imageio unavailable; saved raw frames as %s. Error: %s", fallback, exc
            )


def _make_sb3_callback(
    training_logger: TrainingLogger,
    step_logger: StepLogger,
    env_unwrapped,
    is_mjx: bool = False,
    *,
    wandb_run: Any | None = None,
    wandb_backend: str | None = None,
    norm_env=None,
    cpu_env_factory=None,
    run_dir: pathlib.Path | None = None,
    video_interval: int = 0,
):
    """Build an SB3 callback that logs diagnostics after every episode/step.

    Args:
        training_logger: Per-episode CSV logger.
        step_logger: Per-step CSV logger.
        env_unwrapped: The innermost env (after all wrappers) for CPU envs;
                       or the ``MJXGymnasiumAdapter`` for MJX envs.
        is_mjx: Whether the env is an MJX adapter (affects how info is read).

    Returns:
        An ``sb3.common.callbacks.BaseCallback`` instance.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _DiagnosticCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._ep_rewards: list[float] = []
            self._ep_vel_rewards: list[float] = []
            self._ep_cyclic: list[float] = []
            self._ep_ref_rot: list[float] = []
            self._ep_jang: list[float] = []
            self._ep_heights: list[float] = []
            self._ep_com_vels: list[float] = []
            self._ep_dones: int = 0
            self._ep_len: int = 0
            self._total_steps: int = 0

        def _on_step(self) -> bool:
            self._total_steps += 1
            self._ep_len += 1

            reward = float(self.locals["rewards"][0])
            info = self.locals["infos"][0]
            done = bool(self.locals["dones"][0])

            self._ep_rewards.append(reward)

            if is_mjx:
                # MJXGymnasiumAdapter puts component metrics in info directly
                self._ep_vel_rewards.append(float(info.get("vel_reward", np.nan)))
                self._ep_cyclic.append(float(info.get("cyclic_hip", np.nan)))
                self._ep_ref_rot.append(float(info.get("ref_rot", np.nan)))
                self._ep_jang.append(float(info.get("joint_angle_rew", np.nan)))
            else:
                # CPU env: info contains "rwd_dict" from MyoGymnasiumEnv.step
                rwd = info.get("rwd_dict", {})
                self._ep_vel_rewards.append(float(rwd.get("vel_reward", np.nan)))
                self._ep_cyclic.append(float(rwd.get("cyclic_hip", np.nan)))
                self._ep_ref_rot.append(float(rwd.get("ref_rot", np.nan)))
                self._ep_jang.append(float(rwd.get("joint_angle_rew", np.nan)))

            # Per-step diagnostics (sampled every LOG_STEP_INTERVAL)
            if self._total_steps % LOG_STEP_INTERVAL == 0:
                step_info = dict(info)
                if not is_mjx:
                    # Add CPU physics state to step_info (MyoGymnasiumEnv exposes model/data)
                    try:
                        raw = env_unwrapped
                        raw_data = getattr(raw, "data", None)
                        if raw_data is not None:
                            step_info["qpos"] = raw_data.qpos.copy()
                            step_info["act"] = raw_data.act.copy()
                            step_info["muscle_force"] = np.clip(
                                raw_data.actuator_force / 1000, -100, 100
                            )
                        # Extract height and com_vel from obs_dict if available
                        if hasattr(raw, "obs_dict"):
                            od = raw.obs_dict
                            height_val = np.asarray(od.get("height", np.array(np.nan)))
                            step_info["height"] = (
                                float(height_val.reshape(-1)[0])
                                if height_val.size
                                else np.nan
                            )
                            com_vel_val = np.asarray(
                                od.get("com_vel", np.array([np.nan, np.nan]))
                            )
                            step_info["com_vel_y"] = (
                                float(com_vel_val.reshape(-1)[1])
                                if com_vel_val.size >= 2
                                else np.nan
                            )
                        rwd = info.get("rwd_dict", {})
                        for k in (
                            "vel_reward",
                            "cyclic_hip",
                            "ref_rot",
                            "joint_angle_rew",
                        ):
                            step_info[k] = rwd.get(k, np.nan)
                    except AttributeError:
                        pass
                step_logger.log_step(self._total_steps, reward, step_info)

            # Periodic video rendering
            if (
                video_interval > 0
                and self._total_steps % video_interval == 0
                and cpu_env_factory is not None
                and norm_env is not None
                and run_dir is not None
            ):
                _record_sb3_video(
                    model=self.model,
                    norm_env=norm_env,
                    cpu_env_factory=cpu_env_factory,
                    run_dir=run_dir,
                    timestep=self._total_steps,
                    wandb_run=wandb_run,
                    wandb_backend=wandb_backend,
                )

            if done:
                self._ep_dones += 1

            # On episode end (SB3 uses "episode" key in info after truncation/termination)
            if "episode" in info:
                ep_data = info["episode"]
                row = {
                    "timestep": self._total_steps,
                    "ep_len": self._ep_len,
                    "ep_reward": round(
                        float(ep_data.get("r", sum(self._ep_rewards))), 4
                    ),
                    "vel_reward": round(
                        float(
                            np.nanmean(self._ep_vel_rewards)
                            if self._ep_vel_rewards
                            else np.nan
                        ),
                        4,
                    ),
                    "cyclic_hip": round(
                        float(
                            np.nanmean(self._ep_cyclic) if self._ep_cyclic else np.nan
                        ),
                        4,
                    ),
                    "ref_rot": round(
                        float(
                            np.nanmean(self._ep_ref_rot) if self._ep_ref_rot else np.nan
                        ),
                        4,
                    ),
                    "joint_angle_rew": round(
                        float(np.nanmean(self._ep_jang) if self._ep_jang else np.nan),
                        4,
                    ),
                    "solved_rate": float(ep_data.get("r", 0) > 5.0),
                    "mean_height": round(
                        float(np.nanmean(self._ep_heights))
                        if self._ep_heights
                        else float("nan"),
                        4,
                    ),
                    "mean_com_vel_y": round(
                        float(np.nanmean(self._ep_com_vels))
                        if self._ep_com_vels
                        else float("nan"),
                        4,
                    ),
                    "done_count": self._ep_dones,
                }
                training_logger.log_episode(row)

                if wandb_run is not None and wandb_backend is not None:
                    try:
                        # Log episode-level metrics to Weights & Biases.
                        metrics = {
                            f"{wandb_backend}/{k}": v
                            for k, v in row.items()
                            if k != "timestep"
                        }
                        wandb_run.log(metrics, step=row["timestep"])
                    except Exception:
                        # Wandb is strictly optional; never break training if logging fails.
                        log.debug(
                            "wandb logging failed for backend %s at timestep %d",
                            wandb_backend,
                            row["timestep"],
                        )
                # Reset per-episode accumulators
                self._ep_rewards = []
                self._ep_vel_rewards = []
                self._ep_cyclic = []
                self._ep_ref_rot = []
                self._ep_jang = []
                self._ep_heights = []
                self._ep_com_vels = []
                self._ep_dones = 0
                self._ep_len = 0

            return True

    return _DiagnosticCallback()


# ---------------------------------------------------------------------------
# Phase A: Deterministic rollout — compare physics/obs/reward between backends
# ---------------------------------------------------------------------------


def run_deterministic_comparison(
    output_dir: pathlib.Path,
    sar_dir: pathlib.Path,
    n_steps: int = 200,
    backends: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Run fixed action sequences on all backends; compare obs/reward/qpos.

    This phase is fast (no training) and reveals the raw numerical differences
    between backends for the same physical simulation.

    Args:
        output_dir: Directory to write comparison CSVs and plots.
        sar_dir: Path to pretrained SAR pkl files.
        n_steps: Number of rollout steps per backend.
        backends: List of backends to run (default: all available).

    Returns:
        Dict mapping backend name → reward array of shape ``(n_steps,)``.
    """
    log.info("=== Phase A: Deterministic rollout comparison (%d steps) ===", n_steps)
    output_dir.mkdir(parents=True, exist_ok=True)

    ica, pca, normalizer = load_sar(sar_dir)
    n_syn = pca.components_.shape[0]

    # Fixed random synergy actions
    rng = np.random.default_rng(0)
    syn_actions = rng.uniform(-1.0, 1.0, (n_steps, n_syn)).astype(np.float32)
    muscle_actions = np.array(
        [synergy_inverse_transform(s, ica, pca, normalizer) for s in syn_actions]
    )

    if backends is None:
        backends = ["cpu", "mjx_xla", "mjx_warp", "mjlab"]

    results: dict[str, dict] = {}

    # ---------- CPU backend ----------
    if "cpu" in backends:
        log.info("Running CPU deterministic rollout …")
        results["cpu"] = _run_cpu_rollout(muscle_actions)

    # ---------- MJX backends ----------
    _mjx_available = _check_mjx_available()
    if not _mjx_available:
        log.warning(
            "MJX not available; skipping MJX backends in deterministic comparison"
        )
        backends = [b for b in backends if "mjx" not in b]

    for impl_key, impl_val in [("mjx_xla", None), ("mjx_warp", "warp")]:
        if impl_key not in backends:
            continue
        if impl_val == "warp" and not _check_warp_available():
            log.warning("Warp backend not available; skipping mjx_warp")
            continue
        log.info("Running %s deterministic rollout …", impl_key)
        try:
            results[impl_key] = _run_mjx_rollout(muscle_actions, impl=impl_val)
        except Exception as exc:
            log.error("MJX rollout failed (%s): %s", impl_key, exc, exc_info=True)

    # ---------- mjlab backend ----------
    if "mjlab" in backends:
        if not _check_mjlab_available():
            log.warning(
                "mjlab not available; skipping mjlab backend in deterministic comparison"
            )
        else:
            log.info("Running mjlab deterministic rollout …")
            try:
                results["mjlab"] = _run_mjlab_rollout(muscle_actions)
            except Exception as exc:
                log.error("mjlab rollout failed: %s", exc, exc_info=True)

    # ---------- Save per-backend CSV ----------
    for backend, data in results.items():
        bdir = output_dir / backend
        bdir.mkdir(exist_ok=True)
        _save_rollout_csv(bdir / "deterministic_rollout.csv", data)

    # ---------- Divergence report ----------
    if len(results) > 1:
        _write_divergence_report(output_dir / "divergence_report.txt", results, n_steps)
        _plot_rollout_comparison(output_dir, results)

    return {k: v["rewards"] for k, v in results.items()}


def _run_cpu_rollout(muscle_actions: np.ndarray) -> dict:
    """Run the CPU gymnasium env with fixed muscle actions.

    Args:
        muscle_actions: Shape ``(T, nu)`` float32 muscle activations.

    Returns:
        Dict with arrays: rewards, qpos_traj, vel_reward, cyclic_hip, ref_rot,
        joint_angle_rew, height_traj, com_vel_y_traj, act_traj.
    """
    import myosuite

    myosuite.register_all_envs()
    import gymnasium as gym

    # reset_type="init" selects key_qpos[2] (standing pose), matching MJX and MJLAB
    # which always initialise from keyframe 2.  Default (reset_type=None) uses key_qpos[0].
    env = gym.make("myoLegWalk-v0", reset_type="init")
    env.reset(seed=0)

    # Force deterministic initial state (qpos0 = key_qpos[2], zero vel)
    import mujoco

    unwrapped = env.unwrapped
    # SAR CPU envs used by this benchmark (myoLegWalk-v0) are MyoGymnasiumEnv; use data/model.
    mj_model = getattr(unwrapped, "model", None)
    mj_data = getattr(unwrapped, "data", None)
    if mj_data is None or mj_model is None:
        raise AttributeError(
            "CPU SAR benchmark expects MyoGymnasiumEnv (data/model). "
            "Legacy mj_data/mj_model is no longer supported here."
        )

    mj_data.qpos[:] = mj_model.key_qpos.reshape(mj_model.nkey, mj_model.nq)[2]
    mj_data.qvel[:] = 0.0
    unwrapped.steps = 0
    mujoco.mj_forward(mj_model, mj_data)

    T = len(muscle_actions)
    rewards = np.full(T, np.nan)
    qpos_traj = np.full((T, mj_model.nq), np.nan)
    vel_rewards = np.full(T, np.nan)
    cyclic_hips = np.full(T, np.nan)
    ref_rots = np.full(T, np.nan)
    jang_rews = np.full(T, np.nan)
    heights = np.full(T, np.nan)
    com_vels_y = np.full(T, np.nan)
    act_traj = np.full((T, mj_model.na), np.nan)

    for i, action in enumerate(muscle_actions):
        _, rwd, terminated, truncated, info = env.step(action)
        rewards[i] = rwd
        qpos_traj[i] = mj_data.qpos.copy()
        act_traj[i] = mj_data.act.copy()
        rwd_dict = info.get("rwd_dict", {})
        vel_rewards[i] = rwd_dict.get("vel_reward", np.nan)
        cyclic_hips[i] = rwd_dict.get("cyclic_hip", np.nan)
        ref_rots[i] = rwd_dict.get("ref_rot", np.nan)
        jang_rews[i] = rwd_dict.get("joint_angle_rew", np.nan)

        od = getattr(env.unwrapped, "obs_dict", {})
        if od:
            height_val = np.asarray(od.get("height", np.array(np.nan)))
            heights[i] = float(height_val.reshape(-1)[0]) if height_val.size else np.nan

            com_vel_val = np.asarray(od.get("com_vel", np.array([np.nan, np.nan])))
            com_vels_y[i] = (
                float(com_vel_val.reshape(-1)[1]) if com_vel_val.size >= 2 else np.nan
            )
        else:
            heights[i] = np.nan
            com_vels_y[i] = np.nan
        if terminated or truncated:
            log.info("CPU rollout terminated at step %d", i)
            break

    env.close()
    return {
        "rewards": rewards,
        "qpos": qpos_traj,
        "act": act_traj,
        "vel_reward": vel_rewards,
        "cyclic_hip": cyclic_hips,
        "ref_rot": ref_rots,
        "joint_angle_rew": jang_rews,
        "height": heights,
        "com_vel_y": com_vels_y,
    }


def _run_mjx_rollout(muscle_actions: np.ndarray, impl: str | None = None) -> dict:
    """Run the MJX env with fixed muscle actions (no SAR wrapper needed).

    Args:
        muscle_actions: Shape ``(T, nu)`` float32 muscle activations.
        impl: ``None`` for XLA, ``"warp"`` for Warp backend.

    Returns:
        Same dict structure as ``_run_cpu_rollout``.
    """
    import copy

    import jax
    import jax.numpy as jnp

    from myosuite.envs.myo.backends.mjx import _leg_walk_config
    from myosuite.envs.myo.backends.mjx.walk_env import MjxWalkEnv

    cfg = copy.deepcopy(_leg_walk_config)
    cfg["mjx_impl"] = impl
    cfg["num_envs"] = 1

    env = MjxWalkEnv(config=cfg)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    T = len(muscle_actions)
    rewards = np.full(T, np.nan)
    qpos_traj = np.full((T, env.mj_model.nq), np.nan)
    vel_rewards = np.full(T, np.nan)
    cyclic_hips = np.full(T, np.nan)
    ref_rots = np.full(T, np.nan)
    jang_rews = np.full(T, np.nan)
    heights = np.full(T, np.nan)
    com_vels_y = np.full(T, np.nan)
    act_traj = np.full((T, env.mj_model.na), np.nan)

    for i, action in enumerate(muscle_actions):
        action_jax = jnp.array(action, dtype=jnp.float32)
        state = jit_step(state, action_jax)

        rewards[i] = float(state.reward)
        qpos_traj[i] = np.array(state.data.qpos)
        act_traj[i] = np.array(state.data.act)
        vel_rewards[i] = float(state.metrics.get("vel_reward", jnp.nan))
        cyclic_hips[i] = float(state.metrics.get("cyclic_hip", jnp.nan))
        ref_rots[i] = float(state.metrics.get("ref_rot", jnp.nan))
        jang_rews[i] = float(state.metrics.get("joint_angle_rew", jnp.nan))

        if bool(state.done):
            log.info("MJX (%s) rollout terminated at step %d", impl or "xla", i)
            break

    return {
        "rewards": rewards,
        "qpos": qpos_traj,
        "act": act_traj,
        "vel_reward": vel_rewards,
        "cyclic_hip": cyclic_hips,
        "ref_rot": ref_rots,
        "joint_angle_rew": jang_rews,
        "height": heights,
        "com_vel_y": com_vels_y,
    }


def _run_mjlab_rollout(muscle_actions: np.ndarray) -> dict:
    """Run the mjlab env with fixed muscle actions using the native torch API.

    Creates ``myoLegWalk-v0`` on the mjlab backend and steps it directly with
    ``torch.Tensor`` actions — no gymnasium adapter or numpy round-trips.
    Results are converted to numpy only once, at the end.

    Args:
        muscle_actions: Shape ``(T, nu)`` float32 muscle activations.

    Returns:
        Same dict structure as ``_run_cpu_rollout``.
    """
    import torch
    from myosuite.core.registry import make_env

    mjlab_env = make_env("myoLegWalk-v0", backend="mjlab")
    device = str(getattr(mjlab_env, "device", "cpu"))
    mjlab_env.reset(seed=0)

    T = len(muscle_actions)
    rewards = np.full(T, np.nan)
    qpos_traj = np.full((T, 35), np.nan)  # myolegs nq=35
    vel_rewards = np.full(T, np.nan)
    cyclic_hips = np.full(T, np.nan)
    ref_rots = np.full(T, np.nan)
    jang_rews = np.full(T, np.nan)
    heights = np.full(T, np.nan)
    com_vels_y = np.full(T, np.nan)
    act_traj = np.full((T, 80), np.nan)  # myolegs nu=80

    def _scalar(x) -> float:
        if x is None:
            return np.nan
        return float(x.item() if hasattr(x, "item") else x)

    def _to_np(x) -> np.ndarray:
        if x is None:
            return np.array([])
        if hasattr(x, "cpu"):
            x = x.cpu()
        return np.asarray(x).ravel()

    for i, action in enumerate(muscle_actions):
        action_t = torch.as_tensor(
            action, dtype=torch.float32, device=device
        ).unsqueeze(0)
        result = mjlab_env.step(action_t)
        if len(result) == 5:
            _, rew_raw, term_raw, trunc_raw, info_raw = result
        else:
            _, rew_raw, term_raw, info_raw = result
            trunc_raw = False

        rewards[i] = _scalar(rew_raw)
        terminated = bool(term_raw.item() if hasattr(term_raw, "item") else term_raw)
        truncated = bool(trunc_raw.item() if hasattr(trunc_raw, "item") else trunc_raw)

        extras = info_raw if isinstance(info_raw, dict) else {}
        # mjlab returns reward components inside extras["log"]; fall back to top-level.
        log_dict = extras.get("log", extras)

        qpos = _to_np(extras.get("qpos"))
        if qpos.size:
            qpos_traj[i, : min(len(qpos), 35)] = qpos[: min(len(qpos), 35)]
        act = _to_np(extras.get("act"))
        if act.size:
            act_traj[i, : min(len(act), 80)] = act[: min(len(act), 80)]

        vel_rewards[i] = _scalar(log_dict.get("vel_reward"))
        cyclic_hips[i] = _scalar(log_dict.get("cyclic_hip"))
        ref_rots[i] = _scalar(log_dict.get("ref_rot"))
        jang_rews[i] = _scalar(log_dict.get("joint_angle_rew"))
        heights[i] = _scalar(log_dict.get("height"))
        com_vels_y[i] = _scalar(log_dict.get("com_vel_y"))

        if terminated or truncated:
            log.info("mjlab rollout terminated at step %d", i)
            break

    if hasattr(mjlab_env, "close"):
        mjlab_env.close()
    return {
        "rewards": rewards,
        "qpos": qpos_traj,
        "act": act_traj,
        "vel_reward": vel_rewards,
        "cyclic_hip": cyclic_hips,
        "ref_rot": ref_rots,
        "joint_angle_rew": jang_rews,
        "height": heights,
        "com_vel_y": com_vels_y,
    }


def _save_rollout_csv(path: pathlib.Path, data: dict) -> None:
    """Save rollout data to CSV (one row per step).

    Args:
        path: Output CSV file path.
        data: Dict of arrays from ``_run_cpu_rollout`` / ``_run_mjx_rollout``.
    """
    T = len(data["rewards"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "reward",
                "vel_reward",
                "cyclic_hip",
                "ref_rot",
                "joint_angle_rew",
                "height",
                "com_vel_y",
                "qpos_2",
                "qpos_3",
                "qpos_4",
                "qpos_5",
                "qpos_6",
            ]
        )
        for i in range(T):
            qpos = data["qpos"][i]
            writer.writerow(
                [
                    i,
                    data["rewards"][i],
                    data["vel_reward"][i],
                    data["cyclic_hip"][i],
                    data["ref_rot"][i],
                    data["joint_angle_rew"][i],
                    data["height"][i],
                    data["com_vel_y"][i],
                    qpos[2] if len(qpos) > 2 else np.nan,  # z-root
                    qpos[3] if len(qpos) > 3 else np.nan,  # qw
                    qpos[4] if len(qpos) > 4 else np.nan,  # qx
                    qpos[5] if len(qpos) > 5 else np.nan,  # qy
                    qpos[6] if len(qpos) > 6 else np.nan,  # qz
                ]
            )


def _write_divergence_report(
    path: pathlib.Path,
    results: dict[str, dict],
    n_steps: int,
) -> None:
    """Write a human-readable divergence report comparing all backends.

    Reports max/mean absolute divergence per output field between the CPU
    backend (reference) and each MJX backend, computed over all non-NaN steps.

    Args:
        path: Output text file path.
        results: Dict mapping backend name → rollout data dict.
        n_steps: Total rollout length (for context).
    """
    ref_key = "cpu" if "cpu" in results else next(iter(results))
    ref = results[ref_key]
    fields = ["rewards", "vel_reward", "cyclic_hip", "ref_rot", "joint_angle_rew"]

    lines = [
        "=" * 70,
        "SAR Backend Divergence Report",
        f"Reference backend: {ref_key}  |  Rollout length: {n_steps} steps",
        "=" * 70,
        "",
    ]

    for other_key, other in results.items():
        if other_key == ref_key:
            continue
        lines.append(f"{'─' * 50}")
        lines.append(f"  {ref_key}  vs  {other_key}")
        lines.append(f"{'─' * 50}")
        for field in fields:
            a = np.asarray(ref[field], dtype=float)
            b = np.asarray(other[field], dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            if not mask.any():
                lines.append(f"  {field:<25}  (no valid data)")
                continue
            diff = np.abs(a[mask] - b[mask])
            lines.append(
                f"  {field:<25}  max_diff={diff.max():.6f}  "
                f"mean_diff={diff.mean():.6f}  "
                f"first_diverge={int(np.argmax(diff > 0.01)) if (diff > 0.01).any() else -1}"
            )

        # qpos divergence (all DOFs)
        qp_ref = np.asarray(ref["qpos"])
        qp_other = np.asarray(other["qpos"])
        mask_q = np.all(np.isfinite(qp_ref) & np.isfinite(qp_other), axis=1)
        if mask_q.any():
            qp_diff = np.abs(qp_ref[mask_q] - qp_other[mask_q])
            lines.append(
                f"  {'qpos (all DOFs)':<25}  max_diff={qp_diff.max():.6f}  "
                f"mean_diff={qp_diff.mean():.6f}  "
                f"worst_dof={int(np.argmax(qp_diff.max(axis=0)))}"
            )
        lines.append("")

    lines.append(
        "\nDiagnostic guide:\n"
        "  • max_diff > 0.1 on vel_reward → COM velocity computation mismatch\n"
        "  • max_diff > 0.01 on ref_rot   → quaternion convention difference\n"
        "  • max_diff > 0.01 on qpos      → physics integration divergence\n"
        "  • first_diverge early + qpos diff growing → contact model difference\n"
        "  • first_diverge late + small qpos diff → reward threshold sensitivity\n"
    )

    path.write_text("\n".join(lines))
    log.info("Divergence report written to: %s", path)


def _plot_rollout_comparison(output_dir: pathlib.Path, results: dict) -> None:
    """Generate rollout comparison plots (reward components + qpos).

    Args:
        output_dir: Directory to save PNG files.
        results: Dict mapping backend name → rollout data dict.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping plots")
        return

    colors = {
        "cpu": "#1f77b4",
        "mjx_xla": "#ff7f0e",
        "mjx_warp": "#2ca02c",
        "mjlab": "#d62728",
    }

    # --- Reward components ---
    fields = ["rewards", "vel_reward", "cyclic_hip", "ref_rot", "joint_angle_rew"]
    titles = [
        "Dense reward",
        "vel_reward",
        "cyclic_hip (lower=better)",
        "ref_rot",
        "joint_angle_rew",
    ]
    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 3 * len(fields)))
    for ax, field, title in zip(axes, fields, titles):
        for backend, data in results.items():
            arr = np.asarray(data[field], dtype=float)
            t = np.arange(len(arr))
            valid = np.isfinite(arr)
            ax.plot(
                t[valid],
                arr[valid],
                label=backend,
                color=colors.get(backend, "gray"),
                alpha=0.85,
            )
        ax.set_title(title)
        ax.set_xlabel("Rollout step")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Deterministic Rollout: Reward Components per Backend", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_components.png", dpi=120)
    plt.close(fig)
    log.info("Saved comparison_components.png")

    # --- qpos z-height (index 2) ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for backend, data in results.items():
        qpos = np.asarray(data["qpos"])
        if qpos.shape[1] > 2:
            z = qpos[:, 2]
            t = np.arange(len(z))
            valid = np.isfinite(z)
            ax2.plot(
                t[valid],
                z[valid],
                label=backend,
                color=colors.get(backend, "gray"),
                alpha=0.85,
            )
    ax2.axhline(0.8, color="red", linestyle="--", label="min_height=0.8")
    ax2.set_title("COM Z-height over rollout steps")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Z (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "comparison_physics.png", dpi=120)
    plt.close(fig2)
    log.info("Saved comparison_physics.png")


# ---------------------------------------------------------------------------
# Phase B: Short SAC training comparison
# ---------------------------------------------------------------------------


def run_training_comparison(
    output_dir: pathlib.Path,
    sar_dir: pathlib.Path,
    n_steps: int,
    backends: list[str],
    seed: int = 0,
    wandb_cfg: dict[str, Any] | None = None,
    video_interval: int = VIDEO_INTERVAL_STEPS,
) -> None:
    """Train SB3 SAC with SynergyWrapper on each backend for n_steps.

    All backends use identical SAC hyperparameters and the pretrained SAR
    transform, so differences in reward curves indicate physics divergence.

    Args:
        output_dir: Root output directory.
        sar_dir: Path to pretrained SAR pkl files.
        n_steps: Total SAC training timesteps per backend.
        backends: Which backends to run (``["cpu", "mjx_xla", "mjx_warp"]``).
        seed: Random seed for reproducibility.
        video_interval: Render a video every N training steps (0 = disabled).
    """
    log.info("=== Phase B: SAC training comparison (%d steps) ===", n_steps)
    output_dir.mkdir(parents=True, exist_ok=True)

    ica, pca, normalizer = load_sar(sar_dir)

    if "cpu" in backends:
        _run_cpu_training(
            output_dir / "cpu",
            ica,
            pca,
            normalizer,
            n_steps,
            seed,
            wandb_cfg=wandb_cfg,
            video_interval=video_interval,
        )

    # Run mjlab before MJX so it gets a clean CUDA context (MJX Warp can leave GPU in bad state).
    if "mjlab" in backends:
        if not _check_mjlab_available():
            log.warning(
                "mjlab not available; skipping mjlab backend in training comparison"
            )
        else:
            try:
                _run_mjlab_training(
                    output_dir / "mjlab",
                    ica,
                    pca,
                    normalizer,
                    n_steps,
                    seed,
                    wandb_cfg=wandb_cfg,
                    video_interval=video_interval,
                )
            except Exception as exc:
                log.error("mjlab training failed: %s", exc, exc_info=True)
                if "allocate" in str(exc).lower() or "cuda" in str(exc).lower():
                    log.info(
                        "Hint: GPU memory or CUDA context may be exhausted. "
                        "Try freeing GPU memory or run with --backends cpu mjlab to run only mjlab."
                    )

    _mjx_available = _check_mjx_available()
    if not _mjx_available and any("mjx" in b for b in backends):
        log.warning("MJX not available; skipping MJX backends in training comparison")
        backends = [b for b in backends if "mjx" not in b]

    for impl_key, impl_val in [("mjx_xla", None), ("mjx_warp", "warp")]:
        if impl_key not in backends:
            continue
        if impl_val == "warp" and not _check_warp_available():
            log.warning("Warp not available; skipping %s", impl_key)
            continue
        try:
            _run_mjx_training(
                output_dir / impl_key,
                ica,
                pca,
                normalizer,
                n_steps,
                seed,
                impl=impl_val,
                wandb_cfg=wandb_cfg,
                video_interval=video_interval,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error("MJX training failed (%s): %s", impl_key, exc, exc_info=True)

    # Generate convergence plots
    _plot_training_comparison(output_dir, backends)


def _run_cpu_training(
    run_dir: pathlib.Path,
    ica,
    pca,
    normalizer,
    n_steps: int,
    seed: int,
    *,
    wandb_cfg: dict[str, Any] | None = None,
    video_interval: int = VIDEO_INTERVAL_STEPS,
) -> None:
    """SB3 SAC training on the CPU gymnasium backend.

    Uses ``myoLegWalk-v0`` wrapped with ``SynergyWrapper`` (sklearn).

    Diagnostic info is written to:
        run_dir/training_log.csv  (per-episode)
        run_dir/steps_log.csv     (per step, sampled)

    Args:
        run_dir: Output directory for this backend.
        ica, pca, normalizer: Pretrained SAR components.
        n_steps: Total training timesteps.
        seed: Random seed.
        video_interval: Render a video every N steps (0 = disabled).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Training CPU backend → %s", run_dir)

    import gymnasium as gym
    import myosuite

    myosuite.register_all_envs()
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # Inner env — reset_type="init" selects key_qpos[2] (standing pose), matching MJX/MJLAB
    base_env = gym.make("myoLegWalk-v0", reset_type="init")

    # SAR wrapper (sklearn, CPU-side)
    class _SynergyWrapper(gym.ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            n_syn = pca.components_.shape[0]
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(n_syn,), dtype=np.float32
            )

        def action(self, act):
            return synergy_inverse_transform(act, ica, pca, normalizer)

    wrapped = _SynergyWrapper(base_env)
    monitored = Monitor(wrapped)
    vec_env = DummyVecEnv([lambda: monitored])
    norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    ep_logger = TrainingLogger(run_dir / "training_log.csv", "cpu")
    step_logger = StepLogger(run_dir / "steps_log.csv")
    wandb_run = _init_wandb_run(
        backend="cpu",
        run_dir=run_dir,
        n_steps=n_steps,
        seed=seed,
        wandb_cfg=wandb_cfg,
    )

    def _cpu_env_factory():
        import gymnasium as _gym
        import myosuite as _ms  # noqa: F401

        _base = _gym.make("myoLegWalk-v0", reset_type="init", render_mode="rgb_array")
        return _SynergyWrapper(_base)

    cb = _make_sb3_callback(
        ep_logger,
        step_logger,
        env_unwrapped=base_env.unwrapped,
        is_mjx=False,
        wandb_run=wandb_run,
        wandb_backend="cpu",
        norm_env=norm_env,
        cpu_env_factory=_cpu_env_factory,
        run_dir=run_dir,
        video_interval=video_interval,
    )

    # Force CPU device for the nominal "CPU" backend so tests and benchmarks
    # run correctly even when a CUDA-enabled PyTorch install is present.
    model = SAC("MlpPolicy", norm_env, seed=seed, device="cpu", **_SAC_KWARGS)
    t0 = time.time()
    model.learn(total_timesteps=n_steps, callback=cb, log_interval=None)
    elapsed = time.time() - t0

    model.save(run_dir / "sac_model")
    norm_env.save(run_dir / "vec_normalize.pkl")

    ep_logger.close()
    step_logger.close()
    log.info("CPU training done in %.1f s. Logs: %s", elapsed, run_dir)

    if wandb_run is not None:
        try:
            import wandb as _wandb  # noqa: F401

            wandb_run.finish()
        except Exception:
            log.debug("Failed to finish wandb run for CPU backend")


def _run_mjx_training(
    run_dir: pathlib.Path,
    ica,
    pca,
    normalizer,
    n_steps: int,
    seed: int,
    impl: str | None = None,
    *,
    wandb_cfg: dict[str, Any] | None = None,
    video_interval: int = VIDEO_INTERVAL_STEPS,
) -> None:
    """SB3 SAC training on the MJX backend (XLA or Warp).

    Wraps ``MjxLegWalk-v0`` as a gymnasium env via ``MJXGymnasiumAdapter``,
    then applies the same ``SynergyWrapper`` and SB3 SAC pipeline as CPU.

    This single-env wrapper is slower than vectorised JAX training but ensures
    algorithm comparability with the CPU backend (same SAC, same hyperparams).

    Args:
        run_dir: Output directory for this backend.
        ica, pca, normalizer: Pretrained SAR components.
        n_steps: Total training timesteps.
        seed: Random seed.
        impl: ``None`` for MJX XLA, ``"warp"`` for MuJoCo Warp.
        video_interval: Render a video every N steps (0 = disabled).
    """
    import copy

    run_dir.mkdir(parents=True, exist_ok=True)
    impl_label = "warp" if impl == "warp" else "xla"
    log.info("Training MJX (%s) backend → %s", impl_label, run_dir)

    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from myosuite.envs.myo.backends.mjx import _leg_walk_config
    from myosuite.envs.myo.backends.mjx.walk_env import MjxWalkEnv
    from benchmarks.sar_backends.mjx_gymnasium_adapter import MJXGymnasiumAdapter

    cfg = copy.deepcopy(_leg_walk_config)
    cfg["mjx_impl"] = impl
    cfg["num_envs"] = 1

    mjx_env = MjxWalkEnv(config=cfg)

    # Gymnasium adapter
    gym_env = MJXGymnasiumAdapter(mjx_env, seed=seed)

    # SAR wrapper — same sklearn transform as CPU
    class _SynergyWrapper(gym.ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            n_syn = pca.components_.shape[0]
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(n_syn,), dtype=np.float32
            )

        def action(self, act):
            return synergy_inverse_transform(act, ica, pca, normalizer)

    wrapped = _SynergyWrapper(gym_env)
    monitored = Monitor(wrapped)
    vec_env = DummyVecEnv([lambda: monitored])
    norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    ep_logger = TrainingLogger(run_dir / "training_log.csv", f"mjx_{impl_label}")
    step_logger = StepLogger(run_dir / "steps_log.csv")
    wandb_backend = f"mjx_{impl_label}"
    wandb_run = _init_wandb_run(
        backend=wandb_backend,
        run_dir=run_dir,
        n_steps=n_steps,
        seed=seed,
        wandb_cfg=wandb_cfg,
    )

    # MJX doesn't support rendering; use a CPU env for video capture.
    def _mjx_cpu_env_factory():
        import gymnasium as _gym
        import myosuite as _ms  # noqa: F401

        _base = _gym.make("myoLegWalk-v0", reset_type="init", render_mode="rgb_array")
        return _SynergyWrapper(_base)

    cb = _make_sb3_callback(
        ep_logger,
        step_logger,
        env_unwrapped=gym_env,
        is_mjx=True,
        wandb_run=wandb_run,
        wandb_backend=wandb_backend,
        norm_env=norm_env,
        cpu_env_factory=_mjx_cpu_env_factory,
        run_dir=run_dir,
        video_interval=video_interval,
    )

    model = SAC("MlpPolicy", norm_env, seed=seed, **_SAC_KWARGS)
    t0 = time.time()
    model.learn(total_timesteps=n_steps, callback=cb, log_interval=None)
    elapsed = time.time() - t0

    model.save(run_dir / "sac_model")
    norm_env.save(run_dir / "vec_normalize.pkl")

    ep_logger.close()
    step_logger.close()
    log.info("MJX (%s) training done in %.1f s. Logs: %s", impl_label, elapsed, run_dir)

    if wandb_run is not None:
        try:
            import wandb as _wandb  # noqa: F401

            wandb_run.finish()
        except Exception:
            log.debug("Failed to finish wandb run for MJX backend %s", impl_label)


def _run_mjlab_training(
    run_dir: pathlib.Path,
    ica,
    pca,
    normalizer,
    n_steps: int,
    seed: int,
    *,
    wandb_cfg: dict[str, Any] | None = None,
    video_interval: int = VIDEO_INTERVAL_STEPS,
) -> None:
    """GPU-native RSL-RL PPO training on the mjlab (MuJoCo Warp / Isaac Lab) backend.

    This function keeps **all tensors on GPU** throughout the training loop:
      - Physics simulation: mjlab MuJoCo Warp (GPU)
      - SAR inverse transform: ``SARTorchTransform`` (torch matmul on GPU)
      - Policy forward pass: RSL-RL actor-critic (GPU)
      - Rollout buffer and PPO update: torch (GPU)


    Algorithm note: mjlab uses RSL-RL PPO (on-policy), while CPU/MJX backends
    use SB3 SAC (off-policy).  Reward curves are not directly comparable
    step-for-step, but both should show convergence if SAR is working.

    Diagnostic info is written to:
        run_dir/training_log.csv  (per-iteration: timestep, ep_reward, …)

    Args:
        run_dir: Output directory for this backend.
        ica, pca, normalizer: Pretrained SAR components (sklearn).
        n_steps: Approximate total training timesteps (converted to PPO iterations).
        seed: Random seed (sets torch manual seed).
    """
    import torch
    from tensordict import TensorDict

    from myosuite.core.registry import make_env
    from benchmarks.sar_backends.sar_torch_transform import SARTorchTransform
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _walk_ppo_runner_cfg,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Training mjlab (RSL-RL PPO, GPU-native) → %s", run_dir)

    torch.manual_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        import gc

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    log.info("mjlab training device: %s", device)

    # ── SAR transform (GPU-resident fixed weights) ───────────────────────────
    sar_transform = SARTorchTransform(ica, pca, normalizer, device=device)
    sar_transform.eval()
    n_syn = sar_transform.n_syn
    log.info("SAR: %d synergies → %d muscles", n_syn, sar_transform.n_muscles)

    # ── mjlab environment ────────────────────────────────────────────────────
    mjlab_env = make_env("myoLegWalk-v0", backend="mjlab", device=device)

    # ── SAR env wrapper (accepts synergy actions, applies SAR on GPU) ────────
    class _SAREnvWrapper:
        """Thin wrapper: intercepts step() to apply SAR before env.step().

        Exposes the minimal VecEnv interface expected by ``rsl_rl.OnPolicyRunner``:

          - ``num_envs``, ``num_actions``, ``device``
          - ``get_observations() -> TensorDict`` with keys ``\"policy\"``/``\"critic\"``
          - ``step(actions) -> (TensorDict, rewards, dones, info, extras)``
        """

        def __init__(self, env, sar):
            self._env = env
            self._sar = sar
            self.device = device
            self.num_envs = getattr(env, "num_envs", 1)
            self.num_actions = n_syn
            # Expose underlying mjlab config for RSL-RL logging.
            self.cfg = getattr(env, "cfg", None)
            # Self-tracking accumulators — independent of RSL-RL internals.
            self._step_stats: dict[str, list[float]] = {}
            self._ep_rewards: list[float] = []
            self._ep_lengths: list[int] = []
            self._ep_reward_acc = 0.0
            self._ep_length_acc = 0

        @property
        def episode_length_buf(self):
            return self._env.episode_length_buf

        @episode_length_buf.setter
        def episode_length_buf(self, value):
            self._env.episode_length_buf = value

        @property
        def max_episode_length(self):
            return self._env.max_episode_length

        def get_observations(self):
            """Return current observations as a TensorDict with policy/critic keys."""
            obs_dict = self._env.observation_manager.compute()
            return TensorDict(obs_dict, batch_size=[self.num_envs])

        def step(self, syn_actions: torch.Tensor):
            """Apply SAR to synergy actions and step the underlying mjlab env.

            RSL-RL ``OnPolicyRunner`` expects the environment ``step`` API to
            return a 4-tuple ``(obs, rewards, dones, infos)``. We keep the
            mjlab/Isaac-style ``extras`` dict internal for logging via
            :meth:`get_and_reset_stats` rather than returning it.
            """
            syn_actions = syn_actions.to(self.device)
            if syn_actions.ndim == 1:
                syn_actions = syn_actions.unsqueeze(0)

            with torch.no_grad():
                muscle_actions = self._sar(syn_actions)  # (num_envs, n_muscles)

            obs_dict, rew, term, trunc, extras = self._env.step(muscle_actions)
            term_or_trunc = term | trunc
            dones = term_or_trunc.to(dtype=torch.long)

            obs_td = TensorDict(obs_dict, batch_size=[self.num_envs])
            extras_dict = extras if isinstance(extras, dict) else {}

            # mjlab exposes per-step reward components under extras["log"] with
            # keys like \"Episode_Reward/vel_reward\" or \"Episode_Termination/time_out\".
            # Fall back to the top-level dict when \"log\" is absent.
            log_dict = extras_dict.get("log", extras_dict)

            # Accumulate per-step reward component stats, normalising mjlab's
            # key names (e.g. \"Episode_Reward/vel_reward\" → \"vel_reward\").
            for k, v in log_dict.items():
                val = v.mean().item() if hasattr(v, "mean") else float(v)
                base_k = k
                if "/" in k:
                    # Episode_Reward/vel_reward → vel_reward
                    base_k = k.split("/", 1)[1]
                self._step_stats.setdefault(base_k, []).append(val)

            # Track cumulative episode reward & length for ep-level stats.
            rew_mean = rew.mean().item() if hasattr(rew, "mean") else float(rew)
            self._ep_reward_acc += rew_mean
            self._ep_length_acc += 1
            if dones.any():
                self._ep_rewards.append(self._ep_reward_acc)
                self._ep_lengths.append(self._ep_length_acc)
                self._ep_reward_acc = 0.0
                self._ep_length_acc = 0

            infos: dict = {}
            if log_dict:
                ep_dict: dict[str, float] = {}
                for k, v in log_dict.items():
                    base_k = k
                    if "/" in k:
                        base_k = k.split("/", 1)[1]
                    if base_k.startswith("rew_"):
                        continue
                    val = v.mean().item() if hasattr(v, "mean") else float(v)
                    ep_dict[f"rew_{base_k}"] = val
                if ep_dict:
                    infos["episode"] = ep_dict
            if "time_outs" in extras_dict:
                infos["time_outs"] = extras_dict["time_outs"]

            # VecEnv API (as used by RSL-RL): (obs, rewards, dones, infos)
            return obs_td, rew, dones, infos

        def get_and_reset_stats(self) -> tuple[dict[str, float], float, float]:
            """Return accumulated step/episode stats and reset internal accumulators.

            Returns:
                step_means: Per-step mean of each reward component from extras["log"].
                ep_rew: Mean cumulative reward over completed episodes, or NaN.
                ep_len: Mean length of completed episodes, or NaN.
            """
            step_means = {
                k: float(np.mean(v)) for k, v in self._step_stats.items() if v
            }
            ep_rew = (
                float(np.mean(self._ep_rewards)) if self._ep_rewards else float("nan")
            )
            ep_len = (
                float(np.mean(self._ep_lengths)) if self._ep_lengths else float("nan")
            )
            self._step_stats.clear()
            self._ep_rewards.clear()
            self._ep_lengths.clear()
            return step_means, ep_rew, ep_len

    wrapped_env = _SAREnvWrapper(mjlab_env, sar_transform)

    # ── RSL-RL PPO runner ────────────────────────────────────────────────────
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as e:
        raise ImportError(
            "rsl_rl is required for GPU-native mjlab training. "
            "Install with: pip install rsl-rl"
        ) from e

    train_cfg = _walk_ppo_runner_cfg()
    # Our mjlab walk env only exposes a single observation group "policy".
    # Use it for policy, critic, and actor observation sets so RSL-RL's
    # resolve_obs_groups() finds valid keys without emitting warnings.
    train_cfg.obs_groups = {
        "policy": ("policy",),
        "critic": ("policy",),
        "actor": ("policy",),
    }
    # Convert n_steps (total timesteps) → PPO iterations
    steps_per_iter = train_cfg.num_steps_per_env * wrapped_env.num_envs
    n_iterations = max(1, n_steps // steps_per_iter)
    log.info(
        "RSL-RL PPO: %d iterations × %d steps/env = ~%d total steps",
        n_iterations,
        train_cfg.num_steps_per_env,
        n_iterations * steps_per_iter,
    )

    # rsl_rl.OnPolicyRunner expects a plain dict config; convert the mjlab
    # dataclass config to a nested dict.
    train_cfg_dict = asdict(train_cfg)
    # Older rsl_rl versions do not accept ``cnn_cfg`` in MLPModel.__init__;
    # drop it defensively from actor/critic configs before constructing runner.
    for key in ("actor", "critic"):
        if key in train_cfg_dict and isinstance(train_cfg_dict[key], dict):
            train_cfg_dict[key].pop("cnn_cfg", None)
    # Force TensorBoard logging inside rsl_rl to avoid implicit WandB usage,
    # since the outer benchmark already handles optional wandb logging.
    train_cfg_dict["logger"] = "tensorboard"

    runner = OnPolicyRunner(
        env=wrapped_env,
        train_cfg=train_cfg_dict,
        log_dir=str(run_dir / "rslrl_logs"),
        device=device,
    )

    # ── Per-iteration CSV logger ─────────────────────────────────────────────
    ep_logger = TrainingLogger(run_dir / "training_log.csv", "mjlab")
    _rslrl_global_step = [0]

    _orig_log = runner.log if hasattr(runner, "log") else None

    wandb_run = _init_wandb_run(
        backend="mjlab",
        run_dir=run_dir,
        n_steps=n_steps,
        seed=seed,
        wandb_cfg=wandb_cfg,
    )

    def _log_hook(locs: dict) -> None:
        """Called by RSL-RL OnPolicyRunner.log(locals()) after each iteration.

        Reads metrics primarily from the self-tracking accumulator in
        ``wrapped_env`` (populated every step, independent of RSL-RL internals).
        Also checks RSL-RL ``locs`` as a fallback, supporting both the standard
        RSL-RL key names (``ep_infos``, ``rewbuffer``, ``lenbuffer``) and the
        Isaac Lab variant names (``mean_reward``, ``mean_episode_length``).
        """
        _rslrl_global_step[0] += steps_per_iter

        # Primary: read from self-tracked accumulator (version-independent)
        step_stats, ep_rew, ep_len = wrapped_env.get_and_reset_stats()

        # Fallback: also check locs for RSL-RL-provided aggregate stats
        ep_infos = locs.get("ep_infos", [])
        if ep_infos:
            ep_stats_from_locs: dict[str, list[float]] = {}
            for ep_info in ep_infos:
                for key, val in ep_info.items():
                    v = val.mean().item() if hasattr(val, "mean") else float(val)
                    ep_stats_from_locs.setdefault(key, []).append(v)
            # Merge: fill in any keys absent from step_stats
            for k, vs in ep_stats_from_locs.items():
                step_stats.setdefault(k.removeprefix("rew_"), float(np.mean(vs)))

        # Try RSL-RL alternate key names for episode reward/length
        if np.isnan(ep_rew):
            for key in ("rewbuffer", "mean_reward"):
                buf = locs.get(key)
                if buf:
                    ep_rew = float(np.mean(list(buf)))
                    break
        if np.isnan(ep_len):
            for key in ("lenbuffer", "mean_episode_length"):
                buf = locs.get(key)
                if buf:
                    ep_len = float(np.mean(list(buf)))
                    break

        row = {
            "timestep": _rslrl_global_step[0],
            "ep_len": ep_len,
            "ep_reward": ep_rew,
            "vel_reward": step_stats.get("vel_reward", float("nan")),
            "cyclic_hip": step_stats.get("cyclic_hip", float("nan")),
            "ref_rot": step_stats.get("ref_rot", float("nan")),
            "joint_angle_rew": step_stats.get("joint_angle_rew", float("nan")),
            "done_count": len(ep_infos),
        }
        ep_logger.log_episode(row)

        if wandb_run is not None:
            try:
                metrics = {f"mjlab/{k}": v for k, v in row.items() if k != "timestep"}
                wandb_run.log(metrics, step=row["timestep"])
            except Exception:
                log.debug(
                    "wandb logging failed for mjlab at timestep %d", row["timestep"]
                )
        # Periodic video rendering during training
        if video_interval > 0 and _rslrl_global_step[0] % video_interval == 0:
            _render_mjlab_eval(
                run_dir=run_dir,
                runner=runner,
                wrapped_env=wrapped_env,
                sar_transform=sar_transform,
                n_render_steps=VIDEO_N_EVAL_STEPS,
                wandb_run=wandb_run,
                video_suffix=f"_{_rslrl_global_step[0]:07d}",
            )

        if _orig_log is not None:
            _orig_log(locs)

    if hasattr(runner, "log"):
        runner.log = _log_hook

    t0 = time.time()
    runner.learn(num_learning_iterations=n_iterations, init_at_random_ep_len=True)
    elapsed = time.time() - t0

    runner.save(str(run_dir / "ppo_model.pt"))
    ep_logger.close()
    log.info("mjlab (RSL-RL PPO) training done in %.1f s. Logs: %s", elapsed, run_dir)

    # ── Post-training evaluation with video rendering ────────────────────────
    try:
        _render_mjlab_eval(
            run_dir=run_dir,
            runner=runner,
            wrapped_env=wrapped_env,
            sar_transform=sar_transform,
            n_render_steps=500,
            wandb_run=wandb_run,
        )
    except Exception as exc:
        log.warning("mjlab post-training render failed (non-fatal): %s", exc)

    if wandb_run is not None:
        try:
            import wandb as _wandb  # noqa: F401

            wandb_run.finish()
        except Exception:
            log.debug("Failed to finish wandb run for mjlab backend")


def _get_policy_from_runner(runner: Any) -> Any | None:
    """Extract the actor-critic policy from an RSL-RL OnPolicyRunner.

    RSL-RL stores the policy under different attribute paths depending on
    version (standard RSL-RL vs Isaac Lab fork).  Try each known path in
    order and return the first non-None result.

    Args:
        runner: An ``OnPolicyRunner`` (or compatible) object.

    Returns:
        The policy object if found, otherwise ``None``.
    """
    # Prefer the actor-critic policy object (with .act/.act_inference)
    # over lower-level network exports like alg.get_policy(), which
    # return bare MLPModel instances without an act() method.
    for getter in [
        lambda r: getattr(getattr(r, "alg", None), "policy", None),
        lambda r: getattr(getattr(r, "alg", None), "actor_critic", None),
        lambda r: getattr(r, "actor_critic", None),
        lambda r: getattr(r, "policy", None),
    ]:
        try:
            obj = getter(runner)
            if obj is not None:
                return obj
        except Exception:
            continue
    return None


def _render_mjlab_eval(
    run_dir: pathlib.Path,
    runner,
    wrapped_env,
    sar_transform,
    n_render_steps: int = 500,
    wandb_run: Any = None,
    video_suffix: str = "",
) -> None:
    """Evaluate the mjlab policy on the CPU env and save/upload a video.

    Called both periodically during training (with a ``video_suffix`` like
    ``"_0002500"``) and once after training completes (suffix ``""``).

    Uses ``myoLegWalk-v0`` with ``render_mode="rgb_array"`` for rendering because
    mjlab's MuJoCo Warp backend runs headless on GPU. The trained RSL-RL policy is
    evaluated on the CPU env (obs/actions converted at each step for display only).

    The rendered mp4 is saved to ``run_dir/eval_rollout{video_suffix}.mp4`` and
    optionally uploaded to W&B as ``mjlab/eval_video``.
    """
    import torch

    log.info("Rendering mjlab evaluation%s (%d steps) …", video_suffix, n_render_steps)

    try:
        import myosuite

        myosuite.register_all_envs()
        import gymnasium as gym

        env = gym.make("myoLegWalk-v0", reset_type="init", render_mode="rgb_array")
        obs_cpu, _ = env.reset(seed=42)
    except Exception as exc:
        log.warning("Could not create CPU render env: %s", exc)
        return

    device = wrapped_env.device
    policy = _get_policy_from_runner(runner)
    if policy is None:
        log.warning(
            "Could not access trained policy from runner (tried alg.actor_critic, "
            "actor_critic, policy, alg.policy); skipping render."
        )
        env.close()
        return
    policy.eval()

    frames: list[np.ndarray] = []
    obs_t = torch.as_tensor(obs_cpu, dtype=torch.float32, device=device).unsqueeze(0)

    for _ in range(n_render_steps):
        with torch.no_grad():
            if hasattr(policy, "act_inference"):
                syn_action = policy.act_inference(obs_t)
            else:
                syn_action = policy.act(obs_t)[0]
            muscle_action = sar_transform(syn_action)  # (1, n_muscles)

        obs_cpu, _, terminated, truncated, _ = env.step(
            muscle_action.squeeze(0).cpu().numpy()
        )
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            obs_cpu, _ = env.reset()

        obs_t = torch.as_tensor(obs_cpu, dtype=torch.float32, device=device).unsqueeze(
            0
        )

    env.close()

    if not frames:
        log.warning("No frames captured during mjlab eval render.")
        return

    video_path = run_dir / f"eval_rollout{video_suffix}.mp4"
    _save_frames_to_video(frames, video_path)

    if video_path.exists():
        if wandb_run is not None:
            try:
                import wandb

                wandb_run.log(
                    {
                        "mjlab/eval_video": wandb.Video(
                            str(video_path), fps=50, format="mp4"
                        )
                    },
                    commit=True,
                )
                log.info("Uploaded eval video to W&B.")
            except Exception as exc:
                log.warning("W&B video upload failed: %s", exc)


def load_mjlab_policy_and_sar(
    run_dir: pathlib.Path,
    sar_dir: pathlib.Path,
    device: str = "cuda:0",
) -> tuple[Any, Any]:
    """Load mjlab PPO policy and SAR transform from a Phase B run directory.

    Used by the render script to evaluate the trained mjlab policy on the CPU
    env for video rendering. Reconstructs the runner (env + wrapper + config),
    loads the checkpoint, and returns the actor_critic and SAR transform.

    Args:
        run_dir: Phase B run directory containing ``ppo_model.pt``.
        sar_dir: Directory containing ``ica.pkl``, ``pca.pkl``, ``normalizer.pkl``.
        device: Torch device for policy and SAR (e.g. ``"cuda:0"`` or ``"cpu"``).

    Returns:
        Tuple ``(policy, sar_transform)`` where policy is the RSL-RL actor_critic
        and sar_transform is ``SARTorchTransform``.

    Raises:
        FileNotFoundError: If run_dir or sar_dir is missing required files.
        ImportError: If rsl_rl or mjlab dependencies are not available.
    """
    from tensordict import TensorDict

    from myosuite.core.registry import make_env
    from benchmarks.sar_backends.sar_torch_transform import SARTorchTransform
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _walk_ppo_runner_cfg,
    )

    checkpoint_path = run_dir / "ppo_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"mjlab checkpoint not found: {checkpoint_path}")

    ica, pca, normalizer = load_sar(sar_dir)
    sar_transform = SARTorchTransform(ica, pca, normalizer, device=device)
    sar_transform.eval()
    n_syn = sar_transform.n_syn

    mjlab_env = make_env("myoLegWalk-v0", backend="mjlab", device=device)

    class _SAREnvWrapper:
        def __init__(self, env, sar, dev, n_synergy):
            self._env = env
            self._sar = sar
            self.device = dev
            self.num_envs = getattr(env, "num_envs", 1)
            self.num_actions = n_synergy
            self.cfg = getattr(env, "cfg", None)

        @property
        def episode_length_buf(self):
            return self._env.episode_length_buf

        @episode_length_buf.setter
        def episode_length_buf(self, value):
            self._env.episode_length_buf = value

        @property
        def max_episode_length(self):
            return self._env.max_episode_length

        def get_observations(self):
            obs_dict = self._env.observation_manager.compute()
            return TensorDict(obs_dict, batch_size=[self.num_envs])

        def step(self, syn_actions):
            import torch

            syn_actions = syn_actions.to(self.device)
            if syn_actions.ndim == 1:
                syn_actions = syn_actions.unsqueeze(0)
            with torch.no_grad():
                muscle_actions = self._sar(syn_actions)
            obs_dict, rew, term, trunc, extras = self._env.step(muscle_actions)
            term_or_trunc = term | trunc
            dones = term_or_trunc.to(dtype=torch.long)
            obs_td = TensorDict(obs_dict, batch_size=[self.num_envs])
            extras_dict = extras if isinstance(extras, dict) else {}
            log_dict = extras_dict.get("log", {})
            infos = {}
            if log_dict:
                infos["episode"] = {
                    k: (v.mean().item() if hasattr(v, "mean") else float(v))
                    for k, v in log_dict.items()
                    if not k.startswith("rew_")
                }
            if "time_outs" in extras_dict:
                infos["time_outs"] = extras_dict["time_outs"]
            # Match RSL-RL runner API: return (obs, rewards, dones, infos)
            return obs_td, rew, dones, infos

    wrapped_env = _SAREnvWrapper(mjlab_env, sar_transform, device, n_syn)

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as e:
        raise ImportError(
            "rsl_rl is required to load mjlab policy. pip install rsl-rl"
        ) from e

    train_cfg = _walk_ppo_runner_cfg()
    train_cfg.obs_groups = {
        "policy": ("policy",),
        "critic": ("policy",),
        "actor": ("policy",),
    }

    train_cfg_dict = asdict(train_cfg)
    for key in ("actor", "critic"):
        if key in train_cfg_dict and isinstance(train_cfg_dict[key], dict):
            train_cfg_dict[key].pop("cnn_cfg", None)
    train_cfg_dict["logger"] = "tensorboard"

    runner = OnPolicyRunner(
        env=wrapped_env,
        train_cfg=train_cfg_dict,
        log_dir=str(run_dir / "rslrl_logs"),
        device=device,
    )
    runner.load(str(checkpoint_path))
    # RSL-RL PPO exposes the actor-critic as alg.policy (not actor_critic)
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    if policy is None:
        raise RuntimeError("Could not get policy from loaded mjlab runner")
    policy.eval()
    return policy, sar_transform


def _init_wandb_run(
    backend: str,
    run_dir: pathlib.Path,
    n_steps: int,
    seed: int,
    wandb_cfg: dict[str, Any] | None,
):
    """Initialise a Weights & Biases run for a given backend, if configured.

    Args:
        backend: Backend label (e.g. ``\"cpu\"``, ``\"mjx_xla\"``, ``\"mjlab\"``).
        run_dir: Directory where backend-specific logs/checkpoints are written.
        n_steps: Total training timesteps requested.
        seed: Random seed.
        wandb_cfg: Optional dictionary with configuration keys:
            ``project``, ``entity``, ``group``, ``run_prefix``.

    Returns:
        A wandb Run object if initialisation succeeds, otherwise ``None``.
    """
    if wandb_cfg is None:
        return None

    project = wandb_cfg.get("project")
    if not project:
        # Without an explicit project we treat wandb as disabled.
        return None

    try:
        import wandb
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning(
            "wandb not available; disabling wandb logging for backend %s: %s",
            backend,
            exc,
        )
        return None

    entity = wandb_cfg.get("entity")
    group = wandb_cfg.get("group")
    run_prefix = wandb_cfg.get("run_prefix") or "sar-benchmark"

    run_name = f"{run_prefix}-{backend}-seed{seed}"

    config = {
        "backend": backend,
        "n_steps": n_steps,
        "seed": seed,
        "run_dir": str(run_dir),
    }

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=run_name,
            config=config,
            dir=str(run_dir),
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Failed to initialise wandb run for backend %s: %s", backend, exc)
        return None

    return run


def _plot_training_comparison(output_dir: pathlib.Path, backends: list[str]) -> None:
    """Generate reward convergence plots from training_log.csv files.

    Args:
        output_dir: Root directory containing per-backend subdirs.
        backends: List of backend names to plot.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping training comparison plots")
        return

    colors = {
        "cpu": "#1f77b4",
        "mjx_xla": "#ff7f0e",
        "mjx_warp": "#2ca02c",
        "mjlab": "#d62728",
    }
    fields = ["ep_reward", "vel_reward", "cyclic_hip", "ref_rot", "joint_angle_rew"]
    titles = [
        "Episode reward (dense)",
        "vel_reward (forward velocity)",
        "cyclic_hip (lower=better match)",
        "ref_rot (upright posture)",
        "joint_angle_rew (hip joint penalty)",
    ]

    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 4 * len(fields)))

    for backend in backends:
        csv_path = output_dir / backend / "training_log.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            continue
        ts = np.array([float(r["timestep"]) for r in rows])
        for ax, field, title in zip(axes, fields, titles):
            vals = np.array([float(r.get(field, "nan")) for r in rows])
            # Smooth for readability
            if len(vals) > 10:
                kernel = np.ones(10) / 10
                vals_smooth = np.convolve(vals, kernel, mode="same")
            else:
                vals_smooth = vals
            ax.plot(
                ts,
                vals_smooth,
                label=backend,
                color=colors.get(backend, "gray"),
                alpha=0.85,
            )
            ax.plot(ts, vals, color=colors.get(backend, "gray"), alpha=0.15)

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Training timestep")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "SAR Backend Training Comparison: SAC on myoLegWalk",
        fontsize=14,
    )
    fig.tight_layout()
    out_path = output_dir / "comparison_reward.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved comparison_reward.png → %s", out_path)

    # Per-step component comparison
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9))
    component_fields = ["vel_reward", "cyclic_hip", "ref_rot"]
    for backend in backends:
        csv_path = output_dir / backend / "steps_log.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            continue
        ts = np.array([float(r["timestep"]) for r in rows])
        for ax, field in zip(axes2, component_fields):
            vals = np.array([float(r.get(field, "nan")) for r in rows])
            valid = np.isfinite(vals)
            ax.plot(
                ts[valid],
                vals[valid],
                label=backend,
                color=colors.get(backend, "gray"),
                alpha=0.6,
            )

    for ax, field in zip(axes2, component_fields):
        ax.set_title(f"Per-step {field}")
        ax.set_xlabel("Training timestep")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig2.suptitle("SAR Backend Step-level Reward Components", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(output_dir / "comparison_step_components.png", dpi=120)
    plt.close(fig2)
    log.info("Saved comparison_step_components.png")


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def _check_mjx_available() -> bool:
    try:
        from mujoco import mjx as _  # noqa: F401
        import jax as _jax  # noqa: F401
        import mujoco_playground as _mp  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


def _check_warp_available() -> bool:
    try:
        import warp  # noqa: F401

        return True
    except ImportError:
        return False


def _check_mjlab_available() -> bool:
    """Return True if mjlab backend is usable for myoLegWalk-v0.

    We require:
      - ``mjlab`` and ``torch`` to be importable, and
      - the mjlab scene for ``myoLegWalk-v0`` to compile successfully.

    Some mjlab / MuJoCo combinations currently fail when composing the MyoSuite
    leg model into an mjlab ``Scene`` (``ValueError: repeated name '.../' in key``).
    In that case we treat the mjlab backend as unavailable so the benchmark
    can still run on CPU/MJX backends without crashing.
    """
    try:
        import mjlab  # noqa: F401
        import torch  # noqa: F401  # required alongside mjlab for availability check
    except ImportError:
        return False

    # Smoke-test env creation once to avoid cryptic runtime failures later.
    # Use CPU for the smoke test to avoid spurious GPU OOM / solver errors
    # from heavy Warp initialisation; actual training can still run on CUDA.
    try:
        from myosuite.core.registry import make_env

        env = make_env("myoLegWalk-v0", backend="mjlab", device="cpu")
        # Close immediately; we only care that construction succeeds.
        if hasattr(env, "close"):
            env.close()
    except Exception:  # pragma: no cover - defensive
        return False

    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the benchmark pipeline."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10_000,
        help="Total SAC training timesteps per backend (Phase B). Default: 10000.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("./sar_benchmark_results"),
        help="Directory to write all output CSVs and plots.",
    )
    parser.add_argument(
        "--sar-dir",
        type=pathlib.Path,
        default=_SAR_DIR,
        help="Path to pretrained SAR pkl files (ica.pkl, pca.pkl, normalizer.pkl).",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["cpu", "mjx_xla", "mjx_warp", "mjlab"],
        default=["cpu", "mjx_xla", "mjx_warp", "mjlab"],
        help="Which backends to benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-phase-a",
        action="store_true",
        help="Skip the deterministic rollout comparison (Phase A).",
    )
    parser.add_argument(
        "--skip-phase-b",
        action="store_true",
        help="Skip the SAC training comparison (Phase B).",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=200,
        help="Steps for the deterministic rollout comparison (Phase A). Default: 200.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help=(
            "Weights & Biases project name. If provided, enables wandb logging "
            "for all selected backends."
        ),
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team/user). Optional.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Weights & Biases group name for this benchmark run. Optional.",
    )
    parser.add_argument(
        "--wandb-run-prefix",
        type=str,
        default="sar-benchmark",
        help=(
            "Prefix used when constructing per-backend wandb run names. "
            'Final name is "<prefix>-<backend>-seed<seed>".'
        ),
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=VIDEO_INTERVAL_STEPS,
        help=(
            f"Render a short eval video every N training steps (default: {VIDEO_INTERVAL_STEPS}). "
            "Set to 0 to disable in-training video rendering."
        ),
    )
    args = parser.parse_args()

    output_dir: pathlib.Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log to file too
    file_handler = logging.FileHandler(output_dir / "benchmark.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)

    log.info("SAR Backend Benchmark")
    log.info("  Backends : %s", args.backends)
    log.info("  Steps    : %d (Phase B)", args.steps)
    log.info("  Output   : %s", output_dir)
    log.info("  SAR dir  : %s", args.sar_dir)
    if args.wandb_project:
        log.info(
            "  wandb    : project=%s, entity=%s, group=%s, prefix=%s",
            args.wandb_project,
            args.wandb_entity,
            args.wandb_group,
            args.wandb_run_prefix,
        )
    else:
        log.info("  wandb    : disabled (no --wandb-project provided)")

    wandb_cfg: dict[str, Any] | None = None
    if args.wandb_project:
        wandb_cfg = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "run_prefix": args.wandb_run_prefix,
        }

    # In headless environments (no DISPLAY), disable in-training video rendering
    # by default to avoid GLFW/OpenGL context errors. Users can override by
    # explicitly passing --video-interval > 0 in an environment with a valid
    # OpenGL context.
    if args.video_interval and args.video_interval > 0:
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            log.warning(
                "No DISPLAY detected; disabling in-training video rendering "
                "(set --video-interval 0 to silence this message, or run with "
                "a valid OpenGL context to enable videos)."
            )
            args.video_interval = 0

    # Phase A: deterministic rollout
    if not args.skip_phase_a:
        run_deterministic_comparison(
            output_dir=output_dir / "phase_a_rollout",
            sar_dir=args.sar_dir,
            n_steps=args.rollout_steps,
            backends=args.backends,
        )

    # Phase B: SAC training
    if not args.skip_phase_b:
        run_training_comparison(
            output_dir=output_dir / "phase_b_training",
            sar_dir=args.sar_dir,
            n_steps=args.steps,
            backends=args.backends,
            seed=args.seed,
            wandb_cfg=wandb_cfg,
            video_interval=args.video_interval,
        )

    log.info("Benchmark complete. Results in: %s", output_dir)


if __name__ == "__main__":
    main()
