#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Full SAR (Synergy-based Action Representation) pipeline on MJX with JAX PPO.

This script replicates the pipeline from ``tutorials/sar/run_sar_full.py``
but runs entirely on MJX (JAX) and GPU when available. All four steps
use the same backend for consistency.

Pipeline (matches run_sar_full.py):

  1. **Play phase**: Train JAX PPO on MjxLegWalk-v0 for PLAY_STEPS.
     Saves policy params to ``{output_dir}/play_phase_params.pkl``.

  2. **Activation rollout**: Load the trained policy, roll out ACTIVATION_EPISODES
     on MjxLegWalk-v0, collect muscle activations (state.data.act) from
     episodes above ACTIVATION_PERCENTILE reward. Saves ``muscle_activations.npy``.

  3. **SAR computation**: Fit PCA → ICA → MinMaxScaler on the activation data.
     Optionally plot VAF (variance accounted for) for 1..n_muscles synergies.
     Saves ``ica.pkl``, ``pca.pkl``, ``normalizer.pkl``, and ``vaf_plot.png``.

  4. **SAR-RL**: Train JAX PPO on MjxLegWalk-v0 wrapped with a synergy action
     space (policy outputs N_SYNERGIES dims in [-1, 1]; wrapper converts to
     muscle via the fitted SAR). Saves ``sar_rl_params.pkl``.

  5. **Rendering** (optional): If RENDER_ROLLOUT=1 (default), after the play
     phase and after SAR-RL the script runs a short rollout and saves videos
     to ``{output_dir}/play_rollout.mp4`` and ``{output_dir}/sar_rl_rollout.mp4``.

Requirements:
    pip install wandb scikit-learn matplotlib joblib imageio gymnasium
    pip install -e ".[mjx]"   # brax, mujoco_playground, jax

Usage:
    python scripts/train_sar_jax_ppo.py

    # With GPU and render play + SAR-RL rollouts (default):
    JAX_PLATFORMS=cuda python scripts/train_sar_jax_ppo.py

    # Render video(s) only from existing checkpoint(s) (no training):
    RENDER_ONLY=1 python scripts/train_sar_jax_ppo.py
    # Uses SAR_OUTPUT_DIR/MJX_ENV (default: sar_outputs/MjxLegWalk-v0). Writes
    # play_rollout.mp4 from play_phase_params.pkl; if sar_rl_params.pkl and
    # locomotion_*.pkl exist, also writes sar_rl_rollout.mp4.
    RENDER_ONLY=1 SAR_OUTPUT_DIR=/path/to/sar_outputs python scripts/train_sar_jax_ppo.py

    # Continue from play phase: run with play_phase_params.pkl already in output_dir.
    # The script loads it and runs activation rollout → SAR → SAR-RL (no re-training of play).
    python scripts/train_sar_jax_ppo.py   # same as normal; checkpoint detected automatically

    # SAR-RL only (re-train SAR-RL from existing SAR; skip play + activation + SAR computation):
    # Ensure locomotion_ica.pkl, locomotion_pca.pkl, locomotion_scaler.pkl exist in output_dir.
    RUN_PLAY_PHASE=false python scripts/train_sar_jax_ppo.py

    # Disable video rendering (faster, no video files):
    RENDER_ROLLOUT=0 python scripts/train_sar_jax_ppo.py

    # Custom render length and size:
    RENDER_STEPS=300 RENDER_VIDEO_WIDTH=640 RENDER_VIDEO_HEIGHT=480 python scripts/train_sar_jax_ppo.py

Testing:
    pytest myosuite/tests/mjx/test_train_sar_jax_ppo.py -v -k "not smoke"   # fast SAR helpers
    JAX_PLATFORMS=cpu pytest myosuite/tests/mjx/test_train_sar_jax_ppo.py -v -k smoke  # full pipeline (slow)

Troubleshooting (GPU / CUDA / cuSolver):
    * If you see errors like:
        - gpusolverDnCreate(&handle) failed: cuSolver internal error
        - failed to create cublas handle: the resource allocation failed
      they often indicate GPU memory preallocation issues in XLA/JAX. A simple
      workaround is to disable XLA's preallocation:

        export XLA_PYTHON_CLIENT_PREALLOCATE=false

      Then re-run with JAX_PLATFORMS=cuda. You can also always fall back to:

        export JAX_PLATFORMS=cpu

      to run the full pipeline on CPU only.

Env vars (optional):
    ENV_ID / MJX_ENV: MjxLegWalk-v0 (play and SAR-RL use same env; no MjxHillyTerrain yet).
    PLAY_STEPS: 1_500_000 (play phase timesteps).
    SAR_RL_STEPS: 2_500_000 (SAR-RL phase timesteps).
    ACTIVATION_EPISODES: 1000.
    ACTIVATION_PERCENTILE: 80 (only episodes above this reward percentile used for SAR).
    N_SYNERGIES: 20.
    JAX_PPO_NUM_ENVS: 1024 (vectorized envs for PPO).
    JAX_PLATFORMS: cpu by default; set to cuda for NVIDIA GPU (use cuda not gpu).
    RUN_PLAY_PHASE: true | false (skip play + activation + SAR, only run SAR-RL from existing SAR pkl).
    SKIP_SAR_RL: false (set true to run only steps 1–3).
    WANDB_MODE: offline | online | disabled.
    RENDER_ROLLOUT: 1 | 0 — save play_rollout.mp4 and sar_rl_rollout.mp4 (default 1).
    RENDER_ONLY: 1 | 0 — only load checkpoint(s) and render video(s), then exit (default 0).
    RENDER_STEPS: 500 — number of frames per video.
    RENDER_VIDEO_HEIGHT, RENDER_VIDEO_WIDTH: 320, 480 — video size in pixels.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

# Use CPU by default so script runs without CUDA; set JAX_PLATFORMS=cuda for NVIDIA GPU
# (JAX treats "gpu" as ROCm/AMD; use "cuda" for NVIDIA)
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"
if sys.platform == "darwin":
    # egl is Linux-only; macOS has no EGL backend and would hard-fail on import.
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import FastICA, PCA  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

import myosuite  # noqa: E402

myosuite.register_all_envs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (matches run_sar_full.py where applicable)
# ---------------------------------------------------------------------------

MJX_ENV = os.environ.get("MJX_ENV", "MjxLegWalk-v0")
PLAY_STEPS = int(os.environ.get("PLAY_STEPS", "1500000"))
SAR_RL_STEPS = int(os.environ.get("SAR_RL_STEPS", "2500000"))
ACTIVATION_EPISODES = int(os.environ.get("ACTIVATION_EPISODES", "1000"))
ACTIVATION_PERCENTILE = int(os.environ.get("ACTIVATION_PERCENTILE", "80"))
N_SYNERGIES = int(os.environ.get("N_SYNERGIES", "20"))
SEED = int(os.environ.get("SEED", "0"))
OUTPUT_DIR = Path(os.environ.get("SAR_OUTPUT_DIR", "sar_outputs")) / MJX_ENV

RUN_PLAY_PHASE = os.environ.get("RUN_PLAY_PHASE", "true").lower() in (
    "1",
    "true",
    "yes",
)
SKIP_SAR_RL = os.environ.get("SKIP_SAR_RL", "false").lower() in ("1", "true", "yes")
RUN_WANDB = os.environ.get("WANDB_MODE", "offline").lower() != "disabled"
RENDER_ROLLOUT = os.environ.get("RENDER_ROLLOUT", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
RENDER_ONLY = os.environ.get("RENDER_ONLY", "0").strip().lower() in ("1", "true", "yes")
RENDER_STEPS = int(os.environ.get("RENDER_STEPS", "500"))
RENDER_VIDEO_HEIGHT = int(os.environ.get("RENDER_VIDEO_HEIGHT", "320"))
RENDER_VIDEO_WIDTH = int(os.environ.get("RENDER_VIDEO_WIDTH", "480"))

JAX_PPO_NUM_ENVS = int(os.environ.get("JAX_PPO_NUM_ENVS", "1024"))
JAX_PPO_NUM_EVAL_ENVS = int(os.environ.get("JAX_PPO_NUM_EVAL_ENVS", "128"))


# ---------------------------------------------------------------------------
# SAR transform helpers
# ---------------------------------------------------------------------------


def synergy_to_muscle(
    syn_act: np.ndarray,
    ica: FastICA,
    pca: PCA,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Map synergy-space action [-1, 1] to muscle activations [0, 1].

    Matches tutorials/sar/sar_tutorial_utils SynergyWrapper inverse transform.
    """
    syn = np.atleast_2d(syn_act)
    ica_inv = ica.inverse_transform(scaler.inverse_transform(syn))
    muscle = pca.inverse_transform(ica_inv)
    muscle = np.clip(muscle, 0.0, 1.0)
    return muscle.reshape(-1).astype(np.float32)


def synergy_to_muscle_batch(
    syn_batch: np.ndarray, ica: FastICA, pca: PCA, scaler: MinMaxScaler
) -> np.ndarray:
    """Batch version for (n_envs, n_synergies) -> (n_envs, n_muscles)."""
    syn = np.asarray(syn_batch, dtype=np.float64)
    if syn.ndim == 1:
        syn = syn.reshape(1, -1)
    ica_inv = ica.inverse_transform(scaler.inverse_transform(syn))
    muscle = pca.inverse_transform(ica_inv)
    muscle = np.clip(muscle, 0.0, 1.0)
    return muscle.astype(np.float32)


def _load_play_checkpoint(ckpt_path: Path, mjx_env_name: str) -> tuple[Any, Any] | None:
    """Load play phase checkpoint; return (make_inference_fn, params) or None.

    Supports checkpoint format: dict with "params" and "env_name" (reconstructs
    inference fn), or legacy 2-tuple (make_inference_fn, params).
    """
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        log.warning("Could not load checkpoint %s: %s", ckpt_path, e)
        return None
    if isinstance(data, dict) and "params" in data:
        params = data["params"]
        env_name = data.get("env_name", mjx_env_name)
        make_inference_fn = _make_play_inference_fn(env_name)
        return (make_inference_fn, params)
    if isinstance(data, tuple) and len(data) == 2:
        return data
    log.warning("Unknown checkpoint format in %s", ckpt_path)
    return None


def _load_sar_rl_checkpoint(
    ckpt_path: Path,
    mjx_env_name: str,
    ica: FastICA,
    pca: PCA,
    scaler: MinMaxScaler,
) -> tuple[Any, Any] | None:
    """Load SAR-RL checkpoint; return (make_inference_fn, params) or None."""
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        log.warning("Could not load SAR-RL checkpoint %s: %s", ckpt_path, e)
        return None
    if not isinstance(data, dict) or "params" not in data:
        log.warning("Unknown SAR-RL checkpoint format in %s", ckpt_path)
        return None
    params = data["params"]
    env_name = data.get("env_name", mjx_env_name)
    try:
        import functools
        import jax
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks

        from myosuite.envs.myo.backends.mjx import make, ppo_config
        from mujoco_playground import wrapper

        env = make(env_name, config_overrides={"num_envs": 1})
        brax_env = wrapper.wrap_for_brax_training(
            env,
            episode_length=env._config.max_episode_steps,
            action_repeat=1,
        )
        wrapped = _SynergyWrapper(brax_env, ica, pca, scaler)
        rng = jax.random.PRNGKey(0)
        state = wrapped.reset(rng=rng[None, :])
        obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
        ppo_params = dict(ppo_config)
        if "network_factory" in ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **ppo_params.pop("network_factory")
            )
        else:
            network_factory = ppo_networks.make_ppo_networks
        normalize = running_statistics.normalize
        ppo_network = network_factory(
            obs_shape, wrapped.action_size, preprocess_observations_fn=normalize
        )
        make_inference_fn = ppo_networks.make_inference_fn(
            ppo_network, compute_value=True
        )
        return (make_inference_fn, params)
    except ImportError as e:
        log.warning("Could not build SAR-RL inference fn: %s", e)
        return None


def _make_play_inference_fn(mjx_env_name: str) -> Any:
    """Build Brax PPO inference function from env name (no closure over local objects).

    Used when loading play_phase_params.pkl so we avoid pickling the closure returned
    by brax.training.agents.ppo.train(), which is not pickleable.
    """
    import functools
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from myosuite.envs.myo.backends.mjx import make, ppo_config
    from mujoco_playground import wrapper

    env = make(mjx_env_name, config_overrides={"num_envs": 1})
    brax_env = wrapper.wrap_for_brax_training(
        env,
        episode_length=env._config.max_episode_steps,
        action_repeat=1,
    )
    rng = jax.random.PRNGKey(0)
    state = brax_env.reset(rng=rng[None, :])
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
    ppo_params = dict(ppo_config)
    if "network_factory" in ppo_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.pop("network_factory")
        )
    else:
        network_factory = ppo_networks.make_ppo_networks
    normalize = running_statistics.normalize
    ppo_network = network_factory(
        obs_shape, brax_env.action_size, preprocess_observations_fn=normalize
    )
    return ppo_networks.make_inference_fn(ppo_network, compute_value=True)


# ---------------------------------------------------------------------------
# Step 1 — Play phase: JAX PPO on MjxLegWalk-v0
# ---------------------------------------------------------------------------


def run_play_phase(
    mjx_env_name: str,
    seed: int,
    total_timesteps: int,
    num_envs: int,
    output_dir: Path,
    wandb_run: Any | None = None,
) -> tuple[Any, Any] | None:
    """Train JAX PPO on MJX walk env (play phase). Saves params to output_dir.

    Returns:
        (make_inference_fn, params) for use in activation rollout, or None if imports fail.
    """
    try:
        import functools
        import time
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo

        from myosuite.envs.myo.backends.mjx import make, ppo_config
        from mujoco_playground import wrapper
    except ImportError as e:
        log.warning(
            "Play phase skipped (mjx/brax/mujoco_playground not installed): %s", e
        )
        return None

    env = make(mjx_env_name, config_overrides={"num_envs": num_envs})
    ppo_params = dict(ppo_config)
    ppo_params["num_timesteps"] = total_timesteps

    if "network_factory" in ppo_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.pop("network_factory")
        )
    else:
        network_factory = ppo_networks.make_ppo_networks

    num_eval_envs = ppo_params.pop("num_eval_envs", JAX_PPO_NUM_EVAL_ENVS)
    times: list[float] = [time.monotonic()]

    def progress_fn(num_steps: int, metrics: dict[str, Any]) -> None:
        times.append(time.monotonic())
        if wandb_run is not None:
            wandb_run.log({"play/" + k: v for k, v in metrics.items()}, step=num_steps)
        if "eval/episode_reward" in metrics:
            log.info(
                "Play phase step %d: eval reward=%.3f",
                num_steps,
                metrics["eval/episode_reward"],
            )

    log.info(
        "Step 1 — Play phase: training JAX PPO on %s for %d steps",
        mjx_env_name,
        total_timesteps,
    )
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_envs=env._config.num_envs,
        episode_length=env._config.max_episode_steps,
        progress_fn=progress_fn,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        policy_params_fn=lambda *args: None,
        seed=seed,
        **ppo_params,
    )

    if len(times) > 1:
        log.info(
            "Play phase: JIT compile %.1fs, train %.1fs",
            times[1] - times[0],
            times[-1] - times[1],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "play_phase_params.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {"params": params, "env_name": mjx_env_name},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    log.info("Play phase params saved to %s", ckpt_path)
    return (make_inference_fn, params)


def _run_jax_ppo_training(
    mjx_env_name: str,
    seed: int,
    total_timesteps: int,
    num_envs: int,
    output_dir: Path,
    wandb_run: Any | None = None,
) -> tuple[Any, Any] | None:
    """Train JAX PPO on MJX env (play phase). For notebook/demo use.

    Saves params to output_dir / "jax_ppo_params.pkl" and returns (make_inference_fn, params).
    """
    result = run_play_phase(
        mjx_env_name, seed, total_timesteps, num_envs, output_dir, wandb_run
    )
    if result is None:
        return None
    make_inference_fn, params = result
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "jax_ppo_params.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {"params": params, "env_name": mjx_env_name},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    log.info("JAX PPO params saved to %s", ckpt_path)
    return (make_inference_fn, params)


def _squeeze_obs(obs: Any) -> Any:
    """Remove batch dim from obs (single env). Dict values or array."""
    if isinstance(obs, dict):
        return {
            k: v[0] if hasattr(v, "shape") and len(v.shape) > 0 else v
            for k, v in obs.items()
        }
    if hasattr(obs, "shape") and len(obs.shape) > 0:
        return obs[0]
    return obs


def _unsqueeze_action(action: Any) -> Any:
    """Add batch dim (1, ...) for env.step."""
    if hasattr(action, "shape"):
        if len(action.shape) == 0:
            return action[None]
        if len(action.shape) == 1:
            return action[None, :]
    return action


def run_jax_ppo_rollout(
    make_inference_fn: Any,
    params: Any,
    mjx_env_name: str,
    num_steps: int = 500,
    seed: int = 0,
) -> tuple[list[float], float]:
    """Run a single-env rollout with trained JAX PPO policy; return (rewards, total_reward).

    Uses unbatched obs/key so Brax policy's distribution.sample gets a scalar key (JAX compat).
    """
    try:
        import jax
        from myosuite.envs.myo.backends.mjx import make
        from mujoco_playground import wrapper
    except ImportError as e:
        raise RuntimeError("MJX stack required for rollout") from e

    env = make(mjx_env_name, config_overrides={"num_envs": 1})
    brax_env = wrapper.wrap_for_brax_training(
        env,
        episode_length=env._config.max_episode_steps,
        action_repeat=1,
    )
    policy_fn = make_inference_fn(params)
    rng = jax.random.PRNGKey(seed)
    rng, subkey = jax.random.split(rng)
    state = brax_env.reset(rng=subkey[None, :])
    rewards: list[float] = []
    for _ in range(num_steps):
        rng, subkey = jax.random.split(rng)
        obs_single = _squeeze_obs(state.obs)
        action, _ = policy_fn(obs_single, subkey)
        action_batch = _unsqueeze_action(action)
        state = brax_env.step(state, action_batch)
        rewards.append(float(state.reward[0]))
        if state.done[0]:
            rng, subkey = jax.random.split(rng)
            state = brax_env.reset(rng=subkey[None, :])
    total = sum(rewards)
    return (rewards, total)


def render_rollout_mjx(
    make_inference_fn: Any,
    params: Any,
    mjx_env_name: str,
    output_path: Path,
    num_steps: int = 500,
    seed: int = 0,
    height: int = 320,
    width: int = 480,
    ica: Any = None,
    pca: Any = None,
    scaler: Any = None,
) -> bool:
    """Run a rollout with the given policy and save an MP4 video to output_path.

    If ica, pca, scaler are provided, the env is wrapped with _SynergyWrapper and
    the policy is treated as SAR-RL (synergy-space). Otherwise the policy is the
    play-phase (muscle-space) policy.

    Returns:
        True if the video was written successfully, False otherwise.
    """
    try:
        import jax
        from myosuite.envs.myo.backends.mjx import make
        from myosuite.envs.myo.backends.mjx.utils import make_minimal_state
        from mujoco_playground import wrapper
    except ImportError as e:
        log.warning("Rendering skipped (mjx/utils not available): %s", e)
        return False

    env = make(mjx_env_name, config_overrides={"num_envs": 1})
    brax_env = wrapper.wrap_for_brax_training(
        env,
        episode_length=env._config.max_episode_steps,
        action_repeat=1,
    )
    if ica is not None and pca is not None and scaler is not None:
        brax_env = _SynergyWrapper(brax_env, ica, pca, scaler)

    policy_fn = make_inference_fn(params)
    rng = jax.random.PRNGKey(seed)
    rng, subkey = jax.random.split(rng)
    state = brax_env.reset(rng=subkey[None, :])
    rollout = [make_minimal_state(state)]
    for _ in range(num_steps - 1):
        rng, subkey = jax.random.split(rng)
        obs_single = _squeeze_obs(state.obs)
        action, _ = policy_fn(obs_single, subkey)
        action_batch = _unsqueeze_action(action)
        state = brax_env.step(state, action_batch)
        rollout.append(make_minimal_state(state))
        if state.done[0]:
            rng, subkey = jax.random.split(rng)
            state = brax_env.reset(rng=subkey[None, :])

    try:
        frames = brax_env.render(rollout, height=height, width=width)
    except Exception as e:
        log.warning("brax_env.render failed: %s", e)
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio

        imageio.mimwrite(
            str(output_path), frames, fps=50, format="mp4", codec="libx264"
        )
        log.info("Rendered rollout saved to %s (%d frames)", output_path, len(frames))
        return True
    except Exception as e:
        log.warning("Failed to write video %s: %s", output_path, e)
        return False


# ---------------------------------------------------------------------------
# Step 2 — Activation rollout: collect muscle activations from trained policy
# ---------------------------------------------------------------------------


def collect_activations_mjx(
    make_inference_fn: Any,
    params: Any,
    mjx_env_name: str,
    seed: int,
    episodes: int,
    percentile: int,
    output_dir: Path,
) -> np.ndarray:
    """Roll out trained JAX PPO on MJX, collect state.data.act from high-reward episodes.

    Matches run_sar_full.py get_activations() logic: preview 100 episodes for
    reward threshold, then collect activations from episodes above that percentile.

    Returns:
        Array of shape (T, n_muscles) with concatenated activations.
    """
    try:
        import jax
        from myosuite.envs.myo.backends.mjx import make
        from mujoco_playground import wrapper
    except ImportError as e:
        raise RuntimeError("MJX stack required for activation collection") from e

    env = make(mjx_env_name, config_overrides={"num_envs": 1})
    brax_env = wrapper.wrap_for_brax_training(
        env,
        episode_length=env._config.max_episode_steps,
        action_repeat=1,
    )
    policy_fn = make_inference_fn(params)

    # Preview episodes to compute reward threshold
    log.info("Activation rollout: preview 100 episodes for reward threshold")
    rng = jax.random.PRNGKey(seed)
    preview_rewards: list[float] = []
    for _ in range(100):
        rng, subkey = jax.random.split(rng)
        state = brax_env.reset(rng=subkey[None, :])
        ep_reward = 0.0
        for _ in range(env._config.max_episode_steps):
            rng, subkey = jax.random.split(rng)
            action, _ = policy_fn(state.obs, subkey)
            state = brax_env.step(state, action)
            ep_reward += float(state.reward[0])
            if state.done[0]:
                break
        preview_rewards.append(ep_reward)
    reward_threshold = float(np.percentile(preview_rewards, percentile))
    log.info("Reward threshold (p%d): %.3f", percentile, reward_threshold)

    # Collect activations from episodes above threshold
    solved_acts: list[np.ndarray] = []
    for ep in range(episodes):
        rng, subkey = jax.random.split(rng)
        state = brax_env.reset(rng=subkey[None, :])
        ep_reward = 0.0
        acts: list[np.ndarray] = []
        for _ in range(env._config.max_episode_steps):
            rng, subkey = jax.random.split(rng)
            action, _ = policy_fn(state.obs, subkey)
            state = brax_env.step(state, action)
            # state.data.act is (1, na) after step; squeeze and store
            act = np.asarray(state.data.act[0], dtype=np.float32)
            acts.append(act)
            ep_reward += float(state.reward[0])
            if state.done[0]:
                break
        if ep_reward > reward_threshold:
            solved_acts.extend(acts)
        if (ep + 1) % 100 == 0:
            log.info(
                "Activation rollout: %d/%d episodes, %d frames collected",
                ep + 1,
                episodes,
                len(solved_acts),
            )

    activations = (
        np.stack(solved_acts, axis=0)
        if solved_acts
        else np.zeros((0, env._mjx_model.nu), dtype=np.float32)
    )
    log.info(
        "Collected %d activation frames from high-reward episodes", len(solved_acts)
    )
    return activations


# ---------------------------------------------------------------------------
# Step 3 — SAR computation: VAF plot + PCA/ICA/MinMaxScaler
# ---------------------------------------------------------------------------


def find_synergies(acts: np.ndarray, save_path: Path) -> dict[int, float]:
    """Compute VAF for 1..n_muscles synergies and save plot. Matches run_sar_full.py find_synergies."""
    n_muscles = acts.shape[1]
    log.info("Computing VAF for 1..%d synergies", n_muscles)
    syn_dict: dict[int, float] = {}
    for i in range(n_muscles):
        pca = PCA(n_components=i + 1)
        pca.fit_transform(acts)
        syn_dict[i + 1] = round(float(np.sum(pca.explained_variance_ratio_)), 4)

    plt.figure()
    plt.plot(list(syn_dict.keys()), list(syn_dict.values()))
    plt.title("VAF by N synergies")
    plt.xlabel("# synergies")
    plt.ylabel("VAF")
    plt.grid()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("VAF plot saved to %s", save_path)
    return syn_dict


def compute_sar(
    acts: np.ndarray,
    n_syn: int,
    seed: int,
    output_dir: Path,
    prefix: str = "locomotion",
) -> tuple[FastICA, PCA, MinMaxScaler]:
    """Fit PCA + ICA + MinMaxScaler on activation data. Matches run_sar_full.py compute_SAR."""
    log.info("Fitting SAR with %d synergies", n_syn)
    pca = PCA(n_components=n_syn, random_state=seed)
    pca_act = pca.fit_transform(acts)

    ica = FastICA(n_components=n_syn, random_state=seed, max_iter=1000)
    pcaica_act = ica.fit_transform(pca_act)

    normalizer = MinMaxScaler(feature_range=(-1.0, 1.0))
    normalizer.fit(pcaica_act)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ica, output_dir / f"{prefix}_ica.pkl")
    joblib.dump(pca, output_dir / f"{prefix}_pca.pkl")
    joblib.dump(normalizer, output_dir / f"{prefix}_scaler.pkl")
    log.info("SAR objects saved with prefix '%s' in %s", prefix, output_dir)
    return ica, pca, normalizer


# ---------------------------------------------------------------------------
# Step 4 — SAR-RL: JAX PPO on synergy-wrapped MJX env
# ---------------------------------------------------------------------------


class _SynergyWrapper:
    """Wraps an MJX Brax env so action space is synergy space; converts to muscle in step()."""

    def __init__(self, env: Any, ica: FastICA, pca: PCA, scaler: MinMaxScaler) -> None:
        self._env = env
        self._ica = ica
        self._pca = pca
        self._scaler = scaler
        self._n_syn = pca.components_.shape[0]
        # Unwrap to get raw MJX env and read nu (n_muscles)
        e = env
        while e is not None and not hasattr(e, "_mjx_model"):
            e = getattr(e, "env", getattr(e, "_env", None))
        mj = getattr(e, "_mjx_model", None) if e is not None else None
        self._n_muscles = getattr(mj, "nu", None) if mj is not None else None
        if self._n_muscles is None:
            raise ValueError(
                "Could not infer n_muscles from env (no _mjx_model.nu found)"
            )

    @property
    def action_size(self) -> int:
        return self._n_syn

    @property
    def observation_size(self) -> int:
        """Delegate to inner env for policy network obs dimension."""
        return getattr(self._env, "observation_size", None) or getattr(
            self._env, "obs_size", 0
        )

    def reset(self, rng):
        return self._env.reset(rng)

    def step(self, state, action):
        # action: (batch, n_syn) — convert to muscle (batch, n_muscles)
        action_np = np.asarray(action)
        if action_np.ndim == 1:
            action_np = action_np.reshape(1, -1)
        muscle = synergy_to_muscle_batch(action_np, self._ica, self._pca, self._scaler)
        return self._env.step(state, muscle)


def run_sar_rl(
    mjx_env_name: str,
    ica: FastICA,
    pca: PCA,
    scaler: MinMaxScaler,
    seed: int,
    total_timesteps: int,
    num_envs: int,
    output_dir: Path,
    wandb_run: Any | None = None,
) -> tuple[Any, Any] | None:
    """Train JAX PPO on MJX env with synergy action wrapper (SAR-RL phase)."""
    try:
        import functools
        import time
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo

        from myosuite.envs.myo.backends.mjx import make, ppo_config
        from mujoco_playground import wrapper
    except ImportError as e:
        log.warning("SAR-RL skipped (mjx/brax not installed): %s", e)
        return None

    env = make(mjx_env_name, config_overrides={"num_envs": num_envs})
    brax_env = wrapper.wrap_for_brax_training(
        env,
        episode_length=env._config.max_episode_steps,
        action_repeat=1,
    )
    wrapped = _SynergyWrapper(brax_env, ica, pca, scaler)

    # PPO expects env with action_size; use wrapped
    ppo_params = dict(ppo_config)
    ppo_params["num_timesteps"] = total_timesteps
    if "network_factory" in ppo_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.pop("network_factory")
        )
    else:
        network_factory = ppo_networks.make_ppo_networks
    num_eval_envs = ppo_params.pop("num_eval_envs", JAX_PPO_NUM_EVAL_ENVS)
    times: list[float] = [time.monotonic()]

    def progress_fn(num_steps: int, metrics: dict[str, Any]) -> None:
        times.append(time.monotonic())
        if wandb_run is not None:
            wandb_run.log(
                {"sar_rl/" + k: v for k, v in metrics.items()}, step=num_steps
            )
        if "eval/episode_reward" in metrics:
            log.info(
                "SAR-RL step %d: eval reward=%.3f",
                num_steps,
                metrics["eval/episode_reward"],
            )

    # Brax PPO train() expects env to have attribute used for action size; our wrapper has action_size
    log.info(
        "Step 4 — SAR-RL: training JAX PPO on synergy-wrapped %s for %d steps",
        mjx_env_name,
        total_timesteps,
    )
    make_inference_fn, params, _ = ppo.train(
        environment=wrapped,
        num_envs=num_envs,
        episode_length=env._config.max_episode_steps,
        progress_fn=progress_fn,
        network_factory=network_factory,
        wrap_env_fn=lambda e: e,  # already wrapped
        num_eval_envs=num_eval_envs,
        policy_params_fn=lambda *args: None,
        seed=seed,
        **ppo_params,
    )

    if len(times) > 1:
        log.info(
            "SAR-RL: JIT compile %.1fs, train %.1fs",
            times[1] - times[0],
            times[-1] - times[1],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "sar_rl_params.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {"params": params, "env_name": mjx_env_name},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    log.info("SAR-RL params saved to %s", ckpt_path)
    return (make_inference_fn, params)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_render_only(output_dir: Path) -> None:
    """Load checkpoint(s) from output_dir and render video(s) only (no training).

    Renders play_rollout.mp4 from play_phase_params.pkl if present.
    Renders sar_rl_rollout.mp4 from sar_rl_params.pkl + locomotion_*.pkl if all present.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        log.error("Output dir does not exist: %s", output_dir)
        return
    rendered = []
    # Play policy
    play_ckpt = output_dir / "play_phase_params.pkl"
    if play_ckpt.exists():
        loaded = _load_play_checkpoint(play_ckpt, MJX_ENV)
        if loaded is not None:
            make_fn, params = loaded
            out_path = output_dir / "play_rollout.mp4"
            if render_rollout_mjx(
                make_fn,
                params,
                MJX_ENV,
                out_path,
                num_steps=RENDER_STEPS,
                seed=SEED,
                height=RENDER_VIDEO_HEIGHT,
                width=RENDER_VIDEO_WIDTH,
            ):
                rendered.append(str(out_path))
        else:
            log.warning("Could not load play checkpoint %s", play_ckpt)
    else:
        log.info("No play_phase_params.pkl in %s; skipping play rollout", output_dir)
    # SAR-RL policy (requires SAR artifacts)
    sar_ckpt = output_dir / "sar_rl_params.pkl"
    ica_path = output_dir / "locomotion_ica.pkl"
    pca_path = output_dir / "locomotion_pca.pkl"
    scaler_path = output_dir / "locomotion_scaler.pkl"
    if (
        sar_ckpt.exists()
        and ica_path.exists()
        and pca_path.exists()
        and scaler_path.exists()
    ):
        ica = joblib.load(ica_path)
        pca = joblib.load(pca_path)
        scaler = joblib.load(scaler_path)
        loaded = _load_sar_rl_checkpoint(sar_ckpt, MJX_ENV, ica, pca, scaler)
        if loaded is not None:
            make_fn, params = loaded
            out_path = output_dir / "sar_rl_rollout.mp4"
            if render_rollout_mjx(
                make_fn,
                params,
                MJX_ENV,
                out_path,
                num_steps=RENDER_STEPS,
                seed=SEED,
                height=RENDER_VIDEO_HEIGHT,
                width=RENDER_VIDEO_WIDTH,
                ica=ica,
                pca=pca,
                scaler=scaler,
            ):
                rendered.append(str(out_path))
        else:
            log.warning("Could not load SAR-RL checkpoint %s", sar_ckpt)
    else:
        log.info("SAR-RL checkpoint or SAR artifacts missing; skipping SAR-RL rollout")
    if rendered:
        log.info("Render-only complete. Videos: %s", rendered)
    else:
        log.warning(
            "No videos produced. Ensure play_phase_params.pkl (and optionally SAR-RL files) exist in %s",
            output_dir,
        )


def main() -> None:
    """Run the full SAR pipeline on MJX (play → activation → SAR → SAR-RL)."""
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if RENDER_ONLY:
        run_render_only(output_dir)
        return

    wandb_run = None
    if RUN_WANDB:
        try:
            import wandb

            wandb_run = wandb.init(
                project="myosuite-sar-mjx",
                name=f"SAR-{MJX_ENV}-n{N_SYNERGIES}",
                config={
                    "mjx_env": MJX_ENV,
                    "play_steps": PLAY_STEPS,
                    "sar_rl_steps": SAR_RL_STEPS,
                    "activation_episodes": ACTIVATION_EPISODES,
                    "activation_percentile": ACTIVATION_PERCENTILE,
                    "n_synergies": N_SYNERGIES,
                    "seed": SEED,
                },
            )
        except ImportError:
            pass

    make_inference_fn, play_params = None, None
    muscle_activations_path = output_dir / "muscle_activations.npy"

    # --- Step 1: Play phase ---
    if RUN_PLAY_PHASE:
        play_ckpt = output_dir / "play_phase_params.pkl"
        if play_ckpt.exists():
            log.info("Loading existing play phase params from %s", play_ckpt)
            loaded = _load_play_checkpoint(play_ckpt, MJX_ENV)
            if loaded is not None:
                make_inference_fn, play_params = loaded
            else:
                result = run_play_phase(
                    MJX_ENV, SEED, PLAY_STEPS, JAX_PPO_NUM_ENVS, output_dir, wandb_run
                )
                if result is not None:
                    make_inference_fn, play_params = result
        else:
            result = run_play_phase(
                MJX_ENV, SEED, PLAY_STEPS, JAX_PPO_NUM_ENVS, output_dir, wandb_run
            )
            if result is not None:
                make_inference_fn, play_params = result

        # --- Step 2: Activation rollout ---
        if make_inference_fn is not None and play_params is not None:
            if muscle_activations_path.exists():
                log.info(
                    "Loading existing activations from %s", muscle_activations_path
                )
                muscle_data = np.load(muscle_activations_path)
            else:
                muscle_data = collect_activations_mjx(
                    make_inference_fn,
                    play_params,
                    MJX_ENV,
                    SEED,
                    ACTIVATION_EPISODES,
                    ACTIVATION_PERCENTILE,
                    output_dir,
                )
                np.save(muscle_activations_path, muscle_data)
            log.info("Muscle data shape: %s", muscle_data.shape)

            # Optional: render play policy rollout to video
            if (
                RENDER_ROLLOUT
                and make_inference_fn is not None
                and play_params is not None
            ):
                render_rollout_mjx(
                    make_inference_fn,
                    play_params,
                    MJX_ENV,
                    output_dir / "play_rollout.mp4",
                    num_steps=RENDER_STEPS,
                    seed=SEED,
                    height=RENDER_VIDEO_HEIGHT,
                    width=RENDER_VIDEO_WIDTH,
                )

            # --- Step 3: SAR ---
            vaf_path = output_dir / "vaf_plot.png"
            syn_dict = find_synergies(muscle_data, vaf_path)
            vaf_at_n = syn_dict.get(N_SYNERGIES, float("nan"))
            log.info("VAF at %d synergies: %.4f", N_SYNERGIES, vaf_at_n)

            ica, pca, normalizer = compute_sar(
                muscle_data, N_SYNERGIES, SEED, output_dir, prefix="locomotion"
            )
        else:
            # Play phase did not produce a policy; require existing SAR artifacts
            ica_path = output_dir / "locomotion_ica.pkl"
            pca_path = output_dir / "locomotion_pca.pkl"
            scaler_path = output_dir / "locomotion_scaler.pkl"
            missing = [p for p in (ica_path, pca_path, scaler_path) if not p.exists()]
            if missing:
                log.error(
                    "Play phase produced no policy and SAR artifacts missing in %s: %s. "
                    "Ensure play_phase_params.pkl exists or run the full pipeline.",
                    output_dir,
                    [str(p.name) for p in missing],
                )
                raise FileNotFoundError(
                    f"Missing SAR files: {[p.name for p in missing]}. "
                    "Need play_phase_params.pkl (and run activation+SAR) or locomotion_*.pkl."
                )
            ica = joblib.load(ica_path)
            pca = joblib.load(pca_path)
            normalizer = joblib.load(scaler_path)
            log.info("Loaded SAR from %s", output_dir)
    else:
        # RUN_PLAY_PHASE=false: require existing SAR artifacts
        ica_path = output_dir / "locomotion_ica.pkl"
        pca_path = output_dir / "locomotion_pca.pkl"
        scaler_path = output_dir / "locomotion_scaler.pkl"
        missing = [p for p in (ica_path, pca_path, scaler_path) if not p.exists()]
        if missing:
            log.error(
                "RUN_PLAY_PHASE=false but SAR artifacts missing in %s: %s. "
                "Run the full pipeline once (RUN_PLAY_PHASE=true) to generate them.",
                output_dir,
                [str(p.name) for p in missing],
            )
            raise FileNotFoundError(
                f"Missing SAR files: {[p.name for p in missing]}. "
                "Run without RUN_PLAY_PHASE=false first to create locomotion_*.pkl."
            )
        ica = joblib.load(ica_path)
        pca = joblib.load(pca_path)
        normalizer = joblib.load(scaler_path)
        log.info("Skipping play phase; loaded SAR from %s", output_dir)

    # --- Step 4: SAR-RL ---
    sar_rl_result = None
    if not SKIP_SAR_RL:
        sar_rl_result = run_sar_rl(
            MJX_ENV,
            ica,
            pca,
            normalizer,
            SEED,
            SAR_RL_STEPS,
            JAX_PPO_NUM_ENVS,
            output_dir,
            wandb_run,
        )
        if sar_rl_result is not None and RENDER_ROLLOUT:
            sar_make_fn, sar_params = sar_rl_result
            render_rollout_mjx(
                sar_make_fn,
                sar_params,
                MJX_ENV,
                output_dir / "sar_rl_rollout.mp4",
                num_steps=RENDER_STEPS,
                seed=SEED,
                height=RENDER_VIDEO_HEIGHT,
                width=RENDER_VIDEO_WIDTH,
                ica=ica,
                pca=pca,
                scaler=normalizer,
            )
    else:
        log.info("Skipping SAR-RL (SKIP_SAR_RL=true)")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
    log.info("Full SAR pipeline complete. Outputs in %s", output_dir)
    if RENDER_ROLLOUT:
        log.info("Videos: play_rollout.mp4, sar_rl_rollout.mp4 (if SAR-RL ran)")


def _main_with_cuda_fallback() -> None:
    """Run main(); on CUDA/GPU failure (OOM, cuSolver, no device), re-exec with JAX_PLATFORMS=cpu."""
    try:
        main()
    except (RuntimeError, Exception) as e:
        err_msg = str(e)
        err_lower = err_msg.lower()
        is_gpu_error = (
            "no supported devices" in err_lower
            or "platform cuda" in err_lower
            or "cuda" in err_lower
            or "gpusolver" in err_lower
            or "cusolver" in err_lower
            or "cudnn" in err_lower
        )
        if is_gpu_error and os.environ.get("JAX_PLATFORMS") != "cpu":
            log.warning(
                "GPU/CUDA backend failed: %s. Re-running with JAX_PLATFORMS=cpu.",
                err_msg[:250],
            )
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
        raise


if __name__ == "__main__":
    _main_with_cuda_fallback()
