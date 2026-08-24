# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Standalone script that runs the full SAR locomotion training pipeline.

Pipeline:
  1. Play phase: train on myoLegWalk-v0 for 1.5M steps (seed='0')
  2. Activation rollout: collect 1000 episodes of muscle activations
  3. SAR computation: fit PCA/ICA/scaler with 20 synergies, plot VAF
  4. SAR-RL: train on myoLegHillyTerrainWalk-v0 for 2.5M steps (seed='0')

All outputs (models, normalizers, SAR pkl files, plots) are written to the
directory from which this script is launched.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Force headless rendering backend for mp4 generation in non-GUI sessions.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend for headless execution
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import FastICA, PCA  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Ensure the repo root is on sys.path so ``import myosuite`` works when this
# script is executed from any directory.
_REPO_ROOT = Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import myosuite  # noqa: E402
from myosuite.utils import gym  # noqa: E402
from sar_tutorial_utils import (  # noqa: E402
    SaveSuccesses,
    SynergyWrapper,
    get_vid,
    linear_schedule,
)

myosuite.register_all_envs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
PLAY_ENV = "myoLegWalk-v0"
TARGET_ENV = "myoLegHillyTerrainWalk-v0"
PLAY_STEPS = int(1.5e6)
SAR_RL_STEPS = int(2.5e6)
SEED = "0"
N_SYNERGIES = 20
ACTIVATION_EPISODES = 1000
ACTIVATION_PERCENTILE = 80

# Video rendering (mirrors `SAR_tutorial.ipynb`).
RENDER_VIDEOS = True
PLAY_VIDEO_NAME = "walk_play_period_video"
PLAY_VIDEO_EPISODES = 10
SAR_RL_VIDEO_NAME = "walk_SAR-RL_video"
SAR_RL_VIDEO_EPISODES = 5
RESUME_IF_AVAILABLE = True


# ---------------------------------------------------------------------------
# Step 1 — Play phase training
# ---------------------------------------------------------------------------


def train(env_name: str, policy_name: str, timesteps: int, seed: str) -> None:
    """Train a SAC policy on *env_name* and save model + normalizer.

    Args:
        env_name: Gymnasium environment id.
        policy_name: Prefix used for saved file names.
        timesteps: Total environment steps to train for.
        seed: String seed appended to output file names.
    """
    log.info("Starting play-phase training on %s for %d steps", env_name, timesteps)
    env = gym.make(env_name)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    net_shape = [400, 300]
    policy_kwargs = dict(net_arch=dict(pi=net_shape, qf=net_shape))

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(0.001),
        buffer_size=int(3e5),
        learning_starts=1000,
        batch_size=256,
        tau=0.02,
        gamma=0.98,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    succ_callback = SaveSuccesses(
        check_freq=1,
        env_name=f"{env_name}_{seed}",
        log_dir=f"{policy_name}_successes_{env_name}_{seed}",
    )
    model.set_logger(configure(f"{policy_name}_results_{env_name}_{seed}"))
    model.learn(total_timesteps=timesteps, callback=succ_callback, log_interval=4)
    model.save(f"{policy_name}_model_{env_name}_{seed}")
    env.save(f"{policy_name}_env_{env_name}_{seed}")
    log.info("Play-phase training complete. Model saved.")


# ---------------------------------------------------------------------------
# Step 2 — Muscle activation rollout
# ---------------------------------------------------------------------------


def get_activations(
    name: str,
    env_name: str,
    seed: str,
    episodes: int = 1000,
    percentile: int = 80,
) -> np.ndarray:
    """Roll out a trained policy and collect high-reward muscle activations.

    Args:
        name: Policy prefix (same as used in :func:`train`).
        env_name: Gymnasium environment id.
        seed: String seed matching the saved policy.
        episodes: Number of rollout episodes.
        percentile: Reward percentile threshold; only episodes above this
            threshold contribute activations.

    Returns:
        Array of shape ``(T, n_muscles)`` with concatenated activations.
    """
    log.info("Collecting muscle activations from %s (%d episodes)", env_name, episodes)
    with gym.make(env_name) as env:
        env.reset()
        model = SAC.load(f"{name}_model_{env_name}_{seed}")
        vec = VecNormalize.load(
            f"{name}_env_{env_name}_{seed}", DummyVecEnv([lambda: env])
        )

        # Estimate reward threshold from 100 preview episodes.
        preview_rewards: list[float] = []
        for _ in range(100):
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                obs = reset_out[0]
            else:
                obs = reset_out
            rewards = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                o = vec.normalize_obs(obs)
                a, _ = model.predict(o, deterministic=False)
                step_out = env.step(a)
                if len(step_out) == 5:
                    obs, r, terminated, truncated, _ = step_out
                else:
                    obs, r, done, *_ = step_out
                    terminated = bool(done)
                    truncated = False
                rewards += r
            preview_rewards.append(rewards)
        reward_threshold = float(np.percentile(preview_rewards, percentile))
        log.info("Reward threshold (p%d): %.3f", percentile, reward_threshold)

        solved_acts: list[np.ndarray] = []
        for _ in tqdm(range(episodes), desc="activation rollouts"):
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                obs = reset_out[0]
            else:
                obs = reset_out
            rewards, acts = 0.0, []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                o = vec.normalize_obs(obs)
                a, _ = model.predict(o, deterministic=False)
                step_out = env.step(a)
                if len(step_out) == 5:
                    obs, r, terminated, truncated, _ = step_out
                else:
                    obs, r, done, *_ = step_out
                    terminated = bool(done)
                    truncated = False
                acts.append(env.unwrapped.data.act.copy())
                rewards += r
            if rewards > reward_threshold:
                solved_acts.extend(acts)

    activations = np.array(solved_acts)
    log.info(
        "Collected %d activation frames from high-reward episodes", len(activations)
    )
    return activations


# ---------------------------------------------------------------------------
# Step 3 — SAR computation
# ---------------------------------------------------------------------------


def find_synergies(
    acts: np.ndarray, save_path: str = "vaf_plot.png"
) -> dict[int, float]:
    """Compute VAF for 1…N synergies and save a plot.

    Args:
        acts: Activation array of shape ``(T, n_muscles)``.
        save_path: File path for the VAF figure.

    Returns:
        Mapping from number of synergies to VAF value.
    """
    log.info("Computing VAF for 1..%d synergies", acts.shape[1])
    syn_dict: dict[int, float] = {}
    for i in range(acts.shape[1]):
        pca = PCA(n_components=i + 1)
        pca.fit_transform(acts)
        syn_dict[i + 1] = round(float(sum(pca.explained_variance_ratio_)), 4)

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


def compute_SAR(
    acts: np.ndarray,
    n_syn: int,
    save: bool = True,
    prefix: str = "locomotion",
) -> tuple[FastICA, PCA, MinMaxScaler]:
    """Fit PCA + ICA + normalizer on activation data.

    Args:
        acts: Activation array of shape ``(T, n_muscles)``.
        n_syn: Number of synergies (PCA components).
        save: Whether to serialise the fitted objects via joblib.
        prefix: File name prefix for saved pkl files.

    Returns:
        Tuple of (ica, pca, normalizer).
    """
    log.info("Fitting SAR with %d synergies", n_syn)
    pca = PCA(n_components=n_syn)
    pca_act = pca.fit_transform(acts)

    ica = FastICA()
    pcaica_act = ica.fit_transform(pca_act)

    normalizer = MinMaxScaler((-1, 1))
    normalizer.fit(pcaica_act)

    if save:
        joblib.dump(ica, f"{prefix}_ica.pkl")
        joblib.dump(pca, f"{prefix}_pca.pkl")
        joblib.dump(normalizer, f"{prefix}_scaler.pkl")
        log.info("SAR objects saved with prefix '%s'", prefix)

    return ica, pca, normalizer


# ---------------------------------------------------------------------------
# Step 4 — SAR-RL
# ---------------------------------------------------------------------------


def sar_rl(
    env_name: str,
    policy_name: str,
    timesteps: int,
    seed: str,
    ica: FastICA,
    pca: PCA,
    normalizer: MinMaxScaler,
) -> None:
    """Train SAC with a pure-synergy action wrapper on *env_name*.

    Uses :class:`SynergyWrapper` (phi=1 — fully synergistic actions).

    Args:
        env_name: Gymnasium environment id (target task).
        policy_name: Prefix for saved file names.
        timesteps: Total environment steps.
        seed: String seed.
        ica: Fitted ICA transform.
        pca: Fitted PCA transform.
        normalizer: Fitted MinMaxScaler.
    """
    log.info("Starting SAR-RL on %s for %d steps", env_name, timesteps)
    env = SynergyWrapper(gym.make(env_name), ica, pca, normalizer)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    net_shape = [400, 300]
    policy_kwargs = dict(net_arch=dict(pi=net_shape, qf=net_shape))

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(0.001),
        buffer_size=int(3e5),
        learning_starts=5000,
        batch_size=256,
        tau=0.02,
        gamma=0.98,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    succ_callback = SaveSuccesses(
        check_freq=1,
        env_name=f"{env_name}_{seed}",
        log_dir=f"{policy_name}_successes_{env_name}_{seed}",
    )
    model.set_logger(configure(f"{policy_name}_results_{env_name}_{seed}"))
    model.learn(total_timesteps=timesteps, callback=succ_callback, log_interval=4)
    model.save(f"{policy_name}_model_{env_name}_{seed}")
    env.save(f"{policy_name}_env_{env_name}_{seed}")
    log.info("SAR-RL training complete. Model saved.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full SAR locomotion pipeline end-to-end."""
    os.makedirs("sar_outputs", exist_ok=True)
    os.chdir("sar_outputs")
    log.info("Working directory: %s", os.getcwd())

    # --- Step 1: play phase ---
    play_model_path = Path(f"play_period_model_{PLAY_ENV}_{SEED}.zip")
    play_env_path = Path(f"play_period_env_{PLAY_ENV}_{SEED}")
    have_play_artifacts = play_model_path.exists() and play_env_path.exists()
    if RESUME_IF_AVAILABLE and have_play_artifacts:
        log.info(
            "Skipping play-phase training; found existing artifacts: %s and %s",
            play_model_path,
            play_env_path,
        )
    else:
        train(PLAY_ENV, "play_period", PLAY_STEPS, SEED)
    if RENDER_VIDEOS:
        get_vid(
            name="play_period",
            env_name=PLAY_ENV,
            seed=SEED,
            episodes=PLAY_VIDEO_EPISODES,
            video_name=PLAY_VIDEO_NAME,
        )

    # --- Step 2: activation rollout ---
    activations_path = Path("muscle_activations.npy")
    if RESUME_IF_AVAILABLE and activations_path.exists():
        muscle_data = np.load(activations_path)
        log.info("Skipping activation rollout; loaded existing %s", activations_path)
    else:
        muscle_data = get_activations(
            "play_period", PLAY_ENV, SEED, ACTIVATION_EPISODES, ACTIVATION_PERCENTILE
        )
        np.save(activations_path, muscle_data)
    log.info("Muscle data shape: %s", muscle_data.shape)

    # --- Step 3: SAR ---
    syn_dict = find_synergies(muscle_data, save_path="vaf_plot.png")
    vaf_at_n = syn_dict.get(N_SYNERGIES, float("nan"))
    log.info("VAF at %d synergies: %.4f", N_SYNERGIES, vaf_at_n)

    sar_ica_path = Path("locomotion_ica.pkl")
    sar_pca_path = Path("locomotion_pca.pkl")
    sar_scaler_path = Path("locomotion_scaler.pkl")
    have_sar_artifacts = (
        sar_ica_path.exists() and sar_pca_path.exists() and sar_scaler_path.exists()
    )
    if RESUME_IF_AVAILABLE and have_sar_artifacts:
        log.info("Skipping SAR fit; loading existing locomotion PCA/ICA/scaler")
        ica = joblib.load(sar_ica_path)
        pca = joblib.load(sar_pca_path)
        normalizer = joblib.load(sar_scaler_path)
    else:
        ica, pca, normalizer = compute_SAR(
            muscle_data, N_SYNERGIES, save=True, prefix="locomotion"
        )

    # --- Step 4: SAR-RL on target terrain task ---
    sar_model_path = Path(f"SAR-RL_model_{TARGET_ENV}_{SEED}.zip")
    sar_env_path = Path(f"SAR-RL_env_{TARGET_ENV}_{SEED}")
    have_sar_rl_artifacts = sar_model_path.exists() and sar_env_path.exists()
    if RESUME_IF_AVAILABLE and have_sar_rl_artifacts:
        log.info(
            "Skipping SAR-RL training; found existing artifacts: %s and %s",
            sar_model_path,
            sar_env_path,
        )
    else:
        sar_rl(TARGET_ENV, "SAR-RL", SAR_RL_STEPS, SEED, ica, pca, normalizer)
    if RENDER_VIDEOS:
        get_vid(
            name="SAR-RL",
            env_name=TARGET_ENV,
            seed=SEED,
            episodes=SAR_RL_VIDEO_EPISODES,
            video_name=SAR_RL_VIDEO_NAME,
            determ=False,
            pca=pca,
            ica=ica,
            normalizer=normalizer,
            is_sar=True,
            syn_nosyn=False,
        )

    log.info("Full SAR pipeline complete.")


if __name__ == "__main__":
    main()
