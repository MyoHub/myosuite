# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Standalone script mirroring run_sar_full.py, but for the manipulation
pipeline from SAR_tutorial.ipynb's "Full training example 2" section
(cells 53-69), including an RL-E2E baseline for the SAR-RL vs RL-E2E
comparison the paper reports.

Pipeline (hyperparameters taken verbatim from the notebook cells):
  1. Play phase: train on myoHandReorient8-v0 for 1M steps (seed='0')
  2. Activation rollout: 10 episodes (notebook cell 54's explicit episodes=10,
     much fewer than locomotion's 1000 -- reorientation episodes are longer
     and each yields many more activation frames per episode)
  3. SAR computation: fit PCA/ICA/scaler with 20 synergies
  4. SAR-RL: train on myoHandReorient100-v0 for 1.5M steps using
     SynNoSynWrapper (phi=.66 blend of synergy + task-specific actions,
     notebook cell 61's default) -- NOT the pure SynergyWrapper locomotion
     uses, per SAR_RL()'s syn_nosyn=True default.
  5. RL-E2E baseline: train directly on myoHandReorient100-v0 for 2.5M steps
     (notebook cell 65), for the SAR-RL vs RL-E2E comparison.

All outputs are written to sar_outputs/ under the launch directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "egl"
    if os.environ["MUJOCO_GL"] == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from sklearn.decomposition import FastICA, PCA  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

_REPO_ROOT = Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import myosuite  # noqa: E402
from myosuite.utils import gym  # noqa: E402
from sar_tutorial_utils import SaveSuccesses, SynNoSynWrapper, linear_schedule  # noqa: E402

# Reuse run_sar_full.py's play-phase trainer and activation/SAR helpers
# verbatim -- identical hyperparameters, only the call-site args differ.
from run_sar_full import (  # noqa: E402
    compute_SAR,
    find_synergies,
    get_activations,
    train,
)

myosuite.register_all_envs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline configuration (verbatim from SAR_tutorial.ipynb cells 53-65)
# ---------------------------------------------------------------------------
PLAY_ENV = "myoHandReorient8-v0"
TARGET_ENV = "myoHandReorient100-v0"
PLAY_STEPS = int(1e6)
SAR_RL_STEPS = int(1.5e6)
E2E_STEPS = int(2.5e6)
SEED = "0"
N_SYNERGIES = 20
# Notebook's literal value (10) yields too few post-percentile-filter frames
# for a 20-component PCA fit on this env (verified: 13 samples, n_components
# must be <= n_samples) -- raised to guarantee enough activation frames.
ACTIVATION_EPISODES = 500
ACTIVATION_PERCENTILE = 80
PHI = 0.66
RESUME_IF_AVAILABLE = True


def sar_rl_synnosyn(
    env_name: str,
    policy_name: str,
    timesteps: int,
    seed: str,
    ica: FastICA,
    pca: PCA,
    normalizer: MinMaxScaler,
    phi: float,
) -> None:
    """Train SAC with SynNoSynWrapper (blended synergy + task-specific actions).

    Mirrors run_sar_full.py's ``sar_rl`` but uses SynNoSynWrapper instead of
    the pure SynergyWrapper, matching the manipulation notebook's SAR_RL()
    call (syn_nosyn=True default).
    """
    log.info(
        "Starting SAR-RL (SynNoSyn, phi=%.2f) on %s for %d steps",
        phi,
        env_name,
        timesteps,
    )
    env = SynNoSynWrapper(gym.make(env_name), ica, pca, normalizer, phi)
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


def main() -> None:
    os.makedirs("sar_outputs", exist_ok=True)
    os.chdir("sar_outputs")
    log.info("Working directory: %s", os.getcwd())

    # --- Step 1: play phase ---
    play_model_path = Path(f"play_period_model_{PLAY_ENV}_{SEED}.zip")
    play_env_path = Path(f"play_period_env_{PLAY_ENV}_{SEED}")
    if RESUME_IF_AVAILABLE and play_model_path.exists() and play_env_path.exists():
        log.info("Skipping play-phase training; found existing artifacts")
    else:
        train(PLAY_ENV, "play_period", PLAY_STEPS, SEED)

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
    syn_dict = find_synergies(muscle_data, save_path="manipulation_vaf_plot.png")
    vaf_at_n = syn_dict.get(N_SYNERGIES, float("nan"))
    log.info("VAF at %d synergies: %.4f", N_SYNERGIES, vaf_at_n)

    sar_ica_path = Path("manipulation_ica.pkl")
    sar_pca_path = Path("manipulation_pca.pkl")
    sar_scaler_path = Path("manipulation_scaler.pkl")
    if (
        RESUME_IF_AVAILABLE
        and sar_ica_path.exists()
        and sar_pca_path.exists()
        and sar_scaler_path.exists()
    ):
        log.info("Skipping SAR fit; loading existing manipulation PCA/ICA/scaler")
        ica = joblib.load(sar_ica_path)
        pca = joblib.load(sar_pca_path)
        normalizer = joblib.load(sar_scaler_path)
    else:
        ica, pca, normalizer = compute_SAR(
            muscle_data, N_SYNERGIES, save=True, prefix="manipulation"
        )

    # --- Step 4: SAR-RL on Reorient100 ---
    sar_model_path = Path(f"SAR-RL_model_{TARGET_ENV}_{SEED}.zip")
    sar_env_path = Path(f"SAR-RL_env_{TARGET_ENV}_{SEED}")
    if RESUME_IF_AVAILABLE and sar_model_path.exists() and sar_env_path.exists():
        log.info("Skipping SAR-RL training; found existing artifacts")
    else:
        sar_rl_synnosyn(TARGET_ENV, "SAR-RL", SAR_RL_STEPS, SEED, ica, pca, normalizer, PHI)

    # --- Step 5: RL-E2E baseline on Reorient100 ---
    e2e_model_path = Path(f"RL-E2E_model_{TARGET_ENV}_{SEED}.zip")
    e2e_env_path = Path(f"RL-E2E_env_{TARGET_ENV}_{SEED}")
    if RESUME_IF_AVAILABLE and e2e_model_path.exists() and e2e_env_path.exists():
        log.info("Skipping RL-E2E baseline; found existing artifacts")
    else:
        train(TARGET_ENV, "RL-E2E", E2E_STEPS, SEED)

    log.info("Full SAR manipulation pipeline complete.")


def _dry_run() -> None:
    env = gym.make(PLAY_ENV)
    obs, info = env.reset(seed=0)
    action_shape = env.action_space.shape
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
    log.info(
        "Dry-run ok: %s obs_dim=%s action_dim=%s",
        PLAY_ENV,
        getattr(obs, "shape", type(obs)),
        action_shape,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Full SAR manipulation pipeline (1M + 1.5M + 2.5M SB3 steps). "
            "This takes hours. Use --dry-run to only construct the env."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        _dry_run()
    else:
        main()
