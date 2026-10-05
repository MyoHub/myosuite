"""Train a Stable-Baselines3 policy on one MyoSuite env on the CPU.

The CPU counterpart of ``scripts/train_mjlab.py``::

    python scripts/train_sb3.py myoElbowPose1D6MRandom-v0 --timesteps 500000
    python scripts/train_sb3.py myoLegWalk-v0 --algo sac --n-envs 4 --tensorboard logs/tb

Saves ``<out>.zip`` (default ``logs/sb3/<env_id>/model``) and, with ``--normalize``,
``<out>_vecnormalize.pkl``. Afterwards it scores the deterministic policy: mean return
and the share of episodes that are solved on their final step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import gymnasium as gym
import myosuite  # noqa: F401  (registers the envs)

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("env_id", nargs="?", default="myoElbowPose1D6MRandom-v0")
    p.add_argument("--algo", choices=sorted(ALGOS), default="ppo")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1, help="Parallel CPU envs.")
    p.add_argument("--normalize", action="store_true", help="Normalize observations.")
    p.add_argument("--out", type=Path, default=None, help="Output path without .zip.")
    p.add_argument(
        "--tensorboard", type=Path, default=None, help="TensorBoard log dir."
    )
    p.add_argument(
        "--eval-episodes", type=int, default=20, help="0 skips the evaluation."
    )
    return p.parse_args(argv)


def _make_vec_env(env_id: str, n_envs: int, seed: int) -> DummyVecEnv:
    def factory(rank: int):
        def _init() -> gym.Env:
            env = Monitor(gym.make(env_id))
            env.reset(seed=seed + rank)
            return env

        return _init

    return DummyVecEnv([factory(i) for i in range(n_envs)])


def evaluate(
    model, env_id: str, episodes: int, seed: int, vecnorm=None
) -> tuple[float, float]:
    """Mean return and share of episodes solved on the last step (deterministic policy)."""
    env = gym.make(env_id)
    returns, solved = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        total, info, done = 0.0, {}, False
        while not done:
            net_obs = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            action, _ = model.predict(net_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
        solved.append(float(bool(info.get("solved", False))))
    env.close()
    return float(np.mean(returns)), float(np.mean(solved))


def main(argv: list[str] | None = None) -> Path:
    """Train, save and evaluate; returns the path of the saved ``.zip``."""
    args = parse_args(argv)
    out = args.out or Path("logs") / "sb3" / args.env_id / "model"
    out.parent.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    env = _make_vec_env(args.env_id, args.n_envs, args.seed)
    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = ALGOS[args.algo](
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(args.tensorboard) if args.tensorboard else None,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(out)
    if args.normalize:
        env.save(str(out) + "_vecnormalize.pkl")
    print(f"Saved {out}.zip")

    if args.eval_episodes > 0:
        ret, success = evaluate(
            model,
            args.env_id,
            args.eval_episodes,
            args.seed,
            env if args.normalize else None,
        )
        print(
            f"deterministic policy over {args.eval_episodes} episodes: return {ret:.2f}, "
            f"solved on the final step {100 * success:.1f}%"
        )
    return Path(f"{out}.zip")


if __name__ == "__main__":
    main()
