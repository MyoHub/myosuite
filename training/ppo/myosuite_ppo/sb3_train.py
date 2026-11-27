# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train a Stable-Baselines3 policy on one MyoSuite env on the CPU.

The CPU counterpart of ``myosuite_ppo.mjlab_train`` (``scripts/train_mjlab.py``)::

    myosuite-ppo sb3 myoElbowPose1D6MRandom-v0 --timesteps 500000
    python scripts/train_sb3.py myoLegWalk-v0 --algo sac --n-envs 4 --tensorboard logs/tb

Saves ``<out>.zip`` (default ``logs/sb3/<env_id>/model``) and, with ``--normalize``,
``<out>_vecnormalize.pkl``. Afterwards it scores the deterministic policy: mean return
and the share of episodes that are solved on their final step.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

import myosuite  # noqa: F401  (registers the envs)
from myosuite import make_env
from myosuite.benchmarks import solved_from_info

ALGOS: dict[str, Any] = {"ppo": PPO, "sac": SAC, "td3": TD3}


def _terminate_workers(env: SubprocVecEnv) -> None:
    for process in getattr(env, "processes", ()):
        if process.is_alive():
            process.terminate()
    for process in getattr(env, "processes", ()):
        process.join()
    for remote in (*getattr(env, "remotes", ()), *getattr(env, "work_remotes", ())):
        remote.close()
    env.closed = True


class _SpawnVecEnv(SubprocVecEnv):
    def __init__(self, env_fns: list[Callable[[], gym.Env[Any, Any]]]) -> None:
        try:
            super().__init__(env_fns, start_method="spawn")
        except BaseException:
            _terminate_workers(self)
            raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    desc = (__doc__ or "Train a Stable-Baselines3 policy on CPU.").split("\n\n")[0]
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("env_id", nargs="?", default="myoElbowPose1D6MRandom-v0")
    p.add_argument("--algo", choices=sorted(ALGOS), default="ppo")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1, help="Number of CPU envs.")
    p.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        default="dummy",
        help="Serial envs (default) or one spawned worker per env.",
    )
    p.add_argument("--normalize", action="store_true", help="Normalize observations.")
    p.add_argument("--out", type=Path, default=None, help="Output path without .zip.")
    p.add_argument(
        "--tensorboard", type=Path, default=None, help="TensorBoard log dir."
    )
    p.add_argument(
        "--eval-episodes", type=int, default=20, help="0 skips the evaluation."
    )
    args = p.parse_args(argv)
    if args.n_envs < 1:
        p.error("--n-envs must be positive")
    return args


def _make_vec_env(
    env_id: str, n_envs: int, seed: int, vec_env: Literal["dummy", "subproc"] = "dummy"
) -> VecEnv:
    if n_envs < 1:
        raise ValueError("n_envs must be positive")

    def factory(rank: int) -> Callable[[], gym.Env[Any, Any]]:
        def _init() -> gym.Env[Any, Any]:
            env = Monitor(make_env(env_id))
            try:
                env.reset(seed=seed + rank)
            except BaseException:
                env.close()
                raise
            return env

        return _init

    factories = [factory(i) for i in range(n_envs)]
    return _SpawnVecEnv(factories) if vec_env == "subproc" else DummyVecEnv(factories)


def evaluate(
    model: Any, env_id: str, episodes: int, seed: int, vecnorm: Any = None
) -> tuple[float, float]:
    """Mean return and share of episodes solved on the last step (deterministic policy)."""
    env = make_env(env_id)
    returns, solved = [], []
    try:
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
            solved.append(float(solved_from_info(info)))
    finally:
        env.close()
    return float(np.mean(returns)), float(np.mean(solved))


def main(argv: list[str] | None = None) -> Path:
    """Train, save and evaluate; returns the path of the saved ``.zip``."""
    args = parse_args(argv)
    out = args.out or Path("logs") / "sb3" / args.env_id / "model"
    out.parent.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    base_env = env = _make_vec_env(args.env_id, args.n_envs, args.seed, args.vec_env)
    failed = True
    try:
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
        failed = False
        return Path(f"{out}.zip")
    finally:
        # SB3.close() waits for a pending reply, which a failed worker cannot send.
        if failed and isinstance(base_env, SubprocVecEnv):
            _terminate_workers(base_env)
        else:
            env.close()


if __name__ == "__main__":
    main()
