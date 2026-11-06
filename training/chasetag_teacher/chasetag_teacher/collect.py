# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Roll out :class:`ChaseTagSteeredPair` and write FBVs (obs, action) pairs."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

AGENTS: tuple[str, str] = ("agent_0", "agent_1")


def write_pairs(
    path: Path,
    obs: np.ndarray,
    act: np.ndarray,
    task: np.ndarray,
) -> Path:
    """Save float32 ``obs`` (N, 546), ``act`` (N, 354) and string ``task``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        obs=np.asarray(obs, dtype=np.float32),
        act=np.asarray(act, dtype=np.float32),
        task=np.asarray(task),
    )
    return path.resolve()


def iter_pairs(
    env: Any,
    pair: Any,
    *,
    episodes: int,
    steps: int,
    agent: str = "agent_0",
) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Yield ``(obs[agent], teacher_action, task)`` from scored FBVs."""
    raw = env.unwrapped
    for ep in range(int(episodes)):
        obs, info = env.reset(seed=ep)
        task = str(info["task"])
        pair.reset(raw.model, raw.data, raw._meta, task)
        for t in range(int(steps)):
            acts = pair.actions(raw.model, raw.data, raw._meta, task, t)
            yield (
                np.asarray(obs[agent], dtype=np.float32),
                np.asarray(acts[agent], dtype=np.float32),
                task,
            )
            obs, _, term, trunc, info = env.step(acts)
            if term[agent] or trunc[agent]:
                break


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("pairs.npz"))
    parser.add_argument("--agent", choices=AGENTS, default="agent_0")
    args = parser.parse_args(argv)
    os.environ["MYOSUITE_MUSCLEMIMIC_MODELS_VERSION"] = "1.0.5"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from myosuite import make_env
    from myosuite.integrations.musclemimic.steered_teacher import ChaseTagSteeredPair

    env = make_env("myoChallengeChaseTagFBVs-v0")
    pair = ChaseTagSteeredPair.load(env)
    rows = list(
        iter_pairs(
            env, pair, episodes=args.episodes, steps=args.steps, agent=args.agent
        )
    )
    env.close()
    obs = np.stack([r[0] for r in rows])
    act = np.stack([r[1] for r in rows])
    task = np.array([r[2] for r in rows])
    out = write_pairs(args.out, obs, act, task)
    print(f"wrote {out} n={obs.shape[0]} obs={obs.shape[1]} act={act.shape[1]}")
    return out


if __name__ == "__main__":
    main()
