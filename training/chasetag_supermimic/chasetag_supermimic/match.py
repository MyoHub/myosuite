# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Bouts of myoChallengeChaseTagFBVs-v0 between SuperMimic players, ONNX policies and idle players."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from myosuite import make_env
from myosuite.terms.multiplayer.chase_tag_vs_obs import agent_is_chaser

AGENTS = ("agent_0", "agent_1")
PREFIX = {"agent_0": "a0_", "agent_1": "a1_"}
ENV_ID = "myoChallengeChaseTagFBVs-v0"
BASELINE_REV = "ed40fb4f0a53470ff2106ba5dc2f93f4fd7e68d8"
"""Revision of the shipped FBVs baseline (``onnx/chasetag_fbvs_baseline.onnx``, see docs/baseline_checkpoints.md)."""


class Player(Protocol):
    """What a bout needs of a player."""

    def reset(self, env: Any, agent: str) -> None: ...

    def act(self, env: Any, agent: str, obs: np.ndarray) -> np.ndarray: ...


class OnnxPlayer:
    """A 546 -> 354 ONNX policy on the player's own observation (default: the shipped baseline)."""

    def __init__(self, path: str | None = None):
        import onnxruntime as ort  # noqa: PLC0415

        if path is None:
            from myosuite.core.hf_io import download_baseline_file  # noqa: PLC0415

            path = str(
                download_baseline_file(
                    "onnx/chasetag_fbvs_baseline.onnx", revision=BASELINE_REV
                )
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.s = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.name = self.s.get_inputs()[0].name

    def reset(self, env, agent: str) -> None:
        pass

    def act(self, env, agent: str, obs: np.ndarray) -> np.ndarray:
        return self.s.run(None, {self.name: np.asarray(obs, np.float32)[None]})[0][0]


class IdlePlayer:
    """Zero excitations (the player collapses)."""

    def reset(self, env, agent: str) -> None:
        pass

    def act(self, env, agent: str, obs: np.ndarray) -> np.ndarray:
        return np.zeros(354, np.float32)


class SmPlayer:
    """Adapter of :class:`~chasetag_supermimic.player.SuperMimicPlayer` to the bout loop."""

    def __init__(self, checkpoint: str, lib: Any, **kwargs: Any):
        self.checkpoint, self.lib, self.kwargs, self.p = checkpoint, lib, kwargs, None
        self._by_agent: dict = {}  # one player per side: each tracks its own body (prefix)

    def _xy(self, env: Any, agent: str) -> tuple[np.ndarray, np.ndarray]:
        u = env.unwrapped
        opp = AGENTS[1 - AGENTS.index(agent)]
        sid = u._meta.site_ids
        return u.data.site_xpos[sid[agent]["pelvis_site"]][:2].copy(), u.data.site_xpos[
            sid[opp]["pelvis_site"]
        ][:2].copy()

    def reset(self, env, agent: str) -> None:
        from .player import SuperMimicPlayer  # noqa: PLC0415

        u = env.unwrapped
        if agent not in self._by_agent:
            self._by_agent[agent] = SuperMimicPlayer(self.checkpoint, self.lib, u.model, PREFIX[agent], u._config.ctrl_dt,
                                                     **self.kwargs)  # fmt: skip
        self.p = self._by_agent[agent]
        me, opp = self._xy(env, agent)
        self.p.reset(u.data, me, opp, agent_is_chaser(agent, u._config.current_task))

    def act(self, env, agent: str, obs: np.ndarray) -> np.ndarray:
        u = env.unwrapped
        me, opp = self._xy(env, agent)
        return self.p.act(
            u.data, me, opp, agent_is_chaser(agent, u._config.current_task)
        )


@dataclass
class Bout:
    """Outcome of one bout: agent_0's task, how it ended, its length, the score and the fall times."""

    role0: str
    end: str
    t: float
    score0: float
    fell: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


def play(env: Any, players: dict[str, Player], seed: int, task: str, noise: float = 0.02, max_steps: int = 1000,
         log_every: int = 0, record: Callable[[Any], None] | None = None) -> Bout:  # fmt: skip
    """One bout with ``task`` ("CHASE": agent_0 chases). Non-idle actions get N(0, noise^2), clipped to [0, 1]."""
    u = env.unwrapped
    u._config.task_choice = task
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    for a in AGENTS:
        players[a].reset(env, a)
    fell = {a: None for a in AGENTS}
    log = []
    for step in range(max_steps):
        acts = {}
        for a in AGENTS:
            x = np.asarray(players[a].act(env, a, obs[a]), np.float64)
            if noise > 0 and not isinstance(players[a], IdlePlayer):
                x = x + rng.normal(0.0, noise, x.shape)
            acts[a] = np.clip(x, 0.0, 1.0)
        obs, _, term, trunc, info = env.step(acts)
        if record is not None:
            record(env)
        for a in AGENTS:
            if fell[a] is None and info["fell"].get(a):
                fell[a] = info["elapsed_s"]
        if log_every and step % log_every == 0:
            sid = u._meta.site_ids
            log.append(
                [info["elapsed_s"]]
                + [
                    float(v)
                    for a in AGENTS
                    for v in u.data.site_xpos[sid[a]["pelvis_site"]]
                ]
            )
        if term["agent_0"] or trunc["agent_0"]:
            break
    end = (
        "tag"
        if info["tagged"]
        else ("chaser_fall" if info["chaser_fell"] else "time_up")
    )
    return Bout(task, end, float(info["elapsed_s"]), float(info["score"]), fell, log)


def make() -> Any:
    """The scored 1v1 env."""
    return make_env(ENV_ID)


def player_scores(
    rows: list[dict], player: str, reference: str = "base"
) -> dict[str, float]:
    """Scores of ``player`` in the rows of an evaluation matrix (``scripts/evaluate.py``).

    Args:
        rows: One dict per bout (``chaser``, ``runner``, ``end``, ``t``, ``chaser_fell``, ``runner_fell``).
        player: The player's name.
        reference: The opponent of the head-to-head score.

    Returns:
        ``tag_idle`` (% of the bouts chasing an idle player that end in a tag), ``head_to_head`` (% of the bouts
        against ``reference`` won: a tag as chaser, no tag as runner) and ``time_up_s`` (mean time before a fall in
        the bouts against non-idle opponents, the bout length when it did not fall).
    """
    idle = [
        r["end"] == "tag"
        for r in rows
        if r["chaser"] == player and r["runner"] == "idle"
    ]
    h2h = [
        r["end"] == "tag"
        for r in rows
        if r["chaser"] == player and r["runner"] == reference
    ]
    h2h += [
        r["end"] != "tag"
        for r in rows
        if r["chaser"] == reference and r["runner"] == player
    ]
    fell = [
        (r["chaser_fell"] if r["chaser"] == player else r["runner_fell"], r["t"])
        for r in rows
        if player in (r["chaser"], r["runner"]) and "idle" not in (r["chaser"], r["runner"])
        and r["chaser"] != r["runner"]
    ]  # fmt: skip
    up = [t if f is None else f for f, t in fell]
    mean = lambda x: float(np.mean(x)) if x else float("nan")  # noqa: E731
    return {
        "tag_idle": 100 * mean(idle),
        "head_to_head": 100 * mean(h2h),
        "time_up_s": mean(up),
    }
