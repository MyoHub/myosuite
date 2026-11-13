# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Directional walking and ChaseTag with a steered myoSuperMimic tracker, scored in MyoSuite's CPU envs.

* :func:`evaluate_directional`: ``myoFullBodyDirectional-v0`` with tutorial 5.4's protocol (8 headings x 10 seeds,
  200 steps at 100 Hz from the env's own reset with activations at 0.05; survival, steps alive and the alignment of
  the displacement with the heading), so the numbers compare with the tutorial's walker. With ``turns``, the heading
  changes by each turn after ``turn_at_s``: the displacement's heading error over the last second, and the time until
  the walking direction (over 0.5 s) is within 15 deg of the new heading.
* :func:`evaluate_chasetag`: ``myoChallengeChaseTagFBVs-v0`` matches under its rules (a tag or the pursuer's fall ends
  the match, a runner's fall does not), the tracker heading toward (chase) or away from (evade) the opponent with the
  arena's wall avoidance (:func:`myosuite.integrations.musclemimic.steering.aim_heading_xy`), against a player that does
  nothing, against itself and against an ONNX player (the published baseline); per pairing and role: tags, chaser
  falls, mean length and median fall times, as in ``docs/baseline_checkpoints.md``.
* :func:`evaluate_tracking`: a sim-to-sim check, the tracker on its clips without steering in a CPU host (by default
  the directional env's body), with the task's normalized return and deviation terminations, to compare with
  ``evaluate.py``'s GPU numbers before reading the two tasks' results.
* :func:`evaluate_run`: all of them, for every checkpoint of a training run, as one CSV row each (training curves).

The tracker follows a looped walking clip (:func:`.steering.loop_clip`), turned by :class:`.steering.HeadingSteer`.
"""

from __future__ import annotations

import csv
import dataclasses
import glob
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
import torch

from myosuite.envs.myo.tasks.mimic.tracking import MissingMotionData, kit_root
from myosuite.envs.myo.tasks.mimic.tracking.models import DEFAULT_MODEL, compiled_body

from .heads import StepCtx, _failure, _reward
from .kinematics import SkeletonKinematics
from .steering import (
    TRACKED_BODIES,
    TrackerAgent,
    TrackerPolicy,
    body_index,
    loop_clip,
    read_state,
    write_reference_state,
)

DIRECTIONAL_ENV = "myoFullBodyDirectional-v0"
CHASETAG_ENV = "myoChallengeChaseTagFBVs-v0"
WALK_REL = "KIT/167/walking_medium06_poses.npz"
WALK_REPO = "amathislab/musclemimic-retargeted"
WALK_FILE = f"MyoFullBody/gmr/{WALK_REL}"
"""The directional env's and the ChaseTag teacher's gait clip (held out in MuscleMimic's KIT split)."""
BASELINE_ONNX = "onnx/chasetag_fbvs_baseline.onnx"
BASELINE_REVISION = "ed40fb4f0a53470ff2106ba5dc2f93f4fd7e68d8"
"""The published FBVs baseline in the baselines repo (``docs/baseline_checkpoints.md``)."""
COMPASS: dict[str, int] = {
    "E": 0,
    "NE": 45,
    "N": 90,
    "NW": 135,
    "W": 180,
    "SW": 225,
    "S": 270,
    "SE": 315,
}
AGENTS: tuple[str, str] = ("agent_0", "agent_1")
OPPONENT: dict[str, str] = {"agent_0": "agent_1", "agent_1": "agent_0"}


def default_walk_clip() -> Path:
    """``KIT/167/walking_medium06`` from :func:`kit_root`, or from ``amathislab/musclemimic-retargeted``."""
    try:
        local = kit_root(DEFAULT_MODEL) / WALK_REL
        if local.is_file():
            return local
    except MissingMotionData:
        pass
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return Path(
        hf_hub_download(repo_id=WALK_REPO, filename=WALK_FILE, repo_type="dataset")
    )


def prepare_walk(src: str | Path, dst: str | Path, seconds: float = 15.0) -> dict:
    """Loop a walking clip for the tracker's body (:func:`.steering.loop_clip`) and return the cycle."""
    kin = SkeletonKinematics(TRACKED_BODIES, "cpu", compiled_body(DEFAULT_MODEL))
    return loop_clip(src, dst, kin.info, kin.R_off.double().numpy(), seconds=seconds)


def _angle_deg(xy: np.ndarray, heading: float) -> float:
    """Unsigned angle between a planar vector and a heading (deg); nan for a vector shorter than 1 cm."""
    if float(np.linalg.norm(xy)) < 1e-2:
        return math.nan
    return abs(
        math.degrees(
            (math.atan2(xy[1], xy[0]) - heading + math.pi) % (2 * math.pi) - math.pi
        )
    )


def directional_host() -> mujoco.MjModel:
    """The directional env's compiled model."""
    from myosuite import make_env  # noqa: PLC0415

    env = make_env(DIRECTIONAL_ENV)
    model = env.unwrapped.model
    env.close()
    return model


def evaluate_tracking(
    tracker: TrackerPolicy,
    host: mujoco.MjModel,
    prefix: str = "",
    threshold: float = 1.0,
) -> dict[str, Any]:
    """Each clip of the tracker's library, played once from its first frame in a CPU host, without steering.

    As the task (and ``evaluate.py``): the state starts on the reference, the tracker acts every ``decimation``
    physics steps, an episode fails when a deviation termination (``threshold``: 1 m / 1 m / 1 rad) fires and ends at
    the clip's last frame; the normalized return is the reward sum (``MimicReward``) over its maximum.

    Args:
        tracker: The policy, with the clips to track as its library.
        host: The host model (one body, ground included).
        prefix: The body's name prefix in the host.
        threshold: The deviation terminations' threshold.

    Returns:
        ``per_clip`` (name, normalized return, completed, steps) and ``summary`` (mean normalized return, share
        completed).
    """
    idx = body_index(host, tracker.info, prefix)
    R_off, m, heads, dev = (
        tracker.kin.R_off.double().cpu().numpy(),
        tracker.motion,
        tracker.heads,
        tracker.device,
    )
    rc = tracker.task.reward
    step_weights = rc.step_weights(tracker.dt)
    w_max = sum(step_weights.values())
    fail_names = tuple(tracker.task.failure.thresholds())
    data = mujoco.MjData(host)
    rows = []
    for c, name in enumerate(m.names):
        mujoco.mj_resetData(host, data)
        start, end = int(m.clip_start[c]), int(m.clip_end[c])
        write_reference_state(host, data, idx, tracker, start)
        a = torch.zeros(1, tracker.info.num_muscles, device=dev)
        total, failed, f = 0.0, False, start
        while f < end:
            ft = torch.tensor([f], dtype=torch.long, device=dev)
            e = tracker.act(read_state(host, data, idx, R_off, dev), ft, a)
            data.ctrl[idx.actuators] = e[0].double().cpu().numpy()
            for _ in range(tracker.decimation):
                mujoco.mj_step(host, data)
            mujoco.mj_forward(host, data)
            f += 1
            ctx = StepCtx(
                read_state(host, data, idx, R_off, dev),
                torch.tensor([f], dtype=torch.long, device=dev),
                e,
                a,
            )
            total += sum(
                step_weights[t.name]
                * float(_reward(heads, ctx, name=t.name, kernel=t.kernel))
                for t in rc.terms
            )
            a = e
            if any(
                bool(_failure(heads, ctx, name=fn, threshold=threshold))
                for fn in fail_names
            ):
                failed = True
                break
        rows.append(
            {
                "clip": name,
                "return": total / (w_max * (end - start)),
                "completed": not failed,
                "steps": f - start,
            }
        )
    return {
        "per_clip": rows,
        "summary": {
            "return": float(np.mean([r["return"] for r in rows])),
            "completed": float(np.mean([r["completed"] for r in rows])),
        },
    }


@dataclass
class DirectionalCfg:
    """Tutorial 5.4's evaluation, plus optional turns."""

    headings: tuple[str, ...] = tuple(COMPASS)
    seeds: int = 10
    steps: int = 200
    """Env steps per episode (tutorial 5.4: 200 at 100 Hz)."""
    turns_deg: tuple[float, ...] = ()
    """Heading changes; empty: tutorial 5.4's fixed headings only."""
    turn_at_s: float = 2.0
    turn_steps: int = 600
    max_rate: float = 1.0
    """The steer's largest turn rate (rad/s)."""
    first_s: float = 0.0
    """Earliest reference time of the phase match (the looped clip's cycle start keeps it off the standing start)."""
    env_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "target_speed": 0.65,
            "w_jpos": 0.0,
            "w_jvel": 0.0,
            "w_survival": 0.3,
        }
    )


def _directional_episode(
    tracker: TrackerPolicy,
    cfg: DirectionalCfg,
    theta: float,
    seed: int,
    turn: float | None,
) -> dict[str, Any]:
    from myosuite import make_env  # noqa: PLC0415
    from myosuite.core.config import EnvConfig  # noqa: PLC0415

    steps = cfg.steps if turn is None else cfg.turn_steps
    env = make_env(
        EnvConfig(DIRECTIONAL_ENV, max_episode_steps=max(steps, 500)),
        seed=seed,
        angle_override=theta,
        **cfg.env_kwargs,
    )
    env.reset(seed=seed)
    raw = env.unwrapped
    if raw.model.na:
        raw.data.act[:] = 0.05
        mujoco.mj_forward(raw.model, raw.data)
    dt = raw.model.opt.timestep * raw.frame_skip
    agent = TrackerAgent(tracker, raw.model, dt, max_rate=cfg.max_rate)
    agent.reset(raw.data, first=math.ceil(cfg.first_s / tracker.dt - 1e-9))
    if agent.frames_left() * agent.hold < steps:
        raise ValueError(
            f"the looped clip is too short for {steps} steps: loop it longer"
        )
    t_turn = round(cfg.turn_at_s / dt)
    xy = [raw.data.qpos[agent.idx.root_q : agent.idx.root_q + 2].copy()]
    alive = True
    for t in range(steps):
        heading = theta if turn is None or t < t_turn else theta + turn
        _, _, term, trunc, _ = env.step(
            agent.host_action(raw.data, heading, sigmoid=True)
        )
        xy.append(raw.data.qpos[agent.idx.root_q : agent.idx.root_q + 2].copy())
        if term or trunc:
            alive = not term and t == steps - 1
            break
    env.close()
    xy_arr = np.array(xy)
    n = len(xy_arr) - 1
    disp = xy_arr[-1] - xy_arr[0]
    norm = float(np.linalg.norm(disp))
    row: dict[str, Any] = {
        "heading_deg": round(math.degrees(theta)),
        "seed": seed,
        "turn_deg": None if turn is None else math.degrees(turn),
        "alive": alive,
        "steps_alive": n,
        "speed": norm / (n * dt) if n else 0.0,
        "align": float(disp @ np.array([math.cos(theta), math.sin(theta)])) / norm
        if norm > 1e-4
        else 0.0,
        "heading_err_deg": _angle_deg(disp, theta),
        "start_frame": agent.reset_frame,
    }
    if turn is not None:
        new = theta + turn
        last = round(1.0 / dt)
        row["final_err_deg"] = (
            _angle_deg(xy_arr[-1] - xy_arr[max(0, n - last)], new)
            if n > t_turn + last
            else math.nan
        )
        w = round(0.5 / dt)
        row["align_time_s"] = next(
            (
                (k - t_turn) * dt
                for k in range(t_turn + w, n + 1)
                if _angle_deg(xy_arr[k] - xy_arr[k - w], new) < 15.0
            ),
            math.nan,
        )
    return row


def evaluate_directional(
    tracker: TrackerPolicy, cfg: DirectionalCfg | None = None
) -> dict[str, Any]:
    """Tutorial 5.4's directional evaluation of the steered tracker (and the turns of ``cfg``).

    Returns:
        ``rows`` (one per episode), ``per_heading`` (``surv``, ``mean_steps_alive``, ``align`` as the tutorial prints
        them) and ``summary`` (over every heading; ``turns`` per turn angle if any).
    """
    cfg = cfg or DirectionalCfg()
    rows = [
        _directional_episode(tracker, cfg, math.radians(COMPASS[h]), s, None)
        for h in cfg.headings
        for s in range(cfg.seeds)
    ]
    per = {}
    for h in cfg.headings:
        r = [x for x in rows if x["heading_deg"] == COMPASS[h]]
        per[h] = {
            "surv": np.mean([x["alive"] for x in r]),
            "mean_steps_alive": np.mean([x["steps_alive"] for x in r]),
            "align": np.mean([x["align"] for x in r]),
        }
    summary: dict[str, Any] = {
        "surv": float(np.mean([x["alive"] for x in rows])),
        "mean_steps_alive": float(np.mean([x["steps_alive"] for x in rows])),
        "align": float(np.mean([x["align"] for x in rows])),
        "speed": float(np.mean([x["speed"] for x in rows])),
    }
    if cfg.turns_deg:
        turns = {}
        for turn in cfg.turns_deg:
            tr = [
                _directional_episode(
                    tracker, cfg, math.radians(COMPASS[h]), s, math.radians(turn)
                )
                for h in cfg.headings
                for s in range(cfg.seeds)
            ]
            rows += tr
            turns[turn] = {
                "surv": float(np.mean([x["alive"] for x in tr])),
                "final_err_deg": float(np.nanmedian([x["final_err_deg"] for x in tr]))
                if any(np.isfinite(x["final_err_deg"]) for x in tr)
                else math.nan,
                "aligned": float(np.mean([np.isfinite(x["align_time_s"]) for x in tr])),
                "align_time_s": float(np.nanmedian([x["align_time_s"] for x in tr]))
                if any(np.isfinite(x["align_time_s"]) for x in tr)
                else math.nan,
            }
        summary["turns"] = turns
    return {"rows": rows, "per_heading": per, "summary": summary}


class Player(Protocol):
    """One ChaseTag player: an action (354 excitations in [0, 1], in its actuator order) per env step."""

    def reset(self, raw: Any, agent: str, task: str) -> None: ...

    def action(
        self, raw: Any, agent: str, obs: np.ndarray, task: str
    ) -> np.ndarray: ...


class StillPlayer:
    """Does nothing: every excitation 0."""

    def reset(self, raw: Any, agent: str, task: str) -> None:
        del raw, agent, task

    def action(self, raw: Any, agent: str, obs: np.ndarray, task: str) -> np.ndarray:
        del raw, agent, obs, task
        return np.zeros(354)


class OnnxPlayer:
    """An ONNX submission (scored observation 546 -> 354 excitations), e.g. ``onnx/chasetag_fbvs_baseline.onnx``."""

    def __init__(self, path: str | Path) -> None:
        from myosuite.utils.onnxruntime_utils import require_onnxruntime  # noqa: PLC0415

        ort = require_onnxruntime()
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name

    def reset(self, raw: Any, agent: str, task: str) -> None:
        del raw, agent, task

    def action(self, raw: Any, agent: str, obs: np.ndarray, task: str) -> np.ndarray:
        del raw, agent, task
        out = self.session.run(None, {self.input: np.asarray(obs, np.float32)[None]})[0]
        return np.asarray(out, np.float64).reshape(-1)


class TrackerPlayer:
    """The steered tracker as a ChaseTag player: toward the opponent when chasing, away when evading, off the
    fences (:func:`aim_heading_xy`). It starts from the phase match over the whole looped clip (its standing start
    included), facing where it stands."""

    def __init__(self, tracker: TrackerPolicy, max_rate: float = 1.0) -> None:
        self.tracker, self.max_rate = tracker, max_rate
        self.agents: dict[tuple[int, str], TrackerAgent] = {}
        self._order: dict[tuple[int, str], np.ndarray] = {}

    def reset(self, raw: Any, agent: str, task: str) -> None:
        del task
        key, meta = (id(raw.model), agent), raw._meta
        if key not in self.agents:
            prefix = raw.model.body(meta.pelvis_body_id[agent]).name.removesuffix(
                "pelvis"
            )
            a = TrackerAgent(
                self.tracker,
                raw.model,
                raw._config.ctrl_dt,
                prefix=prefix,
                max_rate=self.max_rate,
            )
            pos = {int(u): k for k, u in enumerate(a.idx.actuators)}
            self._order[key] = np.array([pos[int(u)] for u in meta.act_indices[agent]])
            self.agents[key] = a
        self.agents[key].reset(raw.data)

    def heading(self, raw: Any, agent: str, task: str) -> float:
        from myosuite.integrations.musclemimic.steering import aim_heading_xy  # noqa: PLC0415
        from myosuite.terms.multiplayer.chase_tag_vs_obs import agent_is_chaser  # noqa: PLC0415
        from myosuite.terms.multiplayer.heading import heading_yaw  # noqa: PLC0415

        meta = raw._meta
        me = raw.data.site_xpos[meta.site_ids[agent]["pelvis_site"]][:2]
        opp = raw.data.site_xpos[meta.site_ids[OPPONENT[agent]]["pelvis_site"]][:2]
        return heading_yaw(aim_heading_xy(me, opp, pursue=agent_is_chaser(agent, task)))

    def action(self, raw: Any, agent: str, obs: np.ndarray, task: str) -> np.ndarray:
        del obs
        key = (id(raw.model), agent)
        e = self.agents[key].excitation(raw.data, self.heading(raw, agent, task))
        return e[self._order[key]]


@dataclass
class ChaseTagCfg:
    """Matches per pairing, role and side."""

    episodes: int = 12
    """Episodes per role and side (docs/baseline_checkpoints.md: 12)."""
    max_rate: float = 1.0
    baseline_onnx: str | None = None
    """An ONNX opponent (e.g. the published baseline); None: no such pairing."""
    pairings: tuple[str, ...] = ("still", "self", "onnx")


def play_match(env: Any, players: dict[str, Player], seed: int) -> dict[str, Any]:
    """One match under the env's rules; the fall times are each player's first fall (s)."""
    obs, info = env.reset(seed=seed)
    raw, task = env.unwrapped, str(info["task"])
    for a in AGENTS:
        players[a].reset(raw, a, task)
    fell_at: dict[str, float] = {}
    for _ in range(raw.max_episode_steps):
        acts = {a: players[a].action(raw, a, obs[a], task) for a in AGENTS}
        obs, _, term, trunc, info = env.step(acts)
        for a, f in info["fell"].items():
            if f and a not in fell_at:
                fell_at[a] = float(info["elapsed_s"])
        if term[AGENTS[0]] or trunc[AGENTS[0]]:
            break
    chaser = AGENTS[0] if task == "CHASE" else AGENTS[1]
    return {
        "task": task,
        "tagged": bool(info["tagged"]),
        "chaser_fell": bool(info["chaser_fell"]),
        "length_s": float(info["elapsed_s"]),
        "chaser_fell_at": fell_at.get(chaser, math.nan),
        "runner_fell_at": fell_at.get(OPPONENT[chaser], math.nan),
        "score": float(info["score"]),
    }


def _fbvs_env(task: str) -> Any:
    import gymnasium as gym  # noqa: PLC0415

    import myosuite.envs.myo.tasks.challenge  # noqa: F401, PLC0415
    from myosuite import make_env  # noqa: PLC0415

    cfg = gym.spec(CHASETAG_ENV).kwargs["task_config"]
    return make_env(
        CHASETAG_ENV,
        task_config=dataclasses.replace(
            cfg,
            task_choice=task,
            w_gait_jpos=0.0,
            w_gait_jvel=0.0,
            w_steering=0.0,
        ),
    )


def _summary(rows: list[dict]) -> dict[str, Any]:
    def med(k: str) -> float:
        x = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.median(x)) if x else math.nan

    return {
        "n": len(rows),
        "tags": sum(r["tagged"] for r in rows),
        "chaser_falls": sum(r["chaser_fell"] for r in rows),
        "mean_length_s": float(np.mean([r["length_s"] for r in rows])),
        "chaser_fell_at_s": med("chaser_fell_at"),
        "runner_fell_at_s": med("runner_fell_at"),
    }


def evaluate_chasetag(
    tracker: TrackerPolicy, cfg: ChaseTagCfg | None = None
) -> dict[str, Any]:
    """FBVs matches of the steered tracker.

    Pairings (``side`` is the tracker's player, ``task`` the env's role: CHASE = player A chases):

    * ``still``: the tracker chases a player that does nothing, as A (CHASE) and as B (EVADE);
    * ``self``: the tracker against itself, A chasing (CHASE) and B chasing (EVADE);
    * ``onnx``: the tracker against ``cfg.baseline_onnx`` on both sides and in both roles.

    Returns:
        ``rows`` (one per match) and ``summary`` (per pairing, side and task: matches, tags, chaser falls, mean length,
        median fall times).
    """
    cfg = cfg or ChaseTagCfg()
    me = TrackerPlayer(tracker, cfg.max_rate)
    configs: list[tuple[str, str, str, Player]] = []
    if "still" in cfg.pairings:
        configs += [
            ("still", "agent_0", "CHASE", StillPlayer()),
            ("still", "agent_1", "EVADE", StillPlayer()),
        ]
    if "self" in cfg.pairings:
        configs += [("self", "both", "CHASE", me), ("self", "both", "EVADE", me)]
    if "onnx" in cfg.pairings and cfg.baseline_onnx:
        onnx = OnnxPlayer(cfg.baseline_onnx)
        configs += [
            ("onnx", side, task, onnx) for side in AGENTS for task in ("CHASE", "EVADE")
        ]
    envs = {task: _fbvs_env(task) for task in {c[2] for c in configs}}
    rows, summary = [], {}
    for pairing, side, task, other in configs:
        players = {a: (me if side in (a, "both") else other) for a in AGENTS}
        r = [play_match(envs[task], players, seed) for seed in range(cfg.episodes)]
        for x in r:
            x.update(pairing=pairing, side=side)
        rows += r
        summary[f"{pairing}/{side}/{task}"] = _summary(r)
    for env in envs.values():
        env.close()
    return {"rows": rows, "summary": summary}


def checkpoints(run_dir: str | Path, every: int = 1) -> list[str]:
    """A run's ``model_*.pt``, by iteration, every ``every``-th (the last one always kept)."""
    files = sorted(
        glob.glob(os.path.join(str(run_dir), "model_*.pt")),
        key=lambda f: int(Path(f).stem.split("_")[-1]),
    )
    keep = files[::every]
    if files and keep[-1] != files[-1]:
        keep.append(files[-1])
    return keep


def curve_row(
    tracker: TrackerPolicy,
    directional: dict | None,
    chasetag: dict | None,
    tracking: dict | None = None,
) -> dict[str, Any]:
    """One CSV row: the checkpoint's training clock and the headline numbers of the evaluations."""
    row: dict[str, Any] = {
        "iteration": tracker.iteration,
        "wall_time_min": tracker.wall_time / 60.0,
        "sim_steps": tracker.sim_steps,
    }
    if tracking:
        row.update(
            track_return=tracking["summary"]["return"],
            track_completed=tracking["summary"]["completed"],
        )
    if directional:
        s = directional["summary"]
        row.update(
            dir_surv=s["surv"],
            dir_steps_alive=s["mean_steps_alive"],
            dir_align=s["align"],
            dir_speed=s["speed"],
        )
        for turn, t in s.get("turns", {}).items():
            row.update(
                {
                    f"turn{turn:+.0f}_surv": t["surv"],
                    f"turn{turn:+.0f}_aligned": t["aligned"],
                    f"turn{turn:+.0f}_err_deg": t["final_err_deg"],
                }
            )
    if chasetag:
        for key, s in chasetag["summary"].items():
            k = key.replace("/", "_")
            row.update(
                {
                    f"{k}_tag_rate": s["tags"] / s["n"],
                    f"{k}_chaser_fall_rate": s["chaser_falls"] / s["n"],
                    f"{k}_mean_length_s": s["mean_length_s"],
                }
            )
    return row


def evaluate_run(
    run_dir: str | Path,
    walk: str | Path,
    out_csv: str | Path,
    every: int = 1,
    directional: DirectionalCfg | None = None,
    chasetag: ChaseTagCfg | None = None,
    track: tuple[str, str] | None = None,
) -> list[dict]:
    """:func:`evaluate_directional`, :func:`evaluate_chasetag` and, with ``track`` (a clip list and the directory its
    paths start from), :func:`evaluate_tracking` in the directional env's body, for the checkpoints of a run (each
    may be None to skip it): one :func:`curve_row` each, written to ``out_csv`` as they finish."""
    rows = []
    host = directional_host() if track else None
    for ck in checkpoints(run_dir, every):
        tracker = TrackerPolicy(ck, walk)
        d = evaluate_directional(tracker, directional) if directional else None
        c = evaluate_chasetag(tracker, chasetag) if chasetag else None
        t = (
            evaluate_tracking(TrackerPolicy(ck, track[0], motion_root=track[1]), host)
            if track and host is not None
            else None
        )
        rows.append(curve_row(tracker, d, c, t))
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"{Path(ck).name}: {rows[-1]}", flush=True)
    return rows
