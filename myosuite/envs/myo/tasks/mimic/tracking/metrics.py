# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Success metrics of ``myoMimic-v0`` (independent of the control step).

Protocol: every clip plays once from its first frame to its end, deterministic policy,
no time limit. A clip fails when the failure rule fires (1 m relative sites / 1 m root /
1 rad root orientation). Reported, pooled over the counted states (the post-step states
before a clip's end or failure) and per clip first (suffix ``_clip``):

* ``tracked_to_end``: share of clips without failure; ``tracked_to_end_05m`` the same
  with MuscleMimic's stricter validation rule (mean relative site deviation > 0.5 m);
* ``norm_return``: the proposed reward's return divided by its maximum over the clip, i.e. the reward per
  second of reference motion over its maximum, in [0, 1] (steps after a failure earn 0);
* ``mpjpe_mm``: mean distance of the 16 bodies' offsets from the pelvis (world axes);
  ``mpjpe_root_mm`` with the offsets in the root frame; ``g_mpjpe_mm`` of the 17 world
  positions; ``joint_rmse_deg`` / ``joint_mae_deg`` over the independent joints;
  ``root_ori_rad``; ``body_vel_mps``.

Clips whose retarget metadata carries ``quality_flags`` (curation, none by default)
are scored as usual; ``unflagged`` repeats the report without them.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at
commit 8a6244b, ``src/muscle_supermimic/tasks/tracking/scripts/evaluate.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from myosuite.terms import mimic_track as mt

VALIDATION_REL_SITE = 0.5
"""MuscleMimic's validation threshold of the mean relative site deviation (m)."""


def tracking_errors(
    xp: Any, s: mt.TrackState, f: mt.TrackFeatures, r: mt.TrackRef,
    ref_body_pos: Any, ref_body_lin_vel: Any, indep_cols: Any,
) -> dict[str, Any]:  # fmt: skip
    """Per-state errors against the reference (each ``(...,)``).

    Args:
        xp: Array module.
        s: State.
        f: Its features.
        r: Reference at the state's frame.
        ref_body_pos: ``(..., B, 3)`` reference body positions.
        ref_body_lin_vel: ``(..., B, 3)`` reference body velocities.
        indep_cols: Columns of the independent joints in ``joint_pos``.

    Returns:
        Squared and absolute joint errors (rad), relative-site distances in world axes
        and in the root frame, world body distances (m), root angle (rad), body
        velocity error (m/s) and the relative site deviation.
    """
    dq = s.joint_pos[..., indep_cols] - r.joint_pos[..., indep_cols]
    d = xp.sqrt(((f.site_rpos - r.site_rpos) ** 2).sum(-1))
    loc = (f.site_rpos[..., None, :] @ f.root_rot)[..., 0, :]
    ref_loc = (r.site_rpos[..., None, :] @ r.root_rot)[..., 0, :]
    return {
        "q_sq": (dq**2).mean(-1),
        "q_abs": xp.abs(dq).mean(-1),
        "b_d": d.mean(-1),
        "b_root": xp.sqrt(((loc - ref_loc) ** 2).sum(-1)).mean(-1),
        "g_d": xp.sqrt(((s.body_pos - ref_body_pos) ** 2).sum(-1)).mean(-1),
        "root_ori": mt.rotation_angle(xp, r.root_rot, f.root_rot),
        "body_vel": xp.sqrt(((s.body_lin_vel - ref_body_lin_vel) ** 2).sum(-1)).mean(
            -1
        ),
    }


METRICS: dict[str, tuple[str, Callable[[float], float]]] = {
    "joint_rmse_deg": ("q_sq", lambda x: math.degrees(math.sqrt(x))),
    "joint_mae_deg": ("q_abs", math.degrees),
    "mpjpe_mm": ("b_d", lambda x: 1000.0 * x),
    "mpjpe_root_mm": ("b_root", lambda x: 1000.0 * x),
    "g_mpjpe_mm": ("g_d", lambda x: 1000.0 * x),
    "root_ori_rad": ("root_ori", float),
    "body_vel_mps": ("body_vel", float),
}
"""Reported metric -> (the per-state error it averages, the transform of that average)."""


def summarize(
    names: Sequence[str], sums: dict[str, np.ndarray], counts: np.ndarray,
    returns: np.ndarray, max_return: np.ndarray, failed: np.ndarray, failed_05: np.ndarray,
    flagged: Sequence[bool] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """The report of one evaluation from per-clip accumulators.

    Args:
        names: Clip names.
        sums: Per-clip sums of each :func:`tracking_errors` entry.
        counts: Counted states per clip.
        returns: Return per clip.
        max_return: Largest attainable return per clip.
        failed: Clips the failure rule ended.
        failed_05: Clips that failed or exceeded :data:`VALIDATION_REL_SITE`.
        flagged: Clips with quality flags; when any, ``unflagged`` holds the report
            of the other clips (without ``per_clip``) and ``flagged`` their names.

    Returns:
        The metrics (see the module docstring), ``failed`` (clip names) and ``per_clip``.
    """
    norm = returns / max_return
    out: dict[str, Any] = {
        "tracked_to_end": float(1.0 - failed.mean()),
        "tracked_to_end_05m": float(1.0 - failed_05.mean()),
        "norm_return": float(norm.mean()),
        "norm_return_framew": float(returns.sum() / max_return.sum()),
        "n_clips": len(names),
        "failed": [n for n, x in zip(names, failed) if x],
    }
    counted = counts > 0
    for name, (key, fn) in METRICS.items():
        out[name] = fn(float(sums[key].sum() / max(counts.sum(), 1)))
        vals = [fn(float(v)) for v in sums[key][counted] / counts[counted]]
        out[name + "_clip"] = float(np.mean(vals)) if vals else float("nan")
    out["per_clip"] = {n: round(float(x), 4) for n, x in zip(names, norm)}
    fl = np.zeros(len(names), bool) if flagged is None else np.asarray(flagged, bool)
    if fl.any() and not fl.all():
        k = ~fl
        sub = summarize(
            [n for n, x in zip(names, k) if x], {key: v[k] for key, v in sums.items()},
            counts[k], returns[k], max_return[k], failed[k], failed_05[k],
        )  # fmt: skip
        sub.pop("per_clip")
        out["unflagged"] = sub
        out["flagged"] = [n for n, x in zip(names, fl) if x]
    return out


def evaluate(
    policy: Callable[[np.ndarray], np.ndarray],
    env: Any,
    clips: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate a policy on the CPU env, every clip once from its first frame.

    Args:
        policy: Deterministic ``obs -> action``.
        env: A ``myoMimic-v0`` CPU env (``make_env``), e.g. with
            ``overrides={"motion.clips": "kit_test", "episode_length_s": None}``.
        clips: Clip indices (default: all).

    Returns:
        The report of :func:`summarize`.
    """
    u = env.unwrapped
    m = u.motion
    ids = list(range(len(m.names))) if clips is None else list(clips)
    n = len(ids)
    sums: dict[str, np.ndarray] = {k: np.zeros(n) for k, _ in METRICS.values()}
    counts, returns = np.zeros(n), np.zeros(n)
    failed, failed_05 = np.zeros(n, bool), np.zeros(n, bool)
    cols = np.asarray(u.info.indep_cols)
    for i, c in enumerate(ids):
        obs, _ = env.reset(options={"clip": c})
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            returns[i] += info["task_reward"]
            failed[i] |= terminated
            failed_05[i] |= terminated or info["dev_rel_site"] > VALIDATION_REL_SITE
            done = terminated or truncated
            if done:
                break
            s = u.track_state()
            fr = u.motion.frames
            errs = tracking_errors(
                np, s, mt.features(np, s), m.ref(u.frame),
                fr["body_pos"][u.frame], fr["body_lin_vel"][u.frame], cols,
            )  # fmt: skip
            for k in sums:
                sums[k][i] += float(errs[k])
            counts[i] += 1
    max_return = (m.clip_len[ids] - 1) * u.max_step_reward
    flagged = [bool(m.quality_flags[c]) for c in ids] if m.quality_flags else None
    return summarize(
        [m.names[c] for c in ids], sums, counts, returns, max_return, failed,
        failed_05, flagged,
    )  # fmt: skip
