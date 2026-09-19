#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Render SB3 PPO checkpoints from an ``sb3_all_envs`` sweep to MP4.

Reads ``summary.json`` (or scans ``*/result.json``), loads each ``pass``
env's ``ppo_final.zip``, rolls out deterministically, and writes
``renders/<env_id>.mp4`` plus an ``index.md`` gallery.

Usage::

    python scripts/render_sb3_solutions.py
    python scripts/render_sb3_solutions.py --only myoElbowPose1D6MFixed-v0
    python scripts/render_sb3_solutions.py --max-steps 200 --width 640 --height 480
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# MuJoCo Python bindings expose these members dynamically at runtime.
# pylint: disable=no-member,broad-exception-caught


def _is_descendant(model: Any, body_id: int, root_id: int) -> bool:
    """Return whether ``body_id`` belongs to ``root_id``'s subtree."""
    current = body_id
    while current != 0 and current != root_id:
        current = int(model.body_parentid[current])
    return current == root_id


def _actor_body_ids(model: Any) -> np.ndarray:
    """Find articulated actor bodies while excluding world props and targets."""
    roots = [
        body_id
        for body_id in range(1, int(model.nbody))
        if int(model.body_parentid[body_id]) == 0
    ]
    subtrees = {
        root_id: [
            body_id
            for body_id in range(1, int(model.nbody))
            if _is_descendant(model, body_id, root_id)
        ]
        for root_id in roots
    }
    # Musculoskeletal actors are articulated. Single-body objects may have many
    # geoms, so geom count alone is not a reliable actor discriminator.
    actor_roots = [
        root_id for root_id, body_ids in subtrees.items() if len(body_ids) >= 5
    ]
    if not actor_roots and subtrees:
        actor_roots = [max(subtrees, key=lambda root_id: len(subtrees[root_id]))]
    body_ids = sorted(
        {body_id for root_id in actor_roots for body_id in subtrees[root_id]}
    )
    return np.asarray(body_ids, dtype=np.int32)


def _actor_camera(data: Any, actor_body_ids: np.ndarray) -> Any:
    """Build a close free camera centered on the articulated actor."""
    import mujoco

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    points = np.asarray(data.xpos[actor_body_ids], dtype=np.float64)
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = high - low
    camera.lookat[:] = 0.5 * (low + high)
    camera.azimuth = 135.0
    camera.elevation = -18.0
    camera.distance = max(0.65, 1.55 * float(max(span.max(), 0.35)))
    return camera


def _scene_option() -> Any:
    """Show anatomy/environment groups without internal wrap/collision shapes."""
    import mujoco

    option = mujoco.MjvOption()
    # MyoSuite groups 0–2 contain visible anatomy, environment, and task
    # objects. Groups 3/4 are muscle wrapping and collision-debug geoms; showing
    # them produces large translucent cylinders that obscure the skeleton.
    option.geomgroup[:] = 0
    option.geomgroup[:3] = 1
    option.sitegroup[:] = 1
    option.jointgroup[:] = 1
    option.tendongroup[:] = 1
    return option


def _load_pass_env_ids(sweep_dir: Path, only: list[str] | None) -> list[str]:
    """Return env IDs with status=pass and an on-disk PPO checkpoint."""
    summary = sweep_dir / "summary.json"
    rows: list[dict[str, Any]] = []
    if summary.is_file():
        raw = json.loads(summary.read_text())
        rows = list(raw) if isinstance(raw, list) else list(raw.values())
    else:
        for result in sorted(sweep_dir.glob("*/result.json")):
            rows.append(json.loads(result.read_text()))

    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        eid = row.get("env_id")
        if isinstance(eid, str):
            by_id[eid] = row

    out: list[str] = []
    for eid, row in sorted(by_id.items()):
        if only is not None and eid not in only:
            continue
        if row.get("status") != "pass":
            continue
        ckpt = sweep_dir / eid / "ppo_final.zip"
        if not ckpt.is_file():
            print(f"  skip {eid}: missing {ckpt}", flush=True)
            continue
        out.append(eid)
    return out


def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw status text on a frame when OpenCV is available."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        return frame
    out = frame.copy()
    y0 = 22
    cv2.rectangle(
        out,
        (6, 6),
        (min(out.shape[1] - 6, 520), 12 + 20 * len(lines)),
        (20, 20, 20),
        -1,
    )
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (12, y0 + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 230, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _capture_frame(
    renderer: Any,
    model: Any,
    data: Any,
    *,
    camera: Any,
    scene_option: Any,
    actor_body_ids: np.ndarray,
    overlay_lines: list[str],
) -> np.ndarray:
    from myosuite.viz.mj_renderer import _tune_mjv_scene_for_rgb

    points = np.asarray(data.xpos[actor_body_ids], dtype=np.float64)
    camera.lookat[:] = 0.5 * (points.min(axis=0) + points.max(axis=0))
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    _tune_mjv_scene_for_rgb(model, renderer.scene)
    frame = np.asarray(renderer.render(), dtype=np.uint8).copy()
    return _overlay(frame, overlay_lines)


def _rollout_score(
    env: Any,
    policy: Any,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Evaluate one deterministic rollout without rendering."""
    from myosuite.utils.sb3_sweep import solved_from_info

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    solved = False
    steps = 0
    for steps in range(1, max_steps + 1):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(np.asarray(reward).sum())
        solved = solved or solved_from_info(info)
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "return": episode_return,
        "solved": solved,
        "steps": steps,
    }


def _best_rollout(
    env: Any,
    policy: Any,
    *,
    seed: int,
    candidates: int,
    max_steps: int,
) -> dict[str, Any]:
    """Choose solved rollouts first, then the highest-return rollout."""
    rows = [
        _rollout_score(
            env,
            policy,
            seed=seed + offset,
            max_steps=max_steps,
        )
        for offset in range(candidates)
    ]
    return max(rows, key=lambda row: (bool(row["solved"]), float(row["return"])))


def render_one(
    env_id: str,
    *,
    sweep_dir: Path,
    out_dir: Path,
    seed: int,
    max_steps: int,
    width: int,
    height: int,
    fps: int,
    candidates: int,
    skip_existing: bool,
) -> dict[str, Any]:
    """Render one env checkpoint to MP4. Returns a status dict."""
    import imageio.v2 as imageio
    import mujoco
    from stable_baselines3 import PPO

    from myosuite.utils import gym
    from myosuite.viz.mj_renderer import _tune_mjv_scene_for_rgb

    out_path = out_dir / f"{env_id}.mp4"
    if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
        return {
            "env_id": env_id,
            "status": "skipped_existing",
            "path": str(out_path),
        }

    ckpt = sweep_dir / env_id / "ppo_final.zip"
    t0 = time.time()
    env = gym.make(env_id)
    try:
        uw = env.unwrapped
        model = uw.model
        data = uw.data
        policy = PPO.load(str(ckpt), device="cpu")
        best = _best_rollout(
            env,
            policy,
            seed=seed,
            candidates=candidates,
            max_steps=max_steps,
        )
        obs, _ = env.reset(seed=int(best["seed"]))

        renderer = mujoco.Renderer(model, height=height, width=width)
        _tune_mjv_scene_for_rgb(model, renderer.scene)
        actor_body_ids = _actor_body_ids(model)
        camera = _actor_camera(data, actor_body_ids)
        scene_option = _scene_option()

        frames: list[np.ndarray] = []
        ep_r = 0.0
        solved = False
        from myosuite.utils.sb3_sweep import solved_from_info

        for step in range(max_steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += float(np.asarray(reward).sum())
            solved = solved or solved_from_info(info)
            frames.append(
                _capture_frame(
                    renderer,
                    model,
                    data,
                    camera=camera,
                    scene_option=scene_option,
                    actor_body_ids=actor_body_ids,
                    overlay_lines=[
                        env_id,
                        (
                            f"best of {candidates}  seed={best['seed']}  "
                            f"{'SUCCESS' if best['solved'] else 'UNSOLVED'}"
                        ),
                        f"step={step + 1}  R={ep_r:.2f}",
                    ],
                )
            )
            if terminated or truncated:
                break

        renderer.close()
        out_dir.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames, fps=fps)
        return {
            "env_id": env_id,
            "status": "ok",
            "path": str(out_path),
            "frames": len(frames),
            "return": ep_r,
            "solved": solved,
            "candidate_solved": bool(best["solved"]),
            "candidate_seed": int(best["seed"]),
            "candidates": candidates,
            "elapsed_s": round(time.time() - t0, 2),
        }
    finally:
        env.close()


def _write_index(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write a markdown gallery of rendered videos."""
    success_count = sum(bool(row.get("solved")) for row in rows)
    unsolved_count = sum(
        row.get("status") == "ok" and not row.get("solved") for row in rows
    )
    lines = [
        "# SB3 policy rollout audit",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}.",
        "",
        (
            "These checkpoints passed the reward-improvement smoke gate. "
            "Only rows marked **success** reached the environment's `solved` condition; "
            "unsolved rows are not task solutions."
        ),
        "",
        f"**{success_count} successful rollouts · {unsolved_count} unsolved rollouts**",
        "",
        "| Env | Rollout | Frames | Return | Video |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        eid = row.get("env_id", "?")
        status = row.get("status", "?")
        rollout = (
            "success"
            if row.get("solved")
            else ("unsolved" if status == "ok" else status)
        )
        frames = row.get("frames", "")
        ret = row.get("return", "")
        ret_s = f"{ret:.2f}" if isinstance(ret, float) else ret
        path = row.get("path", "")
        link = (
            f"[mp4]({Path(path).name})"
            if path and status in ("ok", "skipped_existing")
            else ""
        )
        lines.append(f"| `{eid}` | {rollout} | {frames} | {ret_s} | {link} |")
    index = out_dir / "index.md"
    index.write_text("\n".join(lines) + "\n")
    return index


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("runs/sb3_all_envs"),
        help="Sweep directory with summary.json and per-env checkpoints.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for MP4s (default: <sweep-dir>/renders).",
    )
    p.add_argument("--only", nargs="*", default=None, help="Optional env-id filter.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--candidates",
        type=int,
        default=10,
        help="Evaluate this many seeds and render solved/highest-return rollout.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip envs that already have a non-empty MP4.",
    )
    args = p.parse_args(argv)

    sweep_dir = args.sweep_dir.resolve()
    out_dir = (args.out_dir or (sweep_dir / "renders")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import myosuite

    myosuite.register_all_envs()

    env_ids = _load_pass_env_ids(sweep_dir, args.only)
    print(f"rendering {len(env_ids)} pass envs → {out_dir}", flush=True)

    rows: list[dict[str, Any]] = []
    n_ok = 0
    n_err = 0
    for i, eid in enumerate(env_ids, start=1):
        print(f"=== [{i}/{len(env_ids)}] {eid} ===", flush=True)
        try:
            row = render_one(
                eid,
                sweep_dir=sweep_dir,
                out_dir=out_dir,
                seed=args.seed,
                max_steps=args.max_steps,
                width=args.width,
                height=args.height,
                fps=args.fps,
                candidates=args.candidates,
                skip_existing=args.skip_existing,
            )
            print(
                f"  {row['status']}"
                + (
                    f" frames={row.get('frames')} R={row.get('return'):.2f} "
                    f"({row.get('elapsed_s')}s)"
                    if row.get("status") == "ok"
                    else ""
                ),
                flush=True,
            )
            if row["status"] == "ok":
                n_ok += 1
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 — sweep continues on per-env failure
            n_err += 1
            err = {
                "env_id": eid,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            rows.append(err)
            print(f"  ERROR {err['reason']}", flush=True)
            traceback.print_exc()

    manifest = out_dir / "render_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    index = _write_index(out_dir, rows)
    print(
        f"Done. ok={n_ok} error={n_err} " f"manifest={manifest} index={index}",
        flush=True,
    )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
