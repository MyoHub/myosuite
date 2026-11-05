"""Build the boxing skill library: cut the mazen_c3d mocap clips into short skills.

Three subcommands::

    # suggest punch segments (prints a table and writes a draft manifest)
    python scripts/boxing_skills.py detect --out logs/boxing_skills_draft.yaml

    # cut every skill in the manifest into its own padded .npz clip
    python scripts/boxing_skills.py cut

    # render one review video per skill plus an all-skills reel
    python scripts/boxing_skills.py render

The manifest (``boxing_skills.yaml`` next to the BoxingVs task) is the source of truth:
edit a frame range there and re-run ``cut``. Segments stay in mocap coordinates;
re-anchoring to the boxer and blending between skills happen at runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
CLIP_DIR = REPO / "boxing_motions/MyoFullBody/gmr/Transitions_mocap/mazen_c3d"
SKILL_DIR = REPO / "boxing_motions/MyoFullBody/gmr/skills"
MANIFEST = REPO / "myosuite/envs/myo/tasks/challenge/boxing_vs/boxing_skills.yaml"
VIDEO_DIR = REPO / "logs/videos/boxing_skills"

# Per-frame arrays that are sliced; everything else in the .npz is model metadata.
PER_FRAME = (
    "qpos",
    "qvel",
    "xpos",
    "xquat",
    "cvel",
    "subtree_com",
    "site_xpos",
    "site_xmat",
)
VELOCITY_FIELDS = ("qvel", "cvel")

# Frame windows of each clip where the actor punches in place (read off the signal plots).
PUNCH_WINDOWS: dict[str, tuple[int, int]] = {
    "punchboxing_push_poses": (25, 160),
    "punchboxing_stand_poses": (25, 165),
    "punchboxing_walk_poses": (25, 200),
    "punchboxing_kick_poses": (25, 190),
    "punchboxing_jumpinplace_poses": (25, 215),
    "punchboxing_running_poses": (195, 300),
}


@dataclass
class Skill:
    """One manifest entry: a frame range of a source clip, padded at the end."""

    name: str
    clip: str
    start: int
    end: int
    pad: int = 15
    kind: str = "punch_R"  # menu group: punch_R, punch_L, combo, guard, step_fwd, ...
    note: str = ""

    @classmethod
    def from_dict(cls, name: str, d: dict) -> Skill:
        """Build from a manifest entry ``{clip, frames: [start, end], pad, kind, note}``."""
        start, end = d["frames"]
        return cls(
            name=name,
            clip=d["clip"],
            start=int(start),
            end=int(end),
            pad=int(d.get("pad", 15)),
            kind=d.get("kind", "punch_R"),
            note=d.get("note", ""),
        )


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _site(d: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    names = [str(s) for s in d["site_names"]]
    return np.asarray(d["site_xpos"][:, names.index(name)], dtype=np.float64)


def hand_signals(d: np.lib.npyio.NpzFile) -> dict[str, dict[str, np.ndarray]]:
    """Smoothed per-hand reach (horizontal, from pelvis), speed and height above the head."""
    hz = float(d["frequency"])
    pelvis, head = _site(d, "pelvis_mimic"), _site(d, "head_mimic")
    out = {}
    for side, site in (("R", "right_hand_mimic"), ("L", "left_hand_mimic")):
        h = _site(d, site)
        speed = np.r_[0.0, np.linalg.norm(np.diff(h, axis=0), axis=1) * hz]
        out[side] = {
            "reach": _smooth(np.linalg.norm((h - pelvis)[:, :2], axis=1)),
            "speed": _smooth(speed),
            "height": h[:, 2] - head[:, 2],
        }
    return out


def detect_punches(
    d: np.lib.npyio.NpzFile,
    window: tuple[int, int],
    min_reach: float = 0.42,
    min_rise: float = 0.12,
    min_speed: float = 2.0,
) -> list[dict]:
    """Find single punches: a hand-reach peak between two minima of the combined reach.

    Args:
        d: Loaded clip.
        window: ``(start, end)`` frames to search.
        min_reach: Minimum horizontal hand reach at the peak (m).
        min_rise: Minimum reach gained from the lower bracketing minimum (m).
        min_speed: Minimum hand speed within 10 frames of the peak (m/s).

    Returns:
        Punches sorted by start frame: ``{hand, start, peak, end, reach, speed, height}``.
    """
    a, b = window
    sig = hand_signals(d)
    # Cut points are minima of the combined reach, so a cut never lands mid-punch of
    # the other hand.
    both = _smooth(sig["R"]["reach"] + sig["L"]["reach"], 7)
    cuts = (
        [a]
        + [
            i
            for i in range(a + 1, b - 1)
            if both[i] <= both[i - 1] and both[i] < both[i + 1]
        ]
        + [b - 1]
    )
    found = []
    for side, s in sig.items():
        r = s["reach"]
        for p in range(a + 1, b - 1):
            if not (r[p] >= r[p - 1] and r[p] > r[p + 1] and r[p] > min_reach):
                continue
            lo = max(c for c in cuts if c < p)
            hi = min(c for c in cuts if c > p)
            speed = float(s["speed"][max(a, p - 10) : p + 10].max())
            if r[p] - min(r[lo], r[hi]) < min_rise or speed < min_speed:
                continue
            if hi == b - 1:  # runs past the window: punch may be cut short
                continue
            found.append(
                {
                    "hand": side,
                    "start": int(lo),
                    "peak": int(p),
                    "end": int(hi),
                    "reach": round(float(r[p]), 2),
                    "speed": round(speed, 1),
                    "height": round(float(s["height"][p]), 2),
                }
            )
    # Two hands peaking between the same cuts is a double punch: keep the faster one.
    best: dict[tuple[int, int], dict] = {}
    for f in found:
        key = (f["start"], f["end"])
        if key not in best or f["speed"] > best[key]["speed"]:
            best[key] = f
    return sorted(best.values(), key=lambda x: x["start"])


def guard_frame(d: np.lib.npyio.NpzFile, window: tuple[int, int]) -> int:
    """Frame in ``window`` with both hands pulled in and the least joint motion."""
    a, b = window
    sig = hand_signals(d)
    joint_speed = _smooth(np.linalg.norm(d["qvel"][:, 6:], axis=1))
    score = (
        sig["R"]["reach"]
        + sig["L"]["reach"]
        + 0.01 * joint_speed
        + sig["R"]["speed"] * 0.05
    )
    return int(a + np.argmin(score[a:b]))


def cmd_detect(args: argparse.Namespace) -> None:
    """Print detected punches per clip and write them as a draft manifest."""
    draft: dict[str, dict] = {}
    for stem, window in PUNCH_WINDOWS.items():
        d = np.load(CLIP_DIR / f"{stem}.npz", allow_pickle=True)
        short = stem.removeprefix("punchboxing_").removesuffix("_poses")
        print(f"\n{stem}  window={window}  guard frame={guard_frame(d, window)}")
        counts = {"R": 0, "L": 0}
        for p in detect_punches(d, window):
            counts[p["hand"]] += 1
            name = f"punch_{p['hand']}_{short}{counts[p['hand']]}"
            print(
                f"  {name:24s} frames {p['start']:4d}-{p['end']:4d}  peak {p['peak']:4d}"
                f"  reach {p['reach']:.2f} m  speed {p['speed']:.1f} m/s  "
                f"hand-head z {p['height']:+.2f} m"
            )
            draft[name] = {
                "clip": stem,
                "frames": [p["start"], p["end"]],
                "kind": f"punch_{p['hand']}",
                "note": f"peak {p['peak']}, reach {p['reach']} m, {p['speed']} m/s",
            }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump({"skills": draft}, sort_keys=False))
        print(f"\nDraft manifest: {args.out}")


def load_manifest(path: Path = MANIFEST) -> list[Skill]:
    """Read the skill manifest."""
    raw = yaml.safe_load(path.read_text())
    return [Skill.from_dict(k, v) for k, v in raw["skills"].items()]


def cut_skill(skill: Skill) -> dict[str, np.ndarray]:
    """Slice one skill out of its source clip and hold its last frame for ``skill.pad`` frames."""
    src = np.load(CLIP_DIR / f"{skill.clip}.npz", allow_pickle=True)
    n = src["qpos"].shape[0]
    if not 0 <= skill.start < skill.end <= n:
        raise ValueError(
            f"{skill.name}: frames {skill.start}-{skill.end} outside 0-{n}"
        )
    out = {k: src[k] for k in src.files}
    for k in PER_FRAME:
        seg = np.asarray(src[k][skill.start : skill.end])
        hold = np.repeat(seg[-1:], skill.pad, axis=0)
        if k in VELOCITY_FIELDS:
            hold = np.zeros_like(hold)
        out[k] = np.concatenate([seg, hold], axis=0)
    total = out["qpos"].shape[0]
    out["split_points"] = np.array([0, total], dtype=np.int32)
    out["source_clip"] = np.str_(skill.clip)
    out["source_frames"] = np.array([skill.start, skill.end], dtype=np.int32)
    out["skill_kind"] = np.str_(skill.kind)
    return out


def cmd_cut(args: argparse.Namespace) -> None:
    """Write one .npz per manifest skill and check it loads as a MotionClip."""
    from myosuite.core.trajectory_io import check_clip_rate, load_motion_clip

    args.out_dir.mkdir(parents=True, exist_ok=True)
    skills = load_manifest(args.manifest)
    for skill in skills:
        path = args.out_dir / f"{skill.name}.npz"
        np.savez(path, **cut_skill(skill))
        clip = load_motion_clip(path, expected_nq=89, expected_nv=88)
        check_clip_rate(clip.frequency_hz, 0.01, skill.name)
        print(
            f"{skill.name:26s} {skill.kind:8s} {skill.clip}[{skill.start}:{skill.end}]"
            f" + {skill.pad} pad -> {clip.qpos.shape[0]} frames"
        )
    print(f"\n{len(skills)} skills written to {args.out_dir}")


def cmd_render(args: argparse.Namespace) -> None:
    """Render each skill kinematically (side view) and an all-skills reel."""
    import mujoco

    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )
    from myosuite.utils.video_io import add_text_to_frame, write_video

    model, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 150.0, -8.0, 2.6
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reel: list[np.ndarray] = []
    skills = load_manifest(args.manifest)
    if args.only:
        skills = [s for s in skills if s.name in args.only]
    for skill in skills:
        qpos = np.load(SKILL_DIR / f"{skill.name}.npz")["qpos"]
        n_motion = skill.end - skill.start
        frames = []
        for i, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [q[0], q[1], 1.0]
            renderer.update_scene(data, cam, scene_option=opt)
            tag = "pad" if i >= n_motion else f"{skill.start + i}"
            text = f"{skill.name} ({skill.kind})\n{skill.clip} frame {tag}"
            frames.append(
                np.asarray(
                    add_text_to_frame(
                        renderer.render(), text, color=(255, 255, 255), fontsize=16
                    )
                )
            )
        write_video(
            args.out_dir / f"{skill.name}.mp4", frames, fps=int(100 * args.speed)
        )
        reel.extend(frames + [frames[-1]] * 20)
        print(f"rendered {skill.name}")
    write_video(args.out_dir / "all_skills.mp4", reel, fps=int(100 * args.speed))
    renderer.close()
    print(f"Videos in {args.out_dir}")


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run a subcommand."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="Suggest punch segments.")
    d.add_argument("--out", type=Path, default=None, help="Draft manifest path.")
    d.set_defaults(func=cmd_detect)
    c = sub.add_parser("cut", help="Cut manifest skills into .npz clips.")
    c.add_argument("--manifest", type=Path, default=MANIFEST)
    c.add_argument("--out-dir", type=Path, default=SKILL_DIR)
    c.set_defaults(func=cmd_cut)
    r = sub.add_parser("render", help="Render review videos.")
    r.add_argument("--manifest", type=Path, default=MANIFEST)
    r.add_argument("--out-dir", type=Path, default=VIDEO_DIR)
    r.add_argument(
        "--speed", type=float, default=0.5, help="0.5 = half-speed playback."
    )
    r.add_argument("--only", nargs="*", default=None, help="Render just these skills.")
    r.set_defaults(func=cmd_render)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
