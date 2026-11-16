# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Lazy ``mm_evalkit.datasets`` entry-point runners for MyoSuite EMG datasets."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _find_checkout_root() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[3],
        Path.cwd(),
        *Path.cwd().parents,
    ]
    for root in candidates:
        if (root / "scripts" / "validation" / "validate_emg.py").is_file():
            return root
    return None


def _load_wheel_model(spec: str) -> tuple[Any, str]:
    import mujoco

    direct = Path(spec)
    if direct.is_file():
        if direct.suffix.lower() == ".mjb":
            return mujoco.MjModel.from_binary_path(str(direct)), direct.name
        return mujoco.MjModel.from_xml_path(str(direct)), direct.name

    import myo_sim

    if spec.startswith("myo_sim:"):
        name = spec.split(":", 1)[1]
        return myo_sim.load_model(name), spec
    if hasattr(myo_sim, "ALL_MODELS") and spec in myo_sim.ALL_MODELS:
        return myo_sim.load_model(spec), spec

    from myosuite import make_env

    env = make_env(spec)
    try:
        unwrapped = env.unwrapped
        model = (
            unwrapped.model
            if hasattr(unwrapped, "model")
            else unwrapped.sim.model._model
        )
        return copy.deepcopy(model), spec
    finally:
        env.close()


def _build_npz_directory_source(npz_dir: str | Path) -> Any:
    import mujoco

    from myosuite.biomechanics.emg_validation import (
        ChannelBinding,
        EMGChannelMap,
        EMGTrial,
        load_emg_trial,
    )

    class _NpzDirectorySource:
        def __init__(self, root: str | Path) -> None:
            self.root = Path(root)
            self.name = self.root.name or "npz_emg"
            self._paths = {p.stem: p for p in sorted(self.root.glob("*.npz"))}

        def trial_ids(self) -> Sequence[str]:
            return tuple(self._paths.keys())

        def get_trial(
            self, trial_id: str, model: mujoco.MjModel | None = None
        ) -> EMGTrial:
            del model
            return load_emg_trial(self._paths[trial_id])

        def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap | None:
            if not self._paths:
                return None
            first = load_emg_trial(next(iter(self._paths.values())))
            if isinstance(first.meta.get("channel_map"), dict):
                return EMGChannelMap.from_dict(first.meta["channel_map"])
            act_names = {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
                for i in range(model.nu)
            }
            bindings: list[ChannelBinding] = []
            for ch in first.channel_names:
                for cand in (ch, f"{ch}_r", f"{ch}_l"):
                    if cand in act_names:
                        bindings.append(ChannelBinding(channel=ch, actuators=(cand,)))
                        break
            return (
                EMGChannelMap(name=f"{self.name}_auto", bindings=tuple(bindings))
                if bindings
                else None
            )

    return _NpzDirectorySource(npz_dir)


def _wheel_npz_main(argv: Sequence[str]) -> int:
    from myosuite.biomechanics.emg_validation import (
        EMGChannelMap,
        compare_emg_validation_reports,
        export_mm_evalkit_rollout,
        musculoskeletal_fingerprint,
        summarize_emg_validation,
        validate_trial,
        write_mm_evalkit_policy_config,
    )

    ap = argparse.ArgumentParser(
        prog="mm-evalkit evaluate custom-emg",
        description="Evaluate standardized EMGTrial .npz archives against MuJoCo models.",
    )
    ap.add_argument(
        "--npz-dir", required=True, help="Directory of EMGTrial .npz files."
    )
    ap.add_argument(
        "--model", required=True, help="Target XML/MJB, myo_sim model or env_id."
    )
    ap.add_argument(
        "--model-b", help="Optional second model for paired A/B comparison."
    )
    ap.add_argument("--channel-map", help="Optional JSON channel-map file.")
    ap.add_argument(
        "--limit", type=int, default=0, help="Max trials to evaluate (0 = all)."
    )
    ap.add_argument(
        "--trim-s", type=float, default=0.05, help="Trimmed boundary seconds."
    )
    ap.add_argument("--tolerance-band", type=float, default=0.25, help="EMG half-band.")
    ap.add_argument(
        "--rollout-audit", action="store_true", help="Include Mode 3 audit."
    )
    ap.add_argument(
        "--export-mm-evalkit", help="Directory to export mm-evalkit rollouts."
    )
    ap.add_argument(
        "--rollout-solver",
        choices=("static_optimization", "cmc"),
        default="static_optimization",
    )
    ap.add_argument("--out-dir", required=True, help="Output report directory.")
    args = ap.parse_args(list(argv))

    source = _build_npz_directory_source(args.npz_dir)
    ids = list(source.trial_ids())
    if args.limit > 0:
        ids = ids[: args.limit]
    if not ids:
        raise SystemExit(f"no .npz trials found in {args.npz_dir!r}")

    override_cmap = (
        EMGChannelMap.from_dict(
            json.loads(Path(args.channel_map).read_text(encoding="utf-8"))
        )
        if args.channel_map
        else None
    )

    def _eval_one(
        spec: str,
    ) -> tuple[Any, str, list[dict[str, Any]], dict[str, Any], list[Any]]:
        mdl, lbl = _load_wheel_model(spec)
        cmap = override_cmap or source.default_channel_map(mdl)
        reps: list[dict[str, Any]] = []
        trials_out: list[Any] = []
        for tid in ids:
            tr = source.get_trial(tid, model=mdl)
            trials_out.append(tr)
            rep = validate_trial(
                mdl,
                tr,
                channel_map=cmap,
                trim_s=args.trim_s,
                tolerance_band=args.tolerance_band,
                include_rollout_audit=bool(args.rollout_audit),
            )
            rep["model"] = lbl
            reps.append(rep)
        agg = summarize_emg_validation(reps)
        agg["model"] = lbl
        agg["model_fingerprint"] = musculoskeletal_fingerprint(mdl)
        agg["dataset"] = source.name
        return mdl, lbl, reps, agg, trials_out

    mdl_a, lbl_a, reps_a, sum_a, trials_a = _eval_one(args.model)
    payload: dict[str, Any] = sum_a

    policy_entries: list[dict[str, str]] = []
    slug_a = re.sub(r"[^a-zA-Z0-9_-]+", "_", lbl_a).strip("_") or "model_a"
    if args.export_mm_evalkit:
        mm_dir = Path(args.export_mm_evalkit)
        mm_dir.mkdir(parents=True, exist_ok=True)
        ep_a = (mm_dir / f"{slug_a}_episodes.npz").resolve()
        export_mm_evalkit_rollout(
            mdl_a,
            trials_a,
            ep_a,
            solver=args.rollout_solver,
            motion_group=source.name,
        )
        policy_entries.append(
            {
                "id": slug_a,
                "label": lbl_a,
                "trajectory_data_path": str(ep_a),
            }
        )

    if args.model_b:
        mdl_b, lbl_b, reps_b, sum_b, trials_b = _eval_one(args.model_b)
        comp = compare_emg_validation_reports(
            reps_a, reps_b, label_a=lbl_a, label_b=lbl_b
        )
        payload = {
            "dataset": source.name,
            "model_a": sum_a,
            "model_b": sum_b,
            "comparison": comp,
        }
        if args.export_mm_evalkit:
            mm_dir = Path(args.export_mm_evalkit)
            slug_b = re.sub(r"[^a-zA-Z0-9_-]+", "_", lbl_b).strip("_") or "model_b"
            if slug_b == slug_a:
                slug_b = f"{slug_b}_b"
            ep_b = (mm_dir / f"{slug_b}_episodes.npz").resolve()
            export_mm_evalkit_rollout(
                mdl_b,
                trials_b,
                ep_b,
                solver=args.rollout_solver,
                motion_group=source.name,
            )
            policy_entries.append(
                {
                    "id": slug_b,
                    "label": lbl_b,
                    "trajectory_data_path": str(ep_b),
                }
            )

    if policy_entries and args.export_mm_evalkit:
        write_mm_evalkit_policy_config(
            Path(args.export_mm_evalkit) / "policies.yaml", policy_entries
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "emg_validation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def _dispatch_dataset(dataset_key: str, argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    checkout = _find_checkout_root()
    if checkout is not None:
        root_str = str(checkout)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        mod = importlib.import_module("scripts.validation.validate_emg")
        forwarded = (
            args
            if ("--dataset" in args or "--init-manifest" in args)
            else ["--dataset", dataset_key, *args]
        )
        return int(mod.main(forwarded))

    if dataset_key == "custom" or "--npz-dir" in args:
        return _wheel_npz_main(args)

    sys.stderr.write(
        f"MyoSuite dataset adapter {dataset_key!r} requires a MyoSuite source checkout "
        "(scripts/validation/validate_emg.py) or pre-exported trials via custom-emg --npz-dir.\n"
    )
    return 2


def rajagopal2016_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Rajagopal et al. (2016) walking dataset."""
    return _dispatch_dataset("rajagopal2016", argv)


def lencioni2019_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Lencioni et al. (2019) gait dataset."""
    return _dispatch_dataset("lencioni2019", argv)


def dimitrov2023_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Dimitrov et al. (2023) walking dataset."""
    return _dispatch_dataset("dimitrov2023", argv)


def camargo2021_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Camargo et al. (2021) multi-terrain locomotion dataset."""
    return _dispatch_dataset("camargo2021", argv)


def quesada2024_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Quesada et al. (2024) upper-limb reaching dataset."""
    return _dispatch_dataset("quesada2024", argv)


def emg2pose_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Salter et al. (2024) emg2pose hand dataset."""
    return _dispatch_dataset("emg2pose", argv)


def ninapro_db5_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Pizzolato et al. (2017) NinaPro DB5 forearm/hand dataset."""
    return _dispatch_dataset("ninapro_db5", argv)


def grabmyo_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Pradhan et al. (2022) GRABMyo forearm/wrist dataset."""
    return _dispatch_dataset("grabmyo", argv)


def fritsche2026_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Fritsche et al. (2026) ULTRA-MoCap arm dataset."""
    return _dispatch_dataset("fritsche2026", argv)


def maurice2020_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Maurice et al. (2020) AndyData overhead-work dataset."""
    return _dispatch_dataset("maurice2020", argv)


def diaz2026_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Diaz et al. (2026) BHaM hand biomechanics dataset."""
    return _dispatch_dataset("diaz2026", argv)


def custom_emg_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for custom manifest or standardized ``EMGTrial`` ``.npz`` directories."""
    return _dispatch_dataset("custom", argv)
