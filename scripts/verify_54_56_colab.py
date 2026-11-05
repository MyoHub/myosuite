#!/usr/bin/env python3
"""Execute the Colab-critical 5.4 eval + 5.6 export cells (no teacher, no Jupyter)."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)


def _nb(name: str) -> dict:
    return json.loads((REPO / "tutorials" / name).read_text())


def _cell(nb: dict, idx: int) -> str:
    return "".join(nb["cells"][idx]["source"])


def _stub_ipython() -> None:
    ipy = types.ModuleType("IPython")
    display_mod = types.ModuleType("IPython.display")

    def display(*_a, **_k):
        return None

    class Video:
        def __init__(self, *a, **k):
            pass

    display_mod.display = display
    display_mod.Video = Video
    ipy.display = display_mod
    sys.modules.setdefault("IPython", ipy)
    sys.modules.setdefault("IPython.display", display_mod)


def main() -> int:
    _stub_ipython()
    ns: dict = {"__name__": "__main__"}

    nb54 = _nb("5.4_MuscleMimic_Directional_Locomotion.ipynb")
    exec(compile(_cell(nb54, 2), "5.4_setup", "exec"), ns, ns)
    exec(compile(_cell(nb54, 3), "5.4_paths", "exec"), ns, ns)
    ns["SECTOR_DEG"] = list(range(0, 360, 45))
    ns["LABELS"] = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    ns["N_SEEDS"] = 1
    ns["N_STEPS"] = 80
    ns["EVAL_LABELS"] = ["E", "N"]
    if not ns["CKPT"].exists():
        print("FAIL missing CKPT", ns["CKPT"], file=sys.stderr)
        return 2
    print("CKPT", ns["CKPT"])
    exec(compile(_cell(nb54, 15), "5.4_eval", "exec"), ns, ns)
    results = ns.get("results")
    if not results:
        print("FAIL 5.4 eval produced no results (cell printed SKIP)", file=sys.stderr)
        return 1
    east = results["E"]
    print("5.4 results", results)
    if east["mean_steps_alive"] < 60 or east["align"] < 0.5:
        print("FAIL 5.4 East locomotion", east, file=sys.stderr)
        return 1

    nb56 = _nb("5.6_ChaseTag_MyoChallenge.ipynb")
    exec(compile(_cell(nb56, 7), "5.6_export", "exec"), ns, ns)
    onnx = Path(ns.get("ONNX_PATH", REPO / "chasetag_player.onnx"))
    if not onnx.is_file():
        onnx = REPO / "chasetag_player.onnx"
    if not onnx.is_file():
        print("FAIL 5.6 ONNX not written", file=sys.stderr)
        return 1
    print("5.6 ONNX", onnx, "bytes", onnx.stat().st_size)
    print("COLAB_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
