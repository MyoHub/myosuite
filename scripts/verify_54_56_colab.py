#!/usr/bin/env python3
"""Execute the Colab-critical cells of 5.4 (published walker: evaluation, export) and 5.6 (export), no Jupyter."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path


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
    for idx, name in ((2, "setup"), (4, "data")):
        exec(compile(_cell(nb54, idx), f"5.4_{name}", "exec"), ns, ns)
    ns["HAVE_TEACHER"] = (
        False  # the Colab path without clips: no training, the published walker
    )
    for idx, name in ((6, "scene"), (8, "train"), (10, "eval"), (14, "export")):
        exec(compile(_cell(nb54, idx), f"5.4_{name}", "exec"), ns, ns)
    results = ns.get("results", {}).get("published")
    if not results:
        print("FAIL 5.4 eval produced no results", file=sys.stderr)
        return 1
    east = results["E"]
    print(
        "5.4 results",
        {k: (v.survival, v.steps_alive, v.alignment) for k, v in results.items()},
    )
    if east.steps_alive < 150 or east.alignment < 0.5:
        print("FAIL 5.4 East locomotion", east, file=sys.stderr)
        return 1

    nb56 = _nb("5.6_ChaseTag_MyoChallenge.ipynb")
    exec(compile(_cell(nb56, 7), "5.6_export", "exec"), ns, ns)
    onnx = Path(ns.get("ONNX_PATH", REPO / "chasetag_player.onnx"))
    if not onnx.is_file():
        print("FAIL 5.6 ONNX not written", file=sys.stderr)
        return 1
    print("5.6 ONNX", onnx, "bytes", onnx.stat().st_size)
    print("COLAB_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
