# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Hugging Face downloads are pinned to a tested commit (no network: the hub is faked)."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.tier1

_REPO = Path(__file__).resolve().parents[2]
_MM = "amathislab/mm-10m-2"
_CLIPS = "amathislab/musclemimic-retargeted"


class _Stop(Exception):
    """Raised by the fake hub after recording a call."""


@pytest.fixture(name="hub")
def hub_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    """Fake ``hf_hub_download`` / ``snapshot_download`` that record their arguments."""
    calls: list[dict] = []

    def snapshot_download(repo_id: str | None = None, *args: Any, **kw: Any) -> str:
        calls.append({"repo_id": kw.get("repo_id", repo_id), **kw})
        return str(tmp_path)

    def hf_hub_download(repo_id: str | None = None, *args: Any, **kw: Any) -> str:
        calls.append({"repo_id": kw.get("repo_id", repo_id), **kw})
        raise _Stop

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_hub_download)
    return calls


def _assert_pinned(calls: list[dict], repo_id: str) -> None:
    assert calls and all(c["repo_id"] == repo_id for c in calls)
    for call in calls:
        assert call.get("revision") not in (None, "main")


def test_parse_hf_ref_reads_a_revision() -> None:
    from myosuite.core.hf_io import parse_hf_ref

    ref = parse_hf_ref("hf://a/b@abc/sub/dir")
    assert (ref.repo_id, ref.revision, ref.subpath) == ("a/b", "abc", "sub/dir")
    ref = parse_hf_ref("hf://a/b")
    assert (ref.repo_id, ref.revision, ref.subpath) == ("a/b", None, "")


def test_snapshot_paths_are_pinned(hub: list[dict]) -> None:
    from myosuite.core.hf_io import resolve_hf_snapshot
    from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
        resolve_checkpoint_ref,
    )

    resolve_hf_snapshot(f"hf://{_MM}")
    resolve_checkpoint_ref(f"hf://{_MM}")
    _assert_pinned(hub, _MM)
    hub.clear()
    resolve_hf_snapshot(f"hf://{_MM}@abc123")
    assert hub[-1]["revision"] == "abc123"
    resolve_hf_snapshot("hf://someone/unknown")  # an unlisted repo: its default branch
    assert hub[-1]["revision"] is None


def test_dataset_files_are_pinned(hub: list[dict], tmp_path: Path) -> None:
    from myosuite.core.hf_dataset_cache import download_hf_dataset_file
    from myosuite.integrations.musclemimic.hf_demo_cache import download_demo_cache

    download_hf_dataset_file(
        repo_id=_CLIPS, remote_path="x.npz", local_path=tmp_path / "x"
    )
    _assert_pinned(hub, _CLIPS)
    hub.clear()
    download_demo_cache(cache_dir=str(tmp_path))
    _assert_pinned(hub, "amathislab/demo_dataset")


def test_env_reference_clips_are_pinned(hub: list[dict]) -> None:
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
        ChaseTagVsFullbodySteeredTaskConfig,
    )
    from myosuite.envs.myo.tasks.mimic.cpu import MuscleMimicFullbodyDirectionalEnv

    env = SimpleNamespace(_gait_repo=_CLIPS, _gait_file="a.npz")
    for load in ("_load_gait_clip", "_load_circular_clips"):
        with pytest.raises(_Stop):
            getattr(MuscleMimicFullbodyDirectionalEnv, load)(env)
    cfg = SimpleNamespace(
        _gait_qpos_local=None, gait_clip_repo=_CLIPS, gait_clip_filename="a.npz"
    )
    with pytest.raises(_Stop):
        ChaseTagVsFullbodySteeredTaskConfig._ensure_gait_clip(cfg)
    assert len(hub) == 3
    _assert_pinned(hub, _CLIPS)


def test_tracking_dataset_takes_a_revision(hub: list[dict]) -> None:
    from myosuite.envs.myo.tasks.mimic.tracking import download_retargeted_clips

    download_retargeted_clips(f"{_CLIPS}", "myofullbody", "0.2.5")
    _assert_pinned(hub, _CLIPS)
    download_retargeted_clips("owner/data@abc", "myofullbody", "0.2.5")
    assert (hub[-1]["repo_id"], hub[-1]["revision"]) == ("owner/data", "abc")


def test_steered_teacher_downloads_are_pinned(
    hub: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    from myosuite.integrations.musclemimic import fullbody_model
    from myosuite.integrations.musclemimic.steered_teacher import ChaseTagSteeredPair

    monkeypatch.setattr(
        fullbody_model, "compile_mimic_fullbody_mjmodel", lambda cfg: (None, None, None)
    )
    with pytest.raises(_Stop):
        ChaseTagSteeredPair.load(SimpleNamespace(unwrapped=SimpleNamespace()))
    assert [c["repo_id"] for c in hub] == [_MM, _CLIPS]
    for call in hub:
        assert call.get("revision") not in (None, "main")


def test_muscle_meshes_are_pinned(hub: list[dict]) -> None:
    from myosuite.viz.blender_render import MUSCLE_MESH_REPO, _resolve_muscle_mesh

    with pytest.raises(RuntimeError):
        _resolve_muscle_mesh("atlas")
    _assert_pinned(hub, MUSCLE_MESH_REPO)


def test_revisions_are_commit_shas() -> None:
    from myosuite.core.hf_io import HF_REVISIONS
    from myosuite.integrations.musclemimic.terrain_policy import (
        TERRA_REPO,
        TERRA_REVISION,
    )

    assert {_MM, _CLIPS, TERRA_REPO} <= set(HF_REVISIONS)
    assert all(re.fullmatch(r"[0-9a-f]{40}", rev) for rev in HF_REVISIONS.values())
    assert HF_REVISIONS[TERRA_REPO] == TERRA_REVISION


def _unpinned_calls(source: str) -> list[int]:
    """Lines of ``hf_hub_download`` / ``snapshot_download`` calls without ``revision=``."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            keywords = {k.arg for k in node.keywords}
            if (
                name in ("hf_hub_download", "snapshot_download")
                and "revision" not in keywords
            ):
                lines.append(node.lineno)
    return lines


def test_every_download_names_a_revision() -> None:
    """No call in the package, the training projects or the tutorials resolves ``main``."""
    found = {}
    for path in [*(_REPO / "myosuite").rglob("*.py"), *(_REPO / "training").rglob("*.py"),
                 *(_REPO / "tutorials").rglob("*.py")]:  # fmt: skip
        if "tests" in path.parts:
            continue
        if lines := _unpinned_calls(path.read_text(encoding="utf-8")):
            found[str(path.relative_to(_REPO))] = lines
    for path in (_REPO / "tutorials").glob("*.ipynb"):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        for i, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            try:
                lines = _unpinned_calls("".join(cell["source"]))
            except SyntaxError:  # notebook magics
                continue
            if lines:
                found[f"{path.name} cell {i}"] = lines
    assert found == {}
