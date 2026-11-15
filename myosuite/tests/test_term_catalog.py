# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Unit and footprint tests for ``myosuite.terms.catalog``."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from myosuite.terms.catalog import TERM_KINDS, catalog, main, render_markdown, search

pytestmark = pytest.mark.tier1


def test_catalog_discovers_all_kinds_and_features() -> None:
    entries = catalog()
    kinds_found = {e.kind for e in entries}
    assert set(TERM_KINDS).issubset(kinds_found)

    feature_names = {e.name for e in entries if e.kind == "feature"}
    assert {"motor-noise", "fatigue", "sarcopenia", "reafferentation"}.issubset(
        feature_names
    )

    action_names = {e.name for e in catalog(kind="action")}
    assert "sigmoid_muscle_activation" in action_names
    assert "muscle_normalize_action" in action_names


def test_catalog_search_and_invalid_regex() -> None:
    entries = catalog()
    hits = search(entries, r"fatigue|sarcopenia")
    assert len(hits) >= 2

    with pytest.raises(ValueError, match="Invalid regular expression"):
        search(entries, "[unclosed")

    with pytest.raises(ValueError, match="Unknown catalog kind"):
        catalog(kind="nonexistent")


def test_catalog_cli_json_markdown_and_import_footprint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["--kind", "feature", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) >= 4
    assert all(item["kind"] == "feature" for item in payload)

    md = render_markdown(catalog(kind="action"))
    assert "## `action`" in md
    assert "`sigmoid_muscle_activation`" in md

    assert main(["--grep", "[bad"]) == 2
    capsys.readouterr()

    # Subprocess check: running myosuite.terms.catalog must never import torch or mjlab.
    code = (
        "import sys, myosuite.terms.catalog as c; "
        "c.catalog(); "
        "bad = [m for m in ('torch', 'mjlab', 'scipy.spatial') if m in sys.modules]; "
        "assert not bad, f'Forbidden imports loaded: {bad}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
