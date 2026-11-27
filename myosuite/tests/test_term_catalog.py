# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Unit and footprint tests for ``myosuite.terms.catalog``."""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import subprocess
import sys

import pytest

from myosuite.terms.catalog import (
    TERM_KINDS,
    TERM_MODULES,
    _FEATURE_ALIASES,
    _FEATURE_EXCLUSIONS,
    catalog,
    main,
    render_markdown,
    search,
)

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


def test_catalog_covers_term_modules() -> None:
    import myosuite.terms as terms

    # Enumerate files independently: new modules must be classified explicitly.
    discovered = {
        module.name
        for module in pkgutil.walk_packages(terms.__path__, terms.__name__ + ".")
        if not module.ispkg
        and module.name != "myosuite.terms.catalog"
        and not module.name.rsplit(".", 1)[-1].startswith("_")
    }
    indexed = [name for names in TERM_MODULES.values() for name in names]
    assert len(indexed) == len(set(indexed)), "Term modules must have one category"
    assert (
        set(indexed) == discovered
    ), "Update TERM_MODULES for added/moved/removed modules"


def test_catalog_covers_public_wrappers() -> None:
    from gymnasium import Wrapper
    from myosuite.envs import wrappers

    public = {
        name
        for name, cls in inspect.getmembers(wrappers, inspect.isclass)
        if not name.startswith("_")
        and cls.__module__ == wrappers.__name__
        and issubclass(cls, Wrapper)
    }
    assert set(_FEATURE_EXCLUSIONS) <= public, "Remove stale exclusions"
    assert all(_FEATURE_EXCLUSIONS.values()), "Explain each infrastructure exclusion"
    assert {
        name for _, name in _FEATURE_ALIASES
    } <= public, "Remove stale feature aliases"
    entries = catalog(kind="feature")
    assert {entry.qualname for entry in entries} == {
        f"{wrappers.__name__}.{name}" for name in public - _FEATURE_EXCLUSIONS.keys()
    }
    assert len({entry.name for entry in entries}) == len(entries)


def test_catalog_discovers_new_wrapper_without_registry_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gymnasium import Wrapper
    from myosuite.envs import wrappers

    class BalanceAssistWrapper(Wrapper):
        """A newly added scientific feature."""

    BalanceAssistWrapper.__module__ = wrappers.__name__
    monkeypatch.setattr(
        wrappers, "BalanceAssistWrapper", BalanceAssistWrapper, raising=False
    )
    entries = catalog(kind="feature")
    assert any(entry.name == "balance-assist" for entry in entries)


def test_catalog_tracks_live_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("myosuite.terms.base_action")

    def catalog_probe(accessor: object, gain: float = 0.25) -> float:
        """Current parameter documentation."""
        return gain

    catalog_probe.__module__ = module.__name__
    monkeypatch.setattr(module, "catalog_probe", catalog_probe, raising=False)
    entry = next(
        entry for entry in catalog(kind="action") if entry.name == "catalog_probe"
    )
    assert entry.summary == "Current parameter documentation."
    assert [(param.name, param.default) for param in entry.params] == [("gain", "0.25")]
