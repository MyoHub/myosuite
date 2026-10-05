#!/usr/bin/env python3
"""
Check installed myo_sim fragment versions against task declarations.

A ``TaskConfig`` subclass can pin the fragment versions it was audited
against in its ``fragment_versions`` ClassVar. This script fails with a
non-zero exit code if any installed fragment is newer than declared.

Usage:
    python scripts/check_fragment_compat.py

Exit codes:
    0 — all declared fragments compatible; also when myo_sim exposes no
        FragmentRegistry or no task config declares fragment_versions
        (both print a warning that nothing was checked)
    1 — a version mismatch or unknown fragment, or no TaskConfig subclass
        was discovered at all (the scan itself is broken)
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from collections.abc import Iterator
from typing import Any


def _load_myo_sim_registry() -> Any:
    """Return myo_sim.FragmentRegistry or None if unavailable."""
    try:
        import myo_sim

        if hasattr(myo_sim, "FragmentRegistry"):
            return myo_sim.FragmentRegistry
    except ImportError:
        pass
    return None


def _discover_task_configs() -> Iterator[tuple[str, dict[str, int]]]:
    """Yield (class_name, fragment_versions) for every TaskConfig subclass.

    Imports every module under myosuite.envs.myo.tasks.basic and yields each
    TaskConfig subclass found there once, including those that declare no
    versions (an empty dict). A module that fails to import raises, so its
    declarations cannot be skipped silently.
    """
    import myosuite.envs.myo.tasks.basic as myobase_pkg
    from myosuite.core.config import TaskConfig

    seen: set[type] = set()
    for _finder, modname, _ispkg in pkgutil.walk_packages(
        path=myobase_pkg.__path__,
        prefix=myobase_pkg.__name__ + ".",
    ):
        mod = importlib.import_module(modname)
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj in seen or obj is TaskConfig or not issubclass(obj, TaskConfig):
                continue
            seen.add(obj)
            yield obj.__qualname__, dict(obj.fragment_versions)


def main() -> int:
    registry = _load_myo_sim_registry()
    if registry is None:
        print(
            "WARNING: myo_sim.FragmentRegistry not available — "
            "skipping fragment version check.",
            file=sys.stderr,
        )
        return 0

    configs = list(_discover_task_configs())
    if not configs:
        print(
            "Fragment version check FAILED: no TaskConfig subclass found under "
            "myosuite.envs.myo.tasks.basic, so the scan checked nothing.",
            file=sys.stderr,
        )
        return 1
    declared = [(name, versions) for name, versions in configs if versions]
    if not declared:
        print(
            f"WARNING: none of the {len(configs)} task configs declares "
            "fragment_versions; no fragment version was checked.",
            file=sys.stderr,
        )
        return 0

    errors: list[str] = []
    for task_name, declared_versions in declared:
        for fragment_name, declared_ver in declared_versions.items():
            try:
                info = registry.get(fragment_name)
            except KeyError:
                errors.append(
                    f"{task_name}: fragment {fragment_name!r} not found in myo_sim registry"
                )
                continue

            installed_ver = info.version
            if installed_ver > declared_ver:
                errors.append(
                    f"{task_name}: fragment {fragment_name!r} installed version "
                    f"{installed_ver} > declared {declared_ver}. "
                    "Re-audit this task against the new fragment XML and bump "
                    f"fragment_versions[{fragment_name!r}] to {installed_ver}."
                )

    if errors:
        print("Fragment version check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  ✗ {err}", file=sys.stderr)
        return 1

    n_versions = sum(len(versions) for _, versions in declared)
    print(
        f"Fragment version check PASSED — {n_versions} declared fragment "
        f"versions in {len(declared)} task configs are current."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
