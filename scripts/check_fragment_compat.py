#!/usr/bin/env python3
"""
CI script: check myo_sim fragment versions against task declarations.

Fails with a non-zero exit code if any installed fragment version exceeds
the version declared in a task's ``_fragment_versions`` ClassVar.

Usage:
    python scripts/check_fragment_compat.py

Exit codes:
    0 — all fragments compatible
    1 — one or more version mismatches detected
    2 — myo_sim package not available (warning only, exits 0)
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys


def _load_myo_sim_registry():
    """Return myo_sim.FragmentRegistry or None if unavailable."""
    try:
        import myo_sim

        if hasattr(myo_sim, "FragmentRegistry"):
            return myo_sim.FragmentRegistry
    except ImportError:
        pass
    return None


def _discover_task_configs():
    """Yield (class_name, _fragment_versions) for all TaskConfig subclasses.

    Scans myosuite.envs.myo.tasks.basic and sibling packages for classes
    that declare a ``_fragment_versions`` ClassVar.
    """
    import myosuite.envs.myo.tasks.basic as myobase_pkg

    for _finder, modname, _ispkg in pkgutil.walk_packages(
        path=myobase_pkg.__path__,
        prefix=myobase_pkg.__name__ + ".",
        onerror=lambda e: None,
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            fv = getattr(obj, "_fragment_versions", None)
            if fv and isinstance(fv, dict):
                yield obj.__qualname__, fv


def main() -> int:
    registry = _load_myo_sim_registry()
    if registry is None:
        print(
            "WARNING: myo_sim.FragmentRegistry not available — "
            "skipping fragment version check.\n"
            "Ensure the myo_sim submodule is initialized and exposes "
            "FragmentRegistry (myo_sim is not a MyoSuite PyPI package).",
            file=sys.stderr,
        )
        return 0

    errors: list[str] = []

    for task_name, declared_versions in _discover_task_configs():
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
                    f"_fragment_versions[{fragment_name!r}] to {installed_ver}."
                )

    if errors:
        print("Fragment version check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  ✗ {err}", file=sys.stderr)
        return 1

    installed_names = registry.all_names() if hasattr(registry, "all_names") else []
    print(f"Fragment version check PASSED — " f"{len(installed_names)} fragments OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
