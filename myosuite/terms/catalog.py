# Copyright 2025-2026 Embodied AI Nexus (Eden Contributors).
# Modifications Copyright (c) MyoSuite Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Adapted from https://github.com/embodied-ai-nexus/Eden
# (packages/eden/src/eden/managers/catalog.py).
"""Searchable catalog of MyoSuite MDP terms, configs, and physiological features.

Introspects ``myosuite.terms`` modules and registered environment wrapper
features without importing heavy GPU backends (``torch``, ``mjlab``, ``jax``),
and exposes both a Python API (:func:`catalog`, :func:`search`) and the
``myosuite-terms`` CLI entry point.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from gymnasium import Wrapper

TERM_MODULES: dict[str, tuple[str, ...]] = {
    "obs": (
        "myosuite.terms.base_obs",
        "myosuite.terms.mimic_obs",
        "myosuite.terms.opponent_relative_obs",
    ),
    "reward": (
        "myosuite.terms.base_reward",
        "myosuite.terms.mimic_reward",
    ),
    "action": ("myosuite.terms.base_action",),
    "event": ("myosuite.terms.base_event",),
    "termination": ("myosuite.terms.base_termination",),
    "track": (
        "myosuite.terms.mimic_track",
        "myosuite.terms.waypoint",
    ),
    "multiplayer": (
        "myosuite.terms.multiplayer.chase_tag_vs_obs",
        "myosuite.terms.multiplayer.chase_tag_vs_reward",
        "myosuite.terms.multiplayer.common",
        "myosuite.terms.multiplayer.heading",
    ),
}

# Preserve existing CLI names; new wrappers are discovered without an entry here.
_FEATURE_ALIASES: tuple[tuple[str, str], ...] = (
    ("motor-noise", "MotorNoiseWrapper"),
    ("fatigue", "FatigueWrapper"),
    ("sarcopenia", "SarcopeniaWrapper"),
    ("reafferentation", "ReafferentationWrapper"),
    ("perturbation", "PerturbationWrapper"),
    ("dict-obs", "DictObservationWrapper"),
    ("excitation-stage", "ExcitationStageWrapper"),
    ("ctrl-stage", "CtrlStageWrapper"),
)

# Infrastructure wrappers do not represent configurable scientific features.
_FEATURE_EXCLUSIONS = {
    "AttributeForwardingWrapper": "Internal public-attribute forwarding",
    "MjInstabilityTerminationWrapper": "Automatic instability termination",
}

TERM_KINDS: tuple[str, ...] = (*TERM_MODULES.keys(), "feature")

_SKIP_LEADING_PARAMS: frozenset[str] = frozenset({"self", "env", "accessor"})


@dataclass(frozen=True)
class ParamSpec:
    """Signature parameter or dataclass field of a catalog entry."""

    name: str
    annotation: str
    default: str | None
    required: bool


@dataclass(frozen=True)
class CatalogEntry:
    """Metadata record for one MyoSuite term function, config, or feature."""

    kind: str
    name: str
    qualname: str
    module: str
    source: str
    summary: str
    params: tuple[ParamSpec, ...] = ()

    @property
    def haystack(self) -> str:
        param_names = " ".join(p.name for p in self.params)
        return f"{self.kind} {self.name} {self.qualname} {self.summary} {param_names}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _format_source(obj: Any) -> str:
    try:
        file_str = inspect.getsourcefile(obj)
        _, line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return ""
    if not file_str:
        return ""
    path = Path(file_str).resolve()
    root = _repo_root()
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    return f"{rel}:{line}"


def _first_doc_line(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _format_annotation(ann: Any) -> str:
    if ann is inspect.Signature.empty:
        return "Any"
    if isinstance(ann, str):
        return ann
    name = getattr(ann, "__name__", None)
    if isinstance(name, str):
        return name
    return str(ann).replace("typing.", "")


def _format_default(val: Any) -> str:
    if isinstance(val, (int, float, bool, str)) or val is None:
        return repr(val)
    if isinstance(val, tuple) and len(val) <= 4:
        return repr(val)
    return type(val).__name__


def _extract_callable_params(fn: Any) -> tuple[ParamSpec, ...]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ()

    specs: list[ParamSpec] = []
    skipping_leading = True
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if skipping_leading and param.name in _SKIP_LEADING_PARAMS:
            continue
        skipping_leading = False
        has_default = param.default is not inspect.Parameter.empty
        specs.append(
            ParamSpec(
                name=param.name,
                annotation=_format_annotation(param.annotation),
                default=_format_default(param.default) if has_default else None,
                required=not has_default,
            )
        )
    return tuple(specs)


def _extract_dataclass_params(cls: type[Any]) -> tuple[ParamSpec, ...]:
    specs: list[ParamSpec] = []
    for f in dataclasses.fields(cls):
        has_val = f.default is not dataclasses.MISSING
        has_fac = f.default_factory is not dataclasses.MISSING
        default_str: str | None = None
        if has_val:
            default_str = _format_default(f.default)
        elif has_fac:
            fac_name = getattr(f.default_factory, "__name__", "factory")
            default_str = f"{fac_name}()"
        specs.append(
            ParamSpec(
                name=f.name,
                annotation=_format_annotation(f.type),
                default=default_str,
                required=not (has_val or has_fac),
            )
        )
    return tuple(specs)


def _scan_module(mod_name: str, kind: str) -> list[CatalogEntry]:
    mod: ModuleType = importlib.import_module(mod_name)
    entries: list[CatalogEntry] = []
    for attr_name in sorted(dir(mod)):
        if attr_name.startswith("_"):
            continue
        obj = getattr(mod, attr_name)
        if getattr(obj, "__module__", None) != mod_name:
            continue
        if inspect.isfunction(obj):
            entries.append(
                CatalogEntry(
                    kind=kind,
                    name=attr_name,
                    qualname=f"{mod_name}.{attr_name}",
                    module=mod_name,
                    source=_format_source(obj),
                    summary=_first_doc_line(obj),
                    params=_extract_callable_params(obj),
                )
            )
        elif inspect.isclass(obj):
            params = (
                _extract_dataclass_params(obj)
                if dataclasses.is_dataclass(obj)
                else _extract_callable_params(obj.__init__)
            )
            entries.append(
                CatalogEntry(
                    kind=kind,
                    name=attr_name,
                    qualname=f"{mod_name}.{attr_name}",
                    module=mod_name,
                    source=_format_source(obj),
                    summary=_first_doc_line(obj),
                    params=params,
                )
            )
    return entries


def _scan_features() -> list[CatalogEntry]:
    mod_name = "myosuite.envs.wrappers"
    mod = importlib.import_module(mod_name)
    entries: list[CatalogEntry] = []
    aliases = {cls_name: alias for alias, cls_name in _FEATURE_ALIASES}
    for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
        if (
            cls_name.startswith("_")
            or cls.__module__ != mod_name
            or not issubclass(cls, Wrapper)
            or cls_name in _FEATURE_EXCLUSIONS
        ):
            continue
        alias = aliases.get(
            cls_name,
            re.sub(r"(?<!^)(?=[A-Z])", "-", cls_name.removesuffix("Wrapper")).lower(),
        )
        entries.append(
            CatalogEntry(
                kind="feature",
                name=alias,
                qualname=f"{mod_name}.{cls_name}",
                module=mod_name,
                source=_format_source(cls),
                summary=_first_doc_line(cls),
                params=_extract_callable_params(cls.__init__),
            )
        )
    return entries


def _scan_mjlab_optional() -> list[CatalogEntry]:
    try:
        return _scan_module("myosuite.envs.myo.backends.mjlab.mdp", "mjlab")
    except ImportError:
        return []


def catalog(
    kind: str | None = None, *, include_mjlab: bool = False
) -> list[CatalogEntry]:
    """Collect catalog entries across ``myosuite.terms`` and physiological features.

    Args:
        kind: Optional filter in :data:`TERM_KINDS` (or ``"mjlab"`` when
            ``include_mjlab=True``). ``None`` returns all kinds.
        include_mjlab: Whether to also import and scan ``mjlab`` MDP terms when
            installed. Disabled by default to keep import footprint CPU-only.

    Returns:
        Sorted list of :class:`CatalogEntry` items.
    """
    valid_kinds = (*TERM_KINDS, "mjlab") if include_mjlab else TERM_KINDS
    if kind is not None and kind not in valid_kinds:
        raise ValueError(f"Unknown catalog kind {kind!r}; choose from {valid_kinds}.")

    entries: list[CatalogEntry] = []
    for k, modules in TERM_MODULES.items():
        if kind is not None and k != kind:
            continue
        for mod_name in modules:
            entries.extend(_scan_module(mod_name, k))

    if kind in (None, "feature"):
        entries.extend(_scan_features())

    if include_mjlab and kind in (None, "mjlab"):
        entries.extend(_scan_mjlab_optional())

    return sorted(entries, key=lambda e: (e.kind, e.name, e.module))


def search(entries: Sequence[CatalogEntry], pattern: str) -> list[CatalogEntry]:
    """Filter *entries* by case-insensitive regular expression *pattern*."""
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression {pattern!r}: {exc}") from exc
    return [e for e in entries if rx.search(e.haystack)]


def render_text(entries: Sequence[CatalogEntry], *, index_only: bool = False) -> str:
    """Format *entries* as human-readable terminal text."""
    if not entries:
        return "No matching MyoSuite terms found."
    lines: list[str] = []
    for e in entries:
        if index_only:
            lines.append(f"[{e.kind}] {e.name:<34} {e.summary}")
            continue
        sig = ", ".join(
            f"{p.name}: {p.annotation}"
            + (f" = {p.default}" if p.default is not None else "")
            for p in e.params
        )
        loc = f"  ({e.source})" if e.source else ""
        lines.append(f"[{e.kind}] {e.name}({sig}){loc}")
        if e.summary:
            lines.append(f"    {e.summary}")
    return "\n".join(lines)


def render_markdown(entries: Sequence[CatalogEntry]) -> str:
    """Format *entries* as a grouped GitHub-Flavored Markdown reference."""
    if not entries:
        return "_No matching MyoSuite terms found._\n"
    by_kind: dict[str, list[CatalogEntry]] = {}
    for e in entries:
        by_kind.setdefault(e.kind, []).append(e)

    out: list[str] = ["# MyoSuite Term & Feature Catalog", ""]
    for k, group in by_kind.items():
        out.append(f"## `{k}` ({len(group)})")
        out.append("")
        out.append("| Name | Parameters | Source | Summary |")
        out.append("| :--- | :--- | :--- | :--- |")
        for e in group:
            params = ", ".join(
                f"`{p.name}`" + (f"={p.default}" if p.default is not None else "")
                for p in e.params
            )
            summary = e.summary.replace("|", "\\|")
            out.append(f"| `{e.name}` | {params or '—'} | `{e.source}` | {summary} |")
        out.append("")
    return "\n".join(out)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="myosuite-terms",
        description="List and search MyoSuite MDP terms, configs, and features.",
    )
    p.add_argument(
        "--kind",
        default=None,
        help=f"Restrict to one kind ({', '.join(TERM_KINDS)}, or mjlab with --include-mjlab).",
    )
    p.add_argument(
        "--grep",
        default=None,
        help="Case-insensitive regex matched against term name, summary, and parameters.",
    )
    p.add_argument(
        "--index",
        action="store_true",
        help="Compact one-line-per-term summary view.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON array.",
    )
    p.add_argument(
        "--markdown",
        action="store_true",
        help="Emit grouped Markdown tables.",
    )
    p.add_argument(
        "--include-mjlab",
        action="store_true",
        help="Also scan optional mjlab MDP terms when installed.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``myosuite-terms``."""
    args = _build_parser().parse_args(argv)
    try:
        items = catalog(kind=args.kind, include_mjlab=args.include_mjlab)
        if args.grep:
            items = search(items, args.grep)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps([e.to_dict() for e in items], indent=2))
    elif args.markdown:
        print(render_markdown(items), end="")
    else:
        print(render_text(items, index_only=args.index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
