# Copyright (c) 2026 Embodied AI Nexus / Genesis Authors (Eden, Apache-2.0).
# Modified for MyoSuite by the MyoSuite Authors.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Read MANO, SMPL, SMPL+H, and SMPL-X ``.pkl`` files without ``chumpy``.

Official body-model releases pickle ``shapedirs`` as a ``chumpy.ch.Ch`` or a
``chumpy.reordering.Select`` over one. ``chumpy`` is unmaintained and does not build
against modern NumPy, so this module provides lightweight stand-ins for unpickling
and converting model dictionaries to plain NumPy arrays.

Licensing note:
    MyoSuite does **not** bundle, download, or redistribute any SMPL, SMPL+H,
    SMPL-X, or MANO model files or derived parameters. Users must obtain official
    model files directly from the Max Planck Institute for Intelligent Systems and
    comply with their upstream license terms (note that MANO and SMPL-X are
    distributed under non-commercial scientific research licenses):

    - SMPL: https://smpl.is.tue.mpg.de/
    - MANO / SMPL+H: https://mano.is.tue.mpg.de/
    - SMPL-X: https://smpl-x.is.tue.mpg.de/

Adapted from ``eden/utils/chumpy_pickle.py`` (github.com/embodied-ai-nexus/Eden,
Apache-2.0).
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


class _PickledCh:
    """Stand-in for ``chumpy.ch.Ch`` exposing the underlying ``x`` array."""

    x: Any

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        return np.array(self._value(), dtype=dtype, copy=copy)

    def _value(self) -> np.ndarray:
        return np.asarray(self.x)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._value().shape

    def __getitem__(self, key: Any) -> np.ndarray:
        return self._value()[key].copy()


class _PickledSelect(_PickledCh):
    """Stand-in for ``chumpy.reordering.Select``."""

    a: Any
    idxs: Any

    def _value(self) -> np.ndarray:
        selected = np.asarray(self.a).ravel()[np.asarray(self.idxs)]
        shape = getattr(self, "preferred_shape", None)
        return selected if shape is None else selected.reshape(shape)


_CHUMPY_CLASSES: dict[tuple[str, str], type] = {
    ("chumpy.ch", "Ch"): _PickledCh,
    ("chumpy.reordering", "Select"): _PickledSelect,
}


class ChumpyFreeUnpickler(pickle.Unpickler):
    """Unpickler resolving ``chumpy`` array classes to pure-NumPy stand-ins."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "chumpy" or module.startswith("chumpy."):
            cls = _CHUMPY_CLASSES.get((module, name))
            if cls is None:
                known = ", ".join(f"{m}.{n}" for m, n in _CHUMPY_CLASSES)
                raise pickle.UnpicklingError(
                    f"Unsupported chumpy class {module}.{name} (supported: {known}). "
                    "Convert the body model to an .npz archive with plain NumPy arrays."
                )
            return cls
        return super().find_class(module, name)


def _coerce_value(value: Any) -> Any:
    if isinstance(value, _PickledCh):
        return np.array(value)
    if hasattr(value, "toarray") and callable(value.toarray):
        return np.asarray(value.toarray())
    return value


def load_body_model_pkl(path: str | Path) -> dict[str, Any]:
    """Load a body-model ``.pkl`` file into a dictionary of plain NumPy values.

    Handles Python-2 Latin-1 array encodings, ``chumpy`` wrappers, and sparse
    SciPy ``J_regressor`` matrices found in MANO, SMPL, SMPL+H, and SMPL-X releases.

    Args:
        path: Path to the ``.pkl`` file.

    Returns:
        Dictionary with ``chumpy`` and sparse matrices materialized as ``np.ndarray``.
    """
    with Path(path).open("rb") as f:
        raw = ChumpyFreeUnpickler(f, encoding="latin1").load()
    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected dict in body-model pickle {path}, got {type(raw).__name__}."
        )
    return {str(k): _coerce_value(v) for k, v in raw.items()}


def install_chumpy_pickle_modules() -> None:
    """Register minimal ``chumpy`` stand-ins in ``sys.modules`` for ``pickle.load``.

    Leaves an already-imported ``chumpy`` module untouched.
    """
    if sys.modules.get("chumpy") is not None:
        return
    modules = {
        name: types.ModuleType(name)
        for name in ("chumpy", "chumpy.ch", "chumpy.reordering")
    }
    for (module, name), cls in _CHUMPY_CLASSES.items():
        setattr(modules[module], name, cls)
    modules["chumpy"].ch = modules["chumpy.ch"]  # type: ignore[attr-defined]
    modules["chumpy"].reordering = modules["chumpy.reordering"]  # type: ignore[attr-defined]
    sys.modules.update(modules)


__all__ = [
    "ChumpyFreeUnpickler",
    "install_chumpy_pickle_modules",
    "load_body_model_pkl",
]
