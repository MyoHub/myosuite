# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Count torch calls that force a device-to-host sync or a host-to-device copy.

On CUDA each counted call either blocks the host until the device catches up
(``.item()``, ``bool(t)``, ``.cpu()``, ``nonzero``, boolean-mask indexing, ...)
or uploads host data (``torch.as_tensor(numpy_array, device=...)``, indexing
with a NumPy array or a list).  The counter sees them on any device, so
CPU-only CI can pin a hot path's sync budget.  ``torch.from_numpy`` and
``Tensor.to`` are not counted (the first is not dispatched to modes, the second
is a no-op on CPU-only runs).

Every call is attributed to its innermost ``myosuite`` package frame (tests
excluded).  ``HostSyncCounter(package_only=True)`` ignores calls with no such
frame, e.g. mjlab's own ``reset_buf.nonzero()`` in ``env.step``.
"""

from __future__ import annotations

import functools
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

_SYNC_METHODS = {
    torch.Tensor.item: "item",
    torch.Tensor.__bool__: "__bool__",
    torch.Tensor.__int__: "__int__",
    torch.Tensor.__float__: "__float__",
    torch.Tensor.__index__: "__index__",
    torch.Tensor.cpu: "cpu",
    torch.Tensor.numpy: "numpy",
    torch.Tensor.tolist: "tolist",
    torch.Tensor.nonzero: "nonzero",
    torch.nonzero: "nonzero",
}
_HOST_DATA_FACTORIES = {
    torch.as_tensor: "as_tensor",
    torch.tensor: "tensor",
}
_INDEXING = {
    torch.Tensor.__getitem__: "getitem",
    torch.Tensor.__setitem__: "setitem",
}
_PACKAGE = Path(__file__).resolve().parents[2]
_TESTS = Path(__file__).resolve().parents[1]
_OUTSIDE = "<outside myosuite>"


@functools.cache
def _counted_file(filename: str) -> bool:
    """Whether *filename* is MyoSuite package code (not a test)."""
    path = Path(filename).resolve()
    return path.is_relative_to(_PACKAGE) and not path.is_relative_to(_TESTS)


def _index_kinds(index: Any) -> list[str]:
    """Sync kinds of an index: bool masks (implicit nonzero) and host arrays."""
    items = index if isinstance(index, tuple) else (index,)
    kinds = []
    for item in items:
        if isinstance(item, torch.Tensor) and item.dtype == torch.bool and item.dim():
            kinds.append("bool-mask")
        elif isinstance(item, np.ndarray | list):
            kinds.append("host-index")
    return kinds


class HostSyncCounter(TorchFunctionMode):
    """``with HostSyncCounter() as c: ...``; then read ``c.counts`` / ``c.total``.

    ``counts`` holds the sync calls by kind (``item``, ``__bool__``, ``cpu``,
    ``numpy``, ``tolist``, ``nonzero``, ``bool-mask getitem``, ...) and the
    host-to-device copies (``H2D as_tensor`` / ``H2D tensor`` of non-tensor
    data, ``host-index getitem`` / ``host-index setitem``).  ``sites`` holds
    the same calls by kind and MyoSuite call site; ``report()`` formats them.

    Args:
        package_only: Count only calls made (directly or through a library)
            by MyoSuite package code.

    Example:
        >>> with HostSyncCounter(package_only=True) as syncs:
        ...     env.step(action)
        >>> assert syncs.total == 0, syncs.report()
    """

    def __init__(self, package_only: bool = False) -> None:
        super().__init__()
        self.package_only = package_only
        self.counts: Counter[str] = Counter()
        self.sites: Counter[tuple[str, str]] = Counter()

    @property
    def total(self) -> int:
        """Number of counted calls."""
        return sum(self.counts.values())

    def report(self) -> str:
        """The counted calls by kind and call site."""
        return "\n".join(
            f"{n} x {kind} at {site}" for (kind, site), n in self.sites.items()
        )

    def _site(self) -> str:
        frame = sys._getframe(2)
        while frame is not None:
            name = frame.f_code.co_filename
            if _counted_file(name):
                return f"{Path(name).name}:{frame.f_lineno}"
            frame = frame.f_back
        return _OUTSIDE

    def _record(self, kind: str) -> None:
        site = self._site()
        if self.package_only and site == _OUTSIDE:
            return
        self.counts[kind] += 1
        self.sites[(kind, site)] += 1

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        name = _SYNC_METHODS.get(func)
        if name is not None:
            self._record(name)
        elif func in _HOST_DATA_FACTORIES:
            if args and not isinstance(args[0], torch.Tensor):
                self._record(f"H2D {_HOST_DATA_FACTORIES[func]}")
        elif func in _INDEXING and len(args) > 1:
            for kind in _index_kinds(args[1]):
                self._record(f"{kind} {_INDEXING[func]}")
        return func(*args, **kwargs)
