# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Count torch calls that force a device-to-host sync or a host-to-device copy.

On CUDA each counted call either blocks the host until the device catches up
(``.item()``, ``bool(t)``, ``.cpu()``, ``nonzero``, boolean-mask indexing, ...)
or uploads host data (``torch.as_tensor(numpy_array, device=...)``).  The counter
sees them on any device, so CPU-only CI can pin a hot path's sync budget.
``torch.from_numpy`` and ``Tensor.to`` are not counted (the first is not
dispatched to modes, the second is a no-op on CPU-only runs).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

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
    torch.Tensor.__getitem__: "bool-mask getitem",
    torch.Tensor.__setitem__: "bool-mask setitem",
}


def _has_bool_mask(index: Any) -> bool:
    items = index if isinstance(index, tuple) else (index,)
    return any(
        isinstance(i, torch.Tensor) and i.dtype == torch.bool and i.dim() > 0
        for i in items
    )


class HostSyncCounter(TorchFunctionMode):
    """``with HostSyncCounter() as c: ...``; then read ``c.counts`` / ``c.total``.

    ``counts`` holds the sync calls (``item``, ``__bool__``, ``cpu``, ``numpy``,
    ``tolist``, ``nonzero``, bool-mask indexing) and the host-data tensor
    factories (``as_tensor``/``tensor`` of non-tensor data, which
    are H2D copies when the target device is a GPU).
    """

    def __init__(self) -> None:
        super().__init__()
        self.counts: Counter[str] = Counter()

    @property
    def total(self) -> int:
        return sum(self.counts.values())

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
            self.counts[name] += 1
        elif func in _HOST_DATA_FACTORIES:
            if args and not isinstance(args[0], torch.Tensor):
                self.counts[f"H2D {_HOST_DATA_FACTORIES[func]}"] += 1
        elif func in _INDEXING and len(args) > 1 and _has_bool_mask(args[1]):
            self.counts[_INDEXING[func]] += 1
        return func(*args, **kwargs)
