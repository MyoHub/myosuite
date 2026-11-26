# Copyright 2024 Keller Jordan and contributors (MIT License) and
# 2025-2026 Embodied AI Nexus (Eden Contributors, Apache-2.0).
# Modifications Copyright (c) MyoSuite Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See NOTICE.txt in this package.
# Upstream sources:
# - Muon: Jordan et al. (2024), "Muon: An optimizer for hidden layers in neural
#   networks", https://github.com/KellerJordan/Muon (MIT) and PyTorch torch.optim.Muon.
# - Eden: https://github.com/embodied-ai-nexus/Eden (Apache-2.0),
#   packages/eden/src/eden/learning/wrappers/rsl_rl/muon_ppo/ppo.py, and
#   https://github.com/xiaohu-art/Weave.
"""Paired ``torch.optim.Muon`` + ``AdamW`` optimizer and optional RSL-RL ``MuonPPO``.

Adapted upstream components:
    :class:`MuonAdamW` (2-D orthogonalized updates via ``torch.optim.Muon`` with
    ``adjust_lr_fn="match_rms_adamw"`` + 1-D ``AdamW``) and :class:`MuonPPO`
    (adapted from Muon, Weave, and Eden).

MyoSuite-specific integration:
    Unified ``param_groups`` exposure for adaptive-KL learning-rate schedules across
    high-dimensional MyoSuite muscle actor/critic heads.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

MUON_PPO_CLASS_NAME = "eden_rl.muon_ppo:MuonPPO"


class MuonAdamW(torch.optim.Optimizer):
    """Muon for 2-D weight matrices and AdamW for 1-D/scalar parameters under one LR schedule.

    Exposes the concatenated ``param_groups`` of both child optimizers so adaptive-KL
    schedules updating ``group["lr"]`` adjust both optimizers identically.
    """

    def __init__(
        self,
        modules: Sequence[nn.Module],
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        muon_weight_decay: float = 0.1,
        ignore_frozen: bool = True,
    ) -> None:
        if not hasattr(torch.optim, "Muon"):
            raise ImportError(
                f"torch.optim.Muon requires PyTorch >= 2.10 (found {torch.__version__})."
            )
        seen: set[int] = set()
        self.muon_params: list[nn.Parameter] = []
        self.adamw_params: list[nn.Parameter] = []
        for module in modules:
            for param in module.parameters():
                if id(param) in seen or (ignore_frozen and not param.requires_grad):
                    continue
                seen.add(id(param))
                if param.dim() == 2:
                    self.muon_params.append(param)
                else:
                    self.adamw_params.append(param)

        if not self.muon_params and not self.adamw_params:
            raise ValueError("MuonAdamW received no trainable parameters.")

        self.optimizers: list[torch.optim.Optimizer] = []
        if self.muon_params:
            self.optimizers.append(
                torch.optim.Muon(
                    self.muon_params,
                    lr=lr,
                    weight_decay=muon_weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                )
            )
        if self.adamw_params:
            self.optimizers.append(
                torch.optim.AdamW(self.adamw_params, lr=lr, weight_decay=weight_decay)
            )

        super().__init__([*self.muon_params, *self.adamw_params], defaults={})
        self.param_groups = [
            group for opt in self.optimizers for group in opt.param_groups
        ]

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss: float | None = None
        for i, opt in enumerate(self.optimizers):
            res = opt.step(closure if i == 0 else None)
            if loss is None and res is not None:
                loss = float(res)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "optimizers" not in state_dict:
            raise ValueError(
                f"MuonAdamW.load_state_dict expected key 'optimizers', got {sorted(state_dict)}."
            )
        saved = state_dict["optimizers"]
        if len(saved) != len(self.optimizers):
            warnings.warn(
                f"MuonAdamW checkpoint holds {len(saved)} optimizers vs {len(self.optimizers)} active.",
                stacklevel=2,
            )
        for opt, opt_state in zip(self.optimizers, saved):
            opt.load_state_dict(opt_state)
        self.param_groups = [
            group for opt in self.optimizers for group in opt.param_groups
        ]


try:
    from rsl_rl.algorithms import PPO as _RslPPO
except ImportError:  # pragma: no cover - optional [rsl] extra
    _RslPPO = None  # type: ignore[assignment,misc]


if _RslPPO is not None:

    class MuonPPO(_RslPPO):  # type: ignore[misc]
        """RSL-RL PPO replacing Adam with :class:`MuonAdamW` and logging ``explained_variance``."""

        def __init__(
            self,
            *args: Any,
            weight_decay: float = 0.01,
            muon_weight_decay: float = 0.1,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.optimizer = MuonAdamW(
                [self.actor, self.critic],
                lr=self.learning_rate,
                weight_decay=weight_decay,
                muon_weight_decay=muon_weight_decay,
            )

        def update(self) -> dict[str, float]:
            with torch.no_grad():
                values = self.storage.values.flatten()
                returns = self.storage.returns.flatten()
                ev = float(
                    (1.0 - (returns - values).var() / (returns.var() + 1e-8)).item()
                )
            loss_dict: dict[str, float] = super().update()
            loss_dict["explained_variance"] = ev
            return loss_dict
