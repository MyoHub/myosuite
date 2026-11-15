# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0. See NOTICE.txt for upstream attributions.
"""Eden-derived RL backbones, optimizers, estimators, AMP priors, and Genesis muscle bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eden_rl.amp import (
    AmpDiscriminator,
    AmpMotionSampler,
    AmpReplayBuffer,
    FrameNormalizer,
)
from eden_rl.him import HIMActor, HIMEstimator, him_symmetry_losses, sinkhorn
from eden_rl.muon_ppo import MuonAdamW
from eden_rl.simba import (
    GroupEncoder,
    HyperDense,
    HyperEmbedder,
    HyperLERPBlock,
    HyperMLP,
    MLPEncoder,
    Scaler,
    SimBa,
    build_simba_trunk,
    export_simba_to_onnx,
    l2normalize,
    load_simba_policy,
    rebuild_simba_trunk,
    register_simba_rslrl_loader,
)

if TYPE_CHECKING:
    from eden_rl.genesis_bridge import GenesisMuscleBridge

__all__ = [
    "AmpDiscriminator",
    "AmpMotionSampler",
    "AmpReplayBuffer",
    "FrameNormalizer",
    "GenesisMuscleBridge",
    "GroupEncoder",
    "HIMActor",
    "HIMEstimator",
    "HyperDense",
    "HyperEmbedder",
    "HyperLERPBlock",
    "HyperMLP",
    "MLPEncoder",
    "MuonAdamW",
    "Scaler",
    "SimBa",
    "build_simba_trunk",
    "export_simba_to_onnx",
    "him_symmetry_losses",
    "l2normalize",
    "load_simba_policy",
    "rebuild_simba_trunk",
    "register_simba_rslrl_loader",
    "sinkhorn",
]


def __getattr__(name: str) -> Any:
    if name == "GenesisMuscleBridge":
        from eden_rl.genesis_bridge import GenesisMuscleBridge

        return GenesisMuscleBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
