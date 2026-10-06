# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Command-line spelling of ``EnvConfig.features`` for the training and evaluation scripts.

``--feature NAME`` or ``--feature NAME=JSON`` (repeatable) selects a muscle-command feature:

========================  =========================================================
``motor-noise``           ``MotorNoiseWrapper``; the JSON is the noise levels
                          (``{"constant_std": 0.05}``), default the van Beers (2004) levels
``fatigue``               ``FatigueWrapper``; the JSON holds its options
                          (``{"fatigue_reset_random": true}``)
``sarcopenia``            ``SarcopeniaWrapper``; ``{"force_scale": 0.3}``, default 0.5
``reafferentation``       ``ReafferentationWrapper`` (hand models)
========================  =========================================================

The same features the Python call ``make_env(EnvConfig(env_id, features=...))`` takes.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from gymnasium.envs.registration import WrapperSpec

FEATURE_NAMES = ("motor-noise", "fatigue", "sarcopenia", "reafferentation")


def feature_spec(text: str) -> WrapperSpec:
    """Parse one ``NAME`` or ``NAME=JSON`` feature argument into a wrapper spec.

    Args:
        text: For example ``fatigue`` or ``motor-noise={"constant_std": 0.05}``.

    Returns:
        The wrapper spec.

    Raises:
        ValueError: If the name is unknown or the JSON is not an object.
    """
    from myosuite.envs.wrappers import (
        FatigueWrapper,
        MotorNoiseWrapper,
        ReafferentationWrapper,
        SarcopeniaWrapper,
        wrapper_spec,
    )
    from myosuite.terms.base_action import MotorNoiseCfg

    name, _, raw = text.partition("=")
    options: dict[str, Any] = {}
    if raw:
        try:
            options = json.loads(raw)
        except json.JSONDecodeError as err:
            raise ValueError(f"--feature {name}: invalid JSON {raw!r}") from err
        if not isinstance(options, dict):
            raise ValueError(
                f"--feature {name}: the JSON must be an object, got {raw!r}"
            )
    if name == "motor-noise":
        levels = options or dataclasses.asdict(MotorNoiseCfg.van_beers_2004())
        return wrapper_spec(MotorNoiseWrapper, motor_noise=levels)
    wrappers = {
        "fatigue": FatigueWrapper,
        "sarcopenia": SarcopeniaWrapper,
        "reafferentation": ReafferentationWrapper,
    }
    if name not in wrappers:
        raise ValueError(
            f"Unknown feature {name!r}; choose from {', '.join(FEATURE_NAMES)}."
        )
    return wrapper_spec(wrappers[name], **options)


def parse_feature_args(argv: list[str]) -> tuple[tuple[WrapperSpec, ...], list[str]]:
    """Split the ``--feature`` arguments off *argv*.

    Args:
        argv: Command-line arguments (without the program name).

    Returns:
        ``(specs, rest)``: the wrapper specs in command-line order, and *argv* without
        the ``--feature`` arguments.

    Raises:
        ValueError: If a ``--feature`` argument is malformed.
    """
    specs: list[WrapperSpec] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--feature":
            if i + 1 >= len(argv):
                raise ValueError("--feature needs a value, e.g. --feature fatigue")
            specs.append(feature_spec(argv[i + 1]))
            i += 2
        elif arg.startswith("--feature="):
            specs.append(feature_spec(arg.partition("=")[2]))
            i += 1
        else:
            rest.append(arg)
            i += 1
    return tuple(specs), rest
