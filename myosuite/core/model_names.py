# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Shared name aliases for compatible full-body models."""

from collections.abc import Collection, Sequence

# Left-arm muscles: myo_sim names them ``*_l``, musclemimic_models ``*_left``.
_FULLBODY_LEFT_ARM_MUSCLES: tuple[str, ...] = tuple(
    "DELT1 DELT2 DELT3 SUPSP INFSP SUBSC TMIN TMAJ PECM1 PECM2 PECM3 LAT1 LAT2 "
    "LAT3 CORB TRIlong TRIlat TRImed ANC SUP BIClong BICshort BRA BRD ECRL ECRB "
    "ECU FCR FCU PL PT PQ".split()
)

#: Known name differences between the myo_sim and musclemimic_models full
#: bodies, as ``(myo_sim name, musclemimic_models name)``. Each pair is the
#: same joint (type, axis, range, frame) or muscle (gain, dynamics, length
#: range). Pairs apply in both directions and only when the exact name is
#: missing from the other model.
FULLBODY_NAME_ALIASES: tuple[tuple[str, str], ...] = (
    ("elbow_flexion_r", "elbow_flex_r"),
    ("elbow_flexion_l", "elbow_flex_l"),
    *((f"{muscle}_l", f"{muscle}_left") for muscle in _FULLBODY_LEFT_ARM_MUSCLES),
)


def resolve_model_names(
    names: Sequence[str], model_names: Collection[str], *, field_name: str
) -> list[str]:
    """Resolve full-body aliases without changing namespaces or exact names.

    Args:
        names: Source names, possibly repeated for multi-coordinate joints.
        model_names: Names available in the target model.
        field_name: State field to identify in collision errors.

    Returns:
        Target names; unknown names remain for the caller to reject.

    Raises:
        ValueError: Distinct source names resolve to the same target.
    """
    aliases = dict(FULLBODY_NAME_ALIASES)
    aliases.update((b, a) for a, b in FULLBODY_NAME_ALIASES)
    resolved_names: list[str] = []
    claimed: dict[str, str] = {}
    for name in names:
        target = name
        if name not in model_names:
            for source, alias in aliases.items():
                if name == source or name.endswith("_" + source):
                    target = name[: -len(source)] + alias
                    break
        if target in claimed and claimed[target] != name:
            raise ValueError(
                f"Motion clip {field_name} joints {claimed[target]!r} and {name!r} "
                f"both map to model joint {target!r}."
            )
        claimed[target] = name
        resolved_names.append(target)
    return resolved_names
