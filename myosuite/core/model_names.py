# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Shared name aliases for compatible full-body models."""

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
