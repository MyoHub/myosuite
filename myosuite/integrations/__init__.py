# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Optional integration glue for external benchmarks and tooling (e.g. MuscleMimic)."""

from myosuite.integrations.citations import (
    INTEGRATION_CITATIONS,
    INTEGRATION_CITATIONS_BY_KEY,
    MUSCLEMIMIC_CITATION,
    IntegrationCitation,
    get_integration_citation,
)

__all__ = [
    "INTEGRATION_CITATIONS",
    "INTEGRATION_CITATIONS_BY_KEY",
    "IntegrationCitation",
    "MUSCLEMIMIC_CITATION",
    "get_integration_citation",
]
