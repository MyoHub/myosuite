# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Citation metadata for the upstream MuscleMimic project."""

from myosuite.integrations.citations import (
    MUSCLEMIMIC_CITATION as _MUSCLEMIMIC_CITATION,
)

MUSCLEMIMIC_CITATION = _MUSCLEMIMIC_CITATION
MUSCLEMIMIC_PROJECT_URL = MUSCLEMIMIC_CITATION.project_url
MUSCLEMIMIC_ARXIV_URL = MUSCLEMIMIC_CITATION.arxiv_url
MUSCLEMIMIC_CITATION_BIBTEX = MUSCLEMIMIC_CITATION.bibtex

__all__ = [
    "MUSCLEMIMIC_ARXIV_URL",
    "MUSCLEMIMIC_CITATION",
    "MUSCLEMIMIC_CITATION_BIBTEX",
    "MUSCLEMIMIC_PROJECT_URL",
]
