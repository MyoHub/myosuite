# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Central citation metadata for optional integrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationCitation:
    """Citation metadata exposed for docs and package-level discovery."""

    key: str
    name: str
    project_url: str
    bibtex: str
    arxiv_url: str | None = None


MUSCLEMIMIC_CITATION = IntegrationCitation(
    key="musclemimic",
    name="MuscleMimic",
    project_url="https://github.com/amathislab/musclemimic",
    arxiv_url="https://arxiv.org/abs/2603.25544",
    bibtex="""@article{Li2026MuscleMimic,
  title={Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale},
  author={Li, Chengkun and Wang, Cheryl and Ziliotto, Bianca and Simos, Merkourios and Kovecses, Jozsef and Durandau, Guillaume and Mathis, Alexander},
  journal={arXiv preprint arXiv:2603.25544},
  year={2026}
}""",
)

INTEGRATION_CITATIONS = (MUSCLEMIMIC_CITATION,)
INTEGRATION_CITATIONS_BY_KEY = {
    citation.key: citation for citation in INTEGRATION_CITATIONS
}


def get_integration_citation(key: str) -> IntegrationCitation:
    """Return citation metadata for an integration key."""
    try:
        return INTEGRATION_CITATIONS_BY_KEY[key]
    except KeyError as exc:
        known = ", ".join(sorted(INTEGRATION_CITATIONS_BY_KEY))
        raise KeyError(f"Unknown integration citation {key!r}. Known: {known}") from exc


__all__ = [
    "INTEGRATION_CITATIONS",
    "INTEGRATION_CITATIONS_BY_KEY",
    "IntegrationCitation",
    "MUSCLEMIMIC_CITATION",
    "get_integration_citation",
]
