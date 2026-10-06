# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Citation metadata of the integrations: every entry is complete and parses."""

from __future__ import annotations

import re

import pytest

from myosuite.integrations.citations import (
    INTEGRATION_CITATIONS,
    get_integration_citation,
)

pytestmark = pytest.mark.tier1

_DATA_SOURCES = ("amass", "kit", "gmr", "kinesis")


def test_motion_data_sources_are_cited() -> None:
    """The sources of the MuscleMimic motion data (AMASS, KIT, GMR, KINESIS) have entries."""
    keys = {c.key for c in INTEGRATION_CITATIONS}
    assert {"musclemimic", *_DATA_SOURCES} <= keys


@pytest.mark.parametrize("citation", INTEGRATION_CITATIONS, ids=lambda c: c.key)
def test_bibtex_entries_are_well_formed(citation) -> None:
    entries = re.findall(r"@\w+\{([^,]+),", citation.bibtex)
    assert entries, citation.key
    assert citation.bibtex.count("{") == citation.bibtex.count("}")
    for field in ("title", "author", "year"):
        assert citation.bibtex.count(f"{field}=") >= len(entries), (citation.key, field)
    assert citation.project_url.startswith("https://")


def test_unknown_key_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="amass"):
        get_integration_citation("nope")
