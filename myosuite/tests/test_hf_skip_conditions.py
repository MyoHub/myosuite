# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``skip_if_hf_gated`` skips for missing access or a Hub outage, not for a wrong path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from myosuite.envs.myo.tasks.mimic.tracking import MissingMotionData
from myosuite.tests.support.optional_deps import skip_if_hf_gated

pytestmark = pytest.mark.tier1

errors = pytest.importorskip("huggingface_hub.errors")


def _http_error(cls: type[Exception], status: int) -> Exception:
    response = SimpleNamespace(status_code=status, headers={}, request=None)
    return cls("message", response=response)


@pytest.mark.parametrize(
    "exc",
    [
        _http_error(errors.GatedRepoError, 401),
        _http_error(errors.HfHubHTTPError, 429),
        _http_error(errors.HfHubHTTPError, 503),
        MissingMotionData("no clips"),
    ],
    ids=["gated", "rate-limit", "server-error", "missing-clips"],
)
def test_unavailable_data_is_skipped(exc: Exception) -> None:
    with pytest.raises(pytest.skip.Exception):
        skip_if_hf_gated("env", exc)


@pytest.mark.parametrize(
    "exc",
    [
        _http_error(errors.RepositoryNotFoundError, 401),
        _http_error(errors.RevisionNotFoundError, 404),
        _http_error(errors.HfHubHTTPError, 404),
        AssertionError("wrong observation keys"),
        FileNotFoundError("model.xml"),
    ],
    ids=["wrong-repo", "wrong-revision", "wrong-file", "assertion", "missing-file"],
)
def test_a_real_failure_is_not_skipped(exc: Exception) -> None:
    try:
        skip_if_hf_gated("env", exc)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"a real failure was skipped: {skipped}")
