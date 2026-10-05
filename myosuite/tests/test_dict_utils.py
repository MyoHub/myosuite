# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""dict_numpify resolutions (myosuite/utils/dict_utils.py)."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.utils.dict_utils import dict_numpify

pytestmark = pytest.mark.tier1


def test_nested_dicts_use_the_requested_resolutions() -> None:
    """Nested dicts got (i_res, f_res, float16) as (u_res, i_res, f_res)."""
    data = {
        "f": np.array([1.0001]),
        "a": {
            "f": np.array([1.0001]),
            "i": np.array([300]),
            "u": np.array([3], dtype=np.uint16),
        },
    }
    out = dict_numpify(data, u_res=np.uint32, i_res=np.int32, f_res=np.float64)
    assert out["f"].dtype == np.float64
    assert out["a"]["f"].dtype == np.float64 and out["a"]["f"][0] == 1.0001
    assert out["a"]["i"].dtype == np.int32 and out["a"]["i"][0] == 300
    assert out["a"]["u"].dtype == np.uint32


def test_object_arrays_use_the_float_resolution() -> None:
    out = dict_numpify({"x": np.array([None, 1.0001], dtype=object)}, f_res=np.float64)
    assert out["x"].dtype == np.float64
    assert np.isnan(out["x"][0]) and out["x"][1] == 1.0001
