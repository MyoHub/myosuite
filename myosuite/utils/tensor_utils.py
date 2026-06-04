# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from: https://github.com/aravindr93/mjrl
"""Dictionary-of-tensors stack / concat / split utilities used by the logger."""

from __future__ import annotations

import numpy as np


def stack_tensor_dict_list(tensor_dict_list: list[dict]) -> dict:
    """Stack a list of dicts of arrays into a dict of stacked arrays.

    Args:
        tensor_dict_list: Non-empty list of dicts with identical key sets.
            Values may be arrays or nested dicts.

    Returns:
        A dict with the same keys; each value is the result of
        ``np.array([x[k] for x in tensor_dict_list])``.
    """
    keys = list(tensor_dict_list[0].keys())
    ret = {}
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            ret[k] = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            ret[k] = np.array([x[k] for x in tensor_dict_list])
    return ret


def concat_tensor_dict_list(tensor_dict_list: list[dict]) -> dict:
    """Concatenate a list of dicts of arrays along axis 0.

    Args:
        tensor_dict_list: Non-empty list of dicts with identical key sets.

    Returns:
        A dict where each value is ``np.concatenate([x[k] ...], axis=0)``.
    """
    keys = list(tensor_dict_list[0].keys())
    ret = {}
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            ret[k] = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            ret[k] = np.concatenate([x[k] for x in tensor_dict_list], axis=0)
    return ret


def split_tensor_dict_list(tensor_dict: dict) -> list[dict]:
    """Inverse of :func:`stack_tensor_dict_list`.

    Splits a dict of arrays (first axis = time/batch) into a list of
    single-step dicts.

    Args:
        tensor_dict: Dict whose values are arrays or nested dicts.

    Returns:
        A list of dicts, one per element along the first axis.
    """
    keys = list(tensor_dict.keys())
    ret: list[dict] | None = None
    for k in keys:
        vals = tensor_dict[k]
        if isinstance(vals, dict):
            vals = split_tensor_dict_list(vals)
        if ret is None:
            ret = [{k: v} for v in vals]
        else:
            for v, cur_dict in zip(vals, ret):
                cur_dict[k] = v
    return ret or []
