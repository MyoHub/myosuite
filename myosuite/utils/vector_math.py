""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import numpy as np

def calculate_cosine(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Calculates the cosine angle between two vectors.

    This computes cos(theta) = dot(v1, v2) / (norm(v1) * norm(v2))

    Args:
        vec1: The first vector. This can have a batch dimension.
        vec2: The second vector. This can have a batch dimension.

    Returns:
        The cosine angle between the two vectors, with the same batch dimension
        as the given vectors.
    """
    if np.shape(vec1) != np.shape(vec2):
        raise ValueError('{} must have the same shape as {}'.format(vec1, vec2))
    ndim = np.ndim(vec1)
    norm_product = (np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1))
    zero_norms = norm_product == 0
    if np.any(zero_norms):
        if ndim>1:
            norm_product[zero_norms] = 1
        else:
            norm_product = 1
    # Return the batched dot product.
    return np.einsum('...i,...i', vec1, vec2) / norm_product