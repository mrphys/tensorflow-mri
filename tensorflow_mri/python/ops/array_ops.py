# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Array manipulation operations."""

import tensorflow as tf


def ravel_multi_index(multi_index, dims):
  """Converts an array of multi-indices into an array of flat indices.

  Args:
    multi_index: A `Tensor` of shape `[..., N]` containing multi-indices into
      an `N`-dimensional tensor.
    dims: A `Tensor` of shape `[N]`. The shape of the tensor that `multi_index`
      indexes into.

  Returns:
    A `Tensor` of shape `[...]` containing flat indices equivalent to
    `multi_index`.
  """
  strides = tf.math.cumprod(dims, exclusive=True, reverse=True) # pylint:disable=no-value-for-parameter
  return tf.math.reduce_sum(multi_index * strides, axis=-1)


def meshgrid(*args):
  """Return coordinate matrices from coordinate vectors.

  Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
  fields over N-D grids, given one-dimensional coordinate arrays `x1, x2,…, xn`.

  .. note::
    Similar to `tf.meshgrid`, but uses matrix indexing and returns a stacked
    tensor (along axis -1) instead of a list of tensors.

  Args:
    *args: `Tensors` with rank 1.
  
  Returns:
    A `Tensor` of shape `[M1, M2, ..., Mn, N]`, where `N` is the number of
    tensors in `args` and `Mi = tf.size(args[i])`.
  """
  return tf.stack(tf.meshgrid(*args, indexing='ij'), axis=-1)


def cartesian_product(*args):
  """Cartesian product of input tensors.

  Args:
    *args: `Tensors` with rank 1.

  Returns:
    A `Tensor` of shape `[M, N]`, where `N` is the number of tensors in `args`
    and `M` is the product of the sizes of all the tensors in `args`.
  """
  return tf.reshape(meshgrid(*args), [-1, len(args)])
