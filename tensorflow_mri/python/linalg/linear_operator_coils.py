# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Non-uniform Fourier linear operators."""

import warnings

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_nd
from tensorflow_mri.python.linalg import linear_operator_util
from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util
from tensorflow_mri.python.util import types_util


@api_util.export("linalg.LinearOperatorCoils")
@linear_operator_nd.make_mri_operator_nd
class LinearOperatorCoils(linear_operator_nd.LinearOperatorND):
  """Linear operator acting like a [batch] coil array.

  Args:
    maps: A complex `tf.Tensor` of shape `[..., num_coils, *spatial_shape]`.
    batch_dims: An `int`, the number of batch dimensions in `maps`.
    is_non_singular: A boolean, or `None`. Whether this operator is expected
      to be non-singular. Defaults to `None`.
    is_self_adjoint: A boolean, or `None`. Whether this operator is expected
      to be equal to its Hermitian transpose. If `dtype` is real, this is
      equivalent to being symmetric. Defaults to `None`.
    is_positive_definite: A boolean, or `None`. Whether this operator is
      expected to be positive definite, meaning the quadratic form $x^H A x$
      has positive real part for all nonzero $x$. Note that an operator [does
      not need to be self-adjoint to be positive definite](https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices)
      Defaults to `None`.
    is_square: A boolean, or `None`. Expect that this operator acts like a
      square matrix (or a batch of square matrices). Defaults to `False`.
    name: An optional `str`. The name of this operator.
  """
  def __init__(self,
               maps,
               batch_dims=0,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorCoils"):
    parameters = dict(
        maps=maps,
        batch_dims=batch_dims,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    self._maps = tf.convert_to_tensor(maps, name="maps")
    self._batch_dims = int(batch_dims)

    if self._maps.dtype not in (tf.complex64, tf.complex128):
      raise ValueError(
          f"maps must be complex64 or complex128, got {str(self._maps.dtype)}")
    if self._maps.shape.rank is None:
      raise ValueError("maps must have known static rank")
    self._ndim_static = self._maps.shape.rank - self._batch_dims - 1
    if self._ndim_static < 1:
      raise ValueError(
          f"maps must have at least 2 dimensions (excluding batch dimensions), "
          f"got shape: {self._maps.shape}")
    self._coil_axis = -(self._ndim_static + 1)

    super().__init__(
        dtype=maps.dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        parameters=parameters,
        name=name)

  def _matvec_nd(self, x, adjoint=False):
    if adjoint:
      rhs = tf.math.reduce_sum(x * tf.math.conj(self._maps),
                               axis=self._coil_axis)
    else:
      rhs = tf.expand_dims(x, self._coil_axis) * self._maps
    return rhs

  def _ndim(self):
    return self._ndim_static

  def _domain_shape(self):
    return self._maps.shape[self._coil_axis + 1:]

  def _range_shape(self):
    return self._maps.shape[self._coil_axis:]

  def _batch_shape(self):
    return self._maps.shape[:self._coil_axis]

  def _domain_shape_tensor(self):
    return tf.shape(self._maps)[self._coil_axis + 1:]

  def _range_shape_tensor(self):
    return tf.shape(self._maps)[self._coil_axis:]

  def _batch_shape_tensor(self):
    return tf.shape(self._maps)[:self._coil_axis]

  @property
  def num_coils(self):
    return self._maps.shape[self._coil_axis]

  def num_coils_tensor(self):
    return tf.shape(self._maps)[self._coil_axis]


def coils_matrix(maps, batch_dims=0):
  """Constructs a coil array matrix.

  Args:
    maps: A complex `tf.Tensor` of shape `[..., num_coils, *spatial_shape]`.
    batch_dims: An `int`, the number of batch dimensions in `maps`.

  Returns:
    A `LinearOperatorCoils` instance.
  """
  maps = tf.convert_to_tensor(maps, name="maps")
  batch_dims = int(batch_dims)

  # Vectorize N-D maps.
  maps = tf.reshape(
      maps, tf.concat([tf.shape(maps)[:(batch_dims + 1)], [-1]], axis=0))

  # Construct a [batch] matrix for each coil.
  matrix = tf.linalg.diag(maps)

  # Stack the coil matrices.
  matrix = tf.reshape(matrix, tf.concat([tf.shape(maps)[:batch_dims],
                                         [-1, tf.shape(maps)[-1]]], axis=0))

  return matrix
