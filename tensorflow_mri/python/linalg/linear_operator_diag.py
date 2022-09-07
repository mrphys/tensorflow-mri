# Copyright 2021 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Diagonal linear operator."""

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util


@api_util.export("linalg.LinearOperatorDiag")
class LinearOperatorDiag(linear_operator.LinearOperatorMixin,  # pylint: disable=abstract-method
                         tf.linalg.LinearOperatorDiag):
  """Linear operator representing a square diagonal matrix.

  This operator acts like a [batch] diagonal matrix `A` with shape
  `[B1, ..., Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member. For every batch index `(i1, ..., ib)`, `A[i1, ..., ib, : :]` is
  an `N x N` matrix. This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  .. note:
    Similar to `tf.linalg.LinearOperatorDiag`_, but with imaging extensions.

  Args:
    diag: A `tf.Tensor` of shape `[B1, ..., Bb, *S]`.
    rank: An `int`. The rank of `S`. Must be <= `diag.shape.rank`.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose. If `diag` is real, this is auto-set to `True`.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form $x^H A x$ has positive real part for all
      nonzero $x$.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.

  .. _tf.linalg.LinearOperatorDiag: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperatorDiag
  """
  # pylint: disable=invalid-unary-operand-type
  def __init__(self,
               diag,
               rank,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name='LinearOperatorDiag'):
    # pylint: disable=invalid-unary-operand-type
    diag = tf.convert_to_tensor(diag, name='diag')
    self._rank = check_util.validate_rank(rank, name='rank', accept_none=False)
    if self._rank > diag.shape.rank:
      raise ValueError(
          f"Argument `rank` must be <= `diag.shape.rank`, but got: {rank}")

    self._shape_tensor_value = tf.shape(diag)
    self._shape_value = diag.shape
    batch_shape = self._shape_tensor_value[:-self._rank]

    super().__init__(
        diag=tf.reshape(diag, tf.concat([batch_shape, [-1]], 0)),
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)

  def _transform(self, x, adjoint=False):
    diag = tf.math.conj(self.diag) if adjoint else self.diag
    return tf.reshape(diag, self.domain_shape_tensor()) * x

  def _domain_shape(self):
    return self._shape_value[-self._rank:]

  def _range_shape(self):
    return self._shape_value[-self._rank:]

  def _batch_shape(self):
    return self._shape_value[:-self._rank]

  def _domain_shape_tensor(self):
    return self._shape_tensor_value[-self._rank:]

  def _range_shape_tensor(self):
    return self._shape_tensor_value[-self._rank:]

  def _batch_shape_tensor(self):
    return self._shape_tensor_value[:-self._rank]
