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
"""Scaled identity linear operator."""

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util


@api_util.export("linalg.LinearOperatorScaledIdentity")
class LinearOperatorScaledIdentity(linear_operator.LinearOperatorMixin,  # pylint: disable=abstract-method
                                   tf.linalg.LinearOperatorScaledIdentity):
  """Linear operator representing a scaled identity matrix.

  .. note:
    Similar to `tf.linalg.LinearOperatorScaledIdentity`_, but with imaging
    extensions.

  Args:
    shape: Non-negative integer `Tensor`. The shape of the operator.
    multiplier: A `Tensor` of shape `[B1, ..., Bb]`, or `[]` (a scalar).
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form `x^H A x` has positive real part for all
      nonzero `x`.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.  See:
      https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
    is_square:  Expect that this operator acts like square [batch] matrices.
    assert_proper_shapes: Python `bool`.  If `False`, only perform static
      checks that initialization and method arguments have proper shape.
      If `True`, and static checks are inconclusive, add asserts to the graph.
    name: A name for this `LinearOperator`.

  .. _tf.linalg.LinearOperatorScaledIdentity: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperatorScaledIdentity
  """
  def __init__(self,
               shape,
               multiplier,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorScaledIdentity"):

    self._domain_shape_tensor_value = tensor_util.convert_shape_to_tensor(
        shape, name="shape")
    self._domain_shape_value = tf.TensorShape(tf.get_static_value(
        self._domain_shape_tensor_value))

    super().__init__(
        num_rows=tf.math.reduce_prod(shape),
        multiplier=multiplier,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        assert_proper_shapes=assert_proper_shapes,
        name=name)

  def _transform(self, x, adjoint=False):
    domain_rank = tf.size(self.domain_shape_tensor())
    multiplier_shape = tf.concat([
        tf.shape(self.multiplier),
        tf.ones((domain_rank,), dtype=tf.int32)], 0)
    multiplier_matrix = tf.reshape(self.multiplier, multiplier_shape)
    if adjoint:
      multiplier_matrix = tf.math.conj(multiplier_matrix)
    return x * multiplier_matrix

  def _domain_shape(self):
    return self._domain_shape_value

  def _range_shape(self):
    return self._domain_shape_value

  def _batch_shape(self):
    return self.multiplier.shape

  def _domain_shape_tensor(self):
    return self._domain_shape_tensor_value

  def _range_shape_tensor(self):
    return self._domain_shape_tensor_value

  def _batch_shape_tensor(self):
    return tf.shape(self.multiplier)
