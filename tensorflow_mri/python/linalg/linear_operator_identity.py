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
"""Scaled identity linear operator."""

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util


@api_util.export("linalg.LinearOperatorScaledIdentity")
@linear_operator.make_composite_tensor
class LinearOperatorScaledIdentity(linear_operator.LinearOperatorMixin,  # pylint: disable=abstract-method
                                   tf.linalg.LinearOperatorScaledIdentity):
  """Linear operator representing a scaled identity matrix.

  This operator acts like a scaled identity matrix $A = cI$.

  ```{note}
  This operator is a drop-in replacement of
  `tf.linalg.LinearOperatorScaledIdentity`, with extended functionality.
  ```

  Args:
    domain_shape: A 1D integer `Tensor`. The domain/range shape of the operator.
    multiplier: A `tf.Tensor` of arbitrary shape. Its shape will become the
      batch shape of the operator. Its dtype will determine the dtype of the
      operator.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form $x^H A x$ has positive real part for all
      nonzero $x$.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.  See:
      https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
    is_square:  Expect that this operator acts like square [batch] matrices.
    assert_proper_shapes: A boolean.  If `False`, only perform static
      checks that initialization and method arguments have proper shape.
      If `True`, and static checks are inconclusive, add asserts to the graph.
    name: A name for this `LinearOperator`.
  """
  def __init__(self,
               domain_shape,
               multiplier,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorScaledIdentity"):

    self._domain_shape_tensor_value = tensor_util.convert_shape_to_tensor(
        domain_shape, name="domain_shape")
    self._domain_shape_value = tf.TensorShape(tf.get_static_value(
        self._domain_shape_tensor_value))

    super().__init__(
        num_rows=tf.math.reduce_prod(domain_shape),
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

  @property
  def _composite_tensor_fields(self):
    return ("domain_shape", "multiplier", "assert_proper_shapes")

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("domain_shape",)
