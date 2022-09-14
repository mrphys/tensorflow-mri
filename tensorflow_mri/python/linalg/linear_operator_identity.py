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
from tensorflow_mri.python.linalg import linear_operator_algebra
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util
from tensorflow_mri.python.util import types_util


@api_util.export("linalg.LinearOperatorIdentity")
@linear_operator.make_composite_tensor
class LinearOperatorIdentity(linear_operator.LinearOperatorMixin,
                             tf.linalg.LinearOperatorIdentity):
  """Linear operator representing an identity matrix.

  This operator acts like the identity matrix $A = I$ (or a batch of identity
  matrices).

  ```{note}
  This operator is similar to `tf.linalg.LinearOperatorIdentity`, but
  provides additional functionality. See the
  [linear algebra guide](https://mrphys.github.io/tensorflow-mri/guide/linalg/)
  for more details.
  ```

  ```{seealso}
  The scaled identity operator `tfmri.linalg.LinearOperatorScaledIdentity`.
  ```

  Args:
    domain_shape: A 1D integer `tf.Tensor`. The domain/range shape of the
      operator.
    batch_shape: An optional 1D integer `tf.Tensor`. The shape of the leading
      batch dimensions. If `None`, this operator has no leading batch
      dimensions.
    dtype: A `tf.dtypes.DType`. The data type of the matrix that this operator
      represents. Defaults to `float32`.
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
               batch_shape=None,
               dtype=None,
               is_non_singular=True,
               is_self_adjoint=True,
               is_positive_definite=True,
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorIdentity"):
    # Shape inputs must not have reference semantics.
    types_util.assert_not_ref_type(domain_shape, "domain_shape")
    types_util.assert_not_ref_type(batch_shape, "batch_shape")

    # Parse domain shape.
    self._domain_shape_static, self._domain_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(
            domain_shape,
            assert_proper_shape=assert_proper_shapes,
            arg_name='domain_shape'))

    # Parse batch shape.
    if batch_shape is not None:
      # Extra underscore at the end to distinguish from base class property of
      # the same name.
      self._batch_shape_static_, self._batch_shape_dynamic = (
          tensor_util.static_and_dynamic_shapes_from_shape(
              batch_shape,
              assert_proper_shape=assert_proper_shapes,
              arg_name='batch_shape'))
    else:
      self._batch_shape_static_ = tf.TensorShape([])
      self._batch_shape_dynamic = tf.constant([], dtype=tf.int32)

    super().__init__(num_rows=tf.math.reduce_prod(domain_shape),
                     batch_shape=batch_shape,
                     dtype=dtype,
                     is_non_singular=is_non_singular,
                     is_self_adjoint=is_self_adjoint,
                     is_positive_definite=is_positive_definite,
                     is_square=is_square,
                     assert_proper_shapes=assert_proper_shapes,
                     name=name)

  def _transform(self, x, adjoint=False):
    if self.domain_shape.rank is not None:
      rank = self.domain_shape.rank
    else:
      rank = tf.size(self.domain_shape_tensor())
    batch_shape = tf.broadcast_dynamic_shape(
        tf.shape(x)[:-rank], self.batch_shape_tensor())
    output_shape = tf.concat([batch_shape, self.domain_shape_tensor()], axis=0)
    return tf.broadcast_to(x, output_shape)

  def _domain_shape(self):
    return self._domain_shape_static

  def _range_shape(self):
    return self._domain_shape_static

  def _batch_shape(self):
    return self._batch_shape_static_

  def _domain_shape_tensor(self):
    return self._domain_shape_dynamic

  def _range_shape_tensor(self):
    return self._domain_shape_dynamic

  def _batch_shape_tensor(self):
    return self._batch_shape_dynamic

  @property
  def _composite_tensor_fields(self):
    return ("domain_shape", "batch_shape", "dtype", "assert_proper_shapes")

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("domain_shape", "batch_shape")


@api_util.export("linalg.LinearOperatorScaledIdentity")
@linear_operator.make_composite_tensor
class LinearOperatorScaledIdentity(linear_operator.LinearOperatorMixin,  # pylint: disable=abstract-method
                                   tf.linalg.LinearOperatorScaledIdentity):
  """Linear operator representing a scaled identity matrix.

  This operator acts like a scaled identity matrix $A = cI$ (or a batch of
  scaled identity matrices).

  ```{note}
  This operator is similar to `tf.linalg.LinearOperatorScaledIdentity`, but
  provides additional functionality. See the
  [linear algebra guide](https://mrphys.github.io/tensorflow-mri/guide/linalg/)
  for more details.
  ```

  ```{seealso}
  The identity operator `tfmri.linalg.LinearOperatorIdentity`.
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
    # Shape inputs must not have reference semantics.
    types_util.assert_not_ref_type(domain_shape, "domain_shape")

    # Parse domain shape.
    self._domain_shape_static, self._domain_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(
            domain_shape,
            assert_proper_shape=assert_proper_shapes,
            arg_name='domain_shape'))

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
    return self._domain_shape_static

  def _range_shape(self):
    return self._domain_shape_static

  def _batch_shape(self):
    return self.multiplier.shape

  def _domain_shape_tensor(self):
    return self._domain_shape_dynamic

  def _range_shape_tensor(self):
    return self._domain_shape_dynamic

  def _batch_shape_tensor(self):
    return tf.shape(self.multiplier)

  @property
  def _composite_tensor_fields(self):
    return ("domain_shape", "multiplier", "assert_proper_shapes")

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("domain_shape",)


@linear_operator_algebra.RegisterAdjoint(LinearOperatorIdentity)
def adjoint_identity(identity_operator):
  return identity_operator


@linear_operator_algebra.RegisterAdjoint(LinearOperatorScaledIdentity)
def adjoint_scaled_identity(identity_operator):
  multiplier = identity_operator.multiplier
  if multiplier.dtype.is_complex:
    multiplier = tf.math.conj(multiplier)

  return LinearOperatorScaledIdentity(
      domain_shape=identity_operator.domain_shape_tensor(),
      multiplier=multiplier,
      is_non_singular=identity_operator.is_non_singular,
      is_self_adjoint=identity_operator.is_self_adjoint,
      is_positive_definite=identity_operator.is_positive_definite,
      is_square=True)


@linear_operator_algebra.RegisterInverse(LinearOperatorIdentity)
def inverse_identity(identity_operator):
  return identity_operator


@linear_operator_algebra.RegisterInverse(LinearOperatorScaledIdentity)
def inverse_scaled_identity(identity_operator):
  return LinearOperatorScaledIdentity(
      domain_shape=identity_operator.domain_shape_tensor(),
      multiplier=1. / identity_operator.multiplier,
      is_non_singular=identity_operator.is_non_singular,
      is_self_adjoint=True,
      is_positive_definite=identity_operator.is_positive_definite,
      is_square=True)
