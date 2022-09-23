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
"""Addition linear operator."""

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util


@api_util.export("linalg.LinearOperatorAddition")
@linear_operator.make_linear_operator
class LinearOperatorAddition(linear_operator.LinearOperator):
  r"""Adds one or more [batch] linear operators.

  This operator adds one or more linear operators $A_1, A_2, \dots, A_n$ to
  build a new `LinearOperator` $A$ with action defined by:

  $$
  Ax = (A_1 + A_2 + \dots + A_n)(x) = A_1 x + A_2 x + \dots + A_n x
  $$

  All input `operators` must have shape `[..., M, N]` and the resulting
  operator also has shape `[..., M, N]`. The batch shape of the resulting
  operator is the result of broadcasting the batch shape of all input
  operators.

  ```{rubric} Performance
  ```
  In general, performance in matrix-vector multiplication is the sum
  of the individual operators. More efficient implementations may be
  used for specific operators.

  ```{rubric} Matrix properties
  ```
  The properties of this operator are determined by the properties of the
  input operators.

  ```{rubric} Inversion
  ```
  At present, this operator does not implement an efficient algorithm for
  inversion. `solve` and `lstsq` will trigger conversion to a dense matrix.

  Example:
    >>> # Create a 2 x 2 linear operator composed of two 2 x 2 operators.
    >>> op1 = tfmri.linalg.LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
    >>> op2 = tfmri.linalg.LinearOperatorIdentity(2)
    >>> operator = LinearOperatorAddition([op1, op2])
    >>> operator.to_dense().numpy()
    array([[2., 2.],
           [3., 5.]], dtype=float32)

  Args:
    operators: A `list` of `tf.linalg.LinearOperator`s of equal shape and
      dtype. Batch dimensions may vary but must be broadcastable.
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
      square matrix (or a batch of square matrices). Defaults to `None`.
    name: An optional `str`. The name of this operator.

  Raises:
    TypeError: If all operators do not have the same `dtype`.
    ValueError: If `operators` is empty.
  """
  def __init__(self,
               operators,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None):
    """Initialize a `LinearOperatorAddition`."""
    parameters = dict(
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)

    # Validate operators.
    tf.debugging.assert_proper_iterable(operators)
    operators = list(operators)
    if not operators:
      raise ValueError(
          f"Expected a non-empty list of operators. Found: {operators}")
    self._operators = operators

    # Validate dtype.
    dtype = operators[0].dtype
    for operator in operators:
      if operator.dtype != dtype:
        name_type = (str((o.name, o.dtype)) for o in operators)
        raise TypeError(
            "Expected all operators to have the same dtype. Found %s"
            % "   ".join(name_type))

    # Infer operator properties.
    if is_self_adjoint is None:
      # If all operators are self-adjoint, so is the sum.
      if all(operator.is_self_adjoint for operator in operators):
        is_self_adjoint = True
    if is_positive_definite is None:
      # If all operators are positive definite, so is the sum.
      if all(operator.is_positive_definite for operator in operators):
        is_positive_definite = True
    if is_non_singular is None:
      # A positive definite operator is always non-singular.
      if is_positive_definite:
        is_non_singular = True
    if is_square is None:
      # If all operators are square, so is the sum.
      if all(operator.is_square for operator in operators):
        is_square=True

    if name is None:
      name = "_p_".join(operator.name for operator in operators)
    with tf.name_scope(name):
      super().__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  @property
  def operators(self):
    return self._operators

  def _shape(self):
    # Get final matrix shape.
    domain_dimension = self.operators[0].domain_dimension
    range_dimension = self.operators[1].range_dimension
    for operator in self.operators[1:]:
      domain_dimension.assert_is_compatible_with(operator.domain_dimension)
      range_dimension.assert_is_compatible_with(operator.range_dimension)

    matrix_shape = tf.TensorShape([range_dimension, domain_dimension])

    # Get broadcast batch shape.
    # tf.broadcast_static_shape checks for compatibility.
    batch_shape = self.operators[0].batch_shape
    for operator in self.operators[1:]:
      batch_shape = tf.broadcast_static_shape(
          batch_shape, operator.batch_shape)

    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    # Avoid messy broadcasting if possible.
    if self.shape.is_fully_defined():
      return tf.convert_to_tensor(
          self.shape.as_list(), dtype=tf.dtypes.int32, name="shape")

    # Don't check the matrix dimensions.  That would add unnecessary Asserts to
    # the graph.  Things will fail at runtime naturally if shapes are
    # incompatible.
    matrix_shape = tf.stack([
        self.operators[0].range_dimension_tensor(),
        self.operators[0].domain_dimension_tensor()
    ])

    # Dummy Tensor of zeros.  Will never be materialized.
    zeros = tf.zeros(shape=self.operators[0].batch_shape_tensor())
    for operator in self.operators[1:]:
      zeros += tf.zeros(shape=operator.batch_shape_tensor())
    batch_shape = tf.shape(zeros)

    return tf.concat((batch_shape, matrix_shape), 0)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    result = self.operators[0].matmul(
        x, adjoint=adjoint, adjoint_arg=adjoint_arg)
    for operator in self.operators[1:]:
      result += operator.matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
    return result

  @property
  def _composite_tensor_fields(self):
    return ("operators",)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {"operators": [0] * len(self.operators)}
