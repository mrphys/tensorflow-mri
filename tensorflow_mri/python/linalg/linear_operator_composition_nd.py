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
"""Composition of N-Dlinear operators."""

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.linalg import linear_operator_nd
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import doc_util


@api_util.export("linalg.LinearOperatorCompositionND")
@linear_operator_nd.make_linear_operator_nd
class LinearOperatorCompositionND(linear_operator_nd.LinearOperatorND):
  r"""Composes one or more N-dimensional `LinearOperator`s.

  This operator composes one or more linear operators `[op1,...,opJ]`,
  building a new `LinearOperator` with action defined by:

  ```
  op_composed(x) := op1(op2(...(opJ(x)...))
  ```

  If `opj` acts like [batch] matrix `Aj`, then `op_composed` acts like the
  [batch] matrix formed with the multiplication `A1 A2...AJ`.

  If `opj` has shape `batch_shape_j + [M_j, N_j]`, then we must have
  `N_j = M_{j+1}`, in which case the composed operator has shape equal to
  `broadcast_batch_shape + [M_1, N_J]`, where `broadcast_batch_shape` is the
  mutual broadcast of `batch_shape_j`, `j = 1,...,J`, assuming the intermediate
  batch shapes broadcast.  Even if the composed shape is well defined, the
  composed operator's methods may fail due to lack of broadcasting ability in
  the defining operators' methods.

  ```python
  # Create a 2 x 2 linear operator composed of two 2 x 2 operators.
  operator_1 = LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
  operator_2 = LinearOperatorFullMatrix([[1., 0.], [0., 1.]])
  operator = LinearOperatorComposition([operator_1, operator_2])

  operator.to_dense()
  ==> [[1., 2.]
       [3., 4.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 5 linear operators.
  matrix_45 = tf.random.normal(shape=[2, 3, 4, 5])
  operator_45 = LinearOperatorFullMatrix(matrix)

  # Create a [2, 3] batch of 5 x 6 linear operators.
  matrix_56 = tf.random.normal(shape=[2, 3, 5, 6])
  operator_56 = LinearOperatorFullMatrix(matrix_56)

  # Compose to create a [2, 3] batch of 4 x 6 operators.
  operator_46 = LinearOperatorComposition([operator_45, operator_56])

  # Create a shape [2, 3, 6, 2] vector.
  x = tf.random.normal(shape=[2, 3, 6, 2])
  operator.matmul(x)
  ==> Shape [2, 3, 4, 2] Tensor
  ```

  #### Performance

  The performance of `LinearOperatorComposition` on any operation is equal to
  the sum of the individual operators' operations.


  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.

  Args:
    operators: A `list` of `tfmri.linalg.LinearOperatorND` objects, each with
      the same dtype and conformable shapes.
    is_non_singular: A boolean, or `None`. Whether this operator is expected
      to be non-singular. Defaults to `True`.
    is_self_adjoint: A boolean, or `None`. Whether this operator is expected
      to be equal to its Hermitian transpose. If `dtype` is real, this is
      equivalent to being symmetric. Defaults to `False`.
    is_positive_definite: A boolean, or `None`. Whether this operator is
      expected to be positive definite, meaning the quadratic form $x^H A x$
      has positive real part for all nonzero $x$. Note that an operator [does
      not need to be self-adjoint to be positive definite](https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices)
      Defaults to `None`.
    is_square: A boolean, or `None`. Expect that this operator acts like a
      square matrix (or a batch of square matrices). Defaults to `True`.
    name: An optional `str`. The name of this operator.
  """
  def __init__(self,
               operators,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None):
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
            f"Expected all operators to have the same dtype. "
            f"Found: {', '.join(name_type)}")

    # Validate shapes.
    domain_shape = operators[0].domain_shape
    for operator in operators[1:]:
      if not domain_shape.is_compatible_with(operator.range_shape):
        shapes = ', '.join(
            [f'({str(o.range_shape)}, {str(o.domain_shape)})'
             for o in operators])
        raise ValueError(
            f"Expected operators to have conformable shapes for matrix "
            f"multiplication. Found: {shapes}")

    # Get broadcast batch shape (static).
    batch_shape_static = self.operators[0].batch_shape
    for operator in self.operators[1:]:
      batch_shape_static = tf.broadcast_static_shape(
          batch_shape_static, operator.batch_shape)
    self._batch_shape_static = batch_shape_static

    # Get broadcast batch shape (dynamic).
    batch_shape_dynamic = self.operators[0].batch_shape_tensor()
    for operator in self.operators[1:]:
      batch_shape_dynamic = tf.broadcast_dynamic_shape(
          batch_shape_dynamic, operator.batch_shape_tensor())
    self._batch_shape_dynamic = batch_shape_dynamic

    # Infer operator hints.
    is_non_singular = check_hint(
        combined_non_singular_hint(*operators),
        is_non_singular,
        "non-singular")
    is_self_adjoint = check_hint(
        combined_self_adjoint_hint(*operators),
        is_self_adjoint,
        "self-adjoint")
    is_positive_definite = check_hint(
        combined_positive_definite_hint(*operators),
        is_positive_definite,
        "positive-definite")
    is_square = check_hint(
        combined_square_hint(*operators),
        is_square,
        "square")

    # Initialization.
    if name is None:
      name = "_o_".join(operator.name for operator in operators)

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

  def _domain_shape(self):
    return self.operators[-1].domain_shape

  def _range_shape(self):
    return self.operators[0].range_shape

  def _batch_shape(self):
    return self._batch_shape_static

  def _domain_shape_tensor(self):
    return self.operators[-1].domain_shape_tensor()

  def _range_shape_tensor(self):
    return self.oeprators[0].range_shape_tensor()

  def _batch_shape_tensor(self):
    return self._batch_shape_dynamic

  def _matvec_nd(self, x, adjoint=False):
    # If self.operators = [A, B], and not adjoint, then
    # matmul_order_list = [B, A].
    # As a result, we return A.matmul(B.matmul(x))
    if adjoint:
      matmul_order_list = self.operators
    else:
      matmul_order_list = list(reversed(self.operators))

    result = matmul_order_list[0].matvec_nd(x, adjoint=adjoint)
    for operator in matmul_order_list[1:]:
      result = operator.matvec_nd(result, adjoint=adjoint)
    return result

  def _solvevec_nd(self, rhs, adjoint=False):
    # If self.operators = [A, B], and not adjoint, then
    # solve_order_list = [A, B].
    # As a result, we return B.solve(A.solve(x))
    if adjoint:
      solve_order_list = list(reversed(self.operators))
    else:
      solve_order_list = self.operators

    solution = solve_order_list[0].solvevec_nd(rhs, adjoint=adjoint)
    for operator in solve_order_list[1:]:
      solution = operator.solvevec_nd(solution, adjoint=adjoint)
    return solution

  @property
  def _composite_tensor_fields(self):
    return ("operators",)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {"operators": [0] * len(self.operators)}
