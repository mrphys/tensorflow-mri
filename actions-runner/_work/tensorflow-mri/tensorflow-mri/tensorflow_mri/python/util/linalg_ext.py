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
"""Additional linear algebra operators."""

import tensorflow as tf

from tensorflow_mri.python.util import check_util


class LinearOperatorAddition(tf.linalg.LinearOperator):  # pylint: disable=abstract-method
  """Adds one or more `LinearOperators`.

  This operator adds one or more linear operators `op1 + op2 + ... + opJ`,
  building a new `LinearOperator` with action defined by:

  ```
  op_addition(x) := op1(x) + op2(x) + op3(x)
  ```

  If `opj` acts like [batch] matrix `Aj`, then `op_addition` acts like the
  [batch] matrix formed with the addition `A1 + A2 + ... + AJ`.

  If each `opj` has shape `batch_shape_j + [M, N]`, then the addition operator
  has shape equal to `broadcast_batch_shape + [M, N]`, where
  `broadcast_batch_shape` is the mutual broadcast of `batch_shape_j`,
  `j = 1, ..., J`, assuming the intermediate batch shapes broadcast.

  ```python
  # Create a 2 x 2 linear operator composed of two 2 x 2 operators.
  operator_1 = LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
  operator_2 = LinearOperatorFullMatrix([[1., 0.], [0., 1.]])
  operator = LinearOperatorAddition([operator_1, operator_2])

  operator.to_dense()
  ==> [[2., 2.]
       [3., 5.]]

  operator.shape
  ==> [2, 2]
  ```

  #### Performance

  The performance of `LinearOperatorAddition` on any operation is equal to
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
    operators: Iterable of `LinearOperator` objects, each with
      the same shape and dtype.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form `x^H A x` has positive real part for all
      nonzero `x`.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.  See:
      https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
    is_square:  Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.  Default is the individual
      operators names joined with `_p_`.

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
          "Expected a non-empty list of operators. Found: %s" % operators)
    self._operators = operators

    # Validate dtype.
    dtype = operators[0].dtype
    for operator in operators:
      if operator.dtype != dtype:
        name_type = (str((o.name, o.dtype)) for o in operators)
        raise TypeError(
            "Expected all operators to have the same dtype.  Found %s"
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


class _LinearOperatorStackBase(tf.linalg.LinearOperator):  # pylint: disable=abstract-method
  """Base class for `LinearOperator` stacks."""
  def __init__(self,
               operators,
               axis,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name=None):
    parameters = dict(
        operators=operators,
        axis=axis,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    # Validate operators.
    tf.debugging.assert_proper_iterable(operators)
    operators = list(operators)
    if not operators:
      raise ValueError(
          "Expected a non-empty list of operators. Found: %s" % operators)
    self._operators = operators

    # Validate axis.
    self._axis = check_util.validate_enum(
        axis, {'vertical', 'horizontal'}, 'axis')

    # Define diagonal operators, for functions that are shared across blockwise
    # `LinearOperator` types.
    self._diagonal_operators = operators

    # Validate dtype.
    dtype = operators[0].dtype
    for operator in operators:
      if operator.dtype != dtype:
        name_type = (str((o.name, o.dtype)) for o in operators)
        raise TypeError(
            "Expected all operators to have the same dtype.  Found %s"
            % "   ".join(name_type))

    # Validate shapes.
    fixed_axis = -1 if self._axis == 'vertical' else -2
    shape = operators[0].shape
    for operator in operators:
      if operator.shape[fixed_axis] != shape[fixed_axis]:
        raise ValueError(
            "Expected all operators to have the same size at dimension %s.  "
            "Found %s" % (self._axis, [o.shape for o in operators]))

    if name is None:
      # Using s to mean stack.
      name = "_s_".join(operator.name for operator in operators)
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

  @property
  def axis(self):
    return self._axis

  def _shape(self):
    # Get final matrix shape.
    if self.axis == 'vertical':
      domain_dimension = self.operators[0].domain_dimension
      range_dimension = sum(op.range_dimension for op in self.operators)
    else:
      domain_dimension = sum(op.domain_dimension for op in self.operators)
      range_dimension = self.operators[0].range_dimension

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

    if self.axis == 'vertical':
      domain_dimension = self.operators[0].domain_dimension_tensor()
      range_dimension = sum(
          op.range_dimension_tensor() for op in self.operators)
    else:
      domain_dimension = sum(
          op.domain_dimension_tensor() for op in self.operators)
      range_dimension = self.operators[0].range_dimension_tensor()

    matrix_shape = tf.stack([range_dimension, domain_dimension])

    # Dummy Tensor of zeros.  Will never be materialized.
    zeros = tf.zeros(shape=self.operators[0].batch_shape_tensor())
    for operator in self.operators[1:]:
      zeros += tf.zeros(shape=operator.batch_shape_tensor())
    batch_shape = tf.shape(zeros)

    return tf.concat((batch_shape, matrix_shape), 0)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # H-stack or V-stack?
    is_hstack = self.axis == 'horizontal'
    # Exclusive OR logical operator.
    xor = lambda a, b: bool(a) ^ bool(b)

    # Apply each operator, then stack the results.
    if xor(adjoint, is_hstack):  # Adjoint V-stack OR non-adjoint H-stack.
      if is_hstack:  # H-stack operator.
        size_splits = [op.domain_dimension_tensor() for op in self.operators]
      else:  # Adjoint V-stack operator.
        size_splits = [op.range_dimension_tensor() for op in self.operators]
      tensors = tf.split(x, size_splits, axis=-2)  # pylint: disable=redundant-keyword-arg,no-value-for-parameter
      results = [op.matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
                 for op, x in zip(self.operators, tensors)]
      return tf.math.reduce_sum(tf.stack(results, axis=-1), axis=-1)

    # Adjoint H-stack OR non-adjoint V-stack.
    results = [op.matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
                for op in self.operators]
    return tf.concat(results, -2)

  @property
  def _composite_tensor_fields(self):
    return ("operators",)


class LinearOperatorVerticalStack(_LinearOperatorStackBase):  # pylint: disable=abstract-method
  """Stacks one or more `LinearOperators` vertically.

  This operator combines one or more linear operators `[op1, ..., opJ]`,
  building a new `LinearOperator`, whose underlying matrix representation
  has all the operators stacked vertically.

  #### Shape compatibility

  If `opj` acts like a [batch] matrix `Aj`, then `op_combined` acts like
  the [batch] matrix formed by stacking each matrix `Aj`.

  Each `opj` is required to represent a matrix, and hence will have
  shape `batch_shape_j + [M_j, N_j]`.

  If `opj` has shape `batch_shape_j + [M_j, N_j]`, then the combined operator
  has shape `broadcast_batch_shape + [sum M_j, N_j]`, where
  `broadcast_batch_shape` is the mutual broadcast of `batch_shape_j`,
  `j = 1,...,J`, assuming the intermediate batch shapes broadcast.

  Arguments to `matmul`, `matvec`, `solve`, and `solvevec` may either be single
  tensors or lists of tensors that are interpreted as blocks. The `j`th
  element of a blockwise list of tensors must have dimensions that match
  `opj` for the given method. If a list of blocks is input, then a list of
  blocks is returned as well.

  #### Performance

  The performance of `LinearOperatorVerticalStack` on any operation is equal to
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
    operators: Iterable of `LinearOperator` objects, each with
      the same `dtype` and stackable shape. For vertical stacking, all
      operators must have the same domain dimension. For horizontal stacking,
      all operator must have the same range dimension.
    is_non_singular:  Expect that this operator is non-singular.
    is_self_adjoint:  Expect that this operator is equal to its hermitian
      transpose.
    is_positive_definite:  Expect that this operator is positive definite,
      meaning the quadratic form `x^H A x` has positive real part for all
      nonzero `x`.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.  See:
      https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
    is_square:  Expect that this operator acts like square [batch] matrices.
      This is true by default, and will raise a `ValueError` otherwise.
    name: A name for this `LinearOperator`.  Default is the individual
      operators names joined with `_o_`.

  Raises:
    TypeError: If all operators do not have the same `dtype`.
    ValueError: If `operators` is empty.
  """
  def __init__(self,
               operators,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name=None):
    super().__init__(operators, 'vertical',
                     is_non_singular=is_non_singular,
                     is_self_adjoint=is_self_adjoint,
                     is_positive_definite=is_positive_definite,
                     is_square=is_square,
                     name=name)


class LinearOperatorHorizontalStack(_LinearOperatorStackBase):  # pylint: disable=abstract-method
  """Stacks one or more `LinearOperators` horizontally.

  This operator combines one or more linear operators `[op1, ..., opJ]`,
  building a new `LinearOperator`, whose underlying matrix representation
  has all the operators stacked horizontally.

  #### Shape compatibility

  If `opj` acts like a [batch] matrix `Aj`, then `op_combined` acts like
  the [batch] matrix formed by stacking each matrix `Aj`.

  Each `opj` is required to represent a matrix, and hence will have
  shape `batch_shape_j + [M_j, N_j]`.

  If `opj` has shape `batch_shape_j + [M_j, N_j]`, then the combined operator
  has shape `broadcast_batch_shape + [M_j, sum N_j]`, where
  `broadcast_batch_shape` is the mutual broadcast of `batch_shape_j`,
  `j = 1,...,J`, assuming the intermediate batch shapes broadcast.

  Arguments to `matmul`, `matvec`, `solve`, and `solvevec` may either be single
  tensors or lists of tensors that are interpreted as blocks. The `j`th
  element of a blockwise list of tensors must have dimensions that match
  `opj` for the given method. If a list of blocks is input, then a list of
  blocks is returned as well.

  #### Performance

  The performance of `LinearOperatorHorizontalStack` on any operation is equal
  to the sum of the individual operators' operations.


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
    operators: Iterable of `LinearOperator` objects, each with
      the same `dtype` and stackable shape. For vertical stacking, all
      operators must have the same domain dimension. For horizontal stacking,
      all operator must have the same range dimension.
    is_non_singular:  Expect that this operator is non-singular.
    is_self_adjoint:  Expect that this operator is equal to its hermitian
      transpose.
    is_positive_definite:  Expect that this operator is positive definite,
      meaning the quadratic form `x^H A x` has positive real part for all
      nonzero `x`.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.  See:
      https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
    is_square:  Expect that this operator acts like square [batch] matrices.
      This is true by default, and will raise a `ValueError` otherwise.
    name: A name for this `LinearOperator`.  Default is the individual
      operators names joined with `_o_`.

  Raises:
    TypeError: If all operators do not have the same `dtype`.
    ValueError: If `operators` is empty.
  """
  def __init__(self,
               operators,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name=None):
    super().__init__(operators, 'horizontal',
                     is_non_singular=is_non_singular,
                     is_self_adjoint=is_self_adjoint,
                     is_positive_definite=is_positive_definite,
                     is_square=is_square,
                     name=name)


class LinearOperatorFiniteDifference(tf.linalg.LinearOperator):  # pylint: disable=abstract-method
  """Linear operator acting like a difference operator.

  Args:
    domain_dimension: An `int`. The domain dimension of the operator.
    dtype: An optional `string` or `DType`. The data type for this operator.
      Defaults to `float32`.
    name: An optional `string`. A name for this operator.
  """
  def __init__(self,
               domain_dimension,
               dtype=tf.dtypes.float32,
               name="LinearOperatorFiniteDifference"):

    parameters = dict(
      domain_dimension=domain_dimension,
      dtype=dtype,
      name=name
    )

    self._domain_dimension_value = domain_dimension
    self._range_dimension_value = domain_dimension - 1

    super().__init__(dtype,
                     is_non_singular=None,
                     is_self_adjoint=None,
                     is_positive_definite=None,
                     is_square=None,
                     name=name,
                     parameters=parameters)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    if adjoint_arg:
      x = tf.linalg.adjoint(x)
    if adjoint:
      paddings1 = [[1, 0], [0, 0]]
      paddings2 = [[0, 1], [0, 0]]
      x1 = tf.pad(x, paddings1)  # pylint: disable=no-value-for-parameter
      x2 = tf.pad(x, paddings2)  # pylint: disable=no-value-for-parameter
      x = x1 - x2
    else:
      slice1 = [slice(1, None), slice(None)]
      slice2 = [slice(None, -1), slice(None)]
      x1 = x[tuple(slice1)]
      x2 = x[tuple(slice2)]
      x = x1 - x2
    return x

  def _shape(self):
    return tf.TensorShape(
        [self._range_dimension_value, self._domain_dimension_value])
