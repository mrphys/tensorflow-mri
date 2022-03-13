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
"""Linear algebra for images.

Contains the imaging mixin and imaging extensions of basic linear operators.
"""

import abc

import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import linalg_ext
from tensorflow_mri.python.util import tensor_util


class LinalgImagingMixin(tf.linalg.LinearOperator):
  """Mixin for linear operators meant to operate on images."""
  def transform(self, x, adjoint=False, name="transform"):
    """Transform a batch of images.

    Applies this operator to a batch of non-vectorized images `x`.

    Args:
      x: A `Tensor` with compatible shape and same dtype as `self`.
      adjoint: A `boolean`. If `True`, transforms the input using the adjoint
        of the operator, instead of the operator itself.
      name: A name for this operation.

    Returns:
      The transformed `Tensor` with the same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      x = tf.convert_to_tensor(x, name="x")
      self._check_input_dtype(x)
      input_shape = self.range_shape if adjoint else self.domain_shape
      input_shape.assert_is_compatible_with(x.shape[-input_shape.rank:])  # pylint: disable=invalid-unary-operand-type
      return self._transform(x, adjoint=adjoint)

  @property
  def domain_shape(self):
    """Domain shape of this linear operator."""
    return self._domain_shape()

  @property
  def range_shape(self):
    """Range shape of this linear operator."""
    return self._range_shape()

  def domain_shape_tensor(self, name="domain_shape_tensor"):
    """Domain shape of this linear operator, determined at runtime."""
    with self._name_scope(name):  # pylint: disable=not-callable
      # Prefer to use statically defined shape if available.
      if self.domain_shape.is_fully_defined():
        return tensor_util.convert_shape_to_tensor(self.domain_shape.as_list())
      return self._domain_shape_tensor()

  def range_shape_tensor(self, name="range_shape_tensor"):
    """Range shape of this linear operator, determined at runtime."""
    with self._name_scope(name):  # pylint: disable=not-callable
      # Prefer to use statically defined shape if available.
      if self.range_shape.is_fully_defined():
        return tensor_util.convert_shape_to_tensor(self.range_shape.as_list())
      return self._range_shape_tensor()

  def batch_shape_tensor(self, name="batch_shape_tensor"):
    """Batch shape of this linear operator, determined at runtime."""
    with self._name_scope(name):  # pylint: disable=not-callable
      if self.batch_shape.is_fully_defined():
        return tensor_util.convert_shape_to_tensor(self.batch_shape.as_list())
      return self._batch_shape_tensor()

  def adjoint(self, name="adjoint"):
    """Returns the adjoint of this linear operator.

    The returned operator is a valid `LinalgImagingMixin` instance.

    Calling `self.adjoint()` and `self.H` are equivalent.

    Args:
      name: A name for this operation.

    Returns:
      A `LinearOperator` derived from `LinalgImagingMixin`, which
      represents the adjoint of this linear operator.
    """
    if self.is_self_adjoint:
      return self
    with self._name_scope(name):  # pylint: disable=not-callable
      return LinearOperatorAdjoint(self)

  H = property(adjoint, None)

  @abc.abstractmethod
  def _transform(self, x, adjoint=False):
    # Subclasses must override this method.
    raise NotImplementedError("Method `_transform` is not implemented.")

  def _matvec(self, x, adjoint=False):
    # Default implementation of `_matvec` for imaging operator. The vectorized
    # input `x` is first expanded to the its full shape, then transformed, then
    # vectorized again. Typically subclasses should not need to override this
    # method.
    x = self.expand_range_dimension(x) if adjoint else \
        self.expand_domain_dimension(x)
    x = self._transform(x, adjoint=adjoint)
    x = self.flatten_domain_shape(x) if adjoint else \
        self.flatten_range_shape(x)
    return x

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # Default implementation of `matmul` for imaging operator. If outer
    # dimension of argument is 1, call `matvec`. Otherwise raise an error.
    # Typically subclasses should not need to override this method.
    arg_outer_dim = -2 if adjoint_arg else -1

    if x.shape[arg_outer_dim] != 1:
      raise ValueError(
        f"`{self.__class__.__name__}` does not support matrix multiplication.")

    x = tf.squeeze(x, axis=arg_outer_dim)
    x = self.matvec(x, adjoint=adjoint)
    x = tf.expand_dims(x, axis=arg_outer_dim)
    return x

  @abc.abstractmethod
  def _domain_shape(self):
    # Users must override this method.
    return tf.TensorShape(None)

  @abc.abstractmethod
  def _range_shape(self):
    # Users must override this method.
    return tf.TensorShape(None)

  def _batch_shape(self):
    # Users should override this method if this operator has a batch shape.
    return tf.TensorShape([])

  def _domain_shape_tensor(self):
    # Users should override this method if they need to provide a dynamic domain
    # shape.
    raise NotImplementedError("_domain_shape_tensor is not implemented.")

  def _range_shape_tensor(self):
    # Users should override this method if they need to provide a dynamic range
    # shape.
    raise NotImplementedError("_range_shape_tensor is not implemented.")

  def _batch_shape_tensor(self):  # pylint: disable=arguments-differ
    # Users should override this method if they need to provide a dynamic batch
    # shape.
    return tf.constant([], dtype=tf.dtypes.int32)

  def _shape(self):
    # Default implementation of `_shape` for imaging operators. Typically
    # subclasses should not need to override this method.
    return self._batch_shape() + tf.TensorShape(
        [self.range_shape.num_elements(),
         self.domain_shape.num_elements()])

  def _shape_tensor(self):
    # Default implementation of `_shape_tensor` for imaging operators. Typically
    # subclasses should not need to override this method.
    return tf.concat([self.batch_shape_tensor(),
                      [tf.size(self.range_shape_tensor()),
                       tf.size(self.domain_shape_tensor())]], 0)

  def flatten_domain_shape(self, x):
    """Flattens `x` to match the domain dimension of this operator.

    Args:
      x: A `Tensor`. Must have shape `[...] + self.domain_shape`.

    Returns:
      The flattened `Tensor`. Has shape `[..., self.domain_dimension]`.
    """
    # pylint: disable=invalid-unary-operand-type
    self.domain_shape.assert_is_compatible_with(
        x.shape[-self.domain_shape.rank:])

    batch_shape = x.shape[:-self.domain_shape.rank]
    batch_shape_tensor = tf.shape(x)[:-self.domain_shape.rank]

    output_shape = batch_shape + self.domain_dimension
    output_shape_tensor = tf.concat(
        [batch_shape_tensor, [self.domain_dimension_tensor()]], 0)

    x = tf.reshape(x, output_shape_tensor)
    return tf.ensure_shape(x, output_shape)

  def flatten_range_shape(self, x):
    """Flattens `x` to match the range dimension of this operator.

    Args:
      x: A `Tensor`. Must have shape `[...] + self.range_shape`.

    Returns:
      The flattened `Tensor`. Has shape `[..., self.range_dimension]`.
    """
    # pylint: disable=invalid-unary-operand-type
    self.range_shape.assert_is_compatible_with(
        x.shape[-self.range_shape.rank:])

    batch_shape = x.shape[:-self.range_shape.rank]
    batch_shape_tensor = tf.shape(x)[:-self.range_shape.rank]

    output_shape = batch_shape + self.range_dimension
    output_shape_tensor = tf.concat(
        [batch_shape_tensor, [self.range_dimension_tensor()]], 0)

    x = tf.reshape(x, output_shape_tensor)
    return tf.ensure_shape(x, output_shape)

  def expand_domain_dimension(self, x):
    """Expands `x` to match the domain shape of this operator.

    Args:
      x: A `Tensor`. Must have shape `[..., self.domain_dimension]`.

    Returns:
      The expanded `Tensor`. Has shape `[...] + self.domain_shape`.
    """
    self.domain_dimension.assert_is_compatible_with(x.shape[-1])

    batch_shape = x.shape[:-1]
    batch_shape_tensor = tf.shape(x)[:-1]

    output_shape = batch_shape + self.domain_shape
    output_shape_tensor = tf.concat([
        batch_shape_tensor, self.domain_shape_tensor()], 0)

    x = tf.reshape(x, output_shape_tensor)
    return tf.ensure_shape(x, output_shape)

  def expand_range_dimension(self, x):
    """Expands `x` to match the range shape of this operator.

    Args:
      x: A `Tensor`. Must have shape `[..., self.range_dimension]`.

    Returns:
      The expanded `Tensor`. Has shape `[...] + self.range_shape`.
    """
    self.range_dimension.assert_is_compatible_with(x.shape[-1])

    batch_shape = x.shape[:-1]
    batch_shape_tensor = tf.shape(x)[:-1]

    output_shape = batch_shape + self.range_shape
    output_shape_tensor = tf.concat([
        batch_shape_tensor, self.range_shape_tensor()], 0)

    x = tf.reshape(x, output_shape_tensor)
    return tf.ensure_shape(x, output_shape)


@api_util.export("linalg.LinearOperator")
class LinearOperator(LinalgImagingMixin, tf.linalg.LinearOperator):  # pylint: disable=abstract-method
  r"""Base class defining a [batch of] linear operator[s].

  Provides access to common matrix operations without the need to materialize
  the matrix.

  This operator is similar to `tf.linalg.LinearOperator`_, but has additional
  methods to simplify operations on images, while maintaining compatibility
  with the TensorFlow linear algebra framework.

  Inputs and outputs to this linear operator or its subclasses may have
  meaningful non-vectorized N-D shapes. Thus this class defines the additional
  properties `domain_shape` and `range_shape` and the methods
  `domain_shape_tensor` and `range_shape_tensor`. These enrich the information
  provided by the built-in properties `shape`, `domain_dimension`,
  `range_dimension` and methods `domain_dimension_tensor` and
  `range_dimension_tensor`, which only have information about the vectorized 1D
  shapes.

  Subclasses of this operator must define the methods `_domain_shape` and
  `_range_shape`, which return the static domain and range shapes of the
  operator. Optionally, subclasses may also define the methods
  `_domain_shape_tensor` and `_range_shape_tensor`, which return the dynamic
  domain and range shapes of the operator. These two methods will only be called
  if `_domain_shape` and `_range_shape` do not return fully defined static
  shapes.

  Subclasses must define the abstract method `_transform`, which
  applies the operator (or its adjoint) to a [batch of] images. This internal
  method is called by `transform`. In general, subclasses of this operator
  should not define the methods `_matvec` or `_matmul`. These have default
  implementations which call `_transform`.

  Operators derived from this class may be used in any of the following ways:

  1. Using method `transform`, which expects a full-shaped input and returns
     a full-shaped output, i.e. a tensor with shape `[...] + shape`, where
     `shape` is either the `domain_shape` or the `range_shape`. This method is
     unique to operators derived from this class.
  2. Using method `matvec`, which expects a vectorized input and returns a
     vectorized output, i.e. a tensor with shape `[..., n]` where `n` is
     either the `domain_dimension` or the `range_dimension`. This method is
     part of the TensorFlow linear algebra framework.
  3. Using method `matmul`, which expects matrix inputs and returns matrix
     outputs. Note that a matrix is just a column vector in this context, i.e.
     a tensor with shape `[..., n, 1]`, where `n` is either the
     `domain_dimension` or the `range_dimension`. Matrices which are not column
     vectors (i.e. whose last dimension is not 1) are not supported. This
     method is part of the TensorFlow linear algebra framework.

  Operators derived from this class may also be used with the functions
  `tf.linalg.matvec`_ and `tf.linalg.matmul`_, which will call the
  corresponding methods.

  This class also provides the convenience functions `flatten_domain_shape` and
  `flatten_range_shape` to flatten full-shaped inputs/outputs to their
  vectorized form. Conversely, `expand_domain_dimension` and
  `expand_range_dimension` may be used to expand vectorized inputs/outputs to
  their full-shaped form.

  **Subclassing**

  Subclasses must always define `_transform`, which implements this operator's
  functionality (and its adjoint). In general, subclasses should not define the
  methods `_matvec` or `_matmul`. These have default implementations which call
  `_transform`.

  Subclasses must always define `_domain_shape`
  and `_range_shape`, which return the static domain/range shapes of the
  operator. If the subclassed operator needs to provide dynamic domain/range
  shapes and the static shapes are not always fully-defined, it must also define
  `_domain_shape_tensor` and `_range_shape_tensor`, which return the dynamic
  domain/range shapes of the operator. In general, subclasses should not define
  the methods `_shape` or `_shape_tensor`. These have default implementations.

  If the subclassed operator has a non-scalar batch shape, it must also define
  `_batch_shape` which returns the static batch shape. If the static batch shape
  is not always fully-defined, the subclass must also define
  `_batch_shape_tensor`, which returns the dynamic batch shape.

  Args:
    dtype: The `tf.dtypes.DType` of the matrix that this operator represents.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose. If `dtype` is real, this is equivalent to being symmetric.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`. Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.

  .. _tf.linalg.LinearOperator: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperator
  .. _tf.linalg.matvec: https://www.tensorflow.org/api_docs/python/tf/linalg/matvec
  .. _tf.linalg.matmul: https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
  """


@api_util.export("linalg.LinearOperatorAdjoint")
class LinearOperatorAdjoint(LinalgImagingMixin,  # pylint: disable=abstract-method
                            tf.linalg.LinearOperatorAdjoint):
  """Linear operator representing the adjoint of another operator.

  `LinearOperatorAdjoint` is initialized with an operator :math:`A` and
  represents its adjoint :math:`A^H`.

  .. note:
    Similar to `tf.linalg.LinearOperatorAdjoint`_, but with imaging extensions.

  Args:
    operator: A `LinearOperator`.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`. Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`. Default is `operator.name +
      "_adjoint"`.

  .. _tf.linalg.LinearOperatorAdjoint: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperatorAdjoint
  """
  def _transform(self, x, adjoint=False):
    # pylint: disable=protected-access
    return self.operator._transform(x, adjoint=(not adjoint))

  def _domain_shape(self):
    return self.operator.range_shape

  def _range_shape(self):
    return self.operator.domain_shape

  def _batch_shape(self):
    return self.operator.batch_shape

  def _domain_shape_tensor(self):
    return self.operator.range_shape_tensor()

  def _range_shape_tensor(self):
    return self.operator.domain_shape_tensor()

  def _batch_shape_tensor(self):
    return self.operator.batch_shape_tensor()


@api_util.export("linalg.LinearOperatorComposition")
class LinearOperatorComposition(LinalgImagingMixin,  # pylint: disable=abstract-method
                                tf.linalg.LinearOperatorComposition):
  """Composes one or more linear operators.

  `LinearOperatorComposition` is initialized with a list of operators
  :math:`A_1, A_2, ..., A_J` and represents their composition
  :math:`A_1 A_2 ... A_J`.

  .. note:
    Similar to `tf.linalg.LinearOperatorComposition`_, but with imaging
    extensions.

  Args:
    operators: A `list` of `LinearOperator` objects, each with the same `dtype`
      and composable shape.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`. Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.  Default is the individual
      operators names joined with `_o_`.

  .. _tf.linalg.LinearOperatorComposition: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperatorComposition
  """
  def _transform(self, x, adjoint=False):
    # pylint: disable=protected-access
    if adjoint:
      transform_order_list = self.operators
    else:
      transform_order_list = list(reversed(self.operators))

    result = transform_order_list[0]._transform(x, adjoint=adjoint)
    for operator in transform_order_list[1:]:
      result = operator._transform(result, adjoint=adjoint)
    return result

  def _domain_shape(self):
    return self.operators[-1].domain_shape

  def _range_shape(self):
    return self.operators[0].range_shape

  def _batch_shape(self):
    return array_ops.broadcast_static_shapes(
        *[operator.batch_shape for operator in self.operators])

  def _domain_shape_tensor(self):
    return self.operators[-1].domain_shape_tensor()

  def _range_shape_tensor(self):
    return self.operators[0].range_shape_tensor()

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shapes(
        *[operator.batch_shape_tensor() for operator in self.operators])


@api_util.export("linalg.LinearOperatorAddition")
class LinearOperatorAddition(LinalgImagingMixin,  # pylint: disable=abstract-method
                             linalg_ext.LinearOperatorAddition):
  """Adds one or more linear operators.

  `LinearOperatorAddition` is initialized with a list of operators
  :math:`A_1, A_2, ..., A_J` and represents their addition
  :math:`A_1 + A_2 + ... + A_J`.

  Args:
    operators: A `list` of `LinearOperator` objects, each with the same `dtype`
      and shape.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`. Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.  Default is the individual
      operators names joined with `_p_`.
  """
  def _transform(self, x, adjoint=False):
    # pylint: disable=protected-access
    result = self.operators[0]._transform(x, adjoint=adjoint)
    for operator in self.operators[1:]:
      result += operator._transform(x, adjoint=adjoint)
    return result

  def _domain_shape(self):
    return self.operators[0].domain_shape

  def _range_shape(self):
    return self.operators[0].range_shape

  def _batch_shape(self):
    return array_ops.broadcast_static_shapes(
        *[operator.batch_shape for operator in self.operators])

  def _domain_shape_tensor(self):
    return self.operators[0].domain_shape_tensor()

  def _range_shape_tensor(self):
    return self.operators[0].range_shape_tensor()

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shapes(
        *[operator.batch_shape_tensor() for operator in self.operators])


@api_util.export("linalg.LinearOperatorScaledIdentity")
class LinearOperatorScaledIdentity(LinalgImagingMixin,  # pylint: disable=abstract-method
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


@api_util.export("linalg.LinearOperatorDiag")
class LinearOperatorDiag(LinalgImagingMixin, tf.linalg.LinearOperatorDiag): # pylint: disable=abstract-method
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
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`.  Note that we do not require the operator to be
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


@api_util.export("linalg.LinearOperatorGramMatrix")
class LinearOperatorGramMatrix(LinalgImagingMixin,  # pylint: disable=abstract-method
                               tf.linalg.LinearOperator):
  r"""Linear operator representing the Gram matrix of an operator.

  If :math:`A` is a `LinearOperator`, this operator is equivalent to
  :math:`A^H A`.

  The Gram matrix of :math:`A` appears in the normal equation
  :math:`A^H A x = A^H b` associated with the least squares problem
  :math:`{\mathop{\mathrm{argmin}}_x} {\left \| Ax-b \right \|_2^2}`.

  This operator is self-adjoint and positive definite. Therefore, linear systems
  defined by this linear operator can be solved using the conjugate gradient
  method.

  This operator supports the optional addition of a regularization parameter
  :math:`\lambda` and a transform matrix :math:`T`. If these are provided,
  this operator becomes :math:`A^H A + \lambda T^H T`. This appears
  in the regularized normal equation
  :math:`\left ( A^H A + \lambda T^H T \right ) x = A^H b + \lambda T^H T x_0`,
  associated with the regularized least squares problem
  :math:`{\mathop{\mathrm{argmin}}_x} {\left \| Ax-b \right \|_2^2 + \lambda \left \| T(x-x_0) \right \|_2^2}`.

  Args:
    operator: A `LinearOperator`.
    reg_parameter: A `Tensor` of shape `[B1, ..., Bb]` and real dtype.
      The regularization parameter :math:`\lambda`. Defaults to 0.
    reg_operator: A `LinearOperator`. The regularization transform :math:`T`.
      Defaults to the identity.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.
  """
  def __init__(self,
               operator,
               reg_parameter=None,
               reg_operator=None,
               is_non_singular=None,
               is_self_adjoint=True,
               is_positive_definite=True,
               is_square=True,
               name=None):
    parameters = dict(
        operator=operator,
        reg_parameter=reg_parameter,
        reg_operator=reg_operator,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)
    self._operator = operator
    self._reg_parameter = reg_parameter
    self._reg_operator = reg_operator
    self._composed = LinearOperatorComposition(
        operators=[self._operator.H, self._operator])

    if not is_self_adjoint:
      raise ValueError("A Gram matrix is always self-adjoint.")
    if not is_positive_definite:
      raise ValueError("A Gram matrix is always positive-definite.")
    if not is_square:
      raise ValueError("A Gram matrix is always square.")

    if self._reg_parameter is not None:
      reg_operator_gm = LinearOperatorScaledIdentity(
          shape=self._operator.domain_shape,
          multiplier=tf.cast(self._reg_parameter, self._operator.dtype))
      if self._reg_operator is not None:
        reg_operator_gm = LinearOperatorComposition(
            operators=[reg_operator_gm,
                       self._reg_operator.H,
                       self._reg_operator])
      self._composed = LinearOperatorAddition(
          operators=[self._composed, reg_operator_gm])

    super().__init__(operator.dtype,
                     is_non_singular=is_non_singular,
                     is_self_adjoint=is_self_adjoint,
                     is_positive_definite=is_positive_definite,
                     is_square=is_square,
                     parameters=parameters)

  def _transform(self, x, adjoint=False):
    return self._composed.transform(x, adjoint=adjoint)

  def _domain_shape(self):
    return self.operator.domain_shape

  def _range_shape(self):
    return self.operator.domain_shape

  def _batch_shape(self):
    return self.operator.batch_shape

  def _domain_shape_tensor(self):
    return self.operator.domain_shape_tensor()

  def _range_shape_tensor(self):
    return self.operator.domain_shape_tensor()

  def _batch_shape_tensor(self):
    return self.operator.batch_shape_tensor()

  @property
  def operator(self):
    return self._operator


@api_util.export("linalg.LinearOperatorFiniteDifference")
class LinearOperatorFiniteDifference(LinalgImagingMixin,  # pylint: disable=abstract-method
                                     tf.linalg.LinearOperator):
  """Linear operator representing a finite difference matrix.

  Args:
    domain_shape: A `tf.TensorShape` or list of ints. The shape of the input
      images.
    axis: An `int`. The axis along which the finite difference is taken.
      Defaults to -1.
    dtype: A `tf.dtypes.DType`. The data type for this operator. Defaults to
      `float32`.
    name: A `str`. A name for this operator.
  """
  def __init__(self,
               domain_shape,
               axis=-1,
               dtype=tf.dtypes.float32,
               name="LinearOperatorFiniteDifference"):

    parameters = dict(
      domain_shape=domain_shape,
      axis=axis,
      dtype=dtype,
      name=name
    )

    domain_shape = tf.TensorShape(domain_shape)
    self._axis = check_util.validate_axis(axis, domain_shape.rank,
                                          max_length=1,
                                          canonicalize="negative",
                                          scalar_to_list=False)

    range_shape = domain_shape.as_list()
    range_shape[self.axis] = range_shape[self.axis] - 1
    range_shape = tf.TensorShape(range_shape)

    self._domain_shape_value = domain_shape
    self._range_shape_value = range_shape

    super().__init__(dtype,
                     is_non_singular=None,
                     is_self_adjoint=None,
                     is_positive_definite=None,
                     is_square=None,
                     name=name,
                     parameters=parameters)

  def _transform(self, x, adjoint=False):

    if adjoint:
      paddings1 = [[0, 0]] * x.shape.rank
      paddings2 = [[0, 0]] * x.shape.rank
      paddings1[self.axis] = [1, 0]
      paddings2[self.axis] = [0, 1]
      x1 = tf.pad(x, paddings1) # pylint: disable=no-value-for-parameter
      x2 = tf.pad(x, paddings2) # pylint: disable=no-value-for-parameter
      x = x1 - x2
    else:
      slice1 = [slice(None)] * x.shape.rank
      slice2 = [slice(None)] * x.shape.rank
      slice1[self.axis] = slice(1, None)
      slice2[self.axis] = slice(None, -1)
      x1 = x[tuple(slice1)]
      x2 = x[tuple(slice2)]
      x = x1 - x2

    return x

  def _domain_shape(self):
    return self._domain_shape_value

  def _range_shape(self):
    return self._range_shape_value

  @property
  def axis(self):
    return self._axis
