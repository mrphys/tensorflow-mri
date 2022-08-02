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
"""Base linear operator."""

import abc
import functools

import tensorflow as tf
from tensorflow.python.ops.linalg import linear_operator

from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util


class LinearOperatorMixin(tf.linalg.LinearOperator):
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

    The returned operator is a valid `LinearOperatorMixin` instance.

    Calling `self.adjoint()` and `self.H` are equivalent.

    Args:
      name: A name for this operation.

    Returns:
      A `LinearOperator` derived from `LinearOperatorMixin`, which
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
class LinearOperator(LinearOperatorMixin, tf.linalg.LinearOperator):  # pylint: disable=abstract-method
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
class LinearOperatorAdjoint(LinearOperatorMixin,  # pylint: disable=abstract-method
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


make_composite_tensor = functools.partial(
    linear_operator.make_composite_tensor, module_name="tfmri.linalg")