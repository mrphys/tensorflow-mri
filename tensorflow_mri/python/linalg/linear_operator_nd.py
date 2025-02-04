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
"""Utilities for N-D linear operators."""

import functools
import string

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.linalg import linear_operator_algebra
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util


def make_linear_operator_nd(cls):
  """Class decorator for subclasses of `LinearOperatorND`."""
  # Call the original decorator.
  cls = linear_operator.make_linear_operator(cls)

  # Add the N-D specific doclines.
  cls = update_docstring(cls)

  return cls


def update_docstring(op_cls):
  """Adds a notice to the docstring."""
  tfmri_additional_nd_functionality = string.Template("""
  ```{rubric} Additional N-D functionality (TensorFlow MRI)
  ```

  This operator has additional functionality to work with N-dimensional
  problems more easily.

  - Process non-vectorized N-dimensional inputs via `matvec_nd`, `solvevec_nd`
    and `lstsqvec_nd`.
  - Access static N-D shape information via `domain_shape` and `range_shape`.
  - Access dynamic N-D shape information via `domain_shape_tensor` and
    `range_shape_tensor`.
  """).substitute(class_name=op_cls.__name__)

  docstring = op_cls.__doc__
  doclines = docstring.split('\n')
  doclines += tfmri_additional_nd_functionality.split('\n')
  docstring = '\n'.join(doclines)
  op_cls.__doc__ = docstring

  return op_cls


@api_util.export("linalg.LinearOperatorND")
@make_linear_operator_nd
class LinearOperatorND(linear_operator.LinearOperator):
  """Base class defining a [batch of] N-D linear operator(s)."""
  # Overrides of existing methods.
  def matmul(self, x, adjoint=False, adjoint_arg=False, name="matmul"):
    # We define a special implementation for when `x` is a `LinearOperatorND`.
    if isinstance(x, LinearOperatorND):
      left_operator = self.adjoint() if adjoint else self
      right_operator = x.adjoint() if adjoint_arg else x

      tensor_util.assert_broadcast_compatible(
          left_operator.domain_shape,
          right_operator.range_shape,
          message=(
            f"N-D operators are incompatible: "
            f"the domain shape {left_operator.domain_shape} "
            f"of left operator {left_operator.name} is not broadcast-"
            f"compatible with the range shape {right_operator.shape} "
            f"of right operator {right_operator.name}"))

      with self._name_scope(name):  # pylint: disable=not-callable
        return linear_operator_algebra.matmul(left_operator, right_operator)

    # If `x` is not a `LinearOperatorND`, we use the original implementation.
    return super().matmul(
        x, adjoint=adjoint, adjoint_arg=adjoint_arg, name=name)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    """Default implementation of `_matmul` for N-D operator."""
    # Default implementation of `matmul` for N-D operator. Basically we
    # just call `matvec` for each column of `x` (or for each row, if
    # `adjoint_arg` is `True`). `tf.einsum` is used to transpose the input arg.
    batch_shape = tf.broadcast_static_shape(x.shape[:-2], self.batch_shape)
    output_dim = self.domain_dimension if adjoint else self.range_dimension
    if adjoint_arg and x.dtype.is_complex:
      x = tf.math.conj(x)
    x = tf.einsum('...ij->i...j' if adjoint_arg else '...ij->j...i', x)
    y = tf.map_fn(functools.partial(self.matvec, adjoint=adjoint), x,
                  fn_output_signature=tf.TensorSpec(
                      shape=batch_shape + [output_dim],
                      dtype=x.dtype))
    y = tf.einsum('i...j->...ji' if adjoint_arg else 'j...i->...ij', y)
    return y

  def _matvec(self, x, adjoint=False):
    """Default implementation of `_matvec` for N-D operator."""
    # Default implementation of `_matvec` for N-D operator. The vectorized
    # input `x` is first expanded to the its full shape, then transformed, then
    # vectorized again. Typically subclasses should not need to override this
    # method.
    x = (self.expand_range_dimension(x) if adjoint else
         self.expand_domain_dimension(x))
    x = self._matvec_nd(x, adjoint=adjoint)
    x = (self.flatten_domain_shape(x) if adjoint else \
         self.flatten_range_shape(x))
    return x

  def solve(self, rhs, adjoint=False, adjoint_arg=False, name="solve"):
    if self.is_non_singular is False:
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "be singular.")
    if self.is_square is False:
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "not be square.")

    # We define a special implementation for when `rhs` is a `LinearOperatorND`.
    if isinstance(rhs, LinearOperatorND):
      left_operator = self.adjoint() if adjoint else self
      right_operator = rhs.adjoint() if adjoint_arg else rhs

      tensor_util.assert_broadcast_compatible(
          left_operator.domain_shape,
          right_operator.range_shape,
          message=(
            f"N-D operators are incompatible: "
            f"the domain shape {left_operator.domain_shape} "
            f"of left operator {left_operator.name} is not broadcast-"
            f"compatible with the range shape {right_operator.shape} "
            f"of right operator {right_operator.name}"))

      with self._name_scope(name):  # pylint: disable=not-callable
        return linear_operator_algebra.solve(left_operator, right_operator)

    # If `x` is not a `LinearOperatorND`, we use the original implementation.
    return super().solve(
        rhs, adjoint=adjoint, adjoint_arg=adjoint_arg, name=name)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    """Default implementation of `_solve` for N-D operator."""
    # Default implementation of `_solve` for imaging operator. Basically we
    # just call `solvevec` for each column of `rhs` (or for each row, if
    # `adjoint_arg` is `True`). `tf.einsum` is used to transpose the input arg.
    batch_shape = tf.broadcast_static_shape(rhs.shape[:-2], self.batch_shape)
    output_dim = self.range_dimension if adjoint else self.domain_dimension
    if adjoint_arg and rhs.dtype.is_complex:
      rhs = tf.math.conj(rhs)
    rhs = tf.einsum('...ij->i...j' if adjoint_arg else '...ij->j...i', rhs)
    x = tf.map_fn(functools.partial(self.solvevec, adjoint=adjoint), rhs,
                  fn_output_signature=tf.TensorSpec(
                      shape=batch_shape + [output_dim],
                      dtype=rhs.dtype))
    x = tf.einsum('i...j->...ji' if adjoint_arg else 'j...i->...ij', x)
    return x

  def _solvevec(self, rhs, adjoint=False):
    """Default implementation of `_solvevec` for N-D operator."""
    # Default implementation of `_solvevec` for N-D operator. The
    # vectorized input `rhs` is first expanded to the its full shape, then
    # solved, then vectorized again. Typically subclasses should not need to
    # override this method.
    rhs = (self.expand_domain_dimension(rhs) if adjoint else
           self.expand_range_dimension(rhs))
    rhs = self._solvevec_nd(rhs, adjoint=adjoint)
    rhs = (self.flatten_range_shape(rhs) if adjoint else
           self.flatten_domain_shape(rhs))
    return rhs

  def _lstsq(self, rhs, adjoint=False, adjoint_arg=False):
    """Default implementation of `_lstsq` for N-D operator."""
    # Default implementation of `_solve` for N-D operator. Basically we
    # just call `solvevec` for each column of `rhs` (or for each row, if
    # `adjoint_arg` is `True`). `tf.einsum` is used to transpose the input arg.
    batch_shape = tf.broadcast_static_shape(rhs.shape[:-2], self.batch_shape)
    output_dim = self.range_dimension if adjoint else self.domain_dimension
    if adjoint_arg and rhs.dtype.is_complex:
      rhs = tf.math.conj(rhs)
    rhs = tf.einsum('...ij->i...j' if adjoint_arg else '...ij->j...i', rhs)
    x = tf.map_fn(functools.partial(self.lstsqvec, adjoint=adjoint), rhs,
                  fn_output_signature=tf.TensorSpec(
                      shape=batch_shape + [output_dim],
                      dtype=rhs.dtype))
    x = tf.einsum('i...j->...ji' if adjoint_arg else 'j...i->...ij', x)
    return x

  def _lstsqvec(self, rhs, adjoint=False):
    """Default implementation of `_lstsqvec` for N-D operator."""
    # Default implementation of `_solvevec` for N-D operator. The
    # vectorized input `rhs` is first expanded to the its full shape, then
    # solved, then vectorized again. Typically subclasses should not need to
    # override this method.
    rhs = (self.expand_domain_dimension(rhs) if adjoint else
           self.expand_range_dimension(rhs))
    rhs = self._lstsqvec_nd(rhs, adjoint=adjoint)
    rhs = (self.flatten_range_shape(rhs) if adjoint else
           self.flatten_domain_shape(rhs))
    return rhs

  def _shape(self):
    # Default implementation of `_shape` for imaging operators. Typically
    # subclasses should not need to override this method.
    return self._batch_shape().concatenate(tf.TensorShape(
        [self.range_shape.num_elements(),
         self.domain_shape.num_elements()]))

  def _shape_tensor(self):
    # Default implementation of `_shape_tensor` for imaging operators. Typically
    # subclasses should not need to override this method.
    return tf.concat([self.batch_shape_tensor(),
                      [tf.math.reduce_prod(self.range_shape_tensor()),
                       tf.math.reduce_prod(self.domain_shape_tensor())]], 0)

  # New methods.
  def matvec_nd(self, x, adjoint=False, name="matvec_nd"):
    """Transforms [batch] N-D input `x` with left multiplication `x --> Ax`.

    ```{note}
    Similar to `matvec`, but works with non-vectorized N-D inputs `x`.
    ```

    Args:
      x: A `tf.Tensor` with compatible shape and same dtype as `self`.
      adjoint: A boolean. If `True`, transforms the input using the adjoint
        of the operator, instead of the operator itself.
      name: A name for this operation.

    Returns:
      A `tf.Tensor` with same dtype as `x` and shape `[..., *nd_shape]`,
      where `nd_shape` is the equal to `domain_shape` if `adjoint` is `True`
      and `range_shape` otherwise.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      x = tf.convert_to_tensor(x, name="x")
      self._check_input_dtype(x)
      input_shape = self.range_shape if adjoint else self.domain_shape
      input_shape.assert_is_compatible_with(x.shape[-input_shape.rank:])  # pylint: disable=invalid-unary-operand-type
      return self._matvec_nd(x, adjoint=adjoint)

  def _matvec_nd(self, x, adjoint=False):
    # Subclasses must override this method.
    raise NotImplementedError("Method `_matvec_nd` is not implemented.")

  def solvevec_nd(self, rhs, adjoint=False, name="solve"):
    """Solve single equation with N-D right-hand side: `A x = rhs`.

    The returned `tf.Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    ```{note}
    Similar to `solvevec`, but works with non-vectorized N-D inputs `rhs`.
    ```

    Args:
      rhs: A `tf.Tensor` with same `dtype` as this operator.
        `rhs` is treated like a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.  See class docstring
        for definition of compatibility regarding batch dimensions.
      adjoint: A boolean. If `True`, solve the system involving the adjoint of
        this operator: $A^H x = b$. Defaults to `False`.
      name:  A name scope to use for ops added by this method.

    Returns:
      A `tf.Tensor` with same dtype as `x` and shape `[..., *nd_shape]`,
      where `nd_shape` is the equal to `range_shape` if `adjoint` is `True`
      and `domain_shape` otherwise.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      rhs = tf.convert_to_tensor(rhs, name="rhs")
      self._check_input_dtype(rhs)
      input_shape = self.domain_shape if adjoint else self.range_shape
      input_shape.assert_is_compatible_with(rhs.shape[-input_shape.rank:])  # pylint: disable=invalid-unary-operand-type
      return self._solvevec_nd(rhs, adjoint=adjoint)

  def _solvevec_nd(self, rhs, adjoint=False):
    # Subclasses may override this method.
    raise NotImplementedError("Method `_solvevec_nd` is not implemented.")

  def lstsqvec_nd(self, rhs, adjoint=False, name="solve"):
    """Solve single equation with N-D right-hand side: `A x = rhs`.

    The returned `tf.Tensor` is the least squares solution to the system of
    equations.

    ```{note}
    Similar to `solvevec`, but works with non-vectorized N-D inputs `rhs`.
    ```

    Args:
      rhs: A `tf.Tensor` with same `dtype` as this operator.
        `rhs` is treated like a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.  See class docstring
        for definition of compatibility regarding batch dimensions.
      adjoint: A boolean. If `True`, solve the system involving the adjoint of
        this operator: $A^H x = b$. Defaults to `False`.
      name:  A name scope to use for ops added by this method.

    Returns:
      A `tf.Tensor` with same dtype as `x` and shape `[..., *nd_shape]`,
      where `nd_shape` is the equal to `range_shape` if `adjoint` is `True`
      and `domain_shape` otherwise.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      rhs = tf.convert_to_tensor(rhs, name="rhs")
      self._check_input_dtype(rhs)
      input_shape = self.domain_shape if adjoint else self.range_shape
      input_shape.assert_is_compatible_with(rhs.shape[-input_shape.rank:])  # pylint: disable=invalid-unary-operand-type
      return self._lstsqvec_nd(rhs, adjoint=adjoint)

  def _lstsqvec_nd(self, rhs, adjoint=False):
    # Subclasses may override this method.
    raise NotImplementedError("Method `_lstsqvec_nd` is not implemented.")

  @property
  def domain_shape(self):
    """Domain shape of this linear operator, determined statically.

    Returns:
      A `tf.TensorShape` representing the shape of the domain of this operator.
    """
    return self._domain_shape()

  def _domain_shape(self):
    # Users must override this method.
    return tf.TensorShape(None)

  @property
  def range_shape(self):
    """Range shape of this linear operator, determined statically.

    Returns:
      A `tf.TensorShape` representing the shape of the range of this operator.
    """
    return self._range_shape()

  def _range_shape(self):
    # Users must override this method.
    return tf.TensorShape(None)

  def _batch_shape(self):
    # Users should override this method if this operator has a batch shape.
    return tf.TensorShape([])

  def domain_shape_tensor(self, name="domain_shape_tensor"):
    """Domain shape of this linear operator, determined at runtime.

    Args:
      name: A `str`. A name scope to use for ops added by this method.

    Returns:
      A 1D integer `tf.Tensor` representing the shape of the domain of this
      operator.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      # Prefer to use statically defined shape if available.
      if self.domain_shape.is_fully_defined():
        return tensor_util.convert_shape_to_tensor(self.domain_shape.as_list())
      return self._domain_shape_tensor()

  def _domain_shape_tensor(self):
    # Users should override this method if they need to provide a dynamic domain
    # shape.
    raise NotImplementedError("_domain_shape_tensor is not implemented.")

  def range_shape_tensor(self, name="range_shape_tensor"):
    """Range shape of this linear operator, determined at runtime.

    Args:
      name: A `str`. A name scope to use for ops added by this method.

    Returns:
      A 1D integer `tf.Tensor` representing the shape of the range of this
      operator.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      # Prefer to use statically defined shape if available.
      if self.range_shape.is_fully_defined():
        return tensor_util.convert_shape_to_tensor(self.range_shape.as_list())
      return self._range_shape_tensor()

  def _range_shape_tensor(self):
    # Users should override this method if they need to provide a dynamic range
    # shape.
    raise NotImplementedError("_range_shape_tensor is not implemented.")

  def batch_shape_tensor(self, name="batch_shape_tensor"):
    """Batch shape of this linear operator, determined at runtime."""
    with self._name_scope(name):  # pylint: disable=not-callable
      if self.batch_shape.is_fully_defined():
        return tensor_util.convert_shape_to_tensor(self.batch_shape.as_list())
      return self._batch_shape_tensor()

  def _batch_shape_tensor(self):  # pylint: disable=arguments-differ
    # Users should override this method if they need to provide a dynamic batch
    # shape.
    return tf.constant([], dtype=tf.dtypes.int32)

  @property
  def ndim(self):
    """Logical number of dimensions of this linear operator.

    ```{note}
    `ndim` can always be determined statically.
    ```

    ```{attention}
    This number may differ from the number of dimensions in `domain_shape`,
    `range_shape`, or both.
    ```
    """
    return self._ndim()

  def _ndim(self):
    # Users must override this method.
    return None

  def flatten_domain_shape(self, x):
    """Flattens `x` to match the domain dimension of this operator.

    Args:
      x: A `Tensor`. Must have shape `[...] + self.domain_shape`.

    Returns:
      The flattened `Tensor`. Has shape `[..., self.domain_dimension]`.
    """
    # pylint: disable=invalid-unary-operand-type
    domain_rank_static = self.domain_shape.rank
    if domain_rank_static is not None:
      domain_rank_dynamic = domain_rank_static
    else:
      domain_rank_dynamic = tf.shape(self.domain_shape_tensor())[0]

    if domain_rank_static is not None:
      self.domain_shape.assert_is_compatible_with(
          x.shape[-domain_rank_static:])

    if domain_rank_static is not None:
      batch_shape = x.shape[:-domain_rank_static]
    else:
      batch_shape = tf.TensorShape(None)
    batch_shape_tensor = tf.shape(x)[:-domain_rank_dynamic]

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
    range_rank_static = self.range_shape.rank
    if range_rank_static is not None:
      range_rank_dynamic = range_rank_static
    else:
      range_rank_dynamic = tf.shape(self.range_shape_tensor())[0]

    if range_rank_static is not None:
      self.range_shape.assert_is_compatible_with(
          x.shape[-range_rank_static:])

    if range_rank_static is not None:
      batch_shape = x.shape[:-range_rank_static]
    else:
      batch_shape = tf.TensorShape(None)
    batch_shape_tensor = tf.shape(x)[:-range_rank_dynamic]

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


@api_util.export("linalg.LinearOperatorMakeND")
@make_linear_operator_nd
class LinearOperatorMakeND(LinearOperatorND):
  """Adds multidimensional support to a linear operator.

  Adds multidimensional shape information to a `LinearOperator` and support
  for all `LinearOperatorND`-specific functionality, such as `matvec_nd`,
  `solvevec_nd`, `domain_shape` and `range_shape`.

  If the input operator acts like matrix $A$, then this operator also acts
  like matrix $A$. The functionality of the underlying operator is preserved,
  with this operator having a superset of its functionality.

  ```{rubric} Initialization
  ```
  This operator is initialized with a non-ND linear operator (`operator`) and
  range/domain shape information (`range_shape` and `domain_shape`)

  Args:
    operator: A `tfmri.linalg.LinearOperator`. If `operator` is an instance of
      `LinearOperatorND`, then `operator` is returned unchanged.
    range_shape: A `tf.Tensor` representing the range shape of the operator.
      Must be compatible with the range dimension of `operator`.
    domain_shape: A `tf.Tensor` representing the domain shape of the operator.
      Must be compatible with the domain dimension of `operator`.
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
  """
  def __new__(cls, operator, *args, **kwargs):
    # If the input operator is already an ND operator, return it.
    if isinstance(operator, LinearOperatorND):
      return operator
    return super().__new__(cls)

  def __init__(self,
               operator,
               range_shape=None,
               domain_shape=None,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None,
               **kwargs):
    # The arguments `range_shape_` and `domain_shape_` (with trailing
    # underscores) are used when reconstructing the operator from its
    # components.
    if range_shape is None:
      if 'range_shape_' not in kwargs:
        raise ValueError("Argument `range_shape` must be specified.")
      range_shape = kwargs['range_shape_']

    if domain_shape is None:
      if 'domain_shape_' not in kwargs:
        raise ValueError("Argument `domain_shape` must be specified.")
      domain_shape = kwargs['domain_shape_']

    parameters = dict(
        operator=operator,
        range_shape_=range_shape,
        domain_shape_=domain_shape,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)

    if isinstance(operator, LinearOperatorND):
      raise TypeError("operator is already a LinearOperatorND.")
    if not isinstance(operator, linear_operator.LinearOperator):
      raise TypeError(f"operator must be a LinearOperator, but got: {operator}")
    self._operator = operator

    if (is_non_singular is not None and
        operator.is_non_singular is not None and
        is_non_singular != operator.is_non_singular):
      raise ValueError("is_non_singular must match operator.is_non_singular.")
    if is_non_singular is None:
      is_non_singular = operator.is_non_singular

    if (is_self_adjoint is not None and
        operator.is_self_adjoint is not None and
        is_self_adjoint != operator.is_self_adjoint):
      raise ValueError("is_self_adjoint must match operator.is_self_adjoint.")
    if is_self_adjoint is None:
      is_self_adjoint = operator.is_self_adjoint

    if (is_positive_definite is not None and
        operator.is_positive_definite is not None and
        is_positive_definite != operator.is_positive_definite):
      raise ValueError(
          "is_positive_definite must match operator.is_positive_definite.")
    if is_positive_definite is None:
      is_positive_definite = operator.is_positive_definite

    if (is_square is not None and
        operator.is_square is not None and
        is_square != operator.is_square):
      raise ValueError("is_square must match operator.is_square.")
    if is_square is None:
      is_square = operator.is_square

    # Process the domain and range shapes and check that they are compatible.
    self._domain_shape_static, self._domain_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))
    self._range_shape_static, self._range_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(range_shape))

    if (self._domain_shape_static.num_elements() is not None and
        operator.domain_dimension is not None and
        self._domain_shape_static.num_elements() != operator.domain_dimension):
      raise ValueError(
          f"domain_shape must have the same number of elements as "
          f"operator.domain_dimension. "
          f"Found {self._domain_shape_static.num_elements()} "
          f"and {operator.domain_dimension}, respectively.")

    if (self._range_shape_static.num_elements() is not None and
        operator.range_dimension is not None and
        self._range_shape_static.num_elements() != operator.range_dimension):
      raise ValueError(
          f"range_shape must have the same number of elements as "
          f"operator.range_dimension. "
          f"Found {self._range_shape_static.num_elements()} "
          f"and {operator.range_dimension}, respectively.")

    # Initialization.
    if name is None:
      name = operator.name + "ND"

    with tf.name_scope(name):
      super().__init__(
          dtype=operator.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _domain_shape(self):
    return self._domain_shape_static

  def _domain_shape_tensor(self):
    return self._domain_shape_dynamic

  def _range_shape(self):
    return self._range_shape_static

  def _range_shape_tensor(self):
    return self._range_shape_dynamic

  def _batch_shape(self):
    return self.operator.batch_shape

  def _batch_shape_tensor(self):
    return self.operator.batch_shape_tensor()

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    return self.operator.matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _matvec(self, x, adjoint=False):
    return self.operator.matvec(x, adjoint=adjoint)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    return self.operator.solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _solvevec(self, rhs, adjoint=False):
    return self.operator.solvevec(rhs, adjoint=adjoint)

  def _lstsq(self, rhs, adjoint=False, adjoint_arg=False):
    return self.operator.lstsq(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _lstsqvec(self, rhs, adjoint=False):
    return self.oeprator.lstsqvec(rhs, adjoint=adjoint)

  def _matvec_nd(self, x, adjoint=False):
    x = (self.flatten_range_shape(x) if adjoint else \
         self.flatten_domain_shape(x))
    x = self._matvec(x, adjoint=adjoint)
    x = (self.expand_domain_dimension(x) if adjoint else
         self.expand_range_dimension(x))
    return x

  def _solvevec_nd(self, x, adjoint=False):
    x = (self.flatten_domain_shape(x) if adjoint else \
         self.flatten_range_shape(x))
    x = self._solvevec(x, adjoint=adjoint)
    x = (self.expand_range_dimension(x) if adjoint else
         self.expand_domain_dimension(x))
    return x

  def _lstsqvec_nd(self, x, adjoint=False):
    x = (self.flatten_domain_shape(x) if adjoint else \
         self.flatten_range_shape(x))
    x = self._lstsqvec(x, adjoint=adjoint)
    x = (self.expand_range_dimension(x) if adjoint else
         self.expand_domain_dimension(x))
    return x

  def _add_to_tensor(self, x):
    return self.operator.add_to_tensor(x)

  def _assert_non_singular(self):
    return self.operator.assert_non_singular()

  def _assert_self_adjoint(self):
    return self.operator.assert_self_adjoint()

  def _assert_positive_definite(self):
    return self.operator.assert_positive_definite()

  def _cond(self):
    return self.operator.cond()

  def _determinant(self):
    return self.operator.determinant()

  def _diag_part(self):
    return self.operator.diag_part()

  def _eigvals(self):
    return self.operator.eigvals()

  def _log_abs_determinant(self):
    return self.operator.log_abs_determinant()

  def _trace(self):
    return self.operator.trace()

  def _to_dense(self):
    return self.operator.to_dense()

  @property
  def operator(self):
    return self._operator

  @property
  def _composite_tensor_fields(self):
    # We use `domain_shape_` and `range_shape_` for conversion to/from composite tensor.
    return ("operator", "range_shape_", "domain_shape_")

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("range_shape_", "domain_shape_")

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {"operator": 0}
