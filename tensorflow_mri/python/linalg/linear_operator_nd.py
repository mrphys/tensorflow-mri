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
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util


def make_mri_operator_nd(cls):
  """Class decorator for subclasses of `LinearOperatorND`."""
  # Call the original decorator.
  cls = linear_operator.make_mri_operator(cls)

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
@make_mri_operator_nd
class LinearOperatorND(linear_operator.LinearOperator):
  """Base class defining a [batch of] N-D linear operator(s)."""
  # Overrides of existing methods.
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

  def _solvevec_nd(self, x, adjoint=False):
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

  def _lstsqvec_nd(self, x, adjoint=False):
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
