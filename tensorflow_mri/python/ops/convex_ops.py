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
"""Convex operators."""

import abc
import contextlib

import tensorflow as tf

from tensorflow_mri.python.ops import linalg_ops
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.util import check_util


class ConvexFunction():
  """Base class defining a [batch of] convex function[s].

  Represents a closed proper convex function
  :math:`$f : \mathbb{R}^{n}\rightarrow \mathbb{R}$`.

  Subclasses should implement the `_call` and `_prox` methods to define the
  forward pass and the proximal mapping, respectively.
  """
  def __init__(self,
               ndim=None,
               dtype=None,
               name=None):
    """Initialize this `ConvexFunction`."""
    self._ndim = check_util.validate_rank(ndim, 'ndim', accept_none=True)
    self._dtype = tf.dtypes.as_dtype(dtype or tf.dtypes.float32)
    self._name = name or type(self).__name__

  def __call__(self, x):
    return self.call(x)

  def call(self, x, name=None):
    """Evaluate this `ConvexFunction` at input point[s] `x`.

    Args:
      x: A `Tensor` of shape `[..., n]` and same dtype as `self`.
      name: A name for this operation (optional).

    Returns:
      A `Tensor` of shape `[...]` and same dtype as `self`.
    """
    with self._name_scope(name or "call"):
      x = tf.convert_to_tensor(x, name="x")
      self._check_input_shape(x)
      self._check_input_dtype(x)
      return self._call(x)

  def prox(self, x, name=None):
    """Evaluate the proximal operator of this `ConvexFunction` at point[s] `x`.

    Args:
      x: A `Tensor` of shape `[..., n]` and same dtype as `self`.
      name: A name for this operation (optional).

    Returns:
      A `Tensor` of shape `[..., n]` and same dtype as `self`.
    """
    with self._name_scope(name or "prox"):
      x = tf.convert_to_tensor(x, name="x")
      self._check_input_shape(x)
      self._check_input_dtype(x)
      return self._prox(x)

  @abc.abstractmethod
  def _call(self, x):
    # Must be implemented by subclasses.
    raise NotImplementedError("Method `_call` is not implemented.")

  @abc.abstractmethod
  def _prox(self, x):
    # Must be implemented by subclasses.
    raise NotImplementedError("Method `_prox` is not implemented.")

  @property
  def ndim(self):
    """The dimensionality of the domain of this `ConvexFunction`."""
    return self._ndim

  @property
  def dtype(self):
    """The `DType` of `Tensors` handled by this `ConvexFunction`."""
    return self._dtype

  @property
  def name(self):
    """Name prepended to all ops created by this `ConvexFunction`."""
    return self._name

  @contextlib.contextmanager
  def _name_scope(self, name=None):
    """Helper function to standardize op scope."""
    full_name = self.name
    if name is not None:
      full_name += "/" + name
    with tf.name_scope(full_name) as scope:
      yield scope

  def _check_input_shape(self, arg):
    """Check that arg.shape[-1] is compatible with self.ndim."""
    if arg.shape.rank is None:
      raise ValueError(
          "Expected argument to have known rank, but found: %s in tensor %s" %
          (arg.shape.rank, arg))
    if arg.shape.rank < 1:
      raise ValueError(
          "Expected argument to have rank >= 1, but found: %s in tensor %s" %
          (arg.shape.rank, arg))
    if not arg.shape[-1:].is_compatible_with([self.ndim]):
      raise ValueError(
          "Expected argument to have last dimension %d, but found: %d in "
          "tensor %s" % (self.ndim, arg.shape[-1], arg))

  def _check_input_dtype(self, arg):
    """Check that arg.dtype == self.dtype."""
    if arg.dtype.base_dtype != self.dtype:
      raise TypeError(
          "Expected argument to have dtype %s, but found: %s in tensor %s" %
          (self.dtype, arg.dtype, arg))


class ConvexFunctionL1Norm(ConvexFunction):
  """A `ConvexFunction` computing the [scaled] L1-norm of a [batch of] inputs.

  Args:
    scale: A `float`. A scaling factor. Defaults to 1.0.
    ndim: An `int`. The dimensionality of the domain of this `ConvexFunction`.
      Defaults to `None`.
    dtype: A `string` or `DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               scale=1.0,
               ndim=None,
               dtype=None,
               name=None):
    super().__init__(ndim=ndim, dtype=dtype, name=name)
    self._scale = scale

  def _call(self, x):
    return self._scale * tf.norm(x, ord=1, axis=-1)

  def _prox(self, x):
    return soft_threshold(x, self._scale)


class ConvexFunctionL2Norm(ConvexFunction):
  """A `ConvexFunction` computing the [scaled] L2-norm of a [batch of] inputs.

  Args:
    scale: A `float`. A scaling factor. Defaults to 1.0.
    ndim: An `int`. The dimensionality of the domain of this `ConvexFunction`.
      Defaults to `None`.
    dtype: A `string` or `DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               scale=1.0,
               ndim=None,
               dtype=None,
               name=None):
    super().__init__(ndim=ndim, dtype=dtype, name=name)
    self._scale = scale

  def _call(self, x):
    return self._scale * tf.norm(x, ord=2, axis=-1)

  def _prox(self, x):
    return block_soft_threshold(x, self._scale)


class ConvexFunctionQuadratic(ConvexFunction):
  """A `ConvexFunction` representing a generic quadratic function.

  Represents :math:`f(x) = \frac{1}{2} x^{T} A x + b^{T} x + c`.

  Args:
    quadratic_coefficient: A `Tensor` or a `LinearOperator` representing a
      self-adjoint, positive definite matrix `A` with shape `[..., n, n]`. The
      coefficient of the quadratic term.
    linear_coefficient: A `Tensor` representing a vector `b` with shape
      `[..., n]`. The coefficient of the linear term.
    constant_coefficient: A scalar `Tensor` representing the constant term `c`.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               quadratic_coefficient,
               linear_coefficient=None,
               constant_coefficient=None,
               scale=1.0,
               name=None):
    super().__init__(ndim=quadratic_coefficient.shape[-1],
                     dtype=quadratic_coefficient.dtype,
                     name=name)
    self._quadratic_coefficient = quadratic_coefficient
    self._linear_coefficient = self._validate_linear_coefficient(
        linear_coefficient)
    self._constant_coefficient = self._validate_constant_coefficient(
        constant_coefficient)
    self._scale = tf.convert_to_tensor(scale, dtype=self.dtype)
    self._one_over_scale = tf.convert_to_tensor(1.0 / scale, dtype=self.dtype)

    # Operator A^T A + 1 / \lambda * I, used to evaluate the proximal operator.
    self._operator = linalg_ops.LinearOperatorAddition([
        self._quadratic_coefficient,
        tf.linalg.LinearOperatorScaledIdentity(
            num_rows=self._quadratic_coefficient.domain_dimension,
            multiplier=self._one_over_scale)],
        is_self_adjoint=True,
        is_positive_definite=True)

  def _call(self, x):
    # Calculate the quadratic term.
    result = 0.5 * _dot(
        x, tf.linalg.matvec(self._quadratic_coefficient, x))
    # Add the linear term, if there is one.
    if self._linear_coefficient is not None:
      result += _dot(self._linear_coefficient, x)
    # Add the constant term, if there is one.
    if self._constant_coefficient is not None:
      result += self._constant_coefficient
    return self._scale * result

  def _prox(self, x):
    rhs = self._one_over_scale * x
    if self._linear_coefficient is not None:
      rhs -= self._linear_coefficient
    return linalg_ops.conjugate_gradient(self._operator, rhs).x

  def _validate_linear_coefficient(self, coef):
    """Validates the linear coefficient."""
    if coef.shape.rank is None:
      raise ValueError(
          "Expected linear coefficient to have known rank, but found: %s in "
          "tensor %s" % (coef.shape.rank, coef))
    if coef.shape.rank < 1:
      raise ValueError(
          "Expected linear coefficient to have rank >= 1, but found: %s in "
          "tensor %s" % (coef.shape.rank, coef))
    if not coef.shape[-1:].is_compatible_with([self.ndim]):
      raise ValueError(
          "Expected linear coefficient to have last dimension %d, but found: "
          "%d in tensor %s" % (self.ndim, coef.shape[-1], coef))
    if coef.dtype != self.dtype:
      raise ValueError(
          "Expected linear coefficient to have dtype %s, but found: %s in "
          "tensor %s" % (self.dtype, coef.dtype, coef))
    return coef

  def _validate_constant_coefficient(self, coef):
    """Validates the constant coefficient."""
    if coef.dtype != self.dtype:
      raise ValueError(
          "Expected constant coefficient to have dtype %s, but found: %s in "
          "tensor %s" % (self.dtype, coef.dtype, coef))
    return coef


class ConvexFunctionLeastSquares(ConvexFunctionQuadratic):
  """A `ConvexFunction` representing a least squares function.
  
  Represents :math:`f(x) = \frac{1}{2} {\left \| A x - b \right \|}_{2}^{2}`.

  Minimizing `f(x)` is equivalent to finding a solution to the linear system
  :math:`Ax - b`.

  Args:
    operator: A `Tensor` or a `LinearOperator` representing a matrix `A` with
      shape `[..., m, n]`. The linear system operator.
    rhs: A `Tensor` representing a vector `b` with shape `[..., m]`. The
      right-hand side of the linear system.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self, operator, rhs, scale=1.0, name=None):
    quadratic_coefficient = tf.linalg.LinearOperatorComposition(
        [operator.H, operator], is_self_adjoint=True, is_positive_definite=True)
    linear_coefficient = tf.math.negative(tf.linalg.matvec(operator, rhs,
                                                           adjoint_a=True))
    constant_coefficient = tf.constant(0.0, dtype=operator.dtype)
    super().__init__(quadratic_coefficient=quadratic_coefficient,
                     linear_coefficient=linear_coefficient,
                     constant_coefficient=constant_coefficient,
                     scale=scale,
                     name=name)


def block_soft_threshold(x, threshold, name=None):
  """Block soft thresholding operator.

  In the context of proximal gradient methods, this function is the proximal
  operator of :math:`f = {\left\| x \right\|}_{2}` (L2 norm).

  Args:
    x: A `Tensor` of shape `[..., n]`.
    threshold: A `float`.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` of shape `[..., n]` and same dtype as `x`.
  """
  with tf.name_scope(name or 'block_soft_threshold'):
    x = tf.convert_to_tensor(x, name='x')
    threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
    one = tf.constant(1.0, dtype=x.dtype, name='one')
    x_norm = tf.norm(x, ord=2, axis=-1, keepdims=True)
    return tf.math.maximum(one - threshold / x_norm, 0.) * x


def shrinkage(x, threshold, name=None):
  """Shrinkage operator.

  In the context of proximal gradient methods, this function is the proximal
  operator of :math:`f = \frac{1}{2}{\left\| x \right\|}_{2}^{2}`.

  Args:
    x: A `Tensor` of shape `[..., n]`.
    threshold: A `float`.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` of shape `[..., n]` and same dtype as `x`.
  """
  with tf.name_scope(name or 'shrinkage'):
    x = tf.convert_to_tensor(x, name='x')
    threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
    one = tf.constant(1.0, dtype=x.dtype, name='one')
    return x / (one + threshold)


def soft_threshold(x, threshold, name=None):
  """Soft thresholding operator.

  In the context of proximal gradient methods, this function is the proximal
  operator of :math:`f = {\left\| x \right\|}_{1}` (L1 norm).

  Args:
    x: A `Tensor` of shape `[..., n]`.
    threshold: A `float`.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` of shape `[..., n]` and same dtype as `x`.
  """
  with tf.name_scope(name or 'soft_threshold'):
    x = tf.convert_to_tensor(x, name='x')
    threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
    return tf.math.sign(x) * tf.math.maximum(tf.math.abs(x) - threshold, 0.)


def _dot(x, y):
  """Returns the dot product of `x` and `y`."""
  return tf.squeeze(
      tf.linalg.matvec(
          x[..., tf.newaxis],
          y, adjoint_a=True), axis=-1)


class Regularizer():
  """Base class defining a [batch of] regularizer[s]."""
  def __init__(self,
               factor=1.0,
               convex_operator=None,
               linear_operator=None):
    """Initialize this `Regularizer`."""
    self._factor = factor
    self._cvx_op = convex_operator
    self._lin_op = linear_operator

  def __call__(self, x):
    return self.call(x)

  def call(self, x):
    """Compute the regularization term for input `x`."""
    # Apply linear transformation, then convex operator.
    return self._factor * self._cvx_op(tf.linalg.matvec(self._lin_op, x))


class TotalVariationRegularizer(Regularizer):
  """Regularizer calculating the total variation of a [batch of] input[s].

  Returns a value for each element in batch.

  Args:
    factor: A `float`. The regularization factor.
    axis: An `int` or a list of `int`. The axis along which to compute the
      total variation.
    ndim: An `int`. The number of non-batch dimensions. The last `ndim`
      dimensions will be reduced.
  """
  def __init__(self, factor, axis, ndim=None):
    super().__init__(factor=factor)
    self._axis = axis
    self._ndim = ndim

  def call(self, x):
    # Override default implementation of `call` - we use a shortcut here rather
    # than the corresponding linear and convex operators.
    return self._factor * tf.math.reduce_sum(
        image_ops.total_variation(x, axis=self._axis, keepdims=True),
        axis=(list(range(-self._ndim, 0)) # pylint: disable=invalid-unary-operand-type
              if self._ndim is not None else self._ndim))
