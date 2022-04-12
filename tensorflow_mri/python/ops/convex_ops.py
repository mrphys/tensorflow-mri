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

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import linalg_ops
from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import linalg_ext
from tensorflow_mri.python.util import linalg_imaging
from tensorflow_mri.python.util import tensor_util


@api_util.export("convex.ConvexFunction")
class ConvexFunction():
  r"""Base class defining a [batch of] convex function[s].

  Represents a closed proper convex function
  :math:`f : \mathbb{R}^{n}\rightarrow \mathbb{R}` or
  :math:`f : \mathbb{C}^{n}\rightarrow \mathbb{R}`.

  Subclasses should implement the `_call` and `_prox` methods to define the
  forward pass and the proximal mapping, respectively. Gradients are
  provided by TensorFlow's automatic differentiation feature.
  """
  def __init__(self,
               scale=None,
               ndim=None,
               dtype=None,
               name=None):
    """Initialize this `ConvexFunction`."""
    if isinstance(ndim, tf.compat.v1.Dimension):
      ndim = ndim.value
    self._ndim = check_util.validate_rank(ndim, 'ndim', accept_none=True)
    self._dtype = tf.dtypes.as_dtype(dtype or tf.dtypes.float32)
    self._name = name or type(self).__name__
    if scale is None:
      scale = 1.0
    self._scale = tf.convert_to_tensor(scale, dtype=self.dtype.real_dtype)

  def __call__(self, x):
    return self.call(x)

  def call(self, x, name=None):
    """Evaluate this `ConvexFunction` at input point[s] `x`.

    Args:
      x: A `tf.Tensor` of shape `[..., n]` and same dtype as `self`.
      name: A name for this operation (optional).

    Returns:
      A `tf.Tensor` of shape `[...]` and same dtype as `self`.
    """
    with self._name_scope(name or "call"):
      x = tf.convert_to_tensor(x, name="x")
      self._check_input_shape(x)
      self._check_input_dtype(x)
      return self._call(x)

  def prox(self, x, scale=None, name=None, **kwargs):
    """Evaluate the proximal operator of this `ConvexFunction` at point[s] `x`.

    Args:
      x: A `tf.Tensor` of shape `[..., n]` and same dtype as `self`.
      scale: A scalar `float`. Additional scaling factor.
      name: A name for this operation (optional).
      **kwargs: A `dict`. Additional keyword arguments to pass to `_prox`.

    Returns:
      A `tf.Tensor` of shape `[..., n]` and same dtype as `self`.
    """
    with self._name_scope(name or "prox"):
      x = tf.convert_to_tensor(x, name="x")
      self._check_input_shape(x)
      self._check_input_dtype(x)
      return self._prox(x, scale=scale, **kwargs)

  def conj(self, name=None):
    """Returns the convex conjugate of this `ConvexFunction`.

    Args:
      name: A name for this operation (optional).

    Returns:
      A `ConvexFunction` which represents the convex conjugate of `self`.
    """
    with self._name_scope(name or "conj"):
      return self._conj()

  def ndim_tensor(self, name=None):
    """Returns the number of dimensions of this `ConvexFunction`.

    Args:
      name: A name for this operation (optional).

    Returns:
      A scalar integer `tf.Tensor`.
    """
    with self._name_scope(name or "ndim_tensor"):
      # Prefer to use statically defined ndim if available.
      if isinstance(self.ndim, int):
        return tf.constant(self.ndim, dtype=tf.int32)
      return self._ndim_tensor()

  def shape_tensor(self, name=None):
    """Returns the dynamic shape of this `ConvexFunction`.

    Args:
      name: A name for this operation (optional).

    Returns:
      A 1D integer `tf.Tensor`.
    """
    with self._name_scope(name or "shape_tensor"):
      # Prefer to use statically defined shape if available.
      if self.shape.is_fully_defined():
        return tensor_util.convert_shape_to_tensor(
            self.shape.as_list(), name="shape")
      return self._shape_tensor()

  def batch_shape_tensor(self, name=None):
    """Returns the dynamic batch shape of this `ConvexFunction`.

    Args:
      name: A name for this operation (optional).

    Returns:
      A 1D integer `tf.Tensor`.
    """
    with self._name_scope(name or "batch_shape_tensor"):
      return self._batch_shape_tensor()

  @abc.abstractmethod
  def _call(self, x):
    # Must be implemented by subclasses.
    raise NotImplementedError("Method `_call` is not implemented.")

  def _prox(self, x, scale=None):
    # Must be implemented by subclasses.
    raise NotImplementedError("Method `_prox` is not implemented.")

  def _conj(self):
    # Must be implemented by subclasses.
    raise NotImplementedError("Method `_conj` is not implemented.")

  def _shape(self):
    """Returns the static shape of this `ConvexFunction`.

    Defaults to `[ndim]`. Subclasses should override this method if they
    have any batch dimensions.

    Returns:
      A `tf.TensorShape`.
    """
    return tf.TensorShape([self.ndim])

  def _ndim_tensor(self):
    """Returns the dynamic number of dimensions of this `ConvexFunction`.

    Returns:
      A scalar integer `tf.Tensor`.
    """
    raise NotImplementedError("`_ndim_tensor` is not implemented.")

  def _shape_tensor(self):
    """Returns the dynamic shape of this `ConvexFunction`.

    Returns:
      A 1D integer `tf.Tensor`.
    """
    raise NotImplementedError("`_shape_tensor` is not implemented.")

  def _batch_shape_tensor(self):
    """Returns the dynamic batch shape of this `ConvexFunction`.

    Returns:
      A 1D integer `tf.Tensor`.
    """
    if self.batch_shape.is_fully_defined():
      return tensor_util.convert_shape_to_tensor(
          self.batch_shape.as_list(), name="batch_shape")
    return self.shape_tensor()[:-1]

  @property
  def scale(self):
    """The scaling factor."""
    return self._scale

  @property
  def ndim(self):
    """The number of dimensions of this `ConvexFunction`."""
    return self._ndim

  @property
  def shape(self):
    """The static shape of this `ConvexFunction`."""
    return self._shape()

  @property
  def batch_shape(self):
    """The static batch shape of this `ConvexFunction`."""
    return self.shape[:-1]

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

  def _check_input_shape(self, arg):  # pylint: disable=missing-param-doc
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


@api_util.export("convex.ConvexFunctionAffineMappingComposition")
class ConvexFunctionAffineMappingComposition(ConvexFunction):
  """Composes a convex function and an affine mapping.

  Represents :math:`f(Ax + b)`, where :math:`f` is a `ConvexFunction`,
  :math:`A` is a `LinearOperator` and :math:`b` is a constant `Tensor`.

  Args:
    function: A `ConvexFunction`.
    operator: A `LinearOperator`. Defaults to the identity.
    constant: A `Tensor`. Defaults to 0.
    scale: A `float`. A scaling factor.
    name: A name for this operation.
  """
  def __init__(self,
               function,
               operator=None,
               constant=None,
               scale=None,
               name=None):
    ndim = operator.domain_dimension if operator is not None else function.ndim
    super().__init__(scale=scale,
                     ndim=ndim,
                     dtype=function.dtype,
                     name=name)
    self._function = function
    self._operator = operator
    self._constant = constant

  def _call(self, x):
    if self._operator is not None:
      x = self._operator.matvec(x)
    if self._constant is not None:
      x += self._constant
    return self._scale * self._function._call(x)  # pylint: disable=protected-access

  def _prox(self, x, scale=None):
    # Prox difficult to evaluate for general linear operators.
    # TODO(jmontalt): implement prox for specific cases such as orthogonal
    # operators.
    raise NotImplementedError(
       f"The proximal operator of {self.name} is not implemented or "
       f"does not have a closed form expression.")

  @property
  def function(self):
    return self._function

  @property
  def operator(self):
    return self._operator

  @property
  def constant(self):
    return self._constant


@api_util.export("convex.ConvexFunctionLinearOperatorComposition")
class ConvexFunctionLinearOperatorComposition(  # pylint: disable=abstract-method
    ConvexFunctionAffineMappingComposition):
  r"""Composes a convex function and a linear operator.

  Represents :math:`f(Ax)`, where :math:`f` is a `ConvexFunction` and
  :math:`A` is a `LinearOperator`.

  Args:
    function: A `ConvexFunction`.
    operator: A `LinearOperator`. Defaults to the identity.
    scale: A `float`. A scaling factor.
    name: A name for this operation.
  """
  def __init__(self,
               function,
               operator=None,
               scale=None,
               name=None):
    super().__init__(function,
                     operator=operator,
                     scale=scale,
                     name=name)


@api_util.export("convex.ConvexFunctionIndicatorBall")
class ConvexFunctionIndicatorBall(ConvexFunction):  # pylint: disable=abstract-method
  """A `ConvexFunction` representing the indicator function of an Lp ball.

  Args:
    order: A `float`. The order of the norm. Supported values are `1`, `2`,
      `np.inf`.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    ndim: An `int`. The dimensionality of the domain of this `ConvexFunction`.
      Defaults to `None`.
    dtype: A `tf.dtypes.DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               order=2,
               scale=None,
               ndim=None,
               dtype=None,
               name=None):
    super().__init__(scale=scale,
                     ndim=ndim,
                     dtype=dtype,
                     name=name)
    self._order = check_util.validate_enum(order, [1, 2, np.inf], name='order')

  def _call(self, x):
    # Note that the scale has no effect, as the indicator function is always
    # zero or infinity.
    return math_ops.indicator_ball(x, order=self._order)

  def _prox(self, x, scale=None):
    # The proximal operator of the indicator function of a closed convex set
    # (such as the Lp ball) is the projection onto the set.
    return math_ops.project_onto_ball(x, order=self._order)

  def _conj(self):
    # The convex conjugate of the indicator function on the unit ball defined
    # by the Lp-norm is the dual norm function.
    return ConvexFunctionNorm(
        order=_conjugate_exponent(self._order),
        scale=self._scale,
        ndim=self.ndim,
        dtype=self.dtype,
        name=f"{self.name}_conj")


@api_util.export("convex.ConvexFunctionIndicatorL1Ball")
class ConvexFunctionIndicatorL1Ball(ConvexFunctionIndicatorBall):  # pylint: disable=abstract-method
  """A `ConvexFunction` representing the indicator function of an L1 ball.

  Args:
    scale: A `float`. A scaling factor. Defaults to 1.0.
    ndim: An `int`. The dimensionality of the domain of this `ConvexFunction`.
      Defaults to `None`.
    dtype: A `tf.dtypes.DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               scale=None,
               ndim=None,
               dtype=None,
               name=None):
    super().__init__(order=1, scale=scale, ndim=ndim, dtype=dtype, name=name)


@api_util.export("convex.ConvexFunctionIndicatorL2Ball")
class ConvexFunctionIndicatorL2Ball(ConvexFunctionIndicatorBall):  # pylint: disable=abstract-method
  """A `ConvexFunction` representing the indicator function of an L2 ball.

  Args:
    scale: A `float`. A scaling factor. Defaults to 1.0.
    ndim: An `int`. The dimensionality of the domain of this `ConvexFunction`.
      Defaults to `None`.
    dtype: A `tf.dtypes.DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               scale=None,
               ndim=None,
               dtype=None,
               name=None):
    super().__init__(order=2, scale=scale, ndim=ndim, dtype=dtype, name=name)


@api_util.export("convex.ConvexFunctionNorm")
class ConvexFunctionNorm(ConvexFunction):  # pylint: disable=abstract-method
  """A `ConvexFunction` computing the [scaled] Lp-norm of a [batch of] inputs.

  Args:
    order: A `float`. The order of the norm. Supported values are `1`, `2`,
      `np.inf`.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    ndim: An `int`. The dimensionality of the domain of this `ConvexFunction`.
      Defaults to `None`.
    dtype: A `tf.dtypes.DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               order,
               scale=None,
               ndim=None,
               dtype=None,
               name=None):
    super().__init__(scale=scale, ndim=ndim, dtype=dtype, name=name)
    self._order = check_util.validate_enum(order, [1, 2, np.inf], name='order')

  def _call(self, x):
    return self._scale * tf.math.real(tf.norm(x, ord=self._order, axis=-1))

  def _prox(self, x, scale=None):
    combined_scale = self._scale
    if scale is not None:
      combined_scale *= tf.cast(scale, self.dtype.real_dtype)

    if self._order == 1:
      return math_ops.soft_threshold(x, combined_scale)
    if self._order == 2:
      return math_ops.block_soft_threshold(x, combined_scale)
    raise NotImplementedError(
        f"The proximal operator of the L{self._order}-norm is not implemented.")

  def _conj(self):
    # The convex conjugate of the Lp-norm is the indicator function on the unit
    # ball defined by the dual norm.
    return ConvexFunctionIndicatorBall(
        order=_conjugate_exponent(self._order),
        scale=self._scale,
        ndim=self.ndim,
        dtype=self.dtype,
        name=f"{self.name}_conj")


@api_util.export("convex.ConvexFunctionL1Norm")
class ConvexFunctionL1Norm(ConvexFunctionNorm):  # pylint: disable=abstract-method
  """A `ConvexFunction` computing the [scaled] L1-norm of a [batch of] inputs.

  Args:
    scale: A `float`. A scaling factor. Defaults to 1.0.
    ndim: An `int`. The dimensionality of the domain of this `ConvexFunction`.
      Defaults to `None`.
    dtype: A `tf.dtypes.DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               scale=None,
               ndim=None,
               dtype=None,
               name=None):
    super().__init__(order=1, scale=scale, ndim=ndim, dtype=dtype, name=name)


@api_util.export("convex.ConvexFunctionL2Norm")
class ConvexFunctionL2Norm(ConvexFunctionNorm):  # pylint: disable=abstract-method
  """A `ConvexFunction` computing the [scaled] L2-norm of a [batch of] inputs.

  Args:
    scale: A `float`. A scaling factor. Defaults to 1.0.
    ndim: An `int`. The dimensionality of the domain of this `ConvexFunction`.
      Defaults to `None`.
    dtype: A `string` or `DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               scale=None,
               ndim=None,
               dtype=None,
               name=None):
    super().__init__(order=2, scale=scale, ndim=ndim, dtype=dtype, name=name)


@api_util.export("convex.ConvexFunctionL2NormSquared")
class ConvexFunctionL2NormSquared(ConvexFunction):  # pylint: disable=abstract-method
  """A `ConvexFunction` computing the [scaled] squared L2-norm of an input.

  Args:
    scale: A `float`. A scaling factor. Defaults to 1.0.
    ndim: An `int`. The dimensionality of the domain of this `ConvexFunction`.
      Defaults to `None`.
    dtype: A `string` or `DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               scale=None,
               ndim=None,
               dtype=None,
               name=None):
    super().__init__(ndim=ndim, dtype=dtype, name=name)
    self._scale = scale

  def _call(self, x):
    return self._scale * tf.math.reduce_sum(x * tf.math.conj(x), axis=-1)

  def _prox(self, x, scale=None):
    combined_scale = self._scale
    if scale is not None:
      combined_scale *= tf.cast(scale, self.dtype.real_dtype)

    return math_ops.shrinkage(x, 2.0 * combined_scale)


@api_util.export("convex.ConvexFunctionTikhonov")
class ConvexFunctionTikhonov(ConvexFunctionAffineMappingComposition):  # pylint: disable=abstract-method
  r"""A `ConvexFunction` representing a Tikhonov regularization term.

  For a given input :math:`x`, computes
  :math:`\lambda \left\| T(x - x_0) \right\|_2^2`, where :math:`\lambda` is a
  scaling factor, :math:`T` is any linear operator and :math:`x_0` is
  a prior estimate.

  Args:
    transform: A `tf.linalg.LinearOperator`. The Tikhonov operator :math:`T`.
      Defaults to the identity operator.
    prior: A `tf.Tensor`. The prior estimate :math:`x_0`. Defaults to 0.
    scale: A `float`. The scaling factor.
    ndim: An `int`. The dimensionality of the domain of this `ConvexFunction`.
      Defaults to `None`.
    dtype: A `tf.DType`. The dtype of the inputs. Defaults to `float32`.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               transform=None,
               prior=None,
               scale=None,
               ndim=None,
               dtype=tf.float32,
               name=None):
    if ndim is None and transform is not None:
      ndim = transform.range_dimension
    function = ConvexFunctionL2NormSquared(scale=scale,
                                           ndim=ndim,
                                           dtype=dtype)
    # Stored only for external access. Not actually used for computation.
    self._transform = transform
    self._prior = prior
    # Convert to affine transform.
    operator = self._transform
    constant = self._prior
    if self._prior is not None:
      constant = tf.math.negative(constant)
      if operator is not None:
        constant = tf.linalg.matvec(operator, constant)
    super().__init__(function,
                     operator=operator,
                     constant=constant,
                     name=name)

  @property
  def transform(self):
    return self._transform

  @property
  def prior(self):
    return self._prior


@api_util.export("convex.ConvexFunctionTotalVariation")
class ConvexFunctionTotalVariation(ConvexFunctionLinearOperatorComposition):  # pylint: disable=abstract-method
  r"""A `ConvexFunction` representing a total variation regularization term.

  For a given input :math:`x`, computes :math:`\lambda \left\| Dx \right\|_1`,
  where :math:`\lambda` is a scaling factor and :math:`D` is the finite
  difference operator.

  Args:
    scale: A `float`. A scaling factor.
    ndim: An `int`, or a list of `ints`. The dimensionality of the domain.
      The domain may have multiple axes.
    axis: An `int` or a list of `ints`. The axes along which to compute the
      total variation. Defaults to -1.
    dtype: A `tf.DType`. The dtype of the inputs.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               scale=None,
               ndim=None,
               axis=-1,
               dtype=tf.float32,
               name=None):
    domain_shape = tf.TensorShape(ndim)
    axis = check_util.validate_axis(axis, domain_shape.rank,
                                    max_length=domain_shape.rank,
                                    canonicalize="negative")
    operators = [linalg_imaging.LinearOperatorFiniteDifference(
        domain_shape, axis=ax, dtype=dtype) for ax in axis]
    operator = linalg_ext.LinearOperatorVerticalStack(operators)
    function = ConvexFunctionL1Norm(scale=scale,
                                    ndim=operator.range_dimension,
                                    dtype=dtype)
    super().__init__(function,
                     operator=operator,
                     name=name)


@api_util.export("convex.ConvexFunctionQuadratic")
class ConvexFunctionQuadratic(ConvexFunction):  # pylint: disable=abstract-method
  r"""A `ConvexFunction` representing a generic quadratic function.

  Represents :math:`f(x) = \frac{1}{2} x^{T} A x + b^{T} x + c`.

  Args:
    quadratic_coefficient: A `tf.Tensor` or a `tf.linalg.LinearOperator`
      representing a self-adjoint, positive definite matrix `A` with shape
      `[..., n, n]`. The coefficient of the quadratic term.
    linear_coefficient: A `tf.Tensor` representing a vector `b` with shape
      `[..., n]`. The coefficient of the linear term.
    constant_coefficient: A scalar `tf.Tensor` representing the constant term
      `c` with shape `[...]`.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               quadratic_coefficient,
               linear_coefficient=None,
               constant_coefficient=None,
               scale=None,
               name=None):
    super().__init__(scale=scale,
                     ndim=quadratic_coefficient.shape[-1],
                     dtype=quadratic_coefficient.dtype,
                     name=name)
    self._quadratic_coefficient = quadratic_coefficient
    self._linear_coefficient = self._validate_linear_coefficient(
        linear_coefficient)
    self._constant_coefficient = self._validate_constant_coefficient(
        constant_coefficient)

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

  def _prox(self, x, scale=None, solver_kwargs=None):  # pylint: disable=arguments-differ
    combined_scale = self._scale
    if scale is not None:
      combined_scale *= tf.cast(scale, self.dtype.real_dtype)
    one_over_scale = tf.cast(1.0 / combined_scale, self.dtype)

    # Operator A^T A + 1 / \lambda * I.
    self._operator = linalg_ext.LinearOperatorAddition([
        self._quadratic_coefficient,
        tf.linalg.LinearOperatorScaledIdentity(
            num_rows=self._quadratic_coefficient.domain_dimension,
            multiplier=one_over_scale)],
        is_self_adjoint=True,
        is_positive_definite=True)

    rhs = one_over_scale * x
    if self._linear_coefficient is not None:
      rhs -= self._linear_coefficient

    solver_kwargs = solver_kwargs or {}
    state = linalg_ops.conjugate_gradient(self._operator, rhs, **solver_kwargs)

    return state.x

  def _validate_linear_coefficient(self, coef):  # pylint: disable=missing-param-doc
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

  def _validate_constant_coefficient(self, coef):  # pylint: disable=missing-param-doc
    """Validates the constant coefficient."""
    if coef.dtype != self.dtype:
      raise ValueError(
          "Expected constant coefficient to have dtype %s, but found: %s in "
          "tensor %s" % (self.dtype, coef.dtype, coef))
    return coef

  def _shape(self):
    """Returns the static shape of this `ConvexFunction`."""
    batch_shape = array_ops.broadcast_static_shapes(
        self._quadratic_coefficient.shape[:-2],
        self._linear_coefficient.shape[:-1],
        self._constant_coefficient.shape)
    return batch_shape.concatenate(tf.TensorShape([self.ndim]))

  def _shape_tensor(self):
    """Returns the dynamic shape of this `ConvexFunction`."""
    batch_shape_tensor = array_ops.broadcast_dynamic_shapes(
        tensor_util.object_shape(self._quadratic_coefficient)[:-2],
        tf.shape(self._linear_coefficient)[:-1],
        tf.shape(self._constant_coefficient))
    return tf.concat([batch_shape_tensor, [self.ndim_tensor()]], 0)

  @property
  def quadratic_coefficient(self):
    return self._quadratic_coefficient

  @property
  def linear_coefficient(self):
    return self._linear_coefficient

  @property
  def constant_coefficient(self):
    return self._constant_coefficient


@api_util.export("convex.ConvexFunctionLeastSquares")
class ConvexFunctionLeastSquares(ConvexFunctionQuadratic):  # pylint: disable=abstract-method
  r"""A `ConvexFunction` representing a least squares function.

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
  def __init__(self, operator, rhs, scale=None, name=None):
    if isinstance(operator, linalg_imaging.LinalgImagingMixin):
      rhs = operator.flatten_range_shape(rhs)
    quadratic_coefficient = tf.linalg.LinearOperatorComposition(
        [operator.H, operator], is_self_adjoint=True, is_positive_definite=True)
    linear_coefficient = tf.math.negative(
        tf.linalg.matvec(operator, rhs, adjoint_a=True))
    constant_coefficient = tf.constant(0.0, dtype=operator.dtype)
    super().__init__(quadratic_coefficient=quadratic_coefficient,
                     linear_coefficient=linear_coefficient,
                     constant_coefficient=constant_coefficient,
                     scale=scale,
                     name=name)

  @property
  def operator(self):
    return self.quadratic_coefficient

  @property
  def rhs(self):
    return tf.math.negative(self.linear_coefficient)


def _dot(x, y):
  """Returns the dot product of `x` and `y`."""
  return tf.squeeze(
      tf.linalg.matvec(
          x[..., tf.newaxis],
          y, adjoint_a=True), axis=-1)


def _conjugate_exponent(exp):
  """Returns the conjugate exponent of `exp`."""
  if exp == 1.0:
    return np.inf
  if exp == np.inf:
    return 1.0
  return exp / (exp - 1.0)
