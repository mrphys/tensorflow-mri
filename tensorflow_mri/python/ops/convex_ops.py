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
from tensorflow_mri.python.util import deprecation
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

  This class exposes three properties to get static shape information:

  * `shape`: The static shape. Calls `_shape`.
  * `domain_dimension`: The static domain dimension, equal to `shape[-1]`.
  * `batch_shape`: The static batch shape, equal to `shape[:-1]`.

  Additionally there are three equivalent methods to get dynamic shape
  information:

  * `shape_tensor`: The dynamic shape. Calls `_shape_tensor`.
  * `domain_dimension_tensor`: The dynamic domain dimension, equal to
    `shape_tensor()[-1]`.
  * `batch_shape_tensor`: The dynamic batch shape, equal to
    `shape_tensor()[:-1]`.

  Subclasses may implement the methods `_shape` and `_shape_tensor` to provide
  custom static and dynamic shape information, respectively.
  """
  def __init__(self,
               domain_dimension=None,
               scale=None,
               dtype=None,
               name=None):
    """Initialize this `ConvexFunction`."""
    if isinstance(domain_dimension, tf.compat.v1.Dimension):
      domain_dimension = domain_dimension.value
    self._domain_dimension = check_util.validate_rank(
        domain_dimension, 'domain_dimension', accept_none=True)

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

  def domain_dimension_tensor(self, name=None):
    """Returns the dynamic domain dimension of this `ConvexFunction`.

    Subclasses get this for free once they implement `_shape_tensor`.

    Args:
      name: A name for this operation (optional).

    Returns:
      A scalar integer `tf.Tensor`.
    """
    with self._name_scope(name or "domain_dimension_tensor"):
      # Prefer to use statically defined domain_dimension if available.
      if isinstance(self.domain_dimension, int):
        return tf.constant(self.domain_dimension, dtype=tf.int32)
      return self.shape_tensor()[-1]

  @deprecation.deprecated(
      '2022-08-07', 'Use `ConvexFunction.domain_dimension_tensor` instead.')
  def ndim_tensor(self, name=None):  # pylint: disable=unused-argument
    """Returns the dynamic domain dimension of this `ConvexFunction`.

    Args:
      name: A name for this operation (optional).

    Returns:
      A scalar integer `tf.Tensor`.
    """
    return self.domain_dimension_tensor()

  def batch_shape_tensor(self, name=None):
    """Returns the dynamic batch shape of this `ConvexFunction`.

    Subclasses get this for free once they implement `_shape_tensor`.

    Args:
      name: A name for this operation (optional).

    Returns:
      A 1D integer `tf.Tensor`.
    """
    with self._name_scope(name or "batch_shape_tensor"):
      # Prefer to use statically defined shape if available.
      if self.batch_shape.is_fully_defined():
        return tensor_util.convert_shape_to_tensor(
            self.batch_shape.as_list(), name="shape")
      return self.shape_tensor()[:-1]

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

    Has a default implementation based on the domain dimension.

    Returns:
      A `tf.TensorShape`.
    """
    return tf.TensorShape([self._domain_dimension])

  def _shape_tensor(self):
    """Returns the dynamic shape of this `ConvexFunction`.

    Returns:
      A 1D integer `tf.Tensor`.
    """
    raise NotImplementedError("`_shape_tensor` is not implemented.")

  @property
  def scale(self):
    """The scaling factor."""
    return self._scale

  @property
  def shape(self):
    """The static shape of this `ConvexFunction`."""
    return self._shape()

  @property
  def domain_dimension(self):
    """The static domain dimension of this `ConvexFunction`."""
    return self.shape[-1]

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
    """Check that arg.shape[-1] is compatible with self.domain_dimension."""
    if arg.shape.rank is None:
      raise ValueError(
          "Expected argument to have known rank, but found: {} "
          "in tensor {}".format(arg.shape.rank, arg))
    if arg.shape.rank < 1:
      raise ValueError(
          "Expected argument to have rank >= 1, but found: {} "
          "in tensor {}".format(arg.shape.rank, arg))
    if not arg.shape[-1:].is_compatible_with([self.domain_dimension]):
      raise ValueError(
          "Expected argument to have last dimension {}, but found: {} in "
          "tensor {}".format(self.domain_dimension, arg.shape[-1], arg))

  def _check_input_dtype(self, arg):
    """Check that arg.dtype == self.dtype."""
    if arg.dtype.base_dtype != self.dtype:
      raise TypeError(
          "Expected argument to have dtype {}, but found: {} "
          "in tensor {}".format(self.dtype, arg.dtype, arg))


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
    domain_dimension = (operator.domain_dimension if operator is not None
                        else function.domain_dimension)
    super().__init__(domain_dimension=domain_dimension,
                     scale=scale,
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
    domain_dimension: A scalar integer `tf.Tensor`. The dimension of the domain.
    order: A `float`. The order of the norm. Supported values are `1`, `2`,
      `np.inf`.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    dtype: A `tf.dtypes.DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               domain_dimension,
               order,
               scale=None,
               dtype=None,
               name=None):
    super().__init__(scale=scale, dtype=dtype, name=name)
    self._order = check_util.validate_enum(order, [1, 2, np.inf], name='order')

    self._domain_dimension_static, self._domain_dimension_dynamic = (
        _get_static_and_dynamic_dimension(domain_dimension))

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
        domain_dimension=self.domain_dimension,
        order=_conjugate_exponent(self._order),
        scale=self._scale,
        dtype=self.dtype,
        name=f"{self.name}_conj")

  def _shape(self):
    return tf.TensorShape([self._domain_dimension_static])

  def _shape_tensor(self):
    return tf.convert_to_tensor(
        [self._domain_dimension_dynamic], dtype=tf.int32)


@api_util.export("convex.ConvexFunctionIndicatorL1Ball")
class ConvexFunctionIndicatorL1Ball(ConvexFunctionIndicatorBall):  # pylint: disable=abstract-method
  """A `ConvexFunction` representing the indicator function of an L1 ball.

  Args:
    domain_dimension: A scalar integer `tf.Tensor`. The dimension of the domain.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    dtype: A `tf.dtypes.DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               domain_dimension,
               scale=None,
               dtype=None,
               name=None):
    super().__init__(domain_dimension=domain_dimension, order=1,
                     scale=scale, dtype=dtype, name=name)


@api_util.export("convex.ConvexFunctionIndicatorL2Ball")
class ConvexFunctionIndicatorL2Ball(ConvexFunctionIndicatorBall):  # pylint: disable=abstract-method
  """A `ConvexFunction` representing the indicator function of an L2 ball.

  Args:
    scale: A `float`. A scaling factor. Defaults to 1.0.
    domain_dimension: A scalar integer `tf.Tensor`. The dimension of the domain.
    dtype: A `tf.dtypes.DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               domain_dimension,
               scale=None,
               dtype=None,
               name=None):
    super().__init__(domain_dimension=domain_dimension, order=2,
                     scale=scale, dtype=dtype, name=name)


@api_util.export("convex.ConvexFunctionNorm")
class ConvexFunctionNorm(ConvexFunction):  # pylint: disable=abstract-method
  """A `ConvexFunction` computing the [scaled] Lp-norm of a [batch of] inputs.

  Args:
    domain_dimension: A scalar integer `tf.Tensor`. The dimension of the domain.
    order: A `float`. The order of the norm. Supported values are `1`, `2`,
      `np.inf`.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    dtype: A `tf.dtypes.DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               domain_dimension,
               order,
               scale=None,
               dtype=None,
               name=None):
    super().__init__(scale=scale, dtype=dtype, name=name)
    self._order = check_util.validate_enum(order, [1, 2, np.inf], name='order')

    self._domain_dimension_static, self._domain_dimension_dynamic = (
        _get_static_and_dynamic_dimension(domain_dimension))

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
        domain_dimension=self.domain_dimension,
        order=_conjugate_exponent(self._order),
        scale=self._scale,
        dtype=self.dtype,
        name=f"{self.name}_conj")

  def _shape(self):
    return tf.TensorShape([self._domain_dimension_static])

  def _shape_tensor(self):
    return tf.convert_to_tensor(
        [self._domain_dimension_dynamic], dtype=tf.int32)


@api_util.export("convex.ConvexFunctionL1Norm")
class ConvexFunctionL1Norm(ConvexFunctionNorm):  # pylint: disable=abstract-method
  """A `ConvexFunction` computing the [scaled] L1-norm of a [batch of] inputs.

  Args:
    domain_dimension: A scalar integer `tf.Tensor`. The dimension of the domain.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    dtype: A `tf.dtypes.DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               domain_dimension,
               scale=None,
               dtype=None,
               name=None):
    super().__init__(domain_dimension=domain_dimension, order=1,
                     scale=scale, dtype=dtype, name=name)


@api_util.export("convex.ConvexFunctionL2Norm")
class ConvexFunctionL2Norm(ConvexFunctionNorm):  # pylint: disable=abstract-method
  """A `ConvexFunction` computing the [scaled] L2-norm of a [batch of] inputs.

  Args:
    domain_dimension: A scalar integer `tf.Tensor`. The dimension of the domain.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    dtype: A `string` or `DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               domain_dimension,
               scale=None,
               dtype=None,
               name=None):
    super().__init__(domain_dimension=domain_dimension, order=2,
                     scale=scale, dtype=dtype, name=name)


@api_util.export("convex.ConvexFunctionL2NormSquared")
class ConvexFunctionL2NormSquared(ConvexFunction):  # pylint: disable=abstract-method
  """A `ConvexFunction` computing the [scaled] squared L2-norm of an input.

  Args:
    domain_dimension: A scalar integer `tf.Tensor`. The dimension of the domain.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    dtype: A `string` or `DType`. The type of this `ConvexFunction`. Defaults to
      `tf.dtypes.float32`.
    name: A name for this `ConvexFunction`.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.
  """
  def __init__(self,
               domain_dimension,
               scale=None,
               dtype=None,
               name=None):
    super().__init__(domain_dimension=domain_dimension,
                     scale=scale, dtype=dtype, name=name)

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
    domain_dimension: A scalar integer `tf.Tensor`. The dimension of the domain.
    scale: A `float`. The scaling factor.
    dtype: A `tf.DType`. The dtype of the inputs. Defaults to `float32`.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               transform=None,
               prior=None,
               domain_dimension=None,
               scale=None,
               dtype=tf.float32,
               name=None):
    if domain_dimension is None and transform is not None:
      domain_dimension = transform.range_dimension
    function = ConvexFunctionL2NormSquared(domain_dimension=domain_dimension,
                                           scale=scale,
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
    domain_shape: A 1D integer `tf.Tensor`. The shape of the domain. Defaults to
      `None`. The domain of this `ConvexFunction` may have multiple axes.
    axes: An `int` or a list of `ints`. The axes along which to compute the
      total variation. If `None` (default), the total variation is computed
      over all axes.
    scale: A `float`. A scaling factor.
    dtype: A `tf.DType`. The dtype of the inputs.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               domain_shape,
               axes=None,
               scale=None,
               dtype=tf.float32,
               name=None):
    domain_shape_static, _ = (
        tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))
    if axes is None:
      if domain_shape_static.rank is None:
        raise NotImplementedError(
            "Rank of domain_shape must be known statically")
      axes = list(range(domain_shape_static.rank))
    if isinstance(axes, int):
      axes = [axes]
    # `LinearOperatorFiniteDifference` operates along one axis only. So for
    # multiple axes, we create one operator for each axis and vertically stack
    # them.
    operators = [linalg_ops.LinearOperatorFiniteDifference(
        domain_shape, axis=axis, dtype=dtype) for axis in axes]
    operator = linalg_ext.LinearOperatorVerticalStack(operators)
    function = ConvexFunctionL1Norm(
        domain_dimension=operator.range_dimension_tensor(),
        scale=scale,
        dtype=dtype)
    super().__init__(function,
                     operator=operator,
                     name=name)


@api_util.export("convex.ConvexFunctionL1Wavelet")
class ConvexFunctionL1Wavelet(ConvexFunctionLinearOperatorComposition):  # pylint: disable=abstract-method
  r"""A `ConvexFunction` representing an L1 wavelet regularization term.

  For a given input :math:`x`, computes :math:`\lambda \left\| Dx \right\|_1`,
  where :math:`\lambda` is a scaling factor and :math:`D` is a wavelet
  decomposition operator (see `tfmri.linalg.LinearOperatorWavelet`).

  Args:
    domain_shape: A 1D integer `tf.Tensor`. The domain shape of this linear
      operator. This operator may have multiple domain dimensions.
    wavelet: A `str` or a `pywt.Wavelet`_, or a `list` thereof. When passed a
      `list`, different wavelets are applied along each axis in `axes`.
    mode: A `str`. The padding or signal extension mode. Must be one of the
      values supported by `tfmri.signal.wavedec`. Defaults to `'symmetric'`.
    level: An `int` >= 0. The decomposition level. If `None` (default),
      the maximum useful level of decomposition will be used (see
      `tfmri.signal.max_wavelet_level`).
    axes: A `list` of `int`. The axes over which the DWT is computed. Axes refer
      only to domain dimensions without regard for the batch dimensions.
      Defaults to `None` (all domain dimensions).
    scale: A `float`. A scaling factor.
    dtype: A `tf.dtypes.DType`. The dtype of the inputs.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self,
               domain_shape,
               wavelet,
               mode='symmetric',
               level=None,
               axes=None,
               scale=None,
               dtype=tf.dtypes.float32,
               name=None):
    operator = linalg_ops.LinearOperatorWavelet(domain_shape,
                                                wavelet,
                                                mode=mode,
                                                level=level,
                                                axes=axes,
                                                dtype=dtype)
    function = ConvexFunctionL1Norm(
        domain_dimension=operator.range_dimension_tensor(),
        scale=scale,
        dtype=dtype)
    super().__init__(function, operator, name=name)

  def _shape(self):
    return tf.TensorShape([self.operator.shape[-1]])

  def _shape_tensor(self):
    return tf.convert_to_tensor(
        [self.operator.shape_tensor()[-1]], dtype=tf.int32)


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
    super().__init__(domain_dimension=quadratic_coefficient.shape[-1],
                     scale=scale,
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
    if not coef.shape[-1:].is_compatible_with([self._domain_dimension]):
      raise ValueError(
          "Expected linear coefficient to have last dimension %d, but found: "
          "%d in tensor %s" % (self._domain_dimension, coef.shape[-1], coef))
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
    return batch_shape.concatenate(tf.TensorShape([self._domain_dimension]))

  def _shape_tensor(self):
    """Returns the dynamic shape of this `ConvexFunction`."""
    batch_shape_tensor = array_ops.broadcast_dynamic_shapes(
        tensor_util.object_shape(self._quadratic_coefficient)[:-2],
        tf.shape(self._linear_coefficient)[:-1],
        tf.shape(self._constant_coefficient))
    return tf.concat([batch_shape_tensor, [self._domain_dimension]], 0)

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
    operator: A `tf.Tensor` or a `tfmri.linalg.LinearOperator` representing a
      matrix :math:`A` with shape `[..., m, n]`. The linear system operator.
    rhs: A `Tensor` representing a vector `b` with shape `[..., m]`. The
      right-hand side of the linear system.
    gram_operator: A `tf.Tensor` or a `tfmri.linalg.LinearOperator` representing
      the Gram matrix of `operator`. This may be used to provide a specialized
      implementation of the Gram matrix :math:`A^H A`. Defaults to `None`, in
      which case a naive implementation of the Gram matrix is derived from
      `operator`.
    scale: A `float`. A scaling factor. Defaults to 1.0.
    name: A name for this `ConvexFunction`.
  """
  def __init__(self, operator, rhs, gram_operator=None, scale=None, name=None):
    if isinstance(operator, linalg_imaging.LinalgImagingMixin):
      rhs = operator.flatten_range_shape(rhs)
    if gram_operator:
      quadratic_coefficient = gram_operator
    else:
      quadratic_coefficient = tf.linalg.LinearOperatorComposition(
          [operator.H, operator],
          is_self_adjoint=True, is_positive_definite=True)
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


def _get_static_and_dynamic_dimension(dim):  # pylint: disable=missing-param-doc
  """Returns the static and dynamic information from `dim`."""
  # Get static dimension.
  dim_static = tf.get_static_value(dim)
  if dim_static is not None:
    if isinstance(dim_static, np.ndarray):
      try:
        dim_static = dim_static.item()
      except ValueError as err:
        raise ValueError(
            f"domain_dimension must be a scalar integer, "
            f"but got: {dim_static} (type: {type(dim_static)})") from err
    if isinstance(dim_static, (np.int32, np.int64)):
      dim_static = dim_static.item()
    if isinstance(dim_static, tf.compat.v1.Dimension):
      dim_static = dim_static.value
    if not isinstance(dim_static, int):
      raise ValueError(
          f"domain_dimension must be a scalar integer, "
          f"but got: {dim_static} (type: {type(dim_static)})")

  # Get dynamic dimension.
  dim_dynamic = tf.convert_to_tensor(dim, dtype=tf.int32)
  if dim_dynamic.shape.rank != 0:
    raise ValueError(
        f"domain_dimension must be a scalar integer, "
        f"but got: {dim_dynamic} (type: {type(dim_dynamic)})")

  return dim_static, dim_dynamic
