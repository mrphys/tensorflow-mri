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
"""Math operations."""

import functools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util


@api_util.export("math.extract_from_complex")
def extract_from_complex(tensor, part, name='extract_from_complex'):
  """Extract parts from a complex tensor.

  Args:
    tensor: A `Tensor`. Must have type `float32`, `float64`, `complex64` or
      `complex128`.
    part: A `string`. The part of the complex number to extract. Must be one of
      `"real"`, `"imag"`, `"magnitude"`, `"phase"`.
    name: An optional `string`. The name of the op.

  Returns:
    A `Tensor`. The extracted part. Has the same shape as `tensor` and type
    `tensor.dtype.real_dtype`.
  """
  with tf.name_scope(name):
    tensor = check_util.validate_tensor_dtype(
        tf.convert_to_tensor(tensor),
        (tf.float32, tf.float64, tf.complex64, tf.complex128),
        name='tensor')
    part = check_util.validate_enum(
        part, ('magnitude', 'mag', 'phase', 'phs', 'real', 'imaginary', 'imag'),
        'part')

    # Extract the relevant part.
    if part in ('mag', 'magnitude'):
      tensor = tf.math.abs(tensor)
    elif part in ('phs', 'phase'):
      tensor = tf.math.angle(tensor)
    elif part in ('real',):
      tensor = tf.math.real(tensor)
    elif part in ('imag', 'imaginary'):
      tensor = tf.math.imag(tensor)

    return tensor


@api_util.export("math.make_val_and_grad_fn")
def make_val_and_grad_fn(value_fn):
  """Function decorator to compute both function value and gradient.

  Turns function `value_fn` that evaluates and returns a `Tensor` with the value
  of the function evaluated at the input point into one that returns a tuple of
  two `Tensors` with the value and the gradient of the defined function
  evaluated at the input point.

  This is useful for constructing functions for optimization.

  Args:
    value_fn: A Python function to decorate.

  Returns:
    The decorated function.
  """
  @functools.wraps(value_fn)
  def val_and_grad(x):
    return tfp.math.value_and_gradient(value_fn, x)
  return val_and_grad


@api_util.export("math.normalize_no_nan")
def normalize_no_nan(tensor, ord='euclidean', axis=None, name=None):  # pylint: disable=redefined-builtin
  """Normalizes `tensor` along dimension `axis` using specified norm.

  Args:
    tensor: A `Tensor` of type `float32`, `float64`, `complex64`, `complex128`.
    ord: Order of the norm. Supported values are `'fro'`, `'euclidean'`, `1`,
      `2`, `np.inf` and any positive real number yielding the corresponding
      p-norm. Default is `'euclidean'` which is equivalent to Frobenius norm if
      `tensor` is a matrix and equivalent to 2-norm for vectors.
      Some restrictions apply: a) The Frobenius norm `'fro'` is not defined for
        vectors, b) If axis is a 2-tuple (matrix norm), only `'euclidean'`,
        '`fro'`, `1`, `2`, `np.inf` are supported. See the description of `axis`
        on how to compute norms for a batch of vectors or matrices stored in a
        tensor.
    axis: If `axis` is `None` (the default), the input is considered a vector
      and a single vector norm is computed over the entire set of values in the
      tensor, i.e. `norm(tensor, ord=ord)` is equivalent to
      `norm(reshape(tensor, [-1]), ord=ord)`. If `axis` is a Python integer, the
      input is considered a batch of vectors, and `axis` determines the axis in
      `tensor` over which to compute vector norms. If `axis` is a 2-tuple of
      Python integers it is considered a batch of matrices and `axis` determines
      the axes in `tensor` over which to compute a matrix norm.
      Negative indices are supported. Example: If you are passing a tensor that
        can be either a matrix or a batch of matrices at runtime, pass
        `axis=[-2,-1]` instead of `axis=None` to make sure that matrix norms are
        computed.
    name: The name of the op.

  Returns:
    A normalized `Tensor` with the same shape as `tensor`.
  """
  with tf.name_scope(name or 'normalize_no_nan'):
    norm = tf.norm(tensor, ord=ord, axis=axis, keepdims=True)
    norm = tf.cast(norm, tensor.dtype)
    return tf.math.divide_no_nan(tensor, norm)


def scale_by_min_max(tensor,
                     output_min=0.0,
                     output_max=1.0,
                     name='scale_by_min_max'):
  """Rescale tensor values to specified range.

  Values in the input tensor are linearly scaled so that the new minimum value
  is equal to `output_min` and the new maximum value is equal to `output_max`.

  If the input tensor is complex, its magnitude is scaled.

  Args:
    tensor: A `Tensor`. Must have type `float32`, `float64`, `complex64` or
      `complex128`.
    output_min: An optional `float`. The minimum value in the output tensor.
      Defaults to 0.0.
    output_max: An optional `float`. The maximum value in the output tensor.
      Defaults to 1.0.
    name: An optional `string`. The name of the op.

  Returns:
      The rescaled tensor.
  """
  with tf.name_scope(name):
    tensor = tf.convert_to_tensor(tensor)
    output_min = tf.cast(output_min, tensor.dtype.real_dtype)
    output_max = tf.cast(output_max, tensor.dtype.real_dtype)
    scale = output_max - output_min

    checks = [tf.debugging.assert_greater(output_max, output_min)]
    with tf.control_dependencies(checks):
      tensor = tf.identity(tensor)

    def do_rescale(x):
      x = tf.math.divide(
          x - tf.math.reduce_min(x),
          tf.math.reduce_max(x) - tf.math.reduce_min(x))
      x *= scale
      x += output_min
      return x

    if tensor.dtype.is_complex:
      # Rescale magnitude, phase remains unmodified.
      tensor = tf.math.multiply(
          tf.cast(do_rescale(tf.math.abs(tensor)), tensor.dtype),
          tf.math.exp(tf.dtypes.complex(0., tf.math.angle(tensor))))
    else:
      tensor = do_rescale(tensor)

    return tensor


def view_as_complex(x, stacked=True):
  """Returns a view of the input as a complex tensor.

  Returns a new complex-valued input tensor of shape `[M1, M2, ..., Mn]`:

  * If `stacked` is `True`, expects a real-valued tensor of shape
    `[M1, M2, ..., Mn, 2]`, where the last axis has the real and imaginary
    components of the complex numbers.
  * If `stacked` is `False`, expects a real-valued tensor of shape
    `[M1, M2, ..., 2 * Mn], where real and imaginary components are interleaved
    in the channel dimension.

  Args:
    x: A real-valued `Tensor`.
    stacked: A `boolean`. If `True`, real and imaginary components are expected
      to be stacked in their own axis. If `False`, they are expected to be
      interleaved in the channel dimension.

  Returns:
    A complex-valued `Tensor`.
  """
  x = tf.convert_to_tensor(x)
  if not stacked:
    x_shape = tf.shape(x)
    x_shape = tf.concat([x_shape[:-1], [x_shape[-1] // 2], [2]], 0)
    x = tf.reshape(x, x_shape)
  checks = [tf.debugging.assert_equal(tf.shape(x)[-1], 2, message=(
      f"Could not interpret input tensor as complex. Last dimension must be 2, "
      f"but got {tf.shape(x)[-1]}. Perhaps you need to set `stacked` to "
      f"`False`?"))]
  with tf.control_dependencies(checks):
    x = tf.identity(x)
  x = tf.complex(x[..., 0], x[..., 1])
  return x


def view_as_real(x, stacked=True):
  """Returns a view of the input as a real tensor.

  For a complex-valued input tensor of shape `[M1, M2, ..., Mn]`:

  * If `stacked` is `True`, returns a new real-valued tensor of shape
    `[M1, M2, ..., Mn, 2]`, where the last axis has the real and imaginary
    components of the complex numbers.
  * If `stacked` is `False`, returns a new real-valued tensor of shape
    `[M1, M2, ..., 2 * Mn], where real and imaginary components are interleaved
    in the channel dimension.

  Args:
    x: A complex-valued `Tensor`.
    stacked: A `boolean`. If `True`, real and imaginary components are stacked
      along a new axis. If `False`, they are inserted into the channel axis.

  Returns:
    A real-valued `Tensor`.
  """
  x = tf.convert_to_tensor(x)
  static_shape = x.shape
  dynamic_shape = tf.shape(x)

  x = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)

  if not stacked:
    dynamic_shape = tf.concat([dynamic_shape[:-1], 2 * dynamic_shape[-1:]], 0)
    if static_shape[-1] is None:
      static_shape = static_shape[:-1] + [None]
    else:
      static_shape = static_shape[:-1] + [2 * static_shape[-1]]
    x = tf.reshape(x, dynamic_shape)
    x = tf.ensure_shape(x, static_shape)

  return x


@api_util.export("math.block_soft_threshold")
def block_soft_threshold(x, threshold, name=None):
  r"""Block soft thresholding operator.

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
    threshold = tf.convert_to_tensor(threshold, dtype=x.dtype.real_dtype,
                                     name='threshold')
    one = tf.constant(1.0, dtype=x.dtype.real_dtype, name='one')
    reduction_axis = -1 if x.shape.rank > 0 else None
    x_norm = tf.math.real(
        tf.norm(x, ord=2, axis=reduction_axis, keepdims=True))
    return x * tf.cast(tf.math.maximum(
        one - tf.math.divide_no_nan(threshold, x_norm), 0.), x.dtype)


@api_util.export("math.shrinkage")
def shrinkage(x, threshold, name=None):
  r"""Shrinkage operator.

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


@api_util.export("math.soft_threshold")
def soft_threshold(x, threshold, name=None):
  r"""Soft thresholding operator.

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
    threshold = tf.convert_to_tensor(threshold, dtype=x.dtype.real_dtype,
                                     name='threshold')
    return tf.math.sign(x) * tf.cast(
        tf.math.maximum(tf.math.abs(x) - threshold, 0.), x.dtype)


@api_util.export("math.indicator_box")
def indicator_box(x, lower_bound=-1.0, upper_bound=1.0, name=None):
  r"""Indicator function of a box.

  Returns `0` if `x` is in the box, `inf` otherwise.

  The box of radius :math:`r` is defined as the set of points of
  :math:`{R}^{n}` whose components are within the range :math:`[l, u]`.

  .. math::
    \mathcal{C} = \left\{x \in \mathbb{R}^{n} : l \leq x_i \leq u, \forall i = 1, \dots, n \right\}

  Args:
    x: A `tf.Tensor` of shape `[..., n]`.
    lower_bound: A scalar `tf.Tensor`. The lower bound of the box.
      Defaults to -1.
    upper_bound: A scalar `tf.Tensor`. The upper bound of the box.
      Defaults to 1.
    name: A `str`. The name of this operation.

  Returns:
    A `tf.Tensor` of shape `[...]` and dtype equal to `x.dtype.real_dtype`.

  Raises:
    ValueError: If inputs are invalid.
  """  # pylint: disable=line-too-long
  with tf.name_scope(name or 'indicator_box'):
    x = tf.convert_to_tensor(x, name='x')
    lower_bound = tf.convert_to_tensor(
        lower_bound, dtype=x.dtype.real_dtype, name='lower_bound')
    if lower_bound.shape.rank != 0:
      raise ValueError('lower_bound must be a scalar.')
    upper_bound = tf.convert_to_tensor(
        upper_bound, dtype=x.dtype.real_dtype, name='upper_bound')
    if upper_bound.shape.rank != 0:
      raise ValueError('upper_bound must be a scalar.')

    if x.shape.rank == 0:
      within_lbound = tf.math.greater_equal(x, lower_bound)
      within_ubound = tf.math.less_equal(x, upper_bound)
    else:
      within_lbound = tf.math.reduce_all(
          tf.math.greater_equal(x, lower_bound), axis=-1, keepdims=False)
      within_ubound = tf.math.reduce_all(
          tf.math.less_equal(x, upper_bound), axis=-1, keepdims=False)

    zero = tf.constant(0.0, dtype=x.dtype.real_dtype)
    inf = tf.constant(np.inf, dtype=x.dtype.real_dtype)
    cond = tf.math.logical_and(within_lbound, within_ubound)
    return tf.where(cond, zero, inf)


@api_util.export("math.indicator_simplex")
def indicator_simplex(x, radius=1.0, name=None):
  r"""Indicator function of the simplex.

  Returns `0` if `x` is in the simplex, `inf` otherwise.

  The simplex of radius :math:`r` is defined as the set of points of
  :math:`\mathbb{R}^{n}` whose elements are nonnegative and sum up to `r`.

  .. math::
    \Delta_r = \left\{x \in \mathbb{R}^{n} : \sum_{i=1}^{n} x_i = r \text{ and } x_i >= 0, \forall i = 1, \dots, n \right\}

  If :math:`r` is 1, the simplex is also called the unit simplex, standard
  simplex or probability simplex.

  Args:
    x: A `tf.Tensor` of shape `[..., n]`.
    radius: A scalar `tf.Tensor`. The radius of the circumscribed circle of the
      simplex, or the distance to the vertices. Defaults to 1.
    name: A `str`. The name of this operation.

  Returns:
    A `tf.Tensor` of shape `[...]` and dtype equal to `x.dtype.real_dtype`.

  Raises:
    ValueError: If inputs are invalid.
  """  # pylint: disable=line-too-long
  with tf.name_scope(name or 'indicator_ball'):
    x = tf.convert_to_tensor(x, name='x')
    radius = tf.convert_to_tensor(
        radius, dtype=x.dtype.real_dtype, name='radius')
    if radius.shape.rank != 0:
      raise ValueError('radius must be a scalar.')

    if x.shape.rank == 0:
      non_negative = tf.math.greater_equal(x, 0.0)
      sum_equals_radius = tf.math.equal(x, radius)
    else:
      non_negative = tf.math.reduce_all(
          tf.math.greater_equal(x, 0.0), axis=-1, keepdims=False)
      sum_equals_radius = tf.math.equal(
          tf.math.reduce_sum(x, axis=-1, keepdims=False), radius)
    zero = tf.constant(0.0, dtype=x.dtype.real_dtype)
    inf = tf.constant(np.inf, dtype=x.dtype.real_dtype)
    cond = tf.math.logical_and(non_negative, sum_equals_radius)
    return tf.where(cond, zero, inf)


@api_util.export("math.indicator_ball")
def indicator_ball(x, order=2, radius=1.0, name=None):
  r"""Indicator function of the Lp ball.

  Returns `0` if `x` is in the Lp ball, `inf` otherwise.

  The :math:`L_p` ball of radius :math:`r` is defined as the set of points of
  :math:`{R}^{n}` whose distance from the origin, as defined by the :math:`L_p`
  norm, is less than or equal to :math:`r`.

  .. math::
    \mathcal{B}_r = \left\{x \in \mathbb{R}^{n} : \left\|x\right\|_{p} \leq r \right\}

  If :math:`r` is 1, this ball is also called the unit ball of the
  :math`L_p` norm.

  Args:
    x: A `tf.Tensor` of shape `[..., n]`.
    order: A `float`. The order of the norm. Defaults to 2.
    radius: A scalar `tf.Tensor`. The radius of the ball. Defaults to 1.
    name: A `str`. The name of this operation.

  Returns:
    A `tf.Tensor` of shape `[...]` and dtype equal to `x.dtype.real_dtype`.

  Raises:
    ValueError: If inputs are invalid.
  """  # pylint: disable=line-too-long
  with tf.name_scope(name or 'indicator_ball'):
    x = tf.convert_to_tensor(x, name='x')
    radius = tf.convert_to_tensor(
        radius, dtype=x.dtype.real_dtype, name='radius')
    if radius.shape.rank != 0:
      raise ValueError('radius must be a scalar.')

    if x.shape.rank == 0:
      x_norm = tf.math.abs(x)
    else:
      x_norm = tf.math.real(tf.norm(x, ord=order, axis=-1, keepdims=False))
    zero = tf.constant(0.0, dtype=x.dtype.real_dtype)
    inf = tf.constant(np.inf, dtype=x.dtype.real_dtype)
    return tf.where(tf.math.less_equal(x_norm, radius), zero, inf)  # multiplex


@api_util.export("math.project_onto_box")
def project_onto_box(x, lower_bound=-1.0, upper_bound=1.0, name=None):
  """Projects an input vector onto the box.

  Args:
    x: A `tf.Tensor` of shape `[..., n]`.
    lower_bound: A scalar `tf.Tensor` of type `x.dtype.real_dtype`. The lower
      bound of the box. Defaults to -1.0.
    upper_bound: A scalar `tf.Tensor` of type `x.dtype.real_dtype`. The upper
      bound of the box. Defaults to 1.0.
    name: A `str`. The name of this operation.

  Returns:
    A `tf.Tensor` of shape `[..., n]` and dtype equal to `x.dtype`.
  """
  with tf.name_scope(name or 'project_onto_box'):
    return tf.math.minimum(tf.math.maximum(x, lower_bound), upper_bound)


@api_util.export("math.project_onto_simplex")
def project_onto_simplex(x, radius=1.0, name=None):
  """Projects an input vector onto the simplex.

  Args:
    x: A `tf.Tensor` of shape `[..., n]`.
    radius: A scalar `tf.Tensor`. Must have type `x.dtype.real_dtype`. The
      radius of the circumscribed circle of the simplex, or the distance to
      the vertices. Defaults to 1.
    name: A `str`. The name of this operation.

  Returns:
    A `tf.Tensor` of shape `[..., n]` and dtype equal to `x.dtype`.

  Raises:
    ValueError: If inputs are invalid.

  References:
    .. [1] Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008).
      Efficient projections onto the l1-ball for learning in high dimensions.
      In Proceedings of the 25th International Conference on Machine Learning
      (pp. 272-279).
  """
  with tf.name_scope(name or 'project_onto_simplex'):
    x = tf.convert_to_tensor(x, name='x')
    radius = tf.convert_to_tensor(
        radius, dtype=x.dtype.real_dtype, name='radius')
    if radius.shape.rank != 0:
      raise ValueError('radius must be a scalar.')

    if x.shape.rank is None:
      raise ValueError('input must have known rank.')

    if x.shape.rank == 0:
      return radius

    # Sort the input vector[s] in descending order.
    x_sorted = tf.sort(x, axis=-1, direction='DESCENDING')

    # Find the critical indices.
    ndim = tf.shape(x)[-1]  # Dimensionality of inputs.
    j = tf.range(1, ndim + 1)  # [1, 2, ..., n]
    x_sorted_accu = tf.math.cumsum(x_sorted, axis=-1)
    avg = (x_sorted_accu - radius) / tf.cast(j, x.dtype)
    rho = tf.math.reduce_max(tf.where(x_sorted >= avg, j - 1, 0), axis=-1)

    # Compute the threshold.
    threshold = tf.gather(avg, rho, axis=-1, batch_dims=(x.shape.rank - 1))
    threshold = tf.expand_dims(threshold, -1)

    # Compute the projection by shifting and thresholding.
    return tf.math.maximum(x - threshold, 0)


@api_util.export("math.project_onto_ball")
def project_onto_ball(x, order=2, radius=1.0, name=None):
  """Projects an input vector onto the Lp ball.

  Args:
    x: A `tf.Tensor` of shape `[..., n]`.
    order: A `float`. The order of the norm. Must be `1`, `2`, or `np.inf`.
    radius: A scalar `tf.Tensor` of type `x.dtype.real_dtype`. The radius of
      the ball. Defaults to 1.0.
    name: A `str`. The name of this operation.

  Returns:
    A `tf.Tensor` of shape `[..., n]` and dtype equal to `x.dtype`.

  Raises:
    NotImplementedError: If `order` is not `1`, `2`, or `np.inf`.
    ValueError: If inputs are invalid.

  References:
    .. [1] Parikh, N., & Boyd, S. (2014). Proximal algorithms. Foundations and
      Trends in optimization, 1(3), 127-239.

    .. [2] Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008).
      Efficient projections onto the l1-ball for learning in high dimensions.
      In Proceedings of the 25th International Conference on Machine Learning
      (pp. 272-279).
  """
  with tf.name_scope(name or 'project_onto_ball'):
    x = tf.convert_to_tensor(x, name='x')
    radius = tf.convert_to_tensor(
        radius, dtype=x.dtype.real_dtype, name='radius')
    if x.shape.rank is None:
      raise ValueError('input must have known rank.')

    if order == 1:
      proj_simplex = tf.math.sign(x) * project_onto_simplex(
          tf.math.abs(x), radius=radius)
      if x.shape.rank == 0:
        x_norm = tf.math.abs(x)
      else:
        x_norm = tf.math.real(tf.norm(x, ord=1, axis=-1, keepdims=True))
      return tf.where(tf.math.less_equal(x_norm, radius), x, proj_simplex)

    if order == 2:
      if x.shape.rank == 0:
        x_norm = tf.math.abs(x) / radius
      else:
        x_norm = tf.math.real(tf.norm(x, ord=2, axis=-1, keepdims=True))
        x_norm /= radius
      return tf.where(tf.math.less_equal(x_norm, 1.0), x, x / x_norm)

    if order == np.inf:
      # The L-infinity ball is a box.
      return project_onto_box(x, lower_bound=-radius, upper_bound=radius)

    raise NotImplementedError(
        f"Projection onto the L-{order} ball is not implemented.")
