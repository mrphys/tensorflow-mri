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

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_mri.python.util import check_util


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
    stacked: A `bool`. If `True`, real and imaginary components are expected to
      be stacked in their own axis. If `False`, they are expected to be
      interleaved in the channel dimension.

  Returns:
    A complex-valued `Tensor`.
  """
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
    stacked: A `bool`. If `True`, real and imaginary components are stacked
      along a new axis. If `False`, they are inserted into the channel axis.

  Returns:
    A real-valued `Tensor`.
  """
  x = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
  if not stacked:
    x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  return x


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
