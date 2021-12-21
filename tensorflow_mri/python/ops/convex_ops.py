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
import tensorflow_probability as tfp

from tensorflow_mri.python.ops import image_ops


class ConvexFunction():
  """Base class defining a [batch of] convex operator[s].

  Subclasses should implement the `_call` and `_prox` methods to define the
  forward pass and the proximal mapping, respectively.
  """
  def __init__(self,
               dtype,
               name=None):
    """Initialize this `ConvexFunction`."""
    self._dtype = tf.dtypes.as_dtype(dtype)
    self._name = name or type(self).__name__

  def __call__(self, x):
    return self.call(x)

  def call(self, x, name="call"):
    """Apply this `ConvexFunction` to [batch] input `x`.

    Args:
      x: A `Tensor` with compatible shape and same `dtype` as `self`.
      name: A name for this operation.

    Returns:
      A `Tensor` with the same `dtype` as `self`.
    """
    with self._name_scope(name):
      x = tf.convert_to_tensor(x, name="x")
      self._check_input_dtype(x)
      return self._call(x)

  def prox(self, x, name="prox"):
    """Evaluate the proximal operator of this `ConvexFunction` at point `x`.

    Args:
      x: A `Tensor` with compatible shape and same `dtype` as `self`.
      name: A name for this operation.

    Returns:
      A `Tensor` with the same `dtype` as `self`.
    """
    with self._name_scope(name):
      x = tf.convert_to_tensor(x, name="x")
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

  def _check_input_dtype(self, arg):
    """Check that arg.dtype == self.dtype."""
    if arg.dtype.base_dtype != self.dtype:
      raise TypeError(
          "Expected argument to have dtype %s, but found: %s in tensor %s" %
          (self.dtype, arg.dtype, arg))


class ConvexFunctionL1Norm(ConvexFunction):
  """A `ConvexFunction` computing the L1-norm of a [batch of] inputs.

  Args:
    gamma: A `float`. A scaling factor. Defaults to 1.0.
    dtype: A `string` or `DType`. The type of this operator. Defaults to
      `tf.float32`.
    name: A name for this operator.
  """
  def __init__(self,
               gamma=1.0,
               dtype=tf.float32,
               name=None):

    super().__init__(dtype, name=name)
    self._gamma = gamma

  def _call(self, x):
    """Compute the L1-norm of [batch] input `x`.

    Args:
      x: A `Tensor` of shape `[..., n]`.

    Returns:
      A `Tensor` of shape `[...]`. The L1-norm of each element of `x`.
    """
    return self._gamma * tf.norm(x, ord=1, axis=-1)

  def _prox(self, x):
    """Evaluate the proximal operator of the L1-norm at point `x`.
    
    Args:
      x: A `Tensor` of shape `[..., n]`.

    Returns:
      A `Tensor` of shape `[..., n]`.
    """
    return tfp.math.soft_threshold(x, self._gamma)


class ConvexFunctionL2Norm(ConvexFunction):
  """A `ConvexFunction` computing the L2-norm of a [batch of] inputs.

  Args:
    gamma: A `float`. A scaling factor. Defaults to 1.0.
    dtype: A `string` or `DType`. The type of this operator. Defaults to
      `tf.float32`.
    name: A name for this operator.
  """
  def __init__(self,
               gamma=1.0,
               dtype=tf.float32,
               name=None):

    super().__init__(dtype, name=name)
    self._gamma = gamma

  def _call(self, x):
    """Computes the L2-norm of `x`.

    Args:
      x: A `Tensor` of shape `[..., n]`.

    Returns:
      A `Tensor` of shape `[...]`. The L2-norm of each element of `x`.
    """
    return self._gamma * tf.norm(x, ord=2, axis=-1)

  def _prox(self, x):
    """Evaluates the proximal operator of the L2-norm at `x`."""
    return block_soft_threshold(x, self._gamma)


def soft_threshold(x, threshold, name=None):
  """Soft thresholding operator.

  In the context of proximal gradient methods, this function is the proximal
  operator of the L1-norm.

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


def block_soft_threshold(x, threshold, name=None):
  """Block soft thresholding operator.

  In the context of proximal gradient methods, this function is the proximal
  operator of the L2-norm.

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
