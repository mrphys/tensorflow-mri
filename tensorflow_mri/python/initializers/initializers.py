# Copyright 2022 University College London. All Rights Reserved.
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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras initializers.

Contains complex-valued extensions of Keras initializers.
"""

import math
import string

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.util import api_util


_PARTITION_SHAPE = 'partition_shape'
_PARTITION_OFFSET = 'partition_offset'
_ALLOWED_INITIALIZER_KWARGS = [_PARTITION_SHAPE, _PARTITION_OFFSET]


EXTENSION_NOTE = string.Template("""

  .. note::
    This initializer can be used as a drop-in replacement for
    `tf.keras.initializers.${name}`_. However, this one also supports
    initialization of complex-valued weights. Simply pass `dtype='complex64'`
    or `dtype='complex128'` to its `__call__` method.

  .. _tf.keras.initializers.${name}: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/${name}

""")


def complex_variance_scaling(base):
  """Adds complex-valued support to Keras variance scaling initializers.

  Args:
    base: The base class to be extended. Must be a subclass of
      `tf.keras.initializers.VarianceScaling`.

  Returns:
    A subclass of `base` that supports complex-valued initialization.

  Raises:
    ValueError: if `base` is not a subclass of
    `tf.keras.initializers.VarianceScaling`.
  """
  if not issubclass(base, tf.keras.initializers.VarianceScaling):
    raise ValueError(
        f'Expected base class to be a subclass of '
        f'`tf.keras.initializers.VarianceScaling`, but got {base}.')

  # We override the initializer's __call__ method.
  def __call__(self, shape, dtype=None, **kwargs):  # pylint: disable=invalid-name
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point and complex types
        are supported. If not specified, `tf.keras.backend.floatx()` is used,
        which defaults to `float32` unless you configured it otherwise (via
        `tf.keras.backend.set_floatx(float_dtype)`)
      **kwargs: Additional keyword arguments.

    Returns:
      A tensor object initialized as specified by the initializer.
    """
    # pylint: disable=protected-access,no-else-return
    _validate_kwargs(self.__class__.__name__, kwargs)
    dtype = _assert_float_or_complex_dtype(dtype)
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape)
    if _PARTITION_SHAPE in kwargs:  # pylint: disable=consider-using-get
      shape = kwargs[_PARTITION_SHAPE]
    # Compute required variance (in `scale`).
    if self.mode == 'fan_in':
      scale /= max(1., fan_in)
    elif self.mode == 'fan_out':
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == 'truncated_normal':
      if dtype.is_complex:
        # constant is stddev of complex standard normal truncated to 2
        stddev = math.sqrt(scale) / .95311164380491208
        return stddev * _complex_truncated_normal(
            self._random_generator, shape, 2.0, dtype)
      else:
        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        stddev = math.sqrt(scale) / .87962566103423978
        return self._random_generator.truncated_normal(
            shape, 0.0, stddev, dtype)
    elif self.distribution == 'untruncated_normal':
      if dtype.is_complex:
        stddev = math.sqrt(scale)
        return stddev * _complex_normal(self._random_generator, shape, dtype)
      else:
        stddev = math.sqrt(scale)
        return self._random_generator.random_normal(shape, 0.0, stddev, dtype)
    else:
      if dtype.is_complex:
        stddev = math.sqrt(scale)
        return stddev * _complex_uniform(self._random_generator, shape, dtype)
      else:
        limit = math.sqrt(3.0 * scale)
        return self._random_generator.random_uniform(
            shape, -limit, limit, dtype)

  # Dynamically create a subclass of `base` with the same name as `base` and
  # with the overriden `__call__` method.
  subclass = type(base.__name__, (base,), {'__call__': __call__})

  # Copy docs from the base class, adding the extra note.
  docstring = base.__doc__
  doclines = docstring.split('\n')
  doclines[1:1] = EXTENSION_NOTE.substitute(name=base.__name__).splitlines()
  subclass.__doc__ = '\n'.join(doclines)

  return subclass


# Define the variance scaling initializers. We use a composition of three
# decorators:
#  1. `complex_variance_scaling`: Adds complex-valued support to a Keras
#     variance scaling initializer.
#  2. `register_keras_serializable`: Registers the new initializer with the
#     Keras serialization framework.
#  3. `export`: Exports the new initializer to the TFMRI API.
VarianceScaling = api_util.export("initializers.VarianceScaling")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_variance_scaling(tf.keras.initializers.VarianceScaling)))


GlorotNormal = api_util.export("initializers.GlorotNormal")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_variance_scaling(tf.keras.initializers.GlorotNormal)))


GlorotUniform = api_util.export("initializers.GlorotUniform")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_variance_scaling(tf.keras.initializers.GlorotUniform)))


HeNormal = api_util.export("initializers.HeNormal")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_variance_scaling(tf.keras.initializers.HeNormal)))


HeUniform = api_util.export("initializers.HeUniform")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_variance_scaling(tf.keras.initializers.HeUniform)))


LecunNormal = api_util.export("initializers.LecunNormal")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_variance_scaling(tf.keras.initializers.LecunNormal)))


LecunUniform = api_util.export("initializers.LecunUniform")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_variance_scaling(tf.keras.initializers.LecunUniform)))


def _complex_uniform(rng, shape, dtype):
  """Samples random values from a disk on the complex plane.

  The sampled uniform distribution has zero mean and unit variance.

  Args:
    rng: A `keras.backend.RandomGenerator`.
    shape: The shape of the output tensor.
    dtype: The dtype of the output tensor. Must be complex.

  Returns:
    A tensor of shape `shape` and dtype `dtype`.
  """
  radius = tf.math.sqrt(rng.random_uniform(shape, 0.0, 2.0, dtype.real_dtype))
  theta = rng.random_uniform(shape, 0.0, 2 * np.pi, dtype.real_dtype)
  return tf.cast(radius, dtype) * tf.math.exp(tf.dtypes.complex(0.0, theta))


def _complex_normal(rng, shape, dtype):
  """Samples random values from normal distribution on the complex plane.

  The sampled distribution has zero mean and unit variance.

  Args:
    rng: A `keras.backend.RandomGenerator`.
    shape: The shape of the output tensor.
    dtype: The dtype of the output tensor. Must be complex.

  Returns:
    A tensor of shape `shape` and dtype `dtype`.
  """
  sqrt2 = tf.math.sqrt(tf.constant(2.0, dtype=dtype.real_dtype))
  real = rng.random_normal(shape, 0.0, 1.0, dtype=dtype.real_dtype) / sqrt2
  imag = rng.random_normal(shape, 0.0, 1.0, dtype=dtype.real_dtype) / sqrt2
  return tf.dtypes.complex(real, imag)


def _complex_truncated_normal(rng, shape, upper, dtype):
  """Samples random values from truncated normal on the complex plane.

  The modulus is truncated to `upper`. The distribution has zero mean and unit
  variance prior to the truncation.

  Args:
    rng: A `keras.backend.RandomGenerator`.
    shape: The shape of the output tensor.
    upper: The upper bound on the modulus (truncation).
    dtype: The dtype of the output tensor. Must be complex.

  Returns:
    A tensor of shape `shape` and dtype `dtype`.
  """
  t = ((1 - tf.math.exp(tf.constant(-(upper ** 2), dtype.real_dtype))) *
       rng.random_uniform(shape, dtype=dtype.real_dtype))
  radius = tf.math.sqrt(-tf.math.log(1 - t))  # pylint: disable=invalid-unary-operand-type
  theta = rng.random_uniform(shape, 0.0, 2 * np.pi, dtype.real_dtype)
  return tf.cast(radius, dtype) * tf.math.exp(tf.dtypes.complex(0.0, theta))


def _assert_float_or_complex_dtype(dtype):
  """Validate and return floating or complex point type based on `dtype`.

  `dtype` must be a floating point or complex type.

  Args:
    dtype: The data type to validate.

  Returns:
    Validated type.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  if dtype is None:
    dtype = tf.keras.backend.floatx()
  dtype = tf.as_dtype(dtype)
  if not (dtype.is_floating or dtype.is_complex):
    raise ValueError(f'Expected floating point type, got {dtype}.')
  return dtype


def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.

  Args:
    shape: Integer shape tuple or TF tensor shape.

  Returns:
    A tuple of integer scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)


def _validate_kwargs(cls_name, kwargs, support_partition=True):
  for kwarg in kwargs:
    if kwarg not in _ALLOWED_INITIALIZER_KWARGS:  # pylint: disable=no-else-raise
      raise TypeError(f'Unknown keyword arguments: {kwarg}. Allowed keyword '
                      f'arguments: {_ALLOWED_INITIALIZER_KWARGS}.')
    elif not support_partition:
      raise ValueError(f'{cls_name} initializer doesn\'t support '
                       'partition-related arguments.')
