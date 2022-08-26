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
"""Complex-valued activations."""

import inspect

import tensorflow as tf

from tensorflow_mri.python.util import api_util


def complexified(name, split='real_imag'):
  """Returns a decorator to create complex-valued activations."""
  if split not in ('real_imag', 'abs_angle'):
    raise ValueError("split must be one of 'real_imag' or 'abs_angle', but got:"+split)
  def decorator(func):
    def wrapper(x, *args, **kwargs):
      x = tf.convert_to_tensor(x)
      if x.dtype.is_complex:
        if split == 'abs_angle':
          return (tf.cast(func(tf.math.abs(x), *args, **kwargs), x.dtype) *
                  tf.math.exp(1j * tf.math.angle(x)))
        if split == 'real_imag':
          return tf.dtypes.complex(func(tf.math.real(x), *args, **kwargs),
                                   func(tf.math.imag(x), *args, **kwargs))
      return func(x, *args, **kwargs)
    wrapper.__name__ = name
    wrapper.__signature__ = inspect.signature(func)
    return wrapper
  return decorator



complex_relu = api_util.export("activations.complex_relu")(
    complexified(name='complex_relu', split='real_imag')(tf.keras.activations.relu))
complex_relu.__doc__ = (
    """Applies the rectified linear unit activation function.

    With default values, this returns the standard ReLU activation:
    `max(x, 0)`, the element-wise maximum of 0 and the input tensor.

    Modifying default parameters allows you to use non-zero thresholds,
    change the max value of the activation, and to use a non-zero multiple of
    the input for values below the threshold.

    If passed a complex-valued tensor, the ReLU activation is independently
    applied to its real and imaginary parts, i.e., the function returns
    `relu(real(x)) + 1j * relu(imag(x))`.

    ```{note}
    This activation does not preserve the phase of complex inputs.
    ```

    If passed a real-valued tensor, this function falls back to the standard
    `tf.keras.activations.relu`.

    Args:
      x: The input `tf.Tensor`. Can be real or complex.
      alpha: A `float` that governs the slope for values lower than the
        threshold.
      max_value: A `float` that sets the saturation threshold (the largest value
        the function will return).
      threshold: A `float` giving the threshold value of the activation function
        below which values will be damped or set to zero.

    Returns:
      A `tf.Tensor` of the same shape and dtype of input `x`.

    References:
      1. https://arxiv.org/abs/1705.09792
    """
)


mod_relu = api_util.export("activations.mod_relu")(
    complexified(name='mod_relu', split='abs_angle')(tf.keras.activations.relu))
mod_relu.__doc__ = (
    """Applies the rectified linear unit activation function.

    With default values, this returns the standard ReLU activation:
    `max(x, 0)`, the element-wise maximum of 0 and the input tensor.

    Modifying default parameters allows you to use non-zero thresholds,
    change the max value of the activation, and to use a non-zero multiple of
    the input for values below the threshold.

    If passed a complex-valued tensor, the ReLU activation is applied to its
    magnitude, i.e., the function returns `relu(abs(x)) * exp(1j * angle(x))`.

    ```{note}
    This activation preserves the phase of complex inputs.
    ```

    ```{warning}
    With default parameters, this activation is linear, since the magnitude
    of the input is never negative. Usually you will want to set one or more
    of the provided parameters to non-default values.
    ```

    If passed a real-valued tensor, this function falls back to the standard
    `tf.keras.activations.relu`.

    Args:
      x: The input `tf.Tensor`. Can be real or complex.
      alpha: A `float` that governs the slope for values lower than the
        threshold.
      max_value: A `float` that sets the saturation threshold (the largest value
        the function will return).
      threshold: A `float` giving the threshold value of the activation function
        below which values will be damped or set to zero.

    Returns:
      A `tf.Tensor` of the same shape and dtype of input `x`.

    References:
      1. https://arxiv.org/abs/1705.09792
    """
)
