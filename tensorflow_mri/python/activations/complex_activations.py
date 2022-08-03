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

import functools

import tensorflow as tf

from tensorflow_mri.python.util import api_util


def complexified(split='real_imag'):
  """Returns a decorator to create complex-valued activations."""
  if split not in ('real_imag', 'abs_angle'):
    raise ValueError(
        f"split must be one of 'real_imag' or 'abs_angle', but got: {split}")
  def decorator(func):
    @functools.wraps(func)
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
    return wrapper
  return decorator


complex_relu = api_util.export("activations.complex_relu")(
    complexified(split='real_imag')(tf.keras.activations.relu))
