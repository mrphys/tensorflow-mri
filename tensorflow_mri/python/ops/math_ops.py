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

import tensorflow as tf

from tensorflow_mri.python.utils import check_utils


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
    tensor = check_utils.validate_tensor_dtype(
      tf.convert_to_tensor(tensor),
      (tf.float32, tf.float64, tf.complex64, tf.complex128),
      name='tensor')
    part = check_utils.validate_enum(
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

    tf.debugging.assert_greater(output_max, output_min)

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
