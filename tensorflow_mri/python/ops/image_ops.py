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
"""Image operations.

This module contains functions for N-dimensional image processing.
"""

import tensorflow as tf


def central_crop(tensor, shape):
  """Crop the central region of a tensor.

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. The shape of the region to crop. The size of `shape` must
      be equal to the rank of `tensor`. Any component of `shape` can be set to
      the special value -1 to leave the corresponding dimension unchanged.

  Returns:
    A `Tensor`. Has the same type as `tensor`. The centrally cropped tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  tensor_shape = tf.shape(tensor)

  # Calculate output static shape.
  static_shape = None
  if not isinstance(shape, tf.Tensor):
    if isinstance(shape, tf.TensorShape):
      static_shape = [s if not s == -1 else None for s in shape.as_list()]
    else:
      static_shape = [s if not s == -1 else None for s in shape]
    static_shape = tf.TensorShape(static_shape)
    # Complete any unspecified target dimensions with those of the
    # input tensor, if known.
    static_shape = tf.TensorShape(
      [s_target or s_input for (s_target, s_input) in zip(
        static_shape.as_list(), tensor.shape.as_list())])

  shape = tf.convert_to_tensor(shape)

  # Check that ranks are consistent.
  tf.debugging.assert_equal(tf.rank(tensor), tf.size(shape))

  # Crop the tensor.
  slice_begin = tf.where(
    shape >= 0,
    tf.math.maximum(tensor_shape - shape, 0) // 2,
    0)
  slice_size = tf.where(
    shape >= 0,
    tf.math.minimum(tensor_shape, shape),
    -1)
  tensor = tf.slice(tensor, slice_begin, slice_size)

  # Set static shape.
  tensor = tf.ensure_shape(tensor, static_shape)

  return tensor


def resize_with_crop_or_pad(tensor, shape):
  """Crops and/or pads a tensor to a target shape.

  Pads symmetrically or crops centrally the input tensor as necessary to achieve
  the requested shape.

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. The shape of the output tensor. The size of `shape` must
      be equal to the rank of `tensor`. Any component of `shape` can be set to
      the special value -1 to leave the corresponding dimension unchanged.

  Returns:
    A `Tensor`. Has the same type as `tensor`. The symmetrically padded/cropped
    tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  tensor_shape = tf.shape(tensor)
  shape = tf.convert_to_tensor(shape)

  # First do the cropping.
  tensor = central_crop(tensor, shape)

  # Now pad the tensor.
  pad_left = tf.where(
    shape >= 0,
    tf.math.maximum(shape - tensor_shape, 0) // 2,
    0)
  pad_right = tf.where(
    shape >= 0,
    (tf.math.maximum(shape - tensor_shape, 0) + 1) // 2,
    0)

  tensor = tf.pad(tensor, tf.transpose(tf.stack([pad_left, pad_right]))) # pylint: disable=no-value-for-parameter

  return tensor
