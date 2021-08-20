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

import numpy as np
import tensorflow as tf


def psnr(img1, img2, max_val, rank=None, name='psnr'):
  """Computes the peak signal-to-noise ratio (PSNR) between two N-D images.

  This function operates on batches of multi-channel inputs and returns a PSNR
  value for each image in the batch.

  Arguments:
    img1: A `Tensor`. First batch of images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`.
    img2: A `Tensor`. First batch of images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum the and the minimum allowed values).
    rank: An `int`. The number of spatial dimensions. Must be 2 or 3. Defaults
      to `tf.rank(img1) - 2`. In other words, if rank is not explicitly set,
      `img1` and `img2` should have shape `[batch, height, width, channels]`
      if processing 2D images or `[batch, depth, height, width, channels]` if
      processing 3D images.
    name: Namespace to embed the computation in.

  Returns:
    The scalar PSNR between `img1` and `img2`. The returned tensor has type
    `tf.float32` and shape `batch_shape`.
  """
  with tf.name_scope(name):
    # Need to convert the images to float32. Scale max_val accordingly so that
    # PSNR is computed correctly.
    max_val = tf.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, tf.float32)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)

    # Infer the number of spatial dimensions if not specified by the user, then
    # check that the value is valid.
    if rank is None:
      rank = tf.rank(img1) - 2
    tf.debugging.assert_greater_equal(
        rank, 2,
        message=f"`rank` must be >= 2, but got: {rank}")
    tf.debugging.assert_less_equal(
        rank, 3,
        message=f"`rank` must be <= 3, but got: {rank}")

    mse = tf.math.reduce_mean(
      tf.math.squared_difference(img1, img2), tf.range(-rank-1, 0))

    psnr_val = tf.math.subtract(
      20 * tf.math.log(max_val) / tf.math.log(10.0),
      np.float32(10 / np.log(10)) * tf.math.log(mse),
      name='psnr')

    _, _, checks = _verify_compatible_image_shapes(img1, img2, rank)
    with tf.control_dependencies(checks):
      return tf.identity(psnr_val)


def psnr2d(img1, img2, max_val, name='psnr2d'):
  """Computes the peak signal-to-noise ratio (PSNR) between two 2D images.

  Arguments:
    img1: A `Tensor`. First batch of images. Must have rank >= 3 with shape
      `batch_shape + [height, width, channels]`.
    img2: A `Tensor`. First batch of images. Must have rank >= 3 with shape
      `batch_shape + [height, width, channels]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum the and the minimum allowed values).
    name: Namespace to embed the computation in.

  Returns:
    The scalar PSNR between `img1` and `img2`. The returned tensor has type
    `tf.float32` and shape `batch_shape`.
  """
  return psnr(img1, img2, max_val=max_val, rank=2, name=name)


def psnr3d(img1, img2, max_val, name='psnr3d'):
  """Computes the peak signal-to-noise ratio (PSNR) between two 2D images.

  Arguments:
    img1: A `Tensor`. First batch of images. Must have rank >= 4 with shape
      `batch_shape + [height, width, channels]`.
    img2: A `Tensor`. First batch of images. Must have rank >= 4 with shape
      `batch_shape + [height, width, channels]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum the and the minimum allowed values).
    name: Namespace to embed the computation in.

  Returns:
    The scalar PSNR between `img1` and `img2`. The returned tensor has type
    `tf.float32` and shape `batch_shape`.
  """
  return psnr(img1, img2, max_val=max_val, rank=3, name=name)


def _verify_compatible_image_shapes(img1, img2, rank):
  """Checks if two image tensors are compatible for the given rank.

  Checks if two sets of images have rank at least `rank + 1` (spatial
  dimensions plus channel dimension) and if the last `rank + 1` dimensions
  match.

  Args:
    rank: Rank of the images (number of spatial dimensions).
    img1: Tensor containing the first image batch.
    img2: Tensor containing the second image batch.

  Returns:
    A tuple containing: the first tensor shape, the second tensor shape, and
    a list of tf.debugging.Assert() ops implementing the checks.

  Raises:
    ValueError: When static shape check fails.
  """
  rank_p1 = rank + 1

  shape1 = img1.get_shape().with_rank_at_least(rank_p1)
  shape2 = img2.get_shape().with_rank_at_least(rank_p1)
  shape1[-rank_p1:].assert_is_compatible_with(shape2[-rank_p1:])

  if shape1.ndims is not None and shape2.ndims is not None:
    for dim1, dim2 in zip(reversed(shape1.dims[:-rank_p1]),
                          reversed(shape2.dims[:-rank_p1])):
      if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
        raise ValueError(f"Incompatible image shapes: {shape1} and {shape2}")

  # Now assign shape tensors.
  shape1, shape2 = tf.shape_n([img1, img2])

  checks = []
  checks.append(
      tf.debugging.Assert(
          tf.math.greater_equal(tf.size(shape1), rank_p1), [shape1, shape2],
          summarize=10))
  checks.append(
      tf.debugging.Assert(
          tf.math.reduce_all(
              tf.math.equal(shape1[-rank_p1:], shape2[-rank_p1:])),
          [shape1, shape2],
          summarize=10))

  return shape1, shape2, checks


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
