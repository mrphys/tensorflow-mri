# Copyright 2021 University College London. All Rights Reserved.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# Some of the code in this file is adapted from
# tensorflow/python/ops/image_ops_impl.py to support 2D and 3D processing.

import collections

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import geom_ops
from tensorflow_mri.python.util import check_util


def psnr(img1, img2, max_val=None, rank=None, name='psnr'):
  """Computes the peak signal-to-noise ratio (PSNR) between two N-D images.

  This function operates on batches of multi-channel inputs and returns a PSNR
  value for each image in the batch.

  Arguments:
    img1: A `Tensor`. First batch of images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. Can have integer or
      floating point type, with values in the range `[0, max_val]`.
    img2: A `Tensor`. Second batch of images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. Can have integer or
      floating point type, with values in the range `[0, max_val]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
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
    img1 = tf.convert_to_tensor(img1)
    img2 = tf.convert_to_tensor(img2)
    # Default `max_val` to maximum dynamic range for the input dtype.
    if max_val is None:
      max_val = _image_dynamic_range(img1.dtype)
    # Need to convert the images to float32. Scale max_val accordingly so that
    # PSNR is computed correctly.
    max_val = tf.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, tf.float32)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)

    # Infer the number of spatial dimensions if not specified by the user, then
    # check that the value is valid.
    if rank is None:
      rank = img1.shape.rank - 2 # Rank must be defined statically.

    mse = tf.math.reduce_mean(
      tf.math.squared_difference(img1, img2), tf.range(-rank-1, 0)) # pylint: disable=invalid-unary-operand-type

    psnr_val = tf.math.subtract(
      20 * tf.math.log(max_val) / tf.math.log(10.0),
      np.float32(10 / np.log(10)) * tf.math.log(mse),
      name='psnr')

    _, _, checks = _verify_compatible_image_shapes(img1, img2, rank)
    with tf.control_dependencies(checks):
      return tf.identity(psnr_val)


def psnr2d(img1, img2, max_val=None, name='psnr2d'):
  """Computes the peak signal-to-noise ratio (PSNR) between two 2D images.

  This function operates on batches of multi-channel inputs and returns a PSNR
  value for each image in the batch.

  Arguments:
    img1: A `Tensor`. First batch of images. Must have rank >= 3 with shape
      `batch_shape + [height, width, channels]`. Can have integer or
      floating point type, with values in the range `[0, max_val]`.
    img2: A `Tensor`. Second batch of images. Must have rank >= 3 with shape
      `batch_shape + [height, width, channels]`. Can have integer or
      floating point type, with values in the range `[0, max_val]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    name: Namespace to embed the computation in.

  Returns:
    The scalar PSNR between `img1` and `img2`. The returned tensor has type
    `tf.float32` and shape `batch_shape`.
  """
  return psnr(img1, img2, max_val=max_val, rank=2, name=name)


def psnr3d(img1, img2, max_val, name='psnr3d'):
  """Computes the peak signal-to-noise ratio (PSNR) between two 3D images.

  This function operates on batches of multi-channel inputs and returns a PSNR
  value for each image in the batch.

  Arguments:
    img1: A `Tensor`. First batch of images. Must have rank >= 4 with shape
      `batch_shape + [height, width, channels]`. Can have integer or
      floating point type, with values in the range `[0, max_val]`.
    img2: A `Tensor`. Second batch of images. Must have rank >= 4 with shape
      `batch_shape + [height, width, channels]`. Can have integer or
      floating point type, with values in the range `[0, max_val]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    name: Namespace to embed the computation in.

  Returns:
    The scalar PSNR between `img1` and `img2`. The returned tensor has type
    `tf.float32` and shape `batch_shape`.
  """
  return psnr(img1, img2, max_val=max_val, rank=3, name=name)


def ssim(img1,
         img2,
         max_val=None,
         filter_size=11,
         filter_sigma=1.5,
         k1=0.01,
         k2=0.03,
         rank=None,
         name='ssim'):
  """Computes the structural similarity index (SSIM) between two N-D images.

  This function operates on batches of multi-channel inputs and returns an SSIM
  value for each image in the batch.

  .. note::
    As of TensorFlow 2.6.0, 3D inputs with `channels` > 1 can only be processed
    on GPU.

  Args:
    img1: A `Tensor`. First batch of images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `height`, `width` and
      `depth` must be greater than or equal to `filter_size`. Can have integer
      or floating point type, with values in the range `[0, max_val]`.
    img2: A `Tensor`. Second batch of images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `height`, `width` and
      `depth` must be greater than or equal to `filter_size`. Can have integer
      or floating point type, with values in the range `[0, max_val]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    filter_size: The size of the Gaussian filter. Defaults to 11.
    filter_sigma: The standard deviation of the Gaussian filter. Defaults to
      1.5.
    k1: Factor used to calculate the regularization constant for the luminance
      term, as `C1 = (k1 * max_val) ** 2`. Defaults to 0.01.
    k2: Factor used to calculate the regularization constant for the contrast
      term, as `C2 = (k2 * max_val) ** 2`. Defaults to 0.03.
    rank: An `int`. The number of spatial dimensions. Must be 2 or 3. Defaults
      to `tf.rank(img1) - 2`. In other words, if rank is not explicitly set,
      `img1` and `img2` should have shape `[batch, height, width, channels]`
      if processing 2D images or `[batch, depth, height, width, channels]` if
      processing 3D images.
    name: Namespace to embed the computation in.

  Returns:
    A `Tensor` of type `float32` and shape `batch_shape` containing an SSIM
    value for each image in the batch.

  References:
    .. [1] Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image
      quality assessment: from error visibility to structural similarity," in
      IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April
      2004, doi: 10.1109/TIP.2003.819861.
  """
  with tf.name_scope(name):
    img1 = tf.convert_to_tensor(img1)
    img2 = tf.convert_to_tensor(img2)
    # Default `max_val` to maximum dynamic range for the input dtype.
    if max_val is None:
      max_val = _image_dynamic_range(img1.dtype)
    # Need to convert the images to float32. Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = tf.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, tf.float32)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)

    # Infer the number of spatial dimensions if not specified by the user, then
    # check that the value is valid.
    if rank is None:
      rank = img1.shape.rank - 2 # Rank must be defined statically.

    # Check shapes.
    _, _, checks = _verify_compatible_image_shapes(img1, img2, rank)
    with tf.control_dependencies(checks):
      img1 = tf.identity(img1)

    ssim_per_channel, _ = _ssim_per_channel(
      img1, img2, max_val, filter_size, filter_sigma, k1, k2, rank)

    # Compute average over color channels.
    return tf.math.reduce_mean(ssim_per_channel, [-1])


def ssim2d(img1,
           img2,
           max_val=None,
           filter_size=11,
           filter_sigma=1.5,
           k1=0.01,
           k2=0.03,
           name='ssim2d'):
  """Computes the structural similarity index (SSIM) between two 2D images.

  This function operates on batches of multi-channel inputs and returns an SSIM
  value for each image in the batch.

  Args:
    img1: A `Tensor`. First batch of images. Must have rank >= 3 with shape
      `batch_shape + [height, width, channels]`. `height` and `width`
      must be greater than or equal to `filter_size`. Can have integer or
      floating point type, with values in the range `[0, max_val]`.
    img2: A `Tensor`. Second batch of images. Must have rank >= 3 with shape
      `batch_shape + [height, width, channels]`. `height` and `width`
      must be greater than or equal to `filter_size`. Can have integer or
      floating point type, with values in the range `[0, max_val]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    filter_size: The size of the Gaussian filter. Defaults to 11.
    filter_sigma: The standard deviation of the Gaussian filter. Defaults to
      1.5.
    k1: Factor used to calculate the regularization constant for the luminance
      term, as `C1 = (k1 * max_val) ** 2`. Defaults to 0.01.
    k2: Factor used to calculate the regularization constant for the contrast
      term, as `C2 = (k2 * max_val) ** 2`. Defaults to 0.03.
    name: Namespace to embed the computation in.

  Returns:
    A `Tensor` of type `float32` and shape `batch_shape` containing an SSIM
    value for each image in the batch.

  References:
    .. [1] Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image
      quality assessment: from error visibility to structural similarity," in
      IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April
      2004, doi: 10.1109/TIP.2003.819861.
  """
  return ssim(img1,
              img2,
              max_val=max_val,
              filter_size=filter_size,
              filter_sigma=filter_sigma,
              k1=k1,
              k2=k2,
              rank=2,
              name=name)


def ssim3d(img1,
           img2,
           max_val=None,
           filter_size=11,
           filter_sigma=1.5,
           k1=0.01,
           k2=0.03,
           name='ssim3d'):
  """Computes the structural similarity index (SSIM) between two 3D images.

  This function operates on batches of multi-channel inputs and returns an SSIM
  value for each image in the batch.

  .. note::
    As of TensorFlow 2.6.0, 3D inputs with `channels` > 1 can only be processed
    on GPU.

  Args:
    img1: A `Tensor`. First batch of images. Must have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `depth`, `height` and
      `width` must be greater than or equal to `filter_size`. Can have integer
      or floating point type, with values in the range `[0, max_val]`.
    img2: A `Tensor`. Second batch of images. Must have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `depth`, `height` and
      `width` must be greater than or equal to `filter_size`. Can have integer
      or floating point type, with values in the range `[0, max_val]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    filter_size: The size of the Gaussian filter. Defaults to 11.
    filter_sigma: The standard deviation of the Gaussian filter. Defaults to
      1.5.
    k1: Factor used to calculate the regularization constant for the luminance
      term, as `C1 = (k1 * max_val) ** 2`. Defaults to 0.01.
    k2: Factor used to calculate the regularization constant for the contrast
      term, as `C2 = (k2 * max_val) ** 2`. Defaults to 0.03.
    name: Namespace to embed the computation in.

  Returns:
    A `Tensor` of type `float32` and shape `batch_shape` containing an SSIM
    value for each image in the batch.

  References:
    .. [1] Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image
      quality assessment: from error visibility to structural similarity," in
      IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April
      2004, doi: 10.1109/TIP.2003.819861.
  """
  return ssim(img1,
              img2,
              max_val=max_val,
              filter_size=filter_size,
              filter_sigma=filter_sigma,
              k1=k1,
              k2=k2,
              rank=3,
              name=name)


# Default values obtained by Wang et al.
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def ssim_multiscale(img1,
                    img2,
                    max_val=None,
                    power_factors=_MSSSIM_WEIGHTS,
                    filter_size=11,
                    filter_sigma=1.5,
                    k1=0.01,
                    k2=0.03,
                    rank=None,
                    name='ssim_multiscale'):
  """Computes the multiscale SSIM (MS-SSIM) between two N-D images.

  This function operates on batches of multi-channel inputs and returns an
  MS-SSIM value for each image in the batch.

  .. note::
    As of TensorFlow 2.6.0, 3D inputs with `channels` > 1 can only be processed
    on GPU.

  Args:
    img1: A `Tensor`. First batch of images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `height`, `width` and
      `depth` must be greater than or equal to
      `(filter_size - 1) * 2 ** (len(power_factors) - 1) + 1`. Can have integer
      or floating point type, with values in the range `[0, max_val]`.
    img2: A `Tensor`. Second batch of images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `height`, `width` and
      `depth` must be greater than or equal to
      `(filter_size - 1) * 2 ** (len(power_factors) - 1) + 1`. Can have integer
      or floating point type, with values in the range `[0, max_val]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    power_factors: A list of weights for each of the scales. The length of the
      list determines the number of scales. Index 0 is the unscaled resolution's
      weight and each increasing scale corresponds to the image being
      downsampled by 2. Defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
      which are the values obtained in the original paper [1]_.
    filter_size: The size of the Gaussian filter. Defaults to 11.
    filter_sigma: The standard deviation of the Gaussian filter. Defaults to
      1.5.
    k1: Factor used to calculate the regularization constant for the luminance
      term, as `C1 = (k1 * max_val) ** 2`. Defaults to 0.01.
    k2: Factor used to calculate the regularization constant for the contrast
      term, as `C2 = (k2 * max_val) ** 2`. Defaults to 0.03.
    rank: An `int`. The number of spatial dimensions. Must be 2 or 3. Defaults
      to `tf.rank(img1) - 2`. In other words, if rank is not explicitly set,
      `img1` and `img2` should have shape `[batch, height, width, channels]`
      if processing 2D images or `[batch, depth, height, width, channels]` if
      processing 3D images.
    name: Namespace to embed the computation in.

  Returns:
    A `Tensor` of type `float32` and shape `batch_shape` containing an MS-SSIM
    value for each image in the batch.

  References:
    .. [1] Z. Wang, E. P. Simoncelli and A. C. Bovik, "Multiscale structural
      similarity for image quality assessment," The Thrity-Seventh Asilomar
      Conference on Signals, Systems & Computers, 2003, 2003, pp. 1398-1402
      Vol.2, doi: 10.1109/ACSSC.2003.1292216.
  """
  with tf.name_scope(name):
    # Convert to tensor if needed.
    img1 = tf.convert_to_tensor(img1, name='img1')
    img2 = tf.convert_to_tensor(img2, name='img2')
    # Default `max_val` to maximum dynamic range for the input dtype.
    if max_val is None:
      max_val = _image_dynamic_range(img1.dtype)
    # Need to convert the images to float32. Scale max_val accordingly so
    # that SSIM is computed correctly.
    max_val = tf.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, tf.dtypes.float32)
    img1 = tf.image.convert_image_dtype(img1, tf.dtypes.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.dtypes.float32)

    # Infer the number of spatial dimensions if not specified by the user, then
    # check that the value is valid.
    if rank is None:
      rank = img1.shape.rank - 2

    # Shape checking.
    shape1, shape2, checks = _verify_compatible_image_shapes(img1, img2, rank)

    # Check that spatial dimensions are big enough.
    min_dim_size = (filter_size - 1) * 2 ** (len(power_factors) - 1) + 1
    checks.append(tf.debugging.assert_greater_equal(
        shape1[-rank-1:-1], min_dim_size, # pylint: disable=invalid-unary-operand-type
        message=(
          f"All spatial dimensions must have size of at least {min_dim_size}, " # pylint: disable=invalid-unary-operand-type
          f"but got shape: {shape1[-rank-1:-1]}. Try upsampling the image, " # pylint: disable=invalid-unary-operand-type
          f"using a smaller `filter_size` or a smaller number of "
          f"`power_factors`.")))

    with tf.control_dependencies(checks):
      img1 = tf.identity(img1)

    imgs = [img1, img2]
    shapes = [shape1, shape2]

    # img1 and img2 are assumed to be a (multi-dimensional) batch of
    # N-dimensional images. `heads` contain the batch dimensions, and
    # `tails` contain the image dimensions.
    heads = [s[:-(rank+1)] for s in shapes]
    tails = [s[-(rank+1):] for s in shapes]

    divisor = [1] + [2] * rank + [1]
    divisor_tensor = tf.constant(divisor[1:], dtype=tf.dtypes.int32)

    def do_pad(images, remainder):
      padding = tf.expand_dims(remainder, -1)
      padding = tf.pad(padding, [[1, 0], [1, 0]]) # pylint: disable=no-value-for-parameter
      return [tf.pad(x, padding, mode='SYMMETRIC') for x in images] # pylint: disable=unexpected-keyword-arg,no-value-for-parameter

    mcs = []
    for k in range(len(power_factors)):
      with tf.name_scope('Scale%d' % k):
        if k > 0:
          # Avg pool takes rank 4 tensors. Flatten leading dimensions.
          flat_imgs = [
            tf.reshape(x, tf.concat([[-1], t], 0))
            for x, t in zip(imgs, tails)
          ]

          # Pad uneven spatial dimensions.
          remainder = tails[0] % divisor_tensor
          need_padding = tf.math.reduce_any(
            tf.math.not_equal(remainder, 0))
          # pylint: disable=cell-var-from-loop
          padded = tf.cond(
            need_padding,
            lambda: do_pad(flat_imgs, remainder),
            lambda: flat_imgs)
          # pylint: enable=cell-var-from-loop

          # Downscale.
          downscaled = [
            tf.nn.avg_pool(
              x, ksize=divisor, strides=divisor, padding='VALID')
            for x in padded
          ]

          # Restore unflattened batch shape.
          tails = [x[1:] for x in tf.shape_n(downscaled)]
          imgs = [
            tf.reshape(x, tf.concat([h, t], 0))
            for x, h, t in zip(downscaled, heads, tails)
          ]

        # Overwrite previous ssim value since we only need the last one.
        ssim_per_channel, cs = _ssim_per_channel(*imgs,
                                                 max_val,
                                                 filter_size,
                                                 filter_sigma,
                                                 k1,
                                                 k2,
                                                 rank)
        mcs.append(tf.nn.relu(cs))

    # Remove the cs score for the last scale. In the MS-SSIM calculation,
    # we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
    mcs.pop()  # Remove the cs score for the last scale.
    mcs_and_ssim = tf.stack(
      mcs + [tf.nn.relu(ssim_per_channel)], axis=-1)
    # Take weighted geometric mean across the scale axis.
    ms_ssim = tf.math.reduce_prod(
      tf.math.pow(mcs_and_ssim, power_factors), [-1])

    return tf.math.reduce_mean(ms_ssim, [-1])  # Avg over color channels.


def ssim2d_multiscale(img1,
                      img2,
                      max_val=None,
                      power_factors=_MSSSIM_WEIGHTS,
                      filter_size=11,
                      filter_sigma=1.5,
                      k1=0.01,
                      k2=0.03,
                      name='ssim2d_multiscale'):
  """Computes the multiscale SSIM (MS-SSIM) between two 2D images.

  This function operates on batches of multi-channel inputs and returns an
  MS-SSIM value for each image in the batch.

  Args:
    img1: A `Tensor`. First batch of images. Must have rank >= 3 with shape
      `batch_shape + [height, width, channels]`. `height` and `width` must be
      greater than or equal to
      `(filter_size - 1) * 2 ** (len(power_factors) - 1) + 1`. Can have integer
      or floating point type, with values in the range `[0, max_val]`.
    img2: A `Tensor`. Second batch of images. Must have rank >= 3 with shape
      `batch_shape + [height, width, channels]`. `height` and `width` must be
      greater than or equal to
      `(filter_size - 1) * 2 ** (len(power_factors) - 1) + 1`. Can have integer
      or floating point type, with values in the range `[0, max_val]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    power_factors: A list of weights for each of the scales. The length of the
      list determines the number of scales. Index 0 is the unscaled resolution's
      weight and each increasing scale corresponds to the image being
      downsampled by 2. Defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
      which are the values obtained in the original paper [1]_.
    filter_size: The size of the Gaussian filter. Defaults to 11.
    filter_sigma: The standard deviation of the Gaussian filter. Defaults to
      1.5.
    k1: Factor used to calculate the regularization constant for the luminance
      term, as `C1 = (k1 * max_val) ** 2`. Defaults to 0.01.
    k2: Factor used to calculate the regularization constant for the contrast
      term, as `C2 = (k2 * max_val) ** 2`. Defaults to 0.03.
    name: Namespace to embed the computation in.

  Returns:
    A `Tensor` of type `float32` and shape `batch_shape` containing an MS-SSIM
    value for each image in the batch.

  References:
    .. [1] Z. Wang, E. P. Simoncelli and A. C. Bovik, "Multiscale structural
      similarity for image quality assessment," The Thrity-Seventh Asilomar
      Conference on Signals, Systems & Computers, 2003, 2003, pp. 1398-1402
      Vol.2, doi: 10.1109/ACSSC.2003.1292216.
  """
  return ssim_multiscale(img1,
                         img2,
                         max_val=max_val,
                         power_factors=power_factors,
                         filter_size=filter_size,
                         filter_sigma=filter_sigma,
                         k1=k1,
                         k2=k2,
                         rank=2,
                         name=name)


def ssim3d_multiscale(img1,
                      img2,
                      max_val=None,
                      power_factors=_MSSSIM_WEIGHTS,
                      filter_size=11,
                      filter_sigma=1.5,
                      k1=0.01,
                      k2=0.03,
                      name='ssim3d_multiscale'):
  """Computes the multiscale SSIM (MS-SSIM) between two 3D images.

  This function operates on batches of multi-channel inputs and returns an
  MS-SSIM value for each image in the batch.

  .. note::
    As of TensorFlow 2.6.0, 3D inputs with `channels` > 1 can only be processed
    on GPU.

  Args:
    img1: A `Tensor`. First batch of images. Must have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `height`, `width` and
      `depth` must be greater than or equal to
      `(filter_size - 1) * 2 ** (len(power_factors) - 1) + 1`. Can have integer
      or floating point type, with values in the range `[0, max_val]`.
    img2: A `Tensor`. Second batch of images. Must have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `height`, `width` and
      `depth` must be greater than or equal to
      `(filter_size - 1) * 2 ** (len(power_factors) - 1) + 1`. Can have integer
      or floating point type, with values in the range `[0, max_val]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    power_factors: A list of weights for each of the scales. The length of the
      list determines the number of scales. Index 0 is the unscaled resolution's
      weight and each increasing scale corresponds to the image being
      downsampled by 2. Defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
      which are the values obtained in the original paper [1]_.
    filter_size: The size of the Gaussian filter. Defaults to 11.
    filter_sigma: The standard deviation of the Gaussian filter. Defaults to
      1.5.
    k1: Factor used to calculate the regularization constant for the luminance
      term, as `C1 = (k1 * max_val) ** 2`. Defaults to 0.01.
    k2: Factor used to calculate the regularization constant for the contrast
      term, as `C2 = (k2 * max_val) ** 2`. Defaults to 0.03.
    name: Namespace to embed the computation in.

  Returns:
    A `Tensor` of type `float32` and shape `batch_shape` containing an MS-SSIM
    value for each image in the batch.

  References:
    .. [1] Z. Wang, E. P. Simoncelli and A. C. Bovik, "Multiscale structural
      similarity for image quality assessment," The Thrity-Seventh Asilomar
      Conference on Signals, Systems & Computers, 2003, 2003, pp. 1398-1402
      Vol.2, doi: 10.1109/ACSSC.2003.1292216.
  """
  return ssim_multiscale(img1,
                         img2,
                         max_val=max_val,
                         power_factors=power_factors,
                         filter_size=filter_size,
                         filter_sigma=filter_sigma,
                         k1=k1,
                         k2=k2,
                         rank=3,
                         name=name)


def _ssim_per_channel(img1,
                      img2,
                      max_val,
                      filter_size,
                      filter_sigma,
                      k1,
                      k2,
                      rank):
  """Computes SSIM index between img1 and img2 per color channel.

  For the parameters, see `ssim`.

  Returns:
    A pair of tensors containing and channel-wise SSIM and
    contrast-structure values. The shape is [..., channels].

  Raises:
    ValueError: If `rank` is not 2 or 3.
  """
  filter_size = tf.constant(filter_size, dtype=tf.int32)
  filter_sigma = tf.constant(filter_sigma, dtype=img1.dtype)

  shape1, shape2 = tf.shape_n([img1, img2])
  checks = [
    tf.debugging.Assert(
      tf.math.reduce_all(
        tf.math.greater_equal(shape1[-(rank + 1):-1], filter_size)),
      [shape1, filter_size],
      summarize=8),
    tf.debugging.Assert(
      tf.math.reduce_all(
        tf.math.greater_equal(shape2[-(rank + 1):-1], filter_size)),
      [shape2, filter_size],
      summarize=8)
  ]

  # Enforce the check to run before computation.
  with tf.control_dependencies(checks):
    img1 = tf.identity(img1)

  kernel = _fspecial_gauss(filter_size, filter_sigma, rank)
  if rank == 2:
    kernel = tf.tile(kernel, multiples=[1, 1, shape1[-1], 1])
  elif rank == 3:
    # If tf.nn.depthwise_conv3d is ever implemented, the following line
    # should be removed and replaced by the commented line below (also make
    # the change below).
    kernel = tf.tile(kernel, multiples=[1, 1, 1, 1, shape1[-1]])
    # kernel = tf.tile(kernel, multiples=[1, 1, 1, shape1[-1], 1])
  else:
    raise ValueError("Invalid rank: {}".format(rank))

  # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
  # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
  compensation = 1.0

  def reducer(x):
    shape = tf.shape(x)
    x = tf.reshape(x, shape=tf.concat([[-1], shape[-(rank + 1):]], 0))
    if rank == 2:
      y = tf.nn.depthwise_conv2d(
        x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    elif rank == 3:
      # If tf.nn.depthwise_conv3d is ever implemented, the following line
      # should be removed and replaced by the commented line below (also
      # make the change above).
      y = tf.nn.convolution(
        x, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
      # y = nn.depthwise_conv3d(
      #   x, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
    return tf.reshape(
      y, tf.concat([shape[:-(rank + 1)], tf.shape(y)[1:]], 0))

  luminance, cs = _ssim_helper(
    img1, img2, reducer, max_val, compensation, k1, k2)

  # Average over the spatial dimensions.
  axes = tf.range(-(rank + 1), -1)
  ssim_val = tf.math.reduce_mean(luminance * cs, axes)
  cs = tf.math.reduce_mean(cs, axes)
  return ssim_val, cs


def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):
  r"""Helper function for computing SSIM.

  SSIM estimates covariances with weighted sums. The default parameters use a
  biased estimate of the covariance:

  Suppose `reducer` is a weighted sum, then the mean estimators are

    \mu_x = \sum_i w_i x_i,
    \mu_y = \sum_i w_i y_i,

  where w_i's are the weighted-sum weights, and covariance estimator is

    cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)

  with assumption \sum_i w_i = 1. This covariance estimator is biased, since

    E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).

  For SSIM measure with unbiased covariance estimators, pass as `compensation`
  argument (1 - \sum_i w_i ^ 2).

  Arguments:
    x: First set of images.
    y: Second set of images.
    reducer: Function that computes 'local' averages from set of images. For
      non-convolutional version, this is usually
      tf.reduce_mean(x, [1, 2]), and for convolutional version, this is
      usually tf.nn.avg_pool2d or tf.nn.conv2d with weighted-sum kernel.
    max_val: The dynamic range (i.e., the difference between the maximum
      possible allowed value and the minimum allowed value).
    compensation: Compensation factor. See above.
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitive to K2 for lower values,
      so it would be better if we taken the values in range of 0<K2<0.4).

  Returns:
    A pair containing the luminance measure, and the contrast-structur
     measure.
  """
  c1 = (k1 * max_val)**2
  c2 = (k2 * max_val)**2

  # SSIM luminance measure is
  # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
  mean0 = reducer(x)
  mean1 = reducer(y)
  num0 = mean0 * mean1 * 2.0
  den0 = tf.math.square(mean0) + tf.math.square(mean1)
  luminance = (num0 + c1) / (den0 + c1)

  # SSIM contrast-structure measure is
  #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
  # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
  #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
  #      = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
  num1 = reducer(x * y) * 2.0
  den1 = reducer(tf.math.square(x) + tf.math.square(y))
  c2 *= compensation
  cs = (num1 - num0 + c2) / (den1 - den0 + c2)

  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs


def _fspecial_gauss(size, sigma, rank): # pylint: disable=missing-param-doc
  """N-D gaussian filter."""
  if rank == 2:
    return _fspecial_gauss_2d(size, sigma)
  if rank == 3:
    return _fspecial_gauss_3d(size, sigma)
  raise ValueError("Invalid rank: {}".format(rank))


def _fspecial_gauss_2d(size, sigma): # pylint: disable=missing-param-doc
  """2D gaussian filter."""
  size = tf.convert_to_tensor(size, tf.int32)
  sigma = tf.convert_to_tensor(sigma)

  coords = tf.cast(tf.range(size), sigma.dtype)
  coords -= tf.cast(size - 1, sigma.dtype) / 2.0

  g = tf.math.square(coords)
  g *= -0.5 / tf.math.square(sigma)

  g = tf.reshape(g, shape=[1, -1]) + tf.reshape(g, shape=[-1, 1])
  g = tf.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
  g = tf.nn.softmax(g)
  return tf.reshape(g, shape=[size, size, 1, 1])


def _fspecial_gauss_3d(size, sigma): # pylint: disable=missing-param-doc
  """3D gaussian filter."""
  size = tf.convert_to_tensor(size, tf.int32)
  sigma = tf.convert_to_tensor(sigma)

  coords = tf.cast(tf.range(size), sigma.dtype)
  coords -= tf.cast(size - 1, sigma.dtype) / 2.0

  g = tf.math.square(coords)
  g *= -0.5 / tf.math.square(sigma)

  g = (tf.reshape(g, shape=[1, 1, -1]) +
     tf.reshape(g, shape=[1, -1, 1]) +
     tf.reshape(g, shape=[-1, 1, 1]))
  g = tf.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
  g = tf.nn.softmax(g)
  return tf.reshape(g, shape=[size, size, size, 1, 1])


def _verify_compatible_image_shapes(img1, img2, rank):
  """Checks if two image tensors are compatible for the given rank.

  Checks if two sets of images have rank at least `rank + 1` (spatial
  dimensions plus channel dimension) and if the last `rank + 1` dimensions
  match.

  Args:
    img1: Tensor containing the first image batch.
    img2: Tensor containing the second image batch.
    rank: Rank of the images (number of spatial dimensions).

  Returns:
    A tuple containing: the first tensor shape, the second tensor shape, and
    a list of tf.debugging.Assert() ops implementing the checks.
  """
  rank_p1 = rank + 1

  # Now assign shape tensors.
  shape1, shape2 = tf.shape_n([img1, img2])

  checks = []
  checks.append(
      tf.debugging.assert_greater_equal(
          rank, 2,
          message=f"`rank` must be >= 2, but got: {rank}"))
  checks.append(
      tf.debugging.assert_less_equal(
          rank, 3,
          message=f"`rank` must be <= 3, but got: {rank}"))
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


def _image_dynamic_range(dtype):
  """Returns the dynamic range for images of the specified data type.

  Arguments:
    dtype: A `tf.DType` object or a value that can be converted to a `tf.DType`
      object.

  Return:
    A scalar `Tensor` dynamic range, with dtype `dtype`.

  Raises:
    ValueError: If `dtype` is not integer or floating point.
  """
  dtype = tf.dtypes.as_dtype(dtype)
  if dtype.is_integer:
    max_val = dtype.max
  elif dtype.is_floating:
    max_val = 1
  else:
    raise ValueError(f"Invalid image dtype: {dtype.name}")
  return tf.cast(max_val, dtype)


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
  with tf.control_dependencies([tf.debugging.assert_equal(tf.rank(tensor),
                                                          tf.size(shape))]):
    tensor = tf.identity(tensor)

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

  # Set static shape, if possible.
  if static_shape is not None:
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


def total_variation(tensor,
                    axis=None,
                    keepdims=False,
                    name='total_variation'):
  """Calculate and return the total variation along the specified axes.

  The total variation is the sum of the absolute differences for neighboring
  values in the input tensor. This is a measure of the noise present.

  This implements the anisotropic N-D version of the formula described here:

  https://en.wikipedia.org/wiki/Total_variation_denoising

  Args:
    tensor: A `Tensor`. Can have any shape.
    axis: An `int` or a list of `ints`. The axes along which the differences are
      taken. Defaults to `list(range(1, tensor.shape.rank - 1))`, i.e.
      differences are taken along all axes except the first and the last, which
      are assumed to be the batch and channel axes, respectively.
    keepdims: A `bool`. If `True`, keeps the singleton reduced dimensions.
      Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    The total variation of `images`.

  Raises:
    ValueError: If `tensor` has unknown static rank or `axis` is out of bounds.
  """
  with tf.name_scope(name):

    tensor = tf.convert_to_tensor(tensor)
    rank = tensor.shape.rank

    if rank is None:
      raise ValueError(
          "`total_variation` requires known rank for input `tensor`")

    # If `axis` was not specified, compute along all axes.
    if axis is None:
      axis = list(range(rank))
    if isinstance(axis, int):
      axis = [axis]
    for ax in axis:
      if ax < -rank or ax >= rank:
        raise ValueError(
            f"axis {ax} is out of bounds for tensor of rank {rank}")

    # Total variation.
    tot_var = tf.constant(0, dtype=tensor.dtype.real_dtype)

    for ax in axis:
      slice1 = [slice(None)] * rank
      slice2 = [slice(None)] * rank
      slice1[ax] = slice(1, None)
      slice2[ax] = slice(None, -1)

      # Calculate the difference of neighboring values.
      pixel_diff = tensor[slice1] - tensor[slice2]

      # Add the difference for the current axis to total. Use `keepdims=True`
      # to enable broadcasting.
      tot_var += tf.math.reduce_sum(tf.math.abs(pixel_diff),
                                    axis=axis,
                                    keepdims=True)

    # Squeeze the TV dimensions.
    if not keepdims:
      tot_var = tf.squeeze(tot_var, axis=axis)

    return tot_var


def extract_glimpses(images, sizes, offsets):
  """Extract glimpses (patches) from a tensor at the given offsets.

  Args:
    images: A `Tensor`. Must have shape `[batch_size, ..., channels]`, where
      `...` are the `N` spatial dimensions.
    sizes: A list of `ints` of length `N`.
    offsets: A `Tensor` of shape `[M, N]` containing indices into the upper-left
      corners of the patches to be extracted.

  Returns:
    A `Tensor` with shape `[batch_size, M, prod(sizes) * channels]`,
    where the last dimension are the flattened glimpses.
  """
  images = tf.convert_to_tensor(images)
  offsets = tf.convert_to_tensor(offsets)

  # Infer rank from kernel size, then check that `images` and `offsets` are
  # consistent.
  rank = len(sizes)
  checks = []
  checks.append(tf.debugging.assert_rank(images, rank + 2, message=(
      f"`images` must have rank `len(sizes) + 2`, but got: {tf.rank(images)}")))
  checks.append(tf.debugging.assert_equal(tf.shape(offsets)[-1], rank, message=(
      f"The last dimension of `offsets` must be equal to `len(sizes)`, "
      f"but got: {tf.shape(offsets)[-1]}")))
  with tf.control_dependencies(checks):
    images = tf.identity(images)

  # Get batch size and the number of patches.
  batch_size = tf.shape(images)[0]
  num_patches = tf.shape(offsets)[-2]

  # Generate an array of indices into a tensor of shape `[batch_size] + sizes`.
  indices = tf.transpose(tf.reshape(tf.stack(
      tf.meshgrid(*[tf.range(ks) for ks in [batch_size] + sizes],
                  indexing='ij')), [rank + 1, -1]))
  indices = tf.reshape(indices, [batch_size, -1, rank + 1])

  # Replicate (via broadcasting) and offset the indices array.
  offsets = tf.pad(offsets, [[0, 0], [1, 0]]) # pylint:disable=no-value-for-parameter
  offsets = tf.expand_dims(offsets, -2)
  indices = tf.expand_dims(indices, -3)
  indices = indices + tf.cast(offsets, indices.dtype)

  # Gather all patches.
  patches = tf.gather_nd(images, indices)
  patches = tf.reshape(patches, [batch_size, num_patches, -1])
  return patches


def phantom(phantom_type='modified_shepp_logan', # pylint: disable=dangerous-default-value
            shape=[256, 256],
            num_coils=None,
            dtype=tf.dtypes.float32,
            return_sensitivities=False):
  """Generate a phantom image.

  Available 2D phantoms are:

    * **shepp_logan**: The original Shepp-Logan phantom [1]_.
    * **modified_shepp_logan**: A variant of the Shepp-Logan phantom in which
      the contrast is improved for better visual perception [2]_.

  Available 3D phantoms are:

    * **kak_roberts**: A simplified 3D extension of the Shepp-Logan phantom
      [3]_.
    * **modified_kak_roberts**: A variant of the Kak-Roberts phantom in which
      the contrast is improved for better visual perception [4]_.
    * **koay_sarlls_ozarslan**: Same as `modified_kak_roberts`.

  Args:
    phantom_type: A `string`. If 2D, must be one of
      `{'shepp_logan', 'modified_shepp_logan'}`. If 3D, must be one of
      `{'kak_roberts', 'modified_kak_roberts', 'koay_sarlls_ozarslan'}`.
      Defaults to `modified_shepp_logan` for 2D phantoms and
      `modified_kak_roberts` for 3D phantoms.
    shape: A list of `ints`. The shape of the generated phantom. Must have
      length 2 or 3.
    num_coils: An `int`. The number of coils for parallel imaging phantoms. If
      `None`, no coil array will be simulated. Defaults to `None`.
    dtype: A `string` or `tf.DType`. The data type of the generated phantom.
    return_sensitivities: A `bool`. If `True`, returns a tuple containing the
      phantom and the coil sensitivities. If `False`, returns the phantom.
      Defaults to `False`.

  Returns:
    A `Tensor` of type `dtype` containing the generated phantom. Has shape
    `shape` if `num_coils` is `None`, or shape `[num_coils, *shape]` if
    `num_coils` is not `None`.

    If `return_sensitivities` is `True`, returns a tuple of two tensors with
    equal shape and type, the first of which is the phantom and the second the
    coil sensitivities.

  Raises:
    ValueError: If the requested ND phantom is not defined.

  References:
    .. [1] Shepp, L. A., & Logan, B. F. (1974). The Fourier reconstruction of a
      head section. IEEE Transactions on nuclear science, 21(3), 21-43.
    .. [2] Toft, P. (1996). The radon transform. Theory and Implementation
      (Ph. D. Dissertation)(Copenhagen: Technical University of Denmark).
    .. [3] Kak, A. C., & Slaney, M. (2001). Principles of computerized
      tomographic imaging. Society for Industrial and Applied Mathematics.
    .. [4] Koay, C. G., Sarlls, J. E., & Özarslan, E. (2007). Three‐dimensional
      analytical magnetic resonance imaging phantom in the Fourier domain.
      Magnetic Resonance in Medicine, 58(2), 430-436.
  """
  phantom_type = check_util.validate_enum(
      phantom_type,
      {'shepp_logan', 'modified_shepp_logan',
       'kak_roberts', 'modified_kak_roberts', 'koay_sarlls_ozarslan'},
      name='phantom_type')
  if isinstance(shape, tf.TensorShape):
    shape = shape.as_list()
  shape = check_util.validate_list(
      shape, element_type=int, name='shape')

  # If a Shepp-Logan 3D phantom was requested, by default we use the Kak-Roberts
  # 3D extension of the Shepp-Logan phantom.
  if len(shape) == 3 and phantom_type == 'shepp_logan':
    phantom_type = 'kak_roberts'
  elif len(shape) == 3 and phantom_type == 'modified_shepp_logan':
    phantom_type = 'modified_kak_roberts'

  if len(shape) == 2:
    phantom_objects = _PHANTOMS_2D
  elif len(shape) == 3:
    phantom_objects = _PHANTOMS_3D
  else:
    raise ValueError(
        f"`shape` must have rank 2 or 3, but got: {shape}")
  if phantom_type in phantom_objects:
    phantom_objects = phantom_objects[phantom_type]
  else:
    raise ValueError(
        f"No definition for {len(shape)}D phantom of type: {phantom_type}")

  # Initialize image and coordinates arrays.
  image = tf.zeros(shape, dtype=dtype)
  x = array_ops.meshgrid(*[tf.linspace(-1.0, 1.0, s) for s in shape])

  for obj in phantom_objects:

    if isinstance(obj, Ellipse):
      # Apply translation and rotation to coordinates.
      tx = geom_ops.rotate_2d(x - obj.pos, tf.cast(obj.phi, x.dtype))
      # Use object equation to generate a mask.
      mask = tf.math.reduce_sum(
          (tx ** 2) / (tf.convert_to_tensor(obj.size) ** 2), -1) <= 1.0
      # Add current object to image.
      image = tf.where(mask, image + obj.rho, image)
    elif isinstance(obj, Ellipsoid):
      # Apply translation and rotation to coordinates.
      tx = geom_ops.rotate_3d(x - obj.pos, tf.cast(obj.phi, x.dtype))
      # Use object equation to generate a mask.
      mask = tf.math.reduce_sum(
          (tx ** 2) / (tf.convert_to_tensor(obj.size) ** 2), -1) <= 1.0
      # Add current object to image.
      image = tf.where(mask, image + obj.rho, image)

  if num_coils is not None:
    sens = _birdcage_sensitivities(shape, num_coils, dtype=dtype)
    image *= sens

    if return_sensitivities:
      return image, sens

  return image


def _birdcage_sensitivities(shape,
                            num_coils,
                            birdcage_radius=1.5,
                            num_rings=4,
                            dtype=tf.dtypes.complex64):
  """Simulates birdcage coil sensitivities.

  Args:
    shape: A list of `ints` of length 2 or 3. The shape of the coil
      sensitivities.
    num_coils: An `int`. The number of coils.
    birdcage_radius: A `float`. The relative radius of the birdcage. Defaults
      to 1.5.
    num_rings: An `int`. The number of rings along the depth dimension. Defaults
      to 4. Only relevant if `shape` has rank 3.
    dtype: A `string` or a `DType`. The data type of the output tensor.

  Returns:
    An array of shape `[num_coils, *shape]`.

  Raises:
    ValueError: If `shape` does not have a valid rank.
  """
  if isinstance(shape, tf.TensorShape):
    shape = shape.as_list()
  dtype = tf.dtypes.as_dtype(dtype)

  if len(shape) == 2:

    height, width = shape
    c, y, x = tf.meshgrid(
        *[tf.range(n, dtype=dtype.real_dtype) for n in (num_coils,
                                                        height,
                                                        width)],
        indexing='ij')

    coil_x = birdcage_radius * tf.math.cos(c * (2.0 * np.pi / num_coils))
    coil_y = birdcage_radius * tf.math.sin(c * (2.0 * np.pi / num_coils))
    coil_phs = -c * (2.0 * np.pi / num_coils)

    x_co = (x - width / 2.0) / (width / 2.0) - coil_x
    y_co = (y - height / 2.0) / (height / 2.0) - coil_y

    rr = tf.math.sqrt(x_co ** 2 + y_co ** 2)

    if dtype.is_complex:
      phi = tf.math.atan2(x_co, -y_co) + coil_phs
      out = tf.cast(1.0 / rr, dtype) * tf.math.exp(
          tf.dtypes.complex(tf.constant(0.0, dtype=dtype.real_dtype), phi))
    else:
      out = 1.0 / rr

  elif len(shape) == 3:

    num_coils_per_ring = (num_coils + num_rings - 1) // num_rings

    depth, height, width = shape
    c, z, y, x = tf.meshgrid(
        *[tf.range(n, dtype=dtype.real_dtype) for n in (num_coils,
                                                        depth,
                                                        height,
                                                        width)],
        indexing='ij')

    coil_x = birdcage_radius * tf.math.cos(c * (2 * np.pi / num_coils_per_ring))
    coil_y = birdcage_radius * tf.math.sin(c * (2 * np.pi / num_coils_per_ring))
    coil_z = tf.math.floor(c / num_coils_per_ring) - 0.5 * (
        tf.math.ceil(num_coils / num_coils_per_ring) - 1)
    coil_phs = -(c + tf.math.floor(c / num_coils_per_ring)) * (
        2 * np.pi / num_coils_per_ring)

    x_co = (x - width / 2.0) / (width / 2.0) - coil_x
    y_co = (y - height / 2.0) / (height / 2.0) - coil_y
    z_co = (z - depth / 2.0) / (depth / 2.0) - coil_z

    rr = tf.math.sqrt(x_co ** 2 + y_co ** 2 + z_co ** 2)

    if dtype.is_complex:
      phi = tf.math.atan2(x_co, -y_co) + coil_phs
      out = tf.cast(1.0 / rr, dtype) * tf.math.exp(
          tf.dtypes.complex(tf.constant(0.0, dtype=dtype.real_dtype), phi))
    else:
      out = 1.0 / rr

  else:
    raise ValueError(f"`shape` must be of rank 2 or 3, but got: {shape}")

  # Normalize by root sum of squares.
  rss = tf.math.sqrt(tf.math.reduce_sum(out * tf.math.conj(out), axis=0))
  out /= rss

  return tf.cast(out, dtype)


Ellipse = collections.namedtuple(
    'Ellipse', ['rho', 'size', 'pos', 'phi'])
Ellipsoid = collections.namedtuple(
    'Ellipsoid', ['rho', 'size', 'pos', 'phi'])


_PHANTOMS_2D = {
    'shepp_logan': [
        Ellipse(rho=1, size=(.9200, .6900), pos=(0, 0), phi=(0,)),
        Ellipse(rho=-.98, size=(.8740, .6624), pos=(.0184, 0), phi=(0,)),
        Ellipse(rho=-.02, size=(.3100, .1100),
                pos=(0, .22), phi=(.1 * np.pi,)),
        Ellipse(rho=-.02, size=(.4100, .1600),
                pos=(0, -.22), phi=(-.1 * np.pi,)),
        Ellipse(rho=.01, size=(.2500, .2100), pos=(-.35, 0), phi=(0,)),
        Ellipse(rho=.01, size=(.0460, .0460), pos=(-.1, 0), phi=(0,)),
        Ellipse(rho=.01, size=(.0460, .0460), pos=(.1, 0), phi=(0,)),
        Ellipse(rho=.01, size=(.0230, .0460), pos=(.605, -.08), phi=(0,)),
        Ellipse(rho=.01, size=(.0230, .0230), pos=(.606, 0), phi=(0,)),
        Ellipse(rho=.01, size=(.0460, .0230), pos=(.605, .06), phi=(0,))],
    'modified_shepp_logan': [
        Ellipse(rho=1, size=(.9200, .6900), pos=(0, 0), phi=(0,)),
        Ellipse(rho=-.8, size=(.8740, .6624), pos=(.0184, 0), phi=(0,)),
        Ellipse(rho=-.2, size=(.3100, .1100),
                pos=(0, .22), phi=(.1 * np.pi,)),
        Ellipse(rho=-.2, size=(.4100, .1600),
                pos=(0, -.22), phi=(-.1 * np.pi,)),
        Ellipse(rho=.1, size=(.2500, .2100), pos=(-.35, 0), phi=(0,)),
        Ellipse(rho=.1, size=(.0460, .0460), pos=(-.1, 0), phi=(0,)),
        Ellipse(rho=.1, size=(.0460, .0460), pos=(.1, 0), phi=(0,)),
        Ellipse(rho=.1, size=(.0230, .0460), pos=(.605, -.08), phi=(0,)),
        Ellipse(rho=.1, size=(.0230, .0230), pos=(.606, 0), phi=(0,)),
        Ellipse(rho=.1, size=(.0460, .0230), pos=(.605, .06), phi=(0,))]
}


_PHANTOMS_3D = {
    'kak_roberts': [
        Ellipsoid(rho=2, size=(.9000, .9200, .6900),
                  pos=(0, 0, 0), phi=(0, 0, 0)),
        Ellipsoid(rho=-.98, size=(.8800, .8740, .6624),
                  pos=(0, 0, 0), phi=(0, 0, 0)),
        Ellipsoid(rho=-.02, size=(.2100, .1600, .4100),
                  pos=(-.250, 0, -.220), phi=(-.6 * np.pi, 0, 0)),
        Ellipsoid(rho=-.02, size=(.2200, .1100, .3100),
                  pos=(-.250, 0, .220), phi=(-.4 * np.pi, 0, 0)),
        Ellipsoid(rho=.02, size=(.5000, .2500, .2100),
                  pos=(-.250, -.350, 0), phi=(0, 0, 0)),
        Ellipsoid(rho=.02, size=(.0460, .0460, .0460),
                  pos=(-.250, -.100, 0), phi=(0, 0, 0)),
        Ellipsoid(rho=.01, size=(.0200, .0230, .0460),
                  pos=(-.250, .650, -.080), phi=(0, 0, 0)),
        Ellipsoid(rho=.01, size=(.0200, .0230, .0460),
                  pos=(-.250, .650, .060), phi=(-.5 * np.pi, 0, 0)),
        Ellipsoid(rho=.02, size=(.1000, .0400, .0560),
                  pos=(.625, .105, .060), phi=(-.5 * np.pi, 0, 0)),
        Ellipsoid(rho=-.02, size=(.1000, .0560, .0560),
                  pos=(.625, -.100, 0), phi=(0, 0, 0))],
    'modified_kak_roberts': [
        Ellipsoid(rho=2, size=(.9000, .9200, .6900),
                  pos=(0, 0, 0), phi=(0, 0, 0)),
        Ellipsoid(rho=-.8, size=(.8800, .8740, .6624),
                  pos=(0, 0, 0), phi=(0, 0, 0)),
        Ellipsoid(rho=-.2, size=(.2100, .1600, .4100),
                  pos=(-.250, 0, -.220), phi=(-.6 * np.pi, 0, 0)),
        Ellipsoid(rho=-.2, size=(.2200, .1100, .3100),
                  pos=(-.250, 0, .220), phi=(-.4 * np.pi, 0, 0)),
        Ellipsoid(rho=.2, size=(.5000, .2500, .2100),
                  pos=(-.250, -.350, 0), phi=(0, 0, 0)),
        Ellipsoid(rho=.2, size=(.0460, .0460, .0460),
                  pos=(-.250, -.100, 0), phi=(0, 0, 0)),
        Ellipsoid(rho=.1, size=(.0200, .0230, .0460),
                  pos=(-.250, .650, -.080), phi=(0, 0, 0)),
        Ellipsoid(rho=.1, size=(.0200, .0230, .0460),
                  pos=(-.250, .650, .060), phi=(-.5 * np.pi, 0, 0)),
        Ellipsoid(rho=.2, size=(.1000, .0400, .0560),
                  pos=(.625, .105, .060), phi=(-.5 * np.pi, 0, 0)),
        Ellipsoid(rho=-.2, size=(.1000, .0560, .0560),
                  pos=(.625, -.100, 0), phi=(0, 0, 0))]
}
