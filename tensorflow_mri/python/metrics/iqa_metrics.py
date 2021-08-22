# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain img1 copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Image quality assessment metrics.

This module contains metrics and operations for image quality assessment (IQA).
"""

import tensorflow as tf

from tensorflow_mri.python.ops import image_ops


@tf.keras.utils.register_keras_serializable(package="MRI")
class PeakSignalToNoiseRatio(tf.keras.metrics.MeanMetricWrapper):
  """Peak signal-to-noise ratio (PSNR) metric.

  The PSNR is the ratio between the maximum possible power of an image and the
  power of corrupting noise, estimated by comparing to a reference image.

  This metric supports 2D and 3D image inputs, `y_true` and `y_pred`. For 2D
  images, inputs must have rank >= 3 with shape
  `batch_shape + [height, width, channels]`. For 3D images, inputs must have
  rank >= 4 with shape `batch_shape + [depth, height, width, channels]`.

  Args:
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    rank: An `int`. The number of spatial dimensions. Must be 2 or 3. Defaults
      to `tf.rank(y_true) - 2`. In other words, if rank is not explicitly set,
      `y_true` and `y_pred` should have shape `[batch, height, width, channels]`
      if processing 2D images or `[batch, depth, height, width, channels]` if
      processing 3D images.
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """
  def __init__(self,
               max_val=None,
               rank=None,
               name='psnr',
               dtype=None):
    super().__init__(image_ops.psnr,
                     name=name,
                     dtype=dtype,
                     max_val=max_val,
                     rank=rank)


@tf.keras.utils.register_keras_serializable(package="MRI")
class StructuralSimilarity(tf.keras.metrics.MeanMetricWrapper):
  """Structural similarity index (SSIM) metric.

  The SSIM is a method for predicting the perceived quality of an image, based
  on its similarity to a reference image.

  This metric supports 2D and 3D image inputs, `y_true` and `y_pred`. For 2D
  images, inputs must have rank >= 3 with shape
  `batch_shape + [height, width, channels]`. For 3D images, inputs must have
  rank >= 4 with shape `batch_shape + [depth, height, width, channels]`.

  Args:
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
      to `tf.rank(y_true) - 2`. In other words, if rank is not explicitly set,
      `y_true` and `y_pred` should have shape `[batch, height, width, channels]`
      if processing 2D images or `[batch, depth, height, width, channels]` if
      processing 3D images.

  References:
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
      Image quality assessment: from error visibility to structural similarity.
      IEEE transactions on image processing, 13(4), 600-612.
  """
  def __init__(self,
               max_val=None,
               filter_size=11,
               filter_sigma=1.5,
               k1=0.01,
               k2=0.03,
               rank=None,
               name='ssim',
               dtype=None):

    super().__init__(image_ops.ssim,
                     name=name,
                     dtype=dtype,
                     max_val=max_val,
                     filter_size=filter_size,
                     filter_sigma=filter_sigma,
                     k1=k1,
                     k2=k2,
                     rank=rank)


@tf.keras.utils.register_keras_serializable(package="MRI")
class MultiscaleStructuralSimilarity(tf.keras.metrics.MeanMetricWrapper):
  """Multiscale structural similarity index (MS-SSIM) metric.

  Args:
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
    name: String name of the metric instance.
    dtype: Data type of the metric result.

  References:
    .. [1] Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003, November).
      Multiscale structural similarity for image quality assessment. In The
      Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003
      (Vol. 2, pp. 1398-1402). Ieee.
  """
  def __init__(self,
               max_val=None,
               filter_size=11,
               filter_sigma=1.5,
               k1=0.01,
               k2=0.03,
               rank=None,
               name='ms_ssim',
               dtype=None):

    super().__init__(image_ops.ssim_multiscale,
                     name=name,
                     dtype=dtype,
                     max_val=max_val,
                     filter_size=filter_size,
                     filter_sigma=filter_sigma,
                     k1=k1,
                     k2=k2,
                     rank=rank)
