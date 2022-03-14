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
"""IQA losses.

This module contains loss functions for the optimization of image quality.
"""

import tensorflow as tf

from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import keras_util


@api_util.export("losses.StructuralSimilarityLoss")
@tf.keras.utils.register_keras_serializable(package="MRI")
class StructuralSimilarityLoss(keras_util.LossFunctionWrapper):
  """Computes the structural similarity (SSIM) loss.

  The SSIM loss is equal to :math:`1.0 - \textrm{SSIM}`.

  .. warning::
    As of TensorFlow 2.6.0, 3D inputs with `channels` > 1 can only be processed
    on GPU.

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
    reduction: Type of `tf.keras.losses.Reduction` to apply to loss. Default
      value is `AUTO`.
    name: String name of the loss instance.

  References:
    .. [1] Zhao, H., Gallo, O., Frosio, I., & Kautz, J. (2016). Loss functions
      for image restoration with neural networks. IEEE Transactions on
      computational imaging, 3(1), 47-57.
  """
  def __init__(self,
               max_val=None,
               filter_size=11,
               filter_sigma=1.5,
               k1=0.01,
               k2=0.03,
               rank=None,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='ssim_loss'):
    super().__init__(ssim_loss, reduction=reduction, name=name, max_val=max_val,
                     filter_size=filter_size, filter_sigma=filter_sigma,
                     k1=k1, k2=k2, rank=rank)


@api_util.export("losses.MultiscaleStructuralSimilarityLoss")
@tf.keras.utils.register_keras_serializable(package="MRI")
class MultiscaleStructuralSimilarityLoss(keras_util.LossFunctionWrapper):
  """Computes the multiscale structural similarity (MS-SSIM) loss.

  The MS-SSIM loss is equal to :math:`1.0 - \textrm{MS-SSIM}`.

  .. warning::
    As of TensorFlow 2.6.0, 3D inputs with `channels` > 1 can only be processed
    on GPU.

  Args:
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    power_factors: A list of weights for each of the scales. The length of the
      list determines the number of scales. Index 0 is the unscaled resolution's
      weight and each increasing scale corresponds to the image being
      downsampled by 2. Defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
      which are the values obtained in the original paper.
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
    reduction: Type of `tf.keras.losses.Reduction` to apply to loss. Default
      value is `AUTO`.
    name: String name of the loss instance.

  References:
    .. [1] Zhao, H., Gallo, O., Frosio, I., & Kautz, J. (2016). Loss functions
      for image restoration with neural networks. IEEE Transactions on
      computational imaging, 3(1), 47-57.
  """
  def __init__(self,
               max_val=None,
               power_factors=image_ops._MSSSIM_WEIGHTS,
               filter_size=11,
               filter_sigma=1.5,
               k1=0.01,
               k2=0.03,
               rank=None,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='ssim_multiscale_loss'):
    super().__init__(ssim_multiscale_loss, reduction=reduction, name=name,
                     max_val=max_val, power_factors=power_factors,
                     filter_size=filter_size, filter_sigma=filter_sigma,
                     k1=k1, k2=k2, rank=rank)


@api_util.export("losses.ssim_loss")
@tf.keras.utils.register_keras_serializable(package="MRI")
def ssim_loss(y_true, y_pred, max_val=None,
              filter_size=11, filter_sigma=1.5,
              k1=0.01, k2=0.03, rank=None):
  r"""Computes the structural similarity (SSIM) loss.

  The SSIM loss is equal to :math:`1.0 - \textrm{SSIM}`.

  .. warning::
    As of TensorFlow 2.6.0, 3D inputs with `channels` > 1 can only be processed
    on GPU.

  Args:
    y_true: A `Tensor`. Ground truth images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `height`, `width` and
      `depth` must be greater than or equal to `filter_size`. Must have floating
      point type, with values in the range `[0, max_val]`.
    y_pred: A `Tensor`. Predicted images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `height`, `width` and
      `depth` must be greater than or equal to `filter_size`. Must have floating
      point type, with values in the range `[0, max_val]`.
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

  Returns:
    A `Tensor` of type `float32` and shape `batch_shape` containing an SSIM
    value for each image in the batch.

  References:
    .. [1] Zhao, H., Gallo, O., Frosio, I., & Kautz, J. (2016). Loss functions
      for image restoration with neural networks. IEEE Transactions on
      computational imaging, 3(1), 47-57.
  """
  return 1.0 - image_ops.ssim(y_true, y_pred,
                              max_val=max_val,
                              filter_size=filter_size,
                              filter_sigma=filter_sigma,
                              k1=k1,
                              k2=k2,
                              rank=rank)


@api_util.export("losses.ssim_multiscale_loss")
@tf.keras.utils.register_keras_serializable(package="MRI")
def ssim_multiscale_loss(y_true, y_pred, max_val=None,
                         power_factors=image_ops._MSSSIM_WEIGHTS, # pylint: disable=protected-access
                         filter_size=11, filter_sigma=1.5,
                         k1=0.01, k2=0.03, rank=None):
  r"""Computes the multiscale structural similarity (MS-SSIM) loss.

  The MS-SSIM loss is equal to :math:`1.0 - \textrm{MS-SSIM}`.

  .. warning::
    As of TensorFlow 2.6.0, 3D inputs with `channels` > 1 can only be processed
    on GPU.

  Args:
    y_true: A `Tensor`. Ground truth images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `height`, `width` and
      `depth` must be greater than or equal to
      `(filter_size - 1) * 2 ** (len(power_factors) - 1) + 1`. Must have
      floating point type, with values in the range `[0, max_val]`.
    y_pred: A `Tensor`. Predicted images. For 2D images, must have rank >= 3
      with shape `batch_shape + [height, width, channels]`. For 3D images, must
      have rank >= 4 with shape
      `batch_shape + [depth, height, width, channels]`. `height`, `width` and
      `depth` must be greater than or equal to
      `(filter_size - 1) * 2 ** (len(power_factors) - 1) + 1`. Must have
      floating point type, with values in the range `[0, max_val]`.
    max_val: The dynamic range of the images (i.e., the difference between
      the maximum and the minimum allowed values). Defaults to 1 for floating
      point input images and `MAX` for integer input images, where `MAX` is the
      largest positive representable number for the data type.
    power_factors: A list of weights for each of the scales. The length of the
      list determines the number of scales. Index 0 is the unscaled resolution's
      weight and each increasing scale corresponds to the image being
      downsampled by 2. Defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
      which are the values obtained in the original paper.
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

  Returns:
    A `Tensor` of type `float32` and shape `batch_shape` containing an SSIM
    value for each image in the batch.

  References:
    .. [1] Zhao, H., Gallo, O., Frosio, I., & Kautz, J. (2016). Loss functions
      for image restoration with neural networks. IEEE Transactions on
      computational imaging, 3(1), 47-57.
  """
  return 1.0 - image_ops.ssim_multiscale(y_true, y_pred,
                                         max_val=max_val,
                                         power_factors=power_factors,
                                         filter_size=filter_size,
                                         filter_sigma=filter_sigma,
                                         k1=k1,
                                         k2=k2,
                                         rank=rank)
