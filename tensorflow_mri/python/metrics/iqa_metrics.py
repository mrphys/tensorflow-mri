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
"""Image quality assessment metrics.

This module contains metrics and operations for image quality assessment (IQA).
"""

import tensorflow as tf

from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util


class _MeanMetricWrapperIQA(tf.keras.metrics.MeanMetricWrapper):
  """Wraps `tf.keras.metrics.MeanMetricWrapper` to support IQA metrics.

  Adds two new arguments to `MeanMetricWrapper`:

  * **multichannel**: If `True` (default), the input is expected to have a
    channel dimension. If `False`, the input is not expected to have a
    channel dimension. Because the wrapped functions expect a channel
    dimension, this wrapper adds a channel dimension to the inputs if
    `multichannel` is `False`.
  * **complex_part**: If `None` (default), the input is assumed to be real.
    If `'real'`, `'imag'`, `'abs'`, or `'angle'`, the input is assumed to be
    complex and the relevant part is extracted and scaled before passing to the
    wrapped function. `complex_part` must be specified if the input is complex.
  """
  def __init__(self, *args, **kwargs):
    self._max_val = kwargs.get('max_val') or 1.0 # Used during `update_state`.
    self._multichannel = kwargs.pop('multichannel', True)
    self._complex_part = check_util.validate_enum(
        kwargs.pop('complex_part', None),
        [None, 'real', 'imag', 'abs', 'angle'],
        'complex_part')
    super().__init__(*args, **kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates metric statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example.

    Returns:
      Update op.

    Raises:
      ValueError: If `y_true` or `y_pred` are complex and `complex_part` was not
        specified.
    """
    # Add a singleton channel dimension if multichannel is disabled.
    if not self._multichannel:
      y_true = tf.expand_dims(y_true, axis=-1)
      y_pred = tf.expand_dims(y_pred, axis=-1)
    # Extract the relevant complex part, if necessary.
    if self._complex_part is not None:
      y_true = image_ops.extract_and_scale_complex_part(
          y_true, self._complex_part, self._max_val)
      y_pred = image_ops.extract_and_scale_complex_part(
          y_pred, self._complex_part, self._max_val)
    else: # self._complex_part is None
      if y_true.dtype.is_complex or y_pred.dtype.is_complex:
        raise ValueError('complex_part must be specified for complex inputs.')
    return super().update_state(y_true, y_pred, sample_weight)

  def get_config(self):
    """Returns the config of the metric."""
    config = {
        'multichannel': self._multichannel,
        'complex_part': self._complex_part
    }
    base_config = super().get_config()
    return {**base_config, **config}


@api_util.export("metrics.PeakSignalToNoiseRatio")
@tf.keras.utils.register_keras_serializable(package="MRI")
class PeakSignalToNoiseRatio(_MeanMetricWrapperIQA):
  """Peak signal-to-noise ratio (PSNR) metric.

  The PSNR is the ratio between the maximum possible power of an image and the
  power of corrupting noise, estimated by comparing to a reference image.

  This metric supports 2D and 3D image inputs, `y_true` and `y_pred`. For 2D
  images, inputs must have rank >= 3 with shape
  `batch_shape + [height, width, channels]`. For 3D images, inputs must have
  rank >= 4 with shape `batch_shape + [depth, height, width, channels]`. If
  `multichannel` is `False`, the channel dimension should be omitted.

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
    multichannel: A `boolean`. Whether multichannel computation is enabled. If
      `False`, the inputs `y_true` and `y_pred` are not expected to have a
      channel dimension, i.e. they should have shape
      `batch_shape + [height, width]` (2D) or
      `batch_shape + [depth, height, width]` (3D).
    complex_part: The part of a complex input to be used in the computation of
      the metric. Must be one of `'real'`, `'imag'`, `'abs'` or `'angle'`. Note
      that real and imaginary parts, as well as angles, will be scaled to avoid
      negative numbers. This argument must be specified for complex inputs.
    name: String name of the metric instance.
    dtype: Data type of the metric result.
  """
  def __init__(self,
               max_val=None,
               rank=None,
               multichannel=True,
               complex_part=None,
               name='psnr',
               dtype=None):
    super().__init__(image_ops.psnr,
                     name=name,
                     dtype=dtype,
                     max_val=max_val,
                     rank=rank,
                     multichannel=multichannel,
                     complex_part=complex_part)


@api_util.export("metrics.StructuralSimilarity")
@tf.keras.utils.register_keras_serializable(package="MRI")
class StructuralSimilarity(_MeanMetricWrapperIQA):
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
    multichannel: A `boolean`. Whether multichannel computation is enabled. If
      `False`, the inputs `y_true` and `y_pred` are not expected to have a
      channel dimension, i.e. they should have shape
      `batch_shape + [height, width]` (2D) or
      `batch_shape + [depth, height, width]` (3D).
    complex_part: The part of a complex input to be used in the computation of
      the metric. Must be one of `'real'`, `'imag'`, `'abs'` or `'angle'`. Note
      that real and imaginary parts, as well as angles, will be scaled to avoid
      negative numbers. This argument must be specified for complex inputs.
    name: String name of the metric instance.
    dtype: Data type of the metric result.

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
               multichannel=True,
               complex_part=None,
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
                     rank=rank,
                     multichannel=multichannel,
                     complex_part=complex_part)


@api_util.export("metrics.MultiscaleStructuralSimilarity")
@tf.keras.utils.register_keras_serializable(package="MRI")
class MultiscaleStructuralSimilarity(_MeanMetricWrapperIQA):
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
    multichannel: A `boolean`. Whether multichannel computation is enabled. If
      `False`, the inputs `y_true` and `y_pred` are not expected to have a
      channel dimension, i.e. they should have shape
      `batch_shape + [height, width]` (2D) or
      `batch_shape + [depth, height, width]` (3D).
    complex_part: The part of a complex input to be used in the computation of
      the metric. Must be one of `'real'`, `'imag'`, `'abs'` or `'angle'`. Note
      that real and imaginary parts, as well as angles, will be scaled to avoid
      negative numbers. This argument must be specified for complex inputs.
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
               multichannel=True,
               complex_part=None,
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
                     rank=rank,
                     multichannel=multichannel,
                     complex_part=complex_part)
