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
"""Signal processing layers."""

import tensorflow as tf

from tensorflow_mri.python.ops import wavelet_ops
from tensorflow_mri.python.util import api_util


class DWT(tf.keras.layers.Layer):
  """Single-level discrete wavelet transform layer (private base class)."""
  def __init__(self, rank, inverse, wavelet, mode='symmetric', **kwargs):
    super().__init__(**kwargs)
    if not isinstance(wavelet, str):
      raise ValueError('wavelet must be a string')
    if not isinstance(mode, str):
      raise ValueError('mode must be a string')

    self.rank = rank
    self.inverse = inverse
    self.wavelet = wavelet
    self.mode = mode
    self.op = wavelet_ops.idwt if self.inverse else wavelet_ops.dwt
    self.axes = list(range(-(self.rank + 1), -1))

  def call(self, inputs):
    outputs = self.op(inputs,
                      wavelet=self.wavelet,
                      mode=self.mode,
                      axes=self.axes)
    return outputs

  def get_config(self):
    config = {
        'wavelet': self.wavelet,
        'mode': self.mode
    }
    base_config = super().get_config()
    return {**config, **base_config}


@api_util.export("layers.DWT1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class DWT1D(DWT):
  """1D discrete wavelet transform layer."""
  def __init__(self, wavelet, mode='symmetric', **kwargs):
    super().__init__(rank=1,
                     inverse=False,
                     wavelet=wavelet,
                     mode=mode,
                     **kwargs)


@api_util.export("layers.DWT2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class DWT2D(DWT):
  """2D discrete wavelet transform layer."""
  def __init__(self, wavelet, mode='symmetric', **kwargs):
    super().__init__(rank=2,
                     inverse=False,
                     wavelet=wavelet,
                     mode=mode,
                     **kwargs)


@api_util.export("layers.DWT3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class DWT3D(DWT):
  """3D discrete wavelet transform layer."""
  def __init__(self, wavelet, mode='symmetric', **kwargs):
    super().__init__(rank=3,
                     inverse=False,
                     wavelet=wavelet,
                     mode=mode,
                     **kwargs)


@api_util.export("layers.IDWT1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class IDWT1D(DWT):
  """1D inverse discrete wavelet transform layer."""
  def __init__(self, wavelet, mode='symmetric', **kwargs):
    super().__init__(rank=1,
                     inverse=True,
                     wavelet=wavelet,
                     mode=mode,
                     **kwargs)


@api_util.export("layers.IDWT2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class IDWT2D(DWT):
  """2D inverse discrete wavelet transform layer."""
  def __init__(self, wavelet, mode='symmetric', **kwargs):
    super().__init__(rank=2,
                     inverse=True,
                     wavelet=wavelet,
                     mode=mode,
                     **kwargs)


@api_util.export("layers.IDWT3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class IDWT3D(DWT):
  """3D inverse discrete wavelet transform layer."""
  def __init__(self, wavelet, mode='symmetric', **kwargs):
    super().__init__(rank=3,
                     inverse=True,
                     wavelet=wavelet,
                     mode=mode,
                     **kwargs)
