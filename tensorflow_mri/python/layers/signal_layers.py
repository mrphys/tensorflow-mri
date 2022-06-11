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

import string

import tensorflow as tf

from tensorflow_mri.python.ops import wavelet_ops
from tensorflow_mri.python.util import api_util


DWT_KEYS_1D = ['a', 'd']
DWT_KEYS_2D = ['aa', 'ad', 'da', 'dd']
DWT_KEYS_3D = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']


DWT_DOC_TEMPLATE = string.Template(
  """${rank}D ${name_long} (${name_short}) layer.

  Args:
    wavelet: A `str` or a length-${rank} `list` of `str`. When passed a `list`,
      different wavelets are applied along each axis.
    mode: A `str`. The padding or signal extension mode. Must be one of the
      values supported by `tfmri.signal.dwt`. Defaults to `'symmetric'`.
    return_dict: A `boolean`. If `True`, returns a `dict` containing the
      keys ${output_keys}, where `'a'` is for approximation and `'d'` is for
      detail, and the corresponding values. If `False`, returns a `list` of
      values corresponding to the keys above. Defaults to `True`.
  """)


class DWT(tf.keras.layers.Layer):
  """ND discrete wavelet transform layer (private base class)."""
  def __init__(self, rank, inverse, wavelet, mode, return_dict=True, **kwargs):
    super().__init__(**kwargs)
    if not isinstance(wavelet, str):
      raise ValueError('wavelet must be a string')
    if not isinstance(mode, str):
      raise ValueError('mode must be a string')

    self.rank = rank
    self.inverse = inverse
    self.wavelet = wavelet
    self.mode = mode
    self.return_dict = return_dict
    self.op = wavelet_ops.idwt if self.inverse else wavelet_ops.dwt
    self.axes = list(range(-(self.rank + 1), -1))

  def call(self, inputs):  # pylint: disable=missing-function-docstring
    outputs = self.op(inputs,
                      wavelet=self.wavelet,
                      mode=self.mode,
                      axes=self.axes)
    if not self.return_dict:
      if self.rank == 1:
        names = DWT_KEYS_1D
      elif self.rank == 2:
        names = DWT_KEYS_2D
      elif self.rank == 3:
        names = DWT_KEYS_3D
      else:
        raise NotImplementedError('rank must be 1, 2, or 3')
      outputs = [outputs[name] for name in names]
    return outputs

  def get_config(self):
    config = {
        'wavelet': self.wavelet,
        'mode': self.mode,
        'return_dict': self.return_dict
    }
    base_config = super().get_config()
    return {**config, **base_config}


@api_util.export("layers.DWT1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class DWT1D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               return_dict=True,
               **kwargs):
    super().__init__(rank=1,
                     inverse=False,
                     wavelet=wavelet,
                     mode=mode,
                     return_dict=return_dict,
                     **kwargs)


@api_util.export("layers.DWT2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class DWT2D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               return_dict=True,
               **kwargs):
    super().__init__(rank=2,
                     inverse=False,
                     wavelet=wavelet,
                     mode=mode,
                     return_dict=return_dict,
                     **kwargs)


@api_util.export("layers.DWT3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class DWT3D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               return_dict=True,
               **kwargs):
    super().__init__(rank=3,
                     inverse=False,
                     wavelet=wavelet,
                     mode=mode,
                     return_dict=return_dict,
                     **kwargs)


@api_util.export("layers.IDWT1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class IDWT1D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               return_dict=True,
               **kwargs):
    super().__init__(rank=1,
                     inverse=True,
                     wavelet=wavelet,
                     mode=mode,
                     return_dict=return_dict,
                     **kwargs)


@api_util.export("layers.IDWT2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class IDWT2D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               return_dict=True,
               **kwargs):
    super().__init__(rank=2,
                     inverse=True,
                     wavelet=wavelet,
                     mode=mode,
                     return_dict=return_dict,
                     **kwargs)


@api_util.export("layers.IDWT3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class IDWT3D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               return_dict=True,
               **kwargs):
    super().__init__(rank=3,
                     inverse=True,
                     wavelet=wavelet,
                     mode=mode,
                     return_dict=return_dict,
                     **kwargs)


DWT1D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=1,
  name_long='discrete wavelet transform',
  name_short='DWT',
  output_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_1D)))


DWT2D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=2,
  name_long='discrete wavelet transform',
  name_short='DWT',
  output_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_2D)))


DWT3D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=3,
  name_long='discrete wavelet transform',
  name_short='DWT',
  output_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_3D)))


IDWT1D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=1,
  name_long='inverse discrete wavelet transform',
  name_short='IDWT',
  output_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_1D)))


IDWT2D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=2,
  name_long='inverse discrete wavelet transform',
  name_short='IDWT',
  output_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_2D)))


IDWT3D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=3,
  name_long='inverse discrete wavelet transform',
  name_short='IDWT',
  output_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_3D)))
