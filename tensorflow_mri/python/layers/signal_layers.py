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
  """Single-level ${rank}D ${name_long} (${name_short}) layer.

  ${doc_inputs}

  Args:
    wavelet: A `str` or a length-${rank} `list` of `str`. When passed a `list`,
      different wavelets are applied along each axis.
    mode: A `str`. The padding or signal extension mode. Must be one of the
      values supported by `tfmri.signal.dwt`. Defaults to `'symmetric'`.
    format_dict: A `boolean`. If `True`, the ${input_or_output} is a `dict`.
      Otherwise, it is a `list`.
  """)


DWT_FORWARD_DOC = string.Template(
  """
  The input must be a tensor of shape `[batch_size, ${in_dims}, channels]`.

  The output format is determined by the `format_dict` argument. If
  `format_dict` is `True` (default), the output is a `dict` with keys
  ${out_keys}, where `'a'` is for approximation and `'d'` is for
  detail. The value for each key is a tensor of shape
  `[batch_size, ${out_dims}, channels]`. The size of each output
  dimension is determined by `out_dim = (in_dim + filter_len - 1) // 2`,
  where `filter_len` is the length of the decomposition filters for the selected
  wavelet. If `format_dict` is `False`, returns a `list` of tensors
  corresponding to each of the keys above.
  """)


DWT_INVERSE_DOC = string.Template(
  """
  The input format is determined by the `format_dict` argument. If
  `format_dict` is `True` (default), the input should be a `dict` with keys
  ${out_keys}, where `'a'` is for approximation and `'d'` is for detail. The
  value for each key should be a tensor of shape
  `[batch_size, ${in_dims}, channels]`. If `format_dict` is `False`, the input
  should be a `list` of tensors corresponding to each of the keys above.

  The output is a tensor of shape `[batch_size, ${out_dims}, channels]`.
  """)


class DWT(tf.keras.layers.Layer):
  """ND discrete wavelet transform layer (private base class)."""
  def __init__(self, rank, inverse, wavelet, mode, format_dict=True, **kwargs):
    super().__init__(**kwargs)
    if not isinstance(wavelet, str):
      raise ValueError('wavelet must be a string')
    if not isinstance(mode, str):
      raise ValueError('mode must be a string')

    self.rank = rank
    self.inverse = inverse
    self.wavelet = wavelet
    self.mode = mode
    self.format_dict = format_dict
    self.op = wavelet_ops.idwt if self.inverse else wavelet_ops.dwt
    self.axes = list(range(-(self.rank + 1), -1))
    if self.rank == 1:
      self.dict_keys = DWT_KEYS_1D
    elif self.rank == 2:
      self.dict_keys = DWT_KEYS_2D
    elif self.rank == 3:
      self.dict_keys = DWT_KEYS_3D
    else:
      raise NotImplementedError('rank must be 1, 2, or 3')

  def call(self, inputs):  # pylint: disable=missing-function-docstring
    # If not using dict format, convert input to dict.
    if self.inverse and not self.format_dict:
      if not isinstance(inputs, (list, tuple)):
        raise ValueError(f'expected a list of inputs, but got: {type(inputs)}')
      inputs = dict(zip(self.dict_keys, inputs))

    # Compute the (I)DWT.
    outputs = self.op(inputs,
                      wavelet=self.wavelet,
                      mode=self.mode,
                      axes=self.axes)

    # If not using dict format, convert output to list.
    if not self.inverse and not self.format_dict:
      outputs = [outputs[k] for k in self.dict_keys]

    return outputs

  def get_config(self):
    config = {
        'wavelet': self.wavelet,
        'mode': self.mode,
        'format_dict': self.format_dict
    }
    base_config = super().get_config()
    return {**config, **base_config}


@api_util.export("layers.DWT1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class DWT1D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               format_dict=True,
               **kwargs):
    super().__init__(rank=1,
                     inverse=False,
                     wavelet=wavelet,
                     mode=mode,
                     format_dict=format_dict,
                     **kwargs)


@api_util.export("layers.DWT2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class DWT2D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               format_dict=True,
               **kwargs):
    super().__init__(rank=2,
                     inverse=False,
                     wavelet=wavelet,
                     mode=mode,
                     format_dict=format_dict,
                     **kwargs)


@api_util.export("layers.DWT3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class DWT3D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               format_dict=True,
               **kwargs):
    super().__init__(rank=3,
                     inverse=False,
                     wavelet=wavelet,
                     mode=mode,
                     format_dict=format_dict,
                     **kwargs)


@api_util.export("layers.IDWT1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class IDWT1D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               format_dict=True,
               **kwargs):
    super().__init__(rank=1,
                     inverse=True,
                     wavelet=wavelet,
                     mode=mode,
                     format_dict=format_dict,
                     **kwargs)


@api_util.export("layers.IDWT2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class IDWT2D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               format_dict=True,
               **kwargs):
    super().__init__(rank=2,
                     inverse=True,
                     wavelet=wavelet,
                     mode=mode,
                     format_dict=format_dict,
                     **kwargs)


@api_util.export("layers.IDWT3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class IDWT3D(DWT):
  def __init__(self,
               wavelet='haar',
               mode='symmetric',
               format_dict=True,
               **kwargs):
    super().__init__(rank=3,
                     inverse=True,
                     wavelet=wavelet,
                     mode=mode,
                     format_dict=format_dict,
                     **kwargs)


DWT1D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=1,
  name_long='discrete wavelet transform',
  name_short='DWT',
  doc_inputs=DWT_FORWARD_DOC.substitute(
      in_dims='width',
      out_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_1D)),
      out_dims='out_width'),
  input_or_output='output')


DWT2D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=2,
  name_long='discrete wavelet transform',
  name_short='DWT',
  doc_inputs=DWT_FORWARD_DOC.substitute(
      in_dims='height, width',
      out_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_2D)),
      out_dims='out_height, out_width'),
  input_or_output='output')


DWT3D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=3,
  name_long='discrete wavelet transform',
  name_short='DWT',
  doc_inputs=DWT_FORWARD_DOC.substitute(
      in_dims='depth, height, width',
      out_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_3D)),
      out_dims='out_depth, out_height, out_width'),
  input_or_output='output')


IDWT1D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=1,
  name_long='inverse discrete wavelet transform',
  name_short='IDWT',
  doc_inputs=DWT_INVERSE_DOC.substitute(
      in_dims='width',
      out_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_1D)),
      out_dims='out_width'),
  input_or_output='input')


IDWT2D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=2,
  name_long='inverse discrete wavelet transform',
  name_short='IDWT',
  doc_inputs=DWT_INVERSE_DOC.substitute(
      in_dims='height, width',
      out_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_2D)),
      out_dims='out_height, out_width'),
  input_or_output='input')


IDWT3D.__doc__ = DWT_DOC_TEMPLATE.substitute(
  rank=3,
  name_long='inverse discrete wavelet transform',
  name_short='IDWT',
  doc_inputs=DWT_INVERSE_DOC.substitute(
      in_dims='depth, height, width',
      out_keys=', '.join(map(lambda x: f"`'{x}'`", DWT_KEYS_3D)),
      out_dims='out_depth, out_height, out_width'),
  input_or_output='input')
