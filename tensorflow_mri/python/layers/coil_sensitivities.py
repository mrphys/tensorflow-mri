# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Coil sensitivity estimation layer."""

import string

import tensorflow as tf

from tensorflow_mri.python.activations import complex_activations
from tensorflow_mri.python.coils import coil_sensitivities
from tensorflow_mri.python.linalg import linear_operator_mri
from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import doc_util
from tensorflow_mri.python.util import model_util


class CoilSensitivityEstimation(tf.keras.layers.Layer):
  r"""${rank}D coil sensitivity estimation layer.

  This layer extracts a calibration region and estimates the coil sensitivity
  maps.
  """
  def __init__(self,
               rank,
               calib_fn=None,
               algorithm='walsh',
               algorithm_kwargs=None,
               refine_sensitivities=False,
               refinement_network=None,
               normalize_sensitivities=True,
               expand_channel_dim=False,
               reinterpret_complex=False,
               **kwargs):
    super().__init__(**kwargs)
    self.rank = rank
    self.calib_fn = calib_fn
    self.algorithm = algorithm
    self.algorithm_kwargs = algorithm_kwargs or {}
    self.refine_sensitivities = refine_sensitivities
    self.refinement_network = refinement_network
    self.normalize_sensitivities = normalize_sensitivities
    self.expand_channel_dim = expand_channel_dim
    self.reinterpret_complex = reinterpret_complex

    if self.refine_sensitivities and self.refinement_network is None:
      # Default map refinement network.
      dtype = tf.as_dtype(self.dtype)
      network_class = model_util.get_nd_model('UNet', rank)
      network_kwargs = dict(
          filters=[32, 64, 128],
          kernel_size=3,
          activation=('relu' if self.reinterpret_complex else 'complex_relu'),
          output_filters=2 if self.reinterpret_complex else 1,
          dtype=dtype.real_dtype if self.reinterpret_complex else dtype)
      self.refinement_network = tf.keras.layers.TimeDistributed(
          network_class(**network_kwargs))

  def call(self, inputs):
    data, operator, calib_data = parse_inputs(inputs)

    # Compute coil sensitivities.
    maps = coil_sensitivities.estimate_sensitivities_universal(
        data,
        operator,
        calib_data=calib_data,
        calib_fn=self.calib_fn,
        algorithm=self.algorithm,
        **self.algorithm_kwargs)

    # Maybe refine coil sensitivities.
    if self.refine_sensitivities:
      maps = tf.expand_dims(maps, axis=-1)
      if self.reinterpret_complex:
        maps = math_ops.view_as_real(maps, stacked=False)
      maps = self.refinement_network(maps)
      if self.reinterpret_complex:
        maps = math_ops.view_as_complex(maps, stacked=False)
      maps = tf.squeeze(maps, axis=-1)

    # Maybe normalize coil sensitivities.
    if self.normalize_sensitivities:
      coil_axis = -(self.rank + 1)
      maps = math_ops.normalize_no_nan(maps, axis=coil_axis)

    # # Post-processing.
    # if self.expand_channel_dim:
    #   maps = tf.expand_dims(maps, axis=-1)
    # if self.reinterpret_complex and maps.dtype.is_complex:
    #   maps = math_ops.view_as_real(maps, stacked=False)

    return maps

  def get_config(self):
    base_config = super().get_config()
    config = {
        'calib_fn': self.calib_fn,
        'algorithm': self.algorithm,
        'algorithm_kwargs': self.algorithm_kwargs,
        'refine_sensitivities': self.refine_sensitivities,
        'refinement_network': self.refinement_network,
        'normalize_sensitivities': self.normalize_sensitivities,
        'expand_channel_dim': self.expand_channel_dim,
        'reinterpret_complex': self.reinterpret_complex,
    }
    return {**base_config, **config}


def parse_inputs(inputs):
  def _parse_inputs(data, operator, calib_data=None):
    return data, operator, calib_data
  if isinstance(inputs, tuple):
    return _parse_inputs(*inputs)
  elif isinstance(inputs, dict):
    return _parse_inputs(**inputs)
  raise ValueError('inputs must be a tuple or dict')


@api_util.export("layers.CoilSensitivityEstimation2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class CoilSensitivityEstimation2D(CoilSensitivityEstimation):
  def __init__(self, *args, **kwargs):
    super().__init__(2, *args, **kwargs)


@api_util.export("layers.CoilSensitivityEstimation3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class CoilSensitivityEstimation3D(CoilSensitivityEstimation):
  def __init__(self, *args, **kwargs):
    super().__init__(3, *args, **kwargs)


CoilSensitivityEstimation2D.__doc__ = string.Template(
    CoilSensitivityEstimation.__doc__).safe_substitute(rank=2)
CoilSensitivityEstimation3D.__doc__ = string.Template(
    CoilSensitivityEstimation.__doc__).safe_substitute(rank=3)


CoilSensitivityEstimation2D.__signature__ = doc_util.get_nd_layer_signature(
    CoilSensitivityEstimation)
CoilSensitivityEstimation3D.__signature__ = doc_util.get_nd_layer_signature(
    CoilSensitivityEstimation)
