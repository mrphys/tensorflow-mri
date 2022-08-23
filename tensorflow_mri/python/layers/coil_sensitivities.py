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
"""Coil sensitivities layers."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.activations import complex_activations
from tensorflow_mri.python.coils import coil_sensitivities
from tensorflow_mri.python.layers import linear_operator_layer
from tensorflow_mri.python.linalg import linear_operator_mri
from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import model_util


class CoilSensitivityEstimation(linear_operator_layer.LinearOperatorLayer):
  """Coil sensitivity estimation layer.

  This layer extracts a calibration region and estimates the coil sensitivity
  maps.
  """
  def __init__(self,
               rank,
               calib_window,
               calib_method='walsh',
               calib_kwargs=None,
               sens_network='auto',
               reinterpret_complex=False,
               normalize=True,
               operator=linear_operator_mri.LinearOperatorMRI,
               kspace_index=None,
               **kwargs):
    """Initializes the layer."""
    super().__init__(operator=operator, input_indices=kspace_index, **kwargs)
    self.rank = rank
    self.calib_window = calib_window
    self.calib_method = calib_method
    self.calib_kwargs = calib_kwargs or {}
    self.sens_network = sens_network
    self.reinterpret_complex = reinterpret_complex
    self.normalize = normalize

    if self.sens_network == 'auto':
      sens_network_class = model_util.get_nd_model('UNet', rank)
      sens_network_kwargs = dict(
          filters=[32, 64, 128],
          kernel_size=3,
          activation=('relu' if self.reinterpret_complex
                      else complex_activations.complex_relu),
          out_channels=2 if self.reinterpret_complex else 1,
          use_deconv=True,
          dtype=(tf.as_dtype(self.dtype).real_dtype.name
                 if self.reinterpret_complex else self.dtype)
      )
      self._sens_network_layer = tf.keras.layers.TimeDistributed(
          sens_network_class(**sens_network_kwargs))
    else:
      self._sens_network_layer = tf.keras.layers.TimeDistributed(sens_network)

  def call(self, inputs):
    """Applies the layer.

    Args:
      inputs: A `tuple` or `dict` containing the *k*-space data as defined by
        `kspace_index`. If `operator` is a class not an instance, then `inputs`
        must also contain any other arguments to be passed to the constructor of
        `operator`.

    Returns:
      The scaled k-space data.
    """
    kspace, operator = self.parse_inputs(inputs)
    sensitivities = (
        coil_sensitivities.estimate_sensitivities_with_calibration_data(
            kspace,
            operator,
            calib_window=self.calib_window,
            method=self.calib_method,
            **self.calib_kwargs
        )
    )

    if self.sens_network is not None:
      sensitivities = tf.expand_dims(sensitivities, axis=-1)
      if self.reinterpret_complex:
        sensitivities = math_ops.view_as_real(sensitivities, stacked=False)
      sensitivities = self._sens_network_layer(sensitivities)
      if self.reinterpret_complex:
        sensitivities = math_ops.view_as_complex(sensitivities, stacked=False)
      sensitivities = tf.squeeze(sensitivities, axis=-1)

    if self.normalize:
      coil_axis = -(self.rank + 1)
      sensitivities = math_ops.normalize_no_nan(sensitivities, axis=coil_axis)

    return sensitivities

  def get_config(self):
    """Returns the config of the layer.

    Returns:
      A `dict` describing the layer configuration.
    """
    config = {
        'calib_window': self.calib_window,
        'calib_method': self.calib_method,
        'calib_kwargs': self.calib_kwargs,
        'sens_network': self.sens_network,
        'reinterpret_complex': self.reinterpret_complex,
        'normalize': self.normalize
    }
    base_config = super().get_config()
    kspace_index = base_config.pop('input_indices')
    config['kspace_index'] = (
        kspace_index[0] if kspace_index is not None else None)
    return {**config, **base_config}


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
