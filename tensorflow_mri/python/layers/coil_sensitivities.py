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
               sens_network='UNet',
               sens_network_kwargs=None,
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
    self.sens_network_kwargs = sens_network_kwargs or {}

    sens_network_kwargs = _default_sens_network_kwargs(self.sens_network)
    sens_network_kwargs.update(self.sens_network_kwargs)

    if self.sens_network is not None:
      sens_network_class = model_util.get_nd_model(self.sens_network, rank)
      sens_network_kwargs = sens_network_kwargs.copy()
      self._sens_network_layer = tf.keras.layers.TimeDistributed(
          sens_network_class(**sens_network_kwargs))

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
      sensitivities = self._sens_network_layer(sensitivities)
      sensitivities = tf.squeeze(sensitivities, axis=-1)
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
        'sens_network_kwargs': self.sens_network_kwargs
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


def _default_sens_network_kwargs(name):
  return {
      'UNet': dict(
          filters=[32, 64, 128],
          kernel_size=3,
          activation=complex_activations.complex_relu,
          out_channels=1,
          dtype=tf.complex64
      )
  }.get(name, {})
