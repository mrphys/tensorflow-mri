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
"""Coil compression layers."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.coils import coil_compression
from tensorflow_mri.python.layers import linear_operator_layer
from tensorflow_mri.python.linalg import linear_operator_mri
from tensorflow_mri.python.util import api_util


class CoilCompression(linear_operator_layer.LinearOperatorLayer):
  """Coil compression layer.

  This layer extracts a calibration region and compresses the coils.
  """
  def __init__(self,
               rank,
               calib_window='rect',
               calib_region=0.1 * np.pi,
               coil_compression_method='svd',
               coil_compression_kwargs=None,
               operator=linear_operator_mri.LinearOperatorMRI,
               kspace_index=None,
               **kwargs):
    """Initializes the layer."""
    super().__init__(operator=operator, input_indices=kspace_index, **kwargs)
    self.rank = rank
    self.calib_window = calib_window
    self.calib_region = calib_region
    self.coil_compression_method = coil_compression_method
    self.coil_compression_kwargs = coil_compression_kwargs or {}

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
    return coil_compression.compress_coils_with_calibration_data(
        kspace,
        operator,
        calib_window=self.calib_window,
        calib_region=self.calib_region,
        method=self.coil_compression_method,
        **self.coil_compression_kwargs)

  def get_config(self):
    """Returns the config of the layer.

    Returns:
      A `dict` describing the layer configuration.
    """
    config = {
        'calib_window': self.calib_window,
        'calib_region': self.calib_region,
        'coil_compression_method': self.coil_compression_method,
        'coil_compression_kwargs': self.coil_compression_kwargs
    }
    base_config = super().get_config()
    kspace_index = base_config.pop('input_indices')
    config['kspace_index'] = (
        kspace_index[0] if kspace_index is not None else None)
    return {**config, **base_config}


@api_util.export("layers.CoilCompression2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class CoilCompression2D(CoilCompression):
  def __init__(self, *args, **kwargs):
    super().__init__(2, *args, **kwargs)


@api_util.export("layers.CoilCompression3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class CoilCompression3D(CoilCompression):
  def __init__(self, *args, **kwargs):
    super().__init__(3, *args, **kwargs)
