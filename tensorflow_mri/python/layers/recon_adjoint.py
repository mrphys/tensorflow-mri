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
"""Adjoint reconstruction layer."""

import tensorflow as tf

from tensorflow_mri.python.layers import linear_operator_layer
from tensorflow_mri.python.linalg import linear_operator_mri
from tensorflow_mri.python.recon import recon_adjoint
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import keras_util


@api_util.export("layers.ReconAdjoint")
@tf.keras.utils.register_keras_serializable(package="MRI")
class ReconAdjoint(linear_operator_layer.LinearOperatorLayer):
  """Adjoint reconstruction layer.

  This layer reconstructs a signal using the adjoint of the system operator.
  """
  def __init__(self,
               channel_dimension=True,
               operator=linear_operator_mri.LinearOperatorMRI,
               kspace_index=None,
               **kwargs):
    """Initializes the layer."""
    kwargs['dtype'] = kwargs.get('dtype') or keras_util.complexx()
    super().__init__(operator=operator, input_indices=kspace_index, **kwargs)
    self.channel_dimension = channel_dimension

  def call(self, inputs):
    """Applies the layer.

    Args:
      inputs: A `tuple` or `dict` containing the *k*-space data as defined by
        `kspace_index`. If `operator` is a class not an instance, then `inputs`
        must also contain any other arguments to be passed to the constructor of
        `operator`.

    Returns:
      The reconstructed k-space data.
    """
    kspace, operator = self.parse_inputs(inputs)
    image = recon_adjoint.recon_adjoint(kspace, operator)
    if self.channel_dimension:
      image = tf.expand_dims(image, axis=-1)
    return image

  def get_config(self):
    """Returns the config of the layer.

    Returns:
      A `dict` describing the layer configuration.
    """
    config = {
        'channel_dimension': self.channel_dimension
    }
    base_config = super().get_config()
    kspace_index = base_config.pop('input_indices')
    config['kspace_index'] = (
        kspace_index[0] if kspace_index is not None else None)
    return {**config, **base_config}
