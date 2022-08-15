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
"""*k*-space scaling layer."""

import tensorflow as tf

from tensorflow_mri.python.layers import linear_operator_layer
from tensorflow_mri.python.linalg import linear_operator_mri
from tensorflow_mri.python.recon import recon_adjoint


class KSpaceScaling(linear_operator_layer.LinearOperatorLayer):
  """K-space scaling layer.

  This layer scales the *k*-space data so that the adjoint reconstruction has
  values between 0 and 1.
  """
  def __init__(self,
               operator=linear_operator_mri.LinearOperatorMRI,
               kspace_index=None,
               **kwargs):
    """Initializes the layer."""
    super().__init__(operator=operator, input_indices=kspace_index, **kwargs)

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
    image = recon_adjoint.recon_adjoint(kspace, operator)
    return kspace / tf.cast(tf.math.reduce_max(tf.math.abs(image)),
                            kspace.dtype)

  def get_config(self):
    """Returns the config of the layer.

    Returns:
      A `dict` describing the layer configuration.
    """
    config = super().get_config()
    kspace_index = config.pop('input_indices')
    if kspace_index is not None:
      kspace_index = kspace_index[0]
    return config
