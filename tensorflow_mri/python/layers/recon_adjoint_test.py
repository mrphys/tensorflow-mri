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
"""Tests for module `recon_adjoint`."""

import os
import tempfile

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.layers import recon_adjoint as recon_adjoint_layer
from tensorflow_mri.python.recon import recon_adjoint
from tensorflow_mri.python.util import test_util


class ReconAdjointTest(test_util.TestCase):
  @parameterized.product(expand_channel_dim=[True, False])
  def test_recon_adjoint(self, expand_channel_dim):
    # Create layer.
    layer = recon_adjoint_layer.ReconAdjoint2D(
        expand_channel_dim=expand_channel_dim)

    # Generate k-space data.
    image_shape = tf.constant([4, 4])
    kspace = tf.dtypes.complex(
        tf.random.stateless_normal(shape=image_shape, seed=[11, 22]),
        tf.random.stateless_normal(shape=image_shape, seed=[12, 34]))

    # Reconstruct image.
    expected = recon_adjoint.recon_adjoint_mri(kspace, image_shape)
    if expand_channel_dim:
      expected = tf.expand_dims(expected, axis=-1)

    # Test with dict inputs.
    input_data = {'kspace': kspace, 'image_shape': image_shape}
    result = layer(input_data)
    self.assertAllClose(expected, result)

    # Test (de)serialization.
    layer = recon_adjoint_layer.ReconAdjoint2D.from_config(layer.get_config())
    result = layer(input_data)
    self.assertAllClose(expected, result)

    # Test in model.
    inputs = {k: tf.keras.Input(shape=v.shape, dtype=v.dtype)
              for k, v in input_data.items()}
    model = tf.keras.Model(inputs, layer(inputs))
    result = model(input_data)
    self.assertAllClose(expected, result)

    # Test saving/loading.
    saved_model = os.path.join(tempfile.mkdtemp(), 'saved_model')
    model.save(saved_model)
    model = tf.keras.models.load_model(saved_model)
    result = model(input_data)
    self.assertAllClose(expected, result)
