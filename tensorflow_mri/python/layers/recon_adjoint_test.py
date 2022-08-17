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

import tensorflow as tf

from tensorflow_mri.python.layers import recon_adjoint as recon_adjoint_layer
from tensorflow_mri.python.recon import recon_adjoint
from tensorflow_mri.python.util import test_util


class ReconAdjointTest(test_util.TestCase):
  def test_recon_adjoint(self):
    # Create layer.
    layer = recon_adjoint_layer.ReconAdjoint()

    # Generate k-space data.
    image_shape = tf.constant([4, 4])
    kspace = tf.dtypes.complex(
        tf.random.stateless_normal(shape=image_shape, seed=[11, 22]),
        tf.random.stateless_normal(shape=image_shape, seed=[12, 34]))

    # Reconstruct image.
    expected = recon_adjoint.recon_adjoint_mri(kspace, image_shape)

    # Test with tuple inputs.
    inputs = (kspace, image_shape)
    result = layer(inputs)
    self.assertAllClose(expected, result)

    # Test with dict inputs.
    inputs = {'kspace': kspace, 'image_shape': image_shape}
    result = layer(inputs)
    self.assertAllClose(expected, result)

    # Test (de)serialization.
    layer = recon_adjoint_layer.ReconAdjoint.from_config(layer.get_config())
    result = layer(inputs)
    self.assertAllClose(expected, result)
