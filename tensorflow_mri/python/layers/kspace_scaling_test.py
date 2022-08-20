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
"""Tests for module `kspace_scaling`."""

import tensorflow as tf

from tensorflow_mri.python.layers import kspace_scaling
from tensorflow_mri.python.recon import recon_adjoint
from tensorflow_mri.python.util import test_util


class KSpaceScalingTest(test_util.TestCase):
  """Tests for module `kspace_scaling`."""
  def test_kspace_scaling(self):
    """Tests the k-space scaling layer."""
    layer = kspace_scaling.KSpaceScaling()
    self.assertEqual(layer.dtype, "complex64")

    image_shape = tf.convert_to_tensor([4, 4])

    kspace = tf.dtypes.complex(
        tf.random.stateless_normal(shape=image_shape, seed=[11, 22]),
        tf.random.stateless_normal(shape=image_shape, seed=[12, 34]))

    # This mask simulates the default filtering operation.
    mask = tf.constant([[False, False, False, False],
                        [False, False, False, False],
                        [False, False, True, False],
                        [False, False, False, False]], dtype=tf.bool)

    filtered_kspace = tf.where(mask, kspace, tf.zeros_like(kspace))
    image = recon_adjoint.recon_adjoint_mri(filtered_kspace, image_shape)
    expected = kspace / tf.cast(tf.math.reduce_max(tf.math.abs(image)),
                                kspace.dtype)

    # Test with tuple inputs.
    inputs = (kspace, image_shape)
    result = layer(inputs)
    self.assertAllClose(expected, result)

    # Test with dict inputs.
    inputs = {'kspace': kspace, 'image_shape': image_shape}
    result = layer(inputs)
    self.assertAllClose(expected, result)

    # Test (de)serialization.
    layer = kspace_scaling.KSpaceScaling.from_config(layer.get_config())
    result = layer(inputs)
    self.assertAllClose(expected, result)