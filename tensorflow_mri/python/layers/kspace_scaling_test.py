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
    image_shape = [4, 4]

    kspace = tf.dtypes.complex(
        tf.random.stateless_normal(shape=image_shape, seed=[11, 22]),
        tf.random.stateless_normal(shape=image_shape, seed=[12, 34]))
    inputs = (kspace, image_shape)

    result = layer(inputs)

    image = recon_adjoint.recon_adjoint_mri(kspace, image_shape)
    expected = kspace / tf.math.reduce_max(tf.math.abs(image))

    self.assertAllClose(expected, result)
