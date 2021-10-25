# Copyright 2021 University College London. All Rights Reserved.
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
"""Tests for module `conv_endec`."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.layers import conv_endec
from tensorflow_mri.python.util import test_util


class UNetTest(test_util.TestCase):

  @parameterized.parameters((3, 16, 3, 2, None, True, False),
                            (2, 4, 3, 3, None, False, False),
                            (2, 8, 5, 2, 2, False, False),
                            (2, 8, 5, 2, 16, False, True))
  @test_util.run_in_graph_and_eager_modes
  def test_unet_creation(self,
                         scales,
                         base_filters,
                         kernel_size,
                         rank,
                         out_channels,
                         use_deconv,
                         use_residual):
    """Test object creation."""
    inputs = tf.keras.Input(
        shape=(128,) * rank + (16,), batch_size=1)

    network = conv_endec.UNet(
        scales=scales,
        base_filters=base_filters,
        kernel_size=kernel_size,
        rank=rank,
        use_deconv=use_deconv,
        out_channels=out_channels,
        use_residual=use_residual)

    features = network(inputs)
    if out_channels is None:
      out_channels = base_filters

    self.assertAllEqual(features.shape, [1] + [128] * rank + [out_channels])


  def test_serialize_deserialize(self):
    """Test de/serialization."""
    config = dict(
        scales=3,
        base_filters=16,
        kernel_size=2,
        pool_size=2,
        rank=2,
        block_depth=2,
        use_deconv=True,
        activation='tanh',
        kernel_initializer='ones',
        bias_initializer='ones',
        kernel_regularizer='l2',
        bias_regularizer='l1',
        use_batch_norm=True,
        use_sync_bn=True,
        bn_momentum=0.98,
        bn_epsilon=0.002,
        out_channels=1,
        out_activation='relu',
        use_residual=True,
        name='conv_block',
        dtype='float32',
        trainable=True)

    block = conv_endec.UNet(**config)
    self.assertEqual(block.get_config(), config)

    block2 = conv_endec.UNet.from_config(block.get_config())
    self.assertAllEqual(block.get_config(), block2.get_config())


if __name__ == '__main__':
  tf.test.main()
