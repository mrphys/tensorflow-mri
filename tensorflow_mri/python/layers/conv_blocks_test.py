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
"""Tests for module `conv_blocks`."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.layers import conv_blocks
from tensorflow_mri.python.util import test_util


class ConvBlockTest(test_util.TestCase):

  @parameterized.parameters((64, 3, 2), (32, 3, 3))
  @test_util.run_in_graph_and_eager_modes
  def test_conv_block_creation(self, filters, kernel_size, rank):
    """Test object creation."""
    inputs = tf.keras.Input(
        shape=(128,) * rank + (32,), batch_size=1)

    block = conv_blocks.ConvBlock(
        filters=filters, kernel_size=kernel_size)

    features = block(inputs)

    self.assertAllEqual(features.shape, [1] + [128] * rank + [filters])


  def test_serialize_deserialize(self):
    """Test de/serialization."""
    config = dict(
        filters=[32],
        kernel_size=3,
        strides=1,
        rank=2,
        activation='tanh',
        kernel_initializer='ones',
        bias_initializer='ones',
        kernel_regularizer='l2',
        bias_regularizer='l1',
        use_batch_norm=True,
        use_sync_bn=True,
        bn_momentum=0.98,
        bn_epsilon=0.002,
        name='conv_block',
        dtype='float32',
        trainable=True)

    block = conv_blocks.ConvBlock(**config)
    self.assertEqual(block.get_config(), config)

    block2 = conv_blocks.ConvBlock.from_config(block.get_config())
    self.assertAllEqual(block.get_config(), block2.get_config())


if __name__ == '__main__':
  tf.test.main()
