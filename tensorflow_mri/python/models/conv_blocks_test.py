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

from tensorflow_mri.python.activations import complex_activations
from tensorflow_mri.python.models import conv_blocks
from tensorflow_mri.python.util import model_util
from tensorflow_mri.python.util import test_util


class ConvBlockTest(test_util.TestCase):
  """Tests for `ConvBlock`."""
  @test_util.run_all_execution_modes
  @parameterized.named_parameters(
      ("test0", 2, 64, 3),
      ("test1", 3, 32, 3))
  def test_conv_block_creation(self, rank, filters, kernel_size): # pylint: disable=missing-param-doc
    """Test object creation."""
    inputs = tf.keras.Input(
        shape=(128,) * rank + (32,), batch_size=1)

    block = model_util.get_nd_model('ConvBlock', rank)(
        filters=filters, kernel_size=kernel_size)

    features = block(inputs)

    self.assertAllEqual(features.shape, [1] + [128] * rank + [filters])


  def test_complex_valued(self):
    inputs = tf.dtypes.complex(
        tf.random.stateless_normal(shape=(2, 32, 32, 4), seed=[12, 34]),
        tf.random.stateless_normal(shape=(2, 32, 32, 4), seed=[56, 78]))

    block = conv_blocks.ConvBlock2D(
        filters=[6, 6],
        kernel_size=3,
        activation=complex_activations.complex_relu,
        dtype=tf.complex64)

    result = block(inputs)
    self.assertAllClose((2, 32, 32, 6), result.shape)
    self.assertDTypeEqual(result, tf.complex64)


  def test_serialize_deserialize(self):
    """Test de/serialization."""
    config = dict(
        filters=[32],
        kernel_size=3,
        strides=1,
        activation='tanh',
        out_activation='linear',
        use_bias=False,
        kernel_initializer='ones',
        bias_initializer='ones',
        kernel_regularizer='l2',
        bias_regularizer='l1',
        use_batch_norm=True,
        use_sync_bn=True,
        bn_momentum=0.98,
        bn_epsilon=0.002,
        use_residual=True,
        use_dropout=True,
        dropout_rate=0.5,
        dropout_type='spatial')

    block = conv_blocks.ConvBlock2D(**config)
    self.assertEqual(config, block.get_config())

    block2 = conv_blocks.ConvBlock2D.from_config(block.get_config())
    self.assertAllEqual(block2.get_config(), block.get_config())


if __name__ == '__main__':
  tf.test.main()
