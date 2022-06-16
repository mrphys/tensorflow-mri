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
# pylint: disable=missing-param-doc

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.models import conv_endec
from tensorflow_mri.python.util import test_util


class UNetTest(test_util.TestCase):
  """U-Net tests."""
  # pylint disable=missing-param-doc
  @test_util.run_all_execution_modes
  @parameterized.named_parameters(
      ("test0", 2, [16, 32, 64], 3, None, True, False),
      ("test1", 3, [4, 8], 3, None, False, False),
      ("test2", 2, [8, 16], 5, 2, False, False),
      ("test3", 2, [8, 16], 5, 16, False, True))
  def test_unet_creation(self, # pylint: disable=missing-param-doc
                         rank,
                         filters,
                         kernel_size,
                         out_channels,
                         use_deconv,
                         use_global_residual):
    """Test object creation."""
    inputs = tf.keras.Input(
        shape=(128,) * rank + (16,), batch_size=1)

    layer = {
        1: conv_endec.UNet1D,
        2: conv_endec.UNet2D,
        3: conv_endec.UNet3D
    }
    network = layer[rank](
        filters=filters,
        kernel_size=kernel_size,
        use_deconv=use_deconv,
        out_channels=out_channels,
        use_global_residual=use_global_residual)

    features = network(inputs)
    if out_channels is None:
      out_channels = filters[0]

    self.assertAllEqual(features.shape, [1] + [128] * rank + [out_channels])


  @test_util.run_all_execution_modes
  @parameterized.named_parameters(
      ("true", True),
      ("false", False))
  def test_use_bias(self, use_bias):
    """Test `use_bias` argument."""
    model = conv_endec.UNet2D(
        filters=[8, 16],
        kernel_size=3,
        use_bias=use_bias)
    layers = []
    for layer in model.layers:
      if isinstance(layer, tf.keras.Model):
        # Expand nested models (ConvBlock).
        layers.extend(layer.layers)
      else:
        layers.append(layer)
    # For each layer with a `use_bias` attribute, check that it is set
    # correctly.
    for layer in layers:
      if hasattr(layer, 'use_bias'):
        self.assertEqual(use_bias, layer.use_bias)


  def test_serialize_deserialize(self):
    """Test de/serialization."""
    config = dict(
        filters=[16, 32, 64],
        kernel_size=2,
        pool_size=2,
        block_depth=2,
        use_deconv=True,
        activation='tanh',
        use_bias=False,
        kernel_initializer='ones',
        bias_initializer='ones',
        kernel_regularizer='l2',
        bias_regularizer='l1',
        use_batch_norm=True,
        use_sync_bn=True,
        bn_momentum=0.98,
        bn_epsilon=0.002,
        out_channels=1,
        out_kernel_size=1,
        out_activation='relu',
        use_global_residual=True,
        use_dropout=True,
        dropout_rate=0.5,
        dropout_type='spatial',
        use_tight_frame=True)

    block = conv_endec.UNet2D(**config)
    self.assertEqual(block.get_config(), config)

    block2 = conv_endec.UNet2D.from_config(block.get_config())
    self.assertAllEqual(block.get_config(), block2.get_config())


if __name__ == '__main__':
  tf.test.main()
