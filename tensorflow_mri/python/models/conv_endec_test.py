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

from tensorflow_mri.python.activations import complex_activations
from tensorflow_mri.python.layers import convolutional
from tensorflow_mri.python.layers import pooling
from tensorflow_mri.python.layers import reshaping
from tensorflow_mri.python.models import conv_blocks
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
                         output_filters,
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
        output_filters=output_filters,
        use_global_residual=use_global_residual)

    features = network(inputs)
    if output_filters is None:
      output_filters = filters[0]

    self.assertAllEqual(features.shape, [1] + [128] * rank + [output_filters])


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


  def test_complex_valued(self):
    inputs = tf.dtypes.complex(
        tf.random.stateless_normal(shape=(2, 32, 32, 4), seed=[12, 34]),
        tf.random.stateless_normal(shape=(2, 32, 32, 4), seed=[56, 78]))

    block = conv_endec.UNet2D(
        filters=[4, 8],
        kernel_size=3,
        activation=complex_activations.complex_relu,
        dtype=tf.complex64)

    result = block(inputs)
    self.assertAllClose((2, 32, 32, 4), result.shape)
    self.assertDTypeEqual(result, tf.complex64)


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
        output_filters=1,
        output_kernel_size=1,
        output_activation='relu',
        use_global_residual=True,
        use_dropout=True,
        dropout_rate=0.5,
        dropout_type='spatial',
        use_tight_frame=True,
        use_instance_norm=False,
        use_resize_and_concatenate=False)

    block = conv_endec.UNet2D(**config)
    self.assertEqual(config, block.get_config())

    block2 = conv_endec.UNet2D.from_config(block.get_config())
    self.assertAllEqual(block.get_config(), block2.get_config())

  def test_arch(self):
    """Tests basic model arch."""
    tf.keras.backend.clear_session()

    model = conv_endec.UNet2D(filters=[8, 16], kernel_size=3)
    inputs = tf.keras.Input(shape=(32, 32, 1), batch_size=1)
    model = tf.keras.Model(inputs, model.call(inputs))

    expected = [
        # name, type, output_shape, params
        ('input_1', 'InputLayer', [(1, 32, 32, 1)], 0),
        ('conv_block2d', 'ConvBlock2D', (1, 32, 32, 8), 664),
        ('max_pooling2d', 'MaxPooling2D', (1, 16, 16, 8), 0),
        ('conv_block2d_1', 'ConvBlock2D', (1, 16, 16, 16), 3488),
        ('up_sampling2d', 'UpSampling2D', (1, 32, 32, 16), 0),
        ('conv2d_4', 'Conv2D', (1, 32, 32, 8), 1160),
        ('concatenate', 'Concatenate', (1, 32, 32, 16), 0),
        ('conv_block2d_2', 'ConvBlock2D', (1, 32, 32, 8), 1744)]

    self.assertAllEqual(
        [elem[0] for elem in expected],
        [layer.name for layer in get_layers(model)])

    self.assertAllEqual(
        [elem[1] for elem in expected],
        [layer.__class__.__name__ for layer in get_layers(model)])

    self.assertAllEqual(
        [elem[2] for elem in expected],
        [layer.output_shape for layer in get_layers(model)])

    self.assertAllEqual(
        [elem[3] for elem in expected],
        [layer.count_params() for layer in get_layers(model)])

  def test_arch_with_deconv(self):
    """Tests model arch with deconvolution."""
    tf.keras.backend.clear_session()

    model = conv_endec.UNet2D(filters=[8, 16], kernel_size=3, use_deconv=True)
    inputs = tf.keras.Input(shape=(32, 32, 1), batch_size=1)
    model = tf.keras.Model(inputs, model.call(inputs))

    expected = [
        # name, type, output_shape
        ('input_1', 'InputLayer', [(1, 32, 32, 1)], 0),
        ('conv_block2d', 'ConvBlock2D', (1, 32, 32, 8), 664),
        ('max_pooling2d', 'MaxPooling2D', (1, 16, 16, 8), 0),
        ('conv_block2d_1', 'ConvBlock2D', (1, 16, 16, 16), 3488),
        ('conv2d_transpose', 'Conv2DTranspose', (1, 32, 32, 8), 1160),
        ('concatenate', 'Concatenate', (1, 32, 32, 16), 0),
        ('conv_block2d_2', 'ConvBlock2D', (1, 32, 32, 8), 1744)]

    self.assertAllEqual(
        [elem[0] for elem in expected],
        [layer.name for layer in get_layers(model)])

    self.assertAllEqual(
        [elem[1] for elem in expected],
        [layer.__class__.__name__ for layer in get_layers(model)])

    self.assertAllEqual(
        [elem[2] for elem in expected],
        [layer.output_shape for layer in get_layers(model)])

    self.assertAllEqual(
        [elem[3] for elem in expected],
        [layer.count_params() for layer in get_layers(model)])

  def test_arch_with_out_block(self):
    """Tests model arch with output block."""
    tf.keras.backend.clear_session()

    tf.random.set_seed(32)
    model = conv_endec.UNet2D(filters=[8, 16], kernel_size=3, output_filters=2)
    inputs = tf.keras.Input(shape=(32, 32, 1), batch_size=1)
    model = tf.keras.Model(inputs, model.call(inputs))

    expected = [
        # name, type, output_shape, params
        ('input_1', 'InputLayer', [(1, 32, 32, 1)], 0),
        ('conv_block2d', 'ConvBlock2D', (1, 32, 32, 8), 664),
        ('max_pooling2d', 'MaxPooling2D', (1, 16, 16, 8), 0),
        ('conv_block2d_1', 'ConvBlock2D', (1, 16, 16, 16), 3488),
        ('up_sampling2d', 'UpSampling2D', (1, 32, 32, 16), 0),
        ('conv2d_4', 'Conv2D', (1, 32, 32, 8), 1160),
        ('concatenate', 'Concatenate', (1, 32, 32, 16), 0),
        ('conv_block2d_2', 'ConvBlock2D', (1, 32, 32, 8), 1744),
        ('conv_block2d_3', 'ConvBlock2D', (1, 32, 32, 2), 146)]

    self.assertAllEqual(
        [elem[0] for elem in expected],
        [layer.name for layer in get_layers(model)])

    self.assertAllEqual(
        [elem[1] for elem in expected],
        [layer.__class__.__name__ for layer in get_layers(model)])

    self.assertAllEqual(
        [elem[2] for elem in expected],
        [layer.output_shape for layer in get_layers(model)])

    self.assertAllEqual(
        [elem[3] for elem in expected],
        [layer.count_params() for layer in get_layers(model)])

    out_block = model.layers[-1]
    self.assertLen(out_block.layers, 2)
    self.assertIsInstance(out_block.layers[0], convolutional.Conv2D)
    self.assertIsInstance(out_block.layers[1], tf.keras.layers.Activation)
    self.assertEqual(tf.keras.activations.linear,
                     out_block.layers[1].activation)

    input_data = tf.random.stateless_normal((1, 32, 32, 1), [12, 34])
    output_data = model.predict(input_data)

    # New model with output activation.
    tf.random.set_seed(32)
    model = conv_endec.UNet2D(
        filters=[8, 16], kernel_size=3, output_filters=2,
        output_activation='sigmoid')
    inputs = tf.keras.Input(shape=(32, 32, 1), batch_size=1)
    model = tf.keras.Model(inputs, model.call(inputs))

    self.assertAllClose(tf.keras.activations.sigmoid(output_data),
                        model.predict(input_data))

  def test_arch_lstm(self):
    """Tests LSTM model arch."""
    tf.keras.backend.clear_session()

    model = conv_endec.UNetLSTM2D(filters=[8, 16], kernel_size=3)
    inputs = tf.keras.Input(shape=(4, 32, 32, 1), batch_size=1)
    model = tf.keras.Model(inputs, model.call(inputs))

    expected = [
        # name, type, output_shape, params
        ('input_1', tf.keras.layers.InputLayer, [(1, 4, 32, 32, 1)], 0),
        ('conv_block_lstm2d', conv_blocks.ConvBlockLSTM2D, (1, 4, 32, 32, 8), 7264),
        ('time_distributed', tf.keras.layers.TimeDistributed, (1, 4, 16, 16, 8), 0),
        ('conv_block_lstm2d_1', conv_blocks.ConvBlockLSTM2D, (1, 4, 16, 16, 16), 32384),
        ('time_distributed_1', tf.keras.layers.TimeDistributed, (1, 4, 32, 32, 16), 0),
        ('time_distributed_2', tf.keras.layers.TimeDistributed, (1, 4, 32, 32, 8), 1160),
        ('concatenate', tf.keras.layers.Concatenate, (1, 4, 32, 32, 16), 0),
        ('conv_block_lstm2d_2', conv_blocks.ConvBlockLSTM2D, (1, 4, 32, 32, 8), 11584)]

    self._check_layers(expected, model.layers)

    # Check that TimeDistributed wrappers wrap the right layers.
    self.assertIsInstance(model.layers[2].layer, pooling.MaxPooling2D)
    self.assertIsInstance(model.layers[4].layer, reshaping.UpSampling2D)
    self.assertIsInstance(model.layers[5].layer, convolutional.Conv2D)

  def _check_layers(self, expected, actual):
    actual = [
        (layer.name, type(layer), layer.output_shape, layer.count_params())
        for layer in actual]
    self.assertEqual(expected, actual)


def get_layers(model, recursive=False):
  """Gets all layers in a model (expanding nested models)."""
  layers = []
  for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
      if recursive:
        layers.extend(get_layers(layer, recursive=True))
      else:
        layers.append(layer)
    else:
      layers.append(layer)
  return layers


if __name__ == '__main__':
  tf.test.main()
