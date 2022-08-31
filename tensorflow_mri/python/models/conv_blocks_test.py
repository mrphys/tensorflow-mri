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
from tensorflow_mri.python.layers import convolutional
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
        output_activation='linear',
        use_bias=False,
        kernel_initializer='ones',
        bias_initializer='ones',
        kernel_regularizer='l2',
        bias_regularizer='l1',
        use_batch_norm=True,
        use_sync_bn=True,
        use_instance_norm=False,
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

  def test_arch(self):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(32, 32, 4))
    model = conv_blocks.ConvBlock2D(filters=16, kernel_size=3).functional(inputs)

    expected = [
        # name, type, output_shape, params
        ('input_1', tf.keras.layers.InputLayer, [(None, 32, 32, 4)], 0),
        ('conv2d', convolutional.Conv2D, (None, 32, 32, 16), 592),
        ('activation', tf.keras.layers.Activation, (None, 32, 32, 16), 0)
    ]
    self._check_layers(expected, model.layers)

  def test_multilayer(self):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(32, 32, 4))
    model = conv_blocks.ConvBlock2D(filters=[8, 16], kernel_size=3).functional(inputs)

    expected = [
        # name, type, output_shape, params
        ('input_1', tf.keras.layers.InputLayer, [(None, 32, 32, 4)], 0),
        ('conv2d', convolutional.Conv2D, (None, 32, 32, 8), 296),
        ('activation', tf.keras.layers.Activation, (None, 32, 32, 8), 0),
        ('conv2d_1', convolutional.Conv2D, (None, 32, 32, 16), 1168),
        ('activation_1', tf.keras.layers.Activation, (None, 32, 32, 16), 0)
    ]
    self._check_layers(expected, model.layers)

  def test_arch_activation(self):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(32, 32, 4))
    model = conv_blocks.ConvBlock2D(
        filters=16, kernel_size=3, activation='sigmoid').functional(inputs)

    expected = [
        # name, type, output_shape, params
        ('input_1', tf.keras.layers.InputLayer, [(None, 32, 32, 4)], 0),
        ('conv2d', convolutional.Conv2D, (None, 32, 32, 16), 592),
        ('activation', tf.keras.layers.Activation, (None, 32, 32, 16), 0)
    ]
    self._check_layers(expected, model.layers)

    self.assertEqual(tf.keras.activations.sigmoid, model.layers[-1].activation)

  def test_arch_output_activation(self):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(32, 32, 4))
    model = conv_blocks.ConvBlock2D(
        filters=[8, 16],
        kernel_size=5,
        activation='relu',
        output_activation='tanh').functional(inputs)

    expected = [
        # name, type, output_shape, params
        ('input_1', tf.keras.layers.InputLayer, [(None, 32, 32, 4)], 0),
        ('conv2d', convolutional.Conv2D, (None, 32, 32, 8), 808),
        ('activation', tf.keras.layers.Activation, (None, 32, 32, 8), 0),
        ('conv2d_1', convolutional.Conv2D, (None, 32, 32, 16), 3216),
        ('activation_1', tf.keras.layers.Activation, (None, 32, 32, 16), 0)
    ]
    self._check_layers(expected, model.layers)

    self.assertEqual(tf.keras.activations.relu, model.layers[2].activation)
    self.assertEqual(tf.keras.activations.tanh, model.layers[4].activation)

  def test_arch_batch_norm(self):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(32, 32, 4))
    model = conv_blocks.ConvBlock2D(
        filters=16, kernel_size=3, use_batch_norm=True).functional(inputs)

    expected = [
        # name, type, output_shape, params
        ('input_1', tf.keras.layers.InputLayer, [(None, 32, 32, 4)], 0),
        ('conv2d', convolutional.Conv2D, (None, 32, 32, 16), 592),
        ('batch_normalization', tf.keras.layers.BatchNormalization, (None, 32, 32, 16), 64),
        ('activation', tf.keras.layers.Activation, (None, 32, 32, 16), 0)
    ]
    self._check_layers(expected, model.layers)

  def test_arch_dropout(self):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(32, 32, 4))
    model = conv_blocks.ConvBlock2D(
        filters=16, kernel_size=3, use_dropout=True).functional(inputs)

    expected = [
        # name, type, output_shape, params
        ('input_1', tf.keras.layers.InputLayer, [(None, 32, 32, 4)], 0),
        ('conv2d', convolutional.Conv2D, (None, 32, 32, 16), 592),
        ('activation', tf.keras.layers.Activation, (None, 32, 32, 16), 0),
        ('dropout', tf.keras.layers.Dropout, (None, 32, 32, 16), 0)
    ]
    self._check_layers(expected, model.layers)

  def _check_layers(self, expected, actual):
    actual = [
        (layer.name, type(layer), layer.output_shape, layer.count_params())
        for layer in actual]
    self.assertEqual(expected, actual)

  def test_arch_lstm(self):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(None, 32, 32, 4))
    model = conv_blocks.ConvBlockLSTM2D(
        filters=16, kernel_size=3).functional(inputs)

    expected = [
        # name, type, output_shape, params
        ('input_1', tf.keras.layers.InputLayer, [(None, None, 32, 32, 4)], 0),
        ('conv_lstm2d', tf.keras.layers.ConvLSTM2D, (None, None, 32, 32, 16), 11584),
        ('activation', tf.keras.layers.Activation, (None, None, 32, 32, 16), 0),
    ]
    self._check_layers(expected, model.layers)

    self.assertFalse(model.layers[1].stateful)

  def test_arch_lstm_stateful(self):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(6, 32, 32, 4), batch_size=2)
    model = conv_blocks.ConvBlockLSTM2D(
        filters=16, kernel_size=3, stateful=True).functional(inputs)

    expected = [
        # name, type, output_shape, params
        ('input_1', tf.keras.layers.InputLayer, [(2, 6, 32, 32, 4)], 0),
        ('conv_lstm2d', tf.keras.layers.ConvLSTM2D, (2, 6, 32, 32, 16), 11584),
        ('activation', tf.keras.layers.Activation, (2, 6, 32, 32, 16), 0),
    ]
    self._check_layers(expected, model.layers)

    self.assertTrue(model.layers[1].stateful)

  def test_reset_states(self):
    tf.keras.backend.clear_session()
    model = conv_blocks.ConvBlockLSTM2D(
        filters=16, kernel_size=3, stateful=True)

    input_data = tf.random.stateless_normal((2, 6, 32, 32, 4), [12, 34])

    # Test subclassed model directly.
    _ = model(input_data)
    model.reset_states()

    self.assertAllEqual(tf.zeros_like(model.layers[0].states),
                        model.layers[0].states)
    self.assertTrue(model.layers[0].stateful)

    # Test functional model.
    model = model.functional(tf.keras.Input(shape=(6, 32, 32, 4), batch_size=2))
    _ = model(input_data)
    model.reset_states()

    self.assertAllEqual(tf.zeros_like(model.layers[1].states),
                        model.layers[1].states)
    self.assertTrue(model.layers[1].stateful)


if __name__ == '__main__':
  tf.test.main()
