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
"""Tests for module `conv_endec_LSTM`."""
# pylint: disable=missing-param-doc

from absl.testing import parameterized
import tensorflow as tf
# import sys
# sys.path.insert(0,'/workspaces/tensorflow-mri/')
from tensorflow_mri.python.activations import complex_activations
from tensorflow_mri.python.models import conv_endec_LSTM
from tensorflow_mri.python.util import test_util


class LSTMUNetTest(test_util.TestCase):
  """U-Net tests."""
  # pylint disable=missing-param-doc
  @test_util.run_all_execution_modes
  @parameterized.named_parameters(
      ("test0", 2, [16, 32], 3, None, True, False),
      ("test1", 2, [4, 8], 3, None, False, False),
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
        shape=(12,)+(128,) * rank + (16,), batch_size=1)

    layer = conv_endec_LSTM.LSTMUNet2D
    network = layer(
        filters=filters,
        kernel_size=kernel_size,
        use_deconv=use_deconv,
        out_channels=out_channels,
        use_global_residual=use_global_residual)

    features = network(inputs)
    if out_channels is None:
      out_channels = filters[0]

    self.assertAllEqual(features.shape, [1] + [12] + [128] * rank + [out_channels])


  @test_util.run_all_execution_modes
  @parameterized.named_parameters(
      ("true", True),
      ("false", False))
  def test_use_bias(self, use_bias):
    """Test `use_bias` argument."""
    model = conv_endec_LSTM.LSTMUNet2D(
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


  # def test_complex_valued(self):
  #   inputs = tf.dtypes.complex(
  #       tf.random.stateless_normal(shape=(2, 12, 32, 32, 4), seed=[12, 34]),
  #       tf.random.stateless_normal(shape=(2, 12, 32, 32, 4), seed=[56, 78]))

  #   block = conv_endec_LSTM.LSTMUNet2D(
  #       filters=[4, 8],
  #       kernel_size=3,
  #       activation=complex_activations.complex_relu,
  #       dtype=tf.complex64)

  #   result = block(inputs)
  #   self.assertAllClose((2, 12,32, 32, 4), result.shape)
  #   self.assertDTypeEqual(result, tf.complex64)


  def test_serialize_deserialize(self):
    """Test de/serialization."""
    config = dict(
        filters=[16, 32],
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
        stateful=False,
        return_sequences=True,
        use_tight_frame=True,
        use_resize_and_concatenate=True)

    block = conv_endec_LSTM.LSTMUNet2D(**config)
    self.assertEqual(block.get_config(), config)

    block2 = conv_endec_LSTM.LSTMUNet2D.from_config(block.get_config())
    self.assertAllEqual(block.get_config(), block2.get_config())


if __name__ == '__main__':
  tf.test.main()

# # Copyright 2021 University College London. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """Tests for module `conv_endec_LSTM`."""
# # pylint: disable=missing-param-doc
# import sys
# sys.path.insert(0,'/workspaces/tensorflow-mri/')
# from tensorflow_mri.python.models import conv_endec_LSTM
# import tensorflow as tf 
# print(tf.__version__)
# config = dict(
#         filters=[16, 32],
#         kernel_size=3,
#         pool_size=2,
#         block_depth=1,
#         use_deconv=True,
#         activation='tf.keras.layers.LeakyReLU(alpha=0.01)',
#         use_bias=True,
#         kernel_initializer='ones',
#         bias_initializer='he_uniform',
#         recurrent_regularizer='tf.keras.regularizers.L2(0.0001)',
#         #kernel_regularizer='l2',
#         #bias_regularizer='l1',
#         use_batch_norm=False,
#         use_sync_bn=False,
#         bn_momentum=0.98,
#         bn_epsilon=0.002,
#         out_channels=1,
#         out_kernel_size=3,
#         out_activation='relu',
#         #use_global_residual=True,
#         use_dropout=False,
#         dropout_rate=0.5,
#         dropout_type='spatial',
#         use_tight_frame=False)

# block = conv_endec_LSTM.LSTMUNet2D(**config)
# block2 = conv_endec_LSTM.LSTMUNet2D.from_config(block.get_config())

# block.build(input_shape=(1,12,240,240,1))
# block.summary()
# block2.build(input_shape=(1,12,240,240,1))
# block2.summary()
# import tensorflow as tf
# b=block.call(tf.ones((1,12,240,240,1)))