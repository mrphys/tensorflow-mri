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

# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Convolutional neural network blocks."""

import itertools

import tensorflow as tf

from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import layer_util


@api_util.export("layers.ConvBlock")
@tf.keras.utils.register_keras_serializable(package='MRI')
class ConvBlock(tf.keras.layers.Layer):
  """A basic convolution block.

  A Conv + BN + Activation block. The number of convolutional layers is
  determined by `filters`. BN and activation are optional.

  Args:
    filters: A list of `int` numbers or an `int` number of filters. Given an
      `int` input, a single convolution is applied; otherwise a series of
      convolutions are applied.
    kernel_size: An integer or tuple/list of `rank` integers, specifying the
      size of the convolution window. Can be a single integer to specify the
      same value for all spatial dimensions.
    strides: An integer or tuple/list of `rank` integers, specifying the strides
      of the convolution along each spatial dimension. Can be a single integer
      to specify the same value for all spatial dimensions.
    rank: An integer specifying the number of spatial dimensions. Defaults to 2.
    activation: A callable or a Keras activation identifier. The activation to
      use in all layers. Defaults to `'relu'`.
    out_activation: A callable or a Keras activation identifier. The activation
      to use in the last layer. Defaults to `'same'`, in which case we use the
      same activation as in previous layers as defined by `activation`.
    use_bias: A `boolean`, whether the block's layers use bias vectors. Defaults
      to `True`.
    kernel_initializer: A `tf.keras.initializers.Initializer` or a Keras
      initializer identifier. Initializer for convolutional kernels. Defaults to
      `'VarianceScaling'`.
    bias_initializer: A `tf.keras.initializers.Initializer` or a Keras
      initializer identifier. Initializer for bias terms. Defaults to `'Zeros'`.
    kernel_regularizer: A `tf.keras.initializers.Regularizer` or a Keras
      regularizer identifier. Regularizer for convolutional kernels. Defaults to
      `None`.
    bias_regularizer: A `tf.keras.initializers.Regularizer` or a Keras
      regularizer identifier. Regularizer for bias terms. Defaults to `None`.
    use_batch_norm: If `True`, use batch normalization. Defaults to `False`.
    use_sync_bn: If `True`, use synchronised batch normalization. Defaults to
      `False`.
    bn_momentum: A `float`. Momentum for the moving average in batch
      normalization.
    bn_epsilon: A `float`. Small float added to variance to avoid dividing by
      zero during batch normalization.
    use_residual: A `boolean`. If `True`, the input is added to the outputs to
      create a residual learning block. Defaults to `False`.
    use_dropout: A `boolean`. If `True`, a dropout layer is inserted after
      each activation. Defaults to `False`.
    dropout_rate: A `float`. The dropout rate. Only relevant if `use_dropout` is
      `True`. Defaults to 0.3.
    dropout_type: A `str`. The dropout type. Must be one of `'standard'` or
      `'spatial'`. Standard dropout drops individual elements from the feature
      maps, whereas spatial dropout drops entire feature maps. Only relevant if
      `use_dropout` is `True`. Defaults to `'standard'`.
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               rank=2,
               activation='relu',
               out_activation='same',
               use_bias=True,
               kernel_initializer='VarianceScaling',
               bias_initializer='Zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               use_batch_norm=False,
               use_sync_bn=False,
               bn_momentum=0.99,
               bn_epsilon=0.001,
               use_residual=False,
               use_dropout=False,
               dropout_rate=0.3,
               dropout_type='standard',
               **kwargs):
    """Create a basic convolution block."""
    super().__init__(**kwargs)

    self._filters = [filters] if isinstance(filters, int) else filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._rank = rank
    self._activation = activation
    self._out_activation = out_activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_batch_norm = use_batch_norm
    self._use_sync_bn = use_sync_bn
    self._bn_momentum = bn_momentum
    self._bn_epsilon = bn_epsilon
    self._use_residual = use_residual
    self._use_dropout = use_dropout
    self._dropout_rate = dropout_rate
    self._dropout_type = check_util.validate_enum(
        dropout_type, {'standard', 'spatial'}, 'dropout_type')
    self._num_layers = len(self._filters)

    conv = layer_util.get_nd_layer('Conv', self._rank)

    if self._use_batch_norm:
      if self._use_sync_bn:
        bn = tf.keras.layers.experimental.SyncBatchNormalization
      else:
        bn = tf.keras.layers.BatchNormalization

    if self._use_dropout:
      if self._dropout_type == 'standard':
        dropout = tf.keras.layers.Dropout
      elif self._dropout_type == 'spatial':
        dropout = layer_util.get_nd_layer('SpatialDropout', self._rank)

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._channel_axis = -1
    else:
      self._channel_axis = 1

    self._convs = []
    self._norms = []
    self._dropouts = []
    for num_filters in self._filters:
      self._convs.append(
          conv(filters=num_filters,
               kernel_size=self._kernel_size,
               strides=self._strides,
               padding='same',
               data_format=None,
               activation=None,
               use_bias=self._use_bias,
               kernel_initializer=self._kernel_initializer,
               bias_initializer=self._bias_initializer,
               kernel_regularizer=self._kernel_regularizer,
               bias_regularizer=self._bias_regularizer))
      if self._use_batch_norm:
        self._norms.append(
            bn(axis=self._channel_axis,
              momentum=self._bn_momentum,
              epsilon=self._bn_epsilon))
      if self._use_dropout:
        self._dropouts.append(dropout(rate=self._dropout_rate))

    self._activation_fn = tf.keras.activations.get(self._activation)
    if self._out_activation == 'same':
      self._out_activation_fn = self._activation_fn
    else:
      self._out_activation_fn = tf.keras.activations.get(self._out_activation)

  def call(self, inputs, training=None): # pylint: disable=unused-argument, missing-param-doc
    """Runs forward pass on the input tensor."""
    x = inputs

    for i, (conv, norm, dropout) in enumerate(
        itertools.zip_longest(self._convs, self._norms, self._dropouts)):
      # Convolution.
      x = conv(x)
      # Batch normalization.
      if self._use_batch_norm:
        x = norm(x, training=training)
      # Activation.
      if i == self._num_layers - 1: # Last layer.
        x = self._out_activation_fn(x)
      else:
        x = self._activation_fn(x)
      # Dropout.
      if self._use_dropout:
        x = dropout(x, training=training)

    # Residual connection.
    if self._use_residual:
      x += inputs
    return x

  def get_config(self):
    """Gets layer configuration."""
    config = {
        'filters': self._filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'rank': self._rank,
        'activation': self._activation,
        'out_activation': self._out_activation,
        'use_bias': self._use_bias,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'use_batch_norm': self._use_batch_norm,
        'use_sync_bn': self._use_sync_bn,
        'bn_momentum': self._bn_momentum,
        'bn_epsilon': self._bn_epsilon,
        'use_residual': self._use_residual,
        'use_dropout': self._use_dropout,
        'dropout_rate': self._dropout_rate,
        'dropout_type': self._dropout_type
    }
    base_config = super().get_config()
    return {**base_config, **config}
