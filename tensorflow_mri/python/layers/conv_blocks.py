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

import tensorflow as tf

from tensorflow_mri.python.util import layer_util


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
    rank: An integer specifying the number of spatial dimensions.
    activation: A callable or a Keras activation identifier. Defaults to
      `'relu'`.
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
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               rank=2,
               activation='relu',
               kernel_initializer='VarianceScaling',
               bias_initializer='Zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               use_batch_norm=False,
               use_sync_bn=False,
               bn_momentum=0.99,
               bn_epsilon=0.001,
               **kwargs):
    """Create a basic convolution block."""
    super().__init__(**kwargs)

    self._filters = [filters] if isinstance(filters, int) else filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._rank = rank
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_batch_norm = use_batch_norm
    self._use_sync_bn = use_sync_bn
    self._bn_momentum = bn_momentum
    self._bn_epsilon = bn_epsilon

    conv = layer_util.get_nd_layer('Conv', self._rank)
    if use_sync_bn:
      bn = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._channel_axis = -1
    else:
      self._channel_axis = 1

    self._convs = []
    self._norms = []
    for num_filters in self._filters:
      self._convs.append(
          conv(filters=num_filters,
               kernel_size=self._kernel_size,
               strides=self._strides,
               padding='same',
               data_format=None,
               activation=None,
               kernel_initializer=self._kernel_initializer,
               bias_initializer=self._bias_initializer,
               kernel_regularizer=self._kernel_regularizer,
               bias_regularizer=self._bias_regularizer))
      self._norms.append(
          bn(axis=self._channel_axis,
             momentum=self._bn_momentum,
             epsilon=self._bn_epsilon))
      self._activation_fn = tf.keras.activations.get(activation)

  def call(self, inputs, training=None): # pylint: disable=unused-argument
    """Runs forward pass on the input tensor."""
    x = inputs
    for conv, norm in zip(self._convs, self._norms):
      x = conv(x)
      if self._use_batch_norm:
        x = norm(x)
      x = self._activation_fn(x)
    return x

  def get_config(self):
    """Gets layer configuration."""
    config = {
        'filters': self._filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'rank': self._rank,
        'activation': self._activation,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'use_batch_norm': self._use_batch_norm,
        'use_sync_bn': self._use_sync_bn,
        'bn_momentum': self._bn_momentum,
        'bn_epsilon': self._bn_epsilon
    }
    base_config = super().get_config()
    return {**base_config, **config}
