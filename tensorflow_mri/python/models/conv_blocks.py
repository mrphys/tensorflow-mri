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

import string

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_mri.python import activations
from tensorflow_mri.python import initializers
from tensorflow_mri.python.models import graph_like_network
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import doc_util
from tensorflow_mri.python.util import layer_util


ARGS = string.Template("""
    filters: A `int` or a `list` of `int`. Given an `int` input, a single
      convolution is applied; otherwise a series of convolutions are applied.
    kernel_size: An `int` or `list` of ${rank} `int`s, specifying the
      size of the convolution window. Can be a single integer to specify the
      same value for all spatial dimensions.
    strides: An `int` or a `list` of ${rank} `int`s, specifying the strides
      of the convolution along each spatial dimension. Can be a single integer
      to specify the same value for all spatial dimensions.
    activation: A callable or a Keras activation identifier. The activation to
      use in all layers. Defaults to `'relu'`.
    output_activation: A callable or a Keras activation identifier. The activation
      to use in the last layer. Defaults to `'same'`, in which case we use the
      same activation as in previous layers as defined by `activation`.
    use_bias: A `boolean`, whether the block's layers use bias vectors. Defaults
      to `True`.
    kernel_initializer: A `tf.keras.initializers.Initializer` or a Keras
      initializer identifier. Initializer for convolutional kernels. Defaults to
      `'variance_scaling'`.
    bias_initializer: A `tf.keras.initializers.Initializer` or a Keras
      initializer identifier. Initializer for bias terms. Defaults to `'zeros'`.
    kernel_regularizer: A `tf.keras.initializers.Regularizer` or a Keras
      regularizer identifier. Regularizer for convolutional kernels. Defaults to
      `None`.
    bias_regularizer: A `tf.keras.initializers.Regularizer` or a Keras
      regularizer identifier. Regularizer for bias terms. Defaults to `None`.
    use_batch_norm: If `True`, use batch normalization. Defaults to `False`.
    use_sync_bn: If `True`, use synchronised batch normalization. Defaults to
      `False`.
    use_instance_norm: If `True`, use instance normalization. Defaults to
      `False`.
    bn_momentum: A `float`. Momentum for the moving average in batch
      normalization.
    bn_epsilon: A `float`. Small float added to variance to avoid dividing by
      zero during batch normalization.
    use_residual: A boolean. If `True`, the input is added to the outputs to
      create a residual learning block. Defaults to `False`.
    use_dropout: A boolean. If `True`, a dropout layer is inserted after
      each activation. Defaults to `False`.
    dropout_rate: A `float`. The dropout rate. Only relevant if `use_dropout` is
      `True`. Defaults to 0.3.
    dropout_type: A `str`. The dropout type. Must be one of `'standard'` or
      `'spatial'`. Standard dropout drops individual elements from the feature
      maps, whereas spatial dropout drops entire feature maps. Only relevant if
      `use_dropout` is `True`. Defaults to `'standard'`.
""")


class ConvBlock(graph_like_network.GraphLikeNetwork):
  """${rank}D convolutional block.

  A basic Conv + BN + Activation + Dropout block. The number of convolutional
  layers is determined by the length of `filters`. BN and activation are
  optional.

  Args:
    ${args}
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               activation='relu',
               output_activation='same',
               use_bias=True,
               kernel_initializer='variance_scaling',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               use_batch_norm=False,
               use_sync_bn=False,
               use_instance_norm=False,
               bn_momentum=0.99,
               bn_epsilon=0.001,
               use_residual=False,
               use_dropout=False,
               dropout_rate=0.3,
               dropout_type='standard',
               **kwargs):
    """Create a basic convolution block."""
    conv_fn = kwargs.pop('_conv_fn', layer_util.get_nd_layer('Conv', rank))
    conv_kwargs = kwargs.pop('_conv_kwargs', {})
    super().__init__(**kwargs)
    self.rank = rank
    self.filters = [filters] if isinstance(filters, int) else filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.activation = activations.get(activation)
    if output_activation == 'same':
      self.output_activation = self.activation
    else:
      self.output_activation = activations.get(output_activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.use_batch_norm = use_batch_norm
    self.use_sync_bn = use_sync_bn
    self.use_instance_norm = use_instance_norm
    self.bn_momentum = bn_momentum
    self.bn_epsilon = bn_epsilon
    self.use_residual = use_residual
    self.use_dropout = use_dropout
    self.dropout_rate = dropout_rate
    self.dropout_type = check_util.validate_enum(
        dropout_type, {'standard', 'spatial'}, 'dropout_type')

    if use_batch_norm and use_instance_norm:
      raise ValueError('Cannot use both batch and instance normalization.')

    if self.use_batch_norm:
      if self.use_sync_bn:
        bn = tf.keras.layers.experimental.SyncBatchNormalization
      else:
        bn = tf.keras.layers.BatchNormalization

    if self.use_dropout:
      if self.dropout_type == 'standard':
        dropout = tf.keras.layers.Dropout
      elif self.dropout_type == 'spatial':
        dropout = layer_util.get_nd_layer('SpatialDropout', self.rank)

    if tf.keras.backend.image_data_format() == 'channels_last':
      self.channel_axis = -1
    else:
      self.channel_axis = 1

    conv_kwargs.update(dict(
        filters=None,  # To be filled during loop below.
        kernel_size=self.kernel_size,
        strides=self.strides,
        padding='same',
        data_format=None,
        activation=None,
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer,
        bias_initializer=self.bias_initializer,
        kernel_regularizer=self.kernel_regularizer,
        bias_regularizer=self.bias_regularizer,
        dtype=self.dtype))

    self._levels = len(self.filters)
    self._layers = []
    for level in range(self._levels):
      # Convolution.
      conv_kwargs['filters'] = self.filters[level]
      self._layers.append(conv_fn(**conv_kwargs))
      # Normalization.
      if self.use_batch_norm:
        self._layers.append(
            bn(axis=self.channel_axis,
               momentum=self.bn_momentum,
               epsilon=self.bn_epsilon))
      if self.use_instance_norm:
        self._layers.append(tfa.layers.InstanceNormalization(
            axis=self.channel_axis))
      # Activation.
      if level == self._levels - 1:
        # Last level, and `output_activation` is not the same as `activation`.
        self._layers.append(
            tf.keras.layers.Activation(self.output_activation))
      else:
        self._layers.append(
            tf.keras.layers.Activation(self.activation))
      # Dropout.
      if self.use_dropout:
        self._layers.append(dropout(rate=self.dropout_rate))

    # Residual.
    if self.use_residual:
      self._add = tf.keras.layers.Add()

  def call(self, inputs): # pylint: disable=unused-argument, missing-param-doc
    """Runs forward pass on the input tensor."""
    x = inputs

    for layer in self._layers:
      x = layer(x)

    if self.use_residual:
      x = self._add([x, inputs])

    return x

  def get_config(self):
    """Gets layer configuration."""
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'activation': activations.serialize(self.activation),
        'output_activation': activations.serialize(
            self.output_activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': tf.keras.regularizers.serialize(
            self.kernel_regularizer),
        'bias_regularizer': tf.keras.regularizers.serialize(
            self.bias_regularizer),
        'use_batch_norm': self.use_batch_norm,
        'use_sync_bn': self.use_sync_bn,
        'use_instance_norm': self.use_instance_norm,
        'bn_momentum': self.bn_momentum,
        'bn_epsilon': self.bn_epsilon,
        'use_residual': self.use_residual,
        'use_dropout': self.use_dropout,
        'dropout_rate': self.dropout_rate,
        'dropout_type': self.dropout_type
    }
    base_config = super().get_config()
    return {**base_config, **config}


class ConvBlockLSTM(ConvBlock):
  """${rank}D convolutional LSTM block.

  Args:
    ${args}
    stateful: A boolean. If `True`, the last state for each sample at index `i`
      in a batch will be used as initial state for the sample of index `i` in
      the following batch. Defaults to `False`.
    recurrent_regularizer: A `tf.keras.initializers.Regularizer` or a Keras
      regularizer identifier. The regularizer applied to the recurrent kernel.
      Defaults to `None`.
  """
  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               activation='relu',
               output_activation='same',
               use_bias=True,
               kernel_initializer='variance_scaling',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               use_batch_norm=False,
               use_sync_bn=False,
               use_instance_norm=False,
               bn_momentum=0.99,
               bn_epsilon=0.001,
               use_residual=False,
               use_dropout=False,
               dropout_rate=0.3,
               dropout_type='standard',
               stateful=False,
               recurrent_regularizer=None,
               **kwargs):
    self.stateful = stateful
    self.recurrent_regularizer = tf.keras.regularizers.get(
        recurrent_regularizer)
    super().__init__(rank=rank,
                     filters=filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     activation=activation,
                     output_activation=output_activation,
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     use_batch_norm=use_batch_norm,
                     use_sync_bn=use_sync_bn,
                     use_instance_norm=use_instance_norm,
                     bn_momentum=bn_momentum,
                     bn_epsilon=bn_epsilon,
                     use_residual=use_residual,
                     use_dropout=use_dropout,
                     dropout_rate=dropout_rate,
                     dropout_type=dropout_type,
                     _conv_fn=layer_util.get_nd_layer('ConvLSTM', rank),
                     _conv_kwargs=dict(
                        stateful=self.stateful,
                        recurrent_regularizer=self.recurrent_regularizer,
                        return_sequences=True),
                     **kwargs)

  def get_config(self):
    base_config = super().get_config()
    config = {
        'stateful': self.stateful,
        'recurrent_regularizer': tf.keras.regularizers.serialize(
            self.recurrent_regularizer)
    }
    return {**base_config, **config}


@api_util.export("models.ConvBlock1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class ConvBlock1D(ConvBlock):
  def __init__(self, *args, **kwargs):
    super().__init__(1, *args, **kwargs)


@api_util.export("models.ConvBlock2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class ConvBlock2D(ConvBlock):
  def __init__(self, *args, **kwargs):
    super().__init__(2, *args, **kwargs)


@api_util.export("models.ConvBlock3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class ConvBlock3D(ConvBlock):
  def __init__(self, *args, **kwargs):
    super().__init__(3, *args, **kwargs)


@api_util.export("models.ConvBlockLSTM1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class ConvBlockLSTM1D(ConvBlockLSTM):
  def __init__(self, *args, **kwargs):
    super().__init__(1, *args, **kwargs)


@api_util.export("models.ConvBlockLSTM2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class ConvBlockLSTM2D(ConvBlockLSTM):
  def __init__(self, *args, **kwargs):
    super().__init__(2, *args, **kwargs)


@api_util.export("models.ConvBlockLSTM3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class ConvBlockLSTM3D(ConvBlockLSTM):
  def __init__(self, *args, **kwargs):
    super().__init__(3, *args, **kwargs)


ConvBlock1D.__doc__ = string.Template(ConvBlock.__doc__).substitute(
    rank=1, args=ARGS.substitute(rank=1))
ConvBlock2D.__doc__ = string.Template(ConvBlock.__doc__).substitute(
    rank=2, args=ARGS.substitute(rank=2))
ConvBlock3D.__doc__ = string.Template(ConvBlock.__doc__).substitute(
    rank=3, args=ARGS.substitute(rank=3))


ConvBlock1D.__signature__ = doc_util.get_nd_layer_signature(ConvBlock)
ConvBlock2D.__signature__ = doc_util.get_nd_layer_signature(ConvBlock)
ConvBlock3D.__signature__ = doc_util.get_nd_layer_signature(ConvBlock)


ConvBlockLSTM1D.__doc__ = string.Template(ConvBlockLSTM.__doc__).substitute(
    rank=1, args=ARGS.substitute(rank=1))
ConvBlockLSTM2D.__doc__ = string.Template(ConvBlockLSTM.__doc__).substitute(
    rank=2, args=ARGS.substitute(rank=2))
ConvBlockLSTM3D.__doc__ = string.Template(ConvBlockLSTM.__doc__).substitute(
    rank=3, args=ARGS.substitute(rank=3))


ConvBlockLSTM1D.__signature__ = doc_util.get_nd_layer_signature(ConvBlockLSTM)
ConvBlockLSTM2D.__signature__ = doc_util.get_nd_layer_signature(ConvBlockLSTM)
ConvBlockLSTM3D.__signature__ = doc_util.get_nd_layer_signature(ConvBlockLSTM)
