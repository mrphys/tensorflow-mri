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
"""Convolutional encoder-decoder models."""

import string

import tensorflow as tf

from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import model_util  # pylint: disable=cyclic-import
from tensorflow_mri.python.util import layer_util


UNET_DOC_TEMPLATE = string.Template(
  """${rank}D U-Net model.

  Args:
    filters: A `list` of `int`. The number of filters for convolutional layers
      at each scale. The number of scales is inferred as `len(filters)`.
    kernel_size: An integer or tuple/list of ${rank} integers, specifying the
      size of the convolution window. Can be a single integer to specify the
      same value for all spatial dimensions.
    pool_size: The pooling size for the pooling operations. Defaults to 2.
    block_depth: The number of layers in each convolutional block. Defaults to
      2.
    use_deconv: If `True`, transpose convolution (deconvolution) will be used
      instead of up-sampling. This increases the amount memory required during
      training. Defaults to `False`.
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
    out_channels: An `int`. The number of output channels.
    out_kernel_size: An `int` or a list of ${rank} `int`. The size of the
      convolutional kernel for the output layer. Defaults to `kernel_size`.
    out_activation: A callable or a Keras activation identifier. The output
      activation. Defaults to `None`.
    use_global_residual: A `boolean`. If `True`, adds a global residual
      connection to create a residual learning network. Defaults to `False`.
    use_dropout: A `boolean`. If `True`, a dropout layer is inserted after
      each activation. Defaults to `False`.
    dropout_rate: A `float`. The dropout rate. Only relevant if `use_dropout` is
      `True`. Defaults to 0.3.
    dropout_type: A `str`. The dropout type. Must be one of `'standard'` or
      `'spatial'`. Standard dropout drops individual elements from the feature
      maps, whereas spatial dropout drops entire feature maps. Only relevant if
      `use_dropout` is `True`. Defaults to `'standard'`.
    use_tight_frame: A `boolean`. If `True`, creates a tight frame U-Net as
      described in [2]. Defaults to `False`.
    **kwargs: Additional keyword arguments to be passed to base class.

  References:
    .. [1] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net:
      Convolutional networks for biomedical image segmentation. In International
      Conference on Medical image computing and computer-assisted intervention
      (pp. 234-241). Springer, Cham.
    .. [2] Han, Y., & Ye, J. C. (2018). Framing U-Net via deep convolutional
      framelets: Application to sparse-view CT. IEEE transactions on medical
      imaging, 37(6), 1418-1429.
  """)


class UNet(tf.keras.Model):
  """U-Net model (private base class)."""
  def __init__(self,
               rank,
               filters,
               kernel_size,
               pool_size=2,
               block_depth=2,
               use_deconv=False,
               activation='relu',
               use_bias=True,
               kernel_initializer='VarianceScaling',
               bias_initializer='Zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               use_batch_norm=False,
               use_sync_bn=False,
               bn_momentum=0.99,
               bn_epsilon=0.001,
               out_channels=None,
               out_kernel_size=None,
               out_activation=None,
               use_global_residual=False,
               use_dropout=False,
               dropout_rate=0.3,
               dropout_type='standard',
               use_tight_frame=False,
               **kwargs):
    """Creates a UNet model."""
    super().__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._pool_size = pool_size
    self._rank = rank
    self._block_depth = block_depth
    self._use_deconv = use_deconv
    self._activation = activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_batch_norm = use_batch_norm
    self._use_sync_bn = use_sync_bn
    self._bn_momentum = bn_momentum
    self._bn_epsilon = bn_epsilon
    self._out_channels = out_channels
    self._out_kernel_size = out_kernel_size
    self._out_activation = out_activation
    self._use_global_residual = use_global_residual
    self._use_dropout = use_dropout
    self._dropout_rate = dropout_rate
    self._dropout_type = check_util.validate_enum(
        dropout_type, {'standard', 'spatial'}, 'dropout_type')
    self._use_tight_frame = use_tight_frame
    self._dwt_kwargs = {}
    self._dwt_kwargs['format_dict'] = False
    self._scales = len(filters)

    # Check inputs are consistent.
    if use_tight_frame and pool_size != 2:
      raise ValueError('pool_size must be 2 if use_tight_frame is True.')

    block_layer = model_util.get_nd_model('ConvBlock', self._rank)
    block_config = dict(
        filters=None,  # To be filled for each scale.
        kernel_size=self._kernel_size,
        strides=1,
        activation=self._activation,
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        use_batch_norm=self._use_batch_norm,
        use_sync_bn=self._use_sync_bn,
        bn_momentum=self._bn_momentum,
        bn_epsilon=self._bn_epsilon,
        use_dropout=self._use_dropout,
        dropout_rate=self._dropout_rate,
        dropout_type=self._dropout_type)

    # Configure pooling layer.
    if self._use_tight_frame:
      pool_name = 'DWT'
      pool_config = self._dwt_kwargs
    else:
      pool_name = 'MaxPool'
      pool_config = dict(
          pool_size=self._pool_size,
          strides=self._pool_size,
          padding='same')
    pool_layer = layer_util.get_nd_layer(pool_name, self._rank)

    # Configure upsampling layer.
    if self._use_deconv:
      upsamp_name = 'ConvTranspose'
      upsamp_config = dict(
          filters=None,  # To be filled for each scale.
          kernel_size=self._kernel_size,
          strides=self._pool_size,
          padding='same',
          activation=None,
          use_bias=self._use_bias,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      upsamp_name = 'UpSampling'
      upsamp_config = dict(
          size=self._pool_size)
    upsamp_layer = layer_util.get_nd_layer(upsamp_name, self._rank)

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._channel_axis = -1
    else:
      self._channel_axis = 1

    self._enc_blocks = []
    self._dec_blocks = []
    self._pools = []
    self._upsamps = []
    self._concats = []
    if self._use_tight_frame:
      # For tight frame model, we also need to upsample each of the detail
      # components.
      self._detail_upsamps = []

    # Configure backbone and decoder.
    for scale, filt in enumerate(self._filters):
      block_config['filters'] = [filt] * self._block_depth
      self._enc_blocks.append(block_layer(**block_config))

      if scale < len(self._filters) - 1:
        self._pools.append(pool_layer(**pool_config))
        if use_deconv:
          upsamp_config['filters'] = filt
        self._upsamps.append(upsamp_layer(**upsamp_config))
        if self._use_tight_frame:
          # Add one upsampling layer for each detail component. There are 1
          # detail components for 1D, 3 detail components for 2D, and 7 detail
          # components for 3D.
          self._detail_upsamps.append([upsamp_layer(**upsamp_config)
                                       for _ in range(2 ** self._rank - 1)])
        self._concats.append(
            tf.keras.layers.Concatenate(axis=self._channel_axis))
        self._dec_blocks.append(block_layer(**block_config))

    # Configure output block.
    if self._out_channels is not None:
      block_config['filters'] = self._out_channels
    if self._out_kernel_size is not None:
      block_config['kernel_size'] = self._out_kernel_size
    # If network is residual, the activation is performed after the residual
    # addition.
    if self._use_global_residual:
      block_config['activation'] = None
    else:
      block_config['activation'] = self._out_activation
    self._out_block = block_layer(**block_config)

    # Configure residual addition, if requested.
    if self._use_global_residual:
      self._add = tf.keras.layers.Add()
      self._out_activation_fn = tf.keras.activations.get(self._out_activation)

  def call(self, inputs, training=None): # pylint: disable=missing-param-doc,unused-argument
    """Runs forward pass on the input tensors."""
    x = inputs

    # For skip connections to decoder.
    cache = [None] * (self._scales - 1)
    if self._use_tight_frame:
      detail_cache = [None] * (self._scales - 1)

    # Backbone.
    for scale in range(self._scales - 1):
      cache[scale] = self._enc_blocks[scale](x)
      x = self._pools[scale](cache[scale])
      if self._use_tight_frame:
        # Store details for later concatenation, and continue processing
        # approximation coefficients.
        detail_cache[scale] = x[1:]  # details
        x = x[0]  # approximation

    # Lowest resolution scale.
    x = self._enc_blocks[-1](x)

    # Decoder.
    for scale in range(self._scales - 2, -1, -1):
      x = self._upsamps[scale](x)
      concat_inputs = [x, cache[scale]]
      if self._use_tight_frame:
        # Upsample detail components too.
        d = [up(d) for d, up in zip(
            detail_cache[scale], self._detail_upsamps[scale])]
        # Add to concatenation.
        concat_inputs.extend(d)
      x = self._concats[scale](concat_inputs)
      x = self._dec_blocks[scale](x)

    # Head.
    if self._out_channels is not None:
      x = self._out_block(x)

    # Global residual connection.
    if self._use_global_residual:
      x = self._add([x, inputs])
      if self._out_activation is not None:
        x = self._out_activation_fn(x)

    return x

  def get_config(self):
    """Returns model configuration for serialization."""
    config = {
        'filters': self._filters,
        'kernel_size': self._kernel_size,
        'pool_size': self._pool_size,
        'block_depth': self._block_depth,
        'use_deconv': self._use_deconv,
        'activation': self._activation,
        'use_bias': self._use_bias,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'use_batch_norm': self._use_batch_norm,
        'use_sync_bn': self._use_sync_bn,
        'bn_momentum': self._bn_momentum,
        'bn_epsilon': self._bn_epsilon,
        'out_channels': self._out_channels,
        'out_kernel_size': self._out_kernel_size,
        'out_activation': self._out_activation,
        'use_global_residual': self._use_global_residual,
        'use_dropout': self._use_dropout,
        'dropout_rate': self._dropout_rate,
        'dropout_type': self._dropout_type,
        'use_tight_frame': self._use_tight_frame
    }
    base_config = super().get_config()
    return {**base_config, **config}

  @classmethod
  def from_config(cls, config):
    if 'base_filters' in config:
      # Old config format. Convert to new format.
      config['filters'] = [config.pop('base_filters') * (2 ** scale)
                           for scale in config.pop('scales')]
    return super().from_config(config)


@api_util.export("models.UNet1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class UNet1D(UNet):
  def __init__(self, *args, **kwargs):
    super().__init__(1, *args, **kwargs)


@api_util.export("models.UNet2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class UNet2D(UNet):
  def __init__(self, *args, **kwargs):
    super().__init__(2, *args, **kwargs)


@api_util.export("models.UNet3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class UNet3D(UNet):
  def __init__(self, *args, **kwargs):
    super().__init__(3, *args, **kwargs)


UNet1D.__doc__ = UNET_DOC_TEMPLATE.substitute(rank=1)
UNet2D.__doc__ = UNET_DOC_TEMPLATE.substitute(rank=2)
UNet3D.__doc__ = UNET_DOC_TEMPLATE.substitute(rank=3)
