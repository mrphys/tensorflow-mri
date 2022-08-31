# Copyright 2021 The TensorFlow MRI Authors. All Rights Reserved.
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

from tensorflow_mri.python import initializers
from tensorflow_mri.python.layers import concatenate
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import doc_util
from tensorflow_mri.python.util import model_util  # pylint: disable=cyclic-import
from tensorflow_mri.python.util import layer_util


ARGS = string.Template("""
    filters: A `list` of `int`. The number of filters for convolutional layers
      at each scale. The number of scales is inferred as `len(filters)`.
    kernel_size: An `int` or a `list` of ${rank} `int`s, specifying the
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
    bn_momentum: A `float`. Momentum for the moving average in batch
      normalization.
    bn_epsilon: A `float`. Small float added to variance to avoid dividing by
      zero during batch normalization.
    output_filters: An `int`. The number of output channels.
    output_kernel_size: An `int` or a `list` of ${rank} `int`s. The size of the
      convolutional kernel for the output layer. Defaults to `kernel_size`.
    output_activation: A callable or a Keras activation identifier. The output
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
    use_resize_and_concatenate: A `boolean`. If `True`, the upsampled feature
      maps are resized (by cropping) to match the shape of the incoming
      skip connection prior to concatenation. This enables more flexible input
      shapes. Defaults to `True`.
""")


class UNet(tf.keras.Model):
  """${rank}D U-Net model.

  Args:
    ${args}
    **kwargs: Additional keyword arguments to be passed to base class.

  References:
    1. Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net:
       Convolutional networks for biomedical image segmentation. In
       International Conference on Medical image computing and computer-assisted
       intervention (pp. 234-241). Springer, Cham.
    2. Han, Y., & Ye, J. C. (2018). Framing U-Net via deep convolutional
       framelets: Application to sparse-view CT. IEEE transactions on medical
       imaging, 37(6), 1418-1429.
    3. Hauptmann, A., Arridge, S., Lucka, F., Muthurangu, V., & Steeden, J. A.
       (2019). Real-time cardiovascular MR with spatio-temporal artifact
       suppression using deep learning-proof of concept in congenital heart
       disease. Magnetic resonance in medicine, 81(2), 1143-1156.
  """
  def __init__(self,
               rank,
               filters,
               kernel_size,
               pool_size=2,
               block_depth=2,
               use_deconv=False,
               activation='relu',
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
               output_filters=None,
               output_kernel_size=None,
               output_activation=None,
               use_global_residual=False,
               use_dropout=False,
               dropout_rate=0.3,
               dropout_type='standard',
               use_tight_frame=False,
               use_resize_and_concatenate=False,
               **kwargs):
    block_fn = kwargs.pop(
        '_block_fn', model_util.get_nd_model('ConvBlock', rank))
    block_kwargs = kwargs.pop('_block_kwargs', {})
    is_time_distributed = kwargs.pop('_is_time_distributed', False)
    super().__init__(**kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = kernel_size
    self.pool_size = pool_size
    self.block_depth = block_depth
    self.use_deconv = use_deconv
    self.activation = tf.keras.activations.get(activation)
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
    self.output_filters = output_filters
    self.output_kernel_size = output_kernel_size
    self.output_activation = tf.keras.activations.get(output_activation)
    self.use_global_residual = use_global_residual
    self.use_dropout = use_dropout
    self.dropout_rate = dropout_rate
    self.dropout_type = check_util.validate_enum(
        dropout_type, {'standard', 'spatial'}, 'dropout_type')
    self.use_tight_frame = use_tight_frame
    self.use_resize_and_concatenate = use_resize_and_concatenate

    self.scales = len(self.filters)

    # Check inputs are consistent.
    if use_tight_frame and pool_size != 2:
      raise ValueError('pool_size must be 2 if use_tight_frame is True.')

    block_kwargs.update(dict(
        filters=None,  # To be filled for each scale.
        kernel_size=self.kernel_size,
        strides=1,
        activation=self.activation,
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer,
        bias_initializer=self.bias_initializer,
        kernel_regularizer=self.kernel_regularizer,
        bias_regularizer=self.bias_regularizer,
        use_batch_norm=self.use_batch_norm,
        use_sync_bn=self.use_sync_bn,
        use_instance_norm=self.use_instance_norm,
        bn_momentum=self.bn_momentum,
        bn_epsilon=self.bn_epsilon,
        use_dropout=self.use_dropout,
        dropout_rate=self.dropout_rate,
        dropout_type=self.dropout_type,
        dtype=self.dtype))

    # Configure pooling layer.
    if self.use_tight_frame:
      pool_name = 'DWT'
      pool_config = dict(format_dict=False)
    else:
      pool_name = 'MaxPool'
      pool_config = dict(
          pool_size=self.pool_size,
          strides=self.pool_size,
          padding='same',
          dtype=self.dtype)
    pool_fn = layer_util.get_nd_layer(pool_name, self.rank)
    if is_time_distributed:
      pool_fn = wrap_time_distributed(pool_fn)

    # Configure upsampling layer.
    upsamp_config = dict(
        filters=None,  # To be filled for each scale.
        kernel_size=self.kernel_size,
        pool_size=self.pool_size,
        padding='same',
        activation=self.activation,
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer,
        bias_initializer=self.bias_initializer,
        kernel_regularizer=self.kernel_regularizer,
        bias_regularizer=self.bias_regularizer,
        dtype=self.dtype)
    if self.use_deconv:
      # Use transposed convolution for upsampling.
      def upsamp_fn(**config):
        config['strides'] = config.pop('pool_size')
        convt_fn = layer_util.get_nd_layer('ConvTranspose', self.rank)
        if is_time_distributed:
          convt_fn = wrap_time_distributed(convt_fn)
        return convt_fn(**config)
    else:
      # Use upsampling + conv for upsampling.
      def upsamp_fn(**config):
        pool_size = config.pop('pool_size')
        upsamp_fn_ = layer_util.get_nd_layer('UpSampling', rank)
        conv_fn = layer_util.get_nd_layer('Conv', rank)

        if is_time_distributed:
          upsamp_fn_ = wrap_time_distributed(upsamp_fn_)
          conv_fn = wrap_time_distributed(conv_fn)

        upsamp_layer = upsamp_fn_(size=pool_size, dtype=self.dtype)
        conv_layer = conv_fn(**config)
        return (upsamp_layer, conv_layer)

    # Configure concatenation layer.
    if self.use_resize_and_concatenate:
      concat_fn = concatenate.ResizeAndConcatenate
    else:
      concat_fn = tf.keras.layers.Concatenate

    if tf.keras.backend.image_data_format() == 'channels_last':
      self.channel_axis = -1
    else:
      self.channel_axis = 1

    self._enc_blocks = [None] * self.scales
    self._dec_blocks = [None] * (self.scales - 1)
    self._pools = [None] * (self.scales - 1)
    self._upsamps = [None] * (self.scales - 1)
    self._concats = [None] * (self.scales - 1)
    if self.use_tight_frame:
      # For tight frame model, we also need to upsample each of the detail
      # components.
      self._detail_upsamps = [None] * (self.scales - 1)

    # Configure encoder.
    for scale in range(self.scales):
      block_kwargs['filters'] = [filters[scale]] * self.block_depth
      self._enc_blocks[scale] = block_fn(**block_kwargs)

      if scale < len(self.filters) - 1:  # Not the last scale.
        self._pools[scale] = pool_fn(**pool_config)

    # Configure decoder.
    for scale in range(self.scales - 2, -1, -1):
      block_kwargs['filters'] = [filters[scale]] * self.block_depth

      if scale < len(self.filters) - 1:  # Not the last scale.
        # Add upsampling layer.
        upsamp_config['filters'] = filters[scale]
        self._upsamps[scale] = upsamp_fn(**upsamp_config)
        # For tight-frame U-Net only.
        if self.use_tight_frame:
          # Add one upsampling layer for each detail component. There are 1
          # detail components for 1D, 3 detail components for 2D, and 7 detail
          # components for 3D.
          self._detail_upsamps[scale] = [upsamp_fn(**upsamp_config)
                                         for _ in range(2 ** self.rank - 1)]
        # Add concatenation layer.
        self._concats[scale] = concat_fn(axis=self.channel_axis)
        # Add decoding block.
        self._dec_blocks[scale] = block_fn(**block_kwargs)

    # Configure output block.
    if self.output_filters is not None:
      block_kwargs['filters'] = self.output_filters
      if self.output_kernel_size is not None:
        block_kwargs['kernel_size'] = self.output_kernel_size
      # If network is residual, the activation is performed after the residual
      # addition.
      if self.use_global_residual:
        block_kwargs['activation'] = None
      else:
        block_kwargs['activation'] = self.output_activation
      self._out_block = block_fn(**block_kwargs)

    # Configure residual addition, if requested.
    if self.use_global_residual:
      self._add = tf.keras.layers.Add()
      self._out_activation = tf.keras.layers.Activation(self.output_activation)

  def call(self, inputs):  # pylint: disable=missing-param-doc
    """Runs forward pass on the input tensors."""
    x = inputs

    # For skip connections to decoder.
    cache = [None] * (self.scales - 1)
    if self.use_tight_frame:
      detail_cache = [None] * (self.scales - 1)

    # Backbone.
    for scale in range(self.scales - 1):
      cache[scale] = self._enc_blocks[scale](x)
      x = self._pools[scale](cache[scale])
      if self.use_tight_frame:
        # Store details for later concatenation, and continue processing
        # approximation coefficients.
        detail_cache[scale] = x[1:]  # details
        x = x[0]  # approximation

    # Lowest resolution scale.
    x = self._enc_blocks[-1](x)

    # Decoder.
    for scale in range(self.scales - 2, -1, -1):
      # If not using deconv, `self.upsamps[scale]` is a tuple containing two
      # layers (upsampling + conv).
      if self.use_deconv:
        x = self._upsamps[scale](x)
      else:
        x = self._upsamps[scale][0](x)
        x = self._upsamps[scale][1](x)
      concat_inputs = [cache[scale], x]
      if self.use_tight_frame:
        # Upsample detail components too.
        d = [up(d) for d, up in zip(detail_cache[scale],
                                    self._detail_upsamps[scale])]
        # Add to concatenation.
        concat_inputs.extend(d)
      x = self._concats[scale](concat_inputs)
      x = self._dec_blocks[scale](x)

    # Head.
    if self.output_filters is not None:
      x = self._out_block(x)

    # Global residual connection.
    if self.use_global_residual:
      x = self._add([x, inputs])
      if self.output_activation is not None:
        x = self._out_activation(x)

    return x

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.output_filters is not None:
      output_filters = self.output_filters
    else:
      output_filters = self.filters[0]
    return input_shape[:-1].concatenate([output_filters])

  def get_config(self):
    """Returns model configuration for serialization."""
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'pool_size': self.pool_size,
        'block_depth': self.block_depth,
        'use_deconv': self.use_deconv,
        'activation': tf.keras.activations.serialize(self.activation),
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
        'output_filters': self.output_filters,
        'output_kernel_size': self.output_kernel_size,
        'output_activation': tf.keras.activations.serialize(
            self.output_activation),
        'use_global_residual': self.use_global_residual,
        'use_dropout': self.use_dropout,
        'dropout_rate': self.dropout_rate,
        'dropout_type': self.dropout_type,
        'use_tight_frame': self.use_tight_frame,
        'use_resize_and_concatenate': self.use_resize_and_concatenate
    }
    base_config = super().get_config()
    return {**base_config, **config}


class UNetLSTM(UNet):
  """${rank}D LSTM U-Net model.

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
               pool_size=2,
               block_depth=2,
               use_deconv=False,
               activation='relu',
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
               output_filters=None,
               output_kernel_size=None,
               output_activation=None,
               use_global_residual=False,
               use_dropout=False,
               dropout_rate=0.3,
               dropout_type='standard',
               use_tight_frame=False,
               use_resize_and_concatenate=False,
               stateful=False,
               recurrent_regularizer=None,
               **kwargs):
    self.stateful = stateful
    self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
    super().__init__(rank=rank,
                     filters=filters,
                     kernel_size=kernel_size,
                     pool_size=pool_size,
                     block_depth=block_depth,
                     use_deconv=use_deconv,
                     activation=activation,
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
                     output_filters=output_filters,
                     output_kernel_size=output_kernel_size,
                     output_activation=output_activation,
                     use_global_residual=use_global_residual,
                     use_dropout=use_dropout,
                     dropout_rate=dropout_rate,
                     dropout_type=dropout_type,
                     use_tight_frame=use_tight_frame,
                     use_resize_and_concatenate=use_resize_and_concatenate,
                     _block_fn=model_util.get_nd_model('ConvBlockLSTM', rank),
                     _block_kwargs=dict(
                        stateful=self.stateful,
                        recurrent_regularizer=self.recurrent_regularizer),
                     _is_time_distributed=True,
                     **kwargs)

  def get_config(self):
    base_config = super().get_config()
    config = {
        'stateful': self.stateful,
        'recurrent_regularizer': tf.keras.regularizers.serialize(
            self.recurrent_regularizer)
    }
    return {**base_config, **config}


def wrap_time_distributed(fn):
  return lambda *args, **kwargs: (
      tf.keras.layers.TimeDistributed(fn(*args, **kwargs)))


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


@api_util.export("models.UNetLSTM1D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class UNetLSTM1D(UNetLSTM):
  def __init__(self, *args, **kwargs):
    super().__init__(1, *args, **kwargs)


@api_util.export("models.UNetLSTM2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class UNetLSTM2D(UNetLSTM):
  def __init__(self, *args, **kwargs):
    super().__init__(2, *args, **kwargs)


@api_util.export("models.UNetLSTM3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class UNetLSTM3D(UNetLSTM):
  def __init__(self, *args, **kwargs):
    super().__init__(3, *args, **kwargs)


UNet1D.__doc__ = string.Template(UNet.__doc__).substitute(
    rank=1, args=ARGS.substitute(rank=1))
UNet2D.__doc__ = string.Template(UNet.__doc__).substitute(
    rank=2, args=ARGS.substitute(rank=2))
UNet3D.__doc__ = string.Template(UNet.__doc__).substitute(
    rank=3, args=ARGS.substitute(rank=3))


UNet1D.__signature__ = doc_util.get_nd_layer_signature(UNet)
UNet2D.__signature__ = doc_util.get_nd_layer_signature(UNet)
UNet3D.__signature__ = doc_util.get_nd_layer_signature(UNet)


UNetLSTM1D.__doc__ = string.Template(UNetLSTM.__doc__).substitute(
    rank=1, args=ARGS.substitute(rank=1))
UNetLSTM2D.__doc__ = string.Template(UNetLSTM.__doc__).substitute(
    rank=2, args=ARGS.substitute(rank=2))
UNetLSTM3D.__doc__ = string.Template(UNetLSTM.__doc__).substitute(
    rank=3, args=ARGS.substitute(rank=3))


UNetLSTM1D.__signature__ = doc_util.get_nd_layer_signature(UNetLSTM)
UNetLSTM2D.__signature__ = doc_util.get_nd_layer_signature(UNetLSTM)
UNetLSTM3D.__signature__ = doc_util.get_nd_layer_signature(UNetLSTM)
