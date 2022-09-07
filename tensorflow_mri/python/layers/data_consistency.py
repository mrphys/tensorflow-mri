# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Data consistency layers."""

import string

import tensorflow as tf

from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import doc_util


class LeastSquaresGradientDescent(tf.keras.layers.Layer):
  """Least squares gradient descent layer.
  """
  def __init__(self,
               rank,
               scale_initializer=1.0,
               expand_channel_dim=False,
               reinterpret_complex=False,
               **kwargs):
    super().__init__(**kwargs)
    self.rank = rank
    if isinstance(scale_initializer, (float, int)):
      self.scale_initializer = tf.keras.initializers.Constant(scale_initializer)
    else:
      self.scale_initializer = tf.keras.initializers.get(scale_initializer)
    self.expand_channel_dim = expand_channel_dim
    self.reinterpret_complex = reinterpret_complex

  def build(self, input_shape):
    super().build(input_shape)
    self.scale = self.add_weight(
        name='scale',
        shape=(),
        dtype=tf.as_dtype(self.dtype).real_dtype,
        initializer=self.scale_initializer,
        trainable=self.trainable,
        constraint=tf.keras.constraints.NonNeg())

  def call(self, inputs):
    image, data, operator = parse_inputs(inputs)
    if self.reinterpret_complex:
      image = math_ops.view_as_complex(image, stacked=False)
    if self.expand_channel_dim:
      image = tf.squeeze(image, axis=-1)
    image -= tf.cast(self.scale, image.dtype) * operator.transform(
        operator.transform(image) - data, adjoint=True)
    if self.expand_channel_dim:
      image = tf.expand_dims(image, axis=-1)
    if self.reinterpret_complex:
      image = math_ops.view_as_real(image, stacked=False)
    return image

  def get_config(self):
    base_config = super().get_config()
    config = {
        'scale_initializer': tf.keras.initializers.serialize(
            self.scale_initializer),
        'expand_channel_dim': self.expand_channel_dim,
        'reinterpret_complex': self.reinterpret_complex
    }
    return {**base_config, **config}


def parse_inputs(inputs):
  def _parse_inputs(image, data, operator):
    return image, data, operator
  if isinstance(inputs, tuple):
    return _parse_inputs(*inputs)
  elif isinstance(inputs, dict):
    return _parse_inputs(**inputs)
  raise ValueError('inputs must be a tuple or dict')


@api_util.export("layers.LeastSquaresGradientDescent2D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class LeastSquaresGradientDescent2D(LeastSquaresGradientDescent):
  def __init__(self, *args, **kwargs):
    super().__init__(2, *args, **kwargs)


@api_util.export("layers.LeastSquaresGradientDescent3D")
@tf.keras.utils.register_keras_serializable(package='MRI')
class LeastSquaresGradientDescent3D(LeastSquaresGradientDescent):
  def __init__(self, *args, **kwargs):
    super().__init__(3, *args, **kwargs)


LeastSquaresGradientDescent2D.__doc__ = string.Template(
    LeastSquaresGradientDescent.__doc__).safe_substitute(rank=2)
LeastSquaresGradientDescent3D.__doc__ = string.Template(
    LeastSquaresGradientDescent.__doc__).safe_substitute(rank=3)


LeastSquaresGradientDescent2D.__signature__ = doc_util.get_nd_layer_signature(
    LeastSquaresGradientDescent)
LeastSquaresGradientDescent3D.__signature__ = doc_util.get_nd_layer_signature(
    LeastSquaresGradientDescent)
