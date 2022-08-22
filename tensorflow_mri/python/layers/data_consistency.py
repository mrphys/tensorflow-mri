# Copyright 2022 University College London. All Rights Reserved.
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

import tensorflow as tf

from tensorflow_mri.python.layers import linear_operator_layer
from tensorflow_mri.python.linalg import linear_operator_mri
from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import keras_util


@api_util.export("layers.LeastSquaresGradientDescent")
class LeastSquaresGradientDescent(linear_operator_layer.LinearOperatorLayer):
  """Least squares gradient descent layer.
  """
  def __init__(self,
               scale_initializer=1.0,
               ignore_channels=True,
               reinterpret_complex=False,
               operator=linear_operator_mri.LinearOperatorMRI,
               image_index='image',
               kspace_index='kspace',
               **kwargs):
    kwargs['dtype'] = kwargs.get('dtype') or keras_util.complexx()
    super().__init__(operator=operator,
                     input_indices=(image_index, kspace_index),
                     **kwargs)
    if isinstance(scale_initializer, (float, int)):
      self.scale_initializer = tf.keras.initializers.Constant(scale_initializer)
    else:
      self.scale_initializer = tf.keras.initializers.get(scale_initializer)
    self.ignore_channels = ignore_channels
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
    (image, kspace), operator = self.parse_inputs(inputs)
    if self.reinterpret_complex:
      image = math_ops.view_as_complex(image, stacked=False)
    if self.ignore_channels:
      image = tf.squeeze(image, axis=-1)
    image -= tf.cast(self.scale, image.dtype) * operator.transform(
        operator.transform(image) - kspace, adjoint=True)
    if self.ignore_channels:
      image = tf.expand_dims(image, axis=-1)
    if self.reinterpret_complex:
      image = math_ops.view_as_real(image, stacked=False)
    return image

  def get_config(self):
    config = {
        'scale_initializer': tf.keras.initializers.serialize(
            self.scale_initializer),
        'ignore_channels': self.ignore_channels,
        'reinterpret_complex': self.reinterpret_complex
    }
    base_config = super().get_config()
    image_index, kspace_index = base_config.pop('input_indices')
    config['image_index'] = image_index
    config['kspace_index'] = kspace_index
    return {**config, **base_config}
