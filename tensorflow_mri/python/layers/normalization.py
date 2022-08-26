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
"""Normalization layers."""

import tensorflow as tf

from tensorflow_mri.python.util import api_util


@api_util.export("layers.Normalized")
@tf.keras.utils.register_keras_serializable(package='MRI')
class Normalized(tf.keras.layers.Wrapper):
  r"""Applies the wrapped layer with normalized inputs.

  This layer shifts and scales the inputs into a distribution centered around 0
  with a standard deviation of 1 before passing them to the wrapped layer.

  $$
  x = (x - \mu) / \sigma
  $$

  After applying the wrapped layer, the outputs are scaled back to the original
  distribution.

  $$
  y = y \sigma + \mu
  $$

  Args:
    layer: A `tf.keras.layers.Layer`. The wrapped layer.
    axis: An `int` or a `list` thereof. The axis or axes to normalize across.
      Typically this is the features axis/axes. The left-out axes are typically
      the batch axis/axes. Defaults to -1, the last dimension in the input.
    **kwargs: Additional keyword arguments to be passed to the base class.
  """
  def __init__(self, layer, axis=-1, **kwargs):
    super().__init__(layer, **kwargs)
    self.axis = axis

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(input_shape)

  def call(self, inputs, **kwargs):
    mean, variance = tf.nn.moments(inputs, axes=self.axis, keepdims=True)
    std = tf.math.maximum(tf.math.sqrt(variance), tf.keras.backend.epsilon())
    inputs = (inputs - mean) / std
    outputs = self.layer(inputs, **kwargs)
    outputs = outputs * std + mean
    return outputs

  def get_config(self):
    base_config = super().get_config()
    config = {'axis': self.axis}
    return {**base_config, **config}
