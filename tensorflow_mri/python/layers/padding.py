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
"""Padding layers."""

import tensorflow as tf


class DivisorPadding(tf.keras.layers.Layer):
  """Divisor padding layer.

  This layer pads the input tensor so that its spatial dimensions are a multiple
  of the specified divisor.

  Args:
    divisor: An `int` or a `tuple` of `int`. The divisor used to compute the
      output shape.
  """
  def __init__(self, rank, divisor=2, **kwargs):
    super().__init__(**kwargs)
    self.rank = rank
    if isinstance(divisor, int):
      self.divisor = (divisor,) * rank
    elif hasattr(divisor, '__len__'):
      if len(divisor) != rank:
        raise ValueError(f'`divisor` should have {rank} elements. '
                         f'Received: {divisor}')
      self.divisor = divisor
    else:
      raise ValueError(f'`divisor` should be either an int or a '
                       f'a tuple of {rank} ints. '
                       f'Received: {divisor}')
    self.input_spec = tf.keras.layers.InputSpec(ndim=rank + 2)

  def call(self, inputs):  # pylint: disable=missing-function-docstring
    static_input_shape = inputs.shape
    static_output_shape = tuple(
        ((s + d - 1) // d) * d if s is not None else None for s, d in zip(
            static_input_shape[1:-1].as_list(), self.divisor))
    static_output_shape = static_input_shape[:1].concatenate(
        static_output_shape).concatenate(static_input_shape[-1:])

    input_shape = tf.shape(inputs)[1:-1]
    output_shape = (((input_shape + self.divisor - 1) // self.divisor) *
                    self.divisor)
    left_paddings = (output_shape - input_shape) // 2
    right_paddings = (output_shape - input_shape + 1) // 2
    paddings = tf.stack([left_paddings, right_paddings], axis=-1)
    paddings = tf.pad(paddings, [[1, 1], [0, 0]])  # pylint: disable=no-value-for-parameter

    return tf.ensure_shape(tf.pad(inputs, paddings), static_output_shape)  # pylint: disable=no-value-for-parameter

  def get_config(self):
    config = {'divisor': self.divisor}
    base_config = super().get_config()
    return {**config, **base_config}


@tf.keras.utils.register_keras_serializable(package='MRI')
class DivisorPadding1D(DivisorPadding):
  def __init__(self, *args, **kwargs):
    super().__init__(1, *args, **kwargs)


@tf.keras.utils.register_keras_serializable(package='MRI')
class DivisorPadding2D(DivisorPadding):
  def __init__(self, *args, **kwargs):
    super().__init__(2, *args, **kwargs)


@tf.keras.utils.register_keras_serializable(package='MRI')
class DivisorPadding3D(DivisorPadding):
  def __init__(self, *args, **kwargs):
    super().__init__(3, *args, **kwargs)
