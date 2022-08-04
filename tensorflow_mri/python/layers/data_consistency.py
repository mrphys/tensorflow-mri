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

import inspect

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import api_util


@api_util.export("layers.LeastSquaresGradientDescent")
class LeastSquaresGradientDescent(tf.keras.layers.Layer):
  """Least squares gradient descent layer.
  """
  def __init__(self,
               operator,
               scale_initializer=1.0,
               handle_channel_axis=True,
               dtype=None,
               **kwargs):
    if isinstance(operator, linear_operator.LinearOperator):
      # operator is a class instance.
      self.operator = operator
      self._operator_is_class = False
      self._operator_is_instance = True
    elif (inspect.isclass(operator) and
          issubclass(operator, linear_operator.LinearOperator)):
      # operator is a class.
      self.operator = operator
      self._operator_is_class = True
      self._operator_is_instance = False
    else:
      raise TypeError(
          f"operator must be a subclass of `tfmri.linalg.LinearOperator` "
          f"or an instance thereof, but got type: {type(operator)}")

    if isinstance(scale_initializer, (float, int)):
      self.scale_initializer = tf.keras.initializers.Constant(scale_initializer)
    else:
      self.scale_initializer = tf.keras.initializers.get(scale_initializer)

    if self._operator_is_instance:
      if dtype is not None:
        if tf.as_dtype(dtype) != self.operator.dtype:
          raise ValueError(
              f"dtype must be the same as the operator's dtype, but got "
              f"dtype: {dtype} and operator's dtype: {self.operator.dtype}")
      else:
        dtype = self.operator.dtype

    self.handle_channel_axis = handle_channel_axis

    super().__init__(dtype=dtype, **kwargs)

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=(),
        dtype=tf.as_dtype(self.dtype).real_dtype,
        initializer=self.scale_initializer,
        trainable=self.trainable,
        constraint=tf.keras.constraints.NonNeg())
    super().build(input_shape)

  def call(self, inputs):
    x, b, args, kwargs = self._parse_inputs(inputs)
    if self._operator_is_class:
      # operator is a class. Instantiate using any additional arguments.
      operator = self.operator(*args, **kwargs)
    else:
      # operator is an instance, so we can use it directly.
      if args or kwargs:
        raise ValueError(
            f"unexpected arguments in call when linear operator is a class "
            f"instance: {args}, {kwargs}")
      operator = self.operator
    if self.handle_channel_axis:
      x = tf.squeeze(x, axis=-1)
    print(x.shape, operator.domain_shape, operator.range_shape)
    x -= tf.cast(self.scale, self.dtype) * operator.transform(
        operator.transform(x) - b, adjoint=True)
    if self.handle_channel_axis:
      x = tf.expand_dims(x, axis=-1)
    return x

  def _parse_inputs(self, inputs):
    """Parses the inputs to the call method."""
    if isinstance(inputs, dict):
      if 'x' not in inputs or 'b' not in inputs:
        raise ValueError(
            f"inputs dictionary must at least contain the keys 'x' and "
            f"'b', but got keys: {inputs.keys()}")
      x = inputs['x']
      b = inputs['b']
      args, kwargs = (), {k: v for k, v in inputs.items()
                          if k not in {'x', 'b'}}
    elif isinstance(inputs, tuple):
      if len(inputs) < 2:
        raise ValueError(
            f"inputs tuple must contain at least two elements, "
            f"x and b, but got tuple with length: {len(inputs)}")
      x = inputs[0]
      b = inputs[1]
      args, kwargs = inputs[2:], {}
    else:
      raise TypeError("inputs must be a tuple or a dictionary.")
    return x, b, args, kwargs

  def get_config(self):
    config = {
        'operator': self.operator,
        'scale_initializer': tf.keras.initializers.serialize(
            self.scale_initializer)
    }
    base_config = super().get_config()
    return {**config, **base_config}
