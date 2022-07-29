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

from tensorflow_mri.python.linalg import linear_operator


class LeastSquaresGradientDescentStep(tf.keras.layers.Layer):

  def __init__(self,
               operator,
               scale_initializer=1.0,
               dtype=None,
               **kwargs):

    if not isinstance(operator, linear_operator.LinearOperator):
      raise TypeError(
          f"operator must be a `tfmri.linalg.LinearOperator` or a subclass "
          f"thereof, but got type: {type(operator)}")
    self.operator = operator
    if isinstance(scale_initializer, (float, int)):
      self.scale_initializer = tf.keras.initializers.Constant(scale_initializer)
    else:
      self.scale_initializer = tf.keras.initializers.get(scale_initializer)
    if dtype is not None:
      if tf.as_dtype(dtype) != self.operator.dtype:
        raise ValueError(
            f"dtype must be the same as the operator's dtype, but got "
            f"dtype: {dtype} and operator's dtype: {self.operator.dtype}")
    else:
      dtype = self.operator.dtype
    super().__init__(dtype=dtype, **kwargs)

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=(),
        dtype=self.dtype.real_dtype,
        initializer=self.scale_initializer,
        trainable=self.trainable,
        constraint=tf.keras.constraints.NonNeg())
    super().build(input_shape)

  def call(self, inputs):
    x, y, args, kwargs = self._parse_inputs(inputs)
    if args or kwargs:
      raise ValueError(
          f"unexpected arguments in call when GradientDescentStep has a "
          f"predefined operator: {args}, {kwargs}")
    operator = self.operator
    return x - self.scale * operator.transform(
        operator.transform(x) - y, adjoint=True)

  def _parse_inputs(self, inputs):
    if isinstance(inputs, dict):
      if 'x' not in inputs or 'y' not in inputs:
        raise ValueError(
            f"inputs dictionary must at least contain the keys 'x' and "
            f"'y', but got keys: {inputs.keys()}")
      x = inputs.pop('x')
      y = inputs.pop('y')
      args, kwargs = (), inputs
    elif isinstance(inputs, tuple):
      if len(inputs) < 2:
        raise ValueError(
            f"inputs tuple must contain at least two elements, "
            f"x and y, but got tuple with length: {len(inputs)}")
      x = inputs[0]
      y = inputs[1]
      args, kwargs = inputs[2:], {}
    else:
      raise TypeError("inputs must be a tuple or a dictionary.")
    return x, y, args, kwargs

  def get_config(self):
    config = {
        'operator': self.operator,
        'scale_initializer': tf.keras.initializers.serialize(self.scale_initializer)
    }
    base_config = super().get_config()
    return {**config, **base_config}

  # @classmethod
  # def from_config(cls, config):
  #   config = config.copy()
  #   operator = deserialize_linear_operator(config.pop('operator'))
  #   return cls(operator, **config)



# def serialize_linear_operator(operator):
#   if isinstance(operator, linear_operator.LinearOperator):
#     return {
#         'class_name': operator.__class__.__name__,
#         'config': operator.parameters
#     }
#   raise TypeError(
#       f"operator must be a `tfmri.linalg.LinearOperator` or a subclass "
#       f"thereof, but got type: {type(operator)}")


# def deserialize_linear_operator(config):
#   if (not isinstance(config, dict) or
#       set(config.keys()) != {'class_name', 'config'}):
#     raise ValueError(
#         f"config must be a dictionary with keys 'class_name' and 'config', "
#         f"but got: {config}")
#   class_name = config['class_name']
#   config = config['config']
#   if class_name == 'LinearOperator':
#     return linear_operator.LinearOperator(**config)
#   raise ValueError(
#       f"unexpected class name in serialized linear operator: {class_name}")
