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
"""Linear operator layer."""

import inspect

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator


class LinearOperatorLayer(tf.keras.layers.Layer):
  """A layer that uses a linear operator (abstract base class)."""
  def __init__(self, operator, input_indices, **kwargs):
    super().__init__(**kwargs)

    if isinstance(operator, linear_operator.LinearOperator):
      self._operator_class = operator.__class__
      self._operator_instance = operator
    elif (inspect.isclass(operator) and
          issubclass(operator, linear_operator.LinearOperator)):
      self._operator_class = operator
      self._operator_instance = None
    else:
      raise TypeError(
          f"`operator` must be a subclass of `tfmri.linalg.LinearOperator` "
          f"or an instance thereof, but got type: {type(operator)}")

    if isinstance(input_indices, (int, str)):
      input_indices = (input_indices,)
    self._input_indices = input_indices

  def parse_inputs(self, inputs):
    """Parses inputs to the layer.

    This function should typically be called at the beginning of the `call`
    method. It returns the inputs and an instance of the linear operator to be
    used.
    """
    if self._operator_instance is None:
      # operator is a class.
      if not isinstance(inputs, dict):
        raise ValueError(
            f"Layer {self.name} expected a mapping. "
            f"Received: {inputs}")

      if self._input_indices is None:
        input_indices = (tuple(inputs.keys())[0],)
      else:
        input_indices = self._input_indices

      main = tuple(inputs[i] for i in input_indices)
      kwargs = {k: v for k, v in inputs.items() if k not in input_indices}

      # Unpack single input.
      if len(main) == 1:
        main = main[0]

      # Instantiate the operator.
      operator = self._operator_class(**kwargs)

    else:
      # Inputs.
      main = inputs
      operator = self._operator_instance

    return main, operator

  def get_config(self):
    base_config = super().get_config()
    config = {
        'operator': self.get_input_operator(),
        'input_indices': self._input_indices
    }
    return {**config, **base_config}

  def get_input_operator(self):
    """Serializes an operator to a dictionary."""
    if self._operator_instance is None:
      operator = self._operator_class
    else:
      operator = self._operator_instance
    return operator
