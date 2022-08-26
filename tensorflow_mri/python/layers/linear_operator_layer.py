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

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.linalg import linear_operator_mri


LINEAR_OPERATORS = {
    'MRI': linear_operator_mri.LinearOperatorMRI,
    'LinearOperatorMRI': linear_operator_mri.LinearOperatorMRI
}


class LinearOperatorLayer(tf.keras.layers.Layer):
  """A layer that uses a linear operator (abstract base class)."""
  def __init__(self, operator, input_indices=None, **kwargs):
    super().__init__(**kwargs)

    if isinstance(operator, linear_operator.LinearOperator):
      self._operator = operator
    elif isinstance(operator, str):
      if operator not in LINEAR_OPERATORS:
        raise ValueError(
            f"Unknown operator: {operator}. "
            f"Valid strings are: {list(LINEAR_OPERATORS.keys())}")
      self._operator = operator
    elif callable(operator):
      self._operator = operator
    else:
      raise TypeError(
          f"`operator` must be a `tfmri.linalg.LinearOperator`, a `str`, or a "
          f"callable object. Received: {operator}")

    if isinstance(input_indices, (int, str)):
      input_indices = (input_indices,)
    self._input_indices = input_indices

  def parse_inputs(self, inputs):
    """Parses inputs to the layer.

    This function should typically be called at the beginning of the `call`
    method. It returns the inputs and an instance of the linear operator to be
    used.
    """
    if isinstance(self._operator, linear_operator.LinearOperator):
      # Operator already instantiated. Simply return.
      return inputs, self._operator

    # Need to instantiate the operator.
    if not isinstance(inputs, dict):
      raise ValueError(
          f"Layer {self.name} expected a mapping. "
          f"Received: {inputs}")

    # If operator is a string, get corresponding class.
    if isinstance(self._operator, str):
      operator = LINEAR_OPERATORS[self._operator]

    # Get main inputs (defined by input_indices).
    if self._input_indices is None:
      input_indices = (tuple(inputs.keys())[0],)
    else:
      input_indices = self._input_indices
    main = tuple(inputs[i] for i in input_indices)
    if len(main) == 1:
      main = main[0]  # Unpack single inputs.

    # Get remaining inputs and instantiate the operator.
    kwargs = {k: v for k, v in inputs.items() if k not in input_indices}
    operator = operator(**kwargs)

    return main, operator

  def get_config(self):
    base_config = super().get_config()
    config = {
        'operator': self._operator,
        'input_indices': self._input_indices
    }
    return {**config, **base_config}


class LinearTransform(LinearOperatorLayer):
  """A layer that applies a linear transform to its inputs."""
  def __init__(self,
               adjoint=False,
               operator=linear_operator_mri.LinearOperatorMRI,
               input_indices=None,
               **kwargs):
    super().__init__(operator=operator, input_indices=input_indices, **kwargs)
    self.adjoint = adjoint

  def call(self, inputs):
    main, operator = self.parse_inputs(inputs)
    return operator.transform(main, adjoint=self.adjoint)
