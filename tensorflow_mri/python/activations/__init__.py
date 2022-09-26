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
"""Keras activations."""

import keras

from tensorflow_mri.python.activations import complex_activations
from tensorflow_mri.python.util import api_util


TFMRI_ACTIVATIONS = {
    'complex_relu': complex_activations.complex_relu,
    'mod_relu': complex_activations.mod_relu
}


@api_util.export("activations.serialize")
def serialize(activation):
  """Returns the string identifier of an activation function.

  ```{note}
  This function is a drop-in replacement for `tf.keras.activations.serialize`.
  ```

  Example:
    >>> tfmri.activations.serialize(tf.keras.activations.tanh)
    'tanh'
    >>> tfmri.activations.serialize(tf.keras.activations.sigmoid)
    'sigmoid'
    >>> tfmri.activations.serialize(tfmri.activations.complex_relu)
    'complex_relu'
    >>> tfmri.activations.serialize('abcd')
    Traceback (most recent call last):
    ...
    ValueError: ('Cannot serialize', 'abcd')

  Args:
    activation: A function object.

  Returns:
    A `str` denoting the name attribute of the input function.

  Raises:
    ValueError: If the input function is not a valid one.
  """
  return keras.activations.serialize(activation)


@api_util.export("activations.deserialize")
def deserialize(name, custom_objects=None):
  """Returns activation function given a string identifier.

  ```{note}
  This function is a drop-in replacement for
  `tf.keras.activations.deserialize`. The only difference is that this function
  has built-in knowledge of TFMRI activations.
  ```

  Example:
    >>> tfmri.activations.deserialize('linear')
    <function linear at 0x1239596a8>
    >>> tfmri.activations.deserialize('sigmoid')
    <function sigmoid at 0x123959510>
    >>> tfmri.activations.deserialize('complex_relu')
    <function sigmoid at 0x123959510>
    >>> tfmri.activations.deserialize('abcd')
    Traceback (most recent call last):
    ...
    ValueError: Unknown activation function:abcd

  Args:
    name: The name of the activation function.
    custom_objects: Optional `{function_name: function_obj}`
      dictionary listing user-provided activation functions.

  Returns:
    The corresponding activation function.

  Raises:
    ValueError: If the input string does not denote any defined activation
      function.
  """
  custom_objects = {**TFMRI_ACTIVATIONS, **(custom_objects or {})}
  return keras.activations.deserialize(name, custom_objects)


@api_util.export("activations.get")
def get(identifier):
  """Retrieve a Keras activation by its identifier.

  ```{note}
  This function is a drop-in replacement for
  `tf.keras.activations.get`. The only difference is that this function
  has built-in knowledge of TFMRI activations.
  ```

  Args:
    identifier: A function or a string.

  Returns:
    A function corresponding to the input string or input function.

  Example:

    >>> tfmri.activations.get('softmax')
    <function softmax at 0x1222a3d90>
    >>> tfmri.activations.get(tf.keras.activations.softmax)
    <function softmax at 0x1222a3d90>
    >>> tfmri.activations.get(None)
    <function linear at 0x1239596a8>
    >>> tfmri.activations.get(abs)
    <built-in function abs>
    >>> tfmri.activations.get('complex_relu')
    <function complex_relu at 0x123959510>
    >>> tfmri.activations.get('abcd')
    Traceback (most recent call last):
    ...
    ValueError: Unknown activation function:abcd

  Raises:
    ValueError: If the input is an unknown function or string, i.e., the input
      does not denote any defined function.
  """
  if identifier is None:
    return keras.activations.linear
  if isinstance(identifier, (str, dict)):
    return deserialize(identifier)
  if callable(identifier):
    return identifier
  raise ValueError(
      f'Could not interpret activation function identifier: {identifier}')
