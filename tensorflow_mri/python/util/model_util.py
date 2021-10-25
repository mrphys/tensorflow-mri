# Copyright 2021 University College London. All Rights Reserved.
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
"""Layer utilities."""

import tensorflow as tf


def model_from_layers(layers, input_spec):
  """Create a Keras model from the given layers and input specification.

  Args:
    layers: A `tf.keras.layers.Layer` or a list thereof.
    input_spec: A nested structure of `tf.TensorSpec` objects.

  Returns:
    A `tf.keras.Model`.
  """
  if isinstance(layers, tf.keras.layers.Layer):
    layers = [layers]

  # Generate inputs with the passed specification.
  def _make_input(spec):

    if spec.shape == None:
      return tf.keras.Input(shape=None, batch_size=None, dtype=spec.dtype)

    return tf.keras.Input(shape=spec.shape[1:],
                          batch_size=spec.shape[0],
                          dtype=spec.dtype)

  inputs = tf.nest.map_structure(_make_input, input_spec)

  # Forward pass.
  outputs = inputs
  for layer in layers:
    outputs = layer(outputs)

  # Build model using functional API.
  return tf.keras.Model(inputs=inputs, outputs=outputs)
