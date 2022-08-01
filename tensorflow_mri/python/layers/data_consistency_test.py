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
"""Tests for data consistency layers."""

import tempfile

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.layers import data_consistency
from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import test_util


class LeastSquaresGradientDescentTest(test_util.TestCase):
  @parameterized.product(operator_type=['class', 'instance'],
                         input_type=['dict', 'tuple'])
  def test_general(self, operator_type, input_type):
    scale = tf.constant(2.0, dtype=tf.float32)
    dtype = tf.complex64
    if operator_type == 'class':
      # Operator is a class.
      class LinearOperatorScalarMultiplyComplex64(LinearOperatorScalarMultiply):
        # Same as `LinearOperatorScalarMultiply` but dtype is tf.complex64.
        def __init__(self, *args, **kwargs):
          if 'dtype' in kwargs:
            raise ValueError('dtype is not allowed in this class.')
          kwargs['dtype'] = tf.complex64
          super().__init__(*args, **kwargs)

      operator = LinearOperatorScalarMultiplyComplex64
      args = (tf.expand_dims(scale, axis=0),)
      kwargs = {'scale': tf.expand_dims(scale, axis=0)}
    else:
      # Operator is an instance.
      operator = LinearOperatorScalarMultiply(scale, dtype=dtype)
      args, kwargs = (), {}

    # Initialize layer.
    layer = data_consistency.LeastSquaresGradientDescent(
        operator, scale_initializer=0.5, dtype=dtype)

    # All variables have a batch dimension.
    x = tf.constant([[3, 3]], dtype=dtype)
    b = tf.constant([[1, 1]], dtype=dtype)
    expected_output = tf.constant([[-2.0 + 0.0j, -2.0 + 0.0j]], dtype=dtype)

    # Create input data.
    if input_type == 'dict':
      input_data = {'x': x, 'b': b}
      input_data.update(kwargs)
    else:
      input_data = (x, b)
      input_data += args

    # Test layer.
    output = layer(input_data)
    self.assertAllClose(expected_output, output)

    # Test serialization.
    layer_config = layer.get_config()
    layer = data_consistency.LeastSquaresGradientDescent.from_config(
        layer_config)

    # Test layer with tuple inputs.
    output = layer(input_data)
    self.assertAllClose(expected_output, output)

    # Test layer in a model.
    inputs = tf.nest.map_structure(
        lambda x: tf.keras.Input(shape=x.shape[1:], dtype=x.dtype),
        input_data)
    model = tf.keras.Model(inputs=inputs, outputs=layer(inputs))
    output = model(input_data)
    self.assertAllClose(expected_output, output)

    # Test training.
    model.compile(optimizer='sgd', loss='mse')
    model.fit(input_data, expected_output * 2)
    expected_weights = [0.9]
    expected_output = tf.constant([[-6.0 + 0.0j, -6.0 + 0.0j]],
                                  dtype=tf.complex64)
    self.assertAllClose(expected_weights, model.get_weights())
    self.assertAllClose(expected_output, model(input_data))

    # Test model saving.
    with tempfile.TemporaryDirectory() as tmpdir:
      model.save(tmpdir + '/model')
      model = tf.keras.models.load_model(tmpdir + '/model')
      output = model(input_data)
      self.assertAllClose(expected_output, output)


@linear_operator.make_composite_tensor
class LinearOperatorScalarMultiply(linear_operator.LinearOperator):
  def __init__(self, scale, dtype=None, **kwargs):
    parameters = {'scale': scale}
    self.scale = tf.convert_to_tensor(scale)
    super().__init__(dtype=dtype or self.scale.dtype,
                      parameters=parameters,
                      **kwargs)

  def _transform(self, x, adjoint=False):
    if adjoint:
      return x * tf.math.conj(tf.cast(self.scale, x.dtype))
    else:
      return x * tf.cast(self.scale, x.dtype)

  def _domain_shape(self):
    return tf.TensorShape([2])

  def _range_shape(self):
    return self._domain_shape()
  
  def _batch_shape(self):
    return self.scale.shape[:-1]

  @property
  def _composite_tensor_fields(self):
    return ('scale',)
