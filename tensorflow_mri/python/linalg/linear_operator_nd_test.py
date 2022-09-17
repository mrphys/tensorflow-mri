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
"""Tests for module `linear_operator`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import test_util


class LinearOperatorAppendColumn(linear_operator.LinearOperatorMixin,  # pylint: disable=abstract-method
                                 tf.linalg.LinearOperator):
  """Linear operator which appends a column of zeros to the input.

  Used in tests below.
  """
  def __init__(self, domain_shape):
    parameters = {
      'domain_shape': domain_shape}
    range_shape = tf.TensorShape(domain_shape).as_list()
    range_shape[-1] += 1
    self._domain_shape_value = tf.TensorShape(domain_shape)
    self._range_shape_value = tf.TensorShape(range_shape)
    super().__init__(tf.dtypes.float32, parameters=parameters)

  def _transform(self, x, adjoint=False):
    if adjoint:
      # Remove last column.
      return x[..., :-1]
    # Add a column of zeros.
    return tf.pad(x, [[0, 0]] * (x.shape.rank - 1) + [[0, 1]])  # pylint: disable=no-value-for-parameter

  def _domain_shape(self):
    return self._domain_shape_value

  def _range_shape(self):
    return self._range_shape_value


class LinearOperatorMixin(test_util.TestCase):
  """Tests for `LinearOperatorMixin`."""
  @classmethod
  def setUpClass(cls):
    # Test shapes.
    cls.domain_shape = [2, 2]
    cls.range_shape = [2, 3]
    cls.domain_shape_tensor = tf.convert_to_tensor(cls.domain_shape)
    cls.range_shape_tensor = tf.convert_to_tensor(cls.range_shape)
    # Test linear operator.
    cls.linop = LinearOperatorAppendColumn(cls.domain_shape)
    # Test inputs/outputs.
    cls.x = tf.constant([[1., 2.], [3., 4.]])
    cls.y = tf.constant([[1., 2., 0.], [3., 4., 0.]])
    cls.x_vec = tf.reshape(cls.x, [-1])
    cls.y_vec = tf.reshape(cls.y, [-1])
    cls.x_col = tf.reshape(cls.x, [-1, 1])
    cls.y_col = tf.reshape(cls.y, [-1, 1])

  def test_static_shapes(self):
    """Test static shapes."""
    self.assertAllClose(self.linop.domain_shape, self.domain_shape)
    self.assertAllClose(self.linop.range_shape, self.range_shape)

  def test_dynamic_shapes(self):
    """Test dynamic shapes."""
    self.assertAllClose(self.linop.domain_shape_tensor(),
                        self.domain_shape_tensor)
    self.assertAllClose(self.linop.range_shape_tensor(),
                        self.range_shape_tensor)

  def test_transform(self):
    """Test `transform` method."""
    self.assertAllClose(self.linop.transform(self.x), self.y)
    self.assertAllClose(self.linop.transform(self.y, adjoint=True), self.x)

  def test_matvec(self):
    """Test `matvec` method."""
    self.assertAllClose(self.linop.matvec(self.x_vec), self.y_vec)
    self.assertAllClose(self.linop.matvec(self.y_vec, adjoint=True), self.x_vec)

  def test_matmul(self):
    """Test `matmul` method."""
    self.assertAllClose(self.linop.matmul(self.x_col), self.y_col)
    self.assertAllClose(self.linop.matmul(self.y_col, adjoint=True), self.x_col)

  def test_linalg_functions(self):
    """Test `tf.linalg` functions."""
    self.assertAllClose(
        tf.linalg.matvec(self.linop, self.x_vec), self.y_vec)
    self.assertAllClose(
        tf.linalg.matvec(self.linop, self.y_vec, adjoint_a=True), self.x_vec)

    self.assertAllClose(
        tf.linalg.matmul(self.linop, self.x_col), self.y_col)
    self.assertAllClose(
        tf.linalg.matmul(self.linop, self.y_col, adjoint_a=True), self.x_col)

  def test_matmul_operator(self):
    """Test `__matmul__` operator."""
    self.assertAllClose(self.linop @ self.x_col, self.y_col)

  def test_adjoint(self):
    """Test `adjoint` method."""
    self.assertIsInstance(self.linop.adjoint(),
                          linear_operator.LinearOperatorMixin)
    self.assertAllClose(self.linop.adjoint() @ self.y_col, self.x_col)
    self.assertAllClose(self.linop.adjoint().domain_shape, self.range_shape)
    self.assertAllClose(self.linop.adjoint().range_shape, self.domain_shape)
    self.assertAllClose(self.linop.adjoint().domain_shape_tensor(),
                        self.range_shape_tensor)
    self.assertAllClose(self.linop.adjoint().range_shape_tensor(),
                        self.domain_shape_tensor)

  def test_adjoint_property(self):
    """Test `H` property."""
    self.assertIsInstance(self.linop.H, linear_operator.LinearOperatorMixin)
    self.assertAllClose(self.linop.H @ self.y_col, self.x_col)
    self.assertAllClose(self.linop.H.domain_shape, self.range_shape)
    self.assertAllClose(self.linop.H.range_shape, self.domain_shape)
    self.assertAllClose(self.linop.H.domain_shape_tensor(),
                        self.range_shape_tensor)
    self.assertAllClose(self.linop.H.range_shape_tensor(),
                        self.domain_shape_tensor)

  def test_unsupported_matmul(self):
    """Test `matmul` method with a non-column input."""
    message = "does not support matrix multiplication"
    invalid_x = tf.random.normal([4, 4])
    with self.assertRaisesRegex(ValueError, message):
      self.linop.matmul(invalid_x)
    with self.assertRaisesRegex(ValueError, message):
      tf.linalg.matmul(self.linop, invalid_x)
    with self.assertRaisesRegex(ValueError, message):
      self.linop @ invalid_x  # pylint: disable=pointless-statement
