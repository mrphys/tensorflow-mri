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
"""Tests for module `util.linalg_imaging`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import tensorflow as tf

from tensorflow_mri.python.util import linalg_imaging
from tensorflow_mri.python.util import test_util


class LinearOperatorAppendColumn(linalg_imaging.LinalgImagingMixin,  # pylint: disable=abstract-method
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


class LinalgImagingMixin(test_util.TestCase):
  """Tests for `linalg_ops.LinalgImagingMixin`."""
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
                          linalg_imaging.LinalgImagingMixin)
    self.assertAllClose(self.linop.adjoint() @ self.y_col, self.x_col)
    self.assertAllClose(self.linop.adjoint().domain_shape, self.range_shape)
    self.assertAllClose(self.linop.adjoint().range_shape, self.domain_shape)
    self.assertAllClose(self.linop.adjoint().domain_shape_tensor(),
                        self.range_shape_tensor)
    self.assertAllClose(self.linop.adjoint().range_shape_tensor(),
                        self.domain_shape_tensor)

  def test_adjoint_property(self):
    """Test `H` property."""
    self.assertIsInstance(self.linop.H, linalg_imaging.LinalgImagingMixin)
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


class LinearOperatorDiagTest(test_util.TestCase):
  """Tests for `linalg_imaging.LinearOperatorDiag`."""
  def test_transform(self):
    """Test `transform` method."""
    diag = tf.constant([[1., 2.], [3., 4.]])
    diag_linop = linalg_imaging.LinearOperatorDiag(diag, rank=2)
    x = tf.constant([[2., 2.], [2., 2.]])
    self.assertAllClose([[2., 4.], [6., 8.]], diag_linop.transform(x))

  def test_transform_adjoint(self):
    """Test `transform` method with adjoint."""
    diag = tf.constant([[1., 2.], [3., 4.]])
    diag_linop = linalg_imaging.LinearOperatorDiag(diag, rank=2)
    x = tf.constant([[2., 2.], [2., 2.]])
    self.assertAllClose([[2., 4.], [6., 8.]],
                        diag_linop.transform(x, adjoint=True))

  def test_transform_complex(self):
    """Test `transform` method with complex values."""
    diag = tf.constant([[1. + 1.j, 2. + 2.j], [3. + 3.j, 4. + 4.j]],
                       dtype=tf.complex64)
    diag_linop = linalg_imaging.LinearOperatorDiag(diag, rank=2)
    x = tf.constant([[2., 2.], [2., 2.]], dtype=tf.complex64)
    self.assertAllClose([[2. + 2.j, 4. + 4.j], [6. + 6.j, 8. + 8.j]],
                        diag_linop.transform(x))

  def test_transform_adjoint_complex(self):
    """Test `transform` method with adjoint and complex values."""
    diag = tf.constant([[1. + 1.j, 2. + 2.j], [3. + 3.j, 4. + 4.j]],
                       dtype=tf.complex64)
    diag_linop = linalg_imaging.LinearOperatorDiag(diag, rank=2)
    x = tf.constant([[2., 2.], [2., 2.]], dtype=tf.complex64)
    self.assertAllClose([[2. - 2.j, 4. - 4.j], [6. - 6.j, 8. - 8.j]],
                        diag_linop.transform(x, adjoint=True))

  def test_shapes(self):
    """Test shapes."""
    diag = tf.constant([[1., 2.], [3., 4.]])
    diag_linop = linalg_imaging.LinearOperatorDiag(diag, rank=2)
    self.assertIsInstance(diag_linop.domain_shape, tf.TensorShape)
    self.assertIsInstance(diag_linop.range_shape, tf.TensorShape)
    self.assertAllEqual([2, 2], diag_linop.domain_shape)
    self.assertAllEqual([2, 2], diag_linop.range_shape)

  def test_tensor_shapes(self):
    """Test tensor shapes."""
    diag = tf.constant([[1., 2.], [3., 4.]])
    diag_linop = linalg_imaging.LinearOperatorDiag(diag, rank=2)
    self.assertIsInstance(diag_linop.domain_shape_tensor(), tf.Tensor)
    self.assertIsInstance(diag_linop.range_shape_tensor(), tf.Tensor)
    self.assertAllEqual([2, 2], diag_linop.domain_shape_tensor())
    self.assertAllEqual([2, 2], diag_linop.range_shape_tensor())

  def test_batch_shapes(self):
    """Test batch shapes."""
    diag = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    diag_linop = linalg_imaging.LinearOperatorDiag(diag, rank=1)
    self.assertIsInstance(diag_linop.domain_shape, tf.TensorShape)
    self.assertIsInstance(diag_linop.range_shape, tf.TensorShape)
    self.assertIsInstance(diag_linop.batch_shape, tf.TensorShape)
    self.assertAllEqual([3], diag_linop.domain_shape)
    self.assertAllEqual([3], diag_linop.range_shape)
    self.assertAllEqual([2], diag_linop.batch_shape)

  def test_tensor_batch_shapes(self):
    """Test tensor batch shapes."""
    diag = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    diag_linop = linalg_imaging.LinearOperatorDiag(diag, rank=1)
    self.assertIsInstance(diag_linop.domain_shape_tensor(), tf.Tensor)
    self.assertIsInstance(diag_linop.range_shape_tensor(), tf.Tensor)
    self.assertIsInstance(diag_linop.batch_shape_tensor(), tf.Tensor)
    self.assertAllEqual([3], diag_linop.domain_shape)
    self.assertAllEqual([3], diag_linop.range_shape)
    self.assertAllEqual([2], diag_linop.batch_shape)

  def test_name(self):
    """Test names."""
    diag = tf.constant([[1., 2.], [3., 4.]])
    diag_linop = linalg_imaging.LinearOperatorDiag(diag, rank=2)
    self.assertEqual("LinearOperatorDiag", diag_linop.name)
