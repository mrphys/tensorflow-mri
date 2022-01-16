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
"""Tests for module `linalg_ops`."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import linalg

from tensorflow_mri.python.ops import linalg_ops
from tensorflow_mri.python.util import test_util


class LinearOperatorAdditionTest(test_util.TestCase):
  """Tests for `linalg_ops.LinearOperatorAddition`."""
  @parameterized.product(adjoint=[True, False], adjoint_arg=[True, False])
  def test_operator(self, adjoint, adjoint_arg):
    op1 = tf.linalg.LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
    op2 = tf.linalg.LinearOperatorFullMatrix([[4., 1.], [3., 2.]])

    op_sum = linalg_ops.LinearOperatorAddition([op1, op2])

    # Test `shape` and `shape_tensor`.
    self.assertEqual(op_sum.shape, tf.TensorShape([2, 2]))
    self.assertAllEqual(op_sum.shape_tensor(), tf.constant([2, 2]))

    # Test `matmul`.
    x = tf.random.normal((2, 2))
    self.assertAllClose(
        tf.linalg.matmul(op_sum, x, adjoint_a=adjoint, adjoint_b=adjoint_arg),
        (tf.linalg.matmul(op1, x, adjoint_a=adjoint, adjoint_b=adjoint_arg) +
         tf.linalg.matmul(op2, x, adjoint_a=adjoint, adjoint_b=adjoint_arg)))

    # `adjoint_arg` not supported for matvec.
    if adjoint_arg:
      return

    # Test `matvec`.
    x = tf.random.normal((2,))
    self.assertAllClose(
        tf.linalg.matvec(op_sum, x, adjoint_a=adjoint),
        (tf.linalg.matvec(op1, x, adjoint_a=adjoint) +
         tf.linalg.matvec(op2, x, adjoint_a=adjoint)))


class LinearOperatorStackTest(test_util.TestCase):
  """Tests for stack operators."""
  def test_operator(self):
    op1 = tf.linalg.LinearOperatorFullMatrix([[1., 2.],
                                              [3., 4.],
                                              [5., 6.]])
    op2 = tf.linalg.LinearOperatorFullMatrix([[1., 0.],
                                              [0., 1.]])
    op_vstack_ref = tf.linalg.LinearOperatorFullMatrix([[1., 2.],
                                                        [3., 4.],
                                                        [5., 6.],
                                                        [1., 0.],
                                                        [0., 1.]])
    op_hstack_ref = tf.linalg.LinearOperatorFullMatrix([[1., 3., 5., 1., 0.],
                                                        [2., 4., 6., 0., 1.]])
    op_vstack = linalg_ops.LinearOperatorVerticalStack([op1, op2])
    op_hstack = linalg_ops.LinearOperatorHorizontalStack([op1.H, op2.H])

    # Test `shape` and `shape_tensor`.
    self.assertEqual(op_vstack.shape, tf.TensorShape([5, 2]))
    self.assertAllEqual(op_vstack.shape_tensor(), tf.constant([5, 2]))
    self.assertEqual(op_hstack.shape, tf.TensorShape([2, 5]))
    self.assertAllEqual(op_hstack.shape_tensor(), tf.constant([2, 5]))

    # Test `matmul`.
    x = tf.random.normal((2, 2))
    self.assertAllClose(
        tf.linalg.matmul(op_vstack, x, adjoint_a=False),
        tf.linalg.matmul(op_vstack_ref, x, adjoint_a=False))
    self.assertAllClose(
        tf.linalg.matmul(op_hstack, x, adjoint_a=True),
        tf.linalg.matmul(op_hstack_ref, x, adjoint_a=True))

    y = tf.random.normal((5, 3))
    self.assertAllClose(
        tf.linalg.matmul(op_vstack, y, adjoint_a=True),
        tf.linalg.matmul(op_vstack_ref, y, adjoint_a=True))
    self.assertAllClose(
        tf.linalg.matmul(op_hstack, y, adjoint_a=False),
        tf.linalg.matmul(op_hstack_ref, y, adjoint_a=False))

    # Test `matvec`.
    x = tf.random.normal((2,))
    self.assertAllClose(
        tf.linalg.matvec(op_vstack, x, adjoint_a=False),
        tf.linalg.matvec(op_vstack_ref, x, adjoint_a=False))
    self.assertAllClose(
        tf.linalg.matvec(op_hstack, x, adjoint_a=True),
        tf.linalg.matvec(op_hstack_ref, x, adjoint_a=True))

    y = tf.random.normal((5,))
    self.assertAllClose(
        tf.linalg.matvec(op_vstack, y, adjoint_a=True),
        tf.linalg.matvec(op_vstack_ref, y, adjoint_a=True))
    self.assertAllClose(
        tf.linalg.matvec(op_hstack, y, adjoint_a=False),
        tf.linalg.matvec(op_hstack_ref, y, adjoint_a=False))


class LinearOperatorAppendColumn(linalg_ops.LinalgImagingMixin,
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
    else:
      # Add a column of zeros.
      return tf.pad(x, [[0, 0]] * (x.shape.rank - 1) + [[0, 1]])

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
    self.assertAllClose(tf.linalg.matvec(self.linop, self.x_vec),
                        self.y_vec)
    self.assertAllClose(tf.linalg.matvec(self.linop, self.y_vec, adjoint_a=True),
                        self.x_vec)

    self.assertAllClose(tf.linalg.matmul(self.linop, self.x_col),
                        self.y_col)
    self.assertAllClose(tf.linalg.matmul(self.linop, self.y_col, adjoint_a=True),
                        self.x_col)

  def test_matmul_operator(self):
    """Test `__matmul__` operator."""
    self.assertAllClose(self.linop @ self.x_col, self.y_col)

  def test_adjoint(self):
    """Test `adjoint` method."""
    self.assertIsInstance(self.linop.adjoint(),
                          linalg_ops.LinalgImagingMixin)
    self.assertAllClose(self.linop.adjoint() @ self.y_col, self.x_col)
    self.assertAllClose(self.linop.adjoint().domain_shape, self.range_shape)
    self.assertAllClose(self.linop.adjoint().range_shape, self.domain_shape)
    self.assertAllClose(self.linop.adjoint().domain_shape_tensor(),
                        self.range_shape_tensor)
    self.assertAllClose(self.linop.adjoint().range_shape_tensor(),
                        self.domain_shape_tensor)

  def test_adjoint_property(self):
    """Test `H` property."""
    self.assertIsInstance(self.linop.H, linalg_ops.LinalgImagingMixin)
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
      self.linop @ invalid_x


# class LinearOperatorFFTTest(test_util.TestCase):
#   """Tests for FFT linear operator."""

#   @classmethod
#   def setUpClass(cls):

#     super().setUpClass()
#     cls.linop1 = linalg_ops.LinearOperatorFFT([2, 2], norm=None)
#     cls.linop2 = linalg_ops.LinearOperatorFFT(
#         [2, 2], mask=[[False, False], [True, True]], norm=None)
#     cls.linop3 = linalg_ops.LinearOperatorFFT(
#         [2, 2], mask=[[[True, True], [False, False]],
#                       [[False, False], [True, True]],
#                       [[False, True], [True, False]]], norm=None)

#   def test_transform(self):
#     """Test transform method."""
#     signal = tf.constant([1, 2, 4, 4], dtype=tf.complex64)

#     result = tf.linalg.matvec(self.linop1, signal)
#     self.assertAllClose(result, [-1, 5, 1, 11])

#     result = tf.linalg.matvec(self.linop2, signal)
#     self.assertAllClose(result, [0, 0, 1, 11])

#     result = tf.linalg.matvec(self.linop3, signal)
#     self.assertAllClose(result, [[-1, 5, 0, 0], [0, 0, 1, 11], [0, 5, 1, 0]])

#   def test_domain_shape(self):
#     """Test domain shape."""
#     self.assertIsInstance(self.linop1.domain_shape, tf.TensorShape)
#     self.assertAllEqual(self.linop1.domain_shape, [2, 2])
#     self.assertAllEqual(self.linop1.domain_shape_tensor(), [2, 2])

#     self.assertIsInstance(self.linop2.domain_shape, tf.TensorShape)
#     self.assertAllEqual(self.linop2.domain_shape, [2, 2])
#     self.assertAllEqual(self.linop2.domain_shape_tensor(), [2, 2])

#     self.assertIsInstance(self.linop3.domain_shape, tf.TensorShape)
#     self.assertAllEqual(self.linop3.domain_shape, [2, 2])
#     self.assertAllEqual(self.linop3.domain_shape_tensor(), [2, 2])

#   def test_range_shape(self):
#     """Test range shape."""
#     self.assertIsInstance(self.linop1.range_shape, tf.TensorShape)
#     self.assertAllEqual(self.linop1.range_shape, [2, 2])
#     self.assertAllEqual(self.linop1.range_shape_tensor(), [2, 2])

#     self.assertIsInstance(self.linop2.range_shape, tf.TensorShape)
#     self.assertAllEqual(self.linop2.range_shape, [2, 2])
#     self.assertAllEqual(self.linop2.range_shape_tensor(), [2, 2])

#     self.assertIsInstance(self.linop3.range_shape, tf.TensorShape)
#     self.assertAllEqual(self.linop3.range_shape, [2, 2])
#     self.assertAllEqual(self.linop3.range_shape_tensor(), [2, 2])

#   def test_batch_shape(self):
#     """Test batch shape."""
#     self.assertIsInstance(self.linop1.batch_shape, tf.TensorShape)
#     self.assertAllEqual(self.linop1.batch_shape, [])
#     self.assertAllEqual(self.linop1.batch_shape_tensor(), [])

#     self.assertIsInstance(self.linop2.batch_shape, tf.TensorShape)
#     self.assertAllEqual(self.linop2.batch_shape, [])
#     self.assertAllEqual(self.linop2.batch_shape_tensor(), [])

#     self.assertIsInstance(self.linop3.batch_shape, tf.TensorShape)
#     self.assertAllEqual(self.linop3.batch_shape, [3])
#     self.assertAllEqual(self.linop3.batch_shape_tensor(), [3])

#   def test_norm(self):
#     """Test FFT normalization."""
#     linop = linalg_ops.LinearOperatorFFT([2, 2], norm='ortho')
#     x = tf.constant([1 + 2j, 2 - 2j, -1 - 6j, 3 + 4j], dtype=tf.complex64)
#     # With norm='ortho', subsequent application of the operator and its adjoint
#     # should not scale the input.
#     y = tf.linalg.matvec(linop.H, tf.linalg.matvec(linop, x))
#     self.assertAllClose(x, y)


# class LinearOperatorSensitivityModulationTest(test_util.TestCase):
#   """Tests for `linalg_ops.LinearOperatorSensitivityModulation`."""

#   def test_norm(self):
#     """Test normalization."""
#     sens = _random_normal_complex([2, 4, 4])
#     linop = linalg_ops.LinearOperatorSensitivityModulation(sens, norm=True)
#     x = _random_normal_complex([4 * 4])
#     y = tf.linalg.matvec(linop, x)
#     a = tf.linalg.matvec(linop.H, y)
#     self.assertAllClose(x, a)


# def _random_normal_complex(shape):
#   return tf.dtypes.complex(tf.random.normal(shape), tf.random.normal(shape))


class LinearOperatorDifferenceTest(test_util.TestCase):
  """Tests for difference linear operator."""
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.linop1 = linalg_ops.LinearOperatorDifference([4])
    cls.linop2 = linalg_ops.LinearOperatorDifference([4, 4], axis=-2)
    cls.matrix1 = tf.convert_to_tensor([[-1, 1, 0, 0],
                                        [0, -1, 1, 0],
                                        [0, 0, -1, 1]], dtype=tf.float32)

  def test_transform(self):
    """Test transform method."""
    signal = tf.random.normal([4, 4])
    result = self.linop2.transform(signal)
    self.assertAllClose(result, np.diff(signal, axis=-2))

  def test_matvec(self):
    """Test matvec method."""
    signal = tf.constant([1, 2, 4, 8], dtype=tf.float32)

    result = tf.linalg.matvec(self.linop1, signal)
    self.assertAllClose(result, [1, 2, 4])
    self.assertAllClose(result, np.diff(signal))
    self.assertAllClose(result, tf.linalg.matvec(self.matrix1, signal))

    signal2 = tf.range(16, dtype=tf.float32)
    result = tf.linalg.matvec(self.linop2, signal2)
    self.assertAllClose(result, [4] * 12)

  def test_matvec_adjoint(self):
    """Test matvec with adjoint."""
    signal = tf.constant([1, 2, 4], dtype=tf.float32)
    result = tf.linalg.matvec(self.linop1, signal, adjoint_a=True)
    self.assertAllClose(result,
                        tf.linalg.matvec(tf.transpose(self.matrix1), signal))

  def test_shapes(self):
    """Test shapes."""
    self._test_all_shapes(self.linop1, [4], [3])
    self._test_all_shapes(self.linop2, [4, 4], [3, 4])

  def _test_all_shapes(self, linop, domain_shape, range_shape):
    """Test shapes."""
    self.assertIsInstance(linop.domain_shape, tf.TensorShape)
    self.assertAllEqual(linop.domain_shape, domain_shape)
    self.assertAllEqual(linop.domain_shape_tensor(), domain_shape)

    self.assertIsInstance(linop.range_shape, tf.TensorShape)
    self.assertAllEqual(linop.range_shape, range_shape)
    self.assertAllEqual(linop.range_shape_tensor(), range_shape)


if __name__ == '__main__':
  tf.test.main()
