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


class LinearOperatorImagingMixinTest(test_util.TestCase):
  """Tests for `linalg_ops.LinearOperatorImagingMixin`."""

  def test_mixin(self):
    class LinopColumnWiseMultiplication(linalg_ops.LinearOperatorImagingMixin,
                                        tf.linalg.LinearOperator):

      def __init__(self, domain_shape, vector):
        parameters = {
          'domain_shape': domain_shape,
          'vector': vector}
        self._domain_shape_value = tf.TensorShape(shape)
        self._vector = vector
        super().__init__(tf.dtypes.float32, parameters=parameters)

      def _transform(self, x, adjoint=False):
        if adjoint:
          return x / self._vector
        else:
          return x * self._vector

      def _domain_shape(self):
        return self._domain_shape_value

      def _range_shape(self):
        return self._domain_shape_value

    # Initialize linear operator.
    shape = [3, 4]
    vector = tf.range(1, 5, dtype=tf.dtypes.float32)
    linop = LinopColumnWiseMultiplication(shape, vector)

    # Test static shapes.
    self.assertAllClose(linop.domain_shape, shape)
    self.assertAllClose(linop.range_shape, shape)

    # Test dynamic shapes.
    shape_tensor = tf.convert_to_tensor(shape)
    self.assertAllClose(linop.domain_shape_tensor(), shape_tensor)
    self.assertAllClose(linop.range_shape_tensor(), shape_tensor)

    # Test transform method.
    x = tf.random.normal(shape)
    self.assertAllClose(linop.transform(x), x * vector)
    self.assertAllClose(linop.transform(x, adjoint=True), x / vector)

    # Test matvec method.
    to_vector = lambda x: tf.reshape(x, [-1])
    self.assertAllClose(linop.matvec(to_vector(x)),
                        to_vector(x * vector))
    self.assertAllClose(linop.matvec(to_vector(x), adjoint=True),
                        to_vector(x / vector))

    # Test matmul method.
    to_matrix = lambda x: tf.reshape(x, [-1, 1])
    self.assertAllClose(linop.matmul(to_matrix(x)),
                        to_matrix(x * vector))
    self.assertAllClose(linop.matmul(to_matrix(x), adjoint=True),
                        to_matrix(x / vector))

    # Test tf.linalg.matvec.
    self.assertAllClose(tf.linalg.matvec(linop, to_vector(x)),
                        to_vector(x * vector))
    self.assertAllClose(tf.linalg.matvec(linop, to_vector(x), adjoint_a=True),
                        to_vector(x / vector))

    # Test tf.linalg.matmul.
    self.assertAllClose(tf.linalg.matmul(linop, to_matrix(x)),
                        to_matrix(x * vector))
    self.assertAllClose(tf.linalg.matmul(linop, to_matrix(x), adjoint_a=True),
                        to_matrix(x / vector))

    # Test __matmul__ operator.
    self.assertAllClose(linop @ to_matrix(x), to_matrix(x * vector))

    # Test adjointing.
    self.assertAllClose(linop.H.domain_shape, shape)
    self.assertAllClose(linop.H.range_shape, shape)
    self.assertAllClose(linop.H.domain_shape_tensor(), shape_tensor)
    self.assertAllClose(linop.H.range_shape_tensor(), shape_tensor)
    self.assertAllClose(linop.H @ to_matrix(x), to_matrix(x / vector))

    # Test unsupported matmul.
    message = "does not support matrix multiplication"
    invalid_x = tf.random.normal([12, 4])
    with self.assertRaisesRegex(ValueError, message):
      linop.matmul(invalid_x)
    with self.assertRaisesRegex(ValueError, message):
      tf.linalg.matmul(linop, invalid_x)
    with self.assertRaisesRegex(ValueError, message):
      linop @ invalid_x




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
    signal = tf.constant([1, 2, 4, 8], dtype=tf.float32)

    result = tf.linalg.matvec(self.linop1, signal)
    self.assertAllClose(result, [1, 2, 4])
    self.assertAllClose(result, np.diff(signal))
    self.assertAllClose(result, tf.linalg.matvec(self.matrix1, signal))

    signal2 = tf.range(16, dtype=tf.float32)
    result = tf.linalg.matvec(self.linop2, signal2)
    self.assertAllClose(result, [4] * 12)

  def test_transform_adjoint(self):
    """Test adjoint."""
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


# TODO: Add more tests. Much of `linalg_ops` is tested indirectly by recon ops,
# but specific tests should be added here.


if __name__ == '__main__':
  tf.test.main()
