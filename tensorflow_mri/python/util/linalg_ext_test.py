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
"""Tests for module `util.linalg_ext`."""

from absl import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.util import linalg_ext
from tensorflow_mri.python.util import test_util


class LinearOperatorAdditionTest(test_util.TestCase):
  """Tests for `LinearOperatorAddition`."""
  @parameterized.product(adjoint=[True, False], adjoint_arg=[True, False])
  def test_operator(self, adjoint, adjoint_arg):
    op1 = tf.linalg.LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
    op2 = tf.linalg.LinearOperatorFullMatrix([[4., 1.], [3., 2.]])

    op_sum = linalg_ext.LinearOperatorAddition([op1, op2])

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
    op_vstack = linalg_ext.LinearOperatorVerticalStack([op1, op2])
    op_hstack = linalg_ext.LinearOperatorHorizontalStack([op1.H, op2.H])

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


class LinearOperatorDifferenceTest(test_util.TestCase):
  """Tests for difference linear operator."""
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.linop = linalg_ext.LinearOperatorFiniteDifference(4)
    cls.matrix = tf.convert_to_tensor([[-1, 1, 0, 0],
                                       [0, -1, 1, 0],
                                       [0, 0, -1, 1]], dtype=tf.float32)

  def test_matmul(self):
    """Test matmul method."""
    x = tf.constant([[5, 4], [1, 4], [1, 3], [6, 2]], dtype=tf.float32)
    self.assertAllClose(self.linop @ x, self.matrix @ x)

  def test_matmul_adjoint(self):
    """Test matmul method with adjoint."""
    x = tf.constant([[9, 4, 1, 3], [1, 3, 1, 6], [7, 2, 2, 8]],
                    dtype=tf.float32)
    self.assertAllClose(
        tf.linalg.matmul(self.linop, x, adjoint_a=True),
        tf.linalg.matmul(self.matrix, x, adjoint_a=True))
    self.assertAllClose(
        tf.linalg.matmul(self.linop, x, adjoint_a=False, adjoint_b=True),
        tf.linalg.matmul(self.matrix, x, adjoint_a=False, adjoint_b=True))

  def test_matvec(self):
    """Test matvec method."""
    signal = tf.constant([1, 2, 4, 8], dtype=tf.float32)
    result = tf.linalg.matvec(self.linop, signal)
    self.assertAllClose(result, [1, 2, 4])
    self.assertAllClose(result, np.diff(signal))
    self.assertAllClose(result, tf.linalg.matvec(self.matrix, signal))

  def test_matvec_adjoint(self):
    """Test matvec with adjoint."""
    signal = tf.constant([1, 2, 4], dtype=tf.float32)
    result = tf.linalg.matvec(self.linop, signal, adjoint_a=True)
    self.assertAllClose(
        result, tf.linalg.matvec(tf.transpose(self.matrix), signal))

  def test_shapes(self):
    """Test shapes."""
    self.assertAllEqual(self.linop.shape, [3, 4])
    self.assertAllEqual(self.linop.shape_tensor(), [3, 4])
