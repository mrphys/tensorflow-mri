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

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.util import linalg_ext
from tensorflow_mri.python.util import test_util


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
