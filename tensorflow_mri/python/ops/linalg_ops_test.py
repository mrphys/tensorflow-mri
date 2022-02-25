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


class LinearOperatorFFTTest(test_util.TestCase):
  """Tests for FFT linear operator."""

  @classmethod
  def setUpClass(cls):

    super().setUpClass()
    cls.linop1 = linalg_ops.LinearOperatorFFT([2, 2], norm=None)
    cls.linop2 = linalg_ops.LinearOperatorFFT(
        [2, 2], mask=[[False, False], [True, True]], norm=None)
    cls.linop3 = linalg_ops.LinearOperatorFFT(
        [2, 2], mask=[[[True, True], [False, False]],
                      [[False, False], [True, True]],
                      [[False, True], [True, False]]], norm=None)

  def test_transform(self):
    """Test transform method."""
    signal = tf.constant([1, 2, 4, 4], dtype=tf.complex64)

    result = tf.linalg.matvec(self.linop1, signal)
    self.assertAllClose(result, [-1, 5, 1, 11])

    result = tf.linalg.matvec(self.linop2, signal)
    self.assertAllClose(result, [0, 0, 1, 11])

    result = tf.linalg.matvec(self.linop3, signal)
    self.assertAllClose(result, [[-1, 5, 0, 0], [0, 0, 1, 11], [0, 5, 1, 0]])

  def test_domain_shape(self):
    """Test domain shape."""
    self.assertIsInstance(self.linop1.domain_shape, tf.TensorShape)
    self.assertAllEqual(self.linop1.domain_shape, [2, 2])
    self.assertAllEqual(self.linop1.domain_shape_tensor(), [2, 2])

    self.assertIsInstance(self.linop2.domain_shape, tf.TensorShape)
    self.assertAllEqual(self.linop2.domain_shape, [2, 2])
    self.assertAllEqual(self.linop2.domain_shape_tensor(), [2, 2])

    self.assertIsInstance(self.linop3.domain_shape, tf.TensorShape)
    self.assertAllEqual(self.linop3.domain_shape, [2, 2])
    self.assertAllEqual(self.linop3.domain_shape_tensor(), [2, 2])

  def test_range_shape(self):
    """Test range shape."""
    self.assertIsInstance(self.linop1.range_shape, tf.TensorShape)
    self.assertAllEqual(self.linop1.range_shape, [2, 2])
    self.assertAllEqual(self.linop1.range_shape_tensor(), [2, 2])

    self.assertIsInstance(self.linop2.range_shape, tf.TensorShape)
    self.assertAllEqual(self.linop2.range_shape, [2, 2])
    self.assertAllEqual(self.linop2.range_shape_tensor(), [2, 2])

    self.assertIsInstance(self.linop3.range_shape, tf.TensorShape)
    self.assertAllEqual(self.linop3.range_shape, [2, 2])
    self.assertAllEqual(self.linop3.range_shape_tensor(), [2, 2])

  def test_batch_shape(self):
    """Test batch shape."""
    self.assertIsInstance(self.linop1.batch_shape, tf.TensorShape)
    self.assertAllEqual(self.linop1.batch_shape, [])
    self.assertAllEqual(self.linop1.batch_shape_tensor(), [])

    self.assertIsInstance(self.linop2.batch_shape, tf.TensorShape)
    self.assertAllEqual(self.linop2.batch_shape, [])
    self.assertAllEqual(self.linop2.batch_shape_tensor(), [])

    self.assertIsInstance(self.linop3.batch_shape, tf.TensorShape)
    self.assertAllEqual(self.linop3.batch_shape, [3])
    self.assertAllEqual(self.linop3.batch_shape_tensor(), [3])

  def test_norm(self):
    """Test FFT normalization."""
    linop = linalg_ops.LinearOperatorFFT([2, 2], norm='ortho')
    x = tf.constant([1 + 2j, 2 - 2j, -1 - 6j, 3 + 4j], dtype=tf.complex64)
    # With norm='ortho', subsequent application of the operator and its adjoint
    # should not scale the input.
    y = tf.linalg.matvec(linop.H, tf.linalg.matvec(linop, x))
    self.assertAllClose(x, y)


if __name__ == '__main__':
  tf.test.main()
