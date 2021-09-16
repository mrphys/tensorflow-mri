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

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import linalg_ops
from tensorflow_mri.python.util import test_util


class LinearOperatorFFTTest(test_util.TestCase):

  @classmethod
  def setUpClass(cls):

    super().setUpClass()
    cls.linop1 = linalg_ops.LinearOperatorFFT([2, 2])
    cls.linop2 = linalg_ops.LinearOperatorFFT([2, 2], mask=[[False, False],
                                                            [True, True]])
    cls.linop3 = linalg_ops.LinearOperatorFFT(
        [2, 2], mask=[[[True, True], [False, False]],
                      [[False, False], [True, True]],
                      [[False, True], [True, False]]])

  def test_transform(self):

    signal = tf.constant([1, 2, 4, 4], dtype=tf.complex64)

    result = tf.linalg.matvec(self.linop1, signal)
    self.assertAllClose(result, [-1, 5, 1, 11])

    result = tf.linalg.matvec(self.linop2, signal)
    self.assertAllClose(result, [0, 0, 1, 11])

    result = tf.linalg.matvec(self.linop3, signal)
    self.assertAllClose(result, [[-1, 5, 0, 0], [0, 0, 1, 11], [0, 5, 1, 0]])

  def test_domain_shape(self):

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

    self.assertIsInstance(self.linop1.batch_shape, tf.TensorShape)
    self.assertAllEqual(self.linop1.batch_shape, [])
    self.assertAllEqual(self.linop1.batch_shape_tensor(), [])

    self.assertIsInstance(self.linop2.batch_shape, tf.TensorShape)
    self.assertAllEqual(self.linop2.batch_shape, [])
    self.assertAllEqual(self.linop2.batch_shape_tensor(), [])

    self.assertIsInstance(self.linop3.batch_shape, tf.TensorShape)
    self.assertAllEqual(self.linop3.batch_shape, [3])
    self.assertAllEqual(self.linop3.batch_shape_tensor(), [3])


class LinearOperatorDifferenceTest(test_util.TestCase):

  @classmethod
  def setUpClass(cls):

    super().setUpClass()
    cls.linop1 = linalg_ops.LinearOperatorDifference([4])
    cls.linop2 = linalg_ops.LinearOperatorDifference([4, 4], axis=-2)
    cls.matrix1 = tf.convert_to_tensor([[-1, 1, 0, 0],
                                        [0, -1, 1, 0],
                                        [0, 0, -1, 1]], dtype=tf.float32)

  def test_transform(self):

    signal = tf.constant([1, 2, 4, 8], dtype=tf.float32)

    result = tf.linalg.matvec(self.linop1, signal)
    self.assertAllClose(result, [1, 2, 4])
    self.assertAllClose(result, np.diff(signal))
    self.assertAllClose(result, tf.linalg.matvec(self.matrix1, signal))

    signal2 = tf.range(16, dtype=tf.float32)
    result = tf.linalg.matvec(self.linop2, signal2)
    self.assertAllClose(result, [4] * 12)
    
  def test_transform_adjoint(self):

    signal = tf.constant([1, 2, 4], dtype=tf.float32)
    result = tf.linalg.matvec(self.linop1, signal, adjoint_a=True)
    self.assertAllClose(result,
                        tf.linalg.matvec(tf.transpose(self.matrix1), signal))

  def test_shapes(self):
    
    self._test_all_shapes(self.linop1, [4], [3])
    self._test_all_shapes(self.linop2, [4, 4], [3, 4])

  def _test_all_shapes(self, linop, domain_shape, range_shape):

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
