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

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.util import linalg_imaging
from tensorflow_mri.python.util import test_util


class LinearOperatorFiniteDifferenceTest(test_util.TestCase):
  """Tests for difference linear operator."""
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.linop1 = linalg_imaging.LinearOperatorFiniteDifference([4])
    cls.linop2 = linalg_imaging.LinearOperatorFiniteDifference([4, 4], axis=-2)
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
