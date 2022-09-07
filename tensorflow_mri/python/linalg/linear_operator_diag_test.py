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
"""Tests for module `linear_operator_diag`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import test_util


class LinearOperatorDiagTest(test_util.TestCase):
  """Tests for `linear_operator.LinearOperatorDiag`."""
  def test_transform(self):
    """Test `transform` method."""
    diag = tf.constant([[1., 2.], [3., 4.]])
    diag_linop = linear_operator.LinearOperatorDiag(diag, rank=2)
    x = tf.constant([[2., 2.], [2., 2.]])
    self.assertAllClose([[2., 4.], [6., 8.]], diag_linop.transform(x))

  def test_transform_adjoint(self):
    """Test `transform` method with adjoint."""
    diag = tf.constant([[1., 2.], [3., 4.]])
    diag_linop = linear_operator.LinearOperatorDiag(diag, rank=2)
    x = tf.constant([[2., 2.], [2., 2.]])
    self.assertAllClose([[2., 4.], [6., 8.]],
                        diag_linop.transform(x, adjoint=True))

  def test_transform_complex(self):
    """Test `transform` method with complex values."""
    diag = tf.constant([[1. + 1.j, 2. + 2.j], [3. + 3.j, 4. + 4.j]],
                       dtype=tf.complex64)
    diag_linop = linear_operator.LinearOperatorDiag(diag, rank=2)
    x = tf.constant([[2., 2.], [2., 2.]], dtype=tf.complex64)
    self.assertAllClose([[2. + 2.j, 4. + 4.j], [6. + 6.j, 8. + 8.j]],
                        diag_linop.transform(x))

  def test_transform_adjoint_complex(self):
    """Test `transform` method with adjoint and complex values."""
    diag = tf.constant([[1. + 1.j, 2. + 2.j], [3. + 3.j, 4. + 4.j]],
                       dtype=tf.complex64)
    diag_linop = linear_operator.LinearOperatorDiag(diag, rank=2)
    x = tf.constant([[2., 2.], [2., 2.]], dtype=tf.complex64)
    self.assertAllClose([[2. - 2.j, 4. - 4.j], [6. - 6.j, 8. - 8.j]],
                        diag_linop.transform(x, adjoint=True))

  def test_shapes(self):
    """Test shapes."""
    diag = tf.constant([[1., 2.], [3., 4.]])
    diag_linop = linear_operator.LinearOperatorDiag(diag, rank=2)
    self.assertIsInstance(diag_linop.domain_shape, tf.TensorShape)
    self.assertIsInstance(diag_linop.range_shape, tf.TensorShape)
    self.assertAllEqual([2, 2], diag_linop.domain_shape)
    self.assertAllEqual([2, 2], diag_linop.range_shape)

  def test_tensor_shapes(self):
    """Test tensor shapes."""
    diag = tf.constant([[1., 2.], [3., 4.]])
    diag_linop = linear_operator.LinearOperatorDiag(diag, rank=2)
    self.assertIsInstance(diag_linop.domain_shape_tensor(), tf.Tensor)
    self.assertIsInstance(diag_linop.range_shape_tensor(), tf.Tensor)
    self.assertAllEqual([2, 2], diag_linop.domain_shape_tensor())
    self.assertAllEqual([2, 2], diag_linop.range_shape_tensor())

  def test_batch_shapes(self):
    """Test batch shapes."""
    diag = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    diag_linop = linear_operator.LinearOperatorDiag(diag, rank=1)
    self.assertIsInstance(diag_linop.domain_shape, tf.TensorShape)
    self.assertIsInstance(diag_linop.range_shape, tf.TensorShape)
    self.assertIsInstance(diag_linop.batch_shape, tf.TensorShape)
    self.assertAllEqual([3], diag_linop.domain_shape)
    self.assertAllEqual([3], diag_linop.range_shape)
    self.assertAllEqual([2], diag_linop.batch_shape)

  def test_tensor_batch_shapes(self):
    """Test tensor batch shapes."""
    diag = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    diag_linop = linear_operator.LinearOperatorDiag(diag, rank=1)
    self.assertIsInstance(diag_linop.domain_shape_tensor(), tf.Tensor)
    self.assertIsInstance(diag_linop.range_shape_tensor(), tf.Tensor)
    self.assertIsInstance(diag_linop.batch_shape_tensor(), tf.Tensor)
    self.assertAllEqual([3], diag_linop.domain_shape)
    self.assertAllEqual([3], diag_linop.range_shape)
    self.assertAllEqual([2], diag_linop.batch_shape)

  def test_name(self):
    """Test names."""
    diag = tf.constant([[1., 2.], [3., 4.]])
    diag_linop = linear_operator.LinearOperatorDiag(diag, rank=2)
    self.assertEqual("LinearOperatorDiag", diag_linop.name)
