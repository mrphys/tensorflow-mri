# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Tests for module `linear_operator_mask`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import functools

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_mask
from tensorflow_mri.python.linalg import linear_operator_test_util
from tensorflow_mri.python.util import test_util


rng = np.random.RandomState(2016)


class OperatorShapesInfoCoils():
  def __init__(self, image_shape, batch_shape):
    self.image_shape = image_shape
    self.batch_shape = batch_shape

  @property
  def shape(self):
    n = functools.reduce(lambda a, b: a * b, self.image_shape)
    return self.batch_shape + (n, n)

  @property
  def dimension(self):
    return len(self.image_shape)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorMaskMultiplyTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @staticmethod
  def operator_shapes_infos():
    shapes_info = OperatorShapesInfoCoils
    return [
        shapes_info((2, 2), ()),
        shapes_info((2, 4), (3,)),
        shapes_info((4, 2), (1, 2)),
        shapes_info((2, 3), ()),
        shapes_info((2, 2, 2), ()),
        shapes_info((4, 2, 2), (2,))
        # TODO(jmontalt): odd shapes fail tests, investigate
        # shapes_info((2, 3), 5, (2,)),
        # shapes_info((3, 2), 7, ())
    ]

  @staticmethod
  def dtypes_to_test():
    return [tf.float32, tf.float64, tf.complex64, tf.complex128]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
    del use_placeholder

    batch_shape = build_info.batch_shape
    image_shape = build_info.image_shape

    mask = tf.random.uniform(shape=batch_shape + image_shape) > 0.5

    operator = linear_operator_mask.LinearOperatorMask(
        mask=mask, batch_dims=len(batch_shape), dtype=dtype,
        algorithm='multiply')

    matrix = linear_operator_mask.mask_matrix(
        mask=mask, batch_dims=len(batch_shape), dtype=dtype)

    return operator, matrix

  def test_0d_mask_raises_static(self):
    with self.assertRaisesRegex(ValueError, "must be at least 1-D"):
      linear_operator_mask.LinearOperatorMask(
          mask=np.ones(()).astype(np.bool_))

    with self.assertRaisesRegex(ValueError, "must be at least 1-D"):
      linear_operator_mask.LinearOperatorMask(
          mask=np.ones((4, 4)).astype(np.bool_),
          batch_dims=2)

    linear_operator_mask.LinearOperatorMask(
          mask=np.ones((4, 4)).astype(np.bool_),
          batch_dims=1)  # should not raise

  def test_non_bool_mask_raises_static(self):
    with self.assertRaisesRegex(TypeError, "must be boolean"):
      linear_operator_mask.LinearOperatorMask(
          mask=np.ones((4, 4)).astype(np.float32))

  def test_unknown_rank_mask_raises_static(self):
    if tf.executing_eagerly():
      return
    with self.cached_session():
      mask = tf.compat.v1.placeholder_with_default(
          np.ones((3, 4, 4)).astype(np.bool_), shape=None)
      with self.assertRaisesRegex(ValueError, "must have known static rank"):
        operator = linear_operator_mask.LinearOperatorMask(mask=mask)
        self.evaluate(operator.to_dense())

  def test_non_integer_batch_dims_raises_static(self):
    with self.assertRaisesRegex(TypeError, "must be an int"):
      linear_operator_mask.LinearOperatorMask(
          mask=np.ones((3, 4, 4)).astype(np.bool_), batch_dims=1.)

  def test_negative_batch_dims_raises_static(self):
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      linear_operator_mask.LinearOperatorMask(
          mask=np.ones((3, 4, 4)).astype(np.bool_), batch_dims=-1)

  def test_is_x_flags(self):
    operator = linear_operator_mask.LinearOperatorMask(
          mask=np.ones((3, 4, 4)).astype(np.bool_))
    self.assertTrue(operator.is_self_adjoint)
    self.assertFalse(operator.is_non_singular)
    self.assertTrue(operator.is_square)

  def test_solve_raises(self):
    operator = linear_operator_mask.LinearOperatorMask(
          mask=np.ones((1, 4, 4)).astype(np.bool_), is_square=True)
    with self.assertRaisesRegex(NotImplementedError, "singular"):
      operator.solve(tf.ones([16, 1], dtype=tf.bool))

  def test_inverse_raises(self):
    operator = linear_operator_mask.LinearOperatorMask(
          mask=np.ones((1, 4, 4)).astype(np.bool_), is_square=True)
    with self.assertRaisesRegex(ValueError, "singular"):
      operator.inverse()

  def test_adjoint_type(self):
    operator = linear_operator_mask.LinearOperatorMask(
          mask=np.ones((3, 4)).astype(np.bool_))
    self.assertIsInstance(
        operator.adjoint(), linear_operator_mask.LinearOperatorMask)

  def test_convert_variables_to_tensors(self):
    mask = tf.Variable(np.ones((3, 4, 4)).astype(np.bool_))
    operator = linear_operator_mask.LinearOperatorMask(mask=mask)
    with self.cached_session() as sess:
      sess.run([mask.initializer])
      self.check_convert_variables_to_tensors(operator)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorMaskMultiplexTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @staticmethod
  def operator_shapes_infos():
    shapes_info = OperatorShapesInfoCoils
    return [
        shapes_info((2, 2), ()),
        shapes_info((2, 4), (3,)),
        shapes_info((4, 2), (1, 2)),
        shapes_info((2, 3), ()),
        shapes_info((2, 2, 2), ()),
        shapes_info((4, 2, 2), (2,))
        # TODO(jmontalt): odd shapes fail tests, investigate
        # shapes_info((2, 3), 5, (2,)),
        # shapes_info((3, 2), 7, ())
    ]

  @staticmethod
  def dtypes_to_test():
    return [tf.float32, tf.float64, tf.complex64, tf.complex128]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
    del use_placeholder

    batch_shape = build_info.batch_shape
    image_shape = build_info.image_shape

    mask = tf.random.uniform(shape=batch_shape + image_shape) > 0.5

    operator = linear_operator_mask.LinearOperatorMask(
        mask=mask, batch_dims=len(batch_shape), dtype=dtype,
        algorithm='multiplex')

    matrix = linear_operator_mask.mask_matrix(
        mask=mask, batch_dims=len(batch_shape), dtype=dtype)

    return operator, matrix


linear_operator_test_util.add_tests(LinearOperatorMaskMultiplyTest)
linear_operator_test_util.add_tests(LinearOperatorMaskMultiplexTest)


if __name__ == "__main__":
  tf.test.main()
