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
"""Tests for module `linear_operator_coils`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import functools

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_coils
from tensorflow_mri.python.linalg import linear_operator_test_util
from tensorflow_mri.python.util import test_util


rng = np.random.RandomState(2016)


class OperatorShapesInfoCoils():
  def __init__(self, image_shape, num_coils, batch_shape):
    self.image_shape = image_shape
    self.num_coils = num_coils
    self.batch_shape = batch_shape

  @property
  def shape(self):
    n = functools.reduce(lambda a, b: a * b, self.image_shape)
    m = self.num_coils * n
    return self.batch_shape + (m, n)

  @property
  def dimension(self):
    return len(self.image_shape)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorCoilsTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @staticmethod
  def operator_shapes_infos():
    shapes_info = OperatorShapesInfoCoils
    return [
        shapes_info((2, 2), 3, ()),
        shapes_info((2, 4), 4, (3,)),
        shapes_info((4, 2), 3, (1, 2)),
        shapes_info((2, 2), 4, ()),
        shapes_info((2, 2, 2), 4, ()),
        shapes_info((4, 2, 2), 2, (2,))
        # TODO(jmontalt): odd shapes fail tests, investigate
        # shapes_info((2, 3), 5, (2,)),
        # shapes_info((3, 2), 7, ())
    ]

  @staticmethod
  def dtypes_to_test():
    return [tf.complex64, tf.complex128]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
    del use_placeholder

    batch_shape = build_info.batch_shape
    num_coils = build_info.num_coils
    image_shape = build_info.image_shape

    maps = tf.dtypes.complex(
        tf.random.normal(
            shape=batch_shape + (num_coils,) + image_shape,
            dtype=dtype.real_dtype),
        tf.random.normal(
            shape=batch_shape + (num_coils,) + image_shape,
            dtype=dtype.real_dtype)
    )

    operator = linear_operator_coils.LinearOperatorCoils(
        maps=maps, batch_dims=len(batch_shape))

    matrix = linear_operator_coils.coils_matrix(
        maps=maps, batch_dims=len(batch_shape))

    return operator, matrix

  def test_1d_maps_raises_static(self):
    with self.assertRaisesRegex(ValueError, "must be at least 2-D"):
      linear_operator_coils.LinearOperatorCoils(
          maps=np.ones((4,)).astype(np.complex64))

    with self.assertRaisesRegex(ValueError, "must be at least 2-D"):
      linear_operator_coils.LinearOperatorCoils(
          maps=np.ones((3, 4, 4)).astype(np.complex64),
          batch_dims=2)

    linear_operator_coils.LinearOperatorCoils(
          maps=np.ones((3, 4, 4)).astype(np.complex64),
          batch_dims=1)  # should not raise

  def test_non_complex_maps_raises_static(self):
    with self.assertRaisesRegex(TypeError, "must be complex"):
      linear_operator_coils.LinearOperatorCoils(
          maps=np.ones((3, 4, 4)).astype(np.float32))

  def test_unknown_rank_domain_shape_raises_static(self):
    if tf.executing_eagerly():
      return
    with self.cached_session():
      maps = tf.compat.v1.placeholder_with_default(
          np.ones((3, 4, 4)).astype(np.complex64), shape=None)
      with self.assertRaisesRegex(ValueError, "must have known static rank"):
        operator = linear_operator_coils.LinearOperatorCoils(maps=maps)
        self.evaluate(operator.to_dense())

  def test_non_integer_batch_dims_raises_static(self):
    with self.assertRaisesRegex(TypeError, "must be an int"):
      linear_operator_coils.LinearOperatorCoils(
          maps=np.ones((3, 4, 4)).astype(np.complex64), batch_dims=1.)

  def test_negative_batch_dims_raises_static(self):
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      linear_operator_coils.LinearOperatorCoils(
          maps=np.ones((3, 4, 4)).astype(np.complex64), batch_dims=-1)

  def test_is_x_flags(self):
    operator = linear_operator_coils.LinearOperatorCoils(
          maps=np.ones((3, 4, 4)).astype(np.complex64))
    self.assertFalse(operator.is_self_adjoint)

  def test_solve_raises(self):
    operator = linear_operator_coils.LinearOperatorCoils(
          maps=np.ones((1, 4, 4)).astype(np.complex64), is_square=True)
    with self.assertRaisesRegex(ValueError, "not invertible.*lstsq"):
      operator.solve(tf.ones([16, 1], dtype=tf.complex64))

  def test_inverse_raises(self):
    operator = linear_operator_coils.LinearOperatorCoils(
          maps=np.ones((1, 4, 4)).astype(np.complex64), is_square=True)
    with self.assertRaisesRegex(ValueError, "not invertible.*pseudo_inverse"):
      operator.inverse()

  def test_convert_variables_to_tensors(self):
    maps = tf.Variable(np.ones((3, 4, 4)).astype(np.complex64))
    operator = linear_operator_coils.LinearOperatorCoils(maps=maps)
    with self.cached_session() as sess:
      sess.run([maps.initializer])
      self.check_convert_variables_to_tensors(operator)


linear_operator_test_util.add_tests(LinearOperatorCoilsTest)


if __name__ == "__main__":
  tf.test.main()
