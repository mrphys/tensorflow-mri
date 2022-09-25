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
"""Tests for module `linear_operator_addition`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_addition
from tensorflow_mri.python.linalg import linear_operator_full_matrix
from tensorflow_mri.python.linalg import linear_operator_test_util
from tensorflow_mri.python.util import test_util


rng = np.random.RandomState(0)


class SquareLinearOperatorAdditionTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    tf.config.experimental.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    self.tf32_keep_ = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)

  def operator_and_matrix(self, build_info, dtype, use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    shape = list(build_info.shape)

    # Either 1 or 2 matrices, depending.
    num_operators = rng.randint(low=1, high=3)
    if ensure_self_adjoint_and_pd:
      # The random PD matrices are also symmetric. Here we are computing
      # A @ A ... @ A. Since A is symmetric and PD, so are any powers of it.
      matrices = [
          linear_operator_test_util.random_positive_definite_matrix(
              shape, dtype, force_well_conditioned=True)] * num_operators
    else:
      matrices = [
          linear_operator_test_util.random_positive_definite_matrix(
              shape, dtype, force_well_conditioned=True)
          for _ in range(num_operators)
      ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          tf.compat.v1.placeholder_with_default(
              matrix, shape=None) for matrix in matrices]

    operator = linear_operator_addition.LinearOperatorAddition(
        [linear_operator_full_matrix.LinearOperatorFullMatrix(l)
         for l in lin_op_matrices],
        is_positive_definite=True if ensure_self_adjoint_and_pd else None,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_square=True)

    matmul_order_list = list(reversed(matrices))
    mat = matmul_order_list[0]
    for other_mat in matmul_order_list[1:]:
      mat = tf.math.add(other_mat, mat)

    return operator, mat

  @test_util.run_deprecated_v1
  def test_is_x_flags(self):
    expected = {
        'is_non_singular': {
            (True, True): None,
            (True, False): None,
            (True, None): None,
            (False, False): None,
            (False, None): None,
            (None, None): None
        },
        'is_self_adjoint': {
            (True, True): True,
            (True, False): False,
            (True, None): None,
            (False, False): None,
            (False, None): None,
            (None, None): None
        },
        'is_positive_definite': {
            (True, True): True,
            (True, False): None,
            (True, None): None,
            (False, False): None,
            (False, None): None,
            (None, None): None
        },
        'is_square': {
            (True, True): True,
            # (True, False): None,
            (True, None): True,
            (False, False): False,
            (False, None): False,
            (None, None): None
        }
    }
    for name, combinations in expected.items():
      for (flag1, flag2), value in combinations.items():
        with self.subTest(name=name, flag1=flag1, flag2=flag2):
          matrix = tf.compat.v1.placeholder(tf.float32)
          operator1 = linear_operator_full_matrix.LinearOperatorFullMatrix(
              matrix, **{name: flag1})
          operator2 = linear_operator_full_matrix.LinearOperatorFullMatrix(
              matrix, **{name: flag2})
          operator = linear_operator_addition.LinearOperatorAddition(
              [operator1, operator2])

          self.assertIs(getattr(operator, name), value)

  def test_name(self):
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linear_operator_full_matrix.LinearOperatorFullMatrix(
        matrix, name="left")
    operator_2 = linear_operator_full_matrix.LinearOperatorFullMatrix(
        matrix, name="right")

    operator = linear_operator_addition.LinearOperatorAddition(
        [operator_1, operator_2])

    self.assertEqual("left_p_right", operator.name)

  def test_different_dtypes_raises(self):
    operators = [
        linear_operator_full_matrix.LinearOperatorFullMatrix(
            rng.rand(2, 3, 3)),
        linear_operator_full_matrix.LinearOperatorFullMatrix(
            rng.rand(2, 3, 3).astype(np.float32))
    ]
    with self.assertRaisesRegex(TypeError, "same dtype"):
      linear_operator_addition.LinearOperatorAddition(operators)

  def test_empty_operators_raises(self):
    with self.assertRaisesRegex(ValueError, "non-empty"):
      linear_operator_addition.LinearOperatorAddition([])


class NonSquareLinearOperatorAdditionTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    tf.config.experimental.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    self.tf32_keep_ = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
    shape = list(build_info.shape)

    # Ensure that the matrices are well-conditioned by generating
    # random matrices whose singular values are close to 1.
    # The reason to do this is because cond(AB) <= cond(A) * cond(B).
    # By ensuring that each factor has condition number close to 1, we ensure
    # that the condition number of the product isn't too far away from 1.
    def generate_well_conditioned(shape, dtype):
      m, n = shape[-2], shape[-1]
      min_dim = min(m, n)
      # Generate singular values that are close to 1.
      d = linear_operator_test_util.random_normal(
          shape[:-2] + [min_dim],
          mean=1.,
          stddev=0.1,
          dtype=dtype)
      zeros = tf.compat.v1.zeros(shape=shape[:-2] + [m, n], dtype=dtype)
      d = tf.linalg.set_diag(zeros, d)
      u, _ = tf.linalg.qr(linear_operator_test_util.random_normal(
          shape[:-2] + [m, m], dtype=dtype))

      v, _ = tf.linalg.qr(linear_operator_test_util.random_normal(
          shape[:-2] + [n, n], dtype=dtype))
      return tf.matmul(u, tf.matmul(d, v))

    matrices = [
        generate_well_conditioned(shape, dtype=dtype),
        generate_well_conditioned(shape, dtype=dtype),
    ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          tf.compat.v1.placeholder_with_default(
              matrix, shape=None) for matrix in matrices]

    operator = linear_operator_addition.LinearOperatorAddition(
        [linear_operator_full_matrix.LinearOperatorFullMatrix(l)
         for l in lin_op_matrices])

    matmul_order_list = list(reversed(matrices))
    mat = matmul_order_list[0]
    for other_mat in matmul_order_list[1:]:
      mat = tf.math.add(other_mat, mat)

    return operator, mat

  @test_util.run_deprecated_v1
  def test_different_shapes_raises_static(self):
    operators = [
        linear_operator_full_matrix.LinearOperatorFullMatrix(rng.rand(2, 4, 5)),
        linear_operator_full_matrix.LinearOperatorFullMatrix(rng.rand(2, 3, 4))
    ]
    with self.assertRaisesRegex(ValueError, "same shape"):
      linear_operator_addition.LinearOperatorAddition(operators)

  @test_util.run_deprecated_v1
  def test_static_shapes(self):
    operators = [
        linear_operator_full_matrix.LinearOperatorFullMatrix(rng.rand(2, 3, 4)),
        linear_operator_full_matrix.LinearOperatorFullMatrix(rng.rand(2, 3, 4))
    ]
    operator = linear_operator_addition.LinearOperatorAddition(operators)
    self.assertAllEqual((2, 3, 4), operator.shape)

  @test_util.run_deprecated_v1
  def test_shape_tensors_when_statically_available(self):
    operators = [
        linear_operator_full_matrix.LinearOperatorFullMatrix(rng.rand(2, 3, 4)),
        linear_operator_full_matrix.LinearOperatorFullMatrix(rng.rand(2, 3, 4))
    ]
    operator = linear_operator_addition.LinearOperatorAddition(operators)
    with self.cached_session():
      self.assertAllEqual((2, 3, 4), operator.shape_tensor())

  @test_util.run_deprecated_v1
  def test_shape_tensors_when_only_dynamically_available(self):
    mat_1 = rng.rand(1, 2, 3, 4)
    mat_2 = rng.rand(1, 2, 3, 4)
    mat_ph_1 = tf.compat.v1.placeholder(tf.float64)
    mat_ph_2 = tf.compat.v1.placeholder(tf.float64)
    feed_dict = {mat_ph_1: mat_1, mat_ph_2: mat_2}

    operators = [
        linear_operator_full_matrix.LinearOperatorFullMatrix(mat_ph_1),
        linear_operator_full_matrix.LinearOperatorFullMatrix(mat_ph_2)
    ]
    operator = linear_operator_addition.LinearOperatorAddition(operators)
    with self.cached_session():
      self.assertAllEqual(
          (1, 2, 3, 4), operator.shape_tensor().eval(feed_dict=feed_dict))


linear_operator_test_util.add_tests(SquareLinearOperatorAdditionTest)
linear_operator_test_util.add_tests(NonSquareLinearOperatorAdditionTest)


if __name__ == "__main__":
  tf.test.main()
