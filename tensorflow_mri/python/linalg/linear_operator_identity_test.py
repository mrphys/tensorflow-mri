# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for module `linear_operator_identity`.

Adapted from:
  tensorflow/python/kernel_tests/linalg/linear_operator_identity_test.py
"""
# pylint: disable=missing-function-docstring

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.ops.linalg import linear_operator_test_util

from tensorflow_mri.python.linalg import linear_operator_identity


rng = np.random.RandomState(2016)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorIdentityTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    tf.config.experimental.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    self.tf32_keep_ = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)

  @staticmethod
  def dtypes_to_test():
    # TODO(langmore) Test tf.float16 once tf.linalg.solve works in
    # 16bit.
    return [tf.float32, tf.float64, tf.complex64, tf.complex128]

  @staticmethod
  def optional_tests():
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # Identity matrix is already Hermitian Positive Definite.
    del ensure_self_adjoint_and_pd

    shape = list(build_info.shape)
    assert shape[-1] == shape[-2]

    batch_shape = shape[:-2]
    num_rows = shape[-1]

    operator = linear_operator_identity.LinearOperatorIdentity(
        num_rows, batch_shape=batch_shape, dtype=dtype)
    mat = tf.linalg.eye(num_rows, batch_shape=batch_shape, dtype=dtype)

    return operator, mat

  def test_to_dense(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2])
      self.assertAllClose(np.eye(2), self.evaluate(operator.to_dense()))

      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2, 3])
      self.assertAllClose(np.eye(6), self.evaluate(operator.to_dense()))

  def test_shapes(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2, 3], batch_shape=[4, 5])
      self.assertAllEqual([2, 3], operator.domain_shape)
      self.assertAllEqual([2, 3], operator.range_shape)
      self.assertAllEqual([4, 5], operator.batch_shape)
      self.assertAllEqual([4, 5, 6, 6], operator.shape)
      self.assertAllEqual(6, operator.domain_dimension)
      self.assertAllEqual(6, operator.range_dimension)
      self.assertAllEqual([2, 3], self.evaluate(operator.domain_shape_tensor()))
      self.assertAllEqual([2, 3], self.evaluate(operator.range_shape_tensor()))
      self.assertAllEqual([4, 5], self.evaluate(operator.batch_shape_tensor()))
      self.assertAllEqual([4, 5, 6, 6], self.evaluate(operator.shape_tensor()))
      self.assertAllEqual(6, self.evaluate(operator.domain_dimension_tensor()))
      self.assertAllEqual(6, self.evaluate(operator.range_dimension_tensor()))

  def test_shapes_dynamic(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session():
      # Given this x and LinearOperatorIdentity shape of (2, 1, 6, 6), the
      # broadcast shape of operator and 'x' is (2, 2, 3, 4)
      domain_shape = tf.compat.v1.placeholder_with_default((2, 3), shape=None)
      batch_shape = tf.compat.v1.placeholder_with_default((2, 1), shape=None)

      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape, batch_shape=batch_shape, dtype=tf.float64)

      self.assertAllEqual([2, 3], self.evaluate(operator.domain_shape_tensor()))
      self.assertAllEqual([2, 1], self.evaluate(operator.batch_shape_tensor()))
      self.assertAllEqual([2, 1, 6, 6], self.evaluate(operator.shape_tensor()))
      self.assertAllEqual(6, self.evaluate(operator.domain_dimension_tensor()))
      self.assertAllEqual(6, self.evaluate(operator.range_dimension_tensor()))

  def test_assert_positive_definite(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2])
      self.evaluate(operator.assert_positive_definite())  # Should not fail

  def test_assert_non_singular(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2])
      self.evaluate(operator.assert_non_singular())  # Should not fail

  def test_assert_self_adjoint(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2])
      self.evaluate(operator.assert_self_adjoint())  # Should not fail

  # TODO(jmontalt).
  # def test_float16_matmul(self):
  #   # float16 cannot be tested by base test class because tf.linalg.solve does
  #   # not work with float16.
  #   with self.cached_session():
  #     operator = linear_operator_identity.LinearOperatorIdentity(
  #         domain_shape=[2], dtype=tf.float16)
  #     x = rng.randn(2, 3).astype(np.float16)
  #     y = operator.matmul(x)
  #     self.assertAllClose(x, self.evaluate(y))

  def test_non_1d_domain_shape_raises_static(self):
    with self.assertRaisesRegex(
        ValueError, "domain_shape must be a 1-D Tensor"):
      linear_operator_identity.LinearOperatorIdentity(domain_shape=2)

  def test_non_integer_domain_shape_raises_static(self):
    with self.assertRaisesRegex(
        TypeError, "domain_shape must be integer"):
      linear_operator_identity.LinearOperatorIdentity(domain_shape=[2.])

  def test_negative_domain_shape_raises_static(self):
    with self.assertRaisesRegex(
        ValueError, "domain_shape must be non-negative"):
      linear_operator_identity.LinearOperatorIdentity(domain_shape=[-2])

  def test_non_1d_batch_shape_raises_static(self):
    with self.assertRaisesRegex(
        ValueError, "batch_shape must be a 1-D Tensor"):
      linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2], batch_shape=2)

  def test_non_integer_batch_shape_raises_static(self):
    with self.assertRaisesRegex(TypeError, "must be integer"):
      linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2], batch_shape=[2.])

  def test_negative_batch_shape_raises_static(self):
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2], batch_shape=[-2])

  def test_non_1d_domain_shape_raises_dynamic(self):
    with self.cached_session():
      domain_shape = tf.compat.v1.placeholder_with_default(2, shape=None)
      with self.assertRaisesError("must be a 1-D Tensor"):
        operator = linear_operator_identity.LinearOperatorIdentity(
            domain_shape, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_negative_domain_shape_raises_dynamic(self):
    with self.cached_session():
      domain_shape = tf.compat.v1.placeholder_with_default([-2], shape=None)
      with self.assertRaisesError("must be non-negative"):
        operator = linear_operator_identity.LinearOperatorIdentity(
            domain_shape, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_non_1d_batch_shape_raises_dynamic(self):
    with self.cached_session():
      batch_shape = tf.compat.v1.placeholder_with_default(2, shape=None)
      with self.assertRaisesError("must be a 1-D"):
        operator = linear_operator_identity.LinearOperatorIdentity(
            domain_shape=[2], batch_shape=batch_shape,
            assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_negative_batch_shape_raises_dynamic(self):
    with self.cached_session():
      batch_shape = tf.compat.v1.placeholder_with_default([-2], shape=None)
      with self.assertRaisesError("must be non-negative"):
        operator = linear_operator_identity.LinearOperatorIdentity(
            domain_shape=[2], batch_shape=batch_shape,
            assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_wrong_matrix_dimensions_raises_static(self):
    operator = linear_operator_identity.LinearOperatorIdentity(domain_shape=[2])
    x = rng.randn(3, 3).astype(np.float32)
    with self.assertRaisesRegex(ValueError, "Dimensions.*not compatible"):
      operator.matmul(x)

  # TODO(jmontalt).
  # def test_wrong_matrix_dimensions_raises_dynamic(self):
  #   domain_shape = tf.compat.v1.placeholder_with_default([2], shape=None)
  #   x = tf.compat.v1.placeholder_with_default(
  #       rng.rand(3, 3).astype(np.float32), shape=None)

  #   with self.cached_session():
  #     with self.assertRaisesError("Dimensions.*not.compatible"):
  #       operator = linear_operator_identity.LinearOperatorIdentity(
  #           domain_shape, assert_proper_shapes=True)
  #       self.evaluate(operator.matmul(x))

  def test_default_batch_shape_broadcasts_with_everything_static(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session():
      x = tf.random.normal(shape=(1, 2, 3, 4))
      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[3], dtype=x.dtype)

      operator_matmul = operator.matmul(x)
      expected = x

      self.assertAllEqual(operator_matmul.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

  def test_default_batch_shape_broadcasts_with_everything_dynamic(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session():
      x = tf.compat.v1.placeholder_with_default(
          rng.randn(1, 2, 3, 4), shape=None)
      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[3], dtype=x.dtype)

      operator_matmul = operator.matmul(x)
      expected = x

      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

  def test_broadcast_matmul_static_shapes(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session():
      # Given this x and LinearOperatorIdentity shape of (2, 1, 6, 6), the
      # broadcast shape of operator and 'x' is (2, 2, 6, 4)
      x = tf.random.normal(shape=(1, 2, 6, 4))
      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape=(2, 3), batch_shape=(2, 1), dtype=x.dtype)

      # Batch matrix of zeros with the broadcast shape of x and operator.
      zeros = tf.zeros(shape=(2, 2, 6, 4), dtype=x.dtype)

      # Expected result of matmul and solve.
      expected = x + zeros

      operator_matmul = operator.matmul(x)
      self.assertAllEqual(operator_matmul.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

  def test_broadcast_matmul_dynamic_shapes(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session():
      # Given this x and LinearOperatorIdentity shape of (2, 1, 6, 6), the
      # broadcast shape of operator and 'x' is (2, 2, 3, 4)
      x = tf.compat.v1.placeholder_with_default(
          rng.rand(1, 2, 6, 4), shape=None)
      domain_shape = tf.compat.v1.placeholder_with_default((2, 3), shape=None)
      batch_shape = tf.compat.v1.placeholder_with_default((2, 1), shape=None)

      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape, batch_shape=batch_shape, dtype=tf.float64)

      # Batch matrix of zeros with the broadcast shape of x and operator.
      zeros = tf.zeros(shape=(2, 2, 6, 4), dtype=x.dtype)

      # Expected result of matmul and solve.
      expected = x + zeros

      operator_matmul = operator.matmul(x)
      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

  def test_is_x_flags(self):
    # The is_x flags are by default all True.
    operator = linear_operator_identity.LinearOperatorIdentity(domain_shape=[2])
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)

    # Any of them False raises because the identity is always self-adjoint etc..
    with self.assertRaisesRegex(ValueError, "is always non-singular"):
      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2],
          is_non_singular=None,
      )

  def test_identity_adjoint_type(self):
    operator = linear_operator_identity.LinearOperatorIdentity(
        domain_shape=[2], is_non_singular=True)
    self.assertIsInstance(
        operator.adjoint(), linear_operator_identity.LinearOperatorIdentity)

  # TODO(jmontalt).
  # def test_identity_cholesky_type(self):
  #   operator = linear_operator_identity.LinearOperatorIdentity(
  #       domain_shape=[2],
  #       is_positive_definite=True,
  #       is_self_adjoint=True,
  #   )
  #   self.assertIsInstance(
  #       operator.cholesky(), linear_operator_identity.LinearOperatorIdentity)

  def test_identity_inverse_type(self):
    operator = linear_operator_identity.LinearOperatorIdentity(
        domain_shape=[2], is_non_singular=True)
    self.assertIsInstance(
        operator.inverse(), linear_operator_identity.LinearOperatorIdentity)

  def test_ref_type_shape_args_raises(self):
    with self.assertRaisesRegex(TypeError, "domain_shape.*reference"):
      linear_operator_identity.LinearOperatorIdentity(
          domain_shape=tf.Variable([2]))

    with self.assertRaisesRegex(TypeError, "batch_shape.*reference"):
      linear_operator_identity.LinearOperatorIdentity(
          domain_shape=[2], batch_shape=tf.Variable([3]))


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorScaledIdentityTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    tf.config.experimental.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    self.tf32_keep_ = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)

  @staticmethod
  def dtypes_to_test():
    # TODO(langmore) Test tf.float16 once tf.linalg.solve works in
    # 16bit.
    return [tf.float32, tf.float64, tf.complex64, tf.complex128]

  @staticmethod
  def optional_tests():
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):

    shape = list(build_info.shape)
    assert shape[-1] == shape[-2]

    batch_shape = shape[:-2]
    num_rows = shape[-1]

    # Uniform values that are at least length 1 from the origin.  Allows the
    # operator to be well conditioned.
    # Shape batch_shape
    multiplier = linear_operator_test_util.random_sign_uniform(
        shape=batch_shape, minval=1., maxval=2., dtype=dtype)

    if ensure_self_adjoint_and_pd:
      # Abs on complex64 will result in a float32, so we cast back up.
      multiplier = tf.cast(tf.abs(multiplier), dtype=dtype)

    # Nothing to feed since LinearOperatorScaledIdentity takes no Tensor args.
    lin_op_multiplier = multiplier

    if use_placeholder:
      lin_op_multiplier = tf.compat.v1.placeholder_with_default(
          multiplier, shape=None)

    operator = linear_operator_identity.LinearOperatorScaledIdentity(
        num_rows,
        lin_op_multiplier,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None)

    multiplier_matrix = tf.expand_dims(
        tf.expand_dims(multiplier, -1), -1)
    matrix = multiplier_matrix * tf.linalg.eye(
        num_rows, batch_shape=batch_shape, dtype=dtype)

    return operator, matrix

  def test_to_dense(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2], multiplier=1.0)
      self.assertAllClose(np.eye(2), self.evaluate(operator.to_dense()))

      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2, 3], multiplier=2.0)
      self.assertAllClose(2.0 * np.eye(6), self.evaluate(operator.to_dense()))

  def test_shapes(self):
    with self.cached_session():
      multiplier = tf.ones([4, 5])
      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2, 3], multiplier=multiplier)
      self.assertAllEqual([2, 3], operator.domain_shape)
      self.assertAllEqual([2, 3], operator.range_shape)
      self.assertAllEqual([4, 5], operator.batch_shape)
      self.assertAllEqual([4, 5, 6, 6], operator.shape)
      self.assertAllEqual(6, operator.domain_dimension)
      self.assertAllEqual(6, operator.range_dimension)
      self.assertAllEqual([2, 3], self.evaluate(operator.domain_shape_tensor()))
      self.assertAllEqual([2, 3], self.evaluate(operator.range_shape_tensor()))
      self.assertAllEqual([4, 5], self.evaluate(operator.batch_shape_tensor()))
      self.assertAllEqual([4, 5, 6, 6], self.evaluate(operator.shape_tensor()))
      self.assertAllEqual(6, self.evaluate(operator.domain_dimension_tensor()))
      self.assertAllEqual(6, self.evaluate(operator.range_dimension_tensor()))

  def test_shapes_dynamic(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session():
      # Given this x and LinearOperatorIdentity shape of (2, 1, 6, 6), the
      # broadcast shape of operator and 'x' is (2, 2, 3, 4)
      domain_shape = tf.compat.v1.placeholder_with_default((2, 3), shape=None)
      batch_shape = tf.compat.v1.placeholder_with_default((2, 1), shape=None)

      operator = linear_operator_identity.LinearOperatorIdentity(
          domain_shape, batch_shape=batch_shape, dtype=tf.float64)

      self.assertAllEqual([2, 3], self.evaluate(operator.domain_shape_tensor()))
      self.assertAllEqual([2, 1], self.evaluate(operator.batch_shape_tensor()))
      self.assertAllEqual([2, 1, 6, 6], self.evaluate(operator.shape_tensor()))
      self.assertAllEqual(6, self.evaluate(operator.domain_dimension_tensor()))
      self.assertAllEqual(6, self.evaluate(operator.range_dimension_tensor()))

  def test_assert_positive_definite_does_not_raise_when_positive(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2], multiplier=1.)
      self.evaluate(operator.assert_positive_definite())  # Should not fail

  def test_assert_positive_definite_raises_when_negative(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2], multiplier=-1.)
      with self.assertRaisesOpError("not positive definite"):
        self.evaluate(operator.assert_positive_definite())

  def test_assert_non_singular_does_not_raise_when_non_singular(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2], multiplier=[1., 2., 3.])
      self.evaluate(operator.assert_non_singular())  # Should not fail

  def test_assert_non_singular_raises_when_singular(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2], multiplier=[1., 2., 0.])
      with self.assertRaisesOpError("was singular"):
        self.evaluate(operator.assert_non_singular())

  def test_assert_self_adjoint_does_not_raise_when_self_adjoint(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2], multiplier=[1. + 0J])
      self.evaluate(operator.assert_self_adjoint())  # Should not fail

  def test_assert_self_adjoint_raises_when_not_self_adjoint(self):
    with self.cached_session():
      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2], multiplier=[1. + 1J])
      with self.assertRaisesOpError("not self-adjoint"):
        self.evaluate(operator.assert_self_adjoint())

#   def test_float16_matmul(self):
#     # float16 cannot be tested by base test class because tf.linalg.solve does
#     # not work with float16.
#     with self.cached_session():
#       multiplier = rng.rand(3).astype(np.float16)
#       operator = linear_operator_identity.LinearOperatorScaledIdentity(
#           domain_shape=[2], multiplier=multiplier)
#       x = rng.randn(2, 3).astype(np.float16)
#       y = operator.matmul(x)
#       self.assertAllClose(multiplier[..., None, None] * x, self.evaluate(y))

  def test_non_1d_domain_shape_raises_static(self):
    # Many "test_...num_rows" tests are performed in LinearOperatorIdentity.
    with self.assertRaisesRegex(ValueError, "must be a 1-D Tensor"):
      linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=2, multiplier=123.)

  def test_wrong_matrix_dimensions_raises_static(self):
    operator = linear_operator_identity.LinearOperatorScaledIdentity(
        domain_shape=[2], multiplier=2.2)
    x = rng.randn(3, 3).astype(np.float32)
    with self.assertRaisesRegex(ValueError, "Dimensions.*not compatible"):
      operator.matmul(x)

  # TODO(jmontalt): add assertions to `transform` / `matmul`.
  # def test_wrong_matrix_dimensions_raises_dynamic(self):
  #   num_rows = tf.compat.v1.placeholder_with_default(2, shape=None)
  #   x = tf.compat.v1.placeholder_with_default(
  #       rng.rand(3, 3).astype(np.float32), shape=None)

  #   with self.cached_session():
  #     with self.assertRaisesError("Dimensions.*not.compatible"):
  #       operator = linear_operator_identity.LinearOperatorScaledIdentity(
  #           num_rows,
  #           multiplier=[1., 2],
  #           assert_proper_shapes=True)
  #       self.evaluate(operator.matmul(x))

  def test_broadcast_matmul_and_solve(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session():
      # Given this x and LinearOperatorScaledIdentity shape of (2, 1, 6, 6), the
      # broadcast shape of operator and 'x' is (2, 2, 6, 4)
      x = tf.random.normal(shape=(1, 2, 6, 4))

      # operator is 2.2 * identity (with a batch shape).
      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2, 3], multiplier=2.2 * tf.ones((2, 1)))

      # Batch matrix of zeros with the broadcast shape of x and operator.
      zeros = tf.zeros(shape=(2, 2, 6, 4), dtype=x.dtype)

      # Test matmul
      expected = x * 2.2 + zeros
      operator_matmul = operator.matmul(x)
      self.assertAllEqual(operator_matmul.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

      # Test solve
      expected = x / 2.2 + zeros
      operator_solve = operator.solve(x)
      self.assertAllEqual(operator_solve.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_solve, expected]))

  def test_broadcast_matmul_and_solve_scalar_scale_multiplier(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.cached_session():
      # Given this x and LinearOperatorScaledIdentity shape of (6, 6), the
      # broadcast shape of operator and 'x' is (1, 2, 6, 4), which is the same
      # shape as x.
      x = tf.random.normal(shape=(1, 2, 6, 4))

      # operator is 2.2 * identity (with a batch shape).
      operator = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=[2, 3], multiplier=2.2)

      # Test matmul
      expected = x * 2.2
      operator_matmul = operator.matmul(x)
      self.assertAllEqual(operator_matmul.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_matmul, expected]))

      # Test solve
      expected = x / 2.2
      operator_solve = operator.solve(x)
      self.assertAllEqual(operator_solve.shape, expected.shape)
      self.assertAllClose(*self.evaluate([operator_solve, expected]))

  def test_is_x_flags(self):
    operator = linear_operator_identity.LinearOperatorScaledIdentity(
        domain_shape=[2], multiplier=1.,
        is_positive_definite=False, is_non_singular=True)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)  # Auto-set due to real multiplier

  # TODO(jmontalt).
  # def test_identity_matmul(self):
  #   operator1 = linear_operator_identity.LinearOperatorIdentity(domain_shape=[2])
  #   operator2 = linear_operator_identity.LinearOperatorScaledIdentity(
  #       domain_shape=[2], multiplier=3.)
  #   self.assertIsInstance(
  #       operator1.matmul(operator1),
  #       linear_operator_identity.LinearOperatorIdentity)

  #   self.assertIsInstance(
  #       operator1.matmul(operator1),
  #       linear_operator_identity.LinearOperatorIdentity)

  #   self.assertIsInstance(
  #       operator2.matmul(operator2),
  #       linear_operator_identity.LinearOperatorScaledIdentity)

  #   operator_matmul = operator1.matmul(operator2)
  #   self.assertIsInstance(
  #       operator_matmul,
  #       linear_operator_identity.LinearOperatorScaledIdentity)
  #   self.assertAllClose(3., self.evaluate(operator_matmul.multiplier))

  #   operator_matmul = operator2.matmul(operator1)
  #   self.assertIsInstance(
  #       operator_matmul,
  #       linear_operator_identity.LinearOperatorScaledIdentity)
  #   self.assertAllClose(3., self.evaluate(operator_matmul.multiplier))

  # def test_identity_solve(self):
  #   operator1 = linear_operator_identity.LinearOperatorIdentity(
  #       domain_shape=[2])
  #   operator2 = linear_operator_identity.LinearOperatorScaledIdentity(
  #       domain_shape=[2], multiplier=3.)
  #   self.assertIsInstance(
  #       operator1.solve(operator1),
  #       linear_operator_identity.LinearOperatorIdentity)

  #   self.assertIsInstance(
  #       operator2.solve(operator2),
  #       linear_operator_identity.LinearOperatorScaledIdentity)

  #   operator_solve = operator1.solve(operator2)
  #   self.assertIsInstance(
  #       operator_solve,
  #       linear_operator_identity.LinearOperatorScaledIdentity)
  #   self.assertAllClose(3., self.evaluate(operator_solve.multiplier))

  #   operator_solve = operator2.solve(operator1)
  #   self.assertIsInstance(
  #       operator_solve,
  #       linear_operator_identity.LinearOperatorScaledIdentity)
  #   self.assertAllClose(1. / 3., self.evaluate(operator_solve.multiplier))

  # def test_scaled_identity_cholesky_type(self):
  #   operator = linear_operator_identity.LinearOperatorScaledIdentity(
  #       domain_shape=[2],
  #       multiplier=3.,
  #       is_positive_definite=True,
  #       is_self_adjoint=True,
  #   )
  #   self.assertIsInstance(
  #       operator.cholesky(),
  #       linear_operator_identity.LinearOperatorScaledIdentity)

  def test_scaled_identity_inverse_type(self):
    operator = linear_operator_identity.LinearOperatorScaledIdentity(
        domain_shape=[2],
        multiplier=3.,
        is_non_singular=True,
    )
    self.assertIsInstance(
        operator.inverse(),
        linear_operator_identity.LinearOperatorScaledIdentity)

  def test_ref_type_shape_args_raises(self):
    with self.assertRaisesRegex(TypeError, "domain_shape.*reference"):
      linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=tf.Variable(2), multiplier=1.23)

  def test_tape_safe(self):
    multiplier = tf.Variable(1.23)
    operator = linear_operator_identity.LinearOperatorScaledIdentity(
        domain_shape=[2], multiplier=multiplier)
    self.check_tape_safe(operator)

  def test_convert_variables_to_tensors(self):
    multiplier = tf.Variable(1.23)
    operator = linear_operator_identity.LinearOperatorScaledIdentity(
        domain_shape=[2], multiplier=multiplier)
    with self.cached_session() as sess:
      sess.run([multiplier.initializer])
      self.check_convert_variables_to_tensors(operator)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorIdentityTest)
  linear_operator_test_util.add_tests(LinearOperatorScaledIdentityTest)
  tf.test.main()
