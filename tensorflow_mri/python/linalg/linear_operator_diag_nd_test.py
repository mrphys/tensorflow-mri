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
"""Tests for `linear_operator_diag_nd`."""

import tensorflow as tf

from tensorflow.python.framework import test_util

from tensorflow_mri.python.linalg import linear_operator_diag_nd
from tensorflow_mri.python.linalg import linear_operator_identity_nd
from tensorflow_mri.python.linalg import linear_operator_test_util


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorDiagNDTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    tf.config.experimental.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    self.tf32_keep_ = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)

  @staticmethod
  def optional_tests():
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type"
    ]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    shape = list(build_info.shape)
    diag = linear_operator_test_util.random_sign_uniform(
        shape[:-1], minval=1., maxval=2., dtype=dtype)
    batch_dims = len(shape) - 2

    if ensure_self_adjoint_and_pd:
      # Abs on complex64 will result in a float32, so we cast back up.
      diag = tf.cast(tf.math.abs(diag), dtype=dtype)

    lin_op_diag = diag

    if use_placeholder:
      lin_op_diag = tf.compat.v1.placeholder_with_default(
          diag, shape=(None,) * (batch_dims + 1))

    operator = linear_operator_diag_nd.LinearOperatorDiagND(
        lin_op_diag,
        batch_dims=batch_dims,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None)

    matrix = tf.linalg.diag(diag)

    return operator, matrix

  def test_assert_positive_definite_raises_for_zero_eigenvalue(self):
    # Matrix with one positive eigenvalue and one zero eigenvalue.
    with self.cached_session():
      diag = [1.0, 0.0]
      operator = linear_operator_diag_nd.LinearOperatorDiagND(diag)

      # is_self_adjoint should be auto-set for real diag.
      self.assertTrue(operator.is_self_adjoint)
      with self.assertRaisesOpError("non-positive.*not positive definite"):
        operator.assert_positive_definite().run()

  def test_assert_positive_definite_raises_for_negative_real_eigvalues(self):
    with self.cached_session():
      diag_x = [1.0, -2.0]
      diag_y = [0., 0.]  # Imaginary eigenvalues should not matter.
      diag = tf.dtypes.complex(diag_x, diag_y)
      operator = linear_operator_diag_nd.LinearOperatorDiagND(diag)

      # is_self_adjoint should not be auto-set for complex diag.
      self.assertTrue(operator.is_self_adjoint is None)
      with self.assertRaisesOpError("non-positive real.*not positive definite"):
        operator.assert_positive_definite().run()

  def test_assert_positive_definite_does_not_raise_if_pd_and_complex(self):
    with self.cached_session():
      x = [1., 2.]
      y = [1., 0.]
      diag = tf.dtypes.complex(x, y)  # Re[diag] > 0.
      operator = linear_operator_diag_nd.LinearOperatorDiagND(diag)
      # Should not fail
      self.evaluate(operator.assert_positive_definite())

  def test_assert_non_singular_raises_if_zero_eigenvalue(self):
    # Singular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.cached_session():
      diag = [1.0, 0.0]
      operator = linear_operator_diag_nd.LinearOperatorDiagND(
          diag, is_self_adjoint=True)
      with self.assertRaisesOpError("operator is singular"):
        operator.assert_non_singular().run()

  def test_assert_non_singular_does_not_raise_for_complex_nonsingular(self):
    with self.cached_session():
      x = [1., 0.]
      y = [0., 1.]
      diag = tf.dtypes.complex(x, y)
      operator = linear_operator_diag_nd.LinearOperatorDiagND(diag)
      # Should not raise.
      self.evaluate(operator.assert_non_singular())

  def test_assert_self_adjoint_raises_if_diag_has_complex_part(self):
    with self.cached_session():
      x = [1., 0.]
      y = [0., 1.]
      diag = tf.dtypes.complex(x, y)
      operator = linear_operator_diag_nd.LinearOperatorDiagND(diag)
      with self.assertRaisesOpError("imaginary.*not self-adjoint"):
        operator.assert_self_adjoint().run()

  def test_assert_self_adjoint_does_not_raise_for_diag_with_zero_imag(self):
    with self.cached_session():
      x = [1., 0.]
      y = [0., 0.]
      diag = tf.dtypes.complex(x, y)
      operator = linear_operator_diag_nd.LinearOperatorDiagND(diag)
      # Should not raise
      self.evaluate(operator.assert_self_adjoint())

  def test_scalar_diag_raises(self):
    with self.assertRaisesRegex(ValueError, "must be at least 1-D"):
      linear_operator_diag_nd.LinearOperatorDiagND(1.)

  def test_broadcast_matmul_and_solve(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.matmul cannot handle.
    # In particular, tf.matmul does not broadcast.
    with self.cached_session() as sess:
      x = tf.random.normal(shape=(2, 2, 3, 4))

      # This LinearOperatorDiagND will be broadcast to (2, 2, 3, 3) during solve
      # and matmul with 'x' as the argument.
      diag = tf.random.uniform(shape=(2, 1, 3))
      operator = linear_operator_diag_nd.LinearOperatorDiagND(
          diag, batch_dims=2, is_self_adjoint=True)
      self.assertAllEqual((2, 1, 3, 3), operator.shape)

      # Create a batch matrix with the broadcast shape of operator.
      diag_broadcast = tf.concat((diag, diag), 1)
      mat = tf.linalg.diag(diag_broadcast)
      self.assertAllEqual((2, 2, 3, 3), mat.shape)  # being pedantic.

      operator_matmul = operator.matmul(x)
      mat_matmul = tf.matmul(mat, x)
      self.assertAllEqual(operator_matmul.shape, mat_matmul.shape)
      self.assertAllClose(*self.evaluate([operator_matmul, mat_matmul]))

      operator_solve = operator.solve(x)
      mat_solve = tf.linalg.solve(mat, x)
      self.assertAllEqual(operator_solve.shape, mat_solve.shape)
      self.assertAllClose(*self.evaluate([operator_solve, mat_solve]))

  def test_diag_matmul(self):
    operator1 = linear_operator_diag_nd.LinearOperatorDiagND([2., 3.])
    operator2 = linear_operator_diag_nd.LinearOperatorDiagND([1., 2.])
    operator3 = linear_operator_identity_nd.LinearOperatorScaledIdentityND(
        domain_shape=[2], multiplier=3.)
    operator_matmul = operator1.matmul(operator2)
    self.assertTrue(isinstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose([2., 6.], self.evaluate(operator_matmul.diag))

    operator_matmul = operator2.matmul(operator1)
    self.assertTrue(isinstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose([2., 6.], self.evaluate(operator_matmul.diag))

    operator_matmul = operator1.matmul(operator3)
    self.assertTrue(isinstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose([6., 9.], self.evaluate(operator_matmul.diag))

    operator_matmul = operator3.matmul(operator1)
    self.assertTrue(isinstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose([6., 9.], self.evaluate(operator_matmul.diag))

  def test_diag_matmul_nd(self):
    operator1 = linear_operator_diag_nd.LinearOperatorDiagND(
        [[1., 2.], [3., 4.]])
    operator2 = linear_operator_diag_nd.LinearOperatorDiagND(
        [1., 2.])
    operator3 = linear_operator_diag_nd.LinearOperatorDiagND(
        [[1., 2.], [3., 4.]], batch_dims=1)
    operator4 = linear_operator_identity_nd.LinearOperatorScaledIdentityND(
        domain_shape=[2], multiplier=2.)
    operator5 = linear_operator_identity_nd.LinearOperatorScaledIdentityND(
        domain_shape=[2], multiplier=[1., 2., 3.])
    operator6 = linear_operator_identity_nd.LinearOperatorScaledIdentityND(
        domain_shape=[2, 3], multiplier=-1.)

    operator_matmul = operator1.matmul(operator1)
    self.assertIsInstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND)
    self.assertAllClose(
        [[1., 4.], [9., 16.]], self.evaluate(operator_matmul.diag))
    self.assertAllEqual([], operator_matmul.batch_shape)

    operator_matmul = operator1.matmul(operator2)
    self.assertIsInstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND)
    self.assertAllClose(
        [[1., 4.], [3., 8.]], self.evaluate(operator_matmul.diag))
    self.assertAllEqual([], operator_matmul.batch_shape)

    operator_matmul = operator2.matmul(operator1)
    self.assertIsInstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND)
    self.assertAllClose(
        [[1., 4.], [3., 8.]], self.evaluate(operator_matmul.diag))
    self.assertAllEqual([], operator_matmul.batch_shape)

    operator_matmul = operator2.matmul(operator3)
    self.assertIsInstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND)
    self.assertAllClose(
        [[1., 4.], [3., 8.]], self.evaluate(operator_matmul.diag))
    self.assertAllEqual([2], operator_matmul.batch_shape)

    operator_matmul = operator1.matmul(operator3)
    self.assertIsInstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND)
    self.assertAllClose(
        [[[1., 4.], [3., 8.]], [[3., 8.], [9., 16.]]],
        self.evaluate(operator_matmul.diag))
    self.assertAllEqual([2], operator_matmul.batch_shape)

    operator_matmul = operator1.matmul(operator4)
    self.assertTrue(isinstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose(
        [[2., 4.], [6., 8.]], self.evaluate(operator_matmul.diag))
    self.assertAllEqual([2, 2], operator_matmul.domain_shape)
    self.assertAllEqual([], operator_matmul.batch_shape)

    operator_matmul = operator4.matmul(operator1)
    self.assertTrue(isinstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose(
        [[2., 4.], [6., 8.]], self.evaluate(operator_matmul.diag))
    self.assertAllEqual([2, 2], operator_matmul.domain_shape)
    self.assertAllEqual([], operator_matmul.batch_shape)

    operator_matmul = operator2.matmul(operator5)
    self.assertTrue(isinstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose(
        [[1., 2.], [2., 4.], [3., 6.]], self.evaluate(operator_matmul.diag))
    self.assertAllEqual([2], operator_matmul.domain_shape)
    self.assertAllEqual([3], operator_matmul.batch_shape)

    operator_matmul = operator5.matmul(operator2)
    self.assertTrue(isinstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose(
        [[1., 2.], [2., 4.], [3., 6.]], self.evaluate(operator_matmul.diag))
    self.assertAllEqual([2], operator_matmul.domain_shape)
    self.assertAllEqual([3], operator_matmul.batch_shape)

    operator_matmul = operator1.matmul(operator5)
    self.assertTrue(isinstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose(
        [[[1., 2.], [3., 4.]], [[2., 4.], [6., 8.]], [[3., 6.], [9., 12.]]],
        self.evaluate(operator_matmul.diag))
    self.assertAllEqual([2, 2], operator_matmul.domain_shape)
    self.assertAllEqual([3], operator_matmul.batch_shape)

    operator_matmul = operator5.matmul(operator1)
    self.assertTrue(isinstance(
        operator_matmul,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose(
        [[[1., 2.], [3., 4.]], [[2., 4.], [6., 8.]], [[3., 6.], [9., 12.]]],
        self.evaluate(operator_matmul.diag))
    self.assertAllEqual([2, 2], operator_matmul.domain_shape)
    self.assertAllEqual([3], operator_matmul.batch_shape)

    with self.assertRaisesRegex(ValueError, "not broadcast-compatible"):
      operator_matmul = operator1.matmul(operator6)

    with self.assertRaisesRegex(ValueError, "not broadcast-compatible"):
      operator_matmul = operator6.matmul(operator1)

  def test_diag_solve(self):
    operator1 = linear_operator_diag_nd.LinearOperatorDiagND(
        [2., 3.], is_non_singular=True)
    operator2 = linear_operator_diag_nd.LinearOperatorDiagND(
        [1., 2.], is_non_singular=True)
    operator3 = linear_operator_identity_nd.LinearOperatorScaledIdentityND(
        domain_shape=[2], multiplier=3., is_non_singular=True)
    operator_solve = operator1.solve(operator2)
    self.assertTrue(isinstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose([0.5, 2 / 3.], self.evaluate(operator_solve.diag))

    operator_solve = operator2.solve(operator1)
    self.assertTrue(isinstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose([2., 3 / 2.], self.evaluate(operator_solve.diag))

    operator_solve = operator1.solve(operator3)
    self.assertTrue(isinstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose([3 / 2., 1.], self.evaluate(operator_solve.diag))

    operator_solve = operator3.solve(operator1)
    self.assertTrue(isinstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose([2 / 3., 1.], self.evaluate(operator_solve.diag))

  def test_diag_solve_nd(self):
    operator1 = linear_operator_diag_nd.LinearOperatorDiagND(
        [[1., 2.], [3., 4.]])
    operator2 = linear_operator_diag_nd.LinearOperatorDiagND(
        [1., 2.])
    operator3 = linear_operator_diag_nd.LinearOperatorDiagND(
        [[1., 2.], [3., 4.]], batch_dims=1)
    operator4 = linear_operator_identity_nd.LinearOperatorScaledIdentityND(
        domain_shape=[2], multiplier=2.)
    operator5 = linear_operator_identity_nd.LinearOperatorScaledIdentityND(
        domain_shape=[2], multiplier=[1., 2., 3.])
    operator6 = linear_operator_identity_nd.LinearOperatorScaledIdentityND(
        domain_shape=[2, 3], multiplier=-1.)

    operator_solve = operator1.solve(operator1)
    self.assertIsInstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND)
    self.assertAllClose(
        [[1., 1.], [1., 1.]], self.evaluate(operator_solve.diag))
    self.assertAllEqual([], operator_solve.batch_shape)

    operator_solve = operator1.solve(operator2)
    self.assertIsInstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND)
    self.assertAllClose(
        [[1., 1.], [1 / 3, 1 / 2]], self.evaluate(operator_solve.diag))
    self.assertAllEqual([], operator_solve.batch_shape)

    operator_solve = operator2.solve(operator1)
    self.assertIsInstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND)
    self.assertAllClose(
        [[1., 1.], [3., 2.]], self.evaluate(operator_solve.diag))
    self.assertAllEqual([], operator_solve.batch_shape)

    operator_solve = operator2.solve(operator3)
    self.assertIsInstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND)
    self.assertAllClose(
        [[1., 1.], [3., 2.]], self.evaluate(operator_solve.diag))
    self.assertAllEqual([2], operator_solve.batch_shape)

    operator_solve = operator1.solve(operator3)
    self.assertIsInstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND)
    self.assertAllClose(
        [[[1., 1.], [1 / 3, 0.5]], [[3., 2.], [1., 1.]]],
        self.evaluate(operator_solve.diag))
    self.assertAllEqual([2], operator_solve.batch_shape)

    operator_solve = operator1.solve(operator4)
    self.assertTrue(isinstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose(
        [[2., 1.], [2 / 3, 0.5]], self.evaluate(operator_solve.diag))
    self.assertAllEqual([2, 2], operator_solve.domain_shape)
    self.assertAllEqual([], operator_solve.batch_shape)

    operator_solve = operator4.solve(operator1)
    self.assertTrue(isinstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose(
        [[0.5, 1.], [3 / 2, 4 / 2]], self.evaluate(operator_solve.diag))
    self.assertAllEqual([2, 2], operator_solve.domain_shape)
    self.assertAllEqual([], operator_solve.batch_shape)

    operator_solve = operator1.solve(operator5)
    self.assertTrue(isinstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose(
        [[[1., 0.5], [1 / 3, 0.25]],
         [[2., 1.], [2 / 3, 0.5]],
         [[3., 3 / 2], [1., 0.75]]],
        self.evaluate(operator_solve.diag))
    self.assertAllEqual([2, 2], operator_solve.domain_shape)
    self.assertAllEqual([3], operator_solve.batch_shape)

    operator_solve = operator5.solve(operator1)
    self.assertTrue(isinstance(
        operator_solve,
        linear_operator_diag_nd.LinearOperatorDiagND))
    self.assertAllClose(
        [[[1., 2.], [3., 4.]],
         [[0.5, 1.], [3 / 2, 2.]],
         [[1 / 3, 2 / 3], [1., 4 / 3]]],
        self.evaluate(operator_solve.diag))
    self.assertAllEqual([2, 2], operator_solve.domain_shape)
    self.assertAllEqual([3], operator_solve.batch_shape)

    with self.assertRaisesRegex(ValueError, "not broadcast-compatible"):
      operator_solve = operator1.solve(operator6)

    with self.assertRaisesRegex(ValueError, "not broadcast-compatible"):
      operator_solve = operator6.solve(operator1)

  def test_diag_adjoint_type(self):
    diag = [1., 3., 5., 8.]
    operator = linear_operator_diag_nd.LinearOperatorDiagND(
        diag, is_non_singular=True)
    self.assertIsInstance(
        operator.adjoint(), linear_operator_diag_nd.LinearOperatorDiagND)

  def test_diag_cholesky_type(self):
    diag = [1., 3., 5., 8.]
    operator = linear_operator_diag_nd.LinearOperatorDiagND(
        diag,
        is_positive_definite=True,
        is_self_adjoint=True,
    )
    self.assertIsInstance(operator.cholesky(), linear_operator_diag_nd.LinearOperatorDiagND)

  def test_diag_inverse_type(self):
    diag = [1., 3., 5., 8.]
    operator = linear_operator_diag_nd.LinearOperatorDiagND(
        diag, is_non_singular=True)
    self.assertIsInstance(operator.inverse(),
                          linear_operator_diag_nd.LinearOperatorDiagND)

  def test_tape_safe(self):
    diag = tf.Variable([[2.]])
    operator = linear_operator_diag_nd.LinearOperatorDiagND(diag)
    self.check_tape_safe(operator)

  def test_convert_variables_to_tensors(self):
    diag = tf.Variable([[2.]])
    operator = linear_operator_diag_nd.LinearOperatorDiagND(diag)
    with self.cached_session() as sess:
      sess.run([diag.initializer])
      self.check_convert_variables_to_tensors(operator)


linear_operator_test_util.add_tests(LinearOperatorDiagNDTest)


if __name__ == "__main__":
  tf.test.main()
