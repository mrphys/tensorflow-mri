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
"""Tests for `LinearOperatorFFT`."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_adjoint
from tensorflow_mri.python.linalg import linear_operator_fft
from tensorflow_mri.python.linalg import linear_operator_identity
from tensorflow_mri.python.linalg import linear_operator_test_util
from tensorflow_mri.python.util import test_util


rng = np.random.RandomState(2016)


# @test_util.run_all_in_graph_and_eager_modes
# class LinearOperatorFFTTest(
#     linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
#   """Most tests done in the base class LinearOperatorDerivedClassTest."""
#   @staticmethod
#   def skip_these_tests():
#     return [
#         "cholesky",
#         "eigvalsh"
#     ]

#   @staticmethod
#   def dtypes_to_test():
#     return [tf.complex64, tf.complex128]

#   def operator_and_matrix(
#       self, build_info, dtype, use_placeholder,
#       ensure_self_adjoint_and_pd=False):
#     del ensure_self_adjoint_and_pd
#     del use_placeholder
#     shape = list(build_info.shape)
#     assert shape[-1] == shape[-2]

#     batch_shape = shape[:-2]
#     num_rows = shape[-1]

#     operator = linear_operator_fft.LinearOperatorFFT(
#         domain_shape=[num_rows], batch_shape=batch_shape, dtype=dtype)

#     matrix = linear_operator_fft.dft_matrix(
#         num_rows, batch_shape=batch_shape, dtype=dtype, shift=True)

#     return operator, matrix

#   def test_assert_self_adjoint(self):
#     with self.cached_session():
#       operator = linear_operator_fft.LinearOperatorFFT(domain_shape=[4])
#       with self.assertRaisesOpError("not equal to its adjoint"):
#         self.evaluate(operator.assert_self_adjoint())

#   def test_non_1d_domain_shape_raises_static(self):
#     with self.assertRaisesRegex(ValueError, "must be a 1-D"):
#       linear_operator_fft.LinearOperatorFFT(domain_shape=2)

#   def test_non_integer_domain_shape_raises_static(self):
#     with self.assertRaisesRegex(TypeError, "must be integer"):
#       linear_operator_fft.LinearOperatorFFT(domain_shape=[2.])

#   def test_non_1d_domain_shape_raises_static(self):
#     with self.assertRaisesRegex(ValueError, "must be non-negative"):
#       linear_operator_fft.LinearOperatorFFT(domain_shape=[-2])

#   def test_non_1d_batch_shape_raises_static(self):
#     with self.assertRaisesRegex(ValueError, "must be a 1-D"):
#       linear_operator_fft.LinearOperatorFFT(domain_shape=[2], batch_shape=2)

#   def test_non_integer_batch_shape_raises_static(self):
#     with self.assertRaisesRegex(TypeError, "must be integer"):
#       linear_operator_fft.LinearOperatorFFT(domain_shape=[2], batch_shape=[2.])

#   def test_negative_batch_shape_raises_static(self):
#     with self.assertRaisesRegex(ValueError, "must be non-negative"):
#       linear_operator_fft.LinearOperatorFFT(domain_shape=[2], batch_shape=[-2])

#   def test_wrong_matrix_dimensions_raises_static(self):
#     operator = linear_operator_fft.LinearOperatorFFT(domain_shape=[2])
#     x = rng.randn(3, 3).astype(np.complex64)
#     with self.assertRaisesRegex(ValueError, "Dimensions.*not compatible"):
#       operator.matmul(x)

#   def test_is_x_flags(self):
#     operator = linear_operator_fft.LinearOperatorFFT(domain_shape=[2])
#     self.assertTrue(operator.is_non_singular)
#     self.assertFalse(operator.is_self_adjoint)
#     self.assertTrue(operator.is_square)

#   def test_inverse_type(self):
#     operator = linear_operator_fft.LinearOperatorFFT(
#         domain_shape=[4], is_non_singular=True)
#     self.assertIsInstance(
#         operator.inverse(), linear_operator_adjoint.LinearOperatorAdjoint)
#     self.assertIsInstance(
#         operator.inverse().operator, linear_operator_fft.LinearOperatorFFT)

#   def test_identity_matmul(self):
#     operator1 = linear_operator_fft.LinearOperatorFFT(domain_shape=[2])
#     operator2 = linear_operator_identity.LinearOperatorIdentity(num_rows=2)
#     self.assertIsInstance(operator1.matmul(operator2),
#                           linear_operator_fft.LinearOperatorFFT)
#     self.assertIsInstance(operator2.matmul(operator1),
#                           linear_operator_fft.LinearOperatorFFT)

#   def test_ref_type_shape_args_raises(self):
#     with self.assertRaisesRegex(TypeError, "domain_shape.cannot.be.reference"):
#       linear_operator_fft.LinearOperatorFFT(
#           domain_shape=tf.Variable([2]))

#     with self.assertRaisesRegex(TypeError, "batch_shape.cannot.be.reference"):
#       linear_operator_fft.LinearOperatorFFT(
#           domain_shape=[2], batch_shape=tf.Variable([2]))

#   def test_matvec_nd(self):
#     for adjoint in (False, True):
#       with self.subTest(adjoint=adjoint):
#         operator = linear_operator_fft.LinearOperatorFFT(domain_shape=[4, 4])
#         x = tf.constant(rng.randn(4, 4).astype(np.complex64))
#         y = operator.matvec_nd(x, adjoint=adjoint)
#         fn = tf.signal.ifft2d if adjoint else tf.signal.fft2d
#         expected = tf.signal.fftshift(fn(tf.signal.ifftshift(x)))
#         expected = expected * 4 if adjoint else expected / 4
#         self.assertAllClose(expected, y)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorMaskedFFTTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""
  @staticmethod
  def skip_these_tests():
    return [
        "cholesky",
        "cond",
        "eigvalsh",
        "inverse",
        "solve",
        "solve_with_broadcast",
        ##
        "add_to_tensor",
        "adjoint",
        "composite_tensor",
        "det",
        "diag_part",
        "log_abs_det",
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
        "matmul",
        "matmul_with_broadcast",
        "saved_model",
        "slicing",
        "to_dense",
        "trace",
        "solve_ls",
        "solve_ls_with_broadcast"
    ]

  @staticmethod
  def dtypes_to_test():
    return [tf.complex64, tf.complex128]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
    del use_placeholder
    shape = list(build_info.shape)
    assert shape[-1] == shape[-2]

    batch_shape = shape[:-2]
    num_rows = shape[-1]

    mask = rng.binomial(1, 0.5, size=batch_shape + [num_rows]).astype(
        np.bool_)

    operator = linear_operator_fft.LinearOperatorFFT(
        domain_shape=[num_rows], batch_shape=batch_shape, dtype=dtype,
        mask=mask)

    matrix = linear_operator_fft.dft_matrix(
        num_rows, batch_shape=batch_shape, dtype=dtype, shift=True)
    matrix = matrix * mask[..., :, None]

    return operator, matrix

  def test_inverse_raises(self):
    operator = linear_operator_fft.LinearOperatorFFT(
        domain_shape=[2], mask=[True, False])
    with self.assertRaisesRegex(ValueError, "singular matrix"):
      operator.inverse()

  def test_solve_raises(self):
    operator = linear_operator_fft.LinearOperatorFFT(
        domain_shape=[2], mask=[True, False])
    rhs = rng.randn(2, 2).astype(np.complex64)
    with self.assertRaisesRegex(
        NotImplementedError, "Exact solve not implemented.*singular"):
      operator.solve(rhs)

  def test_matvec_nd(self):
    for adjoint in [False, True]:
      with self.subTest(adjoint=adjoint):
        mask = np.eye(4, dtype=np.bool_)
        operator = linear_operator_fft.LinearOperatorFFT(
            domain_shape=[4, 4], mask=mask)
        x = tf.constant(rng.randn(4, 4).astype(np.complex64))
        y = operator.matvec_nd(x, adjoint=adjoint)

        expected = x
        if adjoint:
          expected = tf.where(mask, expected, 0.)
        fn = tf.signal.ifft2d if adjoint else tf.signal.fft2d
        expected = tf.signal.fftshift(fn(tf.signal.ifftshift(expected)))
        expected = expected * 4 if adjoint else expected / 4
        if not adjoint:
          expected = tf.where(mask, expected, 0.)
        self.assertAllClose(expected, y)


# linear_operator_test_util.add_tests(LinearOperatorFFTTest)
linear_operator_test_util.add_tests(LinearOperatorMaskedFFTTest)


if __name__ == "__main__":
  tf.test.main()
