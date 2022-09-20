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
"""Tests for module `linear_operator_nufft`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.geometry import rotation_2d
from tensorflow_mri.python.linalg import linear_operator_nufft
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.util import test_util


  # @staticmethod
  # def skip_these_tests():
  #   return [
  #       # "add_to_tensor",
  #       # "adjoint",
  #       "cholesky", #x
  #       # "cond",
  #       # "composite_tensor",
  #       # "det",
  #       # "diag_part",
  #       "eigvalsh",  #x
  #       # "inverse",
  #       # "log_abs_det",
  #       # "operator_matmul_with_same_type",
  #       # "operator_solve_with_same_type",
  #       # "matmul",
  #       # "matmul_with_broadcast",
  #       # "saved_model",
  #       # "slicing",
  #       # "solve",
  #       # "solve_with_broadcast",
  #       # "to_dense",
  #       # "trace"
  #   ]


# class LinearOperatorNUFFTTest(test_util.TestCase):
#   @parameterized.named_parameters(
#       ("normalized", "ortho"),
#       ("unnormalized", None)
#   )
#   def test_general(self, norm):
#     shape = [8, 12]
#     n_points = 100
#     rank = 2
#     rng = np.random.default_rng()
#     traj = rng.uniform(low=-np.pi, high=np.pi, size=(n_points, rank))
#     traj = traj.astype(np.float32)
#     linop = linear_operator_nufft.LinearOperatorNUFFT(shape, traj, norm=norm)

#     self.assertIsInstance(linop.domain_shape, tf.TensorShape)
#     self.assertIsInstance(linop.domain_shape_tensor(), tf.Tensor)
#     self.assertIsInstance(linop.range_shape, tf.TensorShape)
#     self.assertIsInstance(linop.range_shape_tensor(), tf.Tensor)
#     self.assertIsInstance(linop.batch_shape, tf.TensorShape)
#     self.assertIsInstance(linop.batch_shape_tensor(), tf.Tensor)
#     self.assertAllClose(shape, linop.domain_shape)
#     self.assertAllClose(shape, linop.domain_shape_tensor())
#     self.assertAllClose([n_points], linop.range_shape)
#     self.assertAllClose([n_points], linop.range_shape_tensor())
#     self.assertAllClose([], linop.batch_shape)
#     self.assertAllClose([], linop.batch_shape_tensor())

#     # Check forward.
#     x = (rng.uniform(size=shape).astype(np.float32) +
#          rng.uniform(size=shape).astype(np.float32) * 1j)
#     expected_forward = fft_ops.nufft(x, traj)
#     if norm:
#       expected_forward /= np.sqrt(np.prod(shape))
#     result_forward = linop.transform(x)
#     self.assertAllClose(expected_forward, result_forward, rtol=1e-5, atol=1e-5)

#     # Check adjoint.
#     expected_adjoint = fft_ops.nufft(result_forward, traj, grid_shape=shape,
#                                      transform_type="type_1",
#                                      fft_direction="backward")
#     if norm:
#       expected_adjoint /= np.sqrt(np.prod(shape))
#     result_adjoint = linop.transform(result_forward, adjoint=True)
#     self.assertAllClose(expected_adjoint, result_adjoint, rtol=1e-5, atol=1e-5)


#   @parameterized.named_parameters(
#       ("normalized", "ortho"),
#       ("unnormalized", None)
#   )
#   def test_with_batch_dim(self, norm):
#     shape = [8, 12]
#     n_points = 100
#     batch_size = 4
#     traj_shape = [batch_size, n_points]
#     rank = 2
#     rng = np.random.default_rng()
#     traj = rng.uniform(low=-np.pi, high=np.pi, size=(*traj_shape, rank))
#     traj = traj.astype(np.float32)
#     linop = linear_operator_nufft.LinearOperatorNUFFT(shape, traj, norm=norm)

#     self.assertIsInstance(linop.domain_shape, tf.TensorShape)
#     self.assertIsInstance(linop.domain_shape_tensor(), tf.Tensor)
#     self.assertIsInstance(linop.range_shape, tf.TensorShape)
#     self.assertIsInstance(linop.range_shape_tensor(), tf.Tensor)
#     self.assertIsInstance(linop.batch_shape, tf.TensorShape)
#     self.assertIsInstance(linop.batch_shape_tensor(), tf.Tensor)
#     self.assertAllClose(shape, linop.domain_shape)
#     self.assertAllClose(shape, linop.domain_shape_tensor())
#     self.assertAllClose([n_points], linop.range_shape)
#     self.assertAllClose([n_points], linop.range_shape_tensor())
#     self.assertAllClose([batch_size], linop.batch_shape)
#     self.assertAllClose([batch_size], linop.batch_shape_tensor())

#     # Check forward.
#     x = (rng.uniform(size=shape).astype(np.float32) +
#          rng.uniform(size=shape).astype(np.float32) * 1j)
#     expected_forward = fft_ops.nufft(x, traj)
#     if norm:
#       expected_forward /= np.sqrt(np.prod(shape))
#     result_forward = linop.transform(x)
#     self.assertAllClose(expected_forward, result_forward, rtol=1e-5, atol=1e-5)

#     # Check adjoint.
#     expected_adjoint = fft_ops.nufft(result_forward, traj, grid_shape=shape,
#                                      transform_type="type_1",
#                                      fft_direction="backward")
#     if norm:
#       expected_adjoint /= np.sqrt(np.prod(shape))
#     result_adjoint = linop.transform(result_forward, adjoint=True)
#     self.assertAllClose(expected_adjoint, result_adjoint, rtol=1e-5, atol=1e-5)


#   @parameterized.named_parameters(
#       ("normalized", "ortho"),
#       ("unnormalized", None)
#   )
#   def test_with_extra_dim(self, norm):
#     shape = [8, 12]
#     n_points = 100
#     batch_size = 4
#     traj_shape = [batch_size, n_points]
#     rank = 2
#     rng = np.random.default_rng()
#     traj = rng.uniform(low=-np.pi, high=np.pi, size=(*traj_shape, rank))
#     traj = traj.astype(np.float32)
#     linop = linear_operator_nufft.LinearOperatorNUFFT(
#         [batch_size, *shape], traj, norm=norm)

#     self.assertIsInstance(linop.domain_shape, tf.TensorShape)
#     self.assertIsInstance(linop.domain_shape_tensor(), tf.Tensor)
#     self.assertIsInstance(linop.range_shape, tf.TensorShape)
#     self.assertIsInstance(linop.range_shape_tensor(), tf.Tensor)
#     self.assertIsInstance(linop.batch_shape, tf.TensorShape)
#     self.assertIsInstance(linop.batch_shape_tensor(), tf.Tensor)
#     self.assertAllClose([batch_size, *shape], linop.domain_shape)
#     self.assertAllClose([batch_size, *shape], linop.domain_shape_tensor())
#     self.assertAllClose([batch_size, n_points], linop.range_shape)
#     self.assertAllClose([batch_size, n_points], linop.range_shape_tensor())
#     self.assertAllClose([], linop.batch_shape)
#     self.assertAllClose([], linop.batch_shape_tensor())

#     # Check forward.
#     x = (rng.uniform(size=[batch_size, *shape]).astype(np.float32) +
#          rng.uniform(size=[batch_size, *shape]).astype(np.float32) * 1j)
#     expected_forward = fft_ops.nufft(x, traj)
#     if norm:
#       expected_forward /= np.sqrt(np.prod(shape))
#     result_forward = linop.transform(x)
#     self.assertAllClose(expected_forward, result_forward, rtol=1e-5, atol=1e-5)

#     # Check adjoint.
#     expected_adjoint = fft_ops.nufft(result_forward, traj, grid_shape=shape,
#                                      transform_type="type_1",
#                                      fft_direction="backward")
#     if norm:
#       expected_adjoint /= np.sqrt(np.prod(shape))
#     result_adjoint = linop.transform(result_forward, adjoint=True)
#     self.assertAllClose(expected_adjoint, result_adjoint, rtol=1e-5, atol=1e-5)


#   def test_with_density(self):
#     image_shape = (128, 128)
#     image = image_ops.phantom(shape=image_shape, dtype=tf.complex64)
#     points = traj_ops.radial_trajectory(
#         128, 128, flatten_encoding_dims=True)
#     density = traj_ops.radial_density(
#         128, 128, flatten_encoding_dims=True)
#     weights = tf.cast(tf.math.sqrt(tf.math.reciprocal_no_nan(density)),
#                       tf.complex64)

#     linop = linear_operator_nufft.LinearOperatorNUFFT(
#         image_shape, points=points)
#     linop_d = linear_operator_nufft.LinearOperatorNUFFT(
#         image_shape, points=points, density=density)

#     # Test forward.
#     kspace = linop.transform(image)
#     kspace_d = linop_d.transform(image)
#     self.assertAllClose(kspace * weights, kspace_d)

#     # Test adjoint and preprocess function.
#     recon = linop.transform(
#         linop.preprocess(kspace, adjoint=True) * weights * weights,
#         adjoint=True)
#     recon_d1 = linop_d.transform(kspace_d, adjoint=True)
#     recon_d2 = linop_d.transform(linop_d.preprocess(kspace, adjoint=True),
#                                  adjoint=True)
#     self.assertAllClose(recon, recon_d1)
#     self.assertAllClose(recon, recon_d2)


# class LinearOperatorGramNUFFTTest(test_util.TestCase):
#   @parameterized.product(
#       density=[False, True],
#       norm=[None, 'ortho'],
#       toeplitz=[False, True],
#       batch=[False, True]
#   )
#   def test_general(self, density, norm, toeplitz, batch):
#     with tf.device('/cpu:0'):
#       image_shape = (128, 128)
#       image = image_ops.phantom(shape=image_shape, dtype=tf.complex64)
#       points = traj_ops.radial_trajectory(
#           128, 129, flatten_encoding_dims=True)
#       if density is True:
#         density = traj_ops.radial_density(
#             128, 129, flatten_encoding_dims=True)
#       else:
#         density = None

#       # If testing batches, create new inputs to generate a batch.
#       if batch:
#         image = tf.stack([image, image * 0.5])
#         points = tf.stack([
#             points,
#             rotation_2d.Rotation2D.from_euler([np.pi / 2]).rotate(points)])
#         if density is not None:
#           density = tf.stack([density, density])

#       linop = linear_operator_nufft.LinearOperatorNUFFT(
#           image_shape, points=points, density=density, norm=norm)
#       linop_gram = linear_operator_nufft.LinearOperatorGramNUFFT(
#           image_shape, points=points, density=density, norm=norm,
#           toeplitz=toeplitz)

#       recon = linop.transform(linop.transform(image), adjoint=True)
#       recon_gram = linop_gram.transform(image)

#       if norm is None:
#         # Reduce the magnitude of these values to avoid the need to use a large
#         # tolerance.
#         recon /= tf.cast(tf.math.reduce_prod(image_shape), tf.complex64)
#         recon_gram /= tf.cast(tf.math.reduce_prod(image_shape), tf.complex64)

#       self.assertAllClose(recon, recon_gram, rtol=1e-4, atol=1e-4)


import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_adjoint
from tensorflow_mri.python.linalg import linear_operator_nufft
from tensorflow_mri.python.linalg import linear_operator_identity
from tensorflow_mri.python.linalg import linear_operator_test_util
from tensorflow_mri.python.util import test_util


rng = np.random.RandomState(2016)


# @test_util.run_all_in_graph_and_eager_modes
class LinearOperatorNUFFTTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""
  # _atol = {
  #     tf.complex64: 1e-6,
  #     tf.complex128: 1e-12
  # }

  # _rtol = {
  #     tf.complex64: 1e-6,
  #     tf.complex128: 1e-12
  # }

  @staticmethod
  def skip_these_tests():
    return super(LinearOperatorNUFFTTest, LinearOperatorNUFFTTest).skip_these_tests() + [
        "add_to_tensor",
        "adjoint",
        "cond",
        "composite_tensor",
        "diag_part",
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
        "matmul",
        "matmul_with_broadcast",
        "saved_model",
        "slicing",
        "to_dense",
        "trace",
        "lstsq",
        "lstsq_with_broadcast"
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

    batch_shape = shape[:-2]
    num_rows = shape[-2]
    num_columns = shape[-1]

    points = tf.random.uniform(
        shape=batch_shape + [num_rows, 1],
        minval=-np.pi, maxval=np.pi,
        dtype=dtype.real_dtype)
    density = linear_operator_nufft.density_1d(points)
    op1 = tf.print("points", points)
    op2 = tf.print("density", density)
    with tf.control_dependencies([op1, op2]):
      points = tf.identity(points)


    operator = linear_operator_nufft.LinearOperatorNUFFT(
        domain_shape=[num_columns], points=points, density=density)

    matrix = linear_operator_nufft.nudft_matrix(
        domain_shape=[num_columns], points=points)

    return operator, matrix

  def test_density(self):
    batch_shape = []
    num_rows = 3
    num_columns = 4

    points = tf.random.uniform(
        shape=batch_shape + [num_rows, 1],
        minval=-np.pi, maxval=np.pi,
        dtype=tf.float32)
    # points = tf.sort(points, axis=-2)
    # step = 2 * np.pi / num_rows
    # points = tf.expand_dims(
    #     tf.linspace(-np.pi, np.pi - step, num_rows), 1)
    # density = tf.expand_dims(linear_operator_nufft.density_1d([num_columns], points), -1)
    # weights = tf.cast(tf.math.reciprocal(density), tf.complex64)

    operator = linear_operator_nufft.LinearOperatorNUFFT(
        domain_shape=[num_columns], points=points)

    print(operator.to_dense())
    print("points", points)

    fwd = linear_operator_nufft.nudft_matrix(
        domain_shape=[num_columns], points=points)

    adj = tf.linalg.adjoint(fwd)
    pinv = np.linalg.pinv(fwd)
    cross = tf.linalg.inv(fwd @ adj)

    x = tf.ones([num_columns, 1], dtype=tf.complex64)
    rhs = fwd @ x

    dens = tf.linalg.lstsq(adj, pinv, fast=False)
    # s, u, v = tf.linalg.svd(dens)
    # dens_real = tf.cast(tf.math.real(dens), dens.dtype)
    # dens = tf.linalg.solve(adj, pinv)
    with np.printoptions(precision=3, suppress=True):
      print("dens", dens.numpy(), tf.linalg.diag_part(dens).numpy())
    # print("u", u)
    # print("s", s)
    # print("v", v)

    # # apply dens
    # dens_o_rhs = dens @ rhs
    # print("rhs", rhs.numpy())
    # # print("dens @ rhs", dens_o_rhs.numpy())

    # weights_from_pinv = tf.cast(tf.math.abs(dens_o_rhs / rhs), tf.complex64)
    # print("weights_from_pinv", weights_from_pinv.numpy())

    # # weights_lsqr = tf.linalg.lstsq(fwd @ adj, tf.ones_like(rhs), fast=False)
    # weights_lsqr = tf.linalg.lstsq(spread, tf.ones(x.shape, tf.float32), fast=False)
    # print("weights_lsqr", weights_lsqr.numpy())
    # weights_lsqr = tf.cast(weights_lsqr, tf.complex64)

    r1 = tf.linalg.lstsq(fwd, rhs, fast=False)
    r2 = pinv @ rhs
    r3 = adj @ dens @ rhs
    # r4 = adj @ dens_real @ rhs
    # r5 = adj @ (rhs * weights_from_pinv)
    # r6 = adj @ (rhs * weights)
    # r7 = adj @ (rhs * weights_lsqr)
    r8 = adj @ cross @ rhs

    print("r1", r1)
    print("r2", r2)
    print("r3", r3)
    # print("r4", r4)
    # print("r5", r5)
    # print("r6", r6)
    # print("r7", r7)
    print("r8", r8)

    # print("r1(abs)", tf.math.abs(r1))
    # print("r2(abs)", tf.math.abs(r2))
    # print("r3(abs)", tf.math.abs(r3))
    # # print("r4(abs)", tf.math.abs(r4))
    # print("r5(abs)", tf.math.abs(r5))
    # print("r6(abs)", tf.math.abs(r6))
    # print("r6(abs)", tf.math.abs(r6))


    # # rec1 = tf.linalg.lstsq(rec1)
    # self.assertAllClose()
    # print("dens", dens)

    # print("adj pinv", adj / pinv)
    # print(tf.ones((4, 4)) @ tf.linalg.diag([1., 2., 3., 4.]))
          # def test_nudft_matrix(self):
  #   # shape = [128, 128]
  #   points = tf.random.uniform(
  #       shape=[16, 1], minval=-np.pi, maxval=np.pi, dtype=tf.float32)
  #   other = linear_operator_nufft._nudft_matrix(
  #       points, [16], 'forward') / 4.0
  #   points = tf.stack([points, points], axis=0)
  #   matrix = linear_operator_nufft.nudft_matrix(
  #       domain_shape=[16], points=points)

  #   print(matrix.shape)
  #   self.assertAllEqual(other, matrix[0])
  #   self.assertAllEqual(other, matrix[1])

  # def test_nudft_matrix_dc_only(self):
  #   # shape = [128, 128]
  #   points = tf.random.uniform(
  #       shape=[2, 1], minval=-np.pi, maxval=np.pi, dtype=tf.float32)
  #   other = linear_operator_nufft._nudft_matrix(
  #       points, [1], 'forward') / 4.0
  #   points = tf.stack([points, points], axis=0)
  #   matrix = linear_operator_nufft.nudft_matrix(
  #       domain_shape=[1], points=points)
  #   print(matrix)
  #   # print(matrix.shape)
  #   # self.assertAllEqual(other, matrix[0])
  #   # self.assertAllEqual(other, matrix[1])

  # def test_assert_self_adjoint(self):
  #   with self.cached_session():
  #     operator = linear_operator_nufft.LinearOperatorNUFFT(domain_shape=[4])
  #     with self.assertRaisesOpError("not equal to its adjoint"):
  #       self.evaluate(operator.assert_self_adjoint())

  # def test_non_1d_domain_shape_raises_static(self):
  #   with self.assertRaisesRegex(ValueError, "must be a 1-D"):
  #     linear_operator_nufft.LinearOperatorNUFFT(domain_shape=2)

  # def test_non_integer_domain_shape_raises_static(self):
  #   with self.assertRaisesRegex(TypeError, "must be integer"):
  #     linear_operator_nufft.LinearOperatorNUFFT(domain_shape=[2.])

  # def test_non_1d_domain_shape_raises_static(self):
  #   with self.assertRaisesRegex(ValueError, "must be non-negative"):
  #     linear_operator_nufft.LinearOperatorNUFFT(domain_shape=[-2])

  # def test_non_1d_batch_shape_raises_static(self):
  #   with self.assertRaisesRegex(ValueError, "must be a 1-D"):
  #     linear_operator_nufft.LinearOperatorNUFFT(domain_shape=[2], batch_shape=2)

  # def test_non_integer_batch_shape_raises_static(self):
  #   with self.assertRaisesRegex(TypeError, "must be integer"):
  #     linear_operator_nufft.LinearOperatorNUFFT(domain_shape=[2], batch_shape=[2.])

  # def test_negative_batch_shape_raises_static(self):
  #   with self.assertRaisesRegex(ValueError, "must be non-negative"):
  #     linear_operator_nufft.LinearOperatorNUFFT(domain_shape=[2], batch_shape=[-2])

  # def test_wrong_matrix_dimensions_raises_static(self):
  #   operator = linear_operator_nufft.LinearOperatorNUFFT(domain_shape=[2])
  #   x = rng.randn(3, 3).astype(np.complex64)
  #   with self.assertRaisesRegex(ValueError, "Dimensions.*not compatible"):
  #     operator.matmul(x)

  # def test_is_x_flags(self):
  #   operator = linear_operator_nufft.LinearOperatorNUFFT(domain_shape=[2])
  #   self.assertTrue(operator.is_non_singular)
  #   self.assertFalse(operator.is_self_adjoint)
  #   self.assertTrue(operator.is_square)

  # def test_inverse_type(self):
  #   operator = linear_operator_nufft.LinearOperatorNUFFT(
  #       domain_shape=[4], is_non_singular=True)
  #   self.assertIsInstance(
  #       operator.inverse(), linear_operator_adjoint.LinearOperatorAdjoint)
  #   self.assertIsInstance(
  #       operator.inverse().operator, linear_operator_nufft.LinearOperatorNUFFT)

  # def test_identity_matmul(self):
  #   operator1 = linear_operator_nufft.LinearOperatorNUFFT(domain_shape=[2])
  #   operator2 = linear_operator_identity.LinearOperatorIdentity(num_rows=2)
  #   self.assertIsInstance(operator1.matmul(operator2),
  #                         linear_operator_nufft.LinearOperatorNUFFT)
  #   self.assertIsInstance(operator2.matmul(operator1),
  #                         linear_operator_nufft.LinearOperatorNUFFT)

  # def test_ref_type_shape_args_raises(self):
  #   with self.assertRaisesRegex(TypeError, "domain_shape.cannot.be.reference"):
  #     linear_operator_nufft.LinearOperatorNUFFT(
  #         domain_shape=tf.Variable([2]))

  #   with self.assertRaisesRegex(TypeError, "batch_shape.cannot.be.reference"):
  #     linear_operator_nufft.LinearOperatorNUFFT(
  #         domain_shape=[2], batch_shape=tf.Variable([2]))

  # def test_matvec_nd(self):
  #   for adjoint in (False, True):
  #     with self.subTest(adjoint=adjoint):
  #       operator = linear_operator_nufft.LinearOperatorNUFFT(domain_shape=[4, 4])
  #       x = tf.constant(rng.randn(4, 4).astype(np.complex64))
  #       y = operator.matvec_nd(x, adjoint=adjoint)
  #       fn = tf.signal.ifft2d if adjoint else tf.signal.fft2d
  #       expected = tf.signal.fftshift(fn(tf.signal.ifftshift(x)))
  #       expected = expected * 4 if adjoint else expected / 4
  #       self.assertAllClose(expected, y)


# @test_util.run_all_in_graph_and_eager_modes
# class LinearOperatorMaskedFFTTest(
#     linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
#   """Most tests done in the base class LinearOperatorDerivedClassTest."""
#   @staticmethod
#   def skip_these_tests():
#     return [
#         "cholesky",
#         # cond is infinite for masked FFT.
#         "cond",
#         "eigvalsh",
#         # solve and inverse are not possible because a masked FFT is not
#         # invertible.
#         "inverse",
#         "solve",
#         "solve_with_broadcast"
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

#     mask = rng.binomial(1, 0.5, size=batch_shape + [num_rows]).astype(
#         np.bool_)

#     operator = linear_operator_nufft.LinearOperatorNUFFT(
#         domain_shape=[num_rows], batch_shape=batch_shape, dtype=dtype,
#         mask=mask)

#     matrix = linear_operator_fft.dft_matrix(
#         num_rows, batch_shape=batch_shape, dtype=dtype, shift=True)
#     matrix = matrix * mask[..., :, None]

#     return operator, matrix

#   def test_inverse_raises(self):
#     operator = linear_operator_nufft.LinearOperatorNUFFT(
#         domain_shape=[2], mask=[True, False])
#     with self.assertRaisesRegex(ValueError, "singular matrix"):
#       operator.inverse()

#   def test_solve_raises(self):
#     operator = linear_operator_nufft.LinearOperatorNUFFT(
#         domain_shape=[2], mask=[True, False])
#     rhs = rng.randn(2, 2).astype(np.complex64)
#     with self.assertRaisesRegex(
#         NotImplementedError, "Exact solve not implemented.*singular"):
#       operator.solve(rhs)

#   def test_matvec_nd(self):
#     for adjoint in [False, True]:
#       with self.subTest(adjoint=adjoint):
#         mask = np.eye(4, dtype=np.bool_)
#         operator = linear_operator_nufft.LinearOperatorNUFFT(
#             domain_shape=[4, 4], mask=mask)
#         x = tf.constant(rng.randn(4, 4).astype(np.complex64))
#         y = operator.matvec_nd(x, adjoint=adjoint)

#         expected = x
#         if adjoint:
#           expected = tf.where(mask, expected, 0.)
#         fn = tf.signal.ifft2d if adjoint else tf.signal.fft2d
#         expected = tf.signal.fftshift(fn(tf.signal.ifftshift(expected)))
#         expected = expected * 4 if adjoint else expected / 4
#         if not adjoint:
#           expected = tf.where(mask, expected, 0.)
#         self.assertAllClose(expected, y)


linear_operator_test_util.add_tests(LinearOperatorNUFFTTest)
# linear_operator_test_util.add_tests(LinearOperatorMaskedFFTTest)


if __name__ == "__main__":
  tf.test.main()
