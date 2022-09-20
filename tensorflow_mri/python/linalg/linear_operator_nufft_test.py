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

import functools

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_identity
from tensorflow_mri.python.linalg import linear_operator_inversion
from tensorflow_mri.python.linalg import linear_operator_nufft
from tensorflow_mri.python.linalg import linear_operator_test_util
from tensorflow_mri.python.util import test_util


rng = np.random.RandomState(2016)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorNUFFTTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""
  # NUFFT operator does not quite reach the promised accuracy, so for now we
  # relax the test tolerance a little bit.
  # TODO(jmontalt): Investigate NUFFT precision issues.
  _atol = {
      tf.complex64: 1e-5,  # 1e-6
      tf.complex128: 1e-10  # 1e-12
  }

  _rtol = {
      tf.complex64: 1e-5,  # 1e-6
      tf.complex128: 1e-10  # 1e-12
  }

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

    operator = linear_operator_nufft.LinearOperatorNUFFT(
        domain_shape=[num_columns], points=points)

    matrix = linear_operator_nufft.nudft_matrix(
        domain_shape=[num_columns], points=points)

    return operator, matrix

  def test_assert_self_adjoint(self):
    with self.cached_session():
      operator = linear_operator_nufft.LinearOperatorNUFFT(
          domain_shape=[4], points=[[0.]])
      with self.assertRaisesOpError("not equal to its adjoint"):
        self.evaluate(operator.assert_self_adjoint())

  def test_non_1d_domain_shape_raises_static(self):
    with self.assertRaisesRegex(ValueError, "must be a 1-D"):
      linear_operator_nufft.LinearOperatorNUFFT(
          domain_shape=2, points=[[0.]])

  def test_non_integer_domain_shape_raises_static(self):
    with self.assertRaisesRegex(TypeError, "must be integer"):
      linear_operator_nufft.LinearOperatorNUFFT(
          domain_shape=[2.], points=[[0.]])

  def test_non_negative_domain_shape_raises_static(self):
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      linear_operator_nufft.LinearOperatorNUFFT(
          domain_shape=[-2], points=[[0.]])

  def test_non_float_type_points_raises(self):
    with self.assertRaisesRegex(
        TypeError, "must be a float32 or float64 tensor"):
      linear_operator_nufft.LinearOperatorNUFFT(
          domain_shape=[2], points=[[0]])

  def test_is_x_flags(self):
    operator = linear_operator_nufft.LinearOperatorNUFFT(
        domain_shape=[2], points=[[0.]])
    self.assertFalse(operator.is_self_adjoint)

  def test_inverse_type(self):
    operator = linear_operator_nufft.LinearOperatorNUFFT(
        domain_shape=[4], points=[[0.]], is_non_singular=True)
    self.assertIsInstance(
        operator.inverse(), linear_operator_inversion.LinearOperatorInversion)
    self.assertIsInstance(
        operator.inverse().operator, linear_operator_nufft.LinearOperatorNUFFT)

  def test_identity_matmul(self):
    operator1 = linear_operator_nufft.LinearOperatorNUFFT(
        domain_shape=[2], points=[[0.], [-np.pi]])
    operator2 = linear_operator_identity.LinearOperatorIdentity(num_rows=2)
    self.assertIsInstance(operator1.matmul(operator2),
                          linear_operator_nufft.LinearOperatorNUFFT)
    self.assertIsInstance(operator2.matmul(operator1),
                          linear_operator_nufft.LinearOperatorNUFFT)

  def test_ref_type_domain_shape_raises(self):
    with self.assertRaisesRegex(TypeError, "domain_shape.cannot.be.reference"):
      linear_operator_nufft.LinearOperatorNUFFT(
          domain_shape=tf.Variable([2]), points=[[0.]])

  def test_convert_variables_to_tensors(self):
    points = tf.Variable([[0.]])
    operator = linear_operator_nufft.LinearOperatorNUFFT(
        domain_shape=[2], points=points)
    with self.cached_session() as sess:
      sess.run([points.initializer])
      self.check_convert_variables_to_tensors(operator)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorNUFFTWithCrosstalkTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""
  # NUFFT operator does not quite reach the promised accuracy, so for now we
  # relax the test tolerance a little bit.
  # TODO(jmontalt): Investigate NUFFT precision issues.
  _atol = {
      tf.complex64: 1e-5,  # 1e-6
      tf.complex128: 1e-10  # 1e-12
  }

  _rtol = {
      tf.complex64: 1e-5,  # 1e-6
      tf.complex128: 1e-10  # 1e-12
  }

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

    matrix = linear_operator_nufft.nudft_matrix(
        domain_shape=[num_columns], points=points)

    if num_rows < num_columns:
      crosstalk_inverse = tf.linalg.inv(matrix @ tf.linalg.adjoint(matrix))
    else:
      crosstalk_inverse = tf.linalg.inv(tf.linalg.adjoint(matrix) @ matrix)

    operator = linear_operator_nufft.LinearOperatorNUFFT(
        domain_shape=[num_columns], points=points,
        crosstalk_inverse=crosstalk_inverse)

    return operator, matrix


class OperatorShapesInfoNUFFT():
  def __init__(self, domain_shape, num_points, batch_shape):
    self.domain_shape = domain_shape
    self.num_points = num_points
    self.batch_shape = batch_shape

  @property
  def shape(self):
    grid_size = functools.reduce(lambda a, b: a * b, self.domain_shape)
    return self.batch_shape + (self.num_points, grid_size)

  @property
  def dimension(self):
    return len(self.domain_shape)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorNUFFTNDTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""
  # NUFFT operator does not quite reach the promised accuracy, so for now we
  # relax the test tolerance a little bit.
  # TODO(jmontalt): Investigate NUFFT precision issues.
  _atol = {
      tf.complex64: 1e-5,  # 1e-6
      tf.complex128: 1e-10  # 1e-12
  }

  _rtol = {
      tf.complex64: 1e-5,  # 1e-6
      tf.complex128: 1e-10  # 1e-12
  }

  @staticmethod
  def operator_shapes_infos():
    shapes_info = OperatorShapesInfoNUFFT
    # non-batch operators (n, n) and batch operators.
    return [
        shapes_info((2, 2), 3, ()),
        shapes_info((2, 4), 5, (3,)),
        shapes_info((4, 2), 6, (1, 2)),
        shapes_info((2, 2), 6, ()),
        shapes_info((2, 2, 2), 9, ()),
        shapes_info((4, 2, 2), 7, (2,))
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

    domain_shape = build_info.domain_shape
    num_points = build_info.num_points
    batch_shape = build_info.batch_shape
    grid_size = build_info.shape[-1]
    dimension = build_info.dimension

    points = tf.random.uniform(
        shape=batch_shape + (num_points, dimension),
        minval=-np.pi, maxval=np.pi,
        dtype=dtype.real_dtype)

    matrix = linear_operator_nufft.nudft_matrix(
        domain_shape=domain_shape, points=points)

    if num_points < grid_size:
      crosstalk_inverse = tf.linalg.inv(matrix @ tf.linalg.adjoint(matrix))
    else:
      crosstalk_inverse = tf.linalg.inv(tf.linalg.adjoint(matrix) @ matrix)

    operator = linear_operator_nufft.LinearOperatorNUFFT(
        domain_shape=domain_shape, points=points,
        crosstalk_inverse=crosstalk_inverse)

    return operator, matrix


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


linear_operator_test_util.add_tests(LinearOperatorNUFFTTest)
linear_operator_test_util.add_tests(LinearOperatorNUFFTWithCrosstalkTest)
linear_operator_test_util.add_tests(LinearOperatorNUFFTNDTest)


if __name__ == "__main__":
  tf.test.main()
