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
"""Tests for module `linear_operator_nufft`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_nufft
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.util import test_util


class LinearOperatorNUFFTTest(test_util.TestCase):
  @parameterized.named_parameters(
      ("normalized", "ortho"),
      ("unnormalized", None)
  )
  def test_general(self, norm):
    shape = [8, 12]
    n_points = 100
    rank = 2
    rng = np.random.default_rng()
    traj = rng.uniform(low=-np.pi, high=np.pi, size=(n_points, rank))
    traj = traj.astype(np.float32)
    linop = linear_operator_nufft.LinearOperatorNUFFT(shape, traj, norm=norm)

    self.assertIsInstance(linop.domain_shape, tf.TensorShape)
    self.assertIsInstance(linop.domain_shape_tensor(), tf.Tensor)
    self.assertIsInstance(linop.range_shape, tf.TensorShape)
    self.assertIsInstance(linop.range_shape_tensor(), tf.Tensor)
    self.assertIsInstance(linop.batch_shape, tf.TensorShape)
    self.assertIsInstance(linop.batch_shape_tensor(), tf.Tensor)
    self.assertAllClose(shape, linop.domain_shape)
    self.assertAllClose(shape, linop.domain_shape_tensor())
    self.assertAllClose([n_points], linop.range_shape)
    self.assertAllClose([n_points], linop.range_shape_tensor())
    self.assertAllClose([], linop.batch_shape)
    self.assertAllClose([], linop.batch_shape_tensor())

    # Check forward.
    x = (rng.uniform(size=shape).astype(np.float32) +
         rng.uniform(size=shape).astype(np.float32) * 1j)
    expected_forward = fft_ops.nufft(x, traj)
    if norm:
      expected_forward /= np.sqrt(np.prod(shape))
    result_forward = linop.transform(x)
    self.assertAllClose(expected_forward, result_forward, rtol=1e-5, atol=1e-5)

    # Check adjoint.
    expected_adjoint = fft_ops.nufft(result_forward, traj, grid_shape=shape,
                                     transform_type="type_1",
                                     fft_direction="backward")
    if norm:
      expected_adjoint /= np.sqrt(np.prod(shape))
    result_adjoint = linop.transform(result_forward, adjoint=True)
    self.assertAllClose(expected_adjoint, result_adjoint, rtol=1e-5, atol=1e-5)


  @parameterized.named_parameters(
      ("normalized", "ortho"),
      ("unnormalized", None)
  )
  def test_with_batch_dim(self, norm):
    shape = [8, 12]
    n_points = 100
    batch_size = 4
    traj_shape = [batch_size, n_points]
    rank = 2
    rng = np.random.default_rng()
    traj = rng.uniform(low=-np.pi, high=np.pi, size=(*traj_shape, rank))
    traj = traj.astype(np.float32)
    linop = linear_operator_nufft.LinearOperatorNUFFT(shape, traj, norm=norm)

    self.assertIsInstance(linop.domain_shape, tf.TensorShape)
    self.assertIsInstance(linop.domain_shape_tensor(), tf.Tensor)
    self.assertIsInstance(linop.range_shape, tf.TensorShape)
    self.assertIsInstance(linop.range_shape_tensor(), tf.Tensor)
    self.assertIsInstance(linop.batch_shape, tf.TensorShape)
    self.assertIsInstance(linop.batch_shape_tensor(), tf.Tensor)
    self.assertAllClose(shape, linop.domain_shape)
    self.assertAllClose(shape, linop.domain_shape_tensor())
    self.assertAllClose([n_points], linop.range_shape)
    self.assertAllClose([n_points], linop.range_shape_tensor())
    self.assertAllClose([batch_size], linop.batch_shape)
    self.assertAllClose([batch_size], linop.batch_shape_tensor())

    # Check forward.
    x = (rng.uniform(size=shape).astype(np.float32) +
         rng.uniform(size=shape).astype(np.float32) * 1j)
    expected_forward = fft_ops.nufft(x, traj)
    if norm:
      expected_forward /= np.sqrt(np.prod(shape))
    result_forward = linop.transform(x)
    self.assertAllClose(expected_forward, result_forward, rtol=1e-5, atol=1e-5)

    # Check adjoint.
    expected_adjoint = fft_ops.nufft(result_forward, traj, grid_shape=shape,
                                     transform_type="type_1",
                                     fft_direction="backward")
    if norm:
      expected_adjoint /= np.sqrt(np.prod(shape))
    result_adjoint = linop.transform(result_forward, adjoint=True)
    self.assertAllClose(expected_adjoint, result_adjoint, rtol=1e-5, atol=1e-5)


  @parameterized.named_parameters(
      ("normalized", "ortho"),
      ("unnormalized", None)
  )
  def test_with_extra_dim(self, norm):
    shape = [8, 12]
    n_points = 100
    batch_size = 4
    traj_shape = [batch_size, n_points]
    rank = 2
    rng = np.random.default_rng()
    traj = rng.uniform(low=-np.pi, high=np.pi, size=(*traj_shape, rank))
    traj = traj.astype(np.float32)
    linop = linear_operator_nufft.LinearOperatorNUFFT(
        [batch_size, *shape], traj, norm=norm)

    self.assertIsInstance(linop.domain_shape, tf.TensorShape)
    self.assertIsInstance(linop.domain_shape_tensor(), tf.Tensor)
    self.assertIsInstance(linop.range_shape, tf.TensorShape)
    self.assertIsInstance(linop.range_shape_tensor(), tf.Tensor)
    self.assertIsInstance(linop.batch_shape, tf.TensorShape)
    self.assertIsInstance(linop.batch_shape_tensor(), tf.Tensor)
    self.assertAllClose([batch_size, *shape], linop.domain_shape)
    self.assertAllClose([batch_size, *shape], linop.domain_shape_tensor())
    self.assertAllClose([batch_size, n_points], linop.range_shape)
    self.assertAllClose([batch_size, n_points], linop.range_shape_tensor())
    self.assertAllClose([], linop.batch_shape)
    self.assertAllClose([], linop.batch_shape_tensor())

    # Check forward.
    x = (rng.uniform(size=[batch_size, *shape]).astype(np.float32) +
         rng.uniform(size=[batch_size, *shape]).astype(np.float32) * 1j)
    expected_forward = fft_ops.nufft(x, traj)
    if norm:
      expected_forward /= np.sqrt(np.prod(shape))
    result_forward = linop.transform(x)
    self.assertAllClose(expected_forward, result_forward, rtol=1e-5, atol=1e-5)

    # Check adjoint.
    expected_adjoint = fft_ops.nufft(result_forward, traj, grid_shape=shape,
                                     transform_type="type_1",
                                     fft_direction="backward")
    if norm:
      expected_adjoint /= np.sqrt(np.prod(shape))
    result_adjoint = linop.transform(result_forward, adjoint=True)
    self.assertAllClose(expected_adjoint, result_adjoint, rtol=1e-5, atol=1e-5)


  def test_with_density(self):
    image_shape = (128, 128)
    image = image_ops.phantom(shape=image_shape, dtype=tf.complex64)
    trajectory = traj_ops.radial_trajectory(
        128, 128, flatten_encoding_dims=True)
    density = traj_ops.radial_density(
        128, 128, flatten_encoding_dims=True)
    weights = tf.cast(tf.math.sqrt(tf.math.reciprocal_no_nan(density)),
                      tf.complex64)

    linop = linear_operator_nufft.LinearOperatorNUFFT(
        image_shape, trajectory=trajectory)
    linop_d = linear_operator_nufft.LinearOperatorNUFFT(
        image_shape, trajectory=trajectory, density=density)

    # Test forward.
    kspace = linop.transform(image)
    kspace_d = linop_d.transform(image)
    self.assertAllClose(kspace * weights, kspace_d)

    # Test adjoint and precompensate function.
    recon = linop.transform(linop.precompensate(kspace) * weights * weights,
                            adjoint=True)
    recon_d1 = linop_d.transform(kspace_d, adjoint=True)
    recon_d2 = linop_d.transform(linop_d.precompensate(kspace), adjoint=True)
    self.assertAllClose(recon, recon_d1)
    self.assertAllClose(recon, recon_d2)


if __name__ == '__main__':
  tf.test.main()