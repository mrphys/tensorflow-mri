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
"""Tests for module `linear_operator_mri`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import tensorflow as tf

from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.ops import linalg_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.util import test_util


class LinearOperatorMRITest(test_util.TestCase):
  """Tests for MRI linear operator."""
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.linop1 = linalg_ops.LinearOperatorMRI([2, 2], fft_norm=None)
    cls.linop2 = linalg_ops.LinearOperatorMRI(
        [2, 2], mask=[[False, False], [True, True]], fft_norm=None)
    cls.linop3 = linalg_ops.LinearOperatorMRI(
        [2, 2], mask=[[[True, True], [False, False]],
                      [[False, False], [True, True]],
                      [[False, True], [True, False]]], fft_norm=None)

  def test_fft(self):
    """Test FFT operator."""
    # Test init.
    linop = linalg_ops.LinearOperatorMRI([2, 2], fft_norm=None)

    # Test matvec.
    signal = tf.constant([1, 2, 4, 4], dtype=tf.complex64)
    expected = [-1, 5, 1, 11]
    result = tf.linalg.matvec(linop, signal)
    self.assertAllClose(expected, result)

    # Test domain shape.
    self.assertIsInstance(linop.domain_shape, tf.TensorShape)
    self.assertAllEqual([2, 2], linop.domain_shape)
    self.assertAllEqual([2, 2], linop.domain_shape_tensor())

    # Test range shape.
    self.assertIsInstance(linop.range_shape, tf.TensorShape)
    self.assertAllEqual([2, 2], linop.range_shape)
    self.assertAllEqual([2, 2], linop.range_shape_tensor())

    # Test batch shape.
    self.assertIsInstance(linop.batch_shape, tf.TensorShape)
    self.assertAllEqual([], linop.batch_shape)
    self.assertAllEqual([], linop.batch_shape_tensor())

  def test_fft_with_mask(self):
    """Test FFT operator with mask."""
    # Test init.
    linop = linalg_ops.LinearOperatorMRI(
        [2, 2], mask=[[False, False], [True, True]], fft_norm=None)

    # Test matvec.
    signal = tf.constant([1, 2, 4, 4], dtype=tf.complex64)
    expected = [0, 0, 1, 11]
    result = tf.linalg.matvec(linop, signal)
    self.assertAllClose(expected, result)

    # Test domain shape.
    self.assertIsInstance(linop.domain_shape, tf.TensorShape)
    self.assertAllEqual([2, 2], linop.domain_shape)
    self.assertAllEqual([2, 2], linop.domain_shape_tensor())

    # Test range shape.
    self.assertIsInstance(linop.range_shape, tf.TensorShape)
    self.assertAllEqual([2, 2], linop.range_shape)
    self.assertAllEqual([2, 2], linop.range_shape_tensor())

    # Test batch shape.
    self.assertIsInstance(linop.batch_shape, tf.TensorShape)
    self.assertAllEqual([], linop.batch_shape)
    self.assertAllEqual([], linop.batch_shape_tensor())

  def test_fft_with_batch_mask(self):
    """Test FFT operator with batch mask."""
    # Test init.
    linop = linalg_ops.LinearOperatorMRI(
        [2, 2], mask=[[[True, True], [False, False]],
                      [[False, False], [True, True]],
                      [[False, True], [True, False]]], fft_norm=None)

    # Test matvec.
    signal = tf.constant([1, 2, 4, 4], dtype=tf.complex64)
    expected = [[-1, 5, 0, 0], [0, 0, 1, 11], [0, 5, 1, 0]]
    result = tf.linalg.matvec(linop, signal)
    self.assertAllClose(expected, result)

    # Test domain shape.
    self.assertIsInstance(linop.domain_shape, tf.TensorShape)
    self.assertAllEqual([2, 2], linop.domain_shape)
    self.assertAllEqual([2, 2], linop.domain_shape_tensor())

    # Test range shape.
    self.assertIsInstance(linop.range_shape, tf.TensorShape)
    self.assertAllEqual([2, 2], linop.range_shape)
    self.assertAllEqual([2, 2], linop.range_shape_tensor())

    # Test batch shape.
    self.assertIsInstance(linop.batch_shape, tf.TensorShape)
    self.assertAllEqual([3], linop.batch_shape)
    self.assertAllEqual([3], linop.batch_shape_tensor())

  def test_fft_norm(self):
    """Test FFT normalization."""
    linop = linalg_ops.LinearOperatorMRI([2, 2], fft_norm='ortho')
    x = tf.constant([1 + 2j, 2 - 2j, -1 - 6j, 3 + 4j], dtype=tf.complex64)
    # With norm='ortho', subsequent application of the operator and its adjoint
    # should not scale the input.
    y = tf.linalg.matvec(linop.H, tf.linalg.matvec(linop, x))
    self.assertAllClose(x, y)

  def test_nufft_with_sensitivities(self):
    resolution = 128
    image_shape = [resolution, resolution]
    num_coils = 4
    image, sensitivities = image_ops.phantom(
        shape=image_shape, num_coils=num_coils, dtype=tf.complex64,
        return_sensitivities=True)
    image = image_ops.phantom(shape=image_shape, dtype=tf.complex64)
    trajectory = traj_ops.radial_trajectory(resolution, resolution // 2 + 1,
                                            flatten_encoding_dims=True)
    density = traj_ops.radial_density(resolution, resolution // 2 + 1,
                                      flatten_encoding_dims=True)

    linop = linalg_ops.LinearOperatorMRI(
        image_shape, trajectory=trajectory, density=density,
        sensitivities=sensitivities)

    # Test shapes.
    expected_domain_shape = image_shape
    self.assertAllClose(expected_domain_shape, linop.domain_shape)
    self.assertAllClose(expected_domain_shape, linop.domain_shape_tensor())
    expected_range_shape = [num_coils, (2 * resolution) * (resolution // 2 + 1)]
    self.assertAllClose(expected_range_shape, linop.range_shape)
    self.assertAllClose(expected_range_shape, linop.range_shape_tensor())

    # Test forward.
    weights = tf.cast(tf.math.sqrt(tf.math.reciprocal_no_nan(density)),
                      tf.complex64)
    norm = tf.math.sqrt(tf.cast(tf.math.reduce_prod(image_shape), tf.complex64))
    expected = fft_ops.nufft(image * sensitivities, trajectory) * weights / norm
    kspace = linop.transform(image)
    self.assertAllClose(expected, kspace)

    # Test adjoint.
    expected = tf.math.reduce_sum(
        fft_ops.nufft(
            kspace * weights, trajectory, grid_shape=image_shape,
            transform_type='type_1', fft_direction='backward') / norm *
        tf.math.conj(sensitivities), axis=-3)
    recon = linop.transform(kspace, adjoint=True)
    self.assertAllClose(expected, recon)


if __name__ == '__main__':
  tf.test.main()
