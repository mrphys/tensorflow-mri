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
"""Tests for module `coil_sensitivities`."""

import tensorflow as tf

from tensorflow_mri.python.coils import coil_sensitivities
from tensorflow_mri.python.linalg import linear_operator_mri
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.util import io_util
from tensorflow_mri.python.util import test_util


class EstimateTest(test_util.TestCase):
  """Tests for ops related to estimation of coil sensitivity maps."""
  @classmethod
  def setUpClass(cls):

    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/coil_ops_data.h5')

  @test_util.run_in_graph_and_eager_modes
  def test_walsh(self):
    """Test Walsh's method."""
    # GPU results are close, but about 1-2% of values show deviations up to
    # 1e-3. This is probably related to TF issue:
    # https://github.com/tensorflow/tensorflow/issues/45756
    # In the meantime, we run these tests on the CPU only. Same applies to all
    # other tests in this class.
    with tf.device('/cpu:0'):
      maps = coil_sensitivities.estimate(
          self.data['images'], method='walsh')

    self.assertAllClose(maps, self.data['maps/walsh'], rtol=1e-2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def test_walsh_transposed(self):
    """Test Walsh's method with a transposed array."""
    with tf.device('/cpu:0'):
      maps = coil_sensitivities.estimate(
        tf.transpose(self.data['images'], [2, 0, 1]),
        coil_axis=0, method='walsh')

    self.assertAllClose(maps, tf.transpose(self.data['maps/walsh'], [2, 0, 1]),
                        rtol=1e-2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def test_inati(self):
    """Test Inati's method."""
    with tf.device('/cpu:0'):
      maps = coil_sensitivities.estimate(
          self.data['images'], method='inati')

    self.assertAllClose(maps, self.data['maps/inati'], rtol=1e-4, atol=1e-4)

  @test_util.run_in_graph_and_eager_modes
  def test_espirit(self):
    """Test ESPIRiT method."""
    with tf.device('/cpu:0'):
      maps = coil_sensitivities.estimate(
          self.data['kspace'], method='espirit')

    self.assertAllClose(maps, self.data['maps/espirit'], rtol=1e-2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def test_espirit_transposed(self):
    """Test ESPIRiT method with a transposed array."""
    with tf.device('/cpu:0'):
      maps = coil_sensitivities.estimate(
        tf.transpose(self.data['kspace'], [2, 0, 1]),
        coil_axis=0, method='espirit')

    self.assertAllClose(
        maps, tf.transpose(self.data['maps/espirit'], [2, 0, 1, 3]),
        rtol=1e-2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def test_walsh_3d(self):
    """Test Walsh method with 3D image."""
    with tf.device('/cpu:0'):
      image = image_ops.phantom(shape=[64, 64, 64], num_coils=4)
      # Currently only testing if it runs.
      maps = coil_sensitivities.estimate(image, # pylint: disable=unused-variable
                                         coil_axis=0,
                                         method='walsh')


class EstimateFromKspaceTest(test_util.TestCase):
  def test_estimate_from_kspace(self):
    image_shape = [128, 128]
    image = image_ops.phantom(shape=image_shape, num_coils=4,
                              dtype=tf.complex64)
    kspace = fft_ops.fftn(image, axes=[-2, -1], shift=True)
    mask = traj_ops.accel_mask(image_shape, [2, 2], [32, 32])
    kspace = tf.where(mask, kspace, tf.zeros_like(kspace))

    operator = linear_operator_mri.LinearOperatorMRI(
        image_shape=image_shape, mask=mask)

    # Test with direct *k*-space.
    image = fft_ops.ifftn(kspace, axes=[-2, -1], norm='ortho', shift=True)
    maps = coil_sensitivities.estimate_from_kspace(
        kspace, operator, method='direct')
    self.assertAllClose(image, maps)

    # Test with calibration data.
    calib_mask = traj_ops.centre_mask(image_shape, [32, 32])
    calib_data = tf.where(calib_mask, kspace, tf.zeros_like(kspace))
    calib_image = fft_ops.ifftn(
        calib_data, axes=[-2, -1], norm='ortho', shift=True)
    maps = coil_sensitivities.estimate_from_kspace(
        kspace, operator, calib_data=calib_data, method='direct')
    self.assertAllClose(calib_image, maps)

    # Test with calibration function.
    calib_fn = lambda x, _: tf.where(calib_mask, x, tf.zeros_like(x))
    maps = coil_sensitivities.estimate_from_kspace(
        kspace, operator, calib_fn=calib_fn, method='direct')
    self.assertAllClose(calib_image, maps)

    # Test Walsh.
    expected = coil_sensitivities.estimate(
        calib_image, coil_axis=-3, method='walsh')
    maps = coil_sensitivities.estimate_from_kspace(
        kspace, operator, calib_data=calib_data, method='walsh')
    self.assertAllClose(expected, maps)

    # Test batch.
    kspace_batch = tf.stack([kspace, 2 * kspace], axis=0)
    expected = tf.stack([calib_image, 2 * calib_image], axis=0)
    maps = coil_sensitivities.estimate_from_kspace(
        kspace_batch, operator, calib_fn=calib_fn, method='direct')
    self.assertAllClose(expected, maps)


if __name__ == '__main__':
  tf.test.main()
