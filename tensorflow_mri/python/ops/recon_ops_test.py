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
"""Tests for module `recon_ops`."""

import tensorflow as tf
import tensorflow_nufft as tfft

from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.ops import recon_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.utils import io_utils
from tensorflow_mri.python.utils import test_utils


class ReconstructTest(tf.test.TestCase):
  """Tests for function `reconstruct`."""

  @classmethod
  def setUpClass(cls):
    """Prepare tests."""
    super().setUpClass()
    cls.data = io_utils.read_hdf5('tests/data/recon_ops_data.h5')


  def test_fft(self):
    """Test reconstruction method `fft`."""

    kspace = self.data['fft/kspace']
    sens = self.data['fft/sens']

    # Test single-coil.
    image = recon_ops.reconstruct(kspace[0, ...])
    result = fft_ops.ifftn(kspace[0, ...], shift=True)

    self.assertAllClose(image, result)

    # Test multi-coil, no sensitivities (sum of squares combination).
    image = recon_ops.reconstruct(kspace, multicoil=True)
    result = fft_ops.ifftn(kspace, axes=[-2, -1], shift=True)
    result = tf.math.sqrt(tf.math.reduce_sum(result * tf.math.conj(result), 0))

    self.assertAllClose(image, result)

    # Test multi-coil, no combination.
    image = recon_ops.reconstruct(kspace, multicoil=True, combine_coils=False)
    result = fft_ops.ifftn(kspace, axes=[-2, -1], shift=True)

    self.assertAllClose(image, result)

    # Test multi-coil, with sensitivities.
    image = recon_ops.reconstruct(kspace, sensitivities=sens, method='fft')
    result = fft_ops.ifftn(kspace, axes=[-2, -1], shift=True)
    scale = tf.math.reduce_sum(sens * tf.math.conj(sens), axis=0)
    result = tf.math.divide_no_nan(
      tf.math.reduce_sum(result * tf.math.conj(sens), axis=0), scale)

    self.assertAllClose(image, result)


  def test_nufft(self):
    """Test reconstruction method `nufft`."""

    kspace = self.data['nufft/kspace']
    sens = self.data['nufft/sens']
    traj = self.data['nufft/traj']
    dens = tf.cast(self.data['nufft/dens'], tf.complex64)
    edens = tf.cast(traj_ops.estimate_density(traj, [144, 144]), tf.complex64)

    # Save us some typing.
    inufft = lambda src, pts: tfft.nufft(src, pts,
                                         grid_shape=[144, 144],
                                         transform_type='type_1',
                                         fft_direction='backward')

    # Test no image shape argument.
    with self.assertRaisesRegex(ValueError, "`image_shape` must be provided"):
      image = recon_ops.reconstruct(kspace, trajectory=traj)

    # Test single-coil.
    image = recon_ops.reconstruct(kspace[0, ...], trajectory=traj,
                                  image_shape=[144, 144])
    result = inufft(kspace[0, ...] / edens, traj)

    # Test single-coil with density.
    image = recon_ops.reconstruct(kspace[0, ...], trajectory=traj, density=dens,
                                  image_shape=[144, 144])
    result = inufft(kspace[0, ...] / dens, traj)

    # Test multi-coil, no sensitivities (sum of squares combination).
    image = recon_ops.reconstruct(kspace, trajectory=traj,
                                  image_shape=[144, 144], multicoil=True)
    result = inufft(kspace / edens, traj)
    result = tf.math.sqrt(tf.math.reduce_sum(result * tf.math.conj(result), 0))

    self.assertAllClose(image, result)

    # Test multi-coil, no combination.
    image = recon_ops.reconstruct(kspace, trajectory=traj,
                                  image_shape=[144, 144], multicoil=True,
                                  combine_coils=False)
    result = inufft(kspace / edens, traj)

    self.assertAllClose(image, result)

    # Test multi-coil, with sensitivities.
    image = recon_ops.reconstruct(kspace, trajectory=traj, sensitivities=sens,
                                  image_shape=[144, 144],
                                  method='nufft')
    result = inufft(kspace / edens, traj)
    scale = tf.math.reduce_sum(sens * tf.math.conj(sens), axis=0)
    result = tf.math.divide_no_nan(
      tf.math.reduce_sum(result * tf.math.conj(sens), axis=0), scale)

    self.assertAllClose(image, result)


  @test_utils.parameterized_test(reduction_axis=[[0], [1], [0, 1]],
                                 reduction_factor=[2, 4])
  def test_sense(self, reduction_axis, reduction_factor): # pylint: disable=missing-param-doc
    """Test reconstruction method `sense`."""

    kspace = self.data['sense/kspace']
    sens = self.data['sense/sens']

    mask = tf.where(tf.range(200) % reduction_factor == 0, True, False)
    reduced_kspace = kspace
    for ax in reduction_axis:
      reduced_kspace = tf.boolean_mask(reduced_kspace, mask, axis=ax + 1)

    reduction_factors = [reduction_factor] * len(reduction_axis)

    image = recon_ops.reconstruct(reduced_kspace, sensitivities=sens,
                                  reduction_axis=reduction_axis,
                                  reduction_factor=reduction_factors,
                                  l2_regularizer=0.01)

    result_keys = {
      (2, (0,)): 'sense/result_r2_ax0',
      (2, (1,)): 'sense/result_r2_ax1',
      (2, (0, 1)): 'sense/result_r2_ax01',
      (4, (0,)): 'sense/result_r4_ax0',
      (4, (1,)): 'sense/result_r4_ax1',
      (4, (0, 1)): 'sense/result_r4_ax01',
    }

    result = self.data[result_keys[(reduction_factor, tuple(reduction_axis))]]

    self.assertAllClose(image, result)


  def test_sense_batch(self):
    """Test reconstruction method `sense` with batched inputs."""

    kspace = self.data['sense/cine/reduced_kspace']
    sens = self.data['sense/cine/sens']
    result = self.data['sense/cine/result']

    reduction_axis = 0
    reduction_factor = 2

    # Check batch of k-space data.
    image = recon_ops.reconstruct(kspace, sensitivities=sens,
                                  reduction_axis=reduction_axis,
                                  reduction_factor=reduction_factor,
                                  rank=2,
                                  l2_regularizer=0.01)

    self.assertAllClose(image, result)

    # Check batch of k-space data and batch of sensitivities.
    batch_sens = tf.tile(tf.expand_dims(sens, 0), [19, 1, 1, 1])
    image = recon_ops.reconstruct(kspace, sensitivities=batch_sens,
                                  reduction_axis=reduction_axis,
                                  reduction_factor=reduction_factor,
                                  rank=2,
                                  l2_regularizer=0.01)

    self.assertAllClose(image, result)

    # Check batch of sensitivities without k-space data.
    with self.assertRaisesRegex(ValueError, "incompatible batch shapes"):
      image = recon_ops.reconstruct(kspace[0, ...], sensitivities=batch_sens,
                                    reduction_axis=reduction_axis,
                                    reduction_factor=reduction_factor,
                                    rank=2,
                                    l2_regularizer=0.01)


  def test_cg_sense(self):
    """Test reconstruction method `cg_sense`."""

    kspace = self.data['cg_sense/kspace']
    sens = self.data['cg_sense/sens']
    traj = self.data['cg_sense/traj']
    ref = self.data['cg_sense/result']

    kspace = tf.reshape(kspace, [12, -1])
    traj = tf.reshape(traj, [-1, 2])

    result = recon_ops.reconstruct(kspace, trajectory=traj, sensitivities=sens)

    self.assertAllEqual(result.shape, ref.shape)
    self.assertAllClose(result, ref)


  def test_cg_sense_batch(self):
    """Test reconstruction method `cg_sense` with batched inputs."""

    kspace = self.data['cg_sense/cine/kspace']
    sens = self.data['cg_sense/cine/sens']
    traj = self.data['cg_sense/cine/traj']
    result = self.data['cg_sense/cine/result']

    # Check batch of k-space data and batch of trajectories.
    image = recon_ops.reconstruct(kspace, traj, sensitivities=sens)

    self.assertAllClose(image, result)

    # Check batch of k-space data and batch of sensitivities.
    batch_sens = tf.tile(tf.expand_dims(sens, 0), [19, 1, 1, 1])
    image = recon_ops.reconstruct(kspace, traj, sensitivities=batch_sens)

    self.assertAllClose(image, result)

    # Check batch of k-space data without batch of trajectories (trajectory is
    # equal for all frames in this case).
    image = recon_ops.reconstruct(kspace, traj[0, ...], sensitivities=sens)

    self.assertAllClose(image, result)

    # Check batch of sensitivities/trajectory without batch of k-space. This is
    # disallowed.
    with self.assertRaisesRegex(ValueError, "incompatible batch shapes"):
      image = recon_ops.reconstruct(kspace[0, ...],
                                    traj[0, ...],
                                    sensitivities=batch_sens)
    with self.assertRaisesRegex(ValueError, "incompatible batch shapes"):
      image = recon_ops.reconstruct(kspace[0, ...],
                                    traj,
                                    sensitivities=sens)


if __name__ == '__main__':
  tf.test.main()
