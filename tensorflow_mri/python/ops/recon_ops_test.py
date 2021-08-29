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
    cls.data.update(io_utils.read_hdf5('tests/data/recon_ops_data_2.h5'))


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

    self.assertAllClose(image, result, rtol=1e-5, atol=1e-5)

    # Check batch of k-space data and batch of sensitivities.
    batch_sens = tf.tile(tf.expand_dims(sens, 0), [19, 1, 1, 1])
    image = recon_ops.reconstruct(kspace, traj, sensitivities=batch_sens)

    self.assertAllClose(image, result, rtol=1e-5, atol=1e-5)

    # Check batch of k-space data without batch of trajectories (trajectory is
    # equal for all frames in this case).
    image = recon_ops.reconstruct(kspace, traj[0, ...], sensitivities=sens)

    self.assertAllClose(image, result, rtol=1e-3, atol=1e-3)

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


  @test_utils.parameterized_test(combine_coils=[True, False],
                                 return_kspace=[True, False])
  def test_grappa_2d(self, combine_coils, return_kspace): # pylint:disable=missing-param-doc
    """Test GRAPPA reconstruction (2D, scalar batch)."""
    data = io_utils.read_hdf5('tests/data/brain_2d_multicoil_kspace.h5')
    full_kspace = data['kspace']
    full_kspace = tf.transpose(full_kspace, [2, 0, 1]) # [coils, x, y]

    # Undersampling factor and size of calibration region.
    factor = 2
    calib_size = 24

    # Generate a 1D sampling mask (true means sampled, false means not sampled).
    mask_1d = tf.range(full_kspace.shape[1]) % factor == 0

    # Add ACS region to mask.
    calib_slice = slice(96 - calib_size // 2, 96 + calib_size // 2)
    mask_1d = mask_1d.numpy()
    mask_1d[calib_slice] = True
    mask_1d = tf.convert_to_tensor(mask_1d)

    # Repeat the 1D mask to create a 2D mask.
    mask = tf.reshape(mask_1d, [1, full_kspace.shape[2]])
    mask = tf.tile(mask, [full_kspace.shape[1], 1])

    # Create an undersampled k-space.
    kspace = tf.boolean_mask(full_kspace, mask_1d, axis=2)

    # Create a calibration region.
    calib = full_kspace[:, :, calib_slice]

    # Test op.
    result = recon_ops.reconstruct(kspace,
                                   calib=calib,
                                   mask=mask,
                                   weights_l2_regularizer=0.01,
                                   combine_coils=combine_coils,
                                   return_kspace=return_kspace)

    # Reference result.
    ref = self.data['grappa/2d/result']
    if not return_kspace:
      ref = fft_ops.ifftn(ref, axes=[-2, -1], shift=True)
      if combine_coils:
        ref = tf.math.sqrt(tf.math.reduce_sum(ref * tf.math.conj(ref), 0))

    self.assertAllClose(result, ref)


  def test_grappa_2d_batch(self):
    """Test GRAPPA reconstruction (2D, 1D batch)."""
    data = io_utils.read_hdf5('tests/data/cardiac_cine_2d_multicoil_kspace.h5')
    full_kspace = data['kspace']

    # Undersampling factor and size of calibration region.
    factor = 4
    calib_size = 24

    # Generate a 1D sampling mask (true means sampled, false means not sampled).
    mask_1d = tf.range(full_kspace.shape[-2]) % factor == 0

    # Add ACS region to mask.
    calib_slice = slice(104 - calib_size // 2, 104 + calib_size // 2)
    mask_1d = mask_1d.numpy()
    mask_1d[calib_slice] = True
    mask_1d = tf.convert_to_tensor(mask_1d)

    # Repeat the 1D mask to create a 2D mask.
    mask = tf.reshape(mask_1d, [full_kspace.shape[-2], 1])
    mask = tf.tile(mask, [1, full_kspace.shape[-1]])

    # Create an undersampled k-space.
    kspace = tf.boolean_mask(full_kspace, mask_1d, axis=-2)

    # Create a calibration region. Use the time average.
    calib = full_kspace[:, :, calib_slice, :]
    calib = tf.math.reduce_mean(calib, axis=0)

    # Test op.
    result = recon_ops.reconstruct(kspace, calib=calib, mask=mask,
                                   weights_l2_regularizer=0.0,
                                   return_kspace=True)
    self.assertAllClose(result, self.data['grappa/2d_cine/result'])


class ReconstructPartialKSpaceTest(tf.test.TestCase):
  """Tests for `reconstruct_partial_kspace` operation."""

  @classmethod
  def setUpClass(cls):
    """Prepare tests."""
    super().setUpClass()
    cls.data = io_utils.read_hdf5('tests/data/recon_ops_data.h5')
    cls.data.update(io_utils.read_hdf5('tests/data/recon_ops_data_2.h5'))

  @test_utils.parameterized_test(method=['zerofill', 'homodyne', 'pocs'],
                                 return_complex=[True, False],
                                 return_kspace=[True, False])
  def test_pf(self, method, return_complex, return_kspace): # pylint:disable=missing-param-doc
    """Test PF reconstruction."""
    data = io_utils.read_hdf5('tests/data/brain_2d_multicoil_kspace.h5')
    full_kspace = data['kspace']
    full_kspace = tf.transpose(full_kspace, [2, 0, 1]) # [coils, x, y]

    # PF subsampling with PF factor = 9/16.
    factors = [1.0, 9 / 16]
    kspace = full_kspace[:, :, :(192 * 9 // 16)]

    result = recon_ops.reconstruct_partial_kspace(kspace,
                                                  factors,
                                                  return_complex=return_complex,
                                                  return_kspace=return_kspace,
                                                  method=method)

    ref = self.data['pf/' + method + '/result']

    if return_kspace:
      ref = fft_ops.fftn(ref, axes=[-2, -1], shift=True)
    elif not return_complex:
      if method == 'zerofill':
        ref = tf.math.abs(ref)
      else:
        ref = tf.math.maximum(0.0, tf.math.real(ref))
    self.assertAllClose(result, ref, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
