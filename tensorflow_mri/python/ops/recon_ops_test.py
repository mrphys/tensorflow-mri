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

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_nufft as tfft

from tensorflow_mri.python.ops import convex_ops
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.ops import recon_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.util import io_util
from tensorflow_mri.python.util import test_util


class ReconstructTest(test_util.TestCase):
  """Tests for reconstruction functions."""
  @classmethod
  def setUpClass(cls):
    """Prepare tests."""
    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/recon_ops_data.h5')
    cls.data.update(io_util.read_hdf5('tests/data/recon_ops_data_2.h5'))
    cls.data.update(io_util.read_hdf5('tests/data/recon_ops_data_3.h5'))

  def test_adj_fft(self):
    """Test simple FFT recon."""
    kspace = self.data['fft/kspace']
    sens = self.data['fft/sens']
    image_shape = kspace.shape[-2:]

    # Test single-coil.
    image = recon_ops.reconstruct_adj(kspace[0, ...], image_shape)
    expected = fft_ops.ifftn(kspace[0, ...], norm='ortho', shift=True)

    self.assertAllClose(expected, image)

    # Test multi-coil.
    image = recon_ops.reconstruct_adj(kspace, image_shape, sensitivities=sens)
    expected = fft_ops.ifftn(kspace, axes=[-2, -1], norm='ortho', shift=True)
    scale = tf.math.reduce_sum(sens * tf.math.conj(sens), axis=0)
    expected = tf.math.divide_no_nan(
        tf.math.reduce_sum(expected * tf.math.conj(sens), axis=0), scale)

    self.assertAllClose(expected, image)

  def test_adj_nufft(self):
    """Test simple NUFFT recon."""
    kspace = self.data['nufft/kspace']
    sens = self.data['nufft/sens']
    traj = self.data['nufft/traj']
    dens = self.data['nufft/dens']
    image_shape = [144, 144]
    fft_norm_factor = tf.cast(tf.math.sqrt(144. * 144.), tf.complex64)

    # Save us some typing.
    inufft = lambda src, pts: tfft.nufft(src, pts,
                                         grid_shape=[144, 144],
                                         transform_type='type_1',
                                         fft_direction='backward')

    # Test single-coil.
    image = recon_ops.reconstruct_adj(kspace[0, ...], image_shape,
                                      trajectory=traj,
                                      density=dens)

    expected = inufft(kspace[0, ...] / tf.cast(dens, tf.complex64), traj)
    expected /= fft_norm_factor

    self.assertAllClose(expected, image)

    # Test multi-coil.
    image = recon_ops.reconstruct_adj(kspace, image_shape,
                                      trajectory=traj,
                                      density=dens,
                                      sensitivities=sens)
    expected = inufft(kspace / dens, traj)
    expected /= fft_norm_factor
    scale = tf.math.reduce_sum(sens * tf.math.conj(sens), axis=0)
    expected = tf.math.divide_no_nan(
        tf.math.reduce_sum(expected * tf.math.conj(sens), axis=0), scale)

    self.assertAllClose(expected, image)

  @test_util.run_in_graph_and_eager_modes
  def test_inufft_2d(self):
    """Test inverse NUFFT method with 2D phantom."""
    base_res = 128
    image_shape = [base_res] * 2
    expected = self.data['reconstruct/inufft/shepp_logan_2d/result'] * base_res

    # Create trajectory.
    traj = traj_ops.radial_trajectory(base_res, views=64)
    traj = tf.reshape(traj, [-1, 2])

    # Generate k-space data.
    image = image_ops.phantom(shape=image_shape, dtype=tf.complex64)
    kspace = tfft.nufft(image, traj,
                        transform_type='type_2',
                        fft_direction='forward')

    # Reconstruct.
    image = recon_ops.reconstruct_lstsq(kspace, image_shape,
                                        trajectory=traj,
                                        optimizer_kwargs={'max_iterations': 10})
    self.assertAllClose(expected, image, rtol=3e-3, atol=3e-3)

  @parameterized.product(reduction_axis=[[0], [1], [0, 1]],
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

    image = recon_ops.reconstruct_sense(reduced_kspace,
                                        sensitivities=sens,
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
    image = recon_ops.reconstruct_sense(kspace, sens,
                                        reduction_axis=reduction_axis,
                                        reduction_factor=reduction_factor,
                                        rank=2,
                                        l2_regularizer=0.01)

    self.assertAllClose(image, result)

    # Check batch of k-space data and batch of sensitivities.
    batch_sens = tf.tile(tf.expand_dims(sens, 0), [19, 1, 1, 1])
    image = recon_ops.reconstruct_sense(kspace, batch_sens,
                                        reduction_axis=reduction_axis,
                                        reduction_factor=reduction_factor,
                                        rank=2,
                                        l2_regularizer=0.01)

    self.assertAllClose(image, result)

    # Check batch of sensitivities without k-space data.
    with self.assertRaisesRegex(ValueError, "incompatible batch shapes"):
      image = recon_ops.reconstruct_sense(kspace[0, ...], batch_sens,
                                          reduction_axis=reduction_axis,
                                          reduction_factor=reduction_factor,
                                          rank=2,
                                          l2_regularizer=0.01)

  @parameterized.product(combine_coils=[True, False],
                         return_kspace=[True, False])
  def test_grappa_2d(self, combine_coils, return_kspace): # pylint:disable=missing-param-doc
    """Test GRAPPA reconstruction (2D, scalar batch)."""
    data = io_util.read_hdf5('tests/data/brain_2d_multicoil_kspace.h5')
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
    result = recon_ops.reconstruct_grappa(kspace, mask, calib,
                                          weights_l2_regularizer=0.01,
                                          combine_coils=combine_coils,
                                          return_kspace=return_kspace)

    # Reference result.
    ref = self.data['grappa/2d/result']
    if not return_kspace:
      ref = fft_ops.ifftn(ref, axes=[-2, -1], shift=True)
      if combine_coils:
        ref = tf.math.sqrt(tf.math.reduce_sum(ref * tf.math.conj(ref), 0))

    self.assertAllClose(result, ref, rtol=1e-3, atol=1e-3)

  def test_grappa_2d_batch(self):
    """Test GRAPPA reconstruction (2D, 1D batch)."""
    data = io_util.read_hdf5('tests/data/cardiac_cine_2d_multicoil_kspace.h5')
    full_kspace = data['kspace']

    # Undersampling factor and size of calibration region.
    factor = 4
    calib_size = 24

    # Generate a 1D sampling mask (true means sampled, false means not sampled).
    mask_1d = tf.range(full_kspace.shape[-2]) % factor == 0

    # Add ACS region to mask.
    calib_slice = slice(104 - calib_size // 2, 104 + calib_size // 2)
    mask_1d = tf.concat([mask_1d[:104 - calib_size // 2],
                         tf.fill([calib_size], True),
                         mask_1d[104 + calib_size // 2:]], 0)

    # Repeat the 1D mask to create a 2D mask.
    mask = tf.reshape(mask_1d, [full_kspace.shape[-2], 1])
    mask = tf.tile(mask, [1, full_kspace.shape[-1]])

    # Create an undersampled k-space.
    kspace = tf.boolean_mask(full_kspace, mask_1d, axis=-2)

    # Create a calibration region. Use the time average.
    calib = full_kspace[:, :, calib_slice, :]
    calib = tf.math.reduce_mean(calib, axis=0)

    # Test op.
    result = recon_ops.reconstruct_grappa(kspace, mask, calib,
                                          weights_l2_regularizer=0.0,
                                          return_kspace=True)
    self.assertAllClose(result, self.data['grappa/2d_cine/result'],
                        rtol=1e-4, atol=1e-4)

  def test_cg_sense(self):
    """Test CG-SENSE recon."""
    data = io_util.read_hdf5('tests/data/brain_2d_multicoil_radial_kspace.h5')
    kspace = data['kspace']
    sens = data['sens']
    traj = data['traj']
    expected = data['image/cg_sense']

    image_shape = sens.shape[-2:]
    image = recon_ops.reconstruct_lstsq(
        kspace, image_shape, trajectory=traj, sensitivities=sens,
        optimizer='cg', sens_norm=True)

    self.assertAllEqual(expected.shape, image.shape)
    self.assertAllClose(expected, image, atol=1e-5, rtol=1e-5)

  def test_cg_sense_dens(self):
    """Test CG-SENSE with density compensation."""
    data = io_util.read_hdf5('tests/data/brain_2d_multicoil_radial_kspace.h5')
    kspace = data['kspace']
    sens = data['sens']
    traj = data['traj']
    dens = data['dens']
    expected = data['image/cg_sens_dens']

    image_shape = sens.shape[-2:]
    image = recon_ops.reconstruct_lstsq(
        kspace, image_shape, trajectory=traj, density=dens, sensitivities=sens,
        optimizer='cg', sens_norm=True)

    self.assertAllEqual(expected.shape, image.shape)
    self.assertAllClose(expected, image, atol=1e-5, rtol=1e-5)

  def test_cg_sense_3d(self):
    """Test CG-SENSE in 3D."""
    # TODO: check outputs. Currently just checking it runs.
    base_resolution = 64
    image_shape = [base_resolution] * 3
    traj = traj_ops.radial_trajectory(base_resolution,
                                      views=2000,
                                      ordering='sphere_archimedean')
    traj = tf.reshape(traj, [-1, 3])

    image, sens = image_ops.phantom(shape=image_shape,
                                    num_coils=8,
                                    dtype=tf.complex64,
                                    return_sensitivities=True)
    kspace = tfft.nufft(image, traj,
                        grid_shape=image.shape,
                        transform_type='type_2',
                        fft_direction='forward')

    image = recon_ops.reconstruct_lstsq(kspace,
                                        image_shape,
                                        trajectory=traj,
                                        sensitivities=sens)

  def test_cg_sense_batch(self):
    """Test CG-SENSE with batched inputs."""
    data = io_util.read_hdf5(
        'tests/data/cardiac_cine_2d_multicoil_radial_kspace.h5')
    kspace = data['kspace']
    sens = data['sens']
    traj = data['traj']
    dens = data['dens']
    expected = data['image/cg_sense']
    image_shape = sens.shape[-2:]

    # Check batch of k-space data and batch of trajectories.
    image = recon_ops.reconstruct_lstsq(kspace=kspace,
                                        image_shape=image_shape,
                                        trajectory=traj,
                                        density=dens,
                                        sensitivities=sens,
                                        optimizer_kwargs={'max_iterations': 10})
    self.assertAllClose(expected, image, rtol=1e-5, atol=1e-5)

    # Check batch of k-space data and batch of sensitivities.
    sens_batch = tf.tile(tf.expand_dims(sens, 0), [19, 1, 1, 1])
    image = recon_ops.reconstruct_lstsq(kspace=kspace,
                                        image_shape=image_shape,
                                        trajectory=traj,
                                        density=dens,
                                        sensitivities=sens_batch,
                                        optimizer_kwargs={'max_iterations': 10})
    self.assertAllClose(expected, image, rtol=1e-5, atol=1e-5)

    # Check batch of k-space data without batch of trajectories (trajectory is
    # equal for all frames in this case).
    image = recon_ops.reconstruct_lstsq(kspace=kspace,
                                        image_shape=image_shape,
                                        trajectory=traj[0, ...],
                                        density=dens,
                                        sensitivities=sens_batch,
                                        optimizer_kwargs={'max_iterations': 10})
    self.assertAllClose(expected, image, rtol=1e-3, atol=1e-3)

  def test_cg_sense_reg(self):
    """Test CG-SENSE with regularization."""
    data = io_util.read_hdf5(
        'tests/data/cardiac_cine_2d_multicoil_radial_kspace.h5')
    kspace = data['kspace']
    sens = data['sens']
    traj = data['traj']
    dens = data['dens']
    image_nonreg = data['image/cg_sense']
    expected_null = data['image/cg_sense_reg_null']
    expected_tavg = data['image/cg_sense_reg_tavg']
    image_shape = sens.shape[-2:]

    # Check batch of k-space data and batch of trajectories.
    tavg = tf.math.reduce_mean(image_nonreg, -3)
    regularizer = convex_ops.ConvexFunctionTikhonov(scale=0.5, prior=tavg)
    image = recon_ops.reconstruct_lstsq(kspace=kspace,
                                        image_shape=image_shape,
                                        trajectory=traj,
                                        density=dens,
                                        sensitivities=sens,
                                        regularizer=regularizer,
                                        optimizer_kwargs={'max_iterations': 10})
    self.assertAllClose(expected_tavg, image)

    regularizer = convex_ops.ConvexFunctionTikhonov(scale=0.5)
    image = recon_ops.reconstruct_lstsq(kspace=kspace,
                                        image_shape=image_shape,
                                        trajectory=traj,
                                        density=dens,
                                        sensitivities=sens,
                                        regularizer=regularizer,
                                        optimizer_kwargs={'max_iterations': 10})
    self.assertAllClose(expected_null, image)

  @parameterized.product(optimizer=('admm', 'lbfgs'),
                         execution_mode=('eager', 'graph'))
  def test_lstsq_grasp(self, optimizer, execution_mode):  # pylint: disable=missing-param-doc
    """Test GRASP reconstruction."""
    # Load data.
    data = io_util.read_hdf5(
        'tests/data/liver_dce_2d_multicoil_radial_kspace.h5')
    kspace = data['kspace']
    traj = data['traj']
    dens = data['dens']
    sens = data['sens']
    expected = data[f'image/tv/{optimizer}/i4']

    def _reconstruct():
      regularizer = convex_ops.ConvexFunctionTotalVariation(
          scale=0.001, domain_shape=[28, 384, 384],
          axis=-3, dtype=tf.complex64)
      return recon_ops.reconstruct_lstsq(
          kspace,
          image_shape=[384, 384],
          extra_shape=[28],
          trajectory=traj,
          density=dens,
          sensitivities=sens,
          regularizer=regularizer,
          optimizer=optimizer,
          optimizer_kwargs=dict(max_iterations=4))
    if execution_mode == 'eager':
      fn = _reconstruct
    elif execution_mode == 'graph':
      fn = tf.function(_reconstruct)
    else:
      raise ValueError(f"Invalid execution mode: {execution_mode}")
    image = fn()

    self.assertAllClose(expected, image, rtol=1e-5, atol=1e-5)

  @parameterized.product(optimizer=('cg', 'lbfgs'),
                         execution_mode=('eager', 'graph'),
                         return_optimizer_state=(True, False))
  def test_lstsq_return_optimizer_state(self, optimizer, execution_mode,  # pylint: disable=missing-param-doc
                                        return_optimizer_state):
    """Test the return_optimizer_state argument of `lstsq`."""
    kspace = tf.dtypes.complex(
        tf.random.stateless_uniform([32, 32], [2, 3]),
        tf.random.stateless_uniform([32, 32], [14, 8]))
    mask = tf.cast(
        tf.random.stateless_binomial([32, 32], [1, 1], 1, 0.5), tf.bool)
    kspace *= tf.cast(mask, tf.complex64)

    def _reconstruct():
      return recon_ops.reconstruct_lstsq(
          kspace,
          image_shape=[32, 32],
          mask=mask,
          optimizer=optimizer,
          optimizer_kwargs=dict(max_iterations=1),
          return_optimizer_state=return_optimizer_state)
    if execution_mode == 'eager':
      fn = _reconstruct
    elif execution_mode == 'graph':
      fn = tf.function(_reconstruct)
    else:
      raise ValueError(f"Invalid execution mode: {execution_mode}")

    results = fn()
    if return_optimizer_state:
      self.assertIsInstance(results, tuple)
      self.assertEqual(len(results), 2)
      self.assertIsInstance(results[0], tf.Tensor)
    else:
      self.assertIsInstance(results, tf.Tensor)


class ReconstructPartialKSpaceTest(test_util.TestCase):
  """Tests for `reconstruct_pf` operation."""

  @classmethod
  def setUpClass(cls):
    """Prepare tests."""
    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/recon_ops_data.h5')
    cls.data.update(io_util.read_hdf5('tests/data/recon_ops_data_2.h5'))

  @parameterized.product(method=['zerofill', 'homodyne', 'pocs'],
                         return_complex=[True, False],
                         return_kspace=[True, False])
  def test_pf(self, method, return_complex, return_kspace): # pylint:disable=missing-param-doc
    """Test PF reconstruction."""
    data = io_util.read_hdf5('tests/data/brain_2d_multicoil_kspace.h5')
    full_kspace = data['kspace']
    full_kspace = tf.transpose(full_kspace, [2, 0, 1]) # [coils, x, y]

    # PF subsampling with PF factor = 9/16.
    factors = [1.0, 9 / 16]
    kspace = full_kspace[:, :, :(192 * 9 // 16)]

    result = recon_ops.reconstruct_pf(kspace,
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

  def test_pf_homodyne_weighting(self):
    """Test PF reconstruction with homodyne weighting."""
    kspace = tf.constant([[1, 1, 1, 1]] * 6, tf.complex64)
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='homodyne', return_kspace=True)
    print(result)

if __name__ == '__main__':
  tf.test.main()
