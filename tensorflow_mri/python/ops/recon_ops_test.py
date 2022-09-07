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
"""Tests for module `recon_ops`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from absl.testing import parameterized
import numpy as np
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
    regularizer = convex_ops.ConvexFunctionTikhonov(
        prior=tavg, scale=0.5, dtype=tf.complex64)
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
          domain_shape=[28, 384, 384], axes=-3, scale=0.001, dtype=tf.complex64)
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


class LeastSquaresTest(test_util.TestCase):
  def test_linear_noncart_multicoil_toeplitz(self):
    resolution = 128
    shape = [resolution, resolution]
    image, sens = image_ops.phantom(
        shape=shape, num_coils=4, dtype=tf.complex64, return_sensitivities=True)
    trajectory = traj_ops.radial_trajectory(
        resolution, resolution // 2 + 1, flatten_encoding_dims=True)
    density = traj_ops.radial_density(
        resolution, resolution // 2 + 1, flatten_encoding_dims=True)
    kspace = fft_ops.nufft(image, trajectory)

    recon = recon_ops.reconstruct_lstsq(
        kspace, shape, trajectory=trajectory, density=density,
        sensitivities=sens, toeplitz_nufft=False)

    recon_toep = recon_ops.reconstruct_lstsq(
        kspace, shape, trajectory=trajectory, density=density,
        sensitivities=sens, toeplitz_nufft=True)

    self.assertAllClose(recon, recon_toep, rtol=1e-2, atol=1e-2)

  def test_cs_tv_noncart_toeplitz(self):
    resolution = 128
    shape = [resolution, resolution]
    image, sens = image_ops.phantom(
        shape=shape, num_coils=4, dtype=tf.complex64, return_sensitivities=True)
    trajectory = traj_ops.radial_trajectory(
        resolution, resolution // 2 + 1, flatten_encoding_dims=True)
    density = traj_ops.radial_density(
        resolution, resolution // 2 + 1, flatten_encoding_dims=True)
    kspace = fft_ops.nufft(image, trajectory)

    regularizer = convex_ops.ConvexFunctionTotalVariation(
        domain_shape=shape, scale=0.2, dtype=tf.complex64)

    recon = recon_ops.reconstruct_lstsq(
        kspace, shape, trajectory=trajectory, density=density,
        sensitivities=sens, regularizer=regularizer, toeplitz_nufft=False)

    recon_toep = recon_ops.reconstruct_lstsq(
        kspace, shape, trajectory=trajectory, density=density,
        sensitivities=sens, regularizer=regularizer, toeplitz_nufft=True)

    self.assertAllClose(recon, recon_toep, rtol=1e-2, atol=1e-2)

  def test_compressed_sensing_total_variation(self):
    shape = [8, 8]
    image = image_ops.phantom(shape=shape, dtype=tf.complex64)
    # The mask below was generated randomly using this code. However, to ensure
    # 100% determinism we hardcode the mask (setting the seed is not enough,
    # because NumPy/TensorFlow random generators do not guarantee
    # reproducibility across different versions).
    # density = traj_ops.density_grid(shape,
    #                                 outer_density=0.1,
    #                                 inner_cutoff=0.15,
    #                                 outer_cutoff=0.75)
    # mask = traj_ops.random_sampling_mask(shape=shape,
    #                                     density=density,
    #                                     seed=[1234, 5678])
    mask = [[False, False,  True, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False,  True, False, False,  True, False, False, False],
            [False, False,  True,  True,  True,  True, False, False],
            [ True, False, False,  True,  True,  True, False, False],
            [False, False, False,  True,  True, False, False, False],
            [False, False, False, False, False,  True, False, False],
            [False, False, False, False, False,  True, False, False]]

    kspace = fft_ops.fftn(image, shift=True)
    kspace *= tf.cast(mask, tf.complex64)

    regularizer = convex_ops.ConvexFunctionTotalVariation(
        shape, scale=0.5, dtype=tf.complex64)
    recon = recon_ops.reconstruct_lstsq(
        kspace, shape, mask=mask, regularizer=regularizer)

    expected = [
       [0.47365025+0.35727844j, 0.47362387+0.3574852j ,
        0.47102705+0.35773143j, 1.5927275 +0.03364388j,
        1.5925304 +0.03386804j, 1.5924214 +0.03396048j,
        0.63141483+0.17710306j, 0.47365424+0.24591646j],
       [0.49422392+0.24302912j, 0.49413317+0.24296162j,
        1.592535  +0.03359434j, 1.5924664 +0.03376158j,
        1.5924141 +0.03393522j, 1.5923434 +0.03402908j,
        0.63145584+0.1770497j , 0.47367054+0.24595489j],
       [0.42961526-0.03568281j, 0.42958477-0.03572694j,
        1.5923353 +0.03373172j, 1.5923089 +0.0338942j ,
        1.592305  +0.0341074j , 1.5922856 +0.03428759j,
        0.52442116+0.04065578j, 0.52437794+0.04063034j],
       [0.42960957-0.03569088j, 0.4295284 -0.03570621j,
        1.410736  -0.01676844j, 1.4108166 -0.01677309j,
        1.4109547 -0.01682312j, 1.4110198 -0.01687882j,
        0.5244667 +0.04046582j, 0.52443236+0.04049375j],
       [0.429612  -0.03571485j, 0.42952585-0.03567624j,
        1.4107622 -0.01665446j, 1.4108161 -0.01667437j,
        1.4109243 -0.01672381j, 1.410965  -0.01678341j,
        0.5245532 +0.04018233j, 0.5245165 +0.04022514j],
       [0.4297124 -0.03573817j, 0.42958248-0.03571387j,
        1.4109347 -0.01656535j, 1.4108994 -0.01657733j,
        1.4109379 -0.01667013j, 1.4108921 -0.01673685j,
        0.5246046 +0.03988812j, 0.5245441 +0.04001619j],
       [0.47826728-0.28862503j, 0.6072884 -0.22691496j,
        1.4110522 -0.01650939j, 1.4109974 -0.01655797j,
        1.4109544 -0.01662711j, 1.4108546 -0.01663312j,
        0.5247626 +0.03954637j, 0.5248155 +0.03936924j],
       [0.47841653-0.2886274j , 0.60729015-0.2270134j ,
        1.4111687 -0.01653639j, 1.4111212 -0.01657057j,
        1.4110851 -0.01657632j, 0.3350836 -0.51524085j,
        0.33502012-0.51509833j, 0.33495024-0.514987j  ]]
    self.assertAllClose(expected, recon, rtol=1e-5, atol=1e-5)

  def test_compressed_sensing_l1_wavelet(self):
    shape = [8, 8]
    image = image_ops.phantom(shape=shape, dtype=tf.complex64)
    # The mask below was generated randomly using this code. However, to ensure
    # 100% determinism we hardcode the mask (setting the seed is not enough,
    # because NumPy/TensorFlow random generators do not guarantee
    # reproducibility across different versions).
    # density = traj_ops.density_grid(shape,
    #                                 outer_density=0.1,
    #                                 inner_cutoff=0.15,
    #                                 outer_cutoff=0.75)
    # mask = traj_ops.random_sampling_mask(shape=shape,
    #                                      density=density,
    #                                      seed=[1234, 5678])
    mask = [[False, False,  True, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False,  True, False, False,  True, False, False, False],
            [False, False,  True,  True,  True,  True, False, False],
            [ True, False, False,  True,  True,  True, False, False],
            [False, False, False,  True,  True, False, False, False],
            [False, False, False, False, False,  True, False, False],
            [False, False, False, False, False,  True, False, False]]

    kspace = fft_ops.fftn(image, shift=True)
    kspace *= tf.cast(mask, tf.complex64)

    regularizer = convex_ops.ConvexFunctionL1Wavelet(
        shape, 'haar', level=2, scale=0.5, dtype=tf.complex64)
    recon = recon_ops.reconstruct_lstsq(
        kspace, shape, mask=mask, regularizer=regularizer)

    expected = [
       [ 0.34179926+0.30613357j,  0.3417428 +0.30630958j,
         0.6689252 +0.5598738j ,  0.6686945 +0.5602565j ,
         1.2340636 -0.2148357j ,  1.23368   -0.2146943j ,
         0.15142666+0.2750988j ,  0.08611039+0.3179876j ],
       [ 0.3420096 +0.30618003j,  0.34171504+0.3062238j ,
         1.9308203 +0.03409361j,  1.930787  +0.03424059j,
         2.606547  +0.61097986j,  2.6062891 +0.6112608j ,
         0.1514751 +0.27488872j,  0.08607277+0.31800568j],
       [ 0.2832344 -0.29769248j,  0.28288317-0.29749206j,
         1.4607992 -0.36595207j,  1.4605489 -0.3660729j ,
         1.7979488 -0.28135628j,  1.7979484 -0.28116226j,
        -0.00337884-0.18306382j, -0.00363272-0.1827644j ],
       [ 0.28306255-0.29766917j,  0.28288803-0.2973669j ,
         1.0218956 -0.24774893j,  1.0215473 -0.24767056j,
         1.7981545 -0.2812703j ,  1.797894  -0.28125632j,
        -0.00354373-0.18284853j, -0.00361615-0.18267009j],
       [-0.02226475+0.28652188j, -0.02214793+0.2866598j ,
         1.4652784 +0.33295962j,  1.4650053 +0.33324894j,
         0.6805991 +0.12670842j,  0.680208  +0.12687585j,
         0.33172938+0.1782302j ,  0.3314214 +0.17820854j],
       [-0.021961  +0.2864609j , -0.0221408 +0.28642035j,
         1.4651511 +0.3330814j ,  1.4650812 +0.33326146j,
         1.3525531 +0.33497405j,  1.3522899 +0.33521318j,
         0.33176833+0.17799535j,  0.33142558+0.17818964j],
       [ 0.1972605 -0.33466882j,  0.19683616-0.33455336j,
         2.0666962 -0.52016133j,  2.0664406 -0.52020437j,
         1.922491  +0.19525945j,  1.922551  +0.19537224j,
         0.42206743-0.24393967j,  0.42183053-0.24355185j],
       [ 0.19710806-0.3347677j ,  0.19676188-0.334543j  ,
         1.3019115 -0.0558873j ,  1.3016343 -0.05568542j,
         0.29104483-0.5771697j ,  0.2908303 -0.57718724j,
         0.42195508-0.2436194j ,  0.421938  -0.24345556j]]
    self.assertAllClose(expected, recon, rtol=1e-5, atol=1e-5)

  def test_compressed_sensing_l1_wavelet_multicoi(self):
    shape = [8, 8]
    image, sens = image_ops.phantom(
        shape=shape, num_coils=4, dtype=tf.complex64, return_sensitivities=True)
    # The mask below was generated randomly using this code. However, to ensure
    # 100% determinism we hardcode the mask (setting the seed is not enough,
    # because NumPy/TensorFlow random generators do not guarantee
    # reproducibility across different versions).
    # density = traj_ops.density_grid(shape,
    #                                 outer_density=0.1,
    #                                 inner_cutoff=0.15,
    #                                 outer_cutoff=0.75)
    # mask = traj_ops.random_sampling_mask(shape=shape,
    #                                      density=density,
    #                                      seed=[1234, 5678])
    mask = [[False, False,  True, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False,  True, False, False,  True, False, False, False],
            [False, False,  True,  True,  True,  True, False, False],
            [ True, False, False,  True,  True,  True, False, False],
            [False, False, False,  True,  True, False, False, False],
            [False, False, False, False, False,  True, False, False],
            [False, False, False, False, False,  True, False, False]]

    kspace = fft_ops.fftn(image, axes=[-2, -1], shift=True)
    kspace *= tf.cast(mask, tf.complex64)

    regularizer = convex_ops.ConvexFunctionL1Wavelet(
        shape, 'haar', level=2, scale=0.5, dtype=tf.complex64)
    recon = recon_ops.reconstruct_lstsq(
        kspace, shape, mask=mask, sensitivities=sens, regularizer=regularizer)

    expected = [
       [0.40788233+0.09244551j, 0.40777513+0.09250628j,
        1.5386903 -0.02036293j, 1.538751  -0.02051309j,
        0.8836167 -0.31155926j, 0.88366103-0.31144089j,
        0.2749335 +0.04516386j, 0.274903  +0.04511612j],
       [0.40785035+0.09247903j, 0.40780166+0.09250979j,
        1.5386543 -0.02045235j, 1.5386062 -0.02044583j,
        2.443501  +0.33469176j, 2.4435632 +0.33456582j,
        0.27481675+0.04501313j, 0.27484408+0.04512517j],
       [0.34545314-0.01897199j, 0.34548455-0.01885714j,
        1.8099102 -0.14981847j, 1.8099062 -0.14989099j,
        2.2956266 +0.04863668j, 2.2955754 +0.04873982j,
        0.24760048-0.09828575j, 0.24764319-0.09838261j],
       [0.34555954-0.01890662j, 0.3456543 -0.01903058j,
        1.1428082 -0.11400633j, 1.1427848 -0.1138969j ,
        0.9767417 -0.3125655j , 0.97675574-0.31256628j,
        0.24743372-0.09847797j, 0.24731372-0.09844565j],
       [0.03984646+0.1727991j , 0.03976073+0.17282495j,
        1.3216747 +0.18094535j, 1.3216183 +0.18089586j,
        1.1143698 +0.14693272j, 1.1144072 +0.1468962j ,
        0.26738882+0.00661303j, 0.2675456 +0.0066012j ],
       [0.03972306+0.1728731j , 0.03967386+0.17292835j,
        1.3215551 +0.18083024j, 1.3214544 +0.18088897j,
        1.2834618 +0.17041454j, 1.2836148 +0.17026538j,
        0.26747093+0.00668541j, 0.26741967+0.00680417j],
       [0.2772537 -0.12690209j, 0.27725938-0.12695788j,
        2.7376573 -0.6365766j , 2.7378259 -0.6365111j ,
        2.4262521 +0.44118595j, 2.4260592 +0.4412532j ,
        0.35059568-0.09451335j, 0.35067502-0.09468891j],
       [0.27730492-0.12696016j, 0.27743122-0.12700506j,
        0.3810194 +0.39855784j, 0.3810459 +0.3985247j ,
        0.13833281-0.32683632j, 0.13833402-0.3267697j ,
        0.35064778-0.09473409j, 0.35055292-0.09468822j]]
    self.assertAllClose(expected, recon, rtol=1e-5, atol=1e-5)

  def test_admm_without_regularizer(self):
    shape = [64, 64]
    image = image_ops.phantom(shape=shape, dtype=tf.complex64)
    kspace = fft_ops.fftn(image, axes=[-2, -1], shift=True)
    with self.assertRaisesRegex(
        ValueError, "optimizer 'admm' requires a regularizer"):
      recon_ops.reconstruct_lstsq(kspace, shape, optimizer='admm')


class ReconstructPartialKSpaceTest(test_util.TestCase):
  """Tests for `reconstruct_pf` operation."""
  # pylint: disable=missing-function-docstring
  def test_pf_zerofill(self):
    kspace = tf.constant([[1, 1, 1, 1]] * 6, tf.complex64)
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='zerofill')
    expected = [[0.        , 0.        , 0.        , 0.        ],
                [0.        , 0.        , 0.09567086, 0.        ],
                [0.        , 0.        , 0.17677669, 0.        ],
                [0.        , 0.        , 0.23096989, 0.        ],
                [0.        , 0.        , 0.75      , 0.        ],
                [0.        , 0.        , 0.23096989, 0.        ],
                [0.        , 0.        , 0.17677669, 0.        ],
                [0.        , 0.        , 0.09567086, 0.        ]]
    self.assertAllClose(expected, result)

  def test_pf_zerofill_return_kspace(self):
    kspace = tf.constant([[1, 1, 1, 1]] * 6, tf.complex64)
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='zerofill',
                                      return_kspace=True)
    expected = [[0.4502719 +0.0000000e+00j, 0.4502719 +0.0000000e+00j,
                 0.4502719 +0.0000000e+00j, 0.4502719 +0.0000000e+00j],
                [0.5586583 +1.2828956e-09j, 0.5586583 +1.2828956e-09j,
                 0.5586583 +1.2828956e-09j, 0.5586583 +1.2828956e-09j],
                [0.39644662+0.0000000e+00j, 0.39644662+0.0000000e+00j,
                 0.39644662+0.0000000e+00j, 0.39644662+0.0000000e+00j],
                [0.9413417 +1.2828956e-09j, 0.9413417 +1.2828956e-09j,
                 0.9413417 +1.2828956e-09j, 0.9413417 +1.2828956e-09j],
                [1.756835  +0.0000000e+00j, 1.756835  +0.0000000e+00j,
                 1.756835  +0.0000000e+00j, 1.756835  +0.0000000e+00j],
                [0.9413417 -1.2828956e-09j, 0.9413417 -1.2828956e-09j,
                 0.9413417 -1.2828956e-09j, 0.9413417 -1.2828956e-09j],
                [0.39644662+0.0000000e+00j, 0.39644662+0.0000000e+00j,
                 0.39644662+0.0000000e+00j, 0.39644662+0.0000000e+00j],
                [0.5586583 -1.2828956e-09j, 0.5586583 -1.2828956e-09j,
                 0.5586583 -1.2828956e-09j, 0.5586583 -1.2828956e-09j]]
    self.assertAllClose(expected, result)

  def test_pf_zerofill_with_phase(self):
    kspace = tf.dtypes.complex(
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32),
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32))
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='zerofill')
    expected = [[0.        , 0.        , 0.        , 0.        ],
                [0.        , 0.        , 0.13529903, 0.        ],
                [0.        , 0.        , 0.25      , 0.        ],
                [0.        , 0.        , 0.32664075, 0.        ],
                [0.        , 0.        , 1.0606601 , 0.        ],
                [0.        , 0.        , 0.32664075, 0.        ],
                [0.        , 0.        , 0.25      , 0.        ],
                [0.        , 0.        , 0.13529903, 0.        ]]
    self.assertAllClose(expected, result)

  def test_pf_zerofill_preserve_phase(self):
    kspace = tf.dtypes.complex(
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32),
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32))
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='zerofill',
                                      preserve_phase=True)
    expected = [[ 0.        +0.j        ,  0.        -0.j        ,
                  0.        -0.j        ,  0.        -0.j        ],
                [ 0.        +0.j        ,  0.        -0.j        ,
                 -0.09567086-0.09567086j,  0.        -0.j        ],
                [ 0.        -0.j        ,  0.        +0.j        ,
                  0.17677669+0.1767767j ,  0.        +0.j        ],
                [ 0.        -0.j        ,  0.        +0.j        ,
                  0.23096989+0.2309699j ,  0.        +0.j        ],
                [ 0.        -0.j        ,  0.        +0.j        ,
                  0.74999994+0.75j      ,  0.        +0.j        ],
                [ 0.        -0.j        ,  0.        +0.j        ,
                  0.23096989+0.2309699j ,  0.        +0.j        ],
                [ 0.        -0.j        ,  0.        +0.j        ,
                  0.17677669+0.1767767j ,  0.        +0.j        ],
                [ 0.        +0.j        ,  0.        -0.j        ,
                 -0.09567086-0.09567086j,  0.        -0.j        ]]
    self.assertAllClose(expected, result)

  def test_pf_zerofill_preserve_phase_return_kspace(self):
    kspace = tf.dtypes.complex(
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32),
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32))
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='zerofill',
                                      preserve_phase=True, return_kspace=True)
    expected = [[0.83295524+0.83295536j, 0.83295524+0.83295536j,
                 0.83295524+0.83295536j, 0.83295524+0.83295536j],
                [0.28806016+0.28806016j, 0.28806016+0.28806016j,
                 0.28806016+0.28806016j, 0.28806016+0.28806016j],
                [0.39644656+0.3964466j , 0.39644656+0.3964466j ,
                 0.39644656+0.3964466j , 0.39644656+0.3964466j ],
                [1.2119398 +1.2119398j , 1.2119398 +1.2119398j ,
                 1.2119398 +1.2119398j , 1.2119398 +1.2119398j ],
                [1.3741513 +1.3741515j , 1.3741513 +1.3741515j ,
                 1.3741513 +1.3741515j , 1.3741513 +1.3741515j ],
                [1.2119397 +1.2119398j , 1.2119397 +1.2119398j ,
                 1.2119397 +1.2119398j , 1.2119397 +1.2119398j ],
                [0.39644656+0.3964466j , 0.39644656+0.3964466j ,
                 0.39644656+0.3964466j , 0.39644656+0.3964466j ],
                [0.28806013+0.28806016j, 0.28806013+0.28806016j,
                 0.28806013+0.28806016j, 0.28806013+0.28806016j]]
    self.assertAllClose(expected, result)

  def test_pf_homodyne(self):
    kspace = tf.constant([[1, 1, 1, 1]] * 6, tf.complex64)
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='homodyne')
    expected = np.zeros((8, 4), dtype=np.float32)
    expected[4, 2] = 1.0
    self.assertAllClose(expected, result)

  def test_pf_homodyne_step_weighting(self):
    kspace = tf.constant([[1, 1, 1, 1]] * 6, tf.complex64)
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='homodyne',
                                      weighting_fn='step')
    expected = np.zeros((8, 4), dtype=np.float32)
    expected[4, 2] = 1.0
    self.assertAllClose(expected, result)

  def test_pf_homodyne_return_kspace(self):
    kspace = tf.constant([[1, 1, 1, 1]] * 6, tf.complex64)
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='homodyne',
                                      return_kspace=True)
    expected = np.ones((8, 4), dtype=np.complex64)
    self.assertAllClose(expected, result)

  def test_pf_homodyne_with_phase(self):
    kspace = tf.dtypes.complex(
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32),
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32))
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='homodyne')
    expected = np.zeros((8, 4), dtype=np.float32)
    expected[4, 2] = np.sqrt(2)
    self.assertAllClose(expected, result)

  def test_pf_homodyne_preserve_phase(self):
    kspace = tf.dtypes.complex(
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32),
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32))
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='homodyne',
                                      preserve_phase=True)
    expected = np.zeros((8, 4), dtype=np.complex64)
    expected[4, 2] = 1.0 + 1.0j
    self.assertAllClose(expected, result)

  def test_pf_homodyne_preserve_phase_return_kspace(self):
    kspace = tf.dtypes.complex(
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32),
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32))
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='homodyne',
                                      preserve_phase=True, return_kspace=True)
    expected = np.ones((8, 4)) + 1j * np.ones((8, 4))
    self.assertAllClose(expected, result)

  def test_pf_pocs(self):
    kspace = tf.constant([[1, 1, 1, 1]] * 6, tf.complex64)
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='pocs')
    expected = np.zeros((8, 4), dtype=np.float32)
    expected[4, 2] = 1.0
    self.assertAllClose(expected, result, rtol=1e-3, atol=1e-3)

  def test_pf_pocs_return_kspace(self):
    kspace = tf.constant([[1, 1, 1, 1]] * 6, tf.complex64)
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='pocs',
                                      return_kspace=True)
    expected = np.ones((8, 4), dtype=np.complex64)
    self.assertAllClose(expected, result, rtol=1e-3, atol=1e-3)

  def test_pf_pocs_with_phase(self):
    kspace = tf.dtypes.complex(
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32),
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32))
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='pocs')
    expected = np.zeros((8, 4), dtype=np.float32)
    expected[4, 2] = np.sqrt(2)
    self.assertAllClose(expected, result, rtol=1e-3, atol=1e-3)

  def test_pf_pocs_preserve_phase(self):
    kspace = tf.dtypes.complex(
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32),
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32))
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='pocs',
                                      preserve_phase=True)
    expected = np.zeros((8, 4), dtype=np.complex64)
    expected[4, 2] = 1.0 + 1.0j
    self.assertAllClose(expected, result, rtol=1e-3, atol=1e-3)

  def test_pf_pocs_preserve_phase_return_kspace(self):
    kspace = tf.dtypes.complex(
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32),
        tf.constant([[1, 1, 1, 1]] * 6, tf.float32))
    factors = [0.75, 1.0]
    result = recon_ops.reconstruct_pf(kspace, factors, method='pocs',
                                      preserve_phase=True, return_kspace=True)
    expected = np.ones((8, 4)) + 1j * np.ones((8, 4))
    self.assertAllClose(expected, result, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
