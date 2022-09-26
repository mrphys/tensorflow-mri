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
"""Signal reconstruction (adjoint)."""

import tensorflow as tf
import tensorflow_nufft as tfft

from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.recon import recon_adjoint
from tensorflow_mri.python.util import io_util
from tensorflow_mri.python.util import test_util


class ReconAdjointTest(test_util.TestCase):
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
    image = recon_adjoint.recon_adjoint_mri(kspace[0, ...], image_shape)
    expected = fft_ops.ifftn(kspace[0, ...], norm='ortho', shift=True)

    self.assertAllClose(expected, image)

    # Test multi-coil.
    image = recon_adjoint.recon_adjoint_mri(
        kspace, image_shape, sensitivities=sens)
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
    image = recon_adjoint.recon_adjoint_mri(kspace[0, ...], image_shape,
                                      trajectory=traj,
                                      density=dens)

    expected = inufft(kspace[0, ...] / tf.cast(dens, tf.complex64), traj)
    expected /= fft_norm_factor

    self.assertAllClose(expected, image)

    # Test multi-coil.
    image = recon_adjoint.recon_adjoint_mri(kspace, image_shape,
                                      trajectory=traj,
                                      density=dens,
                                      sensitivities=sens)
    expected = inufft(kspace / dens, traj)
    expected /= fft_norm_factor
    scale = tf.math.reduce_sum(sens * tf.math.conj(sens), axis=0)
    expected = tf.math.divide_no_nan(
        tf.math.reduce_sum(expected * tf.math.conj(sens), axis=0), scale)

    self.assertAllClose(expected, image)
