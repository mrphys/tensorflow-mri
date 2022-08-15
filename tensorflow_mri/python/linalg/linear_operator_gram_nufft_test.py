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
"""Tests for module `linear_operator_gram_nufft`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_gram_nufft
from tensorflow_mri.python.linalg import linear_operator_nufft
from tensorflow_mri.python.ops import geom_ops
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.util import test_util


class LinearOperatorGramNUFFTTest(test_util.TestCase):
  @parameterized.product(
      density=[False, True],
      norm=[None, 'ortho'],
      toeplitz=[False, True],
      batch=[False, True]
  )
  def test_general(self, density, norm, toeplitz, batch):
    with tf.device('/cpu:0'):
      image_shape = (128, 128)
      image = image_ops.phantom(shape=image_shape, dtype=tf.complex64)
      trajectory = traj_ops.radial_trajectory(
          128, 129, flatten_encoding_dims=True)
      if density is True:
        density = traj_ops.radial_density(
            128, 129, flatten_encoding_dims=True)
      else:
        density = None

      # If testing batches, create new inputs to generate a batch.
      if batch:
        image = tf.stack([image, image * 0.5])
        trajectory = tf.stack([
            trajectory, geom_ops.rotate_2d(trajectory, [np.pi / 2])])
        if density is not None:
          density = tf.stack([density, density])

      linop = linear_operator_nufft.LinearOperatorNUFFT(
          image_shape, trajectory=trajectory, density=density, norm=norm)
      linop_gram = linear_operator_gram_nufft.LinearOperatorGramNUFFT(
          image_shape, trajectory=trajectory, density=density, norm=norm,
          toeplitz=toeplitz)

      recon = linop.transform(linop.transform(image), adjoint=True)
      recon_gram = linop_gram.transform(image)

      if norm is None:
        # Reduce the magnitude of these values to avoid the need to use a large
        # tolerance.
        recon /= tf.cast(tf.math.reduce_prod(image_shape), tf.complex64)
        recon_gram /= tf.cast(tf.math.reduce_prod(image_shape), tf.complex64)

      self.assertAllClose(recon, recon_gram, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
