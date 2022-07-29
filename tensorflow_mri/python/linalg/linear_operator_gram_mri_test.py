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
"""Tests for module `linear_operator_gram_mri`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_gram_mri
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.util import test_util


class LinearOperatorGramMRITest(test_util.TestCase):
  @parameterized.product(batch=[False, True], extra=[False, True],
                         toeplitz_nufft=[False, True])
  def test_general(self, batch, extra, toeplitz_nufft):
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
    if batch:
      image = tf.stack([image, image * 2])
      if extra:
        extra_shape = [2]
      else:
        extra_shape = None
    else:
      extra_shape = None

    linop = linear_operator_gram_mri.LinearOperatorMRI(
        image_shape, extra_shape=extra_shape,
        trajectory=trajectory, density=density,
        sensitivities=sensitivities)
    linop_gram = linear_operator_gram_mri.LinearOperatorGramMRI(
        image_shape, extra_shape=extra_shape,
        trajectory=trajectory, density=density,
        sensitivities=sensitivities, toeplitz_nufft=toeplitz_nufft)

    # Test shapes.
    expected_domain_shape = image_shape
    if extra_shape is not None:
      expected_domain_shape = extra_shape + image_shape
    self.assertAllClose(expected_domain_shape, linop_gram.domain_shape)
    self.assertAllClose(expected_domain_shape, linop_gram.domain_shape_tensor())
    self.assertAllClose(expected_domain_shape, linop_gram.range_shape)
    self.assertAllClose(expected_domain_shape, linop_gram.range_shape_tensor())

    # Test transform.
    expected = linop.transform(linop.transform(image), adjoint=True)
    self.assertAllClose(expected, linop_gram.transform(image),
                        rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
