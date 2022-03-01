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
"""Tests for module `preproc_layers`."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.layers import preproc_layers
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.util import test_util


class KSpaceResamplingTest(test_util.TestCase):
  """Tests for layer `KSpaceResampling`."""
  @parameterized.product(dens_algo=['geometric', 'radial', 'jackson', 'pipe'])
  @test_util.run_in_graph_and_eager_modes
  def test_radial_2d(self, dens_algo): # pylint: disable=missing-param-doc
    """Test radial 2D configuration."""
    # TODO: remove this check once the NUFFT segfault issue has been resolved.
    if not tf.executing_eagerly():
      self.skipTest("Skipping test due to NUFFT segfault.")

    image_shape = [256, 256]
    image = image_ops.phantom(shape=image_shape)
    image = tf.expand_dims(image, -1)

    layer = preproc_layers.KSpaceResampling(image_shape=image_shape,
                                            traj_type='radial',
                                            views=403,
                                            angle_range='half',
                                            dens_algo=dens_algo)

    result_max = {
        'geometric': 1.4835564,
        'radial': 1.4835564,
        'jackson': 1.5735059,
        'pipe': 1.4818194
    }

    output = layer(image)
    self.assertAllEqual(output.shape, [256, 256, 1])
    self.assertDTypeEqual(output, 'float32')
    self.assertAllClose(tf.math.reduce_max(output), result_max[dens_algo],
                        rtol=1e-4, atol=1e-4)

  @parameterized.product(dens_algo=['geometric', 'radial', 'jackson', 'pipe'])
  def test_radial_2d_impulse(self, dens_algo): # pylint: disable=missing-param-doc
    """Test radial 2D with impulse function."""
    image = tf.scatter_nd([[96, 96]], [1.0], [192, 192])
    image = tf.expand_dims(image, -1)

    layer = preproc_layers.KSpaceResampling(image_shape=[192, 192],
                                            traj_type='radial',
                                            views=89,
                                            angle_range='half',
                                            dens_algo=dens_algo)

    result_max = {
        'geometric': 1.0000054,
        'radial': 1.0000054,
        'jackson': 0.6805566,
        'pipe': 0.6819394
    }

    output = layer(image)
    self.assertAllEqual(output.shape, [192, 192, 1])
    self.assertDTypeEqual(output, 'float32')
    self.assertAllClose(output[96, 96, 0], result_max[dens_algo],
                        rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
