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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.coils import coil_combination
from tensorflow_mri.python.util import test_util


class CoilCombineTest(test_util.TestCase):
  """Tests for coil combination op."""

  @parameterized.product(coil_axis=[0, -1],
                         keepdims=[True, False])
  @test_util.run_in_graph_and_eager_modes
  def test_sos(self, coil_axis, keepdims): # pylint: disable=missing-param-doc
    """Test sum of squares combination."""

    images = self._random_complex((20, 20, 8))

    combined = coil_combination.combine_coils(
        images, coil_axis=coil_axis, keepdims=keepdims)

    ref = tf.math.sqrt(
        tf.math.reduce_sum(images * tf.math.conj(images),
                           axis=coil_axis, keepdims=keepdims))

    self.assertAllEqual(combined.shape, ref.shape)
    self.assertAllClose(combined, ref)


  @parameterized.product(coil_axis=[0, -1],
                         keepdims=[True, False])
  @test_util.run_in_graph_and_eager_modes
  def test_adaptive(self, coil_axis, keepdims): # pylint: disable=missing-param-doc
    """Test adaptive combination."""

    images = self._random_complex((20, 20, 8))
    maps = self._random_complex((20, 20, 8))

    combined = coil_combination.combine_coils(
      images, maps=maps, coil_axis=coil_axis, keepdims=keepdims)

    ref = tf.math.reduce_sum(images * tf.math.conj(maps),
                             axis=coil_axis, keepdims=keepdims)

    ref /= tf.math.reduce_sum(maps * tf.math.conj(maps),
                              axis=coil_axis, keepdims=keepdims)

    self.assertAllEqual(combined.shape, ref.shape)
    self.assertAllClose(combined, ref)

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)

  def _random_complex(self, shape):
    return tf.dtypes.complex(
      tf.random.normal(shape),
      tf.random.normal(shape))


if __name__ == '__main__':
  tf.test.main()
