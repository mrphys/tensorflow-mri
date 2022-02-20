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
"""Tests for module `convex_ops`."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.ops import convex_ops
from tensorflow_mri.python.util import test_util


@test_util.run_all_in_graph_and_eager_modes
class BlockSoftThresholdTest(test_util.TestCase):

  @parameterized.parameters(
      # x, threshold, expected_y
      (5., 5., 0.),
      (2., 5., 0.),
      (-2., 5., 0.),
      (3., 2.5, 0.5),
      (-3., 2.5, -0.5),
      (-1., 1., 0.),
      (-6., 5., -1.),
      (0., 0., 0.),
      ([4., 3.], 2., [2.4, 1.8])
  )
  def test_block_soft_threshold(self, x, threshold, expected_y):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = convex_ops.block_soft_threshold(x, threshold)
    self.assertAllClose(y, expected_y)
  
  @parameterized.parameters(
      # x, threshold, expected_y
      (2. + 0.j, 2., 0. + 0.j),
      (3. + 0.j, 2., 1. + 0.j),
      (0. - 4.j, 3., 0. - 1.j),
      (4. + 3.j, 1., 3.2 + 2.4j)
  )
  def test_block_soft_threshold_complex(self, x, threshold, expected_y):
    x = tf.convert_to_tensor(x, dtype=tf.complex64)
    y = convex_ops.block_soft_threshold(x, threshold)
    self.assertAllClose(y, expected_y)


@test_util.run_all_in_graph_and_eager_modes
class SoftThresholdTest(test_util.TestCase):

  @parameterized.parameters(
      # x, threshold, expected_y
      (5., 5., 0.),
      (2., 5., 0.),
      (-2., 5., 0.),
      (3., 2.5, 0.5),
      (-3., 2.5, -0.5),
      (-1., 1., 0.),
      (-6., 5., -1.),
      (0., 0., 0.),
      ([4., 3.], 2., [2., 1.])
  )
  def test_soft_threshold(self, x, threshold, expected_y):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = convex_ops.soft_threshold(x, threshold)
    self.assertAllClose(y, expected_y)
  
  @parameterized.parameters(
      # x, threshold, expected_y
      (2. + 0.j, 2., 0. + 0.j),
      (3. + 0.j, 2., 1. + 0.j),
      (0. - 4.j, 3., 0. - 1.j),
      (4. + 3.j, 1., 3.2 + 2.4j)
  )
  def test_soft_threshold_complex(self, x, threshold, expected_y):
    x = tf.convert_to_tensor(x, dtype=tf.complex64)
    y = convex_ops.soft_threshold(x, threshold)
    self.assertAllClose(y, expected_y)


class TikhonovRegularizerTest(test_util.TestCase):
  """Tests for `TikhonovRegularizer`."""
  @parameterized.parameters(
      # x, parameter, transform, prior, expected
      ([3.0, 4.0], 2.0, None, None, 10.0),
      ([3.0, 4.0], 0.5, None, None, 2.5),
      ([3.0, 4.0], 0.5, tf.linalg.LinearOperatorScaledIdentity(2, 2.0), None, 5.0),
      ([3.0, 4.0], 2.0, None, [3.0, 2.0], 4.0),
  )
  def test_call(self, x, parameter, transform, prior, expected):

    reg = convex_ops.TikhonovRegularizer(
        parameter=parameter,
        transform=transform,
        prior=prior)

    result = reg(x)
    self.assertAllClose(result, expected)


class TotalVariationRegularizerTest(test_util.TestCase):
  """Tests for `TotalVariationRegularizer`."""
  def test_call(self):
    x = [[1., 2., 3.],
         [4., 5., 6.]]
    x_flat = tf.reshape(x, [-1])
    reg1 = convex_ops.TotalVariationRegularizer(parameter=0.1,
                                                image_shape=[2, 3],
                                                axis=[0, 1])
    ref1 = 1.3
    res1 = reg1(x_flat)
    self.assertAllClose(res1, ref1)
    
    reg2 = convex_ops.TotalVariationRegularizer(parameter=0.1,
                                                image_shape=[2, 3],
                                                axis=1)
    res2 = reg2(x_flat)
    ref2 = 0.4
    self.assertAllClose(res2, ref2)

    reg3 = convex_ops.TotalVariationRegularizer(parameter=0.5,
                                                image_shape=[3],
                                                axis=-1)
    res3 = reg3(x)
    ref3 = [1.0, 1.0]
    self.assertAllClose(res3, ref3)


if __name__ == '__main__':
  tf.test.main()
