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
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import convex_ops
from tensorflow_mri.python.util import test_util


@test_util.run_all_in_graph_and_eager_modes
class ConvexFunctionL1NormTest(test_util.TestCase):
  """Tests for `ConvexFunctionL1Norm`."""
  # pylint: disable=missing-function-docstring
  @parameterized.parameters(
      # x, scale, expected
      ([3., 4.], 2.0, 14.0),
      ([3., 4.], 1.0, 7.0),
      ([[6., 8.], [4., 3.]], 1.0, [14.0, 7.0]),
      ([3., -4.], 1.0, 7.0)
  )
  def test_call(self, x, scale, expected):
    f = convex_ops.ConvexFunctionL1Norm(scale=scale)
    self.assertAllClose(expected, f(x))

  @parameterized.parameters(
      # x, scale, expected
      ([3., 4.], 2.0, [1., 2.]),
      ([3., 4.], 5.0, [0., 0.]),
      ([[6., 8.], [4., 3.]], 1.0, [[5., 7.], [3., 2.]]),
      ([3., -4.], 2.0, [1., -2.])
  )
  def test_prox(self, x, scale, expected):
    f = convex_ops.ConvexFunctionL1Norm(scale=scale)
    self.assertAllClose(expected, f.prox(x))

  def test_conj(self):
    f = convex_ops.ConvexFunctionL1Norm()
    self.assertIsInstance(f.conj(), convex_ops.ConvexFunctionIndicatorBall)


@test_util.run_all_in_graph_and_eager_modes
class ConvexFunctionL2NormTest(test_util.TestCase):
  """Tests for `ConvexFunctionL2Norm`."""
  # pylint: disable=missing-function-docstring
  @parameterized.parameters(
      # x, scale, expected
      ([3., 4.], 2.0, 10.0),
      ([3., 4.], 1.0, 5.0),
      ([[6., 8.], [4., 3.]], 1.0, [10.0, 5.0])
  )
  def test_call(self, x, scale, expected):
    f = convex_ops.ConvexFunctionL2Norm(scale=scale)
    self.assertAllClose(expected, f(x))

  @parameterized.parameters(
      # x, scale, expected
      ([3., 4.], 2.0, [1.8, 2.4]),
      ([3., 4.], 1.0, [2.4, 3.2]),
      ([[6., 8.], [4., 3.]], 1.0, [[5.4, 7.2], [3.2, 2.4]])
  )
  def test_prox(self, x, scale, expected):
    f = convex_ops.ConvexFunctionL2Norm(scale=scale)
    self.assertAllClose(expected, f.prox(x))


@test_util.run_all_in_graph_and_eager_modes
class ConvexFunctionL2NormSquaredTest(test_util.TestCase):
  """Tests for `ConvexFunctionL2NormSquared`."""
  # pylint: disable=missing-function-docstring
  @parameterized.parameters(
      # x, scale, expected
      ([1., 2., 3.], 1.0, 14.0),
      ([1., 2., 3.], 0.5, 7.0),
      ([[1., 2.], [3., 4.]], 1.0, [5.0, 25.0])
  )
  def test_call(self, x, scale, expected):
    f = convex_ops.ConvexFunctionL2NormSquared(scale=scale)
    self.assertAllClose(expected, f(x))

  @parameterized.parameters(
      # x, scale, expected
      ([1.5, 3.0, 4.5], 1.0, [0.5, 1.0, 1.5]),
      ([1.0, 2.0, 3.0], 2.0, [0.2, 0.4, 0.6])
  )
  def test_prox(self, x, scale, expected):
    f = convex_ops.ConvexFunctionL2NormSquared(scale=scale)
    self.assertAllClose(expected, f.prox(x))


@test_util.run_all_in_graph_and_eager_modes
class ConvexFunctionIndicatorL1BallTest(test_util.TestCase):
  """Tests for `ConvexFunctionIndicatorL1Ball`."""
  # pylint: disable=missing-function-docstring
  @parameterized.parameters(
      # x, scale, expected
      ([1.0], 1.0, 0.0),
      ([0.5], 1.0, 0.0),
      ([1.5], 1.0, np.inf),
      ([0.5], 2.0, 0.0),
      ([1.5], 2.0, np.inf),
      ([1.5, -2.0], 1.0, np.inf),
      ([-0.4, 0.8], 1.0, np.inf),
      ([[1.0, 0.75], [-3., 4.]], 1.0, [np.inf, np.inf]),
      ([[0.1, -0.5, -0.2], [1., 4., -2.]], 1.0, [0.0, np.inf])
  )
  def test_call(self, x, scale, expected):
    f = convex_ops.ConvexFunctionIndicatorL1Ball(scale=scale)
    self.assertAllClose(expected, f(x))

  def test_conj(self):
    f = convex_ops.ConvexFunctionIndicatorL1Ball()
    self.assertIsInstance(f.conj(), convex_ops.ConvexFunctionNorm)
    self.assertEqual(np.inf, f.conj()._order)  # pylint: disable=protected-access


@test_util.run_all_in_graph_and_eager_modes
class ConvexFunctionIndicatorL2BallTest(test_util.TestCase):
  """Tests for `ConvexFunctionIndicatorL2Ball`."""
  # pylint: disable=missing-function-docstring
  @parameterized.parameters(
      # x, scale, expected
      ([1.0], 1.0, 0.0),
      ([0.5], 1.0, 0.0),
      ([1.5], 1.0, np.inf),
      ([0.5], 2.0, 0.0),
      ([1.5], 2.0, np.inf),
      ([1.5, -2.0], 1.0, np.inf),
      ([-0.4, 0.8], 1.0, 0.0),
      ([[1.0, 0.75], [-3., 4.]], 1.0, [np.inf, np.inf]),
      ([[0.1, -0.5, -0.2], [1., 4., -2.]], 1.0, [0.0, np.inf])
  )
  def test_call(self, x, scale, expected):
    f = convex_ops.ConvexFunctionIndicatorL2Ball(scale=scale)
    self.assertAllClose(expected, f(x))

  def test_conj(self):
    f = convex_ops.ConvexFunctionIndicatorL2Ball()
    self.assertIsInstance(f.conj(), convex_ops.ConvexFunctionNorm)
    self.assertEqual(2, f.conj()._order)  # pylint: disable=protected-access


@test_util.run_all_in_graph_and_eager_modes
class ConvexFunctionTikhonovTest(test_util.TestCase):
  """Tests for `ConvexFunctionTikhonov`."""
  # pylint: disable=missing-function-docstring
  @parameterized.parameters(
      # x, scale, transform, prior, expected
      ([3.0, 4.0], 2.0, None, None, 50.0),
      ([3.0, 4.0], 0.5, None, None, 12.5),
      ([3.0, 4.0], 0.5, 2.0, None, 50.0),
      ([3.0, 4.0], 2.0, None, [3.0, 2.0], 8.0))
  def test_call(self, x, scale, transform, prior, expected):
    if isinstance(transform, float):
      x = tf.convert_to_tensor(x)
      transform = tf.linalg.LinearOperatorScaledIdentity(x.shape[-1], transform)
    f = convex_ops.ConvexFunctionTikhonov(
        scale=scale,
        transform=transform,
        prior=prior)
    self.assertAllClose(expected, f(x))


class ConvexFunctionTotalVariationTest(test_util.TestCase):
  """Tests for `ConvexFunctionTotalVariation`."""
  # pylint: disable=missing-function-docstring
  def test_call(self):
    x = [[1., 2., 3.],
         [4., 5., 6.]]
    x_flat = tf.reshape(x, [-1])
    reg1 = convex_ops.ConvexFunctionTotalVariation(scale=0.1,
                                                   ndim=[2, 3],
                                                   axis=[0, 1])
    ref1 = 1.3
    res1 = reg1(x_flat)
    self.assertAllClose(res1, ref1)

    reg2 = convex_ops.ConvexFunctionTotalVariation(scale=0.1,
                                                   ndim=[2, 3],
                                                   axis=1)
    res2 = reg2(x_flat)
    ref2 = 0.4
    self.assertAllClose(res2, ref2)

    reg3 = convex_ops.ConvexFunctionTotalVariation(scale=0.5,
                                                   ndim=[3],
                                                   axis=-1)
    res3 = reg3(x)
    ref3 = [1.0, 1.0]
    self.assertAllClose(res3, ref3)


if __name__ == '__main__':
  tf.test.main()
