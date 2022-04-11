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

  @parameterized.parameters(
      ([0.], 1., [0.]),
      ([0.8], 1., [0.8]),
      ([-4.], 1., [-1.]),
      ([4., 3.], 1., [1.0, 0.0]),
      ([0., 0.5], 1., [0.0, 0.5]),
      ([[-3., 4.], [0.0, -1.5]], 1., [[0.0, 1.0], [0.0, -1.0]])
  )
  def test_prox(self, x, scale, expected):
    f = convex_ops.ConvexFunctionIndicatorL1Ball(scale=scale)
    self.assertAllClose(expected, f.prox(x))

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

  @parameterized.parameters(
      ([0.], 1., [0.]),
      ([0.8], 1., [0.8]),
      ([-4.], 1., [-1.]),
      ([4., 3.], 1., [0.8, 0.6]),
      ([0., 0.5], 1., [0.0, 0.5]),
      ([[-3., 4.], [0.0, -1.5]], 1., [[-0.6, 0.8], [0.0, -1.0]])
  )
  def test_prox(self, x, scale, expected):
    f = convex_ops.ConvexFunctionIndicatorL2Ball(scale=scale)
    self.assertAllClose(expected, f.prox(x))

  def test_conj(self):
    f = convex_ops.ConvexFunctionIndicatorL2Ball()
    self.assertIsInstance(f.conj(), convex_ops.ConvexFunctionNorm)
    self.assertEqual(2, f.conj()._order)  # pylint: disable=protected-access


@test_util.run_all_in_graph_and_eager_modes
class ConvexFunctionQuadraticTest(test_util.TestCase):
  """Tests for `ConvexFunctionQuadratic`."""
  def test_quadratic_simple(self):
    """Tests a simple `ConvexFunctionQuadratic`."""
    # Test operator.
    a = tf.linalg.LinearOperatorFullMatrix([[13., 10.], [10., 5.]])
    b = tf.constant([3., 0.])
    c = tf.constant(2.)
    f = convex_ops.ConvexFunctionQuadratic(a, b, c, scale=1.0)

    # Test value.
    x = tf.constant([-2., 1.])

    self.assertAllClose(4.5, f(x))
    self.assertIsInstance(f.shape, tf.TensorShape)
    self.assertIsInstance(f.batch_shape, tf.TensorShape)
    self.assertAllEqual([2], f.shape)
    self.assertAllEqual([], f.batch_shape)
    self.assertIsInstance(f.shape_tensor(), tf.Tensor)
    self.assertIsInstance(f.batch_shape_tensor(), tf.Tensor)
    self.assertAllEqual([], f.batch_shape_tensor())
    self.assertEqual(2, f.ndim)

  def test_quadratic_batch(self):
    """Tests `ConvexFunctionQuadratic` with batch arguments."""
    # Test operator.
    a = tf.linalg.LinearOperatorFullMatrix(
        [[[13., 10.], [10., 5.]], [[5., -1.], [-1., 1.]]])
    b = tf.constant([3., 0.])
    c = tf.constant(2.)
    f = convex_ops.ConvexFunctionQuadratic(a, b, c, scale=2.0)

    # Test value.
    x = tf.constant([-2., 1.])

    self.assertAllClose([9.0, 17.0], f(x))
    self.assertIsInstance(f.shape, tf.TensorShape)
    self.assertIsInstance(f.batch_shape, tf.TensorShape)
    self.assertAllEqual([2, 2], f.shape)
    self.assertAllEqual([2], f.batch_shape)
    self.assertIsInstance(f.shape_tensor(), tf.Tensor)
    self.assertIsInstance(f.batch_shape_tensor(), tf.Tensor)
    self.assertAllEqual([2], f.batch_shape_tensor())
    self.assertEqual(2, f.ndim)

  def test_prox(self):
    """Tests `prox` method."""
    a = tf.linalg.LinearOperatorFullMatrix([[2., 1.], [1., 2.]])
    b = tf.constant([2., -3.])
    c = tf.constant(2.)
    f = convex_ops.ConvexFunctionQuadratic(a, b, c, scale=1.0)

    # Solution: https://www.wolframalpha.com/input?i=minimize+2+%2B+%282+x+-+3y%29+%2B+1%2F2+*+%28x+%282+x+%2B+1+y%29+%2B+y+%281+x+%2B+2+y%29%29+%2B+1%2F2+*+%28%28x+-+4%29%5E2+%2B+%28y+%2B+4%29%5E2%29
    expected = [7/8, -5/8]

    self.assertAllClose(expected, f.prox([4., -4.]))

  def test_prox_scaled(self):
    """Tests `prox` method with scaling factor."""
    a = tf.linalg.LinearOperatorFullMatrix([[2., 1.], [1., 2.]])
    b = tf.constant([2., -3.])
    c = tf.constant(2.)
    # Solution: https://www.wolframalpha.com/input?i=minimize+2+%2B+%282+x+-+3y%29+%2B+1%2F2+*+%28x+%282+x+%2B+1+y%29+%2B+y+%281+x+%2B+2+y%29%29+%2B+1%2F%282+*+1%2F2%29+*+%28%28x+-+4%29%5E2+%2B+%28y+%2B+4%29%5E2%29
    expected = [97/35, -92/35]

    f = convex_ops.ConvexFunctionQuadratic(a, b, c)
    self.assertAllClose(expected, f.prox([4., -4.], scale=0.25))

    f = convex_ops.ConvexFunctionQuadratic(a, b, c, scale=0.25)
    self.assertAllClose(expected, f.prox([4., -4.]))

    f = convex_ops.ConvexFunctionQuadratic(a, b, c, scale=0.5)
    self.assertAllClose(expected, f.prox([4., -4.], scale=0.5))

  def test_prox_kwargs(self):
    """Tests `prox` method with additional kwargs."""
    a = tf.linalg.LinearOperatorFullMatrix([[2., 1.], [1., 2.]])
    b = tf.constant([2., -3.])
    c = tf.constant(2.)
    f = convex_ops.ConvexFunctionQuadratic(a, b, c, scale=1.0)

    # Solution after a single CG iteration.
    expected = [0.90909094, -0.45454547]

    self.assertAllClose(
        expected, f.prox([4., -4.], solver_kwargs={'max_iterations': 1}))


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
