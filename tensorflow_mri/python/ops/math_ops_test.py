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
"""Tests for module `math_ops`."""

import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.util import test_util


class ScaleMinmaxTest(test_util.TestCase):
  """Tests for function `scale_by_min_max`."""

  @test_util.run_in_graph_and_eager_modes
  def test_scale_by_min_max(self):
    """Test function `scale_by_min_max`."""

    # Create a random tensor.
    x = tf.random.uniform((5, 3, 4), dtype=tf.dtypes.float32)

    # Create a few parameters to test.
    params = {
      'output_min': (0.0, 1.4, 10.0),
      'output_max': (1.0, 1.5, 16.0)
    }

    # Create combinations of the parameters above.
    values = itertools.product(*params.values())
    params = [dict(zip(params.keys(), v)) for v in values]

    for p in params:

      with self.subTest(**p):

        if p['output_min'] >= p['output_max']:
          with self.assertRaises(tf.errors.InvalidArgumentError):
            y = math_ops.scale_by_min_max(x, **p)
          continue

        # Test op.
        y = math_ops.scale_by_min_max(x, **p)

        self.assertAllClose(tf.reduce_min(y), p['output_min'])
        self.assertAllClose(tf.reduce_max(y), p['output_max'])

  @test_util.run_in_graph_and_eager_modes
  def test_scale_by_min_max_complex(self):
    """Test function `scale_by_min_max` with complex numbers."""
    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter

    # Create a random complex tensor.
    x = tf.dtypes.complex(
      tf.random.uniform((5, 3, 4), minval=-1.0, maxval=1.0),
      tf.random.uniform((5, 3, 4), minval=-1.0, maxval=1.0))

    # Create a few parameters to test.
    params = {
      'output_min': (0.0, 1.4, 10.0),
      'output_max': (1.0, 1.5, 16.0)
    }

    # Create combinations of the parameters above.
    values = itertools.product(*params.values())
    params = [dict(zip(params.keys(), v)) for v in values]

    for p in params:

      with self.subTest(**p):

        if p['output_min'] >= p['output_max']:
          with self.assertRaises(tf.errors.InvalidArgumentError):
            y = math_ops.scale_by_min_max(x, **p)
          continue

        # Test op.
        y = math_ops.scale_by_min_max(x, **p)

        # Check magnitude was scaled.
        self.assertAllClose(
          tf.reduce_min(tf.math.abs(y)), p['output_min'])
        self.assertAllClose(
          tf.reduce_max(tf.math.abs(y)), p['output_max'])

        # Check phase did not change. Weight by magnitude, as for very
        # small magnitudes, phase is noisy.
        mag = tf.math.abs(y)
        self.assertAllClose(
          tf.math.angle(y) * mag, tf.math.angle(x) * mag)
        self.assertAllClose(
          tf.math.angle(y) * mag, tf.math.angle(x) * mag)



@test_util.run_all_in_graph_and_eager_modes
class BlockSoftThresholdTest(test_util.TestCase):
  """Tests for `block_soft_threshold` operator."""
  # pylint: disable=missing-function-docstring
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
    y = math_ops.block_soft_threshold(x, threshold)
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
    y = math_ops.block_soft_threshold(x, threshold)
    self.assertAllClose(y, expected_y)


@test_util.run_all_in_graph_and_eager_modes
class SoftThresholdTest(test_util.TestCase):
  """Tests for `soft_threshold` operator."""
  # pylint: disable=missing-function-docstring
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
    y = math_ops.soft_threshold(x, threshold)
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
    y = math_ops.soft_threshold(x, threshold)
    self.assertAllClose(y, expected_y)


@test_util.run_all_in_graph_and_eager_modes
class IndicatorBallTest(test_util.TestCase):
  """Tests for `indicator_ball` operator."""
  @parameterized.parameters(
      # x, radius, expected_l1, expected_l2, expected_linf
      (-1.0, 1.0, 0.0, 0.0, 0.0),
      (0.95, 1.0, 0.0, 0.0, 0.0),
      (1.05, 1.0, np.inf, np.inf, np.inf),
      (-1.5, 2.0, 0.0, 0.0, 0.0),
      ([1.0], 1.0, 0.0, 0.0, 0.0),
      ([0.5], 1.0, 0.0, 0.0, 0.0),
      ([1.5], 1.0, np.inf, np.inf, np.inf),
      ([1.5], 2.0, 0.0, 0.0, 0.0),
      ([1.5, -2.0], 1.0, np.inf, np.inf, np.inf),
      ([1.5, -2.0], 2.5, np.inf, 0.0, 0.0),
      ([1.5, -2.0], 5.0, 0.0, 0.0, 0.0),
      ([[1.0, 0.75], [-3., 4.]], 1.0,
          [np.inf, np.inf], [np.inf, np.inf], [0.0, np.inf]),
      ([[1.0, 0.75], [-3., 4.]], 2.0,
          [0.0, np.inf], [0.0, np.inf], [0.0, np.inf]),
      ([[0.1, -0.5, -0.2], [1., 4., -2.]], 3.0,
          [0.0, np.inf], [0.0, np.inf], [0.0, np.inf]),
  )  # pylint: disable=missing-function-docstring
  def test_indicator_ball(self, x, radius,
                          expected_l1, expected_l2, expected_linf):
    orders = [1, 2, np.inf]
    expecteds = [expected_l1, expected_l2, expected_linf]
    for order, expected in zip(orders, expecteds):
      with self.subTest(order=order):
        y = math_ops.indicator_ball(x, order=order, radius=radius)
        self.assertAllClose(expected, y)


@test_util.run_all_in_graph_and_eager_modes
class ProjectionOntoBoxTest(test_util.TestCase):
  """Tests for `project_onto_box` operator."""
  @parameterized.parameters(
      # x, lbound, ubound, expected
      (0.5, 0.5, 0.5, 0.5),
      (0.5, 0.1, 0.2, 0.2),
      (0.5, 0.0, 1.5, 0.5),
      (1.5, 0.7, 2.5, 1.5),
      ([4.], 1., 2., [2.]),
      ([0.], -4., 1., [0.]),
      ([0.8], 0.7, 0.9, [0.8]),
      ([1.3], 1.5, 2.5, [1.5]),
      ([-6.3], -2.0, 2.5, [-2.0]),
      ([3., 2.], 1., 2.5, [2.5, 2.]),
      ([1., 2., 3.], 1., 2., [1., 2., 2.]),
      ([[2.1, -1.3], [0.7, 2.2]], -2., 2., [[2., -1.3], [0.7, 2.]])
  )  # pylint: disable=missing-function-docstring
  def test_project_onto_simplex(self, x, lbound, ubound, expected):
    y = math_ops.project_onto_box(x, lower_bound=lbound, upper_bound=ubound)
    self.assertAllClose(expected, y)


@test_util.run_all_in_graph_and_eager_modes
class ProjectionOntoSimplexTest(test_util.TestCase):
  """Tests for `project_onto_simplex` operator."""
  @parameterized.parameters(
      # x, radius, expected
      (0.5, 0.5, 0.5),
      (0.5, 0.25, 0.25),
      (0.5, 1.0, 1.0),
      (1.5, 0.75, 0.75),
      ([4.], 1., [1.]),
      ([0.], 1., [1.]),
      ([0.8], 1., [1.]),
      ([-1.3], 2.5, [2.5]),
      ([6.3], 2.5, [2.5]),
      ([3., 2.], 1., [1., 0.]),
      ([0., 0.], 1., [0.5, 0.5]),
      ([4., 1.], 2., [2., 0.]),
      ([-2.5, 1.], 2., [0., 2.]),
      ([1., 2., 3.], 2., [0.0, 0.5, 1.5]),
      ([[2.1, 1.3], [0.7, 2.2]], 1.6, [[1.2, 0.4], [0.05, 1.55]])
  )  # pylint: disable=missing-function-docstring
  def test_project_onto_simplex(self, x, radius, expected):
    y = math_ops.project_onto_simplex(x, radius=radius)
    self.assertAllClose(expected, y)


@test_util.run_all_in_graph_and_eager_modes
class ProjectionOntoBallTest(test_util.TestCase):
  """Tests for `project_onto_ball` operator."""
  @parameterized.parameters(
      # x, radius, expected_l1, expected_l2, expected_linf
      (0.5, 0.5, 0.5, 0.5, 0.5),
      (0.5, 1.0, 0.5, 0.5, 0.5),
      (1.5, 1.0, 1.0, 1.0, 1.0),
      (-2.5, 2.0, -2.0, -2.0, -2.0),
      ([4.], 1., [1.], [1.], [1.]),
      ([0.], 1., [0.], [0.], [0.]),
      ([0.8], 1., [0.8], [0.8], [0.8]),
      ([-0.8], 1., [-0.8], [-0.8], [-0.8]),
      ([-4.], 1., [-1.], [-1.], [-1.]),
      ([1.3], 2.5, [1.3], [1.3], [1.3]),
      ([6.3], 2.5, [2.5], [2.5], [2.5]),
      ([4., 3.], 1., [1.0, 0.0], [0.8, 0.6], [1.0, 1.0]),
      ([0., 0.5], 1., [0.0, 0.5], [0.0, 0.5], [0.0, 0.5]),
      ([-2.5, 1.], 2., [-1.75, 0.25], [-1.8569534, 0.74278134], [-2., 1.]),
      ([[-3., 4.], [0.0, -1.5]], 1.0, [[0.0, 1.0], [0.0, -1.0]],
          [[-0.6, 0.8], [0.0, -1.0]], [[-1., 1.], [0.0, -1.0]]),
      ([[-3., 4.], [0.0, -1.5]], 2.0, [[-0.5, 1.5], [0.0, -1.5]],
          [[-1.2, 1.6], [0.0, -1.5]], [[-2., 2.], [0.0, -1.5]])
  )  # pylint: disable=missing-function-docstring
  def test_project_onto_ball(self, x, radius,
                                expected_l1, expected_l2, expected_linf):
    orders = [1, 2, np.inf]
    expecteds = [expected_l1, expected_l2, expected_linf]
    for order, expected in zip(orders, expecteds):
      with self.subTest(order=order):
        y = math_ops.project_onto_ball(x, order=order, radius=radius)
        self.assertAllClose(expected, y)


if __name__ == '__main__':
  tf.test.main()
