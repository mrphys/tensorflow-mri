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
import odl
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
class ProjectionOntoSimplexTest(test_util.TestCase):
  """Tests for `projection_onto_simplex` operator."""
  @parameterized.parameters(
      # x, radius
      (0.5, 0.5),
      (0.5, 0.25),
      (0.5, 1.0),
      (1.5, 0.75),
      ([4.], 1.),
      ([0.], 1.),
      ([0.8], 1.),
      ([1.3], 2.5),
      ([6.3], 2.5),
      ([3., 2.], 1.),
      ([0., 0.], 1.),
      ([4., 1.], 2.),
      ([1., 2., 3.], 1.),
      ([[2.1, 1.3], [0.7, 2.2]], 1.6)
  )
  def test_projection_onto_simplex(self, x, radius):
    x_flat = np.asarray(x)
    scalar = False
    if x_flat.ndim == 0:
      x_flat = x_flat[None]
      scalar = True
    fn = lambda x: odl.solvers.proj_simplex(odl.vector(x), diameter=radius)
    batch_shape = x_flat.shape[:-1]
    x_flat = x_flat.reshape((-1, x_flat.shape[-1]))
    y_flat = np.apply_along_axis(fn, -1, x_flat)
    expected = y_flat.reshape(batch_shape + (-1,))
    if scalar:
      expected = expected[0]
    y = math_ops.projection_onto_simplex(x, radius=radius)
    self.assertAllClose(expected, y)


def proj_simplex(x, diameter=1, out=None):
    r"""Projection onto simplex.

    Projection onto::

        ``{ x \in X | x_i \geq 0, \sum_i x_i = r}``

    with :math:`r` being the diameter. It is computed by the formula proposed
    in [D+2008].

    Parameters
    ----------
    space : `LinearSpace`
        Space / domain ``X``.
    diameter : positive float, optional
        Diameter of the simplex.

    Returns
    -------
    prox_factory : callable
        Factory for the proximal operator to be initialized.

    Notes
    -----
    The projection onto a simplex is not of closed-form but can be solved by a
    non-iterative algorithm, see [D+2008] for details.

    References
    ----------
    [D+2008] Duchi, J., Shalev-Shwartz, S., Singer, Y., and Chandra, T.
    *Efficient Projections onto the L1-ball for Learning in High dimensions*.
    ICML 2008, pp. 272-279. http://doi.org/10.1145/1390156.1390191

    See Also
    --------
    proj_l1 : projection onto l1-norm ball
    """
    # if out is None:
    #     out = x.space.element()

    # sort values in descending order
    x_sor = x.flatten()
    x_sor.sort()
    x_sor = x_sor[::-1]

    # find critical index
    j = np.arange(1, x.size + 1)
    x_avrg = (1 / j) * (np.cumsum(x_sor) - diameter)
    crit = x_sor - x_avrg
    i = np.argwhere(crit >= 0).flatten().max()
    print(i)

    # output is a shifted and thresholded version of the input
    return np.maximum(x - x_avrg[i], 0)

    # return out

import numpy as np
def euclidean_proj_simplex(v, s=1):
  """ Compute the Euclidean projection on a positive simplex
  Solves the optimisation problem (using the algorithm from [1]):
      min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
  Parameters
  ----------
  v: (n,) numpy array,
      n-dimensional vector to project
  s: int, optional, default: 1,
      radius of the simplex
  Returns
  -------
  w: (n,) numpy array,
      Euclidean projection of v on the simplex
  Notes
  -----
  The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
  Better alternatives exist for high-dimensional sparse vectors (cf. [1])
  However, this implementation still easily scales to millions of dimensions.
  References
  ----------
  [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
      John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
      International Conference on Machine Learning (ICML 2008)
      http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
  """
  assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
  n, = v.shape  # will raise ValueError if v is not 1-D
  # check if we are already on the simplex
  if v.sum() == s and np.alltrue(v >= 0):
      # best projection: itself!
      return v
  # get the array of cumulative sums of a sorted (decreasing) copy of v
  u = np.sort(v)[::-1]
  cssv = np.cumsum(u)
  # get the number of > 0 components of the optimal solution
  rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
  # compute the Lagrange multiplier associated to the simplex constraint
  theta = float(cssv[rho] - s) / rho
  # compute the projection by thresholding v using theta
  w = (v - theta).clip(min=0)
  return w

if __name__ == '__main__':
  tf.test.main()
