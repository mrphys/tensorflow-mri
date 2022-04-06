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
"""Tests for module `optimizer_ops`."""

import tensorflow as tf

from tensorflow_mri.python.ops import convex_ops
from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.ops import optimizer_ops
from tensorflow_mri.python.util import linalg_ext
from tensorflow_mri.python.util import test_util


class GradientDescentTest(test_util.TestCase):
  """Tests for `gradient_descent` op."""
  def test_quadratic(self):
    """Test GD can minimize a quadratic function."""
    @math_ops.make_val_and_grad_fn
    def fn(x):
      return tf.math.reduce_sum(tf.square(x - 2.0), axis=-1)

    state = optimizer_ops.gradient_descent(fn, [0.0], 0.05, max_iterations=100)

    self.assertAllEqual(False, state.converged)
    self.assertAllEqual(100, state.num_iterations)
    self.assertAllClose(0.0, state.objective_value, rtol=1e-3, atol=1e-3)
    self.assertAllClose([0.0], state.objective_gradient, rtol=1e-3, atol=1e-3)
    self.assertAllClose([2.0], state.position, rtol=1e-3, atol=1e-3)

    state = optimizer_ops.gradient_descent(fn, [0.0], 0.05,
                                           grad_tolerance=1e-3,
                                           max_iterations=100)

    self.assertAllEqual(True, state.converged)
    self.assertAllEqual(79, state.num_iterations)
    self.assertAllClose(0.0, state.objective_value, rtol=1e-3, atol=1e-3)
    self.assertAllClose([0.0], state.objective_gradient, rtol=1e-3, atol=1e-3)
    self.assertAllClose([2.0], state.position, rtol=1e-3, atol=1e-3)


class ADMMTest(test_util.TestCase):
  """Tests for `admm_minimize` op."""
  def test_lasso(self):
    """Test ADMM can minimize lasso problem."""
    operator = tf.linalg.LinearOperatorFullMatrix(
        [[-0.69651254, 0.05905978, 0.26406853, -1.44617154],
         [ 1.69614248, 1.79707178, 0.87167329, -0.70116535]])
    x = tf.convert_to_tensor([1.16495351,
                              0.62683908,
                              0.07508015,
                              0.35160690])
    rhs = operator.matvec(x)
    lambda_ = 0.5
    atol = 1e-4
    rtol = 1e-2
    max_iterations = 100

    function_f = convex_ops.ConvexFunctionLeastSquares(operator, rhs)
    function_g = convex_ops.ConvexFunctionL1Norm(scale=lambda_, ndim=4)

    result = optimizer_ops.admm_minimize(function_f, function_g,
                                         atol=atol, rtol=rtol,
                                         max_iterations=max_iterations)
    expected_i = 12
    expected_z = [1.57677657, 0., 0., 0.]

    self.assertAllClose(result.z, expected_z)
    self.assertEqual(result.i, expected_i)

  def test_total_variation(self):
    """Test ADMM can minimize total variation problem."""
    operator = tf.linalg.LinearOperatorIdentity(4)
    x = tf.convert_to_tensor([1.16495351,
                              0.62683908,
                              0.07508015,
                              0.35160690])
    rhs = operator.matvec(x)
    lambda_ = 0.1
    atol = 1e-4
    rtol = 1e-2
    max_iterations = 100

    function_f = convex_ops.ConvexFunctionLeastSquares(operator, rhs)
    function_g = convex_ops.ConvexFunctionL1Norm(scale=lambda_, ndim=3)

    operator_a = linalg_ext.LinearOperatorFiniteDifference(4)

    result = optimizer_ops.admm_minimize(function_f, function_g,
                                         operator_a=operator_a,
                                         atol=atol, rtol=rtol,
                                         max_iterations=max_iterations)

    expected_i = 12
    expected_x = [1.0638748, 0.628781, 0.2630071, 0.26281652]

    self.assertAllClose(result.x, expected_x)
    self.assertEqual(result.i, expected_i)

  def test_linearized(self):
    """Test linearized variation of ADMM."""
    operator = tf.linalg.LinearOperatorIdentity(4)
    x = tf.convert_to_tensor([1.16495351,
                              0.62683908,
                              0.07508015,
                              0.35160690])
    rhs = operator.matvec(x)
    lambda_ = 0.1
    atol = 1e-4
    rtol = 1e-2
    max_iterations = 100

    function_f = convex_ops.ConvexFunctionLeastSquares(operator, rhs)
    function_g = convex_ops.ConvexFunctionL1Norm(scale=lambda_, ndim=3)

    operator_a = linalg_ext.LinearOperatorFiniteDifference(4)

    result = optimizer_ops.admm_minimize(function_f, function_g,
                                         operator_a=operator_a,
                                         atol=atol, rtol=rtol,
                                         max_iterations=max_iterations,
                                         linearized=True)

    expected_i = 100
    expected_x = [1.064954, 0.626839, 0.27508 , 0.251607]

    self.assertAllClose(expected_x, result.x)
    self.assertEqual(expected_i, result.i)


if __name__ == '__main__':
  tf.test.main()
