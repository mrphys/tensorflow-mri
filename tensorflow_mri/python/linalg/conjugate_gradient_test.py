# Copyright 2021 The TensorFlow MRI Authors. All Rights Reserved.
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

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for module `conjugate_gradient`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import conjugate_gradient
from tensorflow_mri.python.util import test_util


@test_util.run_all_in_graph_and_eager_modes
class ConjugateGradientTest(test_util.TestCase):
  """Tests for op `conjugate_gradient`."""
  @parameterized.product(dtype=[np.float32, np.float64],
                         shape=[[1, 1], [4, 4], [10, 10]],
                         use_static_shape=[True, False])
  def test_conjugate_gradient(self, dtype, shape, use_static_shape):  # pylint: disable=missing-param-doc
    """Test CG method."""
    np.random.seed(1)
    a_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype)
    # Make a self-adjoint, positive definite.
    a_np = np.dot(a_np.T, a_np)
    # jacobi preconditioner
    jacobi_np = np.zeros_like(a_np)
    jacobi_np[range(a_np.shape[0]), range(a_np.shape[1])] = (
        1.0 / a_np.diagonal())
    rhs_np = np.random.uniform(
        low=-1.0, high=1.0, size=shape[0]).astype(dtype)
    x_np = np.zeros_like(rhs_np)
    tol = 1e-6 if dtype == np.float64 else 1e-3
    max_iterations = 20

    if use_static_shape:
      a = tf.constant(a_np)
      rhs = tf.constant(rhs_np)
      x = tf.constant(x_np)
      jacobi = tf.constant(jacobi_np)
    else:
      a = tf.compat.v1.placeholder_with_default(a_np, shape=None)
      rhs = tf.compat.v1.placeholder_with_default(rhs_np, shape=None)
      x = tf.compat.v1.placeholder_with_default(x_np, shape=None)
      jacobi = tf.compat.v1.placeholder_with_default(jacobi_np, shape=None)

    operator = tf.linalg.LinearOperatorFullMatrix(
        a, is_positive_definite=True, is_self_adjoint=True)
    preconditioners = [
        None,
        # Preconditioner that does nothing beyond change shape.
        tf.linalg.LinearOperatorIdentity(
            a_np.shape[-1],
            dtype=a_np.dtype,
            is_positive_definite=True,
            is_self_adjoint=True),
        # Jacobi preconditioner.
        tf.linalg.LinearOperatorFullMatrix(
            jacobi,
            is_positive_definite=True,
            is_self_adjoint=True),
    ]
    cg_results = []
    for preconditioner in preconditioners:
      cg_graph = conjugate_gradient.conjugate_gradient(
          operator,
          rhs,
          preconditioner=preconditioner,
          x=x,
          tol=tol,
          max_iterations=max_iterations)
      cg_val = self.evaluate(cg_graph)
      norm_r0 = np.linalg.norm(rhs_np)
      norm_r = np.linalg.norm(cg_val.r)
      self.assertLessEqual(norm_r, tol * norm_r0)
      # Validate that we get an equally small residual norm with numpy
      # using the computed solution.
      r_np = rhs_np - np.dot(a_np, cg_val.x)
      norm_r_np = np.linalg.norm(r_np)
      self.assertLessEqual(norm_r_np, tol * norm_r0)
      cg_results.append(cg_val)

    # Validate that we get same results using identity_preconditioner
    # and None
    self.assertEqual(cg_results[0].i, cg_results[1].i)
    self.assertAlmostEqual(cg_results[0].gamma, cg_results[1].gamma)
    self.assertAllClose(cg_results[0].r, cg_results[1].r, rtol=tol)
    self.assertAllClose(cg_results[0].x, cg_results[1].x, rtol=tol)
    self.assertAllClose(cg_results[0].p, cg_results[1].p, rtol=tol)

  def test_bypass_gradient(self):
    """Tests the `bypass_gradient` argument."""
    dtype = np.float32
    shape = [4, 4]
    np.random.seed(1)
    a_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype)
    # Make a self-adjoint, positive definite.
    a_np = np.dot(a_np.T, a_np)

    rhs_np = np.random.uniform(
        low=-1.0, high=1.0, size=shape[0]).astype(dtype)

    tol = 1e-3
    max_iterations = 20

    a = tf.constant(a_np)
    rhs = tf.constant(rhs_np)
    operator = tf.linalg.LinearOperatorFullMatrix(
        a, is_positive_definite=True, is_self_adjoint=True)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(rhs)
      result = conjugate_gradient.conjugate_gradient(
          operator,
          rhs,
          tol=tol,
          max_iterations=max_iterations)
      result_bypass = conjugate_gradient.conjugate_gradient(
          operator,
          rhs,
          tol=tol,
          max_iterations=max_iterations,
          bypass_gradient=True)

    grad = tape.gradient(result.x, rhs)
    grad_bypass = tape.gradient(result_bypass.x, rhs)
    self.assertAllClose(result, result_bypass)
    self.assertAllClose(grad, grad_bypass, rtol=tol)


if __name__ == '__main__':
  tf.test.main()
