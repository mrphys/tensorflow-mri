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
"""Function optimization operations.

This module contains solvers for nonlinear optimization problems.
"""

import collections

import tensorflow as tf
import tensorflow_probability as tfp


AdmmOptimizerResults = collections.namedtuple(
    'AdmmOptimizerResults', [
        'num_iterations'  # The number of iterations of the ADMM update.
    ]
)


def admm_minimize(function_f, function_g, tolerance=1e-8, max_iterations=50):
  """Applies the ADMM algorithm to minimize `f + g`.

  Minimizes :math:`f(x) + g(z)`, subject to :math:`Ax + Bz = c`.

  If :math:`A`, :math:`B` and :math:`c` are not provided, the constraint
  defaults to :math:`x - z = 0`, in which case the problem is equivalent to
  minimizing :math:`f(x) + g(x)`.

  Args:
    function_f: A `ConvexFunction`.
    function_g: A `ConvexFunction`.
    tolerance: A `float`. Specifies the tolerance of the optimization.
    max_iterations: An `int`. The maximum number of iterations for ADMM updates.
  """
  dtype = tf.dtypes.float32

  tolerance = tf.convert_to_tensor(
      tolerance, dtype=dtype.real_dtype, name='tolerance')
  max_iterations = tf.convert_to_tensor(
      max_iterations, dtype=tf.dtypes.int32, name='max_iterations')

  def _cond(state):
    """Returns `True` if optimization should continue."""
    return state.num_iterations < max_iterations

  def _body(state):
    """A single ADMM step."""
    pass

  # Initial state.
  state = AdmmOptimizerResults(
      num_iterations=tf.constant(0, dtype=tf.dtypes.int32))

  return tf.while_loop(_cond, _body, [state])



lbfgs_minimize = tfp.optimizer.lbfgs_minimize
