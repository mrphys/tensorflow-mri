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
        'i',  # The number of iterations of the ADMM update.
        'x',  # The first primal variable.
        'z',  # The second primal variable.
        'u',  # The scaled dual variable.
        'r',  # The primal residual.
        's',  # The dual residual.
        'primal_tolerance',   # The primal tolerance.
        'dual_tolerance'      # The dual tolerance.
    ]
)


def admm_minimize(function_f, function_g,
                  operator_a=None, operator_b=None, constant_c=None,
                  rho=1.0, abs_tolerance=1e-5, rel_tolerance=1e-5,
                  max_iterations=50):
  """Applies the ADMM algorithm to minimize a separable convex function.

  Minimizes :math:`f(x) + g(z)`, subject to :math:`Ax + Bz = c`.

  If :math:`A`, :math:`B` and :math:`c` are not provided, the constraint
  defaults to :math:`x - z = 0`, in which case the problem is equivalent to
  minimizing :math:`f(x) + g(x)`.

  Args:
    function_f: A `ConvexFunction`.
    function_g: A `ConvexFunction`.
    operator_a: A `LinearOperator`. Defaults to the identity operator.
    operator_b: A `LinearOperator`. Defaults to the negative identity operator.
    constant_c: A scalar `Tensor`. Defaults to 0.0.
    rho: A scalar `Tensor`. The augmented Lagrangian parameter. The step size
      of the dual variable update in the scaled form of ADMM.
    abs_tolerance: A scalar `Tensor`. The absolute tolerance. Defaults to 1e-5.
    rel_tolerance: A scalar `Tensor`. The relative tolerance. Defaults to 1e-5.
    max_iterations: A scalar `Tensor`. The maximum number of iterations for ADMM
      updates.
  """
  dtype = tf.dtypes.float32
  shape = (1,)
  ndim = 1

  abs_tolerance = tf.convert_to_tensor(
      abs_tolerance, dtype=dtype.real_dtype, name='abs_tolerance')
  rel_tolerance = tf.convert_to_tensor(
      rel_tolerance, dtype=dtype.real_dtype, name='rel_tolerance')
  max_iterations = tf.convert_to_tensor(
      max_iterations, dtype=tf.dtypes.int32, name='max_iterations')

  if operator_a is None:
    operator_a = tf.linalg.LinearOperatorScaledIdentity(ndim, 1.0)

  def _stopping_condition(state):
    return tf.math.logical_and(
        tf.norm(state.r, axis=-1) <= state.primal_tolerance,
        tf.norm(state.s, axis=-1) <= state.dual_tolerance)

  def _cond(state):
    """Returns `True` if optimization should continue."""
    print("cond")
    if state.x is None:
      return True
    return (not _stopping_condition(state)) and (state.i < max_iterations)

  def _body(state):
    """A single ADMM step."""
    # x-minimization step.
    print("body")
    print(state)
    state_bz = tf.linalg.matvec(operator_b, state.z)
    x = function_f.prox(
        tf.linalg.matvec(
            operator_a, state_bz - state.u + constant_c,  # TODO: check
            adjoint_a=True))

    # z-minimization step.
    ax = tf.linalg.matvec(operator_a, x)
    z = function_g.prox(
        tf.linalg.matvec(
            operator_b, ax + state.u - constant_c,  # TODO: check
            adjoint_a=False))

    # Dual variable update.
    bz = tf.linalg.matvec(operator_b, z)
    r = ax + bz - constant_c  # TODO: check
    u = state.u + r
    s = rho * tf.norm(tf.linalg.matvec(operator_a, bz - state_bz, adjoint_a=True), axis=-1)

    # Shape of primal and dual variables.
    n = state.x.shape[-1]
    p = state.u.shape[-1]

    # Choose the primal tolerance.
    ax_norm = tf.norm(ax, axis=-1)
    bz_norm = tf.norm(bz, axis=-1)
    c_norm = tf.norm(constant_c, axis=-1)
    max_norm = tf.math.maximum(tf.math.maximum(ax_norm, bz_norm), c_norm)
    primal_tolerance = (abs_tolerance * tf.math.sqrt(p) +
                        rel_tolerance * max_norm)

    # Choose the dual tolerance. 
    aty_norm = tf.math.norm(
        tf.linalg.matvec(operator_a, rho * state.u, adjoint_a=True), axis=-1)
    dual_tolerance = (abs_tolerance * tf.math.sqrt(n) +
                      rel_tolerance * aty_norm)

    return AdmmOptimizerResults(i=state.i + 1,
                                x=x,
                                z=z,
                                u=u,
                                r=r,
                                s=s,
                                primal_tolerance=primal_tolerance,
                                dual_tolerance=dual_tolerance)
  # Initial state.
  state = AdmmOptimizerResults(
      i=tf.constant(0, dtype=tf.dtypes.int32),
      x=None,
      z=tf.constant(0.0, dtype=dtype, shape=shape),
      u=tf.constant(0.0, dtype=dtype, shape=shape),
      r=None,
      s=None,
      primal_tolerance=None,
      dual_tolerance=None)

  return tf.while_loop(_cond, _body, [state])



lbfgs_minimize = tfp.optimizer.lbfgs_minimize
