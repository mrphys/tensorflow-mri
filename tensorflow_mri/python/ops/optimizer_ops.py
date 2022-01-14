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
        'i',      # The number of iterations of the ADMM update.
        'x',      # The first primal variable.
        'z',      # The second primal variable.
        'u',      # The scaled dual variable.
        'r',      # The primal residual.
        's',      # The dual residual.
        'ptol',   # The primal tolerance.
        'dtol'    # The dual tolerance.
    ]
)


def admm_minimize(function_f, function_g,
                  operator_a=None, operator_b=None, constant_c=None,
                  x_shape=None, penalty_rho=1.0, atol=1e-5,
                  rtol=1e-5, max_iterations=50):
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
    penalty_rho: A scalar `Tensor`. The penalty parameter of the augmented
      Lagrangian. Also corresponds to the step size of the dual variable update
      in the scaled form of ADMM.
    atol: A scalar `Tensor`. The absolute tolerance. Defaults to 1e-5.
    rtol: A scalar `Tensor`. The relative tolerance. Defaults to 1e-5.
    max_iterations: A scalar `Tensor`. The maximum number of iterations for ADMM
      updates.

  Returns:
    An namedtuple containing the following fields:
      `i`: The number of iterations of the ADMM update.
      `x`: The first primal variable.
      `z`: The second primal variable.
      `u`: The scaled dual variable.
      `r`: The primal residual.
      `s`: The dual residual.
      `ptol`: The primal tolerance.
      `dtol`: The dual tolerance.

  References:
    Boyd, S., Parikh, N., & Chu, E. (2011). Distributed optimization and
    statistical learning via the alternating direction method of multipliers.
    Now Publishers Inc.
  """
  # Infer the dtype of the variables from the dtype of f.
  dtype = tf.dtypes.as_dtype(function_f.dtype)
  if function_g.dtype != dtype:
    raise ValueError(
        f"`function_g` must have the same dtype as `function_f`, but "
        f"got: {function_g.dtype} and {dtype}")

  # Infer the dimensionality of the primal variables x, z from the
  # dimensionality of the domains and f and g.
  x_ndim = function_f.ndim
  z_ndim = function_g.ndim

  if x_ndim is None:
    raise ValueError("`function_f` must have a known domain dimension.")
  if z_ndim is None:
    raise ValueError("`function_g` must have a known domain dimension.")

  # Provide default values for A and B.
  if operator_a is None:
    operator_a = tf.linalg.LinearOperatorScaledIdentity(x_ndim, 1.0)
  if operator_b is None:
    operator_b = tf.linalg.LinearOperatorScaledIdentity(z_ndim, -1.0)

  # Check that the domain shapes of the A, B operators are consistent with f and
  # g.
  if not operator_a.shape[-1:].is_compatible_with([x_ndim]):
    raise ValueError(
        f"`operator_a` must have the same domain dimension as `function_f`, "
        f"but got: {operator_a.shape[-1]} and {x_ndim}")
  if not operator_b.shape[-1:].is_compatible_with([z_ndim]):
    raise ValueError(
        f"`operator_b` must have the same domain dimension as `function_g`, "
        f"but got: {operator_b.shape[-1]} and {z_ndim}")

  # Infer the dimensionality of the dual variable u from the range shape of
  # operator A.
  u_ndim = operator_a.shape[-2]

  # Check that the range dimension of operator B is compatible with that of
  # operator A.
  if not operator_b.shape[-2:-1].is_compatible_with([u_ndim]):
    raise ValueError(
        f"`operator_b` must have the same range dimension as `operator_a`, "
        f"but got: {operator_b.shape[-2]} and {u_ndim}")

  # Provide default value for constant c.
  if constant_c is None:
    constant_c = tf.constant(0.0, dtype=dtype, shape=[u_ndim])

  # Check that the constant c has the same dimensionality as the dual variable.
  if not constant_c.shape[-1:].is_compatible_with([u_ndim]):
    raise ValueError(
        f"The last dimension of `constant_c` must be equal to the range "
        f"dimension of `operator_a`, but got: {constant_c.shape[-1]} and "
        f"{u_ndim}")

  x_shape = tf.TensorShape([x_ndim])
  z_shape = tf.TensorShape([z_ndim])
  u_shape = tf.TensorShape([u_ndim])

  x_ndim_sqrt = tf.math.sqrt(tf.cast(x_ndim, dtype.real_dtype))
  u_ndim_sqrt = tf.math.sqrt(tf.cast(u_ndim, dtype.real_dtype))

  atol = tf.convert_to_tensor(
      atol, dtype=dtype.real_dtype, name='atol')
  rtol = tf.convert_to_tensor(
      rtol, dtype=dtype.real_dtype, name='rtol')
  max_iterations = tf.convert_to_tensor(
      max_iterations, dtype=tf.dtypes.int32, name='max_iterations')

  def _stopping_condition(state):
    return tf.math.logical_and(
        tf.norm(state.r, axis=-1) <= state.ptol,
        tf.norm(state.s, axis=-1) <= state.dtol)

  def _cond(state):
    """Returns `True` if optimization should continue."""
    return (not _stopping_condition(state)) and (state.i < max_iterations)

  def _body(state):
    """A single ADMM step."""
    # x-minimization step.
    state_bz = tf.linalg.matvec(operator_b, state.z)
    x = function_f.prox(
        tf.linalg.matvec(
            operator_a, constant_c - state.u - state_bz,
            adjoint_a=True))

    # z-minimization step.
    ax = tf.linalg.matvec(operator_a, x)
    z = function_g.prox(
        tf.linalg.matvec(
            operator_b, constant_c - state.u - ax,
            adjoint_a=False))

    # Dual variable update.
    bz = tf.linalg.matvec(operator_b, z)
    r = ax + bz - constant_c
    u = state.u + r
    s = penalty_rho * tf.linalg.matvec(operator_a, bz - state_bz, adjoint_a=True)

    # Choose the primal tolerance.
    ax_norm = tf.math.real(tf.norm(ax, axis=-1))
    bz_norm = tf.math.real(tf.norm(bz, axis=-1))
    c_norm = tf.math.real(tf.norm(constant_c, axis=-1))
    max_norm = tf.math.maximum(tf.math.maximum(ax_norm, bz_norm), c_norm)
    ptol = (atol * u_ndim_sqrt + rtol * max_norm)

    # Choose the dual tolerance. 
    aty_norm = tf.norm(
        tf.linalg.matvec(operator_a, penalty_rho * state.u, adjoint_a=True), axis=-1)
    dtol = (atol * x_ndim_sqrt + rtol * aty_norm)

    return [AdmmOptimizerResults(i=state.i + 1,
                                 x=x,
                                 z=z,
                                 u=u,
                                 r=r,
                                 s=s,
                                 ptol=ptol,
                                 dtol=dtol)]
  # Initial state.
  state = AdmmOptimizerResults(
      i=tf.constant(0, dtype=tf.dtypes.int32),
      x=tf.constant(0.0, dtype=dtype, shape=x_shape),
      z=tf.constant(0.0, dtype=dtype, shape=z_shape),
      u=tf.constant(0.0, dtype=dtype, shape=u_shape),
      r=None,     # Will be set in the next line by calling `_body`.
      s=None,     # Ditto.
      ptol=None,  # Ditto.
      dtol=None)  # Ditto.
  state = _body(state)[0]

  return tf.while_loop(_cond, _body, [state])[0]



lbfgs_minimize = tfp.optimizer.lbfgs_minimize
