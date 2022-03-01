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

from tensorflow_mri.python.ops import convex_ops
from tensorflow_mri.python.ops import linalg_ops
from tensorflow_mri.python.util import linalg_ext


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


def admm_minimize(function_f,
                  function_g,
                  operator_a=None,
                  operator_b=None,
                  constant_c=None,
                  penalty_rho=1.0,
                  atol=1e-5,
                  rtol=1e-5,
                  max_iterations=50,
                  linearized=False):
  r"""Applies the ADMM algorithm to minimize a separable convex function.

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
    linearized: A `bool`. If `True`, use linearized variant of the ADMM
      algorithm. Linearized ADMM solves problems of the form
      :math:`f(x) + g(Ax)` and only requires evaluation of the proximal operator
      of `g(x)`. This is useful when the proximal operator of `g(Ax)` cannot be
      easily evaluated, but the proximal operator of `g(x)` can. Defaults to
      `False`.

  Returns:
    A namedtuple containing the following fields:
      - `i`: The number of iterations of the ADMM update.
      - `x`: The first primal variable.
      - `z`: The second primal variable.
      - `u`: The scaled dual variable.
      - `r`: The primal residual.
      - `s`: The dual residual.
      - `ptol`: The primal tolerance.
      - `dtol`: The dual tolerance.

  References:
    .. [1] Boyd, S., Parikh, N., & Chu, E. (2011). Distributed optimization and
      statistical learning via the alternating direction method of multipliers.
      Now Publishers Inc.

  Raises:
    TypeError: If inputs have incompatible types.
    ValueError: If inputs are incompatible.
  """
  if linearized:
    if operator_b is not None:
      raise ValueError(
          "Linearized ADMM does not support the use of `operator_b`.")
    if constant_c is not None:
      raise ValueError(
          "Linearized ADMM does not support the use of `constant_c`.")

  # Infer the dtype of the variables from the dtype of f.
  dtype = tf.dtypes.as_dtype(function_f.dtype)
  if function_g.dtype != dtype:
    raise TypeError(
        f"`function_f` and `function_g` must have the same dtype, but "
        f"got: {dtype} and {function_g.dtype}")

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
    operator_a = tf.linalg.LinearOperatorScaledIdentity(
        x_ndim, tf.constant(1.0, dtype=dtype))
  if operator_b is None:
    operator_b = tf.linalg.LinearOperatorScaledIdentity(
        z_ndim, tf.constant(-1.0, dtype=dtype))

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

  if linearized:
    x_update_fn = function_f.prox
    z_update_fn = function_g.prox
  else:
    x_update_fn = _get_admm_update_fn(function_f, operator_a)
    z_update_fn = _get_admm_update_fn(function_g, operator_b)

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
        tf.math.real(tf.norm(state.r, axis=-1)) <= state.ptol,
        tf.math.real(tf.norm(state.s, axis=-1)) <= state.dtol)

  def _cond(state):
    """Returns `True` if optimization should continue."""
    return (not _stopping_condition(state)) and (state.i < max_iterations)

  def _body(state):  # pylint: disable=missing-param-doc
    """A single ADMM step."""
    # x-minimization step.
    state_bz = tf.linalg.matvec(operator_b, state.z)
    if linearized:
      v = state.x - tf.linalg.matvec(
          operator_a,
          tf.linalg.matvec(operator_a, state.x) - state.z + state.u,
          adjoint_a=True)
    else:
      v = constant_c - state_bz - state.u
    x = x_update_fn(v, penalty_rho)

    # z-minimization step.
    ax = tf.linalg.matvec(operator_a, x)
    if linearized:
      v = ax + state.u
    else:
      v = constant_c - ax - state.u
    z = z_update_fn(v, penalty_rho)

    # Dual variable update and compute residuals.
    bz = tf.linalg.matvec(operator_b, z)
    r = ax + bz - constant_c
    u = state.u + r
    s = penalty_rho * tf.linalg.matvec(
        operator_a, bz - state_bz, adjoint_a=True)

    # Choose the primal tolerance.
    ax_norm = tf.math.real(tf.norm(ax, axis=-1))
    bz_norm = tf.math.real(tf.norm(bz, axis=-1))
    c_norm = tf.math.real(tf.norm(constant_c, axis=-1))
    max_norm = tf.math.maximum(tf.math.maximum(ax_norm, bz_norm), c_norm)
    ptol = (atol * u_ndim_sqrt + rtol * max_norm)

    # Choose the dual tolerance.
    aty_norm = tf.math.real(tf.norm(
        tf.linalg.matvec(operator_a, penalty_rho * state.u, adjoint_a=True),
        axis=-1))
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
      r=None,     # Will be set in the first call to `_body`.
      s=None,     # Ditto.
      ptol=None,  # Ditto.
      dtol=None)  # Ditto.
  state = _body(state)[0]

  return tf.while_loop(_cond, _body, [state])[0]


def _get_admm_update_fn(function, operator):
  r"""Returns a function for the ADMM update.

  The returned function evaluates the expression
  :math:`{\mathop{\mathrm{argmin}}_x} \left ( f(x) + \frac{\rho}{2} \left\| Ax - v \right\|_2^2 \right )`
  for a given input :math:`v` and penalty parameter :math:`\rho`.

  This function will raise an error if the above expression cannot be easily
  evaluated for the specified convex function and linear operator.

  Args:
    function: A `ConvexFunction` instance.
    operator: A `LinearOperator` instance.

  Returns:
    A function that evaluates the ADMM update.

  Raises:
    NotImplementedError: If no rules exist to evaluate the ADMM update for the
      specified inputs.
  """  # pylint: disable=line-too-long
  if isinstance(operator, tf.linalg.LinearOperatorIdentity):
    def _update_fn(x, rho):
      return function.prox(x, scale=1.0 / rho)
    return _update_fn

  if isinstance(operator, tf.linalg.LinearOperatorScaledIdentity):
    # This is equivalent to multiplication by a scalar, which can be taken out
    # of the norm and pooled with `rho`. If multiplier is negative, we also
    # change the sign of `v` in order to obtain the expression of the proximal
    # operator of f.
    multiplier = operator.multiplier
    def _update_fn(v, rho):  # pylint: disable=function-redefined
      return function.prox(
          tf.math.sign(multiplier) * v, scale=tf.math.abs(multiplier) / rho)
    return _update_fn

  if isinstance(function, convex_ops.ConvexFunctionQuadratic):
    # See ref. [1], section 4.2.
    def _update_fn(v, rho):  # pylint: disable=function-redefined
      # Create operator Q + rho * A^T A, where Q is the quadratic coefficient
      # of the quadratic convex function.
      scaled_identity = tf.linalg.LinearOperatorScaledIdentity(
          operator.shape[-1], tf.cast(rho, operator.dtype))
      ls_operator = tf.linalg.LinearOperatorComposition(
          [scaled_identity, operator.H, operator])
      ls_operator = linalg_ext.LinearOperatorAddition(
          [function.quadratic_coefficient, ls_operator],
          is_self_adjoint=True, is_positive_definite=True)
      # Compute the right-hand side of the linear system.
      rhs = (rho * tf.linalg.matvec(operator, v, adjoint_a=True) -
             function.linear_coefficient)
      # Solve the linear system using CG (see ref [1], section 4.3.4).
      return linalg_ops.conjugate_gradient(ls_operator, rhs).x
    return _update_fn

  raise NotImplementedError(
      f"No rules to evaluate the ADMM update for function "
      f"{function.name} and operator {operator.name}.")


def lbfgs_minimize(*args, **kwargs):
  """Applies the L-BFGS algorithm to minimize a differentiable function.

  For the parameters, see `tfp.optimizer.lbfgs_minimize`.
  """
  return tfp.optimizer.lbfgs_minimize(*args, **kwargs)
