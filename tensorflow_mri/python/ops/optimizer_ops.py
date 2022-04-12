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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_mri.python.ops import convex_ops
from tensorflow_mri.python.ops import linalg_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import linalg_ext
from tensorflow_mri.python.util import prefer_static


GdOptimizerResults = collections.namedtuple(
    'GdOptimizerResults', [
        'converged',
        'num_iterations',
        'objective_gradient',
        'objective_value',
        'position'
    ]
)


AdmmOptimizerResults = collections.namedtuple(
    'AdmmOptimizerResults', [
        'converged',
        'dual_residual',
        'dual_tolerance',
        'f_primal_variable',
        'g_primal_variable',
        'num_iterations',
        'primal_residual',
        'primal_tolerance',
        'scaled_dual_variable'
    ]
)


@api_util.export("optimize.gradient_descent")
def gradient_descent(value_and_gradients_function,
                     initial_position,
                     step_size,
                     max_iterations=50,
                     grad_tolerance=1e-8,
                     x_tolerance=0,
                     f_relative_tolerance=0,
                     f_absolute_tolerance=0,
                     name=None):
  r"""Applies gradient descent to minimize a differentiable function.

  Args:
    value_and_gradients_function: A `callable` that accepts a point as a
      real/complex `tf.Tensor` and returns a tuple of `tf.Tensor` objects
      containing the value of the function (real dtype) and its
      gradient (real/complex dtype) at that point. The function to be minimized.
      The input should be of shape `[..., n]`, where `n` is the size of the
      domain of input points, and all others are batching dimensions. The first
      component of the return value should be a real `tf.Tensor` of matching
      shape `[...]`. The second component (the gradient) should also be of
      shape `[..., n]` like the input value to the function. Given a function
      definition that returns the value of the function to be minimized, the
      value and gradients function may be obtained using the
      `tfmri.math.make_val_and_grad_fn` decorator.
    initial_position: A `tf.Tensor` of shape `[..., n]`. The starting point, or
      points when using batch dimensions, of the search procedure.
    step_size: A scalar real `tf.Tensor`. The step size to use in the gradient
      descent update.
    max_iterations: A scalar integer `tf.Tensor`. The maximum number of gradient
      descent iterations.
    grad_tolerance: A scalar `tf.Tensor` of real dtype. Specifies the gradient
      tolerance for the procedure. If the supremum norm of the gradient vector
      is below this number, the algorithm is stopped.
    x_tolerance: A scalar `tf.Tensor` of real dtype. If the absolute change in
      the position between one iteration and the next is smaller than this
      number, the algorithm is stopped.
    f_relative_tolerance: A scalar `tf.Tensor` of real dtype. If the relative
      change in the objective value between one iteration and the next is
      smaller than this value, the algorithm is stopped.
    f_absolute_tolerance: A scalar `tf.Tensor` of real dtype. If the absolute
      change in the objective value between one iteration and the next is
      smaller than this value, the algorithm is stopped.
    name: A `str`. The name of this operation.

  Returns:
    A `namedtuple` containing the following fields

    - `converged`: A boolean `tf.Tensor` of shape `[...]` indicating whether the
      minimum was found within tolerance for each batch member.
    - `num_iterations`: A scalar integer `tf.Tensor` containing the number of
      iterations of the GD update.
    - `objective_value`: A `tf.Tensor` of shape `[...]` with the value of the
      objective function at the `position`. If the search converged, then
      this is the (local) minimum of the objective function.
    - `objective_gradient`: A `tf.Tensor` of shape `[..., n]` containing the
      gradient of the objective function at the `position`. If the search
      converged the max-norm of this tensor should be below the tolerance.
    - `position`: A `tf.Tensor` of shape `[..., n]` containing the last argument
      value found during the search. If the search converged, then this value
      is the argmin of the objective function.
  """
  with tf.name_scope(name or 'gradient_descent'):
    initial_position = tf.convert_to_tensor(
        initial_position, name='initial_position')
    dtype = initial_position.dtype
    step_size = tf.convert_to_tensor(
        step_size, dtype=dtype.real_dtype, name='step_size')
    max_iterations = tf.convert_to_tensor(
        max_iterations, dtype=tf.int32, name='max_iterations')
    grad_tolerance = tf.convert_to_tensor(
        grad_tolerance, dtype=dtype.real_dtype, name='grad_tolerance')
    x_tolerance = tf.convert_to_tensor(
        x_tolerance, dtype=dtype.real_dtype, name='x_tolerance')
    f_relative_tolerance = tf.convert_to_tensor(
        f_relative_tolerance, dtype=dtype.real_dtype,
        name='f_relative_tolerance')
    f_absolute_tolerance = tf.convert_to_tensor(
        f_absolute_tolerance, dtype=dtype.real_dtype,
        name='f_absolute_tolerance')

    def _cond(state):
      return tf.math.logical_and(
          state.num_iterations < max_iterations,
          tf.math.reduce_any(tf.math.logical_not(state.converged)))

    def _body(state):
      position = state.position - step_size * state.objective_gradient
      objective_value, objective_gradient = value_and_gradients_function(
          position)

      converged = _check_convergence(state.position,
                                     position,
                                     state.objective_value,
                                     objective_value,
                                     objective_gradient,
                                     grad_tolerance,
                                     x_tolerance,
                                     f_relative_tolerance,
                                     f_absolute_tolerance)

      return [GdOptimizerResults(converged=converged,
                                 num_iterations=state.num_iterations + 1,
                                 objective_gradient=objective_gradient,
                                 objective_value=objective_value,
                                 position=position)]

    batch_shape = tf.shape(initial_position)[:-1]
    objective_value, objective_gradient = value_and_gradients_function(
        initial_position)
    state = GdOptimizerResults(converged=tf.fill(batch_shape, False),
                               num_iterations=tf.constant(0, dtype=tf.int32),
                               objective_gradient=objective_gradient,
                               objective_value=objective_value,
                               position=initial_position)

    return tf.while_loop(_cond, _body, [state])[0]


@api_util.export("convex.admm_minimize")
def admm_minimize(function_f,
                  function_g,
                  operator_a=None,
                  operator_b=None,
                  constant_c=None,
                  penalty=1.0,
                  absolute_tolerance=1e-5,
                  relative_tolerance=1e-5,
                  max_iterations=50,
                  linearized=False,
                  f_prox_kwargs=None,
                  g_prox_kwargs=None,
                  name=None):
  r"""Applies the ADMM algorithm to minimize a separable convex function.

  Minimizes :math:`f(x) + g(z)`, subject to :math:`Ax + Bz = c`.

  If :math:`A`, :math:`B` and :math:`c` are not provided, the constraint
  defaults to :math:`x - z = 0`, in which case the problem is equivalent to
  minimizing :math:`f(x) + g(x)`.

  Args:
    function_f: A `tfmri.convex.ConvexFunction` of shape `[..., n]` and real or
      complex dtype.
    function_g: A `tfmri.convex.ConvexFunction` of shape `[..., m]` and real or
      complex dtype.
    operator_a: A `tf.linalg.LinearOperator` of shape `[..., p, n]` and real or
      complex dtype. Defaults to the identity operator.
    operator_b: A `tf.linalg.LinearOperator` of shape `[..., p, m]` and real or
      complex dtype. Defaults to the negated identity operator.
    constant_c: A `tf.Tensor` of shape `[..., p]`. Defaults to 0.0.
    penalty: A scalar `tf.Tensor`. The penalty parameter of the augmented
      Lagrangian. Also corresponds to the step size of the dual variable update
      in the scaled form of ADMM.
    absolute_tolerance: A scalar `tf.Tensor` of real dtype. The absolute
      tolerance. Defaults to 1e-8.
    relative_tolerance: A scalar `tf.Tensor` of real dtype. The relative
      tolerance. Defaults to 1e-8.
    max_iterations: A scalar `tf.Tensor` of integer dtype. The maximum number
      of iterations of the ADMM update.
    linearized: A `boolean`. If `True`, use linearized variant of the ADMM
      algorithm. Linearized ADMM solves problems of the form
      :math:`f(x) + g(Ax)` and only requires evaluation of the proximal operator
      of `g(x)`. This is useful when the proximal operator of `g(Ax)` cannot be
      easily evaluated, but the proximal operator of `g(x)` can. Defaults to
      `False`.
    f_prox_kwargs: A `dict`. Keyword arguments to pass to the proximal operator
      of `function_f` during the x-minimization step.
    g_prox_kwargs: A `dict`. Keyword arguments to pass to the proximal operator
      of `function_g` during the z-minimization step.
    name: A `str`. The name of this operation. Defaults to `'admm_minimize'`.

  Returns:
    A `namedtuple` containing the following fields

    - `converged`: A boolean `tf.Tensor` of shape `[...]` indicating whether the
      minimum was found within tolerance for each batch member.
    - `dual_residual`: A real `tf.Tensor` of shape `[...]` containing the
      last tolerance used to evaluate the primal feasibility condition.
    - `dual_tolerance`: The dual tolerance.
    - `f_primal_variable`: A real or complex `tf.Tensor` of shape `[..., n]`
      containing the last argument value of `f` found during the search for
      each batch member. If the search converged, then this value is the argmin
      of the objective function, subject to the specified constraint.
    - `g_primal_variable`: A real or complex `tf.Tensor` of shape `[..., m]`
      containing the last argument value of `g` found during the search for
      each batch member. If the search converged, then this value is the argmin
      of the objective function, subject to the specified constraint.
    - `num_iterations`: A scalar integer `tf.Tensor` containing the number of
      iterations of the ADMM update.
    - `primal_residual`: A real or complex `tf.Tensor` of shape `[..., p]`
      containing the last primal residual for each batch member.
    - `primal_tolerance`: A real `tf.Tensor` of shape `[...]` containing the
      last tolerance used to evaluate the primal feasibility condition.
    - `scaled_dual_variable`: A `tf.Tensor` of shape `[..., p]` and real or
      complex dtype containing the last value of the scaled dual variable found
      during the search.

  References:
    .. [1] Boyd, S., Parikh, N., & Chu, E. (2011). Distributed optimization and
      statistical learning via the alternating direction method of multipliers.
      Now Publishers Inc.

  Raises:
    TypeError: If inputs have incompatible types.
    ValueError: If inputs are incompatible.
  """
  with tf.name_scope(name or 'admm_minimize'):
    if linearized:
      if operator_b is not None:
        raise ValueError(
            "Linearized ADMM does not support the use of `operator_b`.")
      if constant_c is not None:
        raise ValueError(
            "Linearized ADMM does not support the use of `constant_c`.")

    # Infer the dtype of the variables from the dtype of f.
    dtype = tf.dtypes.as_dtype(function_f.dtype)

    # Check that dtypes of both functions match.
    if function_g.dtype != dtype:
      raise TypeError(
          f"`function_f` and `function_g` must have the same dtype, but "
          f"got: {dtype} and {function_g.dtype}")

    # Check that batch shapes of both functions match.
    batch_shape = prefer_static.batch_shape(function_f)
    batch_shape = prefer_static.broadcast_shape(
        batch_shape, function_g.batch_shape_tensor())

    # Infer the dimensionality of the primal variables x, z from the
    # dimensionality of the domains of f and g.
    x_ndim_static = function_f.ndim
    z_ndim_static = function_g.ndim
    x_ndim = prefer_static.ndim(function_f)
    z_ndim = prefer_static.ndim(function_g)

    # Provide default values for A and B.
    if operator_a is None:
      operator_a = tf.linalg.LinearOperatorScaledIdentity(
          x_ndim, tf.constant(1.0, dtype=dtype))
    if operator_b is None:
      operator_b = tf.linalg.LinearOperatorScaledIdentity(
          z_ndim, tf.constant(-1.0, dtype=dtype))

    # Statically check that the domain shapes of the A, B operators are
    # consistent with f and g.
    if not operator_a.shape[-1:].is_compatible_with([x_ndim_static]):
      raise ValueError(
          f"`operator_a` must have the same domain dimension as `function_f`, "
          f"but got: {operator_a.shape[-1]} and {x_ndim_static}")
    if not operator_b.shape[-1:].is_compatible_with([z_ndim_static]):
      raise ValueError(
          f"`operator_b` must have the same domain dimension as `function_g`, "
          f"but got: {operator_b.shape[-1]} and {z_ndim_static}")

    # Check the batch shapes of the operators.
    batch_shape = prefer_static.broadcast_shape(
        batch_shape, operator_a.batch_shape_tensor())
    batch_shape = prefer_static.broadcast_shape(
        batch_shape, operator_b.batch_shape_tensor())

    # Infer the dimensionality of the dual variable u from the range shape of
    # operator A.
    u_ndim_static = operator_a.range_dimension
    if isinstance(u_ndim_static, tf.compat.v1.Dimension):
      u_ndim_static = u_ndim_static.value
    u_ndim = prefer_static.range_dimension(operator_a)

    # Check that the range dimension of operator B is compatible with that of
    # operator A.
    if not operator_b.shape[-2:-1].is_compatible_with([u_ndim_static]):
      raise ValueError(
          f"`operator_b` must have the same range dimension as `operator_a`, "
          f"but got: {operator_b.shape[-2]} and {u_ndim_static}")

    # Provide default value for constant c.
    if constant_c is None:
      constant_c = tf.constant(0.0, dtype=dtype, shape=[u_ndim])

    # Check that the constant c has the same dimensionality as the dual
    # variable.
    if not constant_c.shape[-1:].is_compatible_with([u_ndim_static]):
      raise ValueError(
          f"The last dimension of `constant_c` must be equal to the range "
          f"dimension of `operator_a`, but got: {constant_c.shape[-1]} and "
          f"{u_ndim_static}")

    if linearized:
      f_update_fn = function_f.prox
      g_update_fn = function_g.prox
    else:
      f_update_fn = _get_admm_update_fn(function_f, operator_a,
                                        prox_kwargs=f_prox_kwargs)
      g_update_fn = _get_admm_update_fn(function_g, operator_b,
                                        prox_kwargs=g_prox_kwargs)

    x_ndim_sqrt = tf.math.sqrt(tf.cast(x_ndim, dtype.real_dtype))
    u_ndim_sqrt = tf.math.sqrt(tf.cast(u_ndim, dtype.real_dtype))

    absolute_tolerance = tf.convert_to_tensor(
        absolute_tolerance, dtype=dtype.real_dtype, name='absolute_tolerance')
    relative_tolerance = tf.convert_to_tensor(
        relative_tolerance, dtype=dtype.real_dtype, name='relative_tolerance')
    max_iterations = tf.convert_to_tensor(
        max_iterations, dtype=tf.dtypes.int32, name='max_iterations')

    def _cond(state):
      """Returns `True` if optimization should continue."""
      return tf.math.logical_and(
          state.num_iterations < max_iterations,
          tf.math.reduce_any(tf.math.logical_not(state.converged)))

    def _body(state):  # pylint: disable=missing-param-doc
      """The ADMM update."""
      # x-minimization step.
      state_bz = tf.linalg.matvec(operator_b, state.g_primal_variable)
      if linearized:
        v = state.f_primal_variable - tf.linalg.matvec(
            operator_a,
            (tf.linalg.matvec(operator_a, state.f_primal_variable) -
             state.g_primal_variable + state.scaled_dual_variable),
            adjoint_a=True)
      else:
        v = constant_c - state_bz - state.scaled_dual_variable
      f_primal_variable = f_update_fn(v, penalty)

      # z-minimization step.
      ax = tf.linalg.matvec(operator_a, f_primal_variable)
      if linearized:
        v = ax + state.scaled_dual_variable
      else:
        v = constant_c - ax - state.scaled_dual_variable
      g_primal_variable = g_update_fn(v, penalty)

      # Dual variable update and compute residuals.
      bz = tf.linalg.matvec(operator_b, g_primal_variable)
      primal_residual = ax + bz - constant_c
      scaled_dual_variable = state.scaled_dual_variable + primal_residual
      dual_residual = penalty * tf.linalg.matvec(
          operator_a, bz - state_bz, adjoint_a=True)

      # Choose the primal tolerance.
      ax_norm = tf.math.real(tf.norm(ax, axis=-1))
      bz_norm = tf.math.real(tf.norm(bz, axis=-1))
      c_norm = tf.math.real(tf.norm(constant_c, axis=-1))
      max_norm = tf.math.maximum(tf.math.maximum(ax_norm, bz_norm), c_norm)
      primal_tolerance = (absolute_tolerance * u_ndim_sqrt +
                          relative_tolerance * max_norm)

      # Choose the dual tolerance.
      aty_norm = tf.math.real(tf.norm(
          tf.linalg.matvec(operator_a, penalty * state.scaled_dual_variable,
                           adjoint_a=True),
          axis=-1))
      dual_tolerance = (absolute_tolerance * x_ndim_sqrt +
                        relative_tolerance * aty_norm)

      # Check convergence.
      converged = tf.math.logical_and(
          tf.math.real(tf.norm(primal_residual, axis=-1)) <= primal_tolerance,
          tf.math.real(tf.norm(dual_residual, axis=-1)) <= dual_tolerance)

      return [AdmmOptimizerResults(converged=converged,
                                   dual_residual=dual_residual,
                                   dual_tolerance=dual_tolerance,
                                   f_primal_variable=f_primal_variable,
                                   g_primal_variable=g_primal_variable,
                                   num_iterations=state.num_iterations + 1,
                                   primal_residual=primal_residual,
                                   primal_tolerance=primal_tolerance,
                                   scaled_dual_variable=scaled_dual_variable)]
    # Initial state.
    x_shape = prefer_static.concat([batch_shape, [x_ndim]], axis=0)
    z_shape = prefer_static.concat([batch_shape, [z_ndim]], axis=0)
    u_shape = prefer_static.concat([batch_shape, [u_ndim]], axis=0)

    state = AdmmOptimizerResults(
        converged=tf.fill(batch_shape, False),
        f_primal_variable=tf.zeros(shape=x_shape, dtype=dtype),
        g_primal_variable=tf.zeros(shape=z_shape, dtype=dtype),
        dual_residual=None,
        dual_tolerance=None,
        num_iterations=tf.constant(0, dtype=tf.dtypes.int32),
        primal_residual=None,
        primal_tolerance=None,
        scaled_dual_variable=tf.zeros(shape=u_shape, dtype=dtype))
    state = _body(state)[0]

    return tf.while_loop(_cond, _body, [state])[0]


def _get_admm_update_fn(function, operator, prox_kwargs=None):
  r"""Returns a function for the ADMM update.

  The returned function evaluates the expression
  :math:`{\mathop{\mathrm{argmin}}_x} \left ( f(x) + \frac{\rho}{2} \left\| Ax - v \right\|_2^2 \right )`
  for a given input :math:`v` and penalty parameter :math:`\rho`.

  This function will raise an error if the above expression cannot be easily
  evaluated for the specified convex function and linear operator.

  Args:
    function: A `ConvexFunction` instance.
    operator: A `LinearOperator` instance.
    prox_kwargs: A `dict` of keyword arguments to pass to the proximal operator
      of `function`.

  Returns:
    A function that evaluates the ADMM update.

  Raises:
    NotImplementedError: If no rules exist to evaluate the ADMM update for the
      specified inputs.
  """  # pylint: disable=line-too-long
  prox_kwargs = prox_kwargs or {}

  if isinstance(operator, tf.linalg.LinearOperatorIdentity):
    def _update_fn(x, rho):
      return function.prox(x, scale=1.0 / rho, **prox_kwargs)
    return _update_fn

  if isinstance(operator, tf.linalg.LinearOperatorScaledIdentity):
    # This is equivalent to multiplication by a scalar, which can be taken out
    # of the norm and pooled with `rho`. If multiplier is negative, we also
    # change the sign of `v` in order to obtain the expression of the proximal
    # operator of f.
    multiplier = operator.multiplier
    def _update_fn(v, rho):  # pylint: disable=function-redefined
      return function.prox(
          tf.math.sign(multiplier) * v, scale=tf.math.abs(multiplier) / rho,
          **prox_kwargs)
    return _update_fn

  if isinstance(function, convex_ops.ConvexFunctionQuadratic):
    # TODO(jmontalt): add prox_kwargs here.
    # See ref. [1], section 4.2.
    def _update_fn(v, rho):  # pylint: disable=function-redefined
      solver_kwargs = prox_kwargs.get('solver_kwargs', {})
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
      return linalg_ops.conjugate_gradient(ls_operator, rhs, **solver_kwargs).x

    return _update_fn

  raise NotImplementedError(
      f"No rules to evaluate the ADMM update for function "
      f"{function.name} and operator {operator.name}.")


@api_util.export("optimize.lbfgs_minimize")
def lbfgs_minimize(*args, **kwargs):
  """Applies the L-BFGS algorithm to minimize a differentiable function.

  For the parameters, see `tfp.optimizer.lbfgs_minimize`_.

  .. _tfp.optimizer.lbfgs_minimize: https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize
  """
  return tfp.optimizer.lbfgs_minimize(*args, **kwargs)


def _check_convergence(current_position,  # pylint: disable=missing-param-doc
                       next_position,
                       current_objective,
                       next_objective,
                       next_gradient,
                       grad_tolerance,
                       x_tolerance,
                       f_relative_tolerance,
                       f_absolute_tolerance):
  """Checks if the minimization has converged."""
  grad_converged = tf.norm(next_gradient, ord=np.inf, axis=-1) <= grad_tolerance
  x_converged = (
      tf.norm(next_position - current_position, ord=np.inf, axis=-1) <=
      x_tolerance)
  f_relative_converged = (
      tf.math.abs(next_objective - current_objective) <=
      f_relative_tolerance * current_objective)
  f_absolute_converged = (
      tf.math.abs(next_objective - current_objective) <= f_absolute_tolerance)
  return (grad_converged |
          x_converged |
          f_relative_converged |
          f_absolute_converged)
