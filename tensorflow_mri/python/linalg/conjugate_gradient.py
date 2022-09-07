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
"""Conjugate gradient solver."""

import collections

import tensorflow as tf

from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.linalg import linear_operator


@api_util.export("linalg.conjugate_gradient")
def conjugate_gradient(operator,
                       rhs,
                       preconditioner=None,
                       x=None,
                       tol=1e-5,
                       max_iterations=20,
                       bypass_gradient=False,
                       name=None):
  r"""Conjugate gradient solver.

  Solves a linear system of equations $Ax = b$ for self-adjoint, positive
  definite matrix $A$ and right-hand side vector $b$, using an
  iterative, matrix-free algorithm where the action of the matrix $A$ is
  represented by `operator`. The iteration terminates when either the number of
  iterations exceeds `max_iterations` or when the residual norm has been reduced
  to `tol` times its initial value, i.e.
  $(\left\| b - A x_k \right\| <= \mathrm{tol} \left\| b \right\|\\)$.

  ```{note}
  This function is similar to
  `tf.linalg.experimental.conjugate_gradient`, except it adds support for
  complex-valued linear systems and for imaging operators.
  ```

  Args:
    operator: A `LinearOperator` that is self-adjoint and positive definite.
    rhs: A `tf.Tensor` of shape `[..., N]`. The right hand-side of the linear
      system.
    preconditioner: A `LinearOperator` that approximates the inverse of `A`.
      An efficient preconditioner could dramatically improve the rate of
      convergence. If `preconditioner` represents matrix `M`(`M` approximates
      `A^{-1}`), the algorithm uses `preconditioner.apply(x)` to estimate
      `A^{-1}x`. For this to be useful, the cost of applying `M` should be
      much lower than computing `A^{-1}` directly.
    x: A `tf.Tensor` of shape `[..., N]`. The initial guess for the solution.
    tol: A float scalar convergence tolerance.
    max_iterations: An `int` giving the maximum number of iterations.
    bypass_gradient: A `boolean`. If `True`, the gradient with respect to `rhs`
      will be computed by applying the inverse of `operator` to the upstream
      gradient with respect to `x` (through CG iteration), instead of relying
      on TensorFlow's automatic differentiation. This may reduce memory usage
      when training neural networks, but `operator` must not have any trainable
      parameters. If `False`, gradients are computed normally. For more details,
      see ref. [1].
    name: A name scope for the operation.

  Returns:
    A `namedtuple` representing the final state with fields

    - i: A scalar `int32` `tf.Tensor`. Number of iterations executed.
    - x: A rank-1 `tf.Tensor` of shape `[..., N]` containing the computed
        solution.
    - r: A rank-1 `tf.Tensor` of shape `[.., M]` containing the residual vector.
    - p: A rank-1 `tf.Tensor` of shape `[..., N]`. `A`-conjugate basis vector.
    - gamma: \\(r \dot M \dot r\\), equivalent to  \\(||r||_2^2\\) when
      `preconditioner=None`.

  Raises:
    ValueError: If `operator` is not self-adjoint and positive definite.

  References:
    1. Aggarwal, H. K., Mani, M. P., & Jacob, M. (2018). MoDL: Model-based
      deep learning architecture for inverse problems. IEEE transactions on
      medical imaging, 38(2), 394-405.
  """
  if bypass_gradient:
    if preconditioner is not None:
      raise ValueError(
          "preconditioner is not supported when bypass_gradient is True.")
    if x is not None:
      raise ValueError("x is not supported when bypass_gradient is True.")

    def _conjugate_gradient_simple(rhs):
      return _conjugate_gradient_internal(operator, rhs,
                                          tol=tol,
                                          max_iterations=max_iterations,
                                          name=name)

    @tf.custom_gradient
    def _conjugate_gradient_internal_grad(rhs):
      result = _conjugate_gradient_simple(rhs)

      def grad(*upstream_grads):
        # upstream_grads has the upstream gradient for each element of the
        # output tuple (i, x, r, p, gamma).
        _, dx, _, _, _ = upstream_grads
        return _conjugate_gradient_simple(dx).x

      return result, grad

    return _conjugate_gradient_internal_grad(rhs)

  return _conjugate_gradient_internal(operator, rhs,
                                      preconditioner=preconditioner,
                                      x=x,
                                      tol=tol,
                                      max_iterations=max_iterations,
                                      name=name)


def _conjugate_gradient_internal(operator,
                                 rhs,
                                 preconditioner=None,
                                 x=None,
                                 tol=1e-5,
                                 max_iterations=20,
                                 name=None):
  """Implementation of `conjugate_gradient`.

  For the parameters, see `conjugate_gradient`.
  """
  if isinstance(operator, linear_operator.LinearOperatorMixin):
    rhs = operator.flatten_domain_shape(rhs)

  if not (operator.is_self_adjoint and operator.is_positive_definite):
    raise ValueError('Expected a self-adjoint, positive definite operator.')

  cg_state = collections.namedtuple('CGState', ['i', 'x', 'r', 'p', 'gamma'])

  def stopping_criterion(i, state):
    return tf.math.logical_and(
        i < max_iterations,
        tf.math.reduce_any(
            tf.math.real(tf.norm(state.r, axis=-1)) > tf.math.real(tol)))

  def dot(x, y):
    return tf.squeeze(
        tf.linalg.matvec(
            x[..., tf.newaxis],
            y, adjoint_a=True), axis=-1)

  def cg_step(i, state):  # pylint: disable=missing-docstring
    z = tf.linalg.matvec(operator, state.p)
    alpha = state.gamma / dot(state.p, z)
    x = state.x + alpha[..., tf.newaxis] * state.p
    r = state.r - alpha[..., tf.newaxis] * z
    if preconditioner is None:
      q = r
    else:
      q = preconditioner.matvec(r)
    gamma = dot(r, q)
    beta = gamma / state.gamma
    p = q + beta[..., tf.newaxis] * state.p
    return i + 1, cg_state(i + 1, x, r, p, gamma)

  # We now broadcast initial shapes so that we have fixed shapes per iteration.

  with tf.name_scope(name or 'conjugate_gradient'):
    broadcast_shape = tf.broadcast_dynamic_shape(
        tf.shape(rhs)[:-1],
        operator.batch_shape_tensor())
    static_broadcast_shape = tf.broadcast_static_shape(
        rhs.shape[:-1],
        operator.batch_shape)
    if preconditioner is not None:
      broadcast_shape = tf.broadcast_dynamic_shape(
          broadcast_shape,
          preconditioner.batch_shape_tensor())
      static_broadcast_shape = tf.broadcast_static_shape(
          static_broadcast_shape,
          preconditioner.batch_shape)
    broadcast_rhs_shape = tf.concat([broadcast_shape, [tf.shape(rhs)[-1]]], -1)
    static_broadcast_rhs_shape = static_broadcast_shape.concatenate(
        [rhs.shape[-1]])
    r0 = tf.broadcast_to(rhs, broadcast_rhs_shape)
    tol *= tf.norm(r0, axis=-1)

    if x is None:
      x = tf.zeros(
          broadcast_rhs_shape, dtype=rhs.dtype.base_dtype)
      x = tf.ensure_shape(x, static_broadcast_rhs_shape)
    else:
      r0 = rhs - tf.linalg.matvec(operator, x)
    if preconditioner is None:
      p0 = r0
    else:
      p0 = tf.linalg.matvec(preconditioner, r0)
    gamma0 = dot(r0, p0)
    i = tf.constant(0, dtype=tf.int32)
    state = cg_state(i=i, x=x, r=r0, p=p0, gamma=gamma0)
    _, state = tf.while_loop(
        stopping_criterion, cg_step, [i, state])

    if isinstance(operator, linear_operator.LinearOperatorMixin):
      x = operator.expand_range_dimension(state.x)
    else:
      x = state.x

    return cg_state(
        state.i,
        x=x,
        r=state.r,
        p=state.p,
        gamma=state.gamma)
