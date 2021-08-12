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
"""Linear algebra operations."""

import abc
import collections

import tensorflow as tf
import tensorflow_nufft as tfft

from tensorflow_mri.python.utils import check_utils
from tensorflow_mri.python.utils import tensor_utils


class LinearOperatorImaging(tf.linalg.LinearOperator):
  """Linear operator meant to operate on images.

  This `LinearOperator` defines some additional methods to simplify operations
  on images, while maintaining compatibility with the TensorFlow linear algebra
  framework.

  Mainly this operator does two things:

  * Acknowledge the fact that its inputs and outputs may have meaningful shapes,
    although they must of course be vectorized before matrix-matrix and
    matrix-vector multiplications. To this end it defines the additional
    properties `domain_shape` and `range_shape`. These enrich the information
    provided by built-in `shape`, `domain_dimension` and `range_dimension`,
    though note that they cannot perform any checks since inputs and outputs are
    always vectorized. The built-in `shape` is overriden as it can be computed
    from `domain_shape` and `range_shape`.
  * Acknowledge the fact that this operator will operate on vectorized images
    and so it will perform primarily matrix-vector multiplications. Therefore
    the requirement to implement matrix-matrix multiplication is removed.
    Nevertheless this operator must still be able to accept matrix arguments to
    be compatible with the TensorFlow linear algebra framework. Thus `matmul`
    is overriden to call `matvec` if the outer dimension of the argument is 1
    and raise an error otherwise.

  To summarize, subclasses of `LinearOperatorImaging` MUST:

  * Override `_domain_shape` and `_range_shape`.
  * Override `_matvec`.

  Optionally, subclasses may also:

  * Override `_batch_shape`, if the operator has a batch shape (default is a
    scalar batch shape).

  Generally, operators should NOT need to:

  * Override `_shape`, this is provided for free.
  * Override `_matmul`, this is provided for free.

  For the parameters, see `tf.linalg.LinearOperator`.
  """
  @property
  def domain_shape(self):
    """Domain shape of this linear operator."""
    return self._domain_shape()

  @property
  def range_shape(self):
    """Range shape of this linear operator."""
    return self._range_shape()

  @abc.abstractmethod
  def _domain_shape(self):
    # Users must override this method.
    return tf.TensorShape(None)

  @abc.abstractmethod
  def _range_shape(self):
    # Users must override this method.
    return tf.TensorShape(None)

  def _batch_shape(self):
    # Users should override this method if this operator has a batch shape.
    return tf.TensorShape([])

  def _shape(self):
    # Default implementation of `_shape` for imaging operators.
    return self._batch_shape() + tf.TensorShape(
      [self.range_shape.num_elements(),
       self.domain_shape.num_elements()])

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # Default implementation of `matmul` for imaging operator. If outer
    # dimension of argument is 1, call `matvec`. Otherwise raise an error.
    arg_outer_dim = -2 if adjoint_arg else -1

    if x.shape[arg_outer_dim] != 1:
      raise ValueError(
        f"`{self.__class__.__name__}` does not support matrix multiplication.")

    x = tf.squeeze(x, axis=arg_outer_dim)
    x = self.matvec(x, adjoint=adjoint)
    x = tf.expand_dims(x, axis=arg_outer_dim)
    return x

  @abc.abstractmethod
  def _matvec(self, x, adjoint=False):
    # Subclasses must override this method.
    pass


class LinearOperatorFFT(LinearOperatorImaging):
  """Linear operator acting like an FFT matrix."""
  def __init__(self, domain_shape):
    raise NotImplementedError(
      "`LinearOperatorFFT` is not implemented.")


class LinearOperatorNUFFT(LinearOperatorImaging):
  """Linear operator acting like an NUFFT matrix.

  Args:
    points: A `Tensor`. Must have type `float32` or `float64`. Must have shape
      `[..., M, N]`, where `N` is the rank (or spatial dimensionality), `M` is
      the number of samples and `...` is the batch shape, which can have any
      number of dimensions.
    domain_shape: A `TensorShape` or a list of `ints`. The domain shape of this
      operator. This is usually the shape of the image but may include
      additional dimensions.
    name: An optional `string`. The name of this operator.
  """
  def __init__(self,
               points,
               domain_shape,
               name="LinearOperatorNUFFT"):

    parameters = dict(
      points=points,
      domain_shape=domain_shape,
      name=name
    )

    self._dshape = tf.TensorShape(domain_shape)
    self._points = check_utils.validate_tensor_dtype(
      tf.convert_to_tensor(points), 'floating', 'points')
    self._rank = self._points.shape[-1]

    # Compute NUFFT batch shape. The NUFFT batch shape is different from this
    # operator's batch shape, and it is included in the operator's inner shape.
    nufft_batch_shape = self.domain_shape[:-self.rank]

    # Batch shape of `points` might have two parts: one that goes into NUFFT
    # batch shape and another that goes into this operator's batch shape.
    points_batch_shape = self.points.shape[:-2]
    # Take operator part of batch shape, then keep remainder.
    self._bshape = points_batch_shape[:-nufft_batch_shape.rank] # pylint: disable=invalid-unary-operand-type
    points_batch_shape = points_batch_shape[-nufft_batch_shape.rank:] # pylint: disable=invalid-unary-operand-type
    # Check that NUFFT part of points batch shape is broadcast compatible with
    # NUFFT batch shape.
    points_batch_shape = points_batch_shape.as_list()
    points_batch_shape = [None if s == 1 else s for s in points_batch_shape]
    points_batch_shape = [None] * (
      nufft_batch_shape.rank - len(points_batch_shape)) + points_batch_shape
    if not nufft_batch_shape.is_compatible_with(points_batch_shape):
      raise ValueError(
        f"The batch shape of `points` must be broadcastable to the batch part "
        f"of `domain_shape`. Received batch shapes "
        f"{str(self.domain_shape[:-self.rank])} and "
        f"{str(self.points.shape[:-2])} for input and `points`, respectively.")
    self._rshape = nufft_batch_shape + self.points.shape[-2:-1]

    is_square = self.domain_dimension == self.range_dimension

    super().__init__(tensor_utils.get_complex_dtype(self.points.dtype),
                     is_non_singular=None,
                     is_self_adjoint=None,
                     is_positive_definite=None,
                     is_square=is_square,
                     name=name,
                     parameters=parameters)

  def _matvec(self, x, adjoint=False):

    arg_batch_shape = x.shape[:-1]
    if adjoint:
      x = tf.reshape(x, arg_batch_shape + self.range_shape)
      x = tfft.nufft(x, self.points,
                     grid_shape=self.domain_shape[-self.rank:],
                     transform_type='type_1',
                     fft_direction='backward')
    else:
      x = tf.reshape(x, arg_batch_shape + self.domain_shape)
      x = tfft.nufft(x, self.points,
                     transform_type='type_2',
                     fft_direction='forward')
    x = tf.reshape(x, arg_batch_shape.as_list() + [-1])
    return x

  def _domain_shape(self):
    return self._dshape

  def _range_shape(self):
    return self._rshape

  def _batch_shape(self):
    return self._bshape

  @property
  def rank(self):
    """Rank (in the sense of spatial dimensionality) of this operator."""
    return self._rank

  @property
  def points(self):
    """Sampling coordinates."""
    return self._points


class LinearOperatorSensitivityModulation(LinearOperatorImaging):
  """Linear operator acting like a sensitivity modulation matrix.

  Args:
    sensitivities: A `Tensor`. The coil sensitivity maps. Must have type
      `complex64` or `complex128`. Must have shape `[..., C, *S]`, where `S`
      is the spatial shape, `C` is the number of coils and `...` is the batch
      shape, which can have any dimensionality. Note that `rank` must be
      specified if you intend to provide any batch dimensions.
    rank: An optional `int`. The rank (in the sense of spatial dimensionality)
      of this operator. Defaults to `sensitivities.shape.rank - 1`. Therefore,
      if `rank` is not specified, axis 0 is interpreted to be the coil axis
      and the remaining dimensions are interpreted to be spatial dimensions.
    name: An optional `string`. The name of this operator.
  """
  def __init__(self,
               sensitivities,
               rank=None,
               name='LinearOperatorSensitivityModulation'):

    parameters = dict(
      sensitivities=sensitivities,
      rank=rank,
      name=name
    )

    self._sensitivities = check_utils.validate_tensor_dtype(
      tf.convert_to_tensor(sensitivities), 'complex', name='sensitivities')

    self._rank = rank or self.sensitivities.shape.rank - 1
    self._image_shape = self.sensitivities.shape[-self.rank:]
    self._num_coils = self.sensitivities.shape[-self.rank-1]
    self._coil_axis = -self.rank-1

    self._dshape = self.image_shape
    self._rshape = tf.TensorShape([self.num_coils]) + self.image_shape
    self._bshape = self.sensitivities.shape[:-self.rank-1]

    super().__init__(self._sensitivities.dtype,
                     is_non_singular=False,
                     is_self_adjoint=False,
                     is_positive_definite=False,
                     is_square=False,
                     name=name,
                     parameters=parameters)

  def _domain_shape(self):
    return self._dshape

  def _range_shape(self):
    return self._rshape

  def _batch_shape(self):
    return self._bshape

  def _matvec(self, x, adjoint=False):

    arg_batch_shape = x.shape[:-1]
    if adjoint:
      x = tf.reshape(x, arg_batch_shape + self.range_shape)
      x *= tf.math.conj(self.sensitivities)
      x = tf.math.reduce_sum(x, axis=self._coil_axis)
    else:
      x = tf.reshape(x, arg_batch_shape + self.domain_shape)
      x = tf.expand_dims(x, -self.domain_shape.rank-1)
      x *= self.sensitivities
    x = tf.reshape(x, arg_batch_shape.as_list() + [-1])
    return x

  @property
  def rank(self):
    """Rank (in the sense of spatial dimensionality) of this operator."""
    return self._rank

  @property
  def image_shape(self):
    """Image shape."""
    return self._image_shape

  @property
  def num_coils(self):
    """Number of coils."""
    return self._num_coils

  @property
  def sensitivities(self):
    """Coil sensitivity maps."""
    return self._sensitivities


class LinearOperatorParallelMRI(tf.linalg.LinearOperatorComposition):
  """Linear operator acting like a parallel MRI matrix.

  Args:
    sensitivities: A `Tensor`. The coil sensitivity maps. Must have type
      `complex64` or `complex128`. Must have shape `[..., C, *S]`, where `S`
      is the spatial shape, `C` is the number of coils and `...` is the batch
      shape, which can have any dimensionality. Note that `rank` must be
      specified if you intend to provide any batch dimensions.
    trajectory: An optional `Tensor`. Must have type `float32` or `float64`.
      Must have shape `[..., M, N]`, where `N` is the rank (or spatial
      dimensionality), `M` is the number of samples and `...` is the batch
      shape, which can have any number of dimensions and must be
      broadcast-compatible with the batch shape of `sensitivities`. If `points`
      is provided, this operator is a non-Cartesian MRI operator. Otherwise,
      this is operator is a Cartesian MRI operator.
    rank: An optional `int`. The rank (in the sense of spatial dimensionality)
      of this operator. If `trajectory` is not `None`, the rank is inferred from
      `trajectory` and this argument is ignored. If `trajectory` is `None`,
      this value defaults to `sensitivities.shape.rank - 1`. `rank` must be
      specified if `sensitivities` has any batch dimensions.
    name: An optional `string`. The name of this operator.
  """
  def __init__(self,
               sensitivities,
               trajectory=None,
               rank=None,
               name='LinearOperatorParallelMRI'):

    sensitivities = tf.convert_to_tensor(sensitivities)

    # Prepare the Fourier operator.
    if trajectory is not None: # Non-Cartesian
      trajectory = check_utils.validate_tensor_dtype(
        tf.convert_to_tensor(trajectory),
        sensitivities.dtype.real_dtype, name='trajectory')
      trajectory = tf.expand_dims(trajectory, -3) # Add coil dimension.
      self._rank = trajectory.shape[-1]
      self._is_cartesian = False
      linop_fourier = LinearOperatorNUFFT(
        trajectory, sensitivities.shape[-self.rank-1:]) # pylint: disable=invalid-unary-operand-type
    else: # Cartesian
      self._rank = rank
      self._is_cartesian = True
      linop_fourier = LinearOperatorFFT(sensitivities.shape[-self.rank-1:]) # pylint: disable=invalid-unary-operand-type

    # Prepare the coil sensitivity operator.
    linop_sens = LinearOperatorSensitivityModulation(
      sensitivities, rank=self.rank)

    super().__init__([linop_fourier, linop_sens], name=name)

  @property
  def rank(self):
    """Rank (in the sense of spatial dimensionality) of this operator."""
    return self._rank

  @property
  def image_shape(self):
    """Image shape."""
    return self.linop_sens.image_shape

  @property
  def num_coils(self):
    """Number of coils."""
    return self.linop_sens.num_coils

  @property
  def is_cartesian(self):
    """Whether this is a Cartesian MRI operator."""
    return self._is_cartesian

  @property
  def is_non_cartesian(self):
    """Whether this is a non-Cartesian MRI operator."""
    return not self._is_cartesian

  @property
  def sensitivities(self):
    """Coil sensitivity maps."""
    return self.linop_sens.sensitivities

  @property
  def trajectory(self):
    """*k*-space trajectory."""
    return self.linop_fourier.points if self.is_non_cartesian else None

  @property
  def linop_fourier(self):
    """Fourier linear operator."""
    return self.operators[0]

  @property
  def linop_sens(self):
    """Sensitivity modulation linear operator."""
    return self.operators[1]


class LinearOperatorRealWeighting(LinearOperatorImaging):
  """Linear operator acting like a real weighting matrix.

  This is a square, self-adjoint operator.

  This operator acts like a diagonal matrix. It does not inherit from
  `LinearOperatorDiag` for efficiency reasons, as the diagonal values may be
  repeated periodically.

  Args:
    weights: A `Tensor`. Must have type `float32` or `float64`.
    arg_shape: A `TensorShape` or a list of `ints`. The domain/range shape.
    dtype: A `DType`. The data type for this operator. Defaults to
      `weights.dtype`.
    name: An optional `string`. The name of this operator.
  """
  def __init__(self,
               weights,
               arg_shape=None,
               dtype=None,
               name='LinearOperatorRealWeighting'):

    parameters = dict(
      weights=weights,
      arg_shape=arg_shape,
      dtype=dtype,
      name=name
    )

    # Only real floating-point types allowed.
    self._weights = check_utils.validate_tensor_dtype(
      tf.convert_to_tensor(weights), 'floating', 'weights')

    # If a dtype was specified, cast weights to it.
    if dtype is not None:
      self._weights = tf.cast(self._weights, dtype)

    if arg_shape is None:
      self._dshape = self._weights.shape
    else:
      self._dshape = tf.TensorShape(arg_shape)
      # Check that the last dimensions of `shape.weights` are broadcastable to
      # this shape.
      weights_shape = self.weights.shape[-self.domain_shape.rank:]
      weights_shape = [None if s == 1 else s for s in weights_shape]
      # weights_shape = [None] * (self.domain_shape.rank - len(weights_shape)) + weights_shape
      if not self.domain_shape.is_compatible_with(weights_shape):
        raise ValueError(
          f"`weights.shape` must be broadcast compatible with `arg_shape`. "
          f"Received shapes {str(weights_shape)} and "
          f"{str(self.domain_shape)}, respectively.")

    self._rshape = self.domain_shape
    self._bshape = self.weights.shape[:-self.domain_shape.rank]

    # This operator acts like a diagonal matrix. It does not inherit from
    # `LinearOperatorDiag` for efficiency reasons, as the diagonal values may
    # be repeated periodically.
    super().__init__(self._weights.dtype,
                     is_non_singular=None,
                     is_self_adjoint=True,
                     is_positive_definite=None,
                     is_square=True,
                     name=name,
                     parameters=parameters)

  def _domain_shape(self):
    return self._dshape

  def _range_shape(self):
    return self._rshape

  def _batch_shape(self):
    return self._bshape

  def _matvec(self, x, adjoint=False):

    arg_batch_shape = x.shape[:-1]
    x = tf.reshape(x, arg_batch_shape + self.domain_shape)
    x *= self.weights
    x = tf.reshape(x, arg_batch_shape.as_list() + [-1])
    return x

  @property
  def weights(self):
    return self._weights



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

def conjugate_gradient(operator,
                       rhs,
                       preconditioner=None,
                       x=None,
                       tol=1e-5,
                       max_iter=20,
                       name='conjugate_gradient'):
  r"""Conjugate gradient solver.

  Solves a linear system of equations `A*x = rhs` for self-adjoint, positive
  definite matrix `A` and right-hand side vector `rhs`, using an iterative,
  matrix-free algorithm where the action of the matrix A is represented by
  `operator`. The iteration terminates when either the number of iterations
  exceeds `max_iter` or when the residual norm has been reduced to `tol`
  times its initial value, i.e. \\(||rhs - A x_k|| <= tol ||rhs||\\).

  .. note::
    This function is mostly equivalent to
    `tf.linalg.experimental.conjugate_gradient`, except it adds support for
    complex-valued linear systems.

  Args:
    operator: A `LinearOperator` that is self-adjoint and positive definite.
    rhs: A possibly batched vector of shape `[..., N]` containing the right-hand
      size vector.
    preconditioner: A `LinearOperator` that approximates the inverse of `A`.
      An efficient preconditioner could dramatically improve the rate of
      convergence. If `preconditioner` represents matrix `M`(`M` approximates
      `A^{-1}`), the algorithm uses `preconditioner.apply(x)` to estimate
      `A^{-1}x`. For this to be useful, the cost of applying `M` should be
      much lower than computing `A^{-1}` directly.
    x: A possibly batched vector of shape `[..., N]` containing the initial
      guess for the solution.
    tol: A float scalar convergence tolerance.
    max_iter: An integer giving the maximum number of iterations.
    name: A name scope for the operation.

  Returns:
    output: A namedtuple representing the final state with fields:
      - i: A scalar `int32` `Tensor`. Number of iterations executed.
      - x: A rank-1 `Tensor` of shape `[..., N]` containing the computed
          solution.
      - r: A rank-1 `Tensor` of shape `[.., M]` containing the residual vector.
      - p: A rank-1 `Tensor` of shape `[..., N]`. `A`-conjugate basis vector.
      - gamma: \\(r \dot M \dot r\\), equivalent to  \\(||r||_2^2\\) when
        `preconditioner=None`.

  Raises:
    ValueError: If `operator` is not self-adjoint and positive definite.
  """
  if not (operator.is_self_adjoint and operator.is_positive_definite):
    raise ValueError('Expected a self-adjoint, positive definite operator.')

  cg_state = collections.namedtuple('CGState', ['i', 'x', 'r', 'p', 'gamma'])

  def stopping_criterion(i, state):
    return tf.math.logical_and(
        i < max_iter,
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

  with tf.name_scope(name):
    broadcast_shape = tf.broadcast_dynamic_shape(
        tf.shape(rhs)[:-1],
        operator.batch_shape_tensor())
    if preconditioner is not None:
      broadcast_shape = tf.broadcast_dynamic_shape(
          broadcast_shape,
          preconditioner.batch_shape_tensor()
      )
    broadcast_rhs_shape = tf.concat([
        broadcast_shape, [tf.shape(rhs)[-1]]], axis=-1)
    r0 = tf.broadcast_to(rhs, broadcast_rhs_shape)
    tol *= tf.norm(r0, axis=-1)

    if x is None:
      x = tf.zeros(
          broadcast_rhs_shape, dtype=rhs.dtype.base_dtype)
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
    return cg_state(
        state.i,
        x=state.x,
        r=state.r,
        p=state.p,
        gamma=state.gamma)
