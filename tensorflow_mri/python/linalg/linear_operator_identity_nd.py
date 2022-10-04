# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
"""(Scaled) identity N-D linear operator."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_nd
from tensorflow_mri.python.ops import control_flow_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util
from tensorflow_mri.python.util import types_util


class BaseLinearOperatorIdentityND(linear_operator_nd.LinearOperatorND):
  """Base class for Identity operators."""

  def _check_domain_shape_possibly_add_asserts(self):
    """Static check of init arg `domain_shape`, possibly add asserts."""
    # Possibly add asserts.
    if self._assert_proper_shapes:
      self._domain_shape_arg = tf.compat.v1.with_dependencies([
          tf.debugging.assert_rank(
              self._domain_shape_arg,
              1,
              message="Argument domain_shape must be a 1-D Tensor."),
          tf.debugging.assert_non_negative(
              self._domain_shape_arg,
              message="Argument domain_shape must be non-negative."),
      ], self._domain_shape_arg)

    # Static checks.
    if not self._domain_shape_arg.dtype.is_integer:
      raise TypeError(f"Argument domain_shape must be integer type. "
                      f"Found: {self._domain_shape_arg}")

    domain_shape_static = self._domain_shape_static

    if domain_shape_static is None:
      return  # Cannot do any other static checks.

    if domain_shape_static.ndim != 1:
      raise ValueError(f"Argument domain_shape must be a 0-D Tensor. "
                       f"Found: {domain_shape_static}")

    if any(s is not None and s < 0 for s in domain_shape_static):
      raise ValueError(f"Argument domain_shape must be non-negative. "
                       f"Found: {domain_shape_static}")

  def _ones_diag(self):
    """Returns the diagonal of this operator as all ones."""
    if self.shape.is_fully_defined():
      diag_shape = self.batch_shape.concatenate([self.domain_dimension])
    else:
      diag_shape = tf.concat(
          [self.batch_shape_tensor(),
           [self.domain_dimension_tensor()]], axis=0)

    return tf.ones(shape=diag_shape, dtype=self.dtype)

  def _check_compatible_input_shape(self, x):
    """Check that an argument to solve/matmul has proper domain shape.

    Adds an assertion op to the graph is `assert_proper_shapes` is `True`.

    Args:
      x: A `tf.Tensor`.

    Returns:
      A `tf.Tensor` with asserted shape.
    """
    # Static checks are done in the base class. Only tensor asserts here.
    if self._assert_proper_shapes:
      assert_compatible_shapes = tf.debugging.assert_equal(
          tf.shape(x)[-self.domain_shape.rank:],
          self.domain_shape_tensor(),
          message="Shapes are incompatible.")
      x = control_flow_ops.with_dependencies([assert_compatible_shapes], x)
    return x


@api_util.export("linalg.LinearOperatorIdentityND")
@linear_operator_nd.make_linear_operator_nd
class LinearOperatorIdentityND(BaseLinearOperatorIdentityND):
  r"""Linear operator acting like a [batch] square identity matrix.

  This operator acts like a batch of identity matrices
  $A = I \in \mathbb{F}^{n \times n}$, where $\mathbb{F}$ may be $\mathbb{R}$
  or $\mathbb{C}$ and $n = n_0 \times n_1 \times \dots \times n_d$, where
  $d$ is the number of dimensions in the domain.

  ```{note}
  The matrix $A$ is not materialized.
  ```

  ```{seealso}
  This operator is similar to `tfmri.linalg.LinearOperatorIdentity`, but
  provides additional functionality to operate with multidimensional inputs.
  ```

  ```{rubric} Initialization
  This operator is initialized with a `domain_shape`, which specifies the
  sizes for the domain dimensions. There may be multiple domain dimensions,
  which does not affect the dense matrix representation of this operator but
  may be convenient to operate with non-vectorized multidimensional inputs.
  This operator may also have a `batch_shape`, which will be relevant for the
  purposes of broadcasting. Use the `dtype` argument to specify this
  operator's data type.

  ```{rubric} Performance
  ```
  - `matvec` is usually $O(1)$, but may be $O(n)$ if broadcasting is needed.
  - `solvevec` is usually $O(1)$, but may be $O(n)$ if broadcasting is needed.
  - `lstsqvec` is usually $O(1)$, but may be $O(n)$ if broadcasting is needed.

  ```{rubric} Properties
  ```
  - This operator is always *non-singular*.
  - This operator is always *self-adjoint*.
  - This operator is always *positive definite*.
  - This operator is always *square*.

  ```{rubric} Inversion
  ```
  The inverse of this operator is equal to the operator itself ($A{-1} = A$).

  Example:
    >>> # Create a 2-D identity operator.
    >>> operator = tfmri.linalg.LinearOperatorIdentityND([2, 2])
    >>> operator.to_dense()
    [[1., 0., 0., 0.],
     [0., 1., 0., 0.]
     [0., 0., 1., 0.],
     [0., 0., 1., 0.]]
    >>> operator.shape
    (4, 4)
    >>> x = tf.reshape(tf.range(4.), (2, 2))
    >>> rhs = operator.matvec_nd(x)
    [[1., 2.],
     [3., 4.]]
    >>> operator.solvevec_nd(rhs)
    [[1., 2.],
     [3., 4.]]

  Args:
    domain_shape: A 1-D non-negative integer `tf.Tensor`. The domain shape
      of this operator.
    batch_shape: A 1-D non-negative integer `tf.Tensor`. The leading batch
      shape of this operator. If `None`, this operator has no
      batch dimensions.
    dtype: A `tf.dtypes.DType`. The data type of the matrix that this operator
      represents.
    is_non_singular: A boolean, or `None`. Whether this operator is expected
      to be non-singular. Defaults to `True`.
    is_self_adjoint: A boolean, or `None`. Whether this operator is expected
      to be equal to its Hermitian transpose. If `dtype` is real, this is
      equivalent to being symmetric. Defaults to `True`.
    is_positive_definite: A boolean, or `None`. Whether this operator is
      expected to be positive definite, meaning the quadratic form $x^H A x$
      has positive real part for all nonzero $x$. Note that an operator [does
      not need to be self-adjoint to be positive definite](https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices)
      Defaults to `True`.
    is_square: A boolean, or `None`. Expect that this operator acts like a
      square matrix (or a batch of square matrices). Defaults to `True`.
    name: An optional `str`. The name of this operator.
  """
  def __init__(self,
               domain_shape,
               batch_shape=None,
               dtype=None,
               is_non_singular=True,
               is_self_adjoint=True,
               is_positive_definite=True,
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorIdentityND"):
    parameters = dict(
        domain_shape=domain_shape,
        batch_shape=batch_shape,
        dtype=dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        assert_proper_shapes=assert_proper_shapes,
        name=name)

    dtype = dtype or tf.dtypes.float32
    self._assert_proper_shapes = assert_proper_shapes

    with tf.name_scope(name):
      dtype = tf.dtypes.as_dtype(dtype)
      if not is_self_adjoint:
        raise ValueError("An identity operator is always self-adjoint.")
      if not is_non_singular:
        raise ValueError("An identity operator is always non-singular.")
      if not is_positive_definite:
        raise ValueError("An identity operator is always positive-definite.")
      if not is_square:
        raise ValueError("An identity operator is always square.")

      super().__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

      types_util.assert_not_ref_type(domain_shape, "domain_shape")
      types_util.assert_not_ref_type(batch_shape, "batch_shape")

      self._domain_shape_static, self._domain_shape_dynamic = (
          tensor_util.static_and_dynamic_shapes_from_shape(
              domain_shape,
              assert_proper_shape=self._assert_proper_shapes,
              arg_name="domain_shape"))
      if self._domain_shape_static.rank is None:
        raise ValueError("domain_shape must have known static rank")

      if batch_shape is None:
        self._batch_shape_static = tf.TensorShape([])
        self._batch_shape_dynamic = tf.constant([], dtype=tf.int32)
      else:
        self._batch_shape_static, self._batch_shape_dynamic = (
            tensor_util.static_and_dynamic_shapes_from_shape(
                batch_shape,
                assert_proper_shape=self._assert_proper_shapes,
                arg_name="batch_shape"))

  def _domain_shape(self):
    return self._domain_shape_static

  def _range_shape(self):
    return self._domain_shape_static

  def _batch_shape(self):
    return self._batch_shape_static

  def _domain_shape_tensor(self):
    return self._domain_shape_dynamic

  def _range_shape_tensor(self):
    return self._domain_shape_dynamic

  def _batch_shape_tensor(self):
    return self._batch_shape_dynamic

  def _assert_non_singular(self):
    return tf.no_op("assert_non_singular")

  def _assert_positive_definite(self):
    return tf.no_op("assert_positive_definite")

  def _assert_self_adjoint(self):
    return tf.no_op("assert_self_adjoint")

  def _possibly_broadcast_batch_shape(self, x):
    """Return 'x', possibly after broadcasting the leading dimensions."""
    # If we have no batch shape, our batch shape broadcasts with everything!
    if self.batch_shape.rank == 0:
      return x

    # Static attempt:
    #   If we determine that no broadcast is necessary, pass x through
    #   If we need a broadcast, add to an array of zeros.
    #
    # special_shape is the shape that, when broadcast with x's shape, will give
    # the correct broadcast_shape.  Note that
    #   We have already verified the second to last dimension of self.shape
    #   matches x's shape in _check_compatible_input_shape.
    #   Also, the final dimension of 'x' can have any shape.
    #   Therefore, the final two dimensions of special_shape are ones.
    special_shape = self.batch_shape.concatenate([1] * self.domain_shape.rank)
    bcast_shape = tf.broadcast_static_shape(x.shape, special_shape)
    if special_shape.is_fully_defined():
      if bcast_shape == x.shape:
        # Input already has correct shape. Broadcasting is not necessary.
        return x
      # Use the built in broadcasting of addition.
      zeros = tf.zeros(shape=special_shape, dtype=self.dtype)
      return x + zeros

    # Dynamic broadcast:
    #   Always add to an array of zeros, rather than using a "cond", since a
    #   cond would require copying data from GPU --> CPU.
    special_shape = tf.concat(
        [self.batch_shape_tensor(), [1] * self.domain_shape.rank], 0)
    zeros = tf.zeros(shape=special_shape, dtype=self.dtype)
    return x + zeros

  def _matvec_nd(self, x, adjoint=False):
    # Note that adjoint has no effect since this matrix is self-adjoint.
    x = self._check_compatible_input_shape(x)
    return self._possibly_broadcast_batch_shape(x)

  def _solvevec_nd(self, rhs, adjoint=False):
    return self._matvec_nd(rhs)

  def _lstsqvec_nd(self, rhs, adjoint=False):
    return self._matvec_nd(rhs)

  def _determinant(self):
    return tf.ones(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _log_abs_determinant(self):
    return tf.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _trace(self):
    if self.batch_shape.is_fully_defined():
      ones = tf.ones(shape=self.batch_shape, dtype=self.dtype)
    else:
      ones = tf.ones(shape=self.batch_shape_tensor(), dtype=self.dtype)

    return ones * tf.cast(self.domain_dimension_tensor(), self.dtype)

  def _diag_part(self):
    return self._ones_diag()

  def add_to_tensor(self, mat, name="add_to_tensor"):
    """Add matrix represented by this operator to `mat`. Equiv to `I + mat`.

    Args:
      mat: A `tf.Tensor` with same `dtype` and shape broadcastable to `self`.
      name: A name to give this `Op`.

    Returns:
      A `tf.Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      mat = tf.convert_to_tensor(mat, name="mat")
      mat_diag = tf.linalg.diag_part(mat)
      new_diag = 1 + mat_diag
      return tf.linalg.set_diag(mat, new_diag)

  def _eigvals(self):
    return self._ones_diag()

  def _cond(self):
    return tf.ones(self.batch_shape_tensor(), dtype=self.dtype)

  def _to_dense(self):
    return tf.eye(
        num_rows=self.domain_dimension_tensor(),
        batch_shape=self.batch_shape_tensor(),
        dtype=self.dtype)

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("domain_shape", "batch_shape")

  @property
  def _composite_tensor_fields(self):
    return ("domain_shape", "batch_shape", "dtype", "assert_proper_shapes")

  def __getitem__(self, slices):
    # Slice the batch shape and return a new LinearOperatorIdentity.
    # Use a proxy tensor and slice it. Use this as the new batch shape.
    new_batch_shape = tf.shape(tf.ones(self._batch_shape_dynamic)[slices])
    parameters = dict(self.parameters, batch_shape=new_batch_shape)
    return LinearOperatorIdentityND(**parameters)


@api_util.export("linalg.LinearOperatorScaledIdentityND")
@linear_operator_nd.make_linear_operator_nd
class LinearOperatorScaledIdentityND(BaseLinearOperatorIdentityND):
  r"""Linear operator acting like a scaled [batch] identity matrix.

  This operator acts like a batch of scaled identity matrices
  $A = \lambda I \in \mathbb{F}^{n \times n}$, where $\lambda$ is a scaling
  constant, $\mathbb{F}$ may be $\mathbb{R}$ or $\mathbb{C}$ and
  $n = n_0 \times n_1 \times \dots \times n_d$, where
  $d$ is the number of dimensions in the domain.

  ```{note}
  The matrix $A$ is not materialized.
  ```

  ```{seealso}
  This operator is similar to `tfmri.linalg.LinearOperatorScaledIdentityND`,
  but provides additional functionality to operate with multidimensional
  inputs.
  ```

  ```{rubric} Initialization
  This operator is initialized with a `domain_shape`, which specifies the
  sizes for the domain dimensions, and a `multiplier`, which specifies the
  scaling constant $\lambda$. `domain_shape` may have multiple dimensions,
  which does not affect the dense matrix representation of this operator but
  may be convenient to operate with non-vectorized multidimensional inputs.
  This operator has the same data type as `multiplier`.

  ```{rubric} Performance
  ```
  - `matvec` is $O(n)$.
  - `solvevec` is $O(n)$.
  - `lstsqvec` is $O(n)$.

  ```{rubric} Properties
  ```
  - This operator is *non-singular* iff multiplier is non-zero.
  - This operator is *self-adjoint* iff multiplier is real or has zero
    imaginary part.
  - This operator is *positive definite* iff multiplier has positive real part.
  - This operator is always *square*.

  ```{rubric} Inversion
  ```
  If this operator is non-singular, its inverse $A^{-1}$ is also a scaled
  identity operator with reciprocal multiplier.

  Example:
    >>> # Create a 2-D identity operator.
    >>> operator = tfmri.linalg.LinearOperatorIdentityND([2, 2])
    >>> operator.to_dense()
    [[1., 0., 0., 0.],
     [0., 1., 0., 0.]
     [0., 0., 1., 0.],
     [0., 0., 1., 0.]]
    >>> operator.shape
    (4, 4)
    >>> x = tf.reshape(tf.range(4.), (2, 2))
    >>> rhs = operator.matvec_nd(x)
    [[1., 2.],
     [3., 4.]]
    >>> operator.solvevec_nd(rhs)
    [[1., 2.],
     [3., 4.]]

  Args:
    domain_shape: A 1-D non-negative integer `tf.Tensor`. The domain shape
      of this operator.
    multiplier: A real or complex `tf.Tensor` of any shape specifying the
      scaling constant for the identity matrix.
    dtype: A `tf.dtypes.DType`. The data type of the matrix that this operator
      represents.
    is_non_singular: A boolean, or `None`. Whether this operator is expected
      to be non-singular. Defaults to `None`.
    is_self_adjoint: A boolean, or `None`. Whether this operator is expected
      to be equal to its Hermitian transpose. If `dtype` is real, this is
      equivalent to being symmetric. Defaults to `None`.
    is_positive_definite: A boolean, or `None`. Whether this operator is
      expected to be positive definite, meaning the quadratic form $x^H A x$
      has positive real part for all nonzero $x$. Note that an operator [does
      not need to be self-adjoint to be positive definite](https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices)
      Defaults to `None`.
    is_square: A boolean, or `None`. Expect that this operator acts like a
      square matrix (or a batch of square matrices). Defaults to `True`.
    name: An optional `str`. The name of this operator.
  """
  def __init__(self,
               domain_shape,
               multiplier,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorScaledIdentityND"):
    parameters = dict(
        domain_shape=domain_shape,
        multiplier=multiplier,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        assert_proper_shapes=assert_proper_shapes,
        name=name)

    self._assert_proper_shapes = assert_proper_shapes

    with tf.name_scope(name):
      # Check domain_shape.
      types_util.assert_not_ref_type(domain_shape, "domain_shape")
      self._domain_shape_static, self._domain_shape_dynamic = (
          tensor_util.static_and_dynamic_shapes_from_shape(
              domain_shape,
              assert_proper_shape=self._assert_proper_shapes,
              arg_name="domain_shape"))
      if self._domain_shape_static.rank is None:
        raise ValueError("domain_shape must have known static rank")

      # Check multiplier.
      self._multiplier = types_util.convert_nonref_to_tensor(
          multiplier, name="multiplier")

      # Check and auto-set hints.
      if not self._multiplier.dtype.is_complex:
        if is_self_adjoint is False:  # pylint: disable=g-bool-id-comparison
          raise ValueError(
              "A real scaled identity operator is always self adjoint.")
        is_self_adjoint = True

      if not is_square:
        raise ValueError("A scaled identity operator is always square.")

      super().__init__(
          dtype=self._multiplier.dtype.base_dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _domain_shape(self):
    return self._domain_shape_static

  def _range_shape(self):
    return self._domain_shape_static

  def _batch_shape(self):
    return self._multiplier.shape

  def _domain_shape_tensor(self):
    return self._domain_shape_dynamic

  def _range_shape_tensor(self):
    return self._domain_shape_dynamic

  def _batch_shape_tensor(self):
    return tf.shape(self._multiplier)

  def _assert_non_singular(self):
    return tf.debugging.assert_positive(
        tf.math.abs(self.multiplier),
        message=("Scaled identity operator is singular: "
                 "multiplier contains zero entries."))

  def _assert_positive_definite(self):
    if self.dtype.is_complex:
      message = ("Scaled identity operator is not positive definite: "
                 "multiplier contains entries with non-positive real part.")
    else:
      message = ("Scaled identity operator is not positive definite: "
                 "multiplier contains non-positive entries.")
    return tf.debugging.assert_positive(
        tf.math.real(self.multiplier), message=message)

  def _assert_self_adjoint(self):
    if not self.dtype.is_complex:
      # A real scaled identity operator is always self-adjoint.
      return tf.no_op("assert_self_adjoint")
    imag_multiplier = tf.math.imag(self.multiplier)
    return tf.debugging.assert_equal(
        tf.zeros_like(imag_multiplier),
        imag_multiplier,
        message=("Scaled identity operator is not self-adjoint: "
                 "multiplier contains entries with non-zero imaginary part."))

  def _matvec_nd(self, x, adjoint=False):
    x = self._check_compatible_input_shape(x)
    return x * self._make_multiplier_matrix(adjoint=adjoint)

  def _solvevec_nd(self, rhs, adjoint=False):
    rhs = self._check_compatible_input_shape(rhs)
    return rhs / self._make_multiplier_matrix(adjoint=adjoint)

  def _lstsqvec_nd(self, rhs, adjoint=False):
    return self._solvevec_nd(rhs, adjoint=adjoint)

  def _make_multiplier_matrix(self, adjoint=False):
    multiplier_matrix = tf.reshape(
        self.multiplier,
        tf.concat([tf.shape(self.multiplier), [1] * self.domain_shape.rank], 0))
    multiplier_matrix = tf.ensure_shape(
        multiplier_matrix, self.multiplier.shape.concatenate(
            [1] * self.domain_shape.rank))
    if adjoint:
      multiplier_matrix = tf.math.conj(multiplier_matrix)
    return multiplier_matrix

  def _determinant(self):
    return self.multiplier ** tf.cast(
        self.domain_dimension_tensor(), self.dtype)

  def _log_abs_determinant(self):
    return (tf.math.log(tf.math.abs(self.multiplier)) *
            tf.cast(self.domain_dimension_tensor(), self.dtype.real_dtype))

  def _trace(self):
    return self.multiplier * tf.cast(self.domain_dimension_tensor(), self.dtype)

  def _diag_part(self):
    return self._ones_diag() * self.multiplier[..., tf.newaxis]

  def add_to_tensor(self, mat, name="add_to_tensor"):
    """Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.

    Args:
      mat:  `Tensor` with same `dtype` and shape broadcastable to `self`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      # Shape [B1,...,Bb, 1]
      multiplier_vector = tf.expand_dims(self.multiplier, -1)
      # Shape [C1,...,Cc, M, M]
      mat = tf.convert_to_tensor(mat, name="mat")
      # Shape [C1,...,Cc, M]
      mat_diag = tf.linalg.diag_part(mat)
      # multiplier_vector broadcasts here.
      new_diag = multiplier_vector + mat_diag
      return tf.linalg.set_diag(mat, new_diag)

  def _eigvals(self):
    return self._ones_diag() * self.multiplier[..., tf.newaxis]

  def _cond(self):
    # Condition number for a scalar time identity matrix is one, except when the
    # scalar is zero.
    return tf.where(
        tf.math.equal(self._multiplier, 0.),
        tf.cast(np.nan, dtype=self.dtype),
        tf.cast(1., dtype=self.dtype))

  def _to_dense(self):
    return self.multiplier[..., tf.newaxis, tf.newaxis] * tf.eye(
        num_rows=self.domain_dimension_tensor(),
        dtype=self.dtype)

  @property
  def multiplier(self):
    """The [batch] scalar `tf.Tensor`, $c$ in $cI$."""
    return self._multiplier

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("domain_shape",)

  @property
  def _composite_tensor_fields(self):
    return ("domain_shape", "multiplier", "assert_proper_shapes")

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {"multiplier": 0}
