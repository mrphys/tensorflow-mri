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

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_nd
from tensorflow_mri.python.linalg import linear_operator_util
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import types_util


@api_util.export("linalg.LinearOperatorDiagND")
@linear_operator_nd.make_linear_operator_nd
class LinearOperatorDiagND(linear_operator_nd.LinearOperatorND):
  r"""Linear operator acting like a [batch] square diagonal matrix.

  This operator acts like a batch of diagonal matrices
  $A \in \mathbb{F}^{n \times n}$, where $\mathbb{F}$ may be $\mathbb{R}$
  or $\mathbb{C}$ and $n = n_0 \times n_1 \times \dots \times n_d$, where
  $d$ is the number of dimensions in the domain.

  ```{note}
  The matrix $A$ is not materialized.
  ```

  ```{seealso}
  This operator is similar to `tfmri.linalg.LinearOperatorDiag`, but provides
  additional functionality to operate with multidimensional inputs.
  ```

  ```{rubric} Initialization
  ```
  This operator is initialized with an array of diagonal elements `diag`.
  `diag` may have multiple domain dimensions, which does not affect the dense
  matrix representation of this operator but may be convenient to operate with
  non-vectorized multidimensional inputs. If `diag` has any leading dimensions
  which should be interpreted as batch dimensions, specify how many using the
  `batch_dims` argument. This operator has the same data type as `diag`.

  ```{rubric} Performance
  ```
  - `matvec` is $O(n)$.
  - `solvevec` is $O(n)$.
  - `lstsqvec` is $O(n)$.

  ```{rubric} Properties
  ```
  - This operator is *non-singular* iff all its diagonal entries are non-zero.
  - This operator is *self-adjoint* iff all its diagonal entries are real or
    have zero imaginary part.
  - This operator is *positive definite* iff all its diagonal entries are
    positive or have positive real part.
  - This operator is always *square*.

  ```{rubric} Inversion
  ```
  If this operator is non-singular, its inverse $A{-1}$ is also a diagonal
  operator whose diagonal entries are the reciprocal of the diagonal entries
  of this operator.

  Example:
    >>> # Create a 2-D diagonal linear operator.
    >>> diag = [[1., -1.], [2., 3.]]
    >>> operator = tfmri.linalg.LinearOperatorDiagND(diag)
    >>> operator.to_dense()
    [[ 1.,  0.,  0.,  0.],
     [ 0., -1.,  0.,  0.],
     [ 0.,  0.,  2.,  0.],
     [ 0.,  0.,  0.,  3.]]
    >>> operator.shape
    (4, 4)
    >>> x = tf.ones(shape=(2, 2))
    >>> rhs = operator.matvec_nd(x)
    [[ 1., -1.],
     [ 2.,  3.]]
    >>> operator.solvevec_nd(rhs)
    [[ 1.,  1.],
     [ 1.,  1.]]

  Args:
    diag: A real or complex `tf.Tensor` of shape `[..., *domain_shape]`.
      The diagonal of the operator.
    batch_dims: An `int`, the number of leading batch dimensions in `diag`.
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
      square matrix (or a batch of square matrices). Defaults to `False`.
    name: An optional `str`. The name of this operator.
  """
  def __init__(self,
               diag,
               batch_dims=0,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorDiag"):
    parameters = dict(
        diag=diag,
        batch_dims=batch_dims,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    with tf.name_scope(name):
      # Check batch_dims.
      self._batch_dims = np.asarray(tf.get_static_value(batch_dims))
      if (not self._batch_dims.ndim == 0 or
          not np.issubdtype(self._batch_dims.dtype, np.integer)):
        raise TypeError(
            f"batch_dims must be an int, but got: {batch_dims}")
      self._batch_dims = self._batch_dims.item()
      if self._batch_dims < 0:
        raise ValueError(
            f"batch_dims must be non-negative, but got: {batch_dims}")

      # Check maps.
      self._diag = types_util.convert_nonref_to_tensor(diag, name="diag")
      if self._diag.shape.rank is None:
        raise ValueError("diag must have known static rank")
      if self._diag.shape.rank < 1:
        raise ValueError(
            f"diag must be at least 1-D, but got shape: {self._diag.shape}")

      # Check and auto-set hints.
      if not self._diag.dtype.is_complex:
        if is_self_adjoint is False:
          raise ValueError("A real diagonal operator is always self adjoint.")
        is_self_adjoint = True

      if is_square is False:
        raise ValueError("Only square diagonal operators currently supported.")
      is_square = True

      super().__init__(
          dtype=self._diag.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _domain_shape(self):
    return self._diag.shape[self._batch_dims:]

  def _range_shape(self):
    return self._diag.shape[self._batch_dims:]

  def _batch_shape(self):
    return self._diag.shape[:self._batch_dims]

  def _domain_shape_tensor(self):
    return tf.shape(self._diag)[self._batch_dims:]

  def _range_shape_tensor(self):
    return tf.shape(self._diag)[self._batch_dims:]

  def _batch_shape_tensor(self):
    return tf.shape(self._diag)[:self._batch_dims]

  def _assert_non_singular(self):
    return linear_operator_util.assert_no_entries_with_modulus_zero(
        self._diag,
        message=(
            "Diagonal operator is singular: "
            "diagonal entries contain zero values."))

  def _assert_positive_definite(self):
    if self.dtype.is_complex:
      message = (
          "Diagonal operator has diagonal entries with non-positive real part, "
          "so it is not positive definite.")
    else:
      message = (
          "Real diagonal operator has non-positive diagonal entries, "
          "so it is not positive definite.")

    return tf.debugging.assert_positive(
        tf.math.real(self._diag), message=message)

  def _assert_self_adjoint(self):
    return linear_operator_util.assert_zero_imag_part(
        self._diag,
        message=(
            "This diagonal operator contains non-zero imaginary values, "
            "so it is not self-adjoint."))

  def _matvec_nd(self, x, adjoint=False):
    diag_term = tf.math.conj(self._diag) if adjoint else self._diag
    return diag_term * x

  def _determinant(self):
    return tf.math.reduce_prod(self._diag, axis=self._diag_axes)

  def _log_abs_determinant(self):
    log_det = tf.math.reduce_sum(
        tf.math.log(tf.math.abs(self._diag)), axis=self._diag_axes)
    if self.dtype.is_complex:
      log_det = tf.cast(log_det, dtype=self.dtype)
    return log_det

  def _solvevec_nd(self, rhs, adjoint=False):
    diag_term = tf.math.conj(self._diag) if adjoint else self._diag
    inv_diag_term = 1. / diag_term
    return inv_diag_term * rhs

  def _lstsqvec_nd(self, rhs, adjoint=False):
    return self._solvevec_nd(rhs, adjoint=adjoint)

  def _to_dense(self):
    return tf.linalg.diag(self._flat_diag)

  def _diag_part(self):
    return self._flat_diag

  def _add_to_tensor(self, x):
    x_diag = tf.linalg.diag_part(x)
    new_diag = self._flat_diag + x_diag
    return tf.linalg.set_diag(x, new_diag)

  def _eigvals(self):
    return tf.convert_to_tensor(self.diag)

  def _cond(self):
    abs_diag = tf.math.abs(self.diag)
    return (tf.math.reduce_max(abs_diag, axis=self._diag_axes) /
            tf.math.reduce_min(abs_diag, axis=self._diag_axes))

  @property
  def diag(self):
    return self._diag

  @property
  def _diag_axes(self):
    return list(range(self._batch_dims, self._diag.shape.rank))

  @property
  def _flat_diag(self):
    return tf.reshape(
        self._diag, tf.concat([self.batch_shape_tensor(), [-1]], 0))

  @property
  def _composite_tensor_fields(self):
    return ("diag", "batch_dims")

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("batch_dims",)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {"diag": self._diag.shape.rank - self._batch_dims}
