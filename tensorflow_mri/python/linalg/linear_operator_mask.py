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
"""Masking linear operator."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_nd
from tensorflow_mri.python.util import api_util


@api_util.export("linalg.LinearOperatorMask")
@linear_operator_nd.make_linear_operator_nd
class LinearOperatorMask(linear_operator_nd.LinearOperatorND):
  """Linear operator acting like a [batch] masking matrix.

  A masking matrix is a diagonal matrix whose diagonal entries are either one
  or zero. This operator is useful for masking out certain entries in a vector
  or matrix.

  ```{tip}
  You can use this operator to mask *k*-space values in undersampled Cartesian
  MRI.
  ```

  ```{rubric} Inversion
  ```
  In general, this operator is singular and cannot be inverted, so `solve`
  and `inverse` will raise an error.

  However, you can use `lstsq` to solve the associated least-squares problem.

  Example:
    >>> mask = [True, False, True, False]
    >>> linop = tfmri.linalg.LinearOperatorMask(mask)
    >>> x = tf.constant([1., 2., 3., 4.])
    >>> linop.matvec_nd(x).numpy()
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 0., 3., 0.], dtype=float32)>

  Args:
    mask: A boolean `tf.Tensor` of shape `[..., *spatial_shape]`.
    batch_dims: An `int`, the number of batch dimensions in `mask`.
    dtype: The `dtype` of the operator. Must be float or complex. If `None`,
      defaults to `float32`.
    algorithm: A `str`, one of `'multiply'` or `'multiplex'`. The algorithm to
      use for masking.
      - `'multiply'` (default) applies the mask by multiplying each value in
        the input tensor by either one or zero. This is often faster, although
        this depends on the specific problem and your hardware.
      - `'multiplex'` applies the mask by using the input mask as a condition
        and multiplexing the input with a zero tensor. See `tf.where` for more
        details.
      ```{attention}
      The IEEE 754 standard for floating-point arithmetic has a
      [signed zero](https://en.wikipedia.org/wiki/Signed_zero). When using
      `'multiply'`, the zeroed out values will be positive zero for positive
      inputs and negative zero for negative inputs. Therefore, the `'multiply'`
      algorithm leaks sign information. If this is a concern in your
      application, use `'multiplex'` instead.
      ```
    is_non_singular: A boolean, or `None`. Whether this operator is expected
      to be non-singular. Defaults to `False`.
    is_self_adjoint: A boolean, or `None`. Whether this operator is expected
      to be equal to its Hermitian transpose. If `dtype` is real, this is
      equivalent to being symmetric. Defaults to `True`.
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
               mask,
               batch_dims=0,
               dtype=None,
               algorithm='multiply',
               is_non_singular=False,
               is_self_adjoint=True,
               is_positive_definite=None,
               is_square=True,
               name='LinearOperatorMask'):
    parameters = dict(
        mask=mask,
        batch_dims=batch_dims,
        dtype=dtype,
        algorithm=algorithm,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    with tf.name_scope(name) as name:
      if dtype is None:
        dtype = tf.float32
      dtype = tf.dtypes.as_dtype(dtype)
      if not dtype.is_floating and not dtype.is_complex:
        raise TypeError(f"dtype must be float or complex, got {str(dtype)}")

      self._batch_dims = np.asarray(tf.get_static_value(batch_dims))
      if (not self._batch_dims.ndim == 0 or
          not np.issubdtype(self._batch_dims.dtype, np.integer)):
        raise TypeError(
            f"batch_dims must be an int, but got: {batch_dims}")
      self._batch_dims = self._batch_dims.item()
      if self._batch_dims < 0:
        raise ValueError(
            f"batch_dims must be non-negative, but got: {batch_dims}")

      self._mask = tf.convert_to_tensor(mask, name="mask")
      if not self._mask.dtype.is_bool:
        raise TypeError(
            f"mask must be boolean, but got dtype: {str(self._mask.dtype)}")
      if self._mask.shape.rank is None:
        raise ValueError("mask must have known static rank")
      self._ndim_static = self._mask.shape.rank - self._batch_dims
      if self._ndim_static < 1:
        raise ValueError(
            f"mask must be at least 1-D (excluding batch dimensions), "
            f"but got shape: {self._mask.shape}")

      if algorithm not in {'multiply', 'multiplex'}:
        raise ValueError(
            f"algorithm must be one of 'multiply' or 'multiplex', "
            f"but got: {algorithm}")
      if algorithm == 'multiply':
        self._mask_mult = tf.cast(self._mask, dtype)
      self._algorithm = algorithm

      if not is_self_adjoint:
        raise ValueError("A mask operator is always self-adjoint.")
      if is_non_singular:
        raise ValueError("A mask operator is always singular.")
      if not is_square:
        raise ValueError("A mask operator is always square.")

      super().__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _matvec_nd(self, x, adjoint=False):
    # This operator is self-adjoint, so we can ignore the adjoint argument.
    if self._algorithm == 'multiply':
      x = x * self._mask_mult
    elif self._algorithm == 'multiplex':
      x = tf.where(self._mask, x, tf.zeros_like(x))
    else:
      raise ValueError(f"Unknown masking algorithm: {self._algorithm}")
    return x

  def _solvevec_nd(self, rhs, adjoint=False):
    raise ValueError(
        f"{self.name} is not invertible. If you intend to solve the "
        f"associated least-squares problem, use `lstsq`, `lstsqvec` or "
        f"`lstsqvec_nd`.")

  def _lstsqvec_nd(self, rhs, adjoint=False):
    # The value of adjoint is irrelevant, but be pedantic.
    return self._matvec_nd(rhs, adjoint=(not adjoint))

  def _ndim(self):
    return self._ndim_static

  def _domain_shape(self):
    return self._mask.shape[self._batch_dims:]

  def _range_shape(self):
    return self._mask.shape[self._batch_dims:]

  def _batch_shape(self):
    return self._mask.shape[:self._batch_dims]

  def _domain_shape_tensor(self):
    return tf.shape(self._mask)[self._batch_dims:]

  def _range_shape_tensor(self):
    return tf.shape(self._mask)[self._batch_dims:]

  def _batch_shape_tensor(self):
    return tf.shape(self._mask)[:self._batch_dims]

  @property
  def mask(self):
    return self._mask

  @property
  def _composite_tensor_fields(self):
    return ('mask', 'batch_dims', 'dtype', 'algorithm')

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ('batch_dims',)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {'mask': self.ndim}


def mask_matrix(mask, batch_dims=0, dtype=None):
  """Constructs a masking matrix.

  Args:
    mask: A complex `tf.Tensor` of shape `[..., *spatial_shape]`.
    batch_dims: An `int`, the number of batch dimensions in `mask`.

  Returns:
    A `tf.Tensor` representing a dense coil array matrix equivalent to
    `LinearOperatorMask`.
  """
  mask = tf.convert_to_tensor(mask, name="mask")
  mask = tf.cast(mask, dtype or tf.float32)

  # Vectorize N-D mask.
  mask = tf.reshape(
      mask, tf.concat([tf.shape(mask)[:batch_dims], [-1]], axis=0))

  # Construct a [batch] diagonal matrix.
  matrix = tf.linalg.diag(mask)

  return matrix
