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
"""Fourier linear operator."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_nd
from tensorflow_mri.python.linalg import slicing
from tensorflow_mri.python.linalg import linear_operator_util
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util
from tensorflow_mri.python.util import types_util


@api_util.export("linalg.LinearOperatorFFT")
@linear_operator_nd.make_linear_operator_nd
class LinearOperatorFFT(linear_operator_nd.LinearOperatorND):
  r"""Linear operator acting like a [batch] DFT matrix.

  If this operator is $A$, then $A x$ computes the Fourier transform of $x$,
  while $A^H x$ computes the inverse Fourier transform of $x$. Note that the
  inverse and the adjoint are equivalent, i.e. $A^H = A^{-1}$.

  The DFT matrix is never materialized internally. Instead matrix-matrix and
  matrix-vector products are computed using the fast Fourier transform (FFT)
  algorithm.

  This operator supports N-dimensional inputs, whose shape must be specified
  through the `domain_shape` argument. This operator also acccepts an optional
  `batch_shape` argument, which will be relevant for broadcasting purposes.

  This operator only supports complex inputs. Specify the desired type using
  the `dtype` argument.

  Example:

  >>> # Create a 2-dimensional 128x128 DFT operator.
  >>> linop = tfmri.linalg.LinearOperatorFFT(domain_shape=[128, 128])

  Args:
    domain_shape: A 1D integer `tf.Tensor`. The domain shape of the operator,
      representing the shape of the inputs to `matvec`.
    batch_shape: A 1D integer `tf.Tensor`. The batch shape of the operator.
      Defaults to `None`, which is equivalent to `[]`.
    dtype: A `tf.dtypes.DType`. Must be complex. Defaults to `complex64`.
    is_non_singular: A boolean, or `None`. Whether this operator is expected
      to be non-singular. Defaults to `True`.
    is_self_adjoint: A boolean, or `None`. Whether this operator is expected
      to be equal to its Hermitian transpose. If `dtype` is real, this is
      equivalent to being symmetric. Defaults to `False`.
    is_positive_definite: A boolean, or `None`. Whether this operator is
      expected to be positive definite, meaning the quadratic form $x^H A x$
      has positive real part for all nonzero $x$. Note that an operator [does
      not need to be self-adjoint to be positive definite](https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices)
      Defaults to `None`.
    is_square: A boolean, or `None`. Expect that this operator acts like a
      square matrix (or a batch of square matrices). Defaults to `True`.
    name: A `name`. The name to give to the ops created by this class.
  """
  def __init__(self,
               domain_shape,
               batch_shape=None,
               dtype=None,
               is_non_singular=True,
               is_self_adjoint=False,
               is_positive_definite=None,
               is_square=True,
               name='LinearOperatorFFT'):

    parameters = dict(
        domain_shape=domain_shape,
        batch_shape=batch_shape,
        dtype=dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)

    dtype = dtype or tf.complex64

    with tf.name_scope(name):
      dtype = tf.dtypes.as_dtype(dtype)
      if not is_non_singular:
        raise ValueError("An FFT operator is always non-singular.")
      if is_self_adjoint:
        raise ValueError("An FFT operator is never self-adjoint.")
      if not is_square:
        raise ValueError("An FFT operator is always square.")

      # Get static/dynamic domain shape.
      types_util.assert_not_ref_type(domain_shape, 'domain_shape')
      self._domain_shape_static, self._domain_shape_dynamic = (
          tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))
      if self._domain_shape_static.rank is None:
        raise ValueError('domain_shape must have known static rank')

      # Get static/dynamic batch shape.
      if batch_shape is not None:
        types_util.assert_not_ref_type(batch_shape, 'batch_shape')
        self._batch_shape_static, self._batch_shape_dynamic = (
            tensor_util.static_and_dynamic_shapes_from_shape(batch_shape))
        if self._batch_shape_static.rank is None:
          raise ValueError('batch_shape must have known static rank')
      else:
        self._batch_shape_static = tf.TensorShape([])
        self._batch_shape_dynamic = tf.constant([], dtype=tf.int32)

      super().__init__(dtype,
                       is_non_singular=is_non_singular,
                       is_self_adjoint=is_self_adjoint,
                       is_positive_definite=is_positive_definite,
                       is_square=is_square,
                       parameters=parameters,
                       name=name)

  def _matvec_nd(self, x, adjoint=False):
    axes = list(range(-self.ndim, 0))

    if adjoint:
      x = fft_ops.ifftn(x, axes=axes, norm='ortho', shift=True)
    else:
      x = fft_ops.fftn(x, axes=axes, norm='ortho', shift=True)

    # For consistent broadcasting semantics.
    if adjoint:
      output_shape = self.domain_shape_tensor()
    else:
      output_shape = self.range_shape_tensor()

    if self.batch_shape.rank > 0:
      x = tf.broadcast_to(
          x, tf.concat([self.batch_shape_tensor(), output_shape], 0))

    return x

  def _solvevec_nd(self, rhs, adjoint=False):
    return self._matvec_nd(rhs, adjoint=(not adjoint))

  def _lstsqvec_nd(self, rhs, adjoint=False):
    return self._solvevec_nd(rhs, adjoint=adjoint)

  def _ndim(self):
    return self.domain_shape.rank

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

  @property
  def _composite_tensor_fields(self):
    return ('domain_shape', 'batch_shape', 'dtype')

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ('domain_shape', 'batch_shape')

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {}

  def __getitem__(self, slices):
    # Support slicing.
    new_batch_shape = tf.shape(tf.ones(self.batch_shape_tensor())[slices])
    return slicing.batch_slice(
        self, params_overrides={'batch_shape': new_batch_shape}, slices=slices)


def dft_matrix(num_rows,
               batch_shape=None,
               dtype=tf.complex64,
               shift=False,
               name=None):
  """Constructs a discrete Fourier transform (DFT) matrix.

  Args:
    num_rows: A non-negative `int32` scalar `tf.Tensor` giving the number
      of rows in each batch matrix.
    batch_shape: A 1D integer `tf.Tensor`. If provided, the returned
      `tf.Tensor` will have leading batch dimensions of this shape.
    dtype: A `tf.dtypes.DType`. The type of an element in the resulting
      `tf.Tensor`. Must be complex. Defaults to `tf.complex64`.
    shift: A boolean. If `True`, returns the matrix for a DC-centred DFT.
    name: A name for this op.

  Returns:
    A `tf.Tensor` of shape `batch_shape + [num_rows, num_rows]` and type
    `dtype` containing a DFT matrix.
  """
  with tf.name_scope(name or "dft_matrix"):
    num_rows = tf.convert_to_tensor(num_rows)
    if batch_shape is not None:
      batch_shape = tensor_util.convert_shape_to_tensor(batch_shape)
    dtype = tf.dtypes.as_dtype(dtype)
    if not dtype.is_complex:
      raise TypeError(f"dtype must be complex, got {str(dtype)}")

    i = tf.range(num_rows, dtype=dtype.real_dtype)
    omegas = tf.reshape(
        tf.math.exp(tf.dtypes.complex(
            tf.constant(0.0, dtype=dtype.real_dtype),
            -2.0 * np.pi * i / tf.cast(num_rows, dtype.real_dtype))), [-1, 1])
    m = omegas ** tf.cast(i, dtype)
    m /= tf.math.sqrt(tf.cast(num_rows, dtype))

    if shift:
      m = tf.signal.fftshift(m)

    if batch_shape is not None:
      m = tf.broadcast_to(m, tf.concat([batch_shape, [num_rows, num_rows]], 0))

    return m
