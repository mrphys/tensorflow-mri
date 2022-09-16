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

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util


@api_util.export("linalg.LinearOperatorFFT")
@linear_operator.make_composite_tensor
class LinearOperatorFFT(linear_operator.LinearOperator):
  """Linear operator acting like a [batch] Fourier transform matrix."""
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

      # Parse domain shape.
      self._domain_shape_static, self._domain_shape_dynamic = (
          tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))
      if batch_shape is not None:
        self._batch_shape_static, self._batch_shape_dynamic = (
            tensor_util.static_and_dynamic_shapes_from_shape(batch_shape))
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

  def _transform(self, x, adjoint=False):
    if self.rank is not None:
      axes = list(range(-self.rank, 0))
    else:
      axes = tf.range(-self.rank_tensor(), 0)
    if adjoint:
      x = fft_ops.ifftn(x, axes=axes, norm='ortho', shift=True)
    else:
      x = fft_ops.fftn(x, axes=axes, norm='ortho', shift=True)
    return x

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
  def rank(self):
    return self.domain_shape.rank

  def rank_tensor(self):
    if self.rank is not None:  # Prefer static rank if available.
      return tf.convert_to_tensor(self.rank, dtype=tf.int32)
    return tf.size(self.domain_shape_tensor())

  @property
  def _composite_tensor_fields(self):
    return ('domain_shape', 'batch_shape', 'dtype')

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("domain_shape", "batch_shape")
