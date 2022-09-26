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
"""Wavelet linear operator."""

import functools

import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import wavelet_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import tensor_util


@api_util.export("linalg.LinearOperatorWavelet")
class LinearOperatorWavelet(linear_operator.LinearOperator):  # pylint: disable=abstract-method
  """Linear operator representing a wavelet decomposition matrix.

  Args:
    domain_shape: A 1D `tf.Tensor` or a `list` of `int`. The domain shape of
      this linear operator.
    wavelet: A `str` or a `pywt.Wavelet`_, or a `list` thereof. When passed a
      `list`, different wavelets are applied along each axis in `axes`.
    mode: A `str`. The padding or signal extension mode. Must be one of the
      values supported by `tfmri.signal.wavedec`. Defaults to `'symmetric'`.
    level: An `int` >= 0. The decomposition level. If `None` (default),
      the maximum useful level of decomposition will be used (see
      `tfmri.signal.max_wavelet_level`).
    axes: A `list` of `int`. The axes over which the DWT is computed. Axes refer
      only to domain dimensions without regard for the batch dimensions.
      Defaults to `None` (all domain dimensions).
    dtype: A `tf.dtypes.DType`. The data type for this operator. Defaults to
      `float32`.
    name: A `str`. A name for this operator.
  """
  def __init__(self,
               domain_shape,
               wavelet,
               mode='symmetric',
               level=None,
               axes=None,
               dtype=tf.dtypes.float32,
               name="LinearOperatorWavelet"):
    # Set parameters.
    parameters = dict(
        domain_shape=domain_shape,
        wavelet=wavelet,
        mode=mode,
        level=level,
        axes=axes,
        dtype=dtype,
        name=name
    )

    # Get the static and dynamic shapes and save them for later use.
    self._domain_shape_static, self._domain_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))
    # At the moment, the wavelet implementation relies on shapes being
    # statically known.
    if not self._domain_shape_static.is_fully_defined():
      raise ValueError(f"static `domain_shape` must be fully defined, "
                       f"but got {self._domain_shape_static}")
    static_rank = self._domain_shape_static.rank

    # Set arguments.
    self.wavelet = wavelet
    self.mode = mode
    self.level = level
    self.axes = check_util.validate_static_axes(axes,
                                                rank=static_rank,
                                                min_length=1,
                                                canonicalize="negative",
                                                must_be_unique=True,
                                                scalar_to_list=True,
                                                none_means_all=True)

    # Compute the coefficient slices needed for adjoint (wavelet
    # reconstruction).
    x = tf.ensure_shape(tf.zeros(self._domain_shape_dynamic, dtype=dtype),
                        self._domain_shape_static)
    x = wavelet_ops.wavedec(x, wavelet=self.wavelet, mode=self.mode,
                            level=self.level, axes=self.axes)
    y, self._coeff_slices = wavelet_ops.coeffs_to_tensor(x, axes=self.axes)

    # Get the range shape.
    self._range_shape_static = y.shape
    self._range_shape_dynamic = tf.shape(y)

    # Call base class.
    super().__init__(dtype,
                     is_non_singular=None,
                     is_self_adjoint=None,
                     is_positive_definite=None,
                     is_square=None,
                     name=name,
                     parameters=parameters)

  def _transform(self, x, adjoint=False):
    # While `wavedec` and `waverec` can transform only a subset of axes (and
    # thus theoretically support batches), there is a caveat due to the
    # `coeff_slices` object required by `waverec`. This object contains
    # information relevant to a specific batch shape. While we could recompute
    # this object for every input batch shape, it is easier to just process
    # each batch independently.
    if x.shape.rank is not None and self._domain_shape_static.rank is not None:
      # Rank of input and this operator are known statically, so we can infer
      # the number of batch dimensions statically too.
      batch_dims = x.shape.rank - self._domain_shape_static.rank
    else:
      # We need to obtain the number of batch dimensions dynamically.
      batch_dims = tf.rank(x) - tf.shape(self._domain_shape_dynamic)[0]
    # Transform each batch.
    x = array_ops.map_fn(
        functools.partial(self._transform_batch, adjoint=adjoint),
        x, batch_dims=batch_dims)
    return x

  def _transform_batch(self, x, adjoint=False):
    if adjoint:
      x = wavelet_ops.tensor_to_coeffs(x, self._coeff_slices)
      x = wavelet_ops.waverec(x, wavelet=self.wavelet, mode=self.mode,
                              axes=self.axes)
    else:
      x = wavelet_ops.wavedec(x, wavelet=self.wavelet, mode=self.mode,
                              level=self.level, axes=self.axes)
      x, _ = wavelet_ops.coeffs_to_tensor(x, axes=self.axes)
    return x

  def _domain_shape(self):
    return self._domain_shape_static

  def _range_shape(self):
    return self._range_shape_static

  def _domain_shape_tensor(self):
    return self._domain_shape_dynamic

  def _range_shape_tensor(self):
    return self._range_shape_dynamic
