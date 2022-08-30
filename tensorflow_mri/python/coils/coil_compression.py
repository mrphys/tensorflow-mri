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
"""Coil sensitivity estimation."""

import abc

import tensorflow as tf

from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util


@api_util.export("coils.compress_coils")
def compress_coils(kspace,
                   coil_axis=-1,
                   out_coils=None,
                   method='svd',
                   **kwargs):
  """Coil compression gateway.

  This function estimates a coil compression matrix and uses it to compress
  `kspace`. If you would like to reuse a coil compression matrix or need to
  calibrate the compression using different data, use
  `tfmri.coils.get_coil_compressor`.

  This function supports the following coil compression methods:

  * **SVD**: Based on direct singular-value decomposition (SVD) of *k*-space
    data [1]_. This coil compression method supports Cartesian and
    non-Cartesian data. This method is resilient to noise, but does not
    achieve optimal compression if there are fully-sampled dimensions.

  ..  * **Geometric**: Performs local compression along fully-sampled dimensions
  ..    to improve compression. This method only supports Cartesian data. This
  ..    method can suffer from low SNR in sections of k-space.
  ..  * **ESPIRiT**: Performs local compression along fully-sampled dimensions
  ..    and is robust to noise. This method only supports Cartesian data.

  Args:
    kspace: A `Tensor`. The multi-coil *k*-space data. Must have type
      `complex64` or `complex128`. Must have shape `[..., Cin]`, where `...` are
      the encoding dimensions and `Cin` is the number of coils. Alternatively,
      the position of the coil axis may be different as long as the `coil_axis`
      argument is set accordingly. If `method` is `"svd"`, `kspace` can be
      Cartesian or non-Cartesian. If `method` is `"geometric"` or `"espirit"`,
      `kspace` must be Cartesian.
    coil_axis: An `int`. Defaults to -1.
    out_coils: An `int`. The desired number of virtual output coils.
    method: A `string`. The coil compression algorithm. Must be `"svd"`.
    **kwargs: Additional method-specific keyword arguments to be passed to the
      coil compressor.

  Returns:
    A `Tensor` containing the compressed *k*-space data. Has shape
    `[..., Cout]`, where `Cout` is determined based on `out_coils` or
    other inputs and `...` are the unmodified encoding dimensions.

  References:
    .. [1] Huang, F., Vijayakumar, S., Li, Y., Hertel, S. and Duensing, G.R.
      (2008). A software channel compression technique for faster reconstruction
      with many channels. Magn Reson Imaging, 26(1): 133-141.
    .. [2] Zhang, T., Pauly, J.M., Vasanawala, S.S. and Lustig, M. (2013), Coil
      compression for accelerated imaging with Cartesian sampling. Magn
      Reson Med, 69: 571-582. https://doi.org/10.1002/mrm.24267
    .. [3] Bahri, D., Uecker, M., & Lustig, M. (2013). ESPIRIT-based coil
      compression for cartesian sampling. In Proceedings of the 21st
      Annual Meeting of ISMRM, Salt Lake City, Utah, USA (Vol. 47).
  """
  return make_coil_compressor(method,
                              coil_axis=coil_axis,
                              out_coils=out_coils,
                              **kwargs).fit_transform(kspace)


class CoilCompressor():
  """Base class for coil compressors.

  Args:
    coil_axis: An `int`. The axis of the coil dimension.
    out_coils: An `int`. The desired number of virtual output coils.
  """
  def __init__(self, coil_axis=-1, out_coils=None):
    self._coil_axis = coil_axis
    self._out_coils = out_coils

  @abc.abstractmethod
  def fit(self, kspace):
    pass

  @abc.abstractmethod
  def transform(self, kspace):
    pass

  def fit_transform(self, kspace):
    return self.fit(kspace).transform(kspace)


@api_util.export("coils.CoilCompressorSVD")
class CoilCompressorSVD(CoilCompressor):
  """SVD-based coil compression.

  This class implements the SVD-based coil compression method [1]_.

  Use this class to compress multi-coil *k*-space data. The method `fit` must
  be used first to calculate the coil compression matrix. The method `transform`
  can then be used to compress *k*-space data. If the data to be used for
  fitting is the same data to be transformed, you can also use the method
  `fit_transform` to fit and transform the data in one step.

  Args:
    coil_axis: An `int`. Defaults to -1.
    out_coils: An `int`. The desired number of virtual output coils. Cannot be
      used together with `variance_ratio`.
    variance_ratio: A `float` between 0.0 and 1.0. The percentage of total
      variance to be retained. The number of virtual coils is automatically
      selected to retain at least this percentage of variance. Cannot be used
      together with `out_coils`.

  References:
    .. [1] Huang, F., Vijayakumar, S., Li, Y., Hertel, S. and Duensing, G.R.
      (2008). A software channel compression technique for faster reconstruction
      with many channels. Magn Reson Imaging, 26(1): 133-141.
  """
  def __init__(self, coil_axis=-1, out_coils=None, variance_ratio=None):
    if out_coils is not None and variance_ratio is not None:
      raise ValueError("Cannot specify both `out_coils` and `variance_ratio`.")
    super().__init__(coil_axis=coil_axis, out_coils=out_coils)
    self._variance_ratio = variance_ratio
    self._singular_values = None
    self._explained_variance = None
    self._explained_variance_ratio = None

  def fit(self, kspace):
    """Fits the coil compression matrix.

    Args:
      kspace: A `Tensor`. The multi-coil *k*-space data. Must have type
        `complex64` or `complex128`.

    Returns:
      The fitted `CoilCompressorSVD` object.
    """
    kspace = tf.convert_to_tensor(kspace)

    # Move coil axis to innermost dimension if not already there.
    kspace, _ = self._permute_coil_axis(kspace)

    # Flatten the encoding dimensions.
    num_coils = tf.shape(kspace)[-1]
    kspace = tf.reshape(kspace, [-1, num_coils])
    num_samples = tf.shape(kspace)[0]

    # Compute singular-value decomposition.
    s, u, v = tf.linalg.svd(kspace)

    # Compresion matrix.
    self._matrix = tf.cond(num_samples > num_coils, lambda: v, lambda: u)

    # Get variance.
    self._singular_values = s
    self._explained_variance = s ** 2 / tf.cast(num_samples - 1, s.dtype)
    total_variance = tf.math.reduce_sum(self._explained_variance)
    self._explained_variance_ratio = self._explained_variance / total_variance

    # Get output coils from variance ratio.
    if self._variance_ratio is not None:
      cum_variance = tf.math.cumsum(self._explained_variance_ratio, axis=0)
      self._out_coils = tf.math.count_nonzero(
          cum_variance <= self._variance_ratio)

    # Remove unnecessary virtual coils.
    if self._out_coils is not None:
      self._matrix = self._matrix[:, :self._out_coils]

    # If possible, set static number of output coils.
    if isinstance(self._out_coils, int):
      self._matrix = tf.ensure_shape(self._matrix, [None, self._out_coils])

    return self

  def transform(self, kspace):
    """Applies the coil compression matrix to the input *k*-space.

    Args:
      kspace: A `Tensor`. The multi-coil *k*-space data. Must have type
        `complex64` or `complex128`.

    Returns:
      The transformed k-space.
    """
    kspace = tf.convert_to_tensor(kspace)
    kspace, inv_perm = self._permute_coil_axis(kspace)

    # Some info.
    encoding_dimensions = tf.shape(kspace)[:-1]
    num_coils = tf.shape(kspace)[-1]
    out_coils = tf.shape(self._matrix)[-1]

    # Flatten the encoding dimensions.
    kspace = tf.reshape(kspace, [-1, num_coils])

    # Apply compression.
    kspace = tf.linalg.matmul(kspace, self._matrix)

    # Restore data shape.
    kspace = tf.reshape(
        kspace,
        tf.concat([encoding_dimensions, [out_coils]], 0))

    if inv_perm is not None:
      kspace = tf.transpose(kspace, inv_perm)

    return kspace

  def _permute_coil_axis(self, kspace):
    """Permutes the coil axis to the last dimension.

    Args:
      kspace: A `Tensor`. The multi-coil *k*-space data.

    Returns:
      A tuple of the permuted k-space and the inverse permutation.
    """
    if self._coil_axis != -1:
      rank = kspace.shape.rank # Rank must be known statically.
      canonical_coil_axis = (
          self._coil_axis + rank if self._coil_axis < 0 else self._coil_axis)
      perm = (
          [ax for ax in range(rank) if not ax == canonical_coil_axis] +
          [canonical_coil_axis])
      kspace = tf.transpose(kspace, perm)
      inv_perm = tf.math.invert_permutation(perm)
      return kspace, inv_perm
    return kspace, None

  @property
  def singular_values(self):
    """The singular values associated with each virtual coil."""
    return self._singular_values

  @property
  def explained_variance(self):
    """The variance explained by each virtual coil."""
    return self._explained_variance

  @property
  def explained_variance_ratio(self):
    """The percentage of variance explained by each virtual coil."""
    return self._explained_variance_ratio


def make_coil_compressor(method, **kwargs):
  """Creates a coil compressor based on the specified method.

  Args:
    method: A `string`. The coil compression algorithm. Must be `"svd"`.
    **kwargs: Additional method-specific keyword arguments to be passed to the
      coil compressor.

  Returns:
    A `CoilCompressor` object.
  """
  method = check_util.validate_enum(
      method, {'svd', 'geometric', 'espirit'}, name='method')
  if method == 'svd':
    return CoilCompressorSVD(**kwargs)
  raise NotImplementedError(f"Method {method} not implemented.")


# def compress_coils_with_calibration_data(
#     kspace,
#     operator,
#     calib_data=None,
#     calib_window=None,
#     method='svd',
#     **kwargs):
#   # For convenience.
#   rank = operator.rank

#   if calib_data is None:
#     # Calibration data was not provided. Get calibration data by low-pass
#     # filtering the input k-space.
#     calib_data = signal_ops.filter_kspace(
#         kspace,
#         trajectory=operator.trajectory,
#         filter_fn=calib_window,
#         filter_rank=rank,
#         separable=True)

#   # Reshape to single batch dimension.
#   coil_axis = -2 if operator.is_non_cartesian else -(rank + 1)
#   batch_shape_static = calib_data.shape[:coil_axis]
#   batch_shape = tf.shape(calib_data)[:coil_axis]
#   calib_shape = tf.shape(calib_data)[coil_axis:]
#   calib_data = tf.reshape(calib_data, tf.concat([[-1], calib_shape], 0))
#   kspace_shape = tf.shape(kspace)[coil_axis:]
#   kspace = tf.reshape(kspace, tf.concat([[-1], kspace_shape], 0))

#   # Apply compression for each element in batch.
#   def compress_coils_fn(inputs):
#     ksp, cal = inputs
#     return get_coil_compressor(method,
#                                coil_axis=coil_axis,
#                                **kwargs).fit(cal).transform(ksp)
#   output_shape = [kwargs.get('out_coils')] + kspace.shape[2:].as_list()
#   fn_output_signature = tf.TensorSpec(shape=output_shape, dtype=kspace.dtype)
#   kspace = tf.map_fn(compress_coils_fn, (kspace, calib_data),
#                      fn_output_signature=fn_output_signature)

#   # Restore batch shape.
#   output_shape = tf.shape(kspace)[1:]
#   output_shape_static = kspace.shape[1:]
#   kspace = tf.reshape(kspace,
#       tf.concat([batch_shape, output_shape], 0))
#   kspace = tf.ensure_shape(
#       kspace, batch_shape_static.concatenate(output_shape_static))

#   return kspace
