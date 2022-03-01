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
"""Coil array operations.

This module contains functions to operate with MR coil arrays, such as
estimating coil sensitivities and combining multi-coil images.
"""

import abc
import collections
import functools

import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.util import check_util


def estimate_coil_sensitivities(input_,
                                coil_axis=-1,
                                method='walsh',
                                **kwargs):
  """Estimate coil sensitivity maps.

  This method supports 2D and 3D inputs.

  Args:
    input_: A `Tensor`. Must have type `complex64` or `complex128`. Must have
      shape `[height, width, coils]` for 2D inputs, or `[depth, height,
      width, coils]` for 3D inputs. Alternatively, this function accepts a
      transposed array by setting the `coil_axis` argument accordingly. Inputs
      should be images if `method` is `'walsh'` or `'inati'`, and k-space data
      if `method` is `'espirit'`.
    coil_axis: An `int`. Defaults to -1.
    method: A `string`. The coil sensitivity estimation algorithm. Must be one
      of: `{'walsh', 'inati', 'espirit'}`. Defaults to `'walsh'`.
    **kwargs: Additional keyword arguments for the coil sensitivity estimation
      algorithm. See Notes.

  Returns:
    A `Tensor`. Has the same type as `input_`. Has shape
    `input_.shape + [num_maps]` if `method` is `'espirit'`, or shape
    `input_.shape` otherwise.

  Notes:

    This function accepts the following method-specific keyword arguments:

    * For `method="walsh"`:

      * **filter_size**: An `int`. The size of the smoothing filter.

    * For `method="inati"`:

      * **filter_size**: An `int`. The size of the smoothing filter.
      * **max_iter**: An `int`. The maximum number of iterations.
      * **tol**: A `float`. The convergence tolerance.

    * For `method="espirit"`:

      * **calib_size**: An `int` or a list of `ints`. The size of the
        calibration region. If `None`, this is set to `input_.shape[:-1]` (ie,
        use full input for calibration). Defaults to 24.
      * **kernel_size**: An `int` or a list of `ints`. The kernel size. Defaults
        to 6.
      * **num_maps**: An `int`. The number of output maps. Defaults to 2.
      * **null_threshold**: A `float`. The threshold used to determine the size
        of the null-space. Defaults to 0.02.
      * **eigen_threshold**: A `float`. The threshold used to determine the
        locations where coil sensitivity maps should be masked out. Defaults
        to 0.95.
      * **image_shape**: A `tf.TensorShape` or a list of `ints`. The shape of
        the output maps. If `None`, this is set to `input_.shape`. Defaults to
        `None`.

  References:
    .. [1] Walsh, D.O., Gmitro, A.F. and Marcellin, M.W. (2000), Adaptive
      reconstruction of phased array MR imagery. Magn. Reson. Med., 43:
      682-690. https://doi.org/10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G

    .. [2] Inati, S.J., Hansen, M.S. and Kellman, P. (2014). A fast optimal
      method for coil sensitivity estimation and adaptive coil combination for
      complex images. Proceedings of the 2014 Joint Annual Meeting
      ISMRM-ESMRMB.

    .. [3] Uecker, M., Lai, P., Murphy, M.J., Virtue, P., Elad, M., Pauly, J.M.,
      Vasanawala, S.S. and Lustig, M. (2014), ESPIRiT—an eigenvalue approach
      to autocalibrating parallel MRI: Where SENSE meets GRAPPA. Magn. Reson.
      Med., 71: 990-1001. https://doi.org/10.1002/mrm.24751
  """
  # pylint: disable=missing-raises-doc
  input_ = tf.convert_to_tensor(input_)
  tf.debugging.assert_rank_at_least(input_, 2, message=(
    f"Argument `input_` must have rank of at least 2, but got shape: "
    f"{input_.shape}"))
  coil_axis = check_util.validate_type(coil_axis, int, name='coil_axis')
  method = check_util.validate_enum(
    method, {'walsh', 'inati', 'espirit'}, name='method')

  # Move coil axis to innermost dimension if not already there.
  if coil_axis != -1:
    rank = input_.shape.rank
    canonical_coil_axis = coil_axis + rank if coil_axis < 0 else coil_axis
    perm = (
      [ax for ax in range(rank) if not ax == canonical_coil_axis] +
      [canonical_coil_axis])
    input_ = tf.transpose(input_, perm)

  if method == 'walsh':
    maps = _estimate_coil_sensitivities_walsh(input_, **kwargs)
  elif method == 'inati':
    maps = _estimate_coil_sensitivities_inati(input_, **kwargs)
  elif method == 'espirit':
    maps = _estimate_coil_sensitivities_espirit(input_, **kwargs)
  else:
    raise RuntimeError("This should never happen.")

  # If necessary, move coil axis back to its original location.
  if coil_axis != -1:
    inv_perm = tf.math.invert_permutation(perm)
    if method == 'espirit':
      # When using ESPIRiT method, output has an additional `maps` dimension.
      inv_perm = tf.concat([inv_perm, [tf.shape(inv_perm)[0]]], 0)
    maps = tf.transpose(maps, inv_perm)

  return maps


def combine_coils(images, maps=None, coil_axis=-1, keepdims=False):
  """Sum of squares or adaptive coil combination.

  Args:
    images: A `Tensor`. The input images.
    maps: A `Tensor`. The coil sensitivity maps. This argument is optional.
      If `maps` is provided, it must have the same shape and type as
      `images`. In this case an adaptive coil combination is performed using
      the specified maps. If `maps` is `None`, a simple estimate of `maps`
      is used (ie, images are combined using the sum of squares method).
    coil_axis: An `int`. The coil axis. Defaults to -1.
    keepdims: A `bool`. If `True`, retains the coil dimension with size 1.

  Returns:
    A `Tensor`. The combined images.

  References:
    .. [1] Roemer, P.B., Edelstein, W.A., Hayes, C.E., Souza, S.P. and
      Mueller, O.M. (1990), The NMR phased array. Magn Reson Med, 16:
      192-225. https://doi.org/10.1002/mrm.1910160203

    .. [2] Bydder, M., Larkman, D. and Hajnal, J. (2002), Combination of signals
      from array coils using image-based estimation of coil sensitivity
      profiles. Magn. Reson. Med., 47: 539-548.
      https://doi.org/10.1002/mrm.10092
  """
  images = tf.convert_to_tensor(images)
  if maps is not None:
    maps = tf.convert_to_tensor(maps)

  if maps is None:
    combined = tf.math.sqrt(
        tf.math.reduce_sum(images * tf.math.conj(images),
                           axis=coil_axis, keepdims=keepdims))

  else:
    combined = tf.math.divide_no_nan(
        tf.math.reduce_sum(images * tf.math.conj(maps),
                           axis=coil_axis, keepdims=keepdims),
        tf.math.reduce_sum(maps * tf.math.conj(maps),
                           axis=coil_axis, keepdims=keepdims))

  return combined


def _estimate_coil_sensitivities_walsh(images, filter_size=5):
  """Estimate coil sensitivity maps using Walsh's method.

  For the parameters, see `estimate_coil_sensitivities`.
  """
  rank = images.shape.rank - 1
  image_shape = tf.shape(images)[:-1]
  num_coils = tf.shape(images)[-1]

  filter_size = check_util.validate_list(
      filter_size, element_type=int, length=rank, name='filter_size')

  # Flatten all spatial dimensions into a single axis, so `images` has shape
  # `[num_pixels, num_coils]`.
  flat_images = tf.reshape(images, [-1, num_coils])

  # Compute covariance matrix for each pixel; with shape
  # `[num_pixels, num_coils, num_coils]`.
  correlation_matrix = tf.math.multiply(
      tf.reshape(flat_images, [-1, num_coils, 1]),
      tf.math.conj(tf.reshape(flat_images, [-1, 1, num_coils])))

  # Smooth the covariance tensor along the spatial dimensions.
  correlation_matrix = tf.reshape(
      correlation_matrix, tf.concat([image_shape, [-1]], 0))
  correlation_matrix = _apply_uniform_filter(correlation_matrix, filter_size)
  correlation_matrix = tf.reshape(correlation_matrix, [-1] + [num_coils] * 2)

  # Get sensitivity maps as the dominant eigenvector.
  _, eigenvectors = tf.linalg.eig(correlation_matrix) # pylint: disable=no-value-for-parameter
  maps = eigenvectors[..., -1]

  # Restore spatial axes.
  maps = tf.reshape(maps, tf.concat([image_shape, [num_coils]], 0))

  return maps


def _estimate_coil_sensitivities_inati(images,
                                       filter_size=5,
                                       max_iter=5,
                                       tol=1e-3):
  """Estimate coil sensitivity maps using Inati's fast method.

  For the parameters, see `estimate_coil_sensitivities`.
  """
  rank = images.shape.rank - 1
  spatial_axes = list(range(rank))
  coil_axis = -1

  # Validate inputs.
  filter_size = check_util.validate_list(
    filter_size, element_type=int, length=rank, name='filter_size')
  max_iter = check_util.validate_type(max_iter, int, name='max_iter')
  tol = check_util.validate_type(tol, float, name='tol')

  d_sum = tf.math.reduce_sum(images, axis=spatial_axes, keepdims=True)
  d_sum /= tf.norm(d_sum, axis=coil_axis, keepdims=True)

  r = tf.math.reduce_sum(
    tf.math.conj(d_sum) * images, axis=coil_axis, keepdims=True)

  eps = tf.cast(
    tnp.finfo(images.dtype).eps * tf.math.reduce_mean(tf.math.abs(images)),
    images.dtype)

  State = collections.namedtuple('State', ['i', 'maps', 'r', 'd'])

  def _cond(i, state):
    return tf.math.logical_and(i < max_iter, state.d >= tol)

  def _body(i, state):
    prev_r = state.r
    r = state.r

    r = tf.math.conj(r)

    maps = images * r
    smooth_maps = _apply_uniform_filter(maps, filter_size)
    d = smooth_maps * tf.math.conj(smooth_maps)

    # Sum over coils.
    r = tf.math.reduce_sum(d, axis=coil_axis, keepdims=True)

    r = tf.math.sqrt(r)
    r = tf.math.reciprocal(r + eps)

    maps = smooth_maps * r

    d = images * tf.math.conj(maps)
    r = tf.math.reduce_sum(d, axis=coil_axis, keepdims=True)

    d = maps * r

    d_sum = tf.math.reduce_sum(d, axis=spatial_axes, keepdims=True)
    d_sum /= tf.norm(d_sum, axis=coil_axis, keepdims=True)

    im_t = tf.math.reduce_sum(
      tf.math.conj(d_sum) * maps, axis=coil_axis, keepdims=True)
    im_t /= (tf.cast(tf.math.abs(im_t), images.dtype) + eps)
    r *= im_t
    im_t = tf.math.conj(im_t)
    maps = maps * im_t

    diff_r = r - prev_r
    d = tf.math.abs(tf.norm(diff_r) / tf.norm(r))

    return i + 1, State(i=i + 1, maps=maps, r=r, d=d)

  i = tf.constant(0, dtype=tf.int32)
  state = State(i=i,
                maps=tf.zeros_like(images),
                r=r,
                d=tf.constant(1.0, dtype=images.dtype.real_dtype))
  [i, state] = tf.while_loop(_cond, _body, [i, state])

  return tf.reshape(state.maps, images.shape)


def _estimate_coil_sensitivities_espirit(kspace,
                                         calib_size=24,
                                         kernel_size=6,
                                         num_maps=2,
                                         null_threshold=0.02,
                                         eigen_threshold=0.95,
                                         image_shape=None):
  """Estimate coil sensitivity maps using the ESPIRiT method.

  For the parameters, see `estimate_coil_sensitivities`.
  """
  kspace = tf.convert_to_tensor(kspace)
  rank = kspace.shape.rank - 1
  spatial_axes = list(range(rank))
  num_coils = tf.shape(kspace)[-1]
  if image_shape is None:
    image_shape = kspace.shape[:-1]
  if calib_size is None:
    calib_size = image_shape.as_list()

  calib_size = check_util.validate_list(
    calib_size, element_type=int, length=rank, name='calib_size')
  kernel_size = check_util.validate_list(
    kernel_size, element_type=int, length=rank, name='kernel_size')

  with tf.control_dependencies([
      tf.debugging.assert_greater(calib_size, kernel_size, message=(
          f"`calib_size` must be greater than `kernel_size`, but got "
          f"{calib_size} and {kernel_size}"))]):
    kspace = tf.identity(kspace)

  # Get calibration region.
  calib = image_ops.central_crop(kspace, calib_size + [-1])

  # Construct the calibration block Hankel matrix.
  conv_size = [cs - ks + 1 for cs, ks in zip(calib_size, kernel_size)]
  calib_matrix = tf.zeros([_prod(conv_size), _prod(kernel_size) * num_coils],
                          dtype=calib.dtype)
  idx = 0
  for nd_inds in np.ndindex(*conv_size):
    slices = [slice(ii, ii + ks) for ii, ks in zip(nd_inds, kernel_size)]
    calib_matrix = tf.tensor_scatter_nd_update(
      calib_matrix, [[idx]], tf.reshape(calib[slices], [1, -1]))
    idx += 1

  # Compute SVD decomposition, threshold singular values and reshape V to create
  # k-space kernel matrix.
  s, _, v = tf.linalg.svd(calib_matrix, full_matrices=True)
  num_values = tf.math.count_nonzero(s >= s[0] * null_threshold)
  v = v[:, :num_values]
  kernel = tf.reshape(v, kernel_size + [num_coils, -1])

  # Rotate kernel to order by maximum variance.
  perm = list(range(kernel.shape.rank))
  perm[-2], perm[-1] = perm[-1], perm[-2]
  kernel = tf.transpose(kernel, perm)
  kernel = tf.reshape(kernel, [-1, num_coils])
  _, _, rot_matrix = tf.linalg.svd(kernel, full_matrices=False)
  kernel = tf.linalg.matmul(kernel, rot_matrix)
  kernel = tf.reshape(kernel, kernel_size + [-1, num_coils])
  kernel = tf.transpose(kernel, perm)

  # Compute inverse FFT of k-space kernel.
  kernel = tf.reverse(kernel, spatial_axes)
  kernel = tf.math.conj(kernel)

  kernel_image = fft_ops.fftn(kernel,
                              shape=image_shape,
                              axes=list(range(rank)),
                              shift=True)

  kernel_image /= tf.cast(tf.sqrt(tf.cast(tf.math.reduce_prod(kernel_size),
                                          kernel_image.dtype.real_dtype)),
                          kernel_image.dtype)

  values, maps, _ = tf.linalg.svd(kernel_image, full_matrices=False)

  # Apply phase modulation.
  maps *= tf.math.exp(tf.complex(tf.constant(0.0, dtype=maps.dtype.real_dtype),
                                 -tf.math.angle(maps[..., 0:1, :])))

  # Undo rotation.
  maps = tf.linalg.matmul(rot_matrix, maps)

  # Keep only the requested number of maps.
  values = values[..., :num_maps]
  maps = maps[..., :num_maps]

  # Apply thresholding.
  mask = tf.expand_dims(values >= eigen_threshold, -2)
  maps *= tf.cast(mask, maps.dtype)

  # If possible, set static number of maps.
  if isinstance(num_maps, int):
    maps_shape = maps.shape.as_list()
    maps_shape[-1] = num_maps
    maps = tf.ensure_shape(maps, maps_shape)

  return maps


def compress_coils(kspace,
                   coil_axis=-1,
                   out_coils=None,
                   method='svd',
                   **kwargs):
  """Coil compression gateway.

  This function estimates a coil compression matrix and uses it to compress
  `kspace`. If you would like to reuse a coil compression matrix or need to
  calibrate the compression using different data, use `tfmri.SVDCoilCompressor`.

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
  # pylint: disable=missing-raises-doc
  kspace = tf.convert_to_tensor(kspace)
  tf.debugging.assert_rank_at_least(kspace, 2, message=(
      f"Argument `kspace` must have rank of at least 2, but got shape: "
      f"{kspace.shape}"))
  coil_axis = check_util.validate_type(coil_axis, int, name='coil_axis')
  method = check_util.validate_enum(
      method, {'svd', 'geometric', 'espirit'}, name='method')

  # Calculate the compression matrix, unless one was already provided.
  if method == 'svd':
    return SVDCoilCompressor(coil_axis=coil_axis,
                             out_coils=out_coils,
                             **kwargs).fit_transform(kspace)

  raise NotImplementedError(f"Method {method} not implemented.")


class _CoilCompressor():
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


class SVDCoilCompressor(_CoilCompressor):
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
      The fitted `SVDCoilCompressor` object.
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


def _apply_uniform_filter(tensor, size=5):
  """Apply a uniform filter.

  Args:
    tensor: A `Tensor`. Must have shape `spatial_shape + [channels]`.
    size: An `int`. The size of the filter. Defaults to 5.

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  rank = tensor.shape.rank - 1

  # Compute filters.
  if isinstance(size, int):
    size = [size] * rank
  filters_shape = size + [1, 1]
  filters = tf.ones(filters_shape, dtype=tensor.dtype.real_dtype)
  filters /= _prod(size)

  # Select appropriate convolution function.
  conv_nd = {
    1: tf.nn.conv1d,
    2: tf.nn.conv2d,
    3: tf.nn.conv3d}[rank]

  # Move channels dimension to batch dimension.
  tensor = tf.transpose(tensor)

  # Add a channels dimension, as required by `tf.nn.conv*` functions.
  tensor = tf.expand_dims(tensor, -1)

  if tensor.dtype.is_complex:
    # For complex input, we filter the real and imaginary parts separately.
    tensor_real = tf.math.real(tensor)
    tensor_imag = tf.math.imag(tensor)

    output_real = conv_nd(tensor_real, filters, [1] * (rank + 2), 'SAME')
    output_imag = conv_nd(tensor_imag, filters, [1] * (rank + 2), 'SAME')

    output = tf.dtypes.complex(output_real, output_imag)
  else:
    output = conv_nd(tensor, filters, [1] * (rank + 2), 'SAME')

  # Remove channels dimension.
  output = output[..., 0]

  # Move channels dimension back to last dimension.
  output = tf.transpose(output)

  return output


_prod = lambda iterable: functools.reduce(lambda x, y: x * y, iterable)
