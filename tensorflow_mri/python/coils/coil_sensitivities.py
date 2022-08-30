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

import collections
import functools

import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.recon import recon_adjoint
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util


@api_util.export("coils.estimate_sensitivities")
def estimate(input_, coil_axis=-1, method='walsh', **kwargs):
  """Estimates coil sensitivity maps.

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

    - For `method="walsh"`:

      - **filter_size**: An `int`. The size of the smoothing filter.

    - For `method="inati"`:

      - **filter_size**: An `int`. The size of the smoothing filter.
      - **max_iter**: An `int`. The maximum number of iterations.
      - **tol**: A `float`. The convergence tolerance.

    - For `method="espirit"`:

      - **calib_size**: An `int` or a list of `ints`. The size of the
        calibration region. If `None`, this is set to `input_.shape[:-1]` (ie,
        use full input for calibration). Defaults to 24.
      - **kernel_size**: An `int` or a list of `ints`. The kernel size. Defaults
        to 6.
      - **num_maps**: An `int`. The number of output maps. Defaults to 2.
      - **null_threshold**: A `float`. The threshold used to determine the size
        of the null-space. Defaults to 0.02.
      - **eigen_threshold**: A `float`. The threshold used to determine the
        locations where coil sensitivity maps should be masked out. Defaults
        to 0.95.
      - **image_shape**: A `tf.TensorShape` or a list of `ints`. The shape of
        the output maps. If `None`, this is set to `input_.shape`. Defaults to
        `None`.

  References:
    1. Walsh, D.O., Gmitro, A.F. and Marcellin, M.W. (2000), Adaptive
       reconstruction of phased array MR imagery. Magn. Reson. Med., 43:
       682-690. https://doi.org/10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G
    2. Inati, S.J., Hansen, M.S. and Kellman, P. (2014). A fast optimal
       method for coil sensitivity estimation and adaptive coil combination for
       complex images. Proceedings of the 2014 Joint Annual Meeting
       ISMRM-ESMRMB.
    3. Uecker, M., Lai, P., Murphy, M.J., Virtue, P., Elad, M., Pauly, J.M.,
       Vasanawala, S.S. and Lustig, M. (2014), ESPIRiT—an eigenvalue approach
       to autocalibrating parallel MRI: Where SENSE meets GRAPPA. Magn. Reson.
       Med., 71: 990-1001. https://doi.org/10.1002/mrm.24751
  """
  # pylint: disable=missing-raises-doc
  with tf.name_scope(kwargs.get("name", "estimate_sensitivities")):
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
      maps = _estimate_walsh(input_, **kwargs)
    elif method == 'inati':
      maps = _estimate_inati(input_, **kwargs)
    elif method == 'espirit':
      maps = _estimate_espirit(input_, **kwargs)
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


def _estimate_walsh(images, filter_size=5):
  """Estimate coil sensitivity maps using Walsh's method.

  For the parameters, see `estimate`.
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


def _estimate_inati(images,
                    filter_size=5,
                    max_iter=5,
                    tol=1e-3):
  """Estimate coil sensitivity maps using Inati's fast method.

  For the parameters, see `estimate`.
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


def _estimate_espirit(kspace,
                      calib_size=24,
                      kernel_size=6,
                      num_maps=2,
                      null_threshold=0.02,
                      eigen_threshold=0.95,
                      image_shape=None):
  """Estimate coil sensitivity maps using the ESPIRiT method.

  For the parameters, see `estimate`.
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
  calib = array_ops.central_crop(kspace, calib_size + [-1])

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


@api_util.export("coils.estimate_sensitivities_universal")
def estimate_universal(
    meas_data,
    operator,
    calib_data=None,
    calib_fn=None,
    method='walsh',
    **kwargs):
  """Estimates coil sensitivities (universal).

  This function is designed to standardize the computation of coil
  sensitivities in different contexts. The `meas_data` argument can accept
  arbitrary measurement data (e.g., N-dimensional, Cartesian/non-Cartesian
  *k*-space tensors). In addition, this function expects a linear `operator`
  which describes the action of the measurement system (e.g., the MR imaging
  experiment).

  This function also accepts an optional `calib_data` tensor or an optional
  `calib_fn` function, in case the calibration should be performed with data
  other than `meas_data`. `calib_data` may be used to provide the calibration
  data directly, whereas `calib_fn` may be used to specify the rules to extract
  it from `meas_data`.

  ```{note}
  This function is part of the
  [universal family](https://mrphys.github.io/tensorflow-mri/guide/universal/)
  of operators designed to work flexibly with any linear system.
  ```

  Example:
    >>> # Create an example image.
    >>> image_shape = [256, 256]
    >>> image = tfmri.image.phantom(shape=image_shape,
    ...                             num_coils=8,
    ...                             dtype=tf.complex64)
    >>> kspace = tfmri.signal.fft(image, axes=[-2, -1], shift=True)
    >>> # Create an acceleration mask with 4x undersampling along the last axis
    >>> # and 24 calibration lines.
    >>> mask = tfmri.sampling.accel_mask(shape=image_shape,
    ...                                  acceleration=[1, 4],
    ...                                  center_size=[256, 24])
    >>> # Create a linear operator describing a basic MR experiment with
    >>> # Cartesian undersampling. This operator maps an image to the
    >>> # corresponding *k*-space data (by performing an FFT and masking the
    >>> # measured values).
    >>> linop_mri = tfmri.linalg.LinearOperatorMRI(
    ...     image_shape=image_shape, mask=mask)
    >>> # Generate *k*-space data using the system operator.
    >>> kspace = linop_mri.transform(image)
    >>> # To compute the sensitivity maps, we typically want to use only the
    >>> # fully-sampled central region of *k*-space. Let's create a mask that
    >>> # retrieves only the 24 calibration lines.
    >>> calib_mask = tfmri.sampling.center_mask(shape=image_shape,
    ...                                         center_size=[256, 24])
    >>> # We can create a function that extracts the calibration data from
    >>> # an arbitrary *k*-space by applying the calibration mask below.
    >>> def calib_fn(meas_data, operator):
    ...   # Returns `meas_data` where `calib_mask` is `True`, 0 otherwise.
    ...   return tf.where(calib_mask, meas_data, tf.zeros_like(meas_data))
    >>> # Finally, compute the coil sensitivities using the above function
    >>> # to extract the calibration data.
    >>> maps = tfmri.coils.estimate_sensitivities_universal(
    ...     kspace, linop_mri, calib_fn=calib_fn)

  Args:
    meas_data: A `tf.Tensor` containing the measurement or observation data.
      Must be compatible with the range of `operator`, i.e., it should be a
      plausible output of the system operator. Accordingly, it should be a
      plausible input for the adjoint of the system operator.
      ```{tip}
      In MRI, this is usually the *k*-space data.
      ```
    operator: A `tfmri.linalg.LinearOperator` describing the action of the
      measurement system, i.e., mapping an object Its range must be compatible with `meas_data`, i.e.,
      its adjoint should be able to process `meas_data` correctly.
      ```{tip}
      In MRI, this is usually an operator mapping images to the corresponding
      *k*-space data. For most MRI experiments, you can use
      `tfmri.linalg.LinearOperatorMRI`.
      ```
    calib_data: A `tf.Tensor` containing the calibration data. Must be
      compatible with `operator`. If `None`, the calibration data will be
      extracted from the `meas_data` tensor using the `calib_fn` function.
      ```{tip}
      In MRI, this is usually the central, fully-sampled region of *k*-space.
      ```
    calib_fn: A callable which extracts the calibration data from the input
      `meas_data`. Must have signature
      `calib_fn(meas_data, operator) -> calib_data`. If `None`, `calib_data`
      will be used for calibration. If `calib_data` is also `None`, `meas_data`
      will be used directly for calibration.
    method: A `str` specifying which coil sensitivity estimation algorithm to
      use. Must be one of `'direct'`, `'walsh'`, `'inati'` or `'espirit'`.
      Defaults to `'walsh'`.
    **kwargs: Additional keyword arguments depending on the `method`. For a
      list of available arguments, see `tfmri.coils.estimate_sensitivites`.

  Returns:
    A `tf.Tensor` of shape `[..., coils, *spatial_dims]` containing the coil
    sensitivities.

  Raises:
    ValueError: If both `calib_data` and `calib_fn` are provided.
  """
  with tf.name_scope(kwargs.get('name', 'estimate_sensitivities_universal')):
    rank = operator.rank
    meas_data = tf.convert_to_tensor(meas_data)

    if calib_data is None and calib_fn is None:
      calib_data = meas_data
    elif calib_data is None and calib_fn is not None:
      calib_data = calib_fn(meas_data, operator)
    elif calib_data is not None and calib_fn is None:
      calib_data = tf.convert_to_tensor(calib_data)
    else:
      raise ValueError(
          "Only one of `calib_data` and `calib_fn` may be specified.")

    # Reconstruct image.
    calib_data = recon_adjoint.recon_adjoint(calib_data, operator)

    # If method is `'direct'`, we simply return the reconstructed calibration
    # data.
    if method == 'direct':
      return calib_data

    # ESPIRiT method takes in k-space data, so convert back to k-space in this
    # case.
    if method == 'espirit':
      axes = list(range(-rank, 0))
      calib_data = fft_ops.fftn(calib_data, axes=axes, norm='ortho', shift=True)

    # Reshape to single batch dimension.
    batch_shape_static = calib_data.shape[:-(rank + 1)]
    batch_shape = tf.shape(calib_data)[:-(rank + 1)]
    calib_shape = tf.shape(calib_data)[-(rank + 1):]
    calib_data = tf.reshape(calib_data, tf.concat([[-1], calib_shape], 0))

    # Apply estimation for each element in batch.
    maps = tf.map_fn(
        functools.partial(estimate,
                          coil_axis=-(rank + 1),
                          method=method,
                          **kwargs),
        calib_data)

    # Restore batch shape.
    output_shape = tf.shape(maps)[1:]
    output_shape_static = maps.shape[1:]
    maps = tf.reshape(maps,
        tf.concat([batch_shape, output_shape], 0))
    maps = tf.ensure_shape(
        maps, batch_shape_static.concatenate(output_shape_static))

    return maps


_prod = lambda iterable: functools.reduce(lambda x, y: x * y, iterable)
