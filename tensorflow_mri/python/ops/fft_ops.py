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
"""Fast Fourier transform operations."""

import tensorflow as tf

from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.util import check_util


def fftn(x, shape=None, axes=None, norm='backward', shift=False):
  """Compute the N-dimensional discrete Fourier Transform.

  This function computes the `N`-dimensional discrete Fourier Transform over any
  number of axes in an `M`-dimensional array by means of the Fast Fourier
  Transform (FFT).

  .. note::
    `N` must be 1, 2 or 3.

  Args:
    x: A `Tensor`. Must be one of the following types: `complex64`,
      `complex128`.
    shape: A `Tensor`, a `TensorShape` or a list of `ints`. Shape (length of
      each transformed axis) of the output (`s[0]` refers to axis 0, `s[1]`
      to axis 1, etc.). Along any axis, if the given shape is smaller than that
      of the input, the input is cropped. If it is larger, the input is padded
      with zeros. If `shape` is not given, the shape of the input along the axes
      specified by `axes` is used.
    axes: A `Tensor`, a `TensorShape` or a list of `ints`. Axes over which to
      compute the FFT. If not given, the last `len(shape)` axes are used, or all
      axes if `shape` is also not specified.
    norm: A `string`. The normalization mode. Must be one of `"forward"`,
      `"backward"` or `"ortho"`. Defaults to `"backward"`. Indicates which
      direction of the forward/backward pair of transforms is scaled and with
      what normalization factor.
    shift: A `bool`. If `True`, perform a "centered" transform by appropriately
      shifting the inputs/outputs (eg, shifting zero-frequency components to the
      center of the spectrum).

  Returns:
    The truncated or zero-padded input tensor, transformed along the axes
    indicated by `axes`, or by a combination of `shape` and `axes`, as explained
    in the parameters section above.

  Raises:
    TypeError: If `x` is not of a complex type.
    InvalidArgumentError: If length of `shape` is greater than the rank of
      `x`.
    InvalidArgumentError: If length of `axes` is greater than the rank of
      `x`.
    InvalidArgumentError: If any element of `axes` is larger than the number
      of axes of `x`.
    InvalidArgumentError: If `shape` and `axes` have different length.
    ValueError: If `norm` is not one of 'forward', 'backward' or 'ortho'.
  """
  return _fft_internal(x, shape, axes, norm, shift, 'forward')


def ifftn(x, shape=None, axes=None, norm='backward', shift=False):
  """Compute the N-dimensional inverse discrete Fourier Transform.

  This function computes the inverse of the `N`-dimensional discrete Fourier
  Transform over any number of axes in an M-dimensional array by means of
  the Fast Fourier Transform (FFT).

  .. note::
    `N` must be 1, 2 or 3.

  Args:
    x: A `Tensor`. Must be one of the following types: `complex64`,
      `complex128`.
    shape: A `Tensor`, a `TensorShape` or a list of `ints`. Shape (length of
      each transformed axis) of the output (`s[0]` refers to axis 0, `s[1]`
      to axis 1, etc.). Along any axis, if the given shape is smaller than that
      of the input, the input is cropped. If it is larger, the input is padded
      with zeros. If `shape` is not given, the shape of the input along the axes
      specified by `axes` is used.
    axes: A `Tensor` or a list of `ints`. Axes over which to compute the FFT.
      If not given, the last `len(shape)` axes are used, or all axes if `shape`
      is also not specified.
    norm: A `string`. The normalization mode. Must be one of `"forward"`,
      `"backward"` or `"ortho"`. Defaults to `"backward"`. Indicates which
      direction of the forward/backward pair of transforms is scaled and with
      what normalization factor.
    shift: A `bool`. If `True`, perform a "centered" transform by appropriately
      shifting the inputs/outputs (eg, shifting zero-frequency components to the
      center of the spectrum).

  Returns:
    The truncated or zero-padded input tensor, transformed along the axes
    indicated by `axes`, or by a combination of `shape` and `axes`, as explained
    in the parameters section above.

  Raises:
    TypeError: If `x` is not of a complex type.
    InvalidArgumentError: If length of `shape` is greater than the rank of
      `x`.
    InvalidArgumentError: If length of `axes` is greater than the rank of
      `x`.
    InvalidArgumentError: If any element of `axes` is larger than the number
      of axes of `x`.
    InvalidArgumentError: If `shape` and `axes` have different length.
    ValueError: If `norm` is not one of 'forward', 'backward' or 'ortho'.
  """
  return _fft_internal(x, shape, axes, norm, shift, 'backward')


def _fft_internal(x, shape, axes, norm, shift, transform): # pylint: disable=missing-raises-doc
  """Compute the N-dimensional (inverse) discrete Fourier Transform.

  Args:
    transform: Transform to compute. One of {'forward', 'backward'}.

  For the other parameters, see `fft` and `ifft`.

  Returns:
    See `fft` and `ifft`.
  """
  # Convert input to tensor.
  x = tf.convert_to_tensor(x)
  input_shape = x.shape
  input_rank_tensor = tf.rank(x)

  # Save original inputs for computation of static shape.
  shape_, axes_ = shape, axes

  if not x.dtype.is_complex:
    raise TypeError((
      "Invalid FFT input: `x` must be of a complex dtype. "
      "Received: {}").format(x.dtype))
  # Convert shape and axes to tensors if specified and run some checks.
  if shape is not None:
    shape = tf.convert_to_tensor(shape, dtype=tf.dtypes.int32)
    checks = []
    checks.append(tf.debugging.assert_less_equal(
        tf.size(shape), input_rank_tensor, message=(
            "Argument `shape` cannot have length greater than the rank of `x`. "
            "Received: {}").format(shape)))
    with tf.control_dependencies(checks):
      shape = tf.identity(shape)
  if axes is not None:
    axes = tf.convert_to_tensor(axes, dtype=tf.dtypes.int32)
    checks = []
    checks.append(tf.debugging.assert_less_equal(
        tf.size(axes), input_rank_tensor, message=(
            "Argument `axes` cannot have length greater than the rank of `x`. "
            "Received: {}").format(axes)))
    checks.append(tf.debugging.assert_less(
        axes, input_rank_tensor, message=(
            "Argument `axes` contains invalid indices. "
            "Received: {}").format(axes)))
    checks.append(tf.debugging.assert_greater_equal(
        axes, -input_rank_tensor, message=(
            "Argument `axes` contains invalid indices. "
            "Received: {}").format(axes)))
    with tf.control_dependencies(checks):
      axes = tf.identity(axes)
  if shape is not None and axes is not None:
    checks = []
    checks.append(
        tf.debugging.assert_equal(tf.size(shape), tf.size(axes), message=(
          "Arguments `shape` and `axes` must have equal length. "
          "Received: {}, {}").format(shape, axes)))
    with tf.control_dependencies(checks):
      shape, axes = tf.identity_n([shape, axes])

  # Default value for `axes`.
  if axes is None:
    if shape is None:
      axes = tf.range(-tf.size(input_shape), 0) # pylint: disable=invalid-unary-operand-type
    else:
      axes = tf.range(-tf.size(shape), 0) # pylint: disable=invalid-unary-operand-type
  # Translate negative axes to positive axes.
  axes = tf.where(tf.math.less(axes, 0), axes + input_rank_tensor, axes)
  # Set flags for which parts of computation to perform. This might allow a
  # slight performance improvement.
  perform_padding = shape is not None
  perform_transpose = tf.math.logical_not(tf.math.reduce_all(tf.math.equal(
    axes, tf.range(input_rank_tensor - tf.size(axes), input_rank_tensor))))
  # Default value for `shape`.
  if shape is None:
    shape = tf.gather(tf.shape(x), axes, axis=0)
  # Rank of the op.
  rank = tf.size(axes)
  with tf.control_dependencies([tf.debugging.assert_less_equal(
      rank, 3, message=("N-D FFT supported only up to 3-D."))]):
    rank = tf.identity(rank)

  # Normalization factor.
  norm = check_util.validate_enum(
    norm, {'forward', 'backward', 'ortho'}, 'norm')
  if norm == 'backward':
    norm_factor = tf.constant(1, x.dtype)
  elif norm == 'forward':
    norm_factor = tf.cast(tf.math.reduce_prod(shape), x.dtype)
  elif norm == 'ortho':
    norm_factor = tf.cast(tf.math.reduce_prod(shape), x.dtype)
    norm_factor = tf.math.sqrt(norm_factor)

  # Apply padding/cropping.
  if perform_padding:
    # `shape` may have less dimensions than input. Fill the remaining outer
    # dimensions with special value -1 to leave those unchanged.
    pad_shape = -tf.ones([input_rank_tensor], dtype=tf.int32)
    pad_shape = tf.tensor_scatter_nd_update(
      pad_shape, tf.expand_dims(axes, -1), shape)
    if shift:
      x = image_ops.resize_with_crop_or_pad(x, pad_shape)
    else:
      x = _right_pad_or_crop(x, pad_shape)

  # Apply input domain FFT shift.
  if shift:
    x = tf.signal.ifftshift(x, axes=axes)

  # Permutation to move op dimensions to the end. The following uses
  # tf.boolean_mask and tf.foldl to get the dimensions which are not in
  # `axes`.
  all_dims = tf.range(input_rank_tensor, dtype=tf.dtypes.int32)
  perm = tf.concat([
    tf.boolean_mask(
      all_dims,
      tf.foldl(
        lambda acc, elem: tf.math.logical_and(
          acc, tf.math.not_equal(all_dims, elem)),
        axes,
        initializer=tf.fill(all_dims.shape, True))),
    axes], 0)

  x = tf.cond(perform_transpose,
              lambda: tf.transpose(x, perm=perm),
              lambda: x)

  # Perform FFT. Currently, only rank up to 3 is supported.
  if transform == 'forward':
    # We need to check static rank here to make sure FFT ops do not complain
    # about input rank being lower than op rank.
    if x.shape.rank == 1:
      x = tf.signal.fft(x)
    elif x.shape.rank == 2:
      x = tf.switch_case(rank - 1, {
        0: lambda: tf.signal.fft(x),
        1: lambda: tf.signal.fft2d(x)})
    else:
      x = tf.switch_case(rank - 1, {
        0: lambda: tf.signal.fft(x),
        1: lambda: tf.signal.fft2d(x),
        2: lambda: tf.signal.fft3d(x)})
    # Apply normalization.
    x = x / norm_factor
  elif transform == 'backward':
    if x.shape.rank == 1:
      x = tf.signal.ifft(x)
    elif x.shape.rank == 2:
      x = tf.switch_case(rank - 1, {
        0: lambda: tf.signal.ifft(x),
        1: lambda: tf.signal.ifft2d(x)})
    else:
      x = tf.switch_case(rank - 1, {
        0: lambda: tf.signal.ifft(x),
        1: lambda: tf.signal.ifft2d(x),
        2: lambda: tf.signal.ifft3d(x)})
    # Apply normalization.
    x = x * norm_factor

  # Undo transpose.
  x = tf.cond(perform_transpose,
              lambda: tf.transpose(x, perm=tf.argsort(perm)),
              lambda: x)

  # Apply output domain FFT shift.
  if shift:
    x = tf.signal.fftshift(x, axes=axes)

  # Set the static shape.
  def _static_output_shape(input_shape, shape, axes):

    # Output shape is equal to input shape, unless `shape` is given.
    output_shape = input_shape.as_list()
    if shape is not None:
      # Get the axes if not given.
      if axes is None:
        axes = list(range(-len(shape), 0))
      if isinstance(shape, tf.Tensor):
        # Dynamic shape. We don't know the output dimensions for the
        # corresponding axes.
        if isinstance(axes, tf.Tensor):
          # Dynamic axes. We don't know which dimensions might be
          # changed, therefore it is not possible to infer anything
          # at all about the output shape except its rank.
          output_shape = [None] * len(output_shape)
        else:
          # Static axes. Only the `axes` dimensions remain unknown.
          for ax in axes:
            output_shape[ax] = None
      else:
        # Static shape. We can infer everything about output shape.
        for idx, ax in enumerate(axes):
          output_shape[ax] = shape[idx]
    return tf.TensorShape(output_shape)

  x = tf.ensure_shape(x, _static_output_shape(input_shape, shape_, axes_))

  return x


def _right_pad_or_crop(tensor, shape):
  """Pad or crop a tensor to the specified shape.

  The tensor will be padded and/or cropped in its right side.

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. The shape of the output tensor. If the size of `shape` is
      smaller than the rank of `tensor`, it is assumed to refer to the innermost
      dimensions of `tensor`.

  Returns:
    A `Tensor`. Has the same type as `tensor`. The symmetrically padded/cropped
    tensor.
  """
  # Get input and output shapes.
  input_shape = tf.shape(tensor)
  shape = tf.convert_to_tensor(shape, dtype=tf.dtypes.int32)

  # Normalize `shape`, which may have less dimensions than input. In this
  # case, `shape` is assumed to refer to the last dimensions in `x`.
  with tf.control_dependencies([tf.debugging.assert_less_equal(
      tf.size(shape), tf.size(input_shape))]):
    shape = tf.identity(shape)
  shape = tf.concat([input_shape[:tf.size(input_shape) - tf.size(shape)],
                     shape], 0)

  # Pad tensor with zeros.
  pad_sizes = tf.math.maximum(shape - input_shape, 0)
  pad_sizes = tf.expand_dims(pad_sizes, -1)
  pad_sizes = tf.concat([tf.zeros(pad_sizes.shape, dtype=tf.dtypes.int32),
                         pad_sizes], -1)
  tensor = tf.pad(tensor, pad_sizes, constant_values=0)

  # Crop tensor.
  begin = tf.zeros(shape.shape, dtype=tf.dtypes.int32)
  tensor = tf.slice(tensor, begin, shape)

  return tensor
