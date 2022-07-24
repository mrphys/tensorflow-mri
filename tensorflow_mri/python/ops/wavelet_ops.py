# ==============================================================================
# Copyright 2022 University College London. All Rights Reserved.
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

# ==============================================================================
# Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
# Copyright (c) 2012-2020 The PyWavelets Developers <https://github.com/PyWavelets/pywt>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Wavelet operators.

Most of the code in this file is taken from the PyWavelets library,
with some modifications in order to use a TensorFlow backend.
"""

import collections.abc
import itertools
import warnings

import pywt
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.util import api_util


@api_util.export("signal.dwt")
def dwt(data, wavelet, mode='symmetric', axes=None):
  """Single-level N-dimensional discrete wavelet transform (DWT).

  Args:
    data: A `tf.Tensor` of real or complex type.
    wavelet: A `str` or a `pywt.Wavelet`_, or a `list` thereof. When passed a
      `list`, different wavelets are applied along each axis in `axes`.
    mode: A `str`. The padding or signal extension mode. Must be one of the
      values supported by `tf.pad`_. Defaults to `'symmetric'`.
    axes: A `list` of `int`. Axes over which to compute the DWT. Repeated
      elements mean the DWT will be performed multiple times along these axes.
      A value of `None` (the default) selects all axes.

  Returns:
    A `dict` where key specifies the transform type on each dimension and value
    is an N-dimensional `tf.Tensor` containing the corresponding coefficients.

    For example, for a 2D case the result will have keys `'aa'` (approximation
    on 1st dimension, approximation on 2nd dimension), `'ad'` (approximation on
    1st dimension, detail on 2nd dimension), `'da'` (detail on 1st dimension,
    approximation on 2nd dimension), and `'dd'` (detail on 1st dimension, detail
    on 2nd dimension).

    For user-specified `axes`, the order of the characters in the
    dictionary keys map to the specified `axes`.

  Raises:
    ValueError: If any of the inputs is not valid.

  .. _pywt.Wavelet: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#pywt.Wavelet
  .. _tf.pad: https://www.tensorflow.org/api_docs/python/tf/pad
  """
  data = tf.convert_to_tensor(data)
  rank = data.shape.rank

  # Handle complex numbers.
  if data.dtype.is_complex:
    real = dwt(tf.math.real(data), wavelet, mode, axes)
    imag = dwt(tf.math.imag(data), wavelet, mode, axes)
    return {k: tf.dtypes.complex(real[k], imag[k]) for k in real.keys()}

  # Canonicalize axes. If None, compute decomposition along all axes.
  if axes is None:
    axes = range(rank)
  axes = [ax + rank if ax < 0 else ax for ax in axes]

  # Get padding mode for each axis.
  wavelets = _wavelets_per_axis(wavelet, axes)
  modes = _modes_per_axis(mode, axes)

  coeffs = [('', data)]
  for ax, wav, mod in zip(axes, wavelets, modes):
    new_coeffs = []
    for subband, x in coeffs:
      c_a, c_d = _dwt_along_axis(x, wav, mod, ax)
      new_coeffs.extend([(subband + 'a', c_a),
                         (subband + 'd', c_d)])
    coeffs = new_coeffs
  return dict(coeffs)


@api_util.export("signal.idwt")
def idwt(coeffs, wavelet, mode='symmetric', axes=None):
  """Single-level N-dimensional inverse discrete wavelet transform (IDWT).

  Args:
    coeffs: A `dict` with the same structure as the output of
      `tfmri.signal.dwt`. Missing or `None` items will be treated as zeros.
    wavelet: A `str` or a `pywt.Wavelet`_, or a `list` thereof. When passed a
      `list`, different wavelets are applied along each axis in `axes`.
    mode: A `str`. The padding or signal extension mode. Must be one of the
      values supported by `tf.pad`_. Defaults to `'symmetric'`.
    axes: A `list` of `int`. Axes over which to compute the DWT. Repeated
      elements mean the DWT will be performed multiple times along these axes.
      A value of `None` (the default) selects all axes.

  Returns:
    A `tf.Tensor` containing the reconstructed signal.

  Raises:
    ValueError: If any of the inputs is not valid.

  .. _pywt.Wavelet: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#pywt.Wavelet
  .. _tf.pad: https://www.tensorflow.org/api_docs/python/tf/pad
  """
  # Drop the keys where value is None.
  coeffs = {k: v for k, v in coeffs.items() if v is not None}

  # Check key combinations.
  coeffs = _fix_coeffs(coeffs)

  # Handle complex numbers.
  if any(v.dtype.is_complex for v in coeffs.values()):
    real = {k: tf.math.real(v) for k, v in coeffs.items()}
    imag = {k: tf.math.imag(v) for k, v in coeffs.items()}
    return tf.dtypes.complex(idwt(real, wavelet, mode, axes),
                             idwt(imag, wavelet, mode, axes))

  # key length matches the number of axes transformed
  rank_transform = max(len(key) for key in coeffs.keys())

  try:
    coeff_shapes = (v.shape for k, v in coeffs.items()
                    if v is not None and len(k) == rank_transform)
    coeff_shape = next(coeff_shapes)
  except StopIteration:
    raise ValueError("`coeffs` must contain at least one non-null wavelet band")  # pylint: disable=raise-missing-from
  if any(s != coeff_shape for s in coeff_shapes):
    raise ValueError("`coeffs` must all be of equal size (or None)")

  if axes is None:
    axes = range(rank_transform)
    ndim = rank_transform
  else:
    ndim = len(coeff_shape)
  axes = [a + ndim if a < 0 else a for a in axes]

  modes = _modes_per_axis(mode, axes)
  wavelets = _wavelets_per_axis(wavelet, axes)
  for key_length, (ax, wav, mod) in reversed(
      list(enumerate(zip(axes, wavelets, modes)))):
    if ax < 0 or ax >= ndim:
      raise ValueError("Axis greater than data dimensions")

    new_coeffs = {}
    new_keys = [''.join(coef)
                for coef in itertools.product('ad', repeat=key_length)]

    for key in new_keys:
      lo = coeffs.get(key + 'a', None)
      hi = coeffs.get(key + 'd', None)
      new_coeffs[key] = _idwt_along_axis(lo, hi, wav, mod, ax)
    coeffs = new_coeffs

  return coeffs['']


@api_util.export("signal.wavedec")
def wavedec(data, wavelet, mode='symmetric', level=None, axes=None):
  """Multilevel N-dimensional discrete wavelet transform (wavelet decomposition).

  Args:
    data: A `tf.Tensor` of real or complex type.
    wavelet: A `str` or a `pywt.Wavelet`_, or a `list` thereof. When passed a
      `list`, different wavelets are applied along each axis in `axes`.
    mode: A `str`. The padding or signal extension mode. Must be one of the
      values supported by `tf.pad`_. Defaults to `'symmetric'`.
    level: An `int` >= 0. The decomposition level. If `None` (default),
      the maximum useful level of decomposition will be used (see
      `tfmri.signal.max_wavelet_level`).
    axes: A `list` of `int`. Axes over which to compute the DWT. Axes may not
      be repeated. A value of `None` (the default) selects all axes.

  Returns:
    A `list` of coefficients such as
    `[approx, {details_level_n}, ..., {details_level_1}]`. The first element
    in the list contains the approximation coefficients at level `n`. The
    remaining elements contain the detail coefficients, listed in descending
    order of decomposition level. Each ``details_level_i`` element is a
    `dict` containing detail coefficients at level ``i`` of the decomposition.
    As a concrete example, a 3D decomposition would have the following set of
    keys in each `details_level_i` `dict`:
    `{'aad', 'ada', 'daa', 'add', 'dad', 'dda', 'ddd'}, where the order of the
    characters in each key map to the specified `axes`.

  Examples:
    >>> import tensorflow as tf
    >>> import tensorflow_mri as tfmri
    >>> coeffs = tfmri.signal.wavedec(tf.ones((4, 4)), 'db1')
    >>> # Levels:
    >>> len(coeffs)-1
    2
    >>> tfmri.signal.waverec(coeffs, 'db1')
    <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
    array([[0.9999999, 0.9999999, 0.9999999, 0.9999999],
           [0.9999999, 0.9999999, 0.9999999, 0.9999999],
           [0.9999999, 0.9999999, 0.9999999, 0.9999999],
           [0.9999999, 0.9999999, 0.9999999, 0.9999999]], dtype=float32)>

  .. _pywt.Wavelet: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#pywt.Wavelet
  .. _tf.pad: https://www.tensorflow.org/api_docs/python/tf/pad
  """
  data = tf.convert_to_tensor(data)
  axes, axes_shapes, rank_transform = _prep_axes_wavedec(data.shape, axes)
  wavelets = _wavelets_per_axis(wavelet, axes)
  dec_lengths = [w.dec_len for w in wavelets]

  level = _check_level(axes_shapes, dec_lengths, level)

  coeffs_list = []

  a = data
  for _ in range(level):
    coeffs = dwt(a, wavelet, mode, axes)
    a = coeffs.pop('a' * rank_transform)
    coeffs_list.append(coeffs)

  coeffs_list.append(a)
  coeffs_list.reverse()

  return coeffs_list


@api_util.export("signal.waverec")
def waverec(coeffs, wavelet, mode='symmetric', axes=None):
  """Multilevel N-dimensional inverse discrete wavelet transform (wavelet reconstruction).

  Args:
    coeffs: A `list` with the same structure as the output of
      `tfmri.signal.wavedec`.
    wavelet: A `str` or a `pywt.Wavelet`_, or a `list` thereof. When passed a
      `list`, different wavelets are applied along each axis in `axes`.
    mode: A `str`. The padding or signal extension mode. Must be one of the
      values supported by `tf.pad`_. Defaults to `'symmetric'`.
    axes: A `list` of `int`. Axes over which to compute the IDWT. Axes may not
        be repeated. A value of `None` (the default) selects all axes.

  Returns:
    A `tf.Tensor` containing the reconstructed signal.

  Examples:
    >>> import tensorflow as tf
    >>> import tensorflow_mri as tfmri
    >>> coeffs = tfmri.signal.wavedec(tf.ones((4, 4)), 'db1')
    >>> # Levels:
    >>> len(coeffs)-1
    2
    >>> tfmri.signal.waverec(coeffs, 'db1')
    <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
    array([[0.9999999, 0.9999999, 0.9999999, 0.9999999],
           [0.9999999, 0.9999999, 0.9999999, 0.9999999],
           [0.9999999, 0.9999999, 0.9999999, 0.9999999],
           [0.9999999, 0.9999999, 0.9999999, 0.9999999]], dtype=float32)>

  Raises:
    ValueError: If passed invalid input values.
  """
  if len(coeffs) < 1:
    raise ValueError(
        "Coefficient list too short (minimum 1 array required).")

  a, ds = coeffs[0], coeffs[1:]

  # this dictionary check must be prior to the call to _fix_coeffs
  if len(ds) > 0 and not all(isinstance(d, dict) for d in ds):
    raise ValueError(
        f"Unexpected detail coefficient type: {type(ds[0])}. Detail "
        f"coefficients must be a dict of arrays as returned by wavedec.")

  # Raise error for invalid key combinations
  ds = list(map(_fix_coeffs, ds))

  if not ds:
    # level 0 transform (just returns the approximation coefficients)
    return coeffs[0]
  if a is None and not any(ds):
    raise ValueError(
        "At least one coefficient must contain a valid value.")

  coeff_ndims = []
  if a is not None:
    a = np.asarray(a)
    coeff_ndims.append(a.ndim)
  for d in ds:
    coeff_ndims += [v.ndim for k, v in d.items()]

  # test that all coefficients have a matching number of dimensions
  unique_coeff_ndims = np.unique(coeff_ndims)
  if len(unique_coeff_ndims) == 1:
    ndim = unique_coeff_ndims[0]
  else:
    raise ValueError(
        "All coefficients must have a matching number of dimensions")

  if np.isscalar(axes):
    axes = (axes, )
  if axes is None:
    axes = range(ndim)
  else:
    axes = tuple(axes)
  if len(axes) != len(set(axes)):
    raise ValueError("The axes passed to waverecn must be unique.")
  rank_transform = len(axes)

  for idx, d in enumerate(ds):
    if a is None and not d:
      continue
    # The following if statement handles the case where the approximation
    # coefficient returned at the previous level may exceed the size of the
    # stored detail coefficients by 1 on any given axis.
    if idx > 0:
      a = _match_coeff_dims(a, d)
    d['a' * rank_transform] = a
    a = idwt(d, wavelet, mode, axes)

  return a


def _dwt_along_axis(x, wavelet, mode, axis):  # pylint: disable=missing-param-doc
  """Computes the DWT along a single axis."""
  # Move axis `axis` to last position.
  perm = list(range(x.shape.rank))
  perm = [ax for ax in perm if ax != axis] + [axis]
  x = tf.transpose(x, perm)

  # Do padding.
  pad_length = len(wavelet) - 1
  paddings = [[0, 0]] * (x.shape.rank - 1) + [[pad_length, pad_length]]
  x = tf.pad(x, paddings, mode=mode)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter

  # Add channel dimension.
  x = tf.expand_dims(x, axis=-1)

  # conv1d requires at least one batch dimension, so add a dummy one.
  scalar_batch = False
  if x.shape.rank == 2:
    scalar_batch = True
    x = tf.expand_dims(x, axis=0)

  # Shift the input tensor by 1. We take care to retain the static shape of the
  # sliced tensor.
  begin = [0] * x.shape.rank
  begin[-2] = 1
  size = tf.shape(x) - begin
  shape = x.shape.as_list()
  if shape[-2] is not None:
    shape[-2] -= 1
  x = tf.slice(x, begin, size)
  x = tf.ensure_shape(x, shape)

  # Get filters.
  f_lo = tf.reverse(tf.reshape(wavelet.dec_lo, [-1, 1, 1]), [0])
  f_hi = tf.reverse(tf.reshape(wavelet.dec_hi, [-1, 1, 1]), [0])
  f_lo = tf.cast(f_lo, x.dtype)
  f_hi = tf.cast(f_hi, x.dtype)

  # Compute approximation and detail coeffs.
  a = tf.nn.conv1d(x, f_lo, 2, 'VALID')
  d = tf.nn.conv1d(x, f_hi, 2, 'VALID')

  # Remove dummy scalar dimension, if necessary.
  if scalar_batch:
    a = tf.squeeze(a, axis=0)
    d = tf.squeeze(d, axis=0)

  # Remove channel dimension.
  a = tf.squeeze(a, axis=-1)
  d = tf.squeeze(d, axis=-1)

  # Invert the original permutation. We use NumPy intentionally here as we
  # want to do this computation statically.
  inv_perm = np.argsort(perm).tolist()
  a = tf.transpose(a, inv_perm)
  d = tf.transpose(d, inv_perm)

  return a, d


def _idwt_along_axis(a, d, wavelet, mode, axis):  # pylint: disable=missing-param-doc,unused-argument
  """Computes the IDWT along a single axis."""
  # Move axis `axis` to last position.
  perm = list(range(a.shape.rank))
  perm = [ax for ax in perm if ax != axis] + [axis]
  a = tf.transpose(a, perm)
  d = tf.transpose(d, perm)

  # Add channel dimension.
  a = tf.expand_dims(a, axis=-1)
  d = tf.expand_dims(d, axis=-1)

  # conv1d requires at least one batch dimension, so add a dummy one.
  scalar_batch = False
  if a.shape.rank == 2:
    scalar_batch = True
    a = tf.expand_dims(a, axis=0)
    d = tf.expand_dims(d, axis=0)

  # Get filters.
  f_lo = tf.reverse(tf.reshape(wavelet.rec_lo, [-1, 1, 1]), [0])
  f_hi = tf.reverse(tf.reshape(wavelet.rec_hi, [-1, 1, 1]), [0])
  f_lo = tf.cast(f_lo, a.dtype)
  f_hi = tf.cast(f_hi, d.dtype)

  # Define length.
  input_length = 2 * tf.shape(a)[-2]
  filter_length = tf.shape(f_lo)[-3]
  output_length = input_length - filter_length + 2

  # Dyadic upsampling.
  a = _dyadic_upsampling(a, axis=-2, indices='even')
  d = _dyadic_upsampling(d, axis=-2, indices='even')

  # Do extra padding to implement "FULL" convolution mode.
  left_padding = len(wavelet.rec_lo) - 1
  right_padding = len(wavelet.rec_hi) - 1
  paddings = [[0, 0]] * a.shape.rank
  paddings[-2] = [left_padding, right_padding]
  a = tf.pad(a, paddings)  # pylint: disable=no-value-for-parameter
  d = tf.pad(d, paddings)  # pylint: disable=no-value-for-parameter

  # Do convolution.
  a = tf.nn.conv1d(a, f_lo, 1, 'VALID')
  d = tf.nn.conv1d(d, f_hi, 1, 'VALID')

  # Keep only part of the output.
  current_length = tf.shape(a)[-2]
  begin = tf.scatter_nd(
      [[tf.rank(a) - 2]], [(current_length - output_length) // 2], [tf.rank(a)])
  size = tf.tensor_scatter_nd_update(
      tf.shape(a), [[tf.rank(a) - 2]], [output_length])
  a = tf.slice(a, begin, size)
  d = tf.slice(d, begin, size)

  # Reconstructed signal.
  x = a + d

  # Remove dummy scalar dimension, if necessary.
  if scalar_batch:
    x = tf.squeeze(x, axis=0)

  # Remove channel dimension.
  x = tf.squeeze(x, axis=-1)

  # Invert the original permutation. We use NumPy intentionally here as we
  # want to do this computation statically.
  inv_perm = np.argsort(perm).tolist()
  x = tf.transpose(x, inv_perm)

  return x


def _dyadic_upsampling(x, axis=-1, indices='odd'):
  """Performs dyadic upsampling along an axis.

  Args:
    x: A `tf.Tensor`.
    axis: An `int`. along which to upsample.
    indices: A `str`. Must be `'odd'` or `'even'`. Controls whether to upsample
      odd- or even-indexed elements.

  Returns:
    The upsampled `tf.Tensor`.
  """
  # Canonicalize axis.
  axis = axis + x.shape.rank if axis < 0 else axis

  # Compute output shape.
  output_shape = tf.tensor_scatter_nd_update(
      tf.shape(x), [[axis]], [2 * tf.shape(x)[axis]])

  zeros = tf.zeros_like(x)
  x = tf.stack([x, zeros], axis=(axis + 1))

  # Reshape to correct shape.
  x = tf.reshape(x, output_shape)
  if indices == 'even':
    begin = tf.zeros([tf.rank(x)], dtype=tf.int32)
    size = tf.tensor_scatter_nd_update(
        tf.shape(x), [[axis]], [tf.shape(x)[axis] - 1])
    x = tf.slice(x, begin, size)

  elif indices == 'odd':
    paddings = tf.zeros([tf.rank(x), 2], dtype=tf.int32)
    paddings = tf.tensor_scatter_nd_update(paddings, [[axis, 0]], [1])
    x = tf.pad(x, paddings)  # pylint: disable=no-value-for-parameter

  return x


def _wavelets_per_axis(wavelet, axes):
  """Initialize Wavelets for each axis to be transformed.

  Args:
    wavelet: A `str` or `Wavelet` or an iterable of `str` or `Wavelet`. If a
      single wavelet is provided, it will used for all axes. Otherwise one
      wavelet per axis must be provided.
    axes: The `list` of axes to be transformed.

  Returns:
    A `list` of wavelets equal in length to `axes`.

  Raises:
    ValueError: If `wavelet` is not valid for the given `axes`.
  """
  axes = tuple(axes)
  if isinstance(wavelet, (str, pywt.Wavelet)):
    # Same wavelet on all axes.
    wavelets = [_as_wavelet(wavelet)] * len(axes)
  elif isinstance(wavelet, collections.abc.Iterable):
    # (potentially) unique wavelet per axis (e.g. for dual-tree DWT)
    if len(wavelet) == 1:
      wavelets = [_as_wavelet(wavelet[0])] * len(axes)
    else:
      if len(wavelet) != len(axes):
        raise ValueError((
            "The number of wavelets must match the number of axes "
            "to be transformed."))
      wavelets = [_as_wavelet(w) for w in wavelet]
  else:
    raise ValueError("wavelet must be a str, Wavelet or iterable")
  return wavelets


def _wavelet_lengths_per_axis(wavelet_or_length, axes):
  """Get wavelet lengths for each axis to be transformed.

  Args:
    wavelet_or_length: An `int`, `str`, `Wavelet` or an iterable of `int`,
      `str` or `Wavelet`. If a scalar input is provided, it will used for
      all axes. Otherwise one input per axis must be provided.
    axes: The `list` of axes to be transformed.

  Returns:
    A `list` of lengths equal in length to `axes`.

  Raises:
    ValueError: If `wavelet_or_length` is not valid for the given `axes`.
  """
  axes = tuple(axes)
  if isinstance(wavelet_or_length, (int, str, pywt.Wavelet)):
    # Same wavelet/length on all axes.
    lengths = [_get_wavelet_length(wavelet_or_length)] * len(axes)
  elif isinstance(wavelet_or_length, collections.abc.Iterable):
    # (potentially) unique wavelet_or_length per axis (e.g. for dual-tree DWT)
    if len(wavelet_or_length) == 1:
      lengths = [_get_wavelet_length(wavelet_or_length[0])] * len(axes)
    else:
      if len(wavelet_or_length) != len(axes):
        raise ValueError((
            "The number of wavelets or lengths must match the number of axes "
            "to be transformed."))
      lengths = [_get_wavelet_length(w) for w in wavelet_or_length]
  else:
    raise ValueError(
        "wavelet_or_length must be an int, str, wavelet or iterable")
  return lengths


def _get_wavelet_length(wavelet_or_length):
  if isinstance(wavelet_or_length, (str, pywt.Wavelet)):
    return _as_wavelet(wavelet_or_length).dec_len
  return wavelet_or_length


def _modes_per_axis(mode, axes):
  """Initialize mode for each axis to be transformed.

  Args:
    mode: A `str` or an iterable of `str`. If a single mode is provided, it
      will used for all axes. Otherwise one mode per axis must be provided.
    axes: The `list` of axes to be transformed.

  Returns:
    A `list` of mode equal in length to `axes`.

  Raises:
    ValueError: If `mode` is not valid for the given `axes`.
  """
  axes = tuple(axes)
  if isinstance(mode, str):
    # same mode on all axes
    mode = [mode] * len(axes)
  elif isinstance(mode, collections.abc.Iterable):
    if len(mode) == 1:
      mode = [mode[0]] * len(axes)
    else:
      if len(mode) != len(axes):
        raise ValueError(
            "The number of mode must match the number "
            "of axes to be transformed.")
    mode = [str(mode) for mode in mode]
  else:
    raise ValueError("mode must be a str or iterable")
  return mode


def _as_wavelet(wavelet):  # pylint: disable=missing-param-doc
  """Convert wavelet name to a Wavelet object."""
  if not isinstance(wavelet, (pywt.ContinuousWavelet, pywt.Wavelet)):
    wavelet = pywt.DiscreteContinuousWavelet(wavelet)
  if isinstance(wavelet, pywt.ContinuousWavelet):
    raise ValueError(
        "A ContinuousWavelet object was provided, but only discrete "
        "Wavelet objects are supported by this function. A list of all "
        "supported discrete wavelets can be obtained by running:\n"
        "print(pywt.wavelist(kind='discrete'))")
  return wavelet


def _fix_coeffs(coeffs):  # pylint: disable=missing-function-docstring
  missing_keys = [k for k, v in coeffs.items() if v is None]
  if missing_keys:
    raise ValueError(
        "The following detail coefficients were set to None:\n"
        "{0}\n"
        "For multilevel transforms, rather than setting\n"
        "\tcoeffs[key] = None\n"
        "use\n"
        "\tcoeffs[key] = np.zeros_like(coeffs[key])\n".format(
            missing_keys))

  invalid_keys = [k for k, v in coeffs.items() if
                  not set(k) <= set('ad')]
  if invalid_keys:
    raise ValueError(
        "The following invalid keys were found in the detail "
        "coefficient dictionary: {}.".format(invalid_keys))

  key_lengths = [len(k) for k in coeffs.keys()]
  if len(np.unique(key_lengths)) > 1:
    raise ValueError(
        "All detail coefficient names must have equal length.")

  return dict((k, tf.convert_to_tensor(v)) for k, v in coeffs.items())


def _check_level(sizes, dec_lens, level):  # pylint: disable=missing-function-docstring
  if np.isscalar(sizes):
    sizes = (sizes, )
  if np.isscalar(dec_lens):
    dec_lens = (dec_lens, )
  max_level = np.min([dwt_max_level(s, d) for s, d in zip(sizes, dec_lens)])
  if level is None:
    level = max_level
  elif level < 0:
    raise ValueError(
        "Level value of %d is too low . Minimum level is 0." % level)
  elif level > max_level:
    warnings.warn(
        ("Level value of {} is too high: all coefficients will experience "
         "boundary effects.").format(level))
  return level


def _prep_axes_wavedec(shape, axes):  # pylint: disable=missing-function-docstring
  rank = shape.rank
  if rank < 1:
    raise ValueError("Expected at least 1D input data.")
  if np.isscalar(axes):
    axes = (axes,)
  if axes is None:
    axes = range(rank)
  else:
    axes = tuple(axes)
  if len(axes) != len(set(axes)):
    raise ValueError("The axes passed to wavedec must be unique.")
  try:
    axes_shapes = [shape[ax] for ax in axes]
  except IndexError:
    raise np.AxisError("Axis greater than data dimensions")  # pylint: disable=raise-missing-from
  rank_transform = len(axes)
  return axes, axes_shapes, rank_transform


def _match_coeff_dims(a_coeff, d_coeff_dict):  # pylint: disable=missing-function-docstring
  # For each axis, compare the approximation coeff shape to one of the
  # stored detail coeffs and truncate the last element along the axis
  # if necessary.
  if a_coeff is None:
    return None
  if not d_coeff_dict:
    return a_coeff
  d_coeff = d_coeff_dict[next(iter(d_coeff_dict))]
  size_diffs = np.subtract(a_coeff.shape, d_coeff.shape)
  if np.any((size_diffs < 0) | (size_diffs > 1)):
    raise ValueError(f"incompatible coefficient array sizes: {size_diffs}")
  return a_coeff[tuple(slice(s) for s in d_coeff.shape)]


@api_util.export("signal.max_wavelet_level")
def dwt_max_level(shape, wavelet_or_length, axes=None):
  """Computes the maximum useful level of wavelet decomposition.

  Returns the maximum level of decomposition suitable for use with
  `tfmri.signal.wavedec`.

  The level returned is the minimum along all axes.

  Examples:
    >>> import tensorflow_mri as tfmri
    >>> tfmri.signal.max_wavelet_level((64, 32), 'db2')
    3

  Args:
    shape: An `int` or a `list` thereof. The input shape.
    wavelet_or_length: A `str`, a `pywt.Wavelet`_. Alternatively, it may also be
      an `int` representing the length of the decomposition filter. This can
      also be a `list` containing a wavelet or filter length for each axis.
    axes: An `list` of `int`. Axes over which the DWT is to be computed.
      If `None` (default), it is assumed that the DWT will be computed along
      all axes.

  Returns:
    An `int` representing the maximum useful level of decomposition.
  """
  # Canonicalize shape.
  if isinstance(shape, int):
    shape = [shape]
  shape = tf.TensorShape(shape)

  # Determine the axes and shape for the transform.
  axes, axes_shapes, _ = _prep_axes_wavedec(shape, axes)

  # Get the filter length for each transformed axis.
  lengths = _wavelet_lengths_per_axis(wavelet_or_length, axes)

  # Maximum level of decomposition per axis.
  max_levels = [_dwt_max_level(input_length, filter_length)
                for input_length, filter_length in zip(axes_shapes, lengths)]
  return min(max_levels)


def _dwt_max_level(input_length, filter_length):
  if filter_length <= 1 or input_length < filter_length - 1:
    return 0

  return _log2_int(input_length // (filter_length - 1))


def _log2_int(x):
  """Returns the integer log2 of x."""
  return x.bit_length() - 1


@api_util.export("signal.wavelet_coeffs_to_tensor")
def coeffs_to_tensor(coeffs, padding=0, axes=None):
  """Arranges a wavelet coefficient list into a single tensor.

  Args:
    coeffs: A `list` of wavelet coefficients as returned by
      `tfmri.signal.wavedec`.
    padding: The value to use for the background if the coefficients cannot be
      tightly packed. If None, raise an error if the coefficients cannot be
      tightly packed.
    axes: Axes over which the DWT that created `coeffs` was performed. The
      default value of `None` corresponds to all axes.

  Returns:
    A `tuple` (`tensor`, `slices`) holding the coefficients
    `tf.Tensor` and a `list` of slices corresponding to each coefficient. For
    example, in a 2D tensor, `tensor[slices[1]['dd']]` would extract
    the first level detail coefficients from `tensor`.

  Raises:
    ValueError: If passed invalid inputs.

  Notes
  -----
  Assume a 2D coefficient dictionary, `c`, from a two-level transform.

  Then all 2D coefficients will be stacked into a single larger 2D array
  as follows::

  .. code-block::

    +---------------+---------------+-------------------------------+
    |               |               |                               |
    |     c[0]      |  c[1]['da']   |                               |
    |               |               |                               |
    +---------------+---------------+           c[2]['da']          |
    |               |               |                               |
    | c[1]['ad']    |  c[1]['dd']   |                               |
    |               |               |                               |
    +---------------+---------------+ ------------------------------+
    |                               |                               |
    |                               |                               |
    |                               |                               |
    |          c[2]['ad']           |           c[2]['dd']          |
    |                               |                               |
    |                               |                               |
    |                               |                               |
    +-------------------------------+-------------------------------+

  If the transform was not performed with mode "periodization" or the signal
  length was not a multiple of ``2**level``, coefficients at each subsequent
  scale will not be exactly 1/2 the size of those at the previous level due
  to additional coefficients retained to handle the boundary condition. In
  these cases, the default setting of `padding=0` indicates to pad the
  individual coefficient arrays with 0 as needed so that they can be stacked
  into a single, contiguous array.

  Examples:
    >>> import tensorflow_mri as tfmri
    >>> image = tfmri.image.phantom()
    >>> coeffs = tfmri.signal.wavedec(image, wavelet='db2', level=3)
    >>> tensor, slices = tfmri.signal.wavelet_coeffs_to_tensor(coeffs)
  """
  coeffs, axes, ndim, ndim_transform = _prepare_coeffs_axes(coeffs, axes)

  # initialize with the approximation coefficients.
  a_coeffs = coeffs[0]
  a_shape = a_coeffs.shape

  if len(coeffs) == 1:
    # only a single approximation coefficient array was found
    return a_coeffs, [tuple([slice(None)] * ndim)]

  # determine size of output and if tight packing is possible
  arr_shape, is_tight_packing = _determine_coeff_array_shape(coeffs, axes)

  # preallocate output array
  if padding is None:
    if not is_tight_packing:
      raise ValueError("array coefficients cannot be tightly packed")
    coeff_tensor = tf.zeros(arr_shape, dtype=a_coeffs.dtype)
  else:
    coeff_tensor = tf.fill(arr_shape, tf.cast(padding, a_coeffs.dtype))

  a_slices = tuple(slice(s) for s in a_shape)
  coeff_tensor = array_ops.update_tensor(coeff_tensor, a_slices, a_coeffs)

  # initialize list of coefficient slices
  coeff_slices = []
  coeff_slices.append(a_slices)

  # loop over the detail cofficients, adding them to coeff_tensor
  ds = coeffs[1:]
  for coeff_dict in ds:
    coeff_slices.append({})  # new dictionary for detail coefficients
    if any(d is None for d in coeff_dict.values()):
      raise ValueError("coeffs_to_tensor does not support missing "
                       "coefficients.")
    d_shape = coeff_dict['d' * ndim_transform].shape
    for key in coeff_dict.keys():
      d = coeff_dict[key]
      slice_array = [slice(None)] * ndim
      for i, let in enumerate(key):
        ax_i = axes[i]  # axis corresponding to this transform index
        if let == 'a':
          slice_array[ax_i] = slice(d.shape[ax_i])
        elif let == 'd':
          slice_array[ax_i] = slice(a_shape[ax_i],
                                    a_shape[ax_i] + d.shape[ax_i])
        else:
          raise ValueError("unexpected letter: {}".format(let))
      slice_array = tuple(slice_array)
      coeff_tensor = array_ops.update_tensor(coeff_tensor, slice_array, d)
      coeff_slices[-1][key] = slice_array
    a_shape = [a_shape[n] + d_shape[n] for n in range(ndim)]
  return coeff_tensor, coeff_slices


@api_util.export("signal.tensor_to_wavelet_coeffs")
def tensor_to_coeffs(coeff_tensor, coeff_slices):
  """Extracts wavelet coefficients from tensor into a list.

  Args:
    coeff_tensor: A `tf.Tensor` containing all wavelet coefficients. This should
      have been generated via `tfmri.signal.wavelet_coeffs_to_tensor`.
    coeff_slices : A `list` of slices corresponding to each coefficient as
      obtained from `tensor_to_wavelet_coeffs`.

  Returns:
    The wavelet coefficients in the format expected by `tfmri.signal.waverec`.

  Raises:
    ValueError: If passed an empty list of coefficients.

  Notes:
    A single large array containing all coefficients will have subsets stored,
    into a `waverecn`` list, c, as indicated below::

    .. code-block::

      +---------------+---------------+-------------------------------+
      |               |               |                               |
      |     c[0]      |  c[1]['da']   |                               |
      |               |               |                               |
      +---------------+---------------+           c[2]['da']          |
      |               |               |                               |
      | c[1]['ad']    |  c[1]['dd']   |                               |
      |               |               |                               |
      +---------------+---------------+ ------------------------------+
      |                               |                               |
      |                               |                               |
      |                               |                               |
      |          c[2]['ad']           |           c[2]['dd']          |
      |                               |                               |
      |                               |                               |
      |                               |                               |
      +-------------------------------+-------------------------------+

  Examples:
    >>> import tensorflow_mri as tfmri
    >>> image = tfmri.image.phantom()
    >>> coeffs = tfmri.signal.wavedec(image, wavelet='db2', level=3)
    >>> tensor, slices = tfmri.signal.wavelet_coeffs_to_tensor(coeffs)
    >>> coeffs_from_arr = tfmri.signal.tensor_to_wavelet_coeffs(tensor, slices)
    >>> image_recon = tfmri.signal.waverec(coeffs_from_arr, wavelet='db2')
    >>> # image and image_recon are equal
  """
  coeff_tensor = tf.convert_to_tensor(coeff_tensor)
  coeffs = []
  if len(coeff_slices) == 0:
    raise ValueError("empty list of coefficient slices")
  coeffs.append(coeff_tensor[coeff_slices[0]])

  # difference coefficients at each level
  for n in range(1, len(coeff_slices)):
    d = {}
    for k, v in coeff_slices[n].items():
      d[k] = coeff_tensor[v]
    coeffs.append(d)
  return coeffs


def _determine_coeff_array_shape(coeffs, axes):  # pylint: disable=missing-param-doc
  """Determines the shape of the coefficients array."""
  arr_shape = np.asarray(coeffs[0].shape)
  axes = np.asarray(axes)  # axes that were transformed
  ndim_transform = len(axes)
  ncoeffs = coeffs[0].shape.num_elements()
  for d in coeffs[1:]:
    arr_shape[axes] += np.asarray(d['d'*ndim_transform].shape)[axes]
    for _, v in d.items():
      ncoeffs += v.shape.num_elements()
  arr_shape = tuple(arr_shape.tolist())
  # if the total number of coefficients doesn't equal the size of the array
  # then tight packing is not possible.
  is_tight_packing = (np.prod(arr_shape) == ncoeffs)
  return arr_shape, is_tight_packing


def _prepare_coeffs_axes(coeffs, axes):  # pylint: disable=missing-param-doc
  """Helper function to check type of coeffs and axes.

  This code is used by both coeffs_to_tensor and ravel_coeffs.
  """
  if not isinstance(coeffs, list) or len(coeffs) == 0:
    raise ValueError("input must be a list of coefficients from wavedec")
  if coeffs[0] is None:
    raise ValueError("coeffs_to_tensor does not support missing "
                     "coefficients.")
  if not isinstance(coeffs[0], tf.Tensor):
    raise ValueError("first list element must be a tensor")
  ndim = coeffs[0].ndim

  if len(coeffs) > 1:
    if not isinstance(coeffs[1], dict):
      raise ValueError("invalid coefficient list")

  if len(coeffs) == 1:
    # no detail coefficients were found
    return coeffs, axes, ndim, None

  # Determine the number of dimensions that were transformed via key length
  ndim_transform = len(list(coeffs[1].keys())[0])
  if axes is None:
    if ndim_transform < ndim:
      raise ValueError(
          "coeffs corresponds to a DWT performed over only a subset of "
          "the axes.  In this case, axes must be specified.")
    axes = np.arange(ndim)

  if isinstance(axes, int):
    axes = [axes]

  if len(axes) != ndim_transform:
    raise ValueError(
        "The length of axes doesn't match the number of dimensions "
        "transformed.")

  return coeffs, axes, ndim, ndim_transform
