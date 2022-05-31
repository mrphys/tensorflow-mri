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

import collections.abc
import itertools

import pywt
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.util import api_util


@api_util.export("signal.dwt")
def dwt(data, wavelet, mode='symmetric', axes=None):
  """Single-level N-dimensional discrete wavelet transform (DWT).

  Args:
    data: A `tf.Tensor`.
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

    For example, for a 2D case the result will look something like this::

    .. code::

      {
        'aa': <coeffs>  # A(LL) - approx. on 1st dim, approx. on 2nd dim
        'ad': <coeffs>  # V(LH) - approx. on 1st dim, detail on 2nd dim
        'da': <coeffs>  # hi(HL) - detail on 1st dim, approx. on 2nd dim
        'dd': <coeffs>  # D(HH) - detail on 1st dim, detail on 2nd dim
      }

    For user-specified `axes`, the order of the characters in the
    dictionary keys map to the specified `axes`.

  .. _pywt.Wavelet: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#pywt.Wavelet
  .. _tf.pad: https://www.tensorflow.org/api_docs/python/tf/pad
  """
  data = tf.convert_to_tensor(data)
  rank = data.shape.rank

  # Handle complex numbers.
  if data.dtype.is_complex:
    real = dwt(data.real, wavelet, mode, axes)
    imag = dwt(data.imag, wavelet, mode, axes)
    return dict((k, tf.dtypes.complex(real, imag)) for k in real.keys())

  # Check dtype.
  if data.dtype not in (tf.float32, tf.float64):
    raise TypeError(f"Data type must be floating, but got {data.dtype}")

  # Canonicalize axes. If None, compute decomposition along all axes.
  if axes is None:
    axes = range(rank)
  axes = [ax + rank if ax < 0 else ax for ax in axes]

  # Get padding mode for each axis.
  wavelets = _wavelets_per_axis(wavelet, axes)
  modes = _modes_per_axis(mode, axes)

  coeffs = [('', data)]
  for axis, wav, mode in zip(axes, wavelets, modes):
    new_coeffs = []
    for subband, x in coeffs:
      c_a, c_d = _dwt_along_axis(x, wav, mode, axis)
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
    A `tf.Tensor` containing the signal reconstructed from the input
    coefficients.

  .. _pywt.Wavelet: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#pywt.Wavelet
  .. _tf.pad: https://www.tensorflow.org/api_docs/python/tf/pad
  """
  # drop the keys corresponding to value = None
  coeffs = {k: v for k, v in coeffs.items() if v is not None}

  # drop the keys corresponding to value = None
  coeffs = dict((k, v) for k, v in coeffs.items() if v is not None)

  # Raise error for invalid key combinations.
  coeffs = _fix_coeffs(coeffs)

  if (any(v.dtype.is_complex for v in coeffs.values())):
    real = dict((k, v.real) for k, v in coeffs.items())
    imag = dict((k, v.imag) for k, v in coeffs.items())
    return tf.dtypes.complex(idwt(real, wavelet, mode, axes),
                             idwt(imag, wavelet, mode, axes))

  # key length matches the number of axes transformed
  ndim_transform = max(len(key) for key in coeffs.keys())

  try:
    coeff_shapes = (v.shape for k, v in coeffs.items()
                    if v is not None and len(k) == ndim_transform)
    coeff_shape = next(coeff_shapes)
  except StopIteration:
    raise ValueError("`coeffs` must contain at least one non-null wavelet "
                     "band")
  if any(s != coeff_shape for s in coeff_shapes):
    raise ValueError("`coeffs` must all be of equal size (or None)")

  if axes is None:
    axes = range(ndim_transform)
    ndim = ndim_transform
  else:
    ndim = len(coeff_shape)
  axes = [a + ndim if a < 0 else a for a in axes]

  modes = _modes_per_axis(mode, axes)
  wavelets = _wavelets_per_axis(wavelet, axes)
  for key_length, (axis, wav, mode) in reversed(
      list(enumerate(zip(axes, wavelets, modes)))):
    if axis < 0 or axis >= ndim:
      raise np.AxisError("Axis greater than data dimensions")

    new_coeffs = {}
    new_keys = [''.join(coef) for coef in itertools.product('ad', repeat=key_length)]

    for key in new_keys:
      lo = coeffs.get(key + 'a', None)
      hi = coeffs.get(key + 'd', None)
      new_coeffs[key] = _idwt_along_axis(lo, hi, wav, mode, axis)
    coeffs = new_coeffs

  return coeffs['']


def _dwt_along_axis(x, wavelet, mode, axis):
  """Computes the DWT along a single axis."""
  # Move axis `axis` to last position.
  perm = list(range(x.shape.rank))
  perm = [ax for ax in perm if ax != axis] + [axis]
  x = tf.transpose(x, perm)

  # Do padding.
  pad_length = len(wavelet) - 1
  paddings = [[0, 0]] * (x.shape.rank - 1) + [[pad_length, pad_length]]
  x = tf.pad(x, paddings, mode=mode)

  # Add channel dimension.
  x = tf.expand_dims(x, axis=-1)

  # conv1d requires at least one batch dimension, so add a dummy one.
  scalar_batch = False
  if x.shape.rank == 2:
    scalar_batch = True
    x = tf.expand_dims(x, axis=0)

  # Shift the input tensor by 1.
  begin = tf.scatter_nd([[tf.rank(x) - 2]], [1], [tf.rank(x)])
  size = tf.shape(x) - begin
  x = tf.slice(x, begin, size)

  # Get filters.
  f_lo = tf.reverse(tf.reshape(wavelet.dec_lo, [-1, 1, 1]), [0])
  f_hi = tf.reverse(tf.reshape(wavelet.dec_hi, [-1, 1, 1]), [0])

  # Compute coefficients.
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


def _idwt_along_axis(a, d, wavelet, mode, axis):
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
  a = tf.pad(a, paddings)
  d = tf.pad(d, paddings)

  # Do convolution.
  a = tf.nn.conv1d(a, f_lo, 1, 'VALID')
  d = tf.nn.conv1d(d, f_hi, 1, 'VALID')

  # Keep only part of the output.
  current_length = tf.shape(a)[-2]
  begin = tf.scatter_nd([[tf.rank(a) - 2]], [(current_length - output_length) // 2], [tf.rank(a)])
  size = tf.tensor_scatter_nd_update(tf.shape(a), [[tf.rank(a) - 2]], [output_length])
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
  """Performs dyadic upsampling along an axis."""
  # Canonicalize axis.
  axis = tf.cond(axis < 0, lambda: tf.rank(x) + axis, lambda: axis)

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
    x = tf.pad(x, paddings)

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
  """
  axes = tuple(axes)
  if isinstance(wavelet, (str, pywt.Wavelet)):
    # Same wavelet on all axes.
    wavelets = [_as_wavelet(wavelet), ] * len(axes)
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


def _modes_per_axis(modes, axes):
  """Initialize mode for each axis to be transformed.

  Args:
    modes: A `str` or an iterable of `str`. If a single mode is provided, it
      will used for all axes. Otherwise one mode per axis must be provided.
    axes: The `list` of axes to be transformed.

  Returns:
    A `list` of modes equal in length to `axes`.
  """
  axes = tuple(axes)
  if isinstance(modes, str):
    # same mode on all axes
    modes = [modes] * len(axes)
  elif isinstance(modes, collections.abc.Iterable):
    if len(modes) == 1:
      modes = [modes[0]] * len(axes)
    else:
      if len(modes) != len(axes):
        raise ValueError(("The number of modes must match the number "
                          "of axes to be transformed."))
    modes = [str(mode) for mode in modes]
  else:
    raise ValueError("modes must be a str or iterable")
  return modes


def _as_wavelet(wavelet):
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


def _fix_coeffs(coeffs):
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
