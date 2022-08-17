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
"""Signal processing operations."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util


@api_util.export("signal.hann")
def hann(arg, name=None):
  """Calculate a Hann window at the specified coordinates.

  The domain of the window is `[-pi, pi]`. Outside this range, its value is 0.
  The center of the window is at 0.

  Args:
    arg: Input tensor.
    name: Name to use for the scope.

  Returns:
    The value of a Hann window at the specified coordinates.
  """
  with tf.name_scope(name or 'hann'):
    return _raised_cosine(arg, 0.5, 0.5)


@api_util.export("signal.hamming")
def hamming(arg, name=None):
  """Calculate a Hamming window at the specified coordinates.

  The domain of the window is `[-pi, pi]`. Outside this range, its value is 0.
  The center of the window is at 0.

  Args:
    arg: Input tensor.
    name: Name to use for the scope.

  Returns:
    The value of a Hamming window at the specified coordinates.
  """
  with tf.name_scope(name or 'hamming'):
    return _raised_cosine(arg, 0.54, 0.46)


def _raised_cosine(arg, a, b):
  """Helper function for computing a raised cosine window.

  Args:
    arg: Input tensor.
    a: The alpha parameter to the raised cosine filter.
    b: The beta parameter to the raised cosine filter.

  Returns:
    A `Tensor` of shape `arg.shape`.
  """
  arg = tf.convert_to_tensor(arg)
  return tf.where(tf.math.abs(arg) <= np.pi,
                  a - b * tf.math.cos(arg + np.pi), 0.0)


@api_util.export("signal.atanfilt")
def atanfilt(arg, cutoff=np.pi, beta=100.0, name=None):
  """Calculate an inverse tangent filter window at the specified coordinates.

  This window has infinite domain.

  Args:
    arg: Input tensor.
    cutoff: A `float` in the range [0, pi]. The cutoff frequency of the filter.
    beta: A `float`. The beta parameter of the filter.
    name: Name to use for the scope.

  Returns:
    A `Tensor` of shape `arg.shape`.

  References:
    .. [1] Pruessmann, K.P., Weiger, M., Börnert, P. and Boesiger, P. (2001),
      Advances in sensitivity encoding with arbitrary k-space trajectories.
      Magn. Reson. Med., 46: 638-651. https://doi.org/10.1002/mrm.1241
  """
  with tf.name_scope(name or 'atanfilt'):
    arg = tf.math.abs(tf.convert_to_tensor(arg))
    return 0.5 + (1.0 / np.pi) * tf.math.atan(beta * (cutoff - arg) / cutoff)


@api_util.export("signal.rect")
def rect(arg, cutoff=np.pi, name=None):
  """Returns the rectangular function.

  The rectangular function is defined as:

  .. math::
    \operatorname{rect}(x) = \Pi(t) =
      \left\{\begin{array}{rl}
        0, & \text{if } |x| > \pi \\
        \frac{1}{2}, & \text{if } |x| = \pi \\
        1, & \text{if } |x| < \pi.
      \end{array}\right.

  Args:
    arg: The input `tf.Tensor`.
    cutoff: A scalar `tf.Tensor` in the range `[0, pi]`.
      The cutoff frequency of the filter.
    name: Name to use for the scope.

  Returns:
    A `tf.Tensor` with the same shape and type as `arg`.
  """
  with tf.name_scope(name or 'rect'):
    arg = tf.convert_to_tensor(arg)
    one = tf.constant(1.0, dtype=arg.dtype)
    zero = tf.constant(0.0, dtype=arg.dtype)
    half = tf.constant(0.5, dtype=arg.dtype)
    return tf.where(tf.math.abs(arg) == cutoff,
                    half, tf.where(tf.math.abs(arg) < cutoff, one, zero))


@api_util.export("signal.separable_window")
def separable_window(func):
  """Returns a function that computes a separable window.

  This function creates a separable N-D filters as the outer product of 1D
  filters along different dimensions.

  Args:
    func: A 1D window function. Must have signature `func(x, *args, **kwargs)`.

  Returns:
    A function that computes a separable window. Has signature
    `func(x, *args, **kwargs)`, where `x` is a `tf.Tensor` of shape `[..., N]`
    and each element of `args` and `kwargs is a `tf.Tensor` of shape `[N, ...]`,
    which will be unpacked along the first dimension.
  """
  def wrapper(x, *args, **kwargs):
    # Convert each input to a tensor.
    args = tuple(tf.convert_to_tensor(arg) for arg in args)
    kwargs = {k: tf.convert_to_tensor(v) for k, v in kwargs.items()}
    def fn(accumulator, current):
      index, value = accumulator
      args, kwargs = current
      return index + 1, value * func(x[..., index], *args, **kwargs)
    initializer = tf.constant(1.0, dtype=x.dtype)
    _, out = tf.foldl(fn, (args, kwargs), initializer=(0, initializer))
    return out
  return wrapper


@api_util.export("signal.filter_kspace")
def filter_kspace(kspace,
                  trajectory=None,
                  filter_fn='hamming',
                  filter_rank=None,
                  filter_kwargs=None,
                  separable=False,
                  name=None):
  """Filter *k*-space.

  Multiplies *k*-space by a filtering function.

  Args:
    kspace: A `Tensor` of any shape. The input *k*-space.
    trajectory: A `Tensor` of shape `kspace.shape + [N]`, where `N` is the
      number of spatial dimensions. If `None`, `kspace` is assumed to be
      Cartesian.
    filter_fn: A `str` (one of `'rect'`, `'hamming'`, `'hann'` or `'atanfilt'`)
      or a callable that accepts a coordinates array and returns corresponding
      filter values. The passed coordinates array will have shape `kspace.shape`
      if `separable=False` and `[*kspace.shape, N]` if `separable=True`.
    filter_rank: An `int`. The rank of the filter. Only relevant if *k*-space is
      Cartesian. Defaults to `kspace.shape.rank`.
    filter_kwargs: A `dict`. Additional keyword arguments to pass to the
      filtering function.
    separable: A `boolean`. If `True`, the input *k*-space will be filtered
      using an N-D separable window instead of a circularly symmetric window.
      If `filter_fn` has one of the default string values, the function is
      automatically made separable. If `filter_fn` is a custom callable, it is
      the responsibility of the user to ensure that the passed callable is
      appropriate.
    name: Name to use for the scope.

  Returns:
    A `Tensor` of shape `kspace.shape`. The filtered *k*-space.
  """
  with tf.name_scope(name or 'filter_kspace'):
    kspace = tf.convert_to_tensor(kspace)
    if trajectory is not None:
      kspace, trajectory = check_util.verify_compatible_trajectory(
          kspace, trajectory)

    # Make a "trajectory" for Cartesian k-spaces.
    is_cartesian = trajectory is None
    if is_cartesian:
      filter_rank = filter_rank or kspace.shape.rank
      vecs = tf.TensorArray(dtype=kspace.dtype.real_dtype,
                            size=filter_rank,
                            infer_shape=False,
                            clear_after_read=False)
      for i in range(-filter_rank, 0):
        size = tf.shape(kspace)[i]
        pi = tf.cast(np.pi, kspace.dtype.real_dtype)
        low = -pi
        high = pi - (2.0 * pi / tf.cast(size, kspace.dtype.real_dtype))
        vecs = vecs.write(i + filter_rank, tf.linspace(low, high, size))
      trajectory = array_ops.dynamic_meshgrid(vecs)

    # For non-separable filters, use the frequency magnitude (circularly
    # symmetric filter).
    if not separable:
      trajectory = tf.norm(trajectory, axis=-1)

    if not callable(filter_fn):
      # filter_fn not a callable, so should be an enum value. Get the
      # corresponding function.
      filter_fn = check_util.validate_enum(
          filter_fn, valid_values={'rect', 'hamming', 'hann', 'atanfilt'},
          name='filter_fn')
      filter_fn = {
          'rect': rect,
          'hamming': hamming,
          'hann': hann,
          'atanfilt': atanfilt
      }[filter_fn]

      if separable:
        # The above functions are 1D. If `separable` is `True`, make them N-D
        # by wrapping them with `separable_window`.
        filter_fn = separable_window(filter_fn)

    filter_kwargs = filter_kwargs or {}  # Make sure it's a dict.
    filter_values = filter_fn(trajectory, **filter_kwargs)
    return kspace * tf.cast(filter_values, kspace.dtype)


@api_util.export("signal.crop_kspace")
def crop_kspace(kspace, trajectory=None, cutoff=None, mode='low_pass'):  # pylint: disable=missing-raises-doc
  """Crop *k*-space.

  Crops all frequencies above or below the specified frequency.

  Args:
    kspace: A `Tensor` of any shape. The input *k*-space.
    trajectory: A `Tensor` of shape `kspace.shape + [N]`, where `N` is the
      number of spatial dimensions.
    cutoff: A `float` between `-pi` and `pi`. The cutoff frequency.
    mode: A `str`. Must be one of `low_pass` or `high_pass`.

  Returns:
    A `Tensor`. The cropped *k*-space.
  """
  # TODO: add support for Cartesian *k*-space.
  if trajectory is None:
    raise ValueError('`trajectory` must be specified.')
  mode = check_util.validate_enum(mode, {'low_pass', 'high_pass'}, 'mode')
  if cutoff is None:
    cutoff = 0.0 if mode == 'high_pass' else np.pi
  traj_norm = tf.norm(trajectory, axis=-1)
  if mode == 'low_pass':
    mask = traj_norm < cutoff
  elif mode == 'high_pass':
    mask = traj_norm > cutoff
  filt_kspace = tf.gather_nd(kspace, tf.where(mask))
  filt_traj = tf.gather_nd(trajectory, tf.where(mask))
  return filt_kspace, filt_traj
