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
from tensorflow_mri.python.util import check_util


def hann(arg, name=None):
  """Calculate a Hann window at the specified coordinates.

  Coordinates should be in the range `[-pi, pi]`. The center of the window
  is at 0.

  Args:
    arg: Input tensor.
    name: Name to use for the scope.

  Returns:
    The value of a Hann window at the specified coordinates.
  """
  with tf.name_scope(name or 'hann'):
    return _raised_cosine(arg, 0.5, 0.5)


def hamming(arg, name=None):
  """Calculate a Hamming window at the specified coordinates.

  Coordinates should be in the range `[-pi, pi]`. The center of the window
  is at 0.

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
  return a - b * tf.math.cos(arg + np.pi)


def atanfilt(arg, cutoff=np.pi, beta=100.0, name=None):
  """Calculate an inverse tangent filter window at the specified coordinates.

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
    return 0.5 + (1.0 / np.pi) * tf.math.atan(beta * (cutoff - arg) / cutoff)


def filter_kspace(kspace,
                  traj=None,
                  filter_type='hamming',
                  filter_rank=None,
                  filter_kwargs=None):
  """Filter *k*-space.

  Multiplies *k*-space by a filtering function.

  Args:
    kspace: A `Tensor` of any shape. The input *k*-space.
    traj: A `Tensor` of shape `kspace.shape + [N]`, where `N` is the number of
      spatial dimensions. If `None`, `kspace` is assumed to be Cartesian.
    filter_type: A `str`. Must be one of `"hamming"`, `"hann"` or `"atanfilt"`.
    filter_rank: An `int`. The rank of the filter. Only relevant if *k*-space is
      Cartesian. Defaults to `kspace.shape.rank`.
    filter_kwargs: A `dict`. Additional keyword arguments to pass to the
      filtering function.

  Returns:
    A `Tensor` of shape `kspace.shape`. The filtered *k*-space.
  """
  kspace = tf.convert_to_tensor(kspace)

  # Make a "trajectory" for Cartesian k-spaces.
  is_cartesian = traj is None
  if is_cartesian:
    filter_rank = filter_rank or kspace.shape.rank
    vecs = [tf.linspace(-np.pi, np.pi - (2.0 * np.pi / s), s)
            for s in kspace.shape[-filter_rank:]]  # pylint: disable=invalid-unary-operand-type
    traj = array_ops.meshgrid(*vecs)

  filter_type = check_util.validate_enum(
      filter_type, valid_values={'hamming', 'hann', 'atanfilt'},
      name='filter_type')
  filter_func = {
      'hamming': hamming,
      'hann': hann,
      'atanfilt': atanfilt
  }[filter_type]
  filter_kwargs = filter_kwargs or {}

  traj_norm = tf.norm(traj, axis=-1)
  return kspace * tf.cast(filter_func(traj_norm, **filter_kwargs), kspace.dtype)


def crop_kspace(kspace, traj=None, cutoff=None, mode='low_pass'):  # pylint: disable=missing-raises-doc
  """Crop *k*-space.

  Crops all frequencies above or below the specified frequency.

  Args:
    kspace: A `Tensor` of any shape. The input *k*-space.
    traj: A `Tensor` of shape `kspace.shape + [N]`, where `N` is the number of
      spatial dimensions.
    cutoff: A `float` between `-pi` and `pi`. The cutoff frequency.
    mode: A `str`. Must be one of `low_pass` or `high_pass`.

  Returns:
    A `Tensor`. The cropped *k*-space.
  """
  # TODO: add support for Cartesian *k*-space.
  if traj is None:
    raise ValueError('`traj` must be specified.')
  mode = check_util.validate_enum(mode, {'low_pass', 'high_pass'}, 'mode')
  if cutoff is None:
    cutoff = 0.0 if mode == 'high_pass' else np.pi
  traj_norm = tf.norm(traj, axis=-1)
  if mode == 'low_pass':
    mask = traj_norm < cutoff
  elif mode == 'high_pass':
    mask = traj_norm > cutoff
  filt_kspace = tf.gather_nd(kspace, tf.where(mask))
  filt_traj = tf.gather_nd(traj, tf.where(mask))
  return filt_kspace, filt_traj
