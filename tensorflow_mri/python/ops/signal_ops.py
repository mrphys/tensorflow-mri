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

from tensorflow_mri.python.util import check_util


def hamming(arg, name='hamming'):
  """Calculate a Hamming window at the specified coordinates.

  Coordinates should be in the range `[-pi, pi]`. The center of the window
  is at 0.

  Args:
    arg: Input tensor.
    name: Name to use for the scope.

  Returns:
    The value of a Hamming window at the specified coordinates.
  """
  return _raised_cosine(arg, 0.54, 0.46, name=name)


def _raised_cosine(arg, a, b, name=None):
  """Helper function for computing a raised cosine window.

  Args:
    arg: Input tensor.
    a: The alpha parameter to the raised cosine filter.
    b: The beta parameter to the raised cosine filter.
    name: Name to use for the scope.

  Returns:
    A `Tensor` of shape `arg.shape`.
  """
  with tf.name_scope(name):
    return a - b * tf.math.cos(arg + np.pi)


def filter_kspace(kspace, traj, filter_type='hamming'):
  """Filter *k*-space.

  Multiplies *k*-space by a filtering function.

  Args:
    kspace: A `Tensor` of any shape. The input *k*-space.
    traj: A `Tensor` of shape `kspace.shape + [N]`, where `N` is the number of
      spatial dimensions.
    filter_type: A `str`. Must be one of `"hamming"`.

  Returns:
    A `Tensor` of shape `kspace.shape`. The filtered *k*-space.
  """
  # TODO: add support for Cartesian *k*-space.
  kspace = tf.convert_to_tensor(kspace)

  filter_type = check_util.validate_enum(
      filter_type, valid_values={'hamming'}, name='filter_type')
  filter_func = {
      'hamming': hamming
  }[filter_type]

  traj_norm = tf.norm(traj, axis=-1)
  return kspace * tf.cast(filter_func(traj_norm), kspace.dtype)


def crop_kspace(kspace, traj, cutoff, mode='low_pass'):
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
  mode = check_util.validate_enum(mode, {'low_pass', 'high_pass'}, 'mode')
  traj_norm = tf.norm(traj, axis=-1)
  if mode == 'low_pass':
    mask = traj_norm < cutoff
  elif mode == 'high_pass':
    mask = traj_norm > cutoff
  filt_kspace = tf.gather_nd(kspace, tf.where(mask))
  filt_traj = tf.gather_nd(traj, tf.where(mask))
  return filt_kspace, filt_traj
