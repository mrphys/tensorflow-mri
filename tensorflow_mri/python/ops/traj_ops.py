# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""K-space trajectory ops."""

import math

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_2d
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d


def radial_trajectory(base_resolution,
                      views=1,
                      phases=None,
                      spacing='linear',
                      domain='full',
                      readout_os=2.0):
  """Calculate a radial trajectory.

  Args:
    base_resolution: An `int`. The base resolution, or number of pixels in the
      readout dimension.
    views: An `int`. Number of radial views per k-space.
    phases: An `int`. Number of phases for cine acquisitions. If `None`, this is
      assumed to be a non-cine acquisition with no time dimension.
    spacing: A `string`. Spacing type. Must be one of: `{'linear', 'golden',
      'tiny', 'sorted'}`.
    domain: A `string`. Rotation domain. Must be one of: `{'full', 'half'}`. If
      `domain` is `'full'`, the full circle is included in the rotation domain
      (`2 * pi`). If `domain` is `'half'`, only half circle is included (`pi`).
    readout_os: A `float`. Readout oversampling factor.

  Returns:
    A `Tensor` of type `float32` and shape `[views, samples, 2]` if `phases` is
    `None`, or of shape `[phases, views, samples, 2]` if `phases` is not `None`.
    `samples` is equal to `base_resolution * readout_os`. The units are
    radians/voxel, ie, values are in the range `[-pi, pi]`.
  """
  return _kspace_trajectory('radial',
                            {'base_resolution': base_resolution,
                             'readout_os': readout_os},
                            views=views,
                            phases=phases,
                            spacing=spacing,
                            domain=domain)


def _kspace_trajectory(traj_type,
                       waveform_params,
                       views=1,
                       phases=None,
                       spacing='linear',
                       domain='full'):
  """Calculate a k-space trajectory.

  Args:
    traj_type: Trajectory type. One of {'radial', 'spiral'}.
    waveform_params: A Python dict containing the parameters to calculate
      the view waveform. The accepted parameters depends on the trajectory
      type: see `radial_waveform` and `spiral_waveform`.
    views: Number of views per k-space.
    phases: Number of phases for cine acquisitions. If None, this is assumed
      to be a non-cine acquisition with no time dimension.
    spacing: Spacing type. One of {'linear', 'golden', 'tiny', 'sorted'}.
    domain: Angle domain. One of {'full', 'half'}, to include the full
      circle (2*pi) or half circle (pi).

  Returns:
    A k-space trajectory for the given parameters.

  Raises:
    NotImplementedError: If `traj_type` is `'spiral'`.
  """
  # Check inputs.
  traj_type = _validate_enum(
    traj_type, {'radial', 'spiral'}, 'traj_type')
  spacing = _validate_enum(
    spacing, {'linear', 'golden', 'tiny', 'sorted'}, 'spacing')
  domain = _validate_enum(
    domain, {'full', 'half'}, 'domain')

  # Calculate waveform.
  if traj_type == 'radial':
    waveform_func = radial_waveform
  elif traj_type == 'spiral':
    raise NotImplementedError("Spiral trajectories are not implemented.")
  waveform = waveform_func(**waveform_params)

  # Compute angles.
  theta = _trajectory_angles(views,
                             phases=phases,
                             spacing=spacing,
                             domain=domain)

  # Rotate waveform.
  traj = _rotate_waveform_2d(waveform, theta)

  return traj


def radial_waveform(base_resolution, readout_os=2.0):
  """Calculate a radial readout waveform.

  Args:
    base_resolution: An `int`. The base resolution, or number of pixels in the
      readout dimension.
    readout_os: A `float`. Readout oversampling factor.

  Returns:
    A `Tensor` of type `float32` and shape `[samples, 2]`, where `samples` is
    equal to `base_resolution * readout_os`. The units are radians/voxel, ie,
    values are in the range `[-pi, pi]`.
  """
  # Number of samples with oversampling.
  samples = int(base_resolution * readout_os + 0.5)

  # Compute 1D spoke.
  waveform = tf.range(-samples // 2, samples // 2, dtype=tf.float32)
  waveform /= samples

  # Add y dimension.
  waveform = tf.expand_dims(waveform, axis=1)
  waveform = tf.concat([waveform, tf.zeros((samples, 1))], axis=1)

  # Scale to [-pi, pi] (radians/voxel).
  waveform *= 2.0 * math.pi

  return waveform


def radial_density(base_resolution,
                   views=1,
                   phases=None,
                   spacing='linear',
                   domain='full',
                   readout_os=2.0):
  """Calculate density compensation weights for radial trajectories.

  For the parameters, see `radial_trajectory`.

  Returns:
    A `Tensor` of type `float32` and shape `[views, samples]` if `phases` is
    `None`, or of shape `[phases, views, samples]` if `phases` is not `None`.
    `samples` is equal to `base_resolution * readout_os`.
  """
  # Get angles.
  theta = _trajectory_angles(views, phases or 1, spacing=spacing, domain=domain)

  # Oversampling.
  samples = int(base_resolution * readout_os + 0.5)

  # Compute weights.
  weights = tf.map_fn(lambda t: _radial_density_from_theta(samples, t), theta)

  # Remove time dimension if there are no phases.
  if phases is None:
    weights = weights[0, ...]

  # Scale weights (valid for 2D radial only).
  scale = (samples * views) / (samples ** 2)
  weights *= scale

  return weights


def _radial_density_from_theta(samples, theta):
  """Compute density compensation weights for a single radial frame.

  Args:
    samples: Number of samples, including oversampling.
    theta: A 1D `Tensor` of rotation angles.

  Returns:
    A `Tensor` of shape `[views, samples]`, where `views = theta.shape`.
  """
  tf.debugging.assert_rank(theta, 1, message=(
    "`theta` must be of rank 1, but received shape: {}").format(theta.shape))

  # Operate in numpy.
  theta = theta.numpy()

  # Infer number of views from number of angles.
  views = tf.size(theta)

  # Validate angles and move to range [-pi, +pi].
  theta %= (2.0 * math.pi)
  theta = tf.where(theta < 0, theta + 2.0 * math.pi, theta)
  theta = tf.where(theta > math.pi, theta - math.pi, theta)

  # Get sorting indices.
  sort_inds = tf.expand_dims(tf.argsort(theta), -1)

  # Relative weights for each spoke, based on distance.
  dists = tf.zeros(theta.shape)
  inds = tf.expand_dims(tf.range(views), -1)
  # dists[sort_inds] = (
  #   theta[sort_inds[(inds + 1) % views]] -
  #   theta[sort_inds[(inds - 1) % views]])
  dists = tf.tensor_scatter_nd_update(
    dists,
    sort_inds,
    tf.gather_nd(theta, tf.gather_nd(sort_inds, (inds + 1) % views)))
  dists = tf.tensor_scatter_nd_sub(
    dists,
    sort_inds,
    tf.gather_nd(theta, tf.gather_nd(sort_inds, (inds - 1) % views)))

  # First and last elements will be negative, so add pi.
  # dists[sort_inds[0]] += math.pi
  # dists[sort_inds[-1]] += math.pi
  dists = tf.tensor_scatter_nd_add(dists, [sort_inds[0]], [math.pi])
  dists = tf.tensor_scatter_nd_add(dists, [sort_inds[-1]], [math.pi])

  # Divide by 2 * pi, so that all distances add up to 1.
  dists /= (2.0 * math.pi)

  # Radius.
  radii = tf.abs(tf.range(-samples // 2, samples // 2, dtype=tf.float32))

  # Compute weights.
  dists = np.expand_dims(dists, axis=1) # For broadcasting.
  radii = tf.expand_dims(radii, axis=0) # For broadcasting.
  weights = 8.0 * radii * dists

  # Special calculation for DC component.
  echo = samples // 2
  # w[:, echo] = 1.0 / views
  weights = tf.transpose(weights)
  weights = tf.tensor_scatter_nd_update(
    weights,
    [[echo]],
    tf.ones([1, views], dtype=tf.float32) / tf.cast(views, dtype=tf.float32))
  weights = tf.transpose(weights)

  return weights


def _trajectory_angles(views, phases=None, spacing='linear', domain='full'):
  """Compute angles for k-space trajectory.

  Args:
    views: Number of views.
    phases: Number of phases. If None, there is no time dimension.
    spacing: Spacing type. One of {'linear', 'golden', 'tiny', 'sorted'}.
    domain: Size of domain. One of {'full', 'half'}, to include the full
      circle (2*pi) or half circle (pi).

  Returns:
    An array of angles of shape (phases, views) if `phases` is not
    None, or of shape (views,) otherwise.
  """
  # Golden ratio.
  phi = 2.0 / (1.0 + tf.sqrt(5.0))

  # Get spacing.
  if spacing == 'linear':
    delta = 1.0 / views
  elif spacing == 'golden':
    delta = phi
  elif spacing == 'sorted':
    delta = phi
  elif spacing == 'tiny':
    order = 7
    delta = 1.0 / (phi + order)

  if domain == 'full':
    theta_domain = 2.0 * math.pi
  elif domain == 'half':
    theta_domain = math.pi

  # Compute azimuthal angles.
  theta = tf.range(views * (phases or 1), dtype=tf.float32)
  theta *= delta
  theta %= 1.0
  theta -= 0.5
  theta *= theta_domain
  theta = tf.reshape(theta, (phases or 1, views))

  # For sorted GA, sort each frame.
  if spacing == 'sorted':
    theta = tf.sort(theta, axis=1)
  if phases is None:
    theta = theta[0, ...]

  return theta


def _rotate_waveform_2d(waveform, theta):
  """Rotate a 2D waveform.

  Args:
    waveform: The waveform to rotate. Must have shape `[N, 2]`, where `N` is the
      number of samples.
    theta: The azimuthal rotation angles, with shape `[A1, A1, ..., An]`.

  Returns:
    Rotated waveform(s).
  """
  # Create Euler angle array.
  euler_angles = tf.expand_dims(theta, -1)
  euler_angles = tf.expand_dims(euler_angles, -2)

  # Compute rotation matrix.
  rot_matrix = rotation_matrix_2d.from_euler(euler_angles)

  # Apply rotation to trajectory.
  return rotation_matrix_2d.rotate(waveform, rot_matrix)


def _rotate_waveform_3d(waveform, theta):
  """Rotate a 3D waveform.

  Args:
    waveform: The waveform to rotate. Must have shape `[N, 3]`, where `N` is the
      number of samples.
    theta: The azimuthal rotation angles, with shape `[A1, A1, ..., An]`.

  Returns:
    Rotated waveform(s).
  """
  # Create Euler angle array.
  euler_angles = tf.zeros(theta.shape + (2,))
  theta = tf.expand_dims(theta, -1)
  euler_angles = tf.concat([euler_angles, theta], axis=-1)
  euler_angles = tf.expand_dims(euler_angles, -2)

  # Compute rotation matrix.
  rot_matrix = rotation_matrix_3d.from_euler(euler_angles)

  # Apply rotation to trajectory.
  return rotation_matrix_3d.rotate(waveform, rot_matrix)


def _validate_enum(value, valid_values, name):
  """Validates that value is in a list of valid values.

  Args:
    value: The value to validate.
    valid_values: The list of valid values.
    name: The name of the argument being validated. This is only used to
      format error messages.

  Returns:
    A valid enum value.

  Raises:
    ValueError: If a value not in the list of valid values was passed.
  """
  if value not in valid_values:
    raise ValueError((
      "The `{}` argument must be one of {}. "
      "Received: {}").format(name, valid_values, value))
  return value
