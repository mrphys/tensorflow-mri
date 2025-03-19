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
"""*k*-space trajectory operations.

This module contains functions to operate with *k*-space trajectories, such as
calculation of trajectories and sampling density.
"""

import math
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_nufft as tfft
from tensorflow_graphics.geometry.transformation import rotation_matrix_2d # pylint: disable=wrong-import-order
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d # pylint: disable=wrong-import-order

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import geom_ops
from tensorflow_mri.python.ops import signal_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import math_util
from tensorflow_mri.python.util import sys_util
from tensorflow_mri.python.util import tensor_util


if sys_util.is_op_library_enabled():
  _mri_ops = tf.load_op_library(
      tf.compat.v1.resource_loader.get_path_to_datafile('_mri_ops.so'))


@api_util.export("sampling.density_grid")
def density_grid(shape,
                 inner_density=1.0,
                 outer_density=1.0,
                 inner_cutoff=0.0,
                 outer_cutoff=1.0,
                 transition_type='linear',
                 name=None):
  """Returns a density grid.

  Creates a tensor containing a density grid. The density grid is a tensor of
  shape `shape` containing a density value for each point in the grid. The
  density of a given point can be interpreted as the probability of it being
  sampled.

  The density grid has an inner region and an outer region with constant
  densities defined by `inner_density` and `outer_density`, respectively. Both
  regions are separated by a variable-density transition region whose extent
  is determined by `inner_cutoff` and `outer_cutoff`. The density variation
  in the transition region is controlled by the `transition_type`.

  The output of this function may be passed to `random_sampling_mask` to
  generate a boolean sampling mask.

  Args:
    shape: A `tf.TensorShape` or a list of `ints`. The shape of the output
      density grid.
    inner_density: A `float` between 0.0 and 1.0. The density of the inner
      region.
    outer_density: A `float` between 0.0 and 1.0. The density of the outer
      region.
    inner_cutoff: A `float` between 0.0 and 1.0. The cutoff defining the
      limit between the inner region and the transition region.
    outer_cutoff: A `float` between 0.0 and 1.0. The cutoff defining the
      limit between the transition region and the outer region.
    transition_type: A string. The type of transition to use. Must be one of
      'linear', 'quadratic', or 'hann'.
    name: A name for this op.

  Returns:
    A tensor containing the density grid.
  """
  with tf.name_scope(name or 'density_grid'):
    shape = tf.TensorShape(shape).as_list()
    transition_type = check_util.validate_enum(
        transition_type, ['linear', 'quadratic', 'hann'],
        name='transition_type')

    vecs = [tf.linspace(-1.0, 1.0 - 2.0 / n, n) for n in shape]
    grid = array_ops.meshgrid(*vecs)
    radius = tf.norm(grid, axis=-1)

    scaled_radius = (outer_cutoff - radius) / (outer_cutoff - inner_cutoff)
    if transition_type == 'linear':
      scaled_density = scaled_radius
    elif transition_type == 'quadratic':
      scaled_density = scaled_radius ** 2
    elif transition_type == 'hann':
      scaled_density = signal_ops.hann(np.pi * (1.0 - scaled_radius))
    density = (inner_density - outer_density) * scaled_density + outer_density

    density = tf.where(radius < inner_cutoff, inner_density, density)
    density = tf.where(radius > outer_cutoff, outer_density, density)

    return density


@api_util.export("sampling.random_mask")
def random_sampling_mask(shape, density=1.0, seed=None, rng=None, name=None):
  """Returns a random sampling mask with the given density.

  Args:
    shape: A 1D integer `Tensor` or array. The shape of the output mask.
    density: A `Tensor`. A density grid. After broadcasting with `shape`,
      each point in the grid represents the probability that a given point will
      be sampled. For example, if `density` is a scalar, then each point in the
      mask will be sampled with probability `density`. A non-scalar `density`
      may be used to create variable-density sampling masks.
      `tfmri.sampling.density_grid` can be used to create density grids.
    seed: A `Tensor` of shape `[2]`. The seed for the stateless RNG. `seed` and
      `rng` may not be specified at the same time.
    rng: A `tf.random.Generator`. The stateful RNG to use. `seed` and `rng` may
      not be specified at the same time. If neither `seed` nor `rng` are
      provided, the global RNG will be used.
    name: A name for this op.

  Returns:
    A boolean tensor containing the sampling mask.

  Raises:
    ValueError: If both `seed` and `rng` are specified.
  """
  with tf.name_scope(name or 'sampling_mask'):
    if seed is not None and rng is not None:
      raise ValueError("Cannot provide both `seed` and `rng`.")
    counts = tf.ones(shape, dtype=density.dtype)
    if seed is not None:  # Use stateless RNG.
      mask = tf.random.stateless_binomial(shape, seed, counts, density)
    else:  # Use stateful RNG.
      rng = rng or tf.random.get_global_generator()
      mask = rng.binomial(shape, counts, density)
    return tf.cast(mask, tf.bool)


@api_util.export("sampling.radial_trajectory")
def radial_trajectory(base_resolution,
                      views=1,
                      phases=None,
                      ordering='linear',
                      angle_range='full',
                      tiny_number=7,
                      readout_os=2.0,
                      flatten_encoding_dims=False):
  """Calculate a radial trajectory.

  This function supports the following 2D ordering methods:

  * **linear**: Uniformly spaced radial views. Views are interleaved if there
    are multiple phases.
  * **golden**: Consecutive views are spaced by the golden angle (222.49
    degrees if `angle_range` is `'full'` and 111.25 degrees if `angle_range` is
    `'half'`) [1]_.
  * **golden_half**: Variant of `'golden'` in which views are spaced by 111.25
    degrees even if `angle_range` is `'full'` [1]_.
  * **tiny**: Consecutive views are spaced by the n-th tiny golden angle, where
    `n` is given by `tiny_number` [2]_. The default tiny number is 7 (47.26
    degrees if `angle_range` is `'full'` and 23.63 degrees if `angle_range` is
    `'half'`).
  * **tiny_half**: Variant of `'tiny'` in which views are spaced by a half angle
    even if `angle_range` is `'full'` [2]_ (23.63 degrees for `tiny_number`
    equal to 7).
  * **sorted**: Like `golden`, but views within each phase are sorted by their
    angle in ascending order. Can be an alternative to `'tiny'` ordering in
    applications where small angle increments are required.
  * **sorted_half**: Variant of `'sorted'` in which angles are computed on a
    half range but modified to continuously cover the full range.

  This function also supports the following 3D ordering methods:

  * **sphere_archimedean**: 3D radial trajectory ("koosh-ball"). The starting
    points of consecutive views trace an Archimedean spiral trajectory along
    the surface of a sphere, if `angle_range` is `'full'`, or a hemisphere, if
    `angle_range` is `'half'` [3]_. Views are interleaved if there are multiple
    phases.

  Args:
    base_resolution: An `int`. The base resolution, or number of pixels in the
      readout dimension.
    views: An `int`. The number of radial views per phase.
    phases: An `int`. The number of phases for cine acquisitions. If `None`,
      this is assumed to be a non-cine acquisition with no time dimension.
    ordering: A `str`. The ordering type. Must be one of: `{'linear',
      'golden', 'tiny', 'sorted', 'sorted_half', 'sphere_archimedean'}`.
    angle_range: A `str`. The range of the rotation angle. Must be one of:
      `{'full', 'half'}`. If `angle_range` is `'full'`, the full circle/sphere
      is included in the range. If `angle_range` is `'half'`, only a
      semicircle/hemisphere is included.
    tiny_number: An `int`. The tiny golden angle number. Only used if `ordering`
      is `'tiny'` or `'tiny_half'`. Must be >= 2. Defaults to 7.
    readout_os: A `float`. The readout oversampling factor. Defaults to 2.0.
    flatten_encoding_dims: A `boolean`. If `True`, the encoding dimensions are
      flattened to a single dimension. Defaults to `False`.

  Returns:
    A `Tensor` of type `float32` and shape `[views, samples, 2]` if `phases` is
    `None`, or of shape `[phases, views, samples, 2]` if `phases` is not `None`.
    `samples` is equal to `base_resolution * readout_os`. The units are
    radians/voxel, ie, values are in the range `[-pi, pi]`.

  References:
    .. [1] Winkelmann, S., Schaeffter, T., Koehler, T., Eggers, H. and
      Doessel, O. (2007), An optimal radial profile order based on the golden
      ratio for time-resolved MRI. IEEE Transactions on Medical Imaging,
      26(1): 68-76, https://doi.org/10.1109/TMI.2006.885337
    .. [2] Wundrak, S., Paul, J., Ulrici, J., Hell, E., Geibel, M.-A.,
      Bernhardt, P., Rottbauer, W. and Rasche, V. (2016), Golden ratio sparse
      MRI using tiny golden angles. Magn. Reson. Med., 75: 2372-2378.
      https://doi.org/10.1002/mrm.25831
    .. [3] Wong, S.T.S. and Roos, M.S. (1994), A strategy for sampling on a
      sphere applied to 3D selective RF pulse design. Magn. Reson. Med.,
      32: 778-784. https://doi.org/10.1002/mrm.1910320614
  """
  return _kspace_trajectory('radial',
                            {'base_resolution': base_resolution,
                             'readout_os': readout_os},
                            views=views,
                            phases=phases,
                            ordering=ordering,
                            angle_range=angle_range,
                            tiny_number=tiny_number,
                            flatten_encoding_dims=flatten_encoding_dims)


@api_util.export("sampling.spiral_trajectory")
def spiral_trajectory(base_resolution,
                      spiral_arms,
                      field_of_view,
                      max_grad_ampl,
                      min_rise_time,
                      dwell_time,
                      views=1,
                      phases=None,
                      ordering='linear',
                      angle_range='full',
                      tiny_number=7,
                      readout_os=2.0,
                      gradient_delay=0.0,
                      larmor_const=42.577478518,
                      vd_inner_cutoff=1.0,
                      vd_outer_cutoff=1.0,
                      vd_outer_density=1.0,
                      vd_type='linear',
                      flatten_encoding_dims=False):
  """Calculate a spiral trajectory.

  Args:
    base_resolution: An `int`. The base resolution, or number of pixels in the
      readout dimension.
    spiral_arms: An `int`. The number of spiral arms that a fully sampled
      k-space should be divided into.
    field_of_view: A `float`. The field of view, in mm.
    max_grad_ampl: A `float`. The maximum allowed gradient amplitude, in mT/m.
    min_rise_time: A `float`. The minimum allowed rise time, in us/(mT/m).
    dwell_time: A `float`. The digitiser's real dwell time, in us. This does not
      include oversampling. The effective dwell time (with oversampling) is
      equal to `dwell_time * readout_os`.
    views: An `int`. The number of radial views per phase.
    phases: An `int`. The number of phases for cine acquisitions. If `None`,
      this is assumed to be a non-cine acquisition with no time dimension.
    ordering: A `str`. The ordering type. Must be one of: `{'linear',
      'golden', 'tiny', 'sorted'}`.
    angle_range: A `str`. The range of the rotation angle. Must be one of:
      `{'full', 'half'}`. If `angle_range` is `'full'`, the full circle/sphere
      is included in the range. If `angle_range` is `'half'`, only a
      semicircle/hemisphere is included.
    tiny_number: An `int`. The tiny golden angle number. Only used if `ordering`
      is `'tiny'` or `'tiny_half'`. Must be >= 2. Defaults to 7.
    readout_os: A `float`. The readout oversampling factor. Defaults to 2.0.
    gradient_delay: A `float`. The system's gradient delay relative to the ADC,
      in us. Defaults to 0.0.
    larmor_const: A `float`. The Larmor constant of the imaging nucleus, in
      MHz/T. Defaults to 42.577478518 (the Larmor constant of the 1H nucleus).
    vd_inner_cutoff: Defines the inner, high-density portion of *k*-space.
      Must be between 0.0 and 1.0, where 0.0 is the center of *k*-space and 1.0
      is the edge. Between 0.0 and `vd_inner_cutoff`, *k*-space will be sampled
      at the Nyquist rate.
    vd_outer_cutoff: Defines the outer, low-density portion of *k*-space. Must
      be between 0.0 and 1.0, where 0.0 is the center of *k*-space and 1.0 is
      the edge. Between `vd_outer_cutoff` and 1.0, *k*-space will be sampled at
      a rate `vd_outer_density` times the Nyquist rate.
    vd_outer_density: Defines the sampling density in the outer portion of
      *k*-space. Must be > 0.0. Higher means more densely sampled. Multiplies
      the Nyquist rate: 1.0 means sampling at the Nyquist rate, < 1.0 means
      undersampled and > 1.0 means oversampled.
    vd_type: Defines the rate of variation of the sampling density the
      variable-density portion of *k*-space, i.e., between `vd_inner_cutoff`
      and `vd_outer_cutoff`. Must be one of `'linear'`, `'quadratic'` or
      `'hanning'`.
    flatten_encoding_dims: A `boolean`. If `True`, the encoding dimensions are
      flattened to a single dimension. Defaults to `False`.

  Returns:
    A `Tensor` of type `float32` and shape `[views, samples, 2]` if `phases` is
    `None`, or of shape `[phases, views, samples, 2]` if `phases` is not `None`.
    `samples` is equal to `base_resolution * readout_os`. The units are
    radians/voxel, ie, values are in the range `[-pi, pi]`.

  References:
    .. [1] Pipe, J.G. and Zwart, N.R. (2014), Spiral trajectory design: A
      flexible numerical algorithm and base analytical equations. Magn. Reson.
      Med, 71: 278-285. https://doi.org/10.1002/mrm.24675
  """
  return _kspace_trajectory('spiral',
                            {'base_resolution': base_resolution,
                             'spiral_arms': spiral_arms,
                             'field_of_view': field_of_view,
                             'max_grad_ampl': max_grad_ampl,
                             'min_rise_time': min_rise_time,
                             'dwell_time': dwell_time,
                             'readout_os': readout_os,
                             'gradient_delay': gradient_delay,
                             'larmor_const': larmor_const,
                             'vd_inner_cutoff': vd_inner_cutoff,
                             'vd_outer_cutoff': vd_outer_cutoff,
                             'vd_outer_density': vd_outer_density,
                             'vd_type': vd_type},
                            views=views,
                            phases=phases,
                            ordering=ordering,
                            angle_range=angle_range,
                            tiny_number=tiny_number,
                            flatten_encoding_dims=flatten_encoding_dims)


def _kspace_trajectory(traj_type,
                       waveform_params,
                       views=1,
                       phases=None,
                       ordering='linear',
                       angle_range='full',
                       tiny_number=7,
                       flatten_encoding_dims=False):
  """Calculate a k-space trajectory.

  Args:
    traj_type: A `str`. The trajectory type. Must be one of: `{'radial',
      'spiral'}`.
    waveform_params: A `dict`. Must contain the parameters needed to calculate
      the view waveform. The accepted parameters depend on the trajectory type:
      see `radial_waveform` and `spiral_waveform`.

  For the other parameters, see `radial_trajectory` or `spiral_trajectory`.

  Returns:
    A k-space trajectory for the given parameters.
  """
  # Valid orderings.
  orderings_2d = {'linear', 'golden', 'tiny', 'sorted'} # 2D, radial or spiral
  orderings_3d = set()                                  # 3D, radial or spiral
  orderings_radial_2d = {'golden_half', 'tiny_half',
                         'sorted_half'}                 # 2D, radial only
  orderings_spiral_2d = set()                           # 2D, spiral only
  orderings_radial_3d = {'sphere_archimedean'}          # 3D, radial only
  orderings_spiral_3d = set()                           # 3D, spiral only
  all_orderings = orderings_2d | orderings_3d
  if traj_type == 'radial':
    all_orderings |= orderings_radial_2d | orderings_radial_3d
  elif traj_type == 'spiral':
    all_orderings |= orderings_spiral_2d | orderings_spiral_3d

  # Check inputs.
  traj_type = check_util.validate_enum(
    traj_type, {'radial', 'spiral'}, name='traj_type')
  ordering = check_util.validate_enum(
    ordering, all_orderings, name='ordering')
  angle_range = check_util.validate_enum(
    angle_range, {'full', 'half'}, name='angle_range')

  # Is this a 3D trajectory?
  if ordering in orderings_3d | orderings_radial_3d | orderings_spiral_3d:
    rank = 3
  else:
    rank = 2

  if sys_util.is_assistant_enabled():
    if (ordering in {'golden', 'golden_half', 'sorted'} and
        not math_util.is_fibonacci_number(views)):
      fibonacci_string = ', '.join(map(str, math_util.fibonacci_sequence(10)))
      fibonacci_string += '...'
      warnings.warn(
          f"When using golden angle ordering, optimal k-space filling "
          f"is achieved when the number of views is a member of the Fibonacci "
          f"sequence: {fibonacci_string}, but the specified number ({views}) "
          f"is not a member of this sequence.")
    if (ordering in {'tiny', 'tiny_half'} and
        not math_util.is_generalized_fibonacci_number(views, tiny_number)):
      fibonacci_string = ', '.join(
          map(str, math_util.generalized_fibonacci_sequence(10, tiny_number)))
      fibonacci_string += '...'
      warnings.warn(
          f"When using tiny golden angle ordering, optimal k-space filling "
          f"is achieved when the number of views is a member of the "
          f"generalized Fibonacci sequence: {fibonacci_string}, but the "
          f"specified number ({views}) is not a member of this sequence.")

  # Calculate waveform.
  if traj_type == 'radial':
    waveform_func = radial_waveform
    waveform_params['rank'] = rank
  elif traj_type == 'spiral':
    waveform_func = spiral_waveform
  waveform = waveform_func(**waveform_params)

  # Compute angles.
  angles = _trajectory_angles(views,
                              phases=phases,
                              ordering=ordering,
                              angle_range=angle_range,
                              tiny_number=tiny_number)

  # Rotate waveform.
  if rank == 3:
    if traj_type == 'radial':
      traj = _radial_trajectory_from_spherical_coordinates(waveform, angles)
    else:
      traj = _rotate_waveform_3d(waveform, angles)
  else:
    traj = _rotate_waveform_2d(waveform, angles)

  if flatten_encoding_dims:
    traj = flatten_trajectory(traj)

  return traj


@api_util.export("sampling.radial_density")
def radial_density(base_resolution,
                   views=1,
                   phases=None,
                   ordering='linear',
                   angle_range='full',
                   tiny_number=7,
                   readout_os=2.0,
                   flatten_encoding_dims=False):
  """Calculate sampling density for radial trajectories.

  This is an exact density calculation method based on geometrical
  considerations.

  For the parameters, see `radial_trajectory`.

  Returns:
    A `Tensor` of type `float32` and shape `[views, samples]` if `phases` is
    `None`, or of shape `[phases, views, samples]` if `phases` is not `None`.
    `samples` is equal to `base_resolution * readout_os`.

  Raises:
    ValueError: If any of the input arguments are invalid.
  """
  orderings_2d = {'linear', 'golden', 'tiny', 'sorted', 'golden_half',
                  'tiny_half'}
  if ordering not in orderings_2d:
    raise ValueError(f"Ordering `{ordering}` is not implemented.")

  # Get angles.
  angles = _trajectory_angles(views, phases or 1, ordering=ordering,
                              angle_range=angle_range, tiny_number=tiny_number)

  # Compute weights.
  weights = tf.map_fn(
      lambda t: _radial_density_from_theta(base_resolution, t,
                                           readout_os=readout_os),
      angles[..., 0])

  # Remove time dimension if there are no phases.
  if phases is None:
    weights = weights[0, ...]

  density = tf.math.reciprocal(weights)

  if flatten_encoding_dims:
    density = flatten_density(density)

  return density


def _radial_density_from_theta(base_resolution, theta, readout_os=2.0):
  """Compute density compensation weights for a single radial frame.

  Args:
    base_resolution: Number of samples, including oversampling.
    theta: A 1D `Tensor` of rotation angles.
    readout_os: A `float`. The readout oversampling factor. Defaults to 2.0.

  Returns:
    A `Tensor` of shape `[views, samples]`, where `views = theta.shape`.
  """
  # See https://github.com/tensorflow/tensorflow/issues/43038
  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter

  # Oversampling.
  samples = int(base_resolution * readout_os + 0.5)

  tf.debugging.assert_rank(theta, 1, message=(
    "`theta` must be of rank 1, but received shape: {}").format(theta.shape))

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
  dists = tf.expand_dims(dists, axis=1) # For broadcasting.
  radii = tf.expand_dims(radii, axis=0) # For broadcasting.
  weights = 4.0 * radii * dists

  # Special calculation for DC component.
  echo = samples // 2
  # w[:, echo] = 1.0 / views
  weights = tf.transpose(weights)
  weights = tf.tensor_scatter_nd_update(
      weights,
      [[echo]],
      tf.ones([1, views], dtype=tf.float32) / tf.cast(views, dtype=tf.float32))
  weights = tf.transpose(weights)

  # Compensate for oversampling.
  weights /= (readout_os ** 2)

  return weights


@api_util.export("sampling.estimate_radial_density")
def estimate_radial_density(points, readout_os=2.0):
  """Estimate the sampling density of a radial *k*-space trajectory.

  This function estimates the density based on the radius of each sample, but
  does not take into account the relationships between different spokes or
  views. This function should work well as long as the spacing between radial
  spokes is reasonably uniform. If this is not the case, consider also
  `tfmri.sampling.radial_density` or `tfmri.sampling.estimate_density`.

  This function supports 2D and 3D ("koosh-ball") radial trajectories.

  .. warning::
    This function assumes that `points` represents a radial trajectory, but
    cannot verify that. If used with trajectories other than radial, it will
    not fail but the result will be invalid.

  Args:
    points: A `Tensor`. Must be one of the following types: `float32`,
      `float64`. The coordinates at which the sampling density should be
      estimated. Must have shape `[..., views, samples, rank]`. The coordinates
      should be in radians/pixel, ie, in the range `[-pi, pi]`. Must represent a
      radial trajectory.
    readout_os: A `float`. The readout oversampling factor. Defaults to 2.0.

  Returns:
    A `Tensor` of shape `[...]` containing the density of `points`.

  Raises:
    ValueError: If any of the passed inputs is invalid.
  """
  views, samples, rank = points.shape[-3:]

  if rank not in (2, 3):
    raise ValueError(
        f"Rank must be 2 or 3, but received trajectory with rank: {rank}")

  radius = tf.norm(points, axis=-1) / np.pi * (samples // 2)

  if rank == 2:
    weights = tf.where(radius < 0.5, 1.0, 4.0 * radius)
  elif rank == 3:
    weights = tf.where(radius < 0.5, 1.0, 12.0 * radius ** 2 + 1.0)

  weights /= views * (readout_os ** 2)

  return tf.math.reciprocal_no_nan(weights)


@api_util.export("sampling.radial_waveform")
def radial_waveform(base_resolution, readout_os=2.0, rank=2):
  """Calculate a radial readout waveform.

  Args:
    base_resolution: An `int`. The base resolution, or number of pixels in the
      readout dimension.
    readout_os: A `float`. The readout oversampling factor. Defaults to 2.0.
    rank: An `int`. The rank of the waveform. Must be 2 or 3. Defaults to 2.

  Returns:
    A `Tensor` of type `float32` and shape `[samples, rank]`, where `samples` is
    equal to `base_resolution * readout_os`. The units are radians/voxel, ie,
    values are in the range `[-pi, pi]`.

  Raises:
    ValueError: If any of the input arguments has an invalid value.
  """
  # See https://github.com/tensorflow/tensorflow/issues/43038
  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter

  # Number of samples with oversampling.
  samples = int(base_resolution * readout_os + 0.5)

  # Compute 1D spoke.
  waveform = tf.range(-samples // 2, samples // 2, dtype=tf.float32)
  waveform /= samples

  # Add y/z dimensions.
  waveform = tf.expand_dims(waveform, axis=1)
  if rank == 2:
    waveform = tf.pad(waveform, [[0, 0], [0, 1]])
  elif rank == 3:
    waveform = tf.pad(waveform, [[0, 0], [1, 1]])
  else:
    raise ValueError("`rank` must be 2 or 3.")

  # Scale to [-pi, pi] (radians/voxel).
  waveform *= 2.0 * math.pi

  return waveform


if sys_util.is_op_library_enabled():
  spiral_waveform = _mri_ops.spiral_waveform


def _trajectory_angles(views,
                       phases=None,
                       ordering='linear',
                       angle_range='full',
                       tiny_number=7):
  """Compute angles for k-space trajectory.

  For the parameters, see `_kspace_trajectory`.

  Returns:
    An array of angles of shape `(phases, views)` if `phases` is not
    `None`, or of shape `(views,)` otherwise.

  Raises:
    ValueError: If unexpected inputs are received.
  """
  if not isinstance(tiny_number, int) or tiny_number < 2:
    raise ValueError(
        f"`tiny_number` must be an integer >= 2. Received: {tiny_number}")

  # Constants.
  pi = math.pi
  pi2 = math.pi * 2.0
  phi = 2.0 / (1.0 + tf.sqrt(5.0))    # Golden ratio.
  phi_n = 1.0 / (phi + tiny_number)   # n-th tiny golden ratio.

  if angle_range == 'full':
    default_max = pi2
  elif angle_range == 'half':
    default_max = pi
  else:
    raise ValueError(f"Unexpected rotation range: {angle_range}")

  def _angles_2d(angle_delta, angle_max, interleave=False):
    # Compute azimuthal angles [0, 2 * pi] (full) or [0, pi] (half).
    angles = tf.range(views * (phases or 1), dtype=tf.float32)
    angles *= angle_delta
    angles %= angle_max
    if interleave:
      angles = tf.transpose(tf.reshape(angles, (views, phases or 1)))
    else:
      angles = tf.reshape(angles, (phases or 1, views))
    angles = tf.expand_dims(angles, -1)
    return angles

  # Get ordering.
  if ordering == 'linear':
    angles = _angles_2d(default_max / (views * (phases or 1)), default_max,
                        interleave=True)
  elif ordering == 'golden':
    angles = _angles_2d(phi * default_max, default_max)
  elif ordering == 'golden_half':
    angles = _angles_2d(phi * pi, default_max)
  elif ordering == 'sorted':
    angles = _angles_2d(phi * default_max, default_max)
    angles = tf.sort(angles, axis=1)
  elif ordering == 'sorted_half':
    if angle_range != 'full':
      raise ValueError("ordering='sorted_half' requires angle_range='full'")
    angles = _angles_2d(phi * pi, pi)
    angles = tf.sort(angles, axis=1)
    def _scan_fn(prev, curr):
      # curr is the list of angles for the current phase (1D vector).
      # prev is the list of angles for the previous phase (1D vector).
      n = tf.shape(curr)[0]  # length of vector, number of views
      # Angles are in range [0, pi]. Cover range [0, 2*pi] by replicating the
      # angles and adding pi.
      curr = tf.concat([curr, curr + np.pi], 0)
      # Find the index of the first angle in the current phase that is larger
      # than the last angle in the previous phase.
      start = _find_first_greater_than(curr, prev[-1])
      # Roll the list so that `start` is the first angle.
      curr = tf.roll(curr, shift=-start, axis=0)
      # Retrieve the last `n` angles.
      curr = tf.slice(curr, [0], [n])
      return curr
    angles = tf.squeeze(angles, -1)
    angles = tf.scan(_scan_fn, angles)
    angles = tf.expand_dims(angles, -1)
  elif ordering == 'tiny':
    angles = _angles_2d(phi_n * default_max, default_max)
  elif ordering == 'tiny_half':
    angles = _angles_2d(phi_n * pi, default_max)
  elif ordering == 'sphere_archimedean':
    projections = views * (phases or 1)
    full_projections = 2 * projections if angle_range == 'half' else projections
    # Computation is sensitive to floating-point errors, so we use float64 to
    # ensure sufficient accuracy.
    indices = tf.range(projections, dtype=tf.float64)
    dh = 2.0 * indices / full_projections - 1.0
    pol = math.pi - tf.math.acos(dh)
    az = tf.math.divide_no_nan(tf.constant(3.6, dtype=tf.float64),
                               tf.math.sqrt(full_projections * (1.0 - dh * dh)))
    az = tf.math.floormod(tf.math.cumsum(az), 2.0 * math.pi) # pylint: disable=no-value-for-parameter
    # Interleave the readouts.
    def _interleave(arg):
      return tf.transpose(tf.reshape(arg, (views, phases or 1)))
    pol = _interleave(pol)
    az = _interleave(az)
    angles = tf.stack([pol, az], axis=-1)
    angles = tf.cast(angles, tf.float32)
  else:
    raise ValueError(f"Unexpected ordering method: {ordering}")

  if phases is None:
    angles = angles[0, ...]

  return angles


def _spherical_to_euler(angles):
  """Convert angles from spherical coordinates to Euler angles."""
  pol = angles[..., 0]
  az = angles[..., 1]
  el = math.pi / 2.0 - pol  # Polar angle to elevation [-pi/2, pi/2].
  euler_z = az  # Azimuthal angle = rotation of base waveform about z axis.
  euler_y = el  # Elevation angle = rotation of base waveform about y axis.
  euler_x = tf.zeros_like(az)   # Base waveform is along x axis.
  return tf.stack([euler_z, euler_x, euler_y], -1)


def _rotate_waveform_2d(waveform, angles):
  """Rotate a 2D waveform.

  Args:
    waveform: The waveform to rotate. Must have shape `[N, 2]`, where `N` is the
      number of samples.
    angles: The Euler rotation angles, with shape `[A1, A1, ..., An, 1]`.

  Returns:
    Rotated waveform(s).
  """
  # Prepare for broadcasting.
  angles = tf.expand_dims(angles, -2)

  # Compute rotation matrix.
  rot_matrix = rotation_matrix_2d.from_euler(angles)

  # Add leading singleton dimensions to `waveform` to match the batch shape of
  # `angles`. This prevents a broadcasting error later.
  waveform = tf.reshape(waveform,
      tf.concat([tf.ones([tf.rank(angles) - 2], dtype=tf.int32),
                 tf.shape(waveform)], 0))

  # Apply rotation.
  return rotation_matrix_2d.rotate(waveform, rot_matrix)


def _rotate_waveform_3d(waveform, angles):
  """Rotate a 3D waveform.

  Args:
    waveform: The waveform to rotate. Must have shape `[N, 3]`, where `N` is the
      number of samples.
    angles: The Euler rotation angles, with shape `[A1, A1, ..., An, 3]`.

  Returns:
    Rotated waveform(s).
  """
  # See https://github.com/tensorflow/tensorflow/issues/43038
  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter

  # Prepare for broadcasting.
  angles = tf.expand_dims(angles, -2)

  # Compute rotation matrix.
  rot_matrix = geom_ops.euler_to_rotation_matrix_3d(angles, order='ZYX')

  # Apply rotation to trajectory.
  waveform = rotation_matrix_3d.rotate(waveform, rot_matrix)

  return waveform


def _radial_trajectory_from_spherical_coordinates(waveform, angles):
  """Create a 3D radial trajectory from spherical coordinates.

  Args:
    waveform: A `Tensor` of shape `[samples, 3]` containing a 3D radial
      waveform.
    angles: A `Tensor` of shape `[..., 2]` where `angles[..., 0]` are the polar
      angles and `angles[..., 1]` are the azimuthal angles.

  Returns:
    A `Tensor` of shape `[..., samples, 3]` containing a 3D radial trajectory
    with dimensions z, x, y.
  """
  r = waveform[..., 1]
  theta = angles[..., 0]
  phi = angles[..., 1]

  theta = tf.expand_dims(theta, -1)
  phi = tf.expand_dims(phi, -1)

  x = r * tf.cos(phi) * tf.sin(theta)
  y = r * tf.sin(phi) * tf.sin(theta)
  z = r * tf.cos(theta)

  return tf.stack([z, x, y], -1)


@api_util.export("sampling.estimate_density")
def estimate_density(points, grid_shape, method='jackson', max_iter=50):
  """Estimate the density of an arbitrary set of points.

  Args:
    points: A `Tensor`. Must be one of the following types: `float32`,
      `float64`. The coordinates at which the sampling density should be
      estimated. Must have shape `[..., M, N]`, where `M` is the number of
      points, `N` is the number of dimensions and `...` is an arbitrary batch
      shape. `N` must be 1, 2 or 3. The coordinates should be in radians/pixel,
      ie, in the range `[-pi, pi]`.
    grid_shape: A `tf.TensorShape` or list of `ints`. The shape of the image
      corresponding to this *k*-space.
    method: A `str`. The estimation algorithm to use. Must be `"jackson"`
      or `"pipe"`. Method `"pipe"` may be more accurate but it is slower.
    max_iter: Maximum number of iterations. Only relevant if `method` is
      `"pipe"`.

  Returns:
    A `Tensor` of shape `[..., M]` containing the density of `points`.

  References:
    .. [1] Jackson, J.I., Meyer, C.H., Nishimura, D.G. and Macovski, A. (1991),
      Selection of a convolution function for Fourier inversion using gridding
      (computerised tomography application). IEEE Transactions on Medical
      Imaging, 10(3): 473-478. https://doi.org/10.1109/42.97598
    .. [2] Pipe, J.G. and Menon, P. (1999), Sampling density compensation in
      MRI: Rationale and an iterative numerical solution. Magn. Reson. Med.,
      41: 179-186. https://doi.org/10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V
  """
  method = check_util.validate_enum(
      method, {'jackson', 'pipe'}, name='method')

  # We do not check inputs here, the NUFFT op will do it for us.
  batch_shape = tf.shape(points)[:-2]

  # Calculate an appropriate grid shape.
  grid_shape = tf.TensorShape(grid_shape) # Canonicalize.
  grid_shape = tf.TensorShape(
      [_next_smooth_int(2 * s) for s in grid_shape.as_list()])

  if method in ('jackson', 'pipe'):
    # Create a k-space of ones.
    ones = tf.ones(tf.concat([batch_shape, tf.shape(points)[-2:-1]], 0),
                   dtype=tensor_util.get_complex_dtype(points.dtype))

    # Spread ones to grid and interpolate back.
    density = tfft.interp(tfft.spread(ones, points, grid_shape), points)

  if method == 'pipe':

    def _cond(i, weights): # pylint: disable=unused-argument
      return i < max_iter

    def _body(i, weights):
      weights /= tfft.interp(tfft.spread(weights, points, grid_shape), points)
      return i + 1, weights

    i = tf.constant(0, dtype=tf.dtypes.int32)
    weights = tf.math.reciprocal_no_nan(density)
    _, weights = tf.while_loop(_cond, _body, [i, weights])
    density = tf.math.reciprocal_no_nan(weights)

  # Apply scaling.
  density *= np.pi

  # Get real part and make sure there are no (slightly) negative numbers.
  density = tf.math.abs(tf.math.real(density))

  # For numerical stability: set any value smaller than a threshold to 0.
  thresh = 1e-3
  density = tf.where(density < thresh, 0.0, density)

  return density


def _next_smooth_int(n):
  """Find the next even integer with prime factors no larger than 5.

  Args:
    n: An `int`.

  Returns:
    The smallest `int` that is larger than or equal to `n`, even and with no
    prime factors larger than 5.
  """
  if n <= 2:
    return 2
  if n % 2 == 1:
    n += 1    # Even.
  n -= 2      # Cancel out +2 at the beginning of the loop.
  ndiv = 2    # Dummy value that is >1.
  while ndiv > 1:
    n += 2
    ndiv = n
    while ndiv % 2 == 0:
      ndiv /= 2
    while ndiv % 3 == 0:
      ndiv /= 3
    while ndiv % 5 == 0:
      ndiv /= 5
  return n


@api_util.export("sampling.flatten_trajectory")
def flatten_trajectory(trajectory):
  """Flatten a trajectory tensor.

  Args:
    trajectory: A `Tensor`. Must have shape `[..., views, samples, ndim]`.

  Returns:
    A reshaped `Tensor` with shape `[..., views * samples, ndim]`.
  """
  batch_shape = trajectory.shape[:-3]
  views, samples, rank = trajectory.shape[-3:]
  new_shape = batch_shape + [views*samples, rank]
  return tf.reshape(trajectory, new_shape)


@api_util.export("sampling.flatten_density")
def flatten_density(density):
  """Flatten a density tensor.

  Args:
    density: A `Tensor`. Must have shape `[..., views, samples]`.

  Returns:
    A reshaped `Tensor` with shape `[..., views * samples]`.
  """
  batch_shape = density.shape[:-2]
  views, samples = density.shape[-2:]
  new_shape = batch_shape + [views*samples]
  return tf.reshape(density, new_shape)


@api_util.export("sampling.expand_trajectory")
def expand_trajectory(trajectory, samples):
  """Expands a trajectory tensor.

  Args:
    trajectory: A `Tensor`. Must have shape `[..., views * samples, ndim]`.
    samples: An `int`. The number of samples in each view.

  Returns:
    A reshaped `Tensor` with shape `[..., views, samples, ndim]`.
  """
  batch_shape = tf.shape(trajectory)[:-2]
  rank = tf.shape(trajectory)[-1]
  shape = tf.concat([batch_shape, [-1, samples, rank]], 0)
  return tf.reshape(trajectory, shape)


@api_util.export("sampling.expand_density")
def expand_density(density, samples):
  """Expands a density tensor.

  Args:
    density: A `Tensor`. Must have shape `[..., views * samples]`.
    samples: An `int`. The number of samples in each view.

  Returns:
    A reshaped `Tensor` with shape `[..., views, samples]`.
  """
  batch_shape = tf.shape(density)[:-1]
  shape = tf.concat([batch_shape, [-1, samples]], 0)
  return tf.reshape(density, shape)


def _find_first_greater_than(x, y):
  """Returns the index of the first element in `x` that is greater than `y`."""
  x = x - y
  x = tf.where(x < 0, np.inf, x)
  return tf.math.argmin(x)
