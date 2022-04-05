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
"""Preprocessing layers."""

import tensorflow as tf
import tensorflow_nufft as tfft

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import tensor_util


@tf.keras.utils.register_keras_serializable(package='MRI')
class AddChannelDimension(tf.keras.layers.Layer):
  """Adds a channel dimension to input tensor.

  Args:
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def call(self, inputs):
    """Runs forward pass on the input tensor."""
    return tf.expand_dims(inputs, -1)


@tf.keras.utils.register_keras_serializable(package='MRI')
class Cast(tf.keras.layers.Layer):
  """Casts input tensor to target dtype.

  Args:
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def call(self, inputs):
    """Runs forward pass on the input tensor."""
    return tf.cast(inputs, self.dtype)


@tf.keras.utils.register_keras_serializable(package='MRI')
class ExpandDims(tf.keras.layers.Layer):
  """Insert a new axis at specified index.

  Args:
    axis: An `int` specifying the dimension index at which to expand the shape
      of input.
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self, axis, **kwargs):
    """Initializes layer."""
    super().__init__(**kwargs)
    self._axis = axis

  def call(self, inputs):
    """Runs forward pass on the input tensor."""
    return tf.expand_dims(inputs, self._axis)

  def get_config(self):
    """Gets layer configuration."""
    config = {'axis': self._axis}
    base_config = super().get_config()
    return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='MRI')
class KSpaceResampling(tf.keras.layers.Layer):
  """K-space resampling layer.

  This layer receives an image, resamples its k-space according to the
  configuration and outputs a new image.

  This layer operates in the spatial dimensions. Inputs are assumed to have
  shape `[..., *spatial_dims, channels].`

  Args:
    image_shape: An `tf.TensorShape` or a list of ints. The spatial dimensions
      of the input/output images.
    traj_type: A `str`. The type of the trajectory. Must be `radial` or
      `spiral`.
    views: An `int`. The number of radial views per phase.
    phases: An `int`. The number of phases for cine acquisitions. If `None`,
      this is assumed to be a non-cine acquisition with no time dimension.
    ordering: A `string`. The ordering type. Must be one of: `{'linear',
      'golden', 'tiny', 'sorted'}`.
    angle_range: A `string`. The range of the rotation angle. Must be one of:
      `{'full', 'half'}`. If `angle_range` is `'full'`, the full circle/sphere
      is included in the range. If `angle_range` is `'half'`, only a
      semicircle/hemisphere is included.
    readout_os: A `float`. The readout oversampling factor. Defaults to 2.0.
    spiral_arms: An `int`. The number of spiral arms that a fully sampled
      k-space should be divided into. Must be specified for spiral trajectories.
    field_of_view: A `float`. The field of view, in mm. Must be specified for
      spiral trajectories.
    max_grad_ampl: A `float`. The maximum allowed gradient amplitude, in mT/m.
      Must be specified for spiral trajectories.
    min_rise_time: A `float`. The minimum allowed rise time, in us/(mT/m). Must
      be specified for spiral trajectories.
    dwell_time: A `float`. The digitiser's real dwell time, in us. This does not
      include oversampling. The effective dwell time (with oversampling) is
      equal to `dwell_time * readout_os`. Must be specified for spiral
      trajectories.
    gradient_delay: A `float`. The system's gradient delay relative to the ADC,
      in us. Defaults to 0.0. Only relevant for spiral trajectories.
    larmor_const: A `float`. The Larmor constant of the imaging nucleus, in
      MHz/T. Defaults to 42.577478518 (the Larmor constant of the 1H nucleus).
      Only relevant for spiral trajectories.
    vd_inner_cutoff: Defines the inner, high-density portion of *k*-space.
      Must be between 0.0 and 1.0, where 0.0 is the center of *k*-space and 1.0
      is the edge. Between 0.0 and `vd_inner_cutoff`, *k*-space will be sampled
      at the Nyquist rate. Only relevant for spiral trajectories.
    vd_outer_cutoff: Defines the outer, low-density portion of *k*-space. Must
      be between 0.0 and 1.0, where 0.0 is the center of *k*-space and 1.0 is
      the edge. Between `vd_outer_cutoff` and 1.0, *k*-space will be sampled at
      a rate `vd_outer_density` times the Nyquist rate. Only relevant for spiral
      trajectories.
    vd_outer_density: Defines the sampling density in the outer portion of
      *k*-space. Must be > 0.0. Higher means more densely sampled. Multiplies
      the Nyquist rate: 1.0 means sampling at the Nyquist rate, < 1.0 means
      undersampled and > 1.0 means oversampled. Only relevant for spiral
      trajectories.
    vd_type: Defines the rate of variation of the sampling density the
      variable-density portion of *k*-space, i.e., between `vd_inner_cutoff`
      and `vd_outer_cutoff`. Must be one of `'linear'`, `'quadratic'` or
      `'hanning'`. Only relevant for spiral trajectories.
    dens_algo: A `str`. The density estimation algorithm. Must be one of
      `'geometric'`, `'radial'`, `'jackson'` or `'pipe'`. `'geometric'` supports
      2D radial trajectories only (see also `tfmri.sampling.radial_density`).
      `'radial'` supports 2D/3D radial trajectories (see also
      `tfmri.sampling.estimate_radial_density`). `'jackson'` and `'pipe'`
      support all trajectories (see also `tfmri.sampling.estimate_density`).
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self,
               image_shape,
               traj_type,
               views,
               phases=None,
               ordering='linear',
               angle_range='full',
               readout_os=2.0,
               spiral_arms=None,
               field_of_view=None,
               max_grad_ampl=None,
               min_rise_time=None,
               dwell_time=None,
               gradient_delay=0.0,
               larmor_const=42.577478518,
               vd_inner_cutoff=1.0,
               vd_outer_cutoff=1.0,
               vd_outer_density=1.0,
               vd_type='linear',
               dens_algo='jackson',
               **kwargs):
    """Initializes layer."""
    super().__init__(**kwargs)

    check_not_none_if_spiral = lambda value, name: _check_if(
        traj_type == 'spiral', _check_not_none, value, name)

    self._image_shape = tf.TensorShape(image_shape)
    self._traj_type = check_util.validate_enum(
        traj_type, {'radial', 'spiral'}, 'traj_type')
    self._views = views
    self._phases = phases
    self._ordering = ordering
    self._angle_range = angle_range
    self._readout_os = readout_os
    self._spiral_arms = check_not_none_if_spiral(
        spiral_arms, 'spiral_arms')
    self._field_of_view = check_not_none_if_spiral(
        field_of_view, 'field_of_view')
    self._max_grad_ampl = check_not_none_if_spiral(
        max_grad_ampl, 'max_grad_ampl')
    self._min_rise_time = check_not_none_if_spiral(
        min_rise_time, 'min_rise_time')
    self._dwell_time = check_not_none_if_spiral(
        dwell_time, 'dwell_time')
    self._gradient_delay = gradient_delay
    self._larmor_const = larmor_const
    self._vd_inner_cutoff = vd_inner_cutoff
    self._vd_outer_cutoff = vd_outer_cutoff
    self._vd_outer_density = vd_outer_density
    self._vd_type = vd_type
    self._dens_algo = check_util.validate_enum(
        dens_algo, {'geometric', 'radial', 'jackson', 'pipe'})

    self._rank = self._image_shape.rank
    base_resolution = self._image_shape[-2]

    self._image_size = tf.math.reciprocal(
        tf.cast(self._image_shape.num_elements(),
        tensor_util.get_complex_dtype(self.dtype)))

    if self._traj_type == 'radial':
      self._points = traj_ops.radial_trajectory(
          base_resolution,
          views=self._views,
          phases=self._phases,
          ordering=self._ordering,
          angle_range=self._angle_range,
          readout_os=self._readout_os)

      if self._dens_algo == 'geometric':
        if self._rank == 3:
          raise ValueError(
              f"Density algorithm '{self._dens_algo}' does not support 3D "
              f"radial trajectories.")
        density = traj_ops.radial_density(
            base_resolution,
            views=self._views,
            phases=self._phases,
            ordering=self._ordering,
            angle_range=self._angle_range,
            readout_os=self._readout_os)
        self._points = traj_ops.flatten_trajectory(self._points)
        density = traj_ops.flatten_density(density)
      elif self._dens_algo == 'radial':
        density = traj_ops.estimate_radial_density(self._points,
                                                   readout_os=self._readout_os)
        self._points = traj_ops.flatten_trajectory(self._points)
        density = traj_ops.flatten_density(density)
      else:
        self._points = traj_ops.flatten_trajectory(self._points)
        density = traj_ops.estimate_density(self._points, self._image_shape,
                                            method=self._dens_algo)

    elif self._traj_type == 'spiral':
      self._points = traj_ops.spiral_trajectory(
          base_resolution,
          self._spiral_arms,
          self._field_of_view,
          self._max_grad_ampl,
          self._min_rise_time,
          self._dwell_time,
          views=self._views,
          phases=self._phases,
          ordering=self._ordering,
          angle_range=self._angle_range,
          readout_os=self._readout_os,
          gradient_delay=self._gradient_delay,
          larmor_const=self._larmor_const,
          vd_inner_cutoff=self._vd_inner_cutoff,
          vd_outer_cutoff=self._vd_outer_cutoff,
          vd_outer_density=self._vd_outer_density,
          vd_type=self._vd_type)

      self._points = traj_ops.flatten_trajectory(self._points)

      if self._dens_algo not in ('jackson', 'pipe'):
        raise ValueError(
            f"Density algorithm {self._dens_algo} does not support spiral "
            f"trajectories.")

      density = traj_ops.estimate_density(self._points, self._image_shape,
                                          method=self._dens_algo)

    else:
      raise ValueError(f"Unknown trajectory type: {self._traj_type}")

    self._weights = tf.cast(tf.math.reciprocal_no_nan(density),
                            tensor_util.get_complex_dtype(self.dtype))

    traj_rank = self._points.shape[-1]
    if traj_rank != self._rank:
      raise ValueError(
          f"`image_shape` has rank {self._rank}, but the specified trajectory "
          f"has rank {traj_rank}")

  def call(self, inputs): # pylint: disable=missing-param-doc
    """Runs forward pass on the input tensor."""
    x = inputs
    input_rank = x.shape.rank
    layer_dtype = tf.as_dtype(self.dtype)

    # Move channel dimension to beginning.
    perm = [input_rank - 1] + list(range(input_rank - 1))
    x = tf.transpose(x, perm)

    # Cast real inputs to complex.
    if not layer_dtype.is_complex:
      x = tf.dtypes.complex(x, tf.constant(0.0, dtype=self.dtype))

    # Forward NUFFT.
    x = tfft.nufft(x, self._points,
                   transform_type='type_2',
                   fft_direction='forward')

    # Additional k-space processing steps. This does not do anything in this
    # class but may be used by subclasses.
    x = self.process_kspace(x)

    # Density compensation.
    x *= self._weights

    # Adjoint NUFFT.
    x = tfft.nufft(x, self._points,
                   grid_shape=self._image_shape,
                   transform_type='type_1',
                   fft_direction='backward')

    # FFT normalization.
    x *= self._image_size

    # If layer is real, cast complex tensor to real.
    if not layer_dtype.is_complex:
      x = tf.math.abs(x)

    # Move channel dimension back to the end.
    inv_perm = list(range(1, input_rank)) + [0]
    x = tf.transpose(x, inv_perm)
    return x

  def process_kspace(self, kspace):
    """Additional k-space processing.

    Subclasses that wish to do additional k-space processing should override
    this method.

    Args:
      kspace: A tensor containing the resampled k-space data.

    Returns:
      A processed k-space tensor. Has the same shape and type as `kspace`.
    """
    return kspace

  def get_config(self):
    """Gets layer configuration."""
    config = {
        'image_shape': self._image_shape,
        'traj_type': self._traj_type,
        'views': self._views,
        'phases': self._phases,
        'ordering': self._ordering,
        'angle_range': self._angle_range,
        'readout_os': self._readout_os,
        'spiral_arms': self._spiral_arms,
        'field_of_views': self._field_of_view,
        'max_grad_ampl': self._max_grad_ampl,
        'min_rise_time': self._min_rise_time,
        'dwell_time': self._dwell_time,
        'gradient_delay': self._gradient_delay,
        'larmor_const': self._larmor_const,
        'vd_inner_cutoff': self._vd_inner_cutoff,
        'vd_outer_cutoff': self._vd_outer_cutoff,
        'vd_outer_density': self._vd_outer_density,
        'vd_type': self._vd_type,
        'dens_algo': self._dens_algo
    }
    base_config = super().get_config()
    return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='MRI')
class RepeatTensor(tf.keras.layers.Layer):
  """Repeats the input.

  Returns a list with the repeated inputs.

  Args:
    repeats: An `int`. The number of times the input should be repeated.
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self, repeats, **kwargs):
    """Initializes layer."""
    super().__init__(**kwargs)
    self._repeats = repeats

  def call(self, inputs):
    """Runs forward pass on the input tensor."""
    return [inputs] * self._repeats

  def get_config(self):
    """Gets layer configuration."""
    config = {'repeats': self._repeats}
    base_config = super().get_config()
    return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='MRI')
class ResizeWithCropOrPad(tf.keras.layers.Layer):
  """Crops and/or pads to target shape.

  Pads symmetrically or crops centrally to the target shape.

  This operation is applied along the spatial dimensions. The inputs are assumed
  to have shape `[..., *spatial_dims, channels]`.

  Args:
    shape: A list of `int` or a `tf.TensorShape`. The target shape. Each
      dimension can be `None`, in which case it is left unmodified.
    padding_mode: A `str`. Must be one of `'constant'`, `'reflect'` or
      `'symmetric'`.
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self, shape, padding_mode='constant', **kwargs):
    """Initializes layer."""
    super().__init__(**kwargs)
    self._shape = shape
    self._shape_internal = [s or -1 for s in tf.TensorShape(shape).as_list()]
    self._shape_internal += [-1]
    self._padding_mode = padding_mode

  def call(self, inputs):
    """Runs forward pass on the input tensor."""
    return array_ops.resize_with_crop_or_pad(inputs, self._shape_internal,
                                             padding_mode=self._padding_mode)

  def get_config(self):
    """Gets layer configuration."""
    config = {'shape': self._shape}
    base_config = super().get_config()
    return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='MRI')
class ScaleByMinMax(tf.keras.layers.Layer):
  """Scale input into a specified range.

  Args:
    output_min: An optional `float`. The minimum value in the output tensor.
      Defaults to 0.0.
    output_max: An optional `float`. The maximum value in the output tensor.
      Defaults to 1.0.
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self, output_min=0.0, output_max=1.0, **kwargs):
    """Initializes layer."""
    super().__init__(**kwargs)
    self._output_min = output_min
    self._output_max = output_max

  def call(self, inputs):
    """Runs forward pass on the input tensor."""
    return math_ops.scale_by_min_max(inputs, self._output_min, self._output_max)

  def get_config(self):
    """Gets layer configuration."""
    config = {'output_min': self._output_min,
              'output_max': self._output_max}
    base_config = super().get_config()
    return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='MRI')
class Transpose(tf.keras.layers.Layer):
  """Transpose the input tensor.

  Args:
    perm: A list of `int`. A permutation of the dimensions of the input tensor.
    conjugate: An optional `bool`. Defaults to `False`.
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self, perm=None, conjugate=False, **kwargs):
    """Initializes layer."""
    super().__init__(**kwargs)
    self._perm = perm
    self._conjugate = conjugate

  def call(self, inputs):
    """Runs forward pass on the input tensor."""
    return tf.transpose(inputs, self._perm, conjugate=self._conjugate)

  def get_config(self):
    """Gets layer configuration."""
    config = {'perm': self._perm,
              'conjugate': self._conjugate}
    base_config = super().get_config()
    return {**base_config, **config}


def _check_not_none(value, name):
  """Checks that value is not None."""
  if value is None:
    raise ValueError(f"`{name}` must be specified.")
  return value


def _check_if(cond, fn, value, name):
  """Performs a check if condition is true."""
  return fn(value, name) if cond else value
