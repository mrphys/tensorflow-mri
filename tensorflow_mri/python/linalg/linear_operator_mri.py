# Copyright 2021 The TensorFlow MRI Authors. All Rights Reserved.
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
"""MRI linear operator."""

import warnings

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_nufft
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import tensor_util


_WARNED_IGNORED_BATCH_DIMENSIONS = {}


@api_util.export("linalg.LinearOperatorMRI")
@linear_operator.make_composite_tensor
class LinearOperatorMRI(linear_operator.LinearOperator):  # pylint: disable=abstract-method
  r"""Linear operator acting like an MRI measurement system.

  The MRI operator, $A$, maps a [batch of] images, $x$ to a
  [batch of] measurement data (*k*-space), $b$.

  $$
  A x = b
  $$

  This object may represent an undersampled MRI operator and supports
  Cartesian and non-Cartesian *k*-space sampling. The user may provide a
  sampling `mask` to represent an undersampled Cartesian operator, or a
  `trajectory` to represent a non-Cartesian operator.

  This object may represent a multicoil MRI operator by providing coil
  `sensitivities`. Note that `mask`, `trajectory` and `density` should never
  have a coil dimension, including in the case of multicoil imaging. The coil
  dimension will be handled automatically.

  The domain shape of this operator is `extra_shape + image_shape`. The range
  of this operator is `extra_shape + [num_coils] + image_shape`, for
  Cartesian imaging, or `extra_shape + [num_coils] + [num_samples]`, for
  non-Cartesian imaging. `[num_coils]` is optional and only present for
  multicoil operators. This operator supports batches of images and will
  vectorize operations when possible.

  Args:
    image_shape: A 1D integer `tf.Tensor`. The shape of the images
      that this operator acts on. Must have length 2 or 3.
    extra_shape: An optional 1D integer `tf.Tensor`. Additional
      dimensions that should be included within the operator domain. Note that
      `extra_shape` is not needed to reconstruct independent batches of images.
      However, it is useful when this operator is used as part of a
      reconstruction that performs computation along non-spatial dimensions,
      e.g. for temporal regularization. Defaults to `None`.
    mask: An optional `tf.Tensor` of type `tf.bool`. The sampling mask. Must
      have shape `[..., *S]`, where `S` is the `image_shape` and `...` is
      the batch shape, which can have any number of dimensions. If `mask` is
      passed, this operator represents an undersampled MRI operator.
    trajectory: An optional `tf.Tensor` of type `float32` or `float64`. Must
      have shape `[..., M, N]`, where `N` is the rank (number of spatial
      dimensions), `M` is the number of samples in the encoded space and `...`
      is the batch shape, which can have any number of dimensions. If
      `trajectory` is passed, this operator represents a non-Cartesian MRI
      operator.
    density: An optional `tf.Tensor` of type `float32` or `float64`. The
      sampling densities. Must have shape `[..., M]`, where `M` is the number of
      samples and `...` is the batch shape, which can have any number of
      dimensions. This input is only relevant for non-Cartesian MRI operators.
      If passed, the non-Cartesian operator will include sampling density
      compensation. If `None`, the operator will not perform sampling density
      compensation.
    sensitivities: An optional `tf.Tensor` of type `complex64` or `complex128`.
      The coil sensitivity maps. Must have shape `[..., C, *S]`, where `S`
      is the `image_shape`, `C` is the number of coils and `...` is the batch
      shape, which can have any number of dimensions.
    phase: An optional `tf.Tensor` of type `float32` or `float64`. A phase
      estimate for the image. If provided, this operator will be
      phase-constrained.
    fft_norm: FFT normalization mode. Must be `None` (no normalization)
      or `'ortho'`. Defaults to `'ortho'`.
    sens_norm: A `boolean`. Whether to normalize coil sensitivities. Defaults to
      `True`.
    intensity_correction: A `boolean`. Whether to correct for overall receiver
      coil sensitivity. Defaults to `True`. Has no effect if `sens_norm` is also
      `True`.
    dynamic_domain: A `str`. The domain of the dynamic dimension, if present.
      Must be one of `'time'` or `'frequency'`. May only be provided together
      with a non-scalar `extra_shape`. The dynamic dimension is the last
      dimension of `extra_shape`. The `'time'` mode (default) should be
      used for regular dynamic reconstruction. The `'frequency'` mode should be
      used for reconstruction in x-f space.
    dtype: A `tf.dtypes.DType`. The dtype of this operator. Must be `complex64`
      or `complex128`. Defaults to `complex64`.
    name: An optional `str`. The name of this operator.
  """
  def __init__(self,
               image_shape,
               extra_shape=None,
               mask=None,
               trajectory=None,
               density=None,
               sensitivities=None,
               phase=None,
               fft_norm='ortho',
               sens_norm=True,
               intensity_correction=True,
               dynamic_domain=None,
               dtype=tf.complex64,
               name=None):
    # pylint: disable=invalid-unary-operand-type
    parameters = dict(
        image_shape=image_shape,
        extra_shape=extra_shape,
        mask=mask,
        trajectory=trajectory,
        density=density,
        sensitivities=sensitivities,
        phase=phase,
        fft_norm=fft_norm,
        sens_norm=sens_norm,
        intensity_correction=intensity_correction,
        dynamic_domain=dynamic_domain,
        dtype=dtype,
        name=name)
    super().__init__(dtype, name=name, parameters=parameters)

    # Set dtype.
    dtype = tf.as_dtype(dtype)
    if dtype not in (tf.complex64, tf.complex128):
      raise ValueError(
          f"`dtype` must be `complex64` or `complex128`, but got: {str(dtype)}")

    # Batch dimensions in `image_shape` and `extra_shape` are not supported.
    # However, it is convenient to allow them to have batch dimensions anyway.
    # This helps when this operator is used in Keras models, where all inputs
    # may be automatically batched. If there are any batch dimensions, we simply
    # ignore them by taking the first element. The first time this happens
    # we also emit a warning.
    image_shape = self._ignore_batch_dims_in_shape(image_shape, "image_shape")
    extra_shape = self._ignore_batch_dims_in_shape(extra_shape, "extra_shape")

    # Set image shape, rank and extra shape.
    self._image_shape_static, self._image_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(image_shape))
    self._rank = self._image_shape_static.rank
    if self._rank not in (2, 3):
      raise ValueError(f"Rank must be 2 or 3, but got: {self._rank}")
    self._image_axes = list(range(-self._rank, 0))  # pylint: disable=invalid-unary-operand-type
    if extra_shape is None:
      extra_shape = []
    self._extra_shape_static, self._extra_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(extra_shape))

    # Set initial batch shape, then update according to inputs.
    # We include the "extra" dimensions in the batch shape for now, so that
    # they are also included in the broadcasting operations below. However,
    # note that the "extra" dimensions are not in fact part of the batch shape
    # and they will be removed later.
    self._batch_shape_static = self._extra_shape_static
    self._batch_shape_dynamic = self._extra_shape_dynamic

    # Set sampling mask after checking dtype and static shape.
    if mask is not None:
      mask = tf.convert_to_tensor(mask)
      if mask.dtype != tf.bool:
        raise TypeError(
            f"`mask` must have dtype `bool`, but got: {str(mask.dtype)}")
      if not mask.shape[-self._rank:].is_compatible_with(
          self._image_shape_static):
        raise ValueError(
            f"Expected the last dimensions of `mask` to be compatible with "
            f"{self._image_shape_static}], but got: {mask.shape[-self._rank:]}")
      self._batch_shape_static = tf.broadcast_static_shape(
          self._batch_shape_static, mask.shape[:-self._rank])
      self._batch_shape_dynamic = tf.broadcast_dynamic_shape(
          self._batch_shape_dynamic, tf.shape(mask)[:-self._rank])
    self._mask = mask

    # Set sampling trajectory after checking dtype and static shape.
    if trajectory is not None:
      if mask is not None:
        raise ValueError("`mask` and `trajectory` cannot be both passed.")
      trajectory = tf.convert_to_tensor(trajectory)
      if trajectory.dtype != dtype.real_dtype:
        raise TypeError(
            f"Expected `trajectory` to have dtype `{str(dtype.real_dtype)}`, "
            f"but got: {str(trajectory.dtype)}")
      if trajectory.shape[-1] != self._rank:
        raise ValueError(
            f"Expected the last dimension of `trajectory` to be "
            f"{self._rank}, but got {trajectory.shape[-1]}")
      self._batch_shape_static = tf.broadcast_static_shape(
          self._batch_shape_static, trajectory.shape[:-2])
      self._batch_shape_dynamic = tf.broadcast_dynamic_shape(
          self._batch_shape_dynamic, tf.shape(trajectory)[:-2])
    self._trajectory = trajectory

    # Set sampling density after checking dtype and static shape.
    if density is not None:
      if self._trajectory is None:
        raise ValueError("`density` must be passed with `trajectory`.")
      density = tf.convert_to_tensor(density)
      if density.dtype != dtype.real_dtype:
        raise TypeError(
            f"Expected `density` to have dtype `{str(dtype.real_dtype)}`, "
            f"but got: {str(density.dtype)}")
      if density.shape[-1] != self._trajectory.shape[-2]:
        raise ValueError(
            f"Expected the last dimension of `density` to be "
            f"{self._trajectory.shape[-2]}, but got {density.shape[-1]}")
      self._batch_shape_static = tf.broadcast_static_shape(
          self._batch_shape_static, density.shape[:-1])
      self._batch_shape_dynamic = tf.broadcast_dynamic_shape(
          self._batch_shape_dynamic, tf.shape(density)[:-1])
    self._density = density

    # Set sensitivity maps after checking dtype and static shape.
    if sensitivities is not None:
      sensitivities = tf.convert_to_tensor(sensitivities)
      if sensitivities.dtype != dtype:
        raise TypeError(
            f"Expected `sensitivities` to have dtype `{str(dtype)}`, but got: "
            f"{str(sensitivities.dtype)}")
      if not sensitivities.shape[-self._rank:].is_compatible_with(
          self._image_shape_static):
        raise ValueError(
            f"Expected the last dimensions of `sensitivities` to be "
            f"compatible with {self._image_shape_static}, but got: "
            f"{sensitivities.shape[-self._rank:]}")
      self._batch_shape_static = tf.broadcast_static_shape(
          self._batch_shape_static, sensitivities.shape[:-(self._rank + 1)])
      self._batch_shape_dynamic = tf.broadcast_dynamic_shape(
          self._batch_shape_dynamic, tf.shape(sensitivities)[:-(self._rank + 1)])
    self._sensitivities = sensitivities

    if phase is not None:
      phase = tf.convert_to_tensor(phase)
      if phase.dtype != dtype.real_dtype:
        raise TypeError(
            f"Expected `phase` to have dtype `{str(dtype.real_dtype)}`, "
            f"but got: {str(phase.dtype)}")
      if not phase.shape[-self._rank:].is_compatible_with(
          self._image_shape_static):
        raise ValueError(
            f"Expected the last dimensions of `phase` to be "
            f"compatible with {self._image_shape_static}, but got: "
            f"{phase.shape[-self._rank:]}")
      self._batch_shape_static = tf.broadcast_static_shape(
          self._batch_shape_static, phase.shape[:-self._rank])
      self._batch_shape_dynamic = tf.broadcast_dynamic_shape(
          self._batch_shape_dynamic, tf.shape(phase)[:-self._rank])
    self._phase = phase

    # Set batch shapes.
    extra_dims = self._extra_shape_static.rank
    if extra_dims is None:
      raise ValueError("rank of `extra_shape` must be known statically.")
    if extra_dims > 0:
      self._batch_shape_static = self._batch_shape_static[:-extra_dims]
      self._batch_shape_dynamic = self._batch_shape_dynamic[:-extra_dims]

    # If multicoil, add coil dimension to mask, trajectory and density.
    if self._sensitivities is not None:
      if self._mask is not None:
        self._mask = tf.expand_dims(self._mask, axis=-(self._rank + 1))
      if self._trajectory is not None:
        self._trajectory = tf.expand_dims(self._trajectory, axis=-3)
      if self._density is not None:
        self._density = tf.expand_dims(self._density, axis=-2)
      if self._phase is not None:
        self._phase = tf.expand_dims(self._phase, axis=-(self._rank + 1))

    # Save some tensors for later use during computation.
    if self._mask is not None:
      self._mask_linop_dtype = tf.cast(self._mask, dtype)
    if self._density is not None:
      self._dens_weights_sqrt = tf.cast(
          tf.math.sqrt(tf.math.reciprocal_no_nan(self._density)), dtype)
    if self._phase is not None:
      self._phase_rotator = tf.math.exp(
          tf.complex(tf.constant(0.0, dtype=phase.dtype), phase))

    # Set normalization.
    self._fft_norm = check_util.validate_enum(
        fft_norm, {None, 'ortho'}, 'fft_norm')
    if self._fft_norm == 'ortho':  # Compute normalization factors.
      self._fft_norm_factor = tf.math.reciprocal(
          tf.math.sqrt(tf.cast(
              tf.math.reduce_prod(self._image_shape_dynamic), dtype)))

    # Normalize coil sensitivities.
    self._sens_norm = sens_norm
    if self._sensitivities is not None and self._sens_norm:
      self._sensitivities = math_ops.normalize_no_nan(
          self._sensitivities, axis=-(self._rank + 1))

    # Intensity correction.
    self._intensity_correction = intensity_correction
    if self._sensitivities is not None and self._intensity_correction:
      self._intensity_weights_sqrt = tf.math.reciprocal_no_nan(
          tf.math.sqrt(tf.norm(self._sensitivities, axis=-(self._rank + 1))))

    # Set dynamic domain.
    if dynamic_domain is not None and self._extra_shape.rank == 0:
      raise ValueError(
          "Argument `dynamic_domain` requires a non-scalar `extra_shape`.")
    if dynamic_domain is not None:
      self._dynamic_domain = check_util.validate_enum(
          dynamic_domain, {'time', 'frequency'}, name='dynamic_domain')
    else:
      self._dynamic_domain = None

    # This variable is used by `LinearOperatorGramMRI` to disable the NUFFT.
    self._skip_nufft = False

  def _transform(self, x, adjoint=False):
    """Transform [batch] input `x`.

    Args:
      x: A `tf.Tensor` of type `self.dtype` and shape
        `[..., *self.domain_shape]` containing images, if `adjoint` is `False`,
        or a `tf.Tensor` of type `self.dtype` and shape
        `[..., *self.range_shape]` containing *k*-space data, if `adjoint` is
        `True`.
      adjoint: A `boolean` indicating whether to apply the adjoint of the
        operator.

    Returns:
      A `tf.Tensor` of type `self.dtype` and shape `[..., *self.range_shape]`
      containing *k*-space data, if `adjoint` is `False`, or a `tf.Tensor` of
      type `self.dtype` and shape `[..., *self.domain_shape]` containing
      images, if `adjoint` is `True`.
    """
    if adjoint:
      # Apply density compensation.
      if self._density is not None and not self._skip_nufft:
        x *= self._dens_weights_sqrt

      # Apply adjoint Fourier operator.
      if self.is_non_cartesian:  # Non-Cartesian imaging, use NUFFT.
        if not self._skip_nufft:
          x = fft_ops.nufft(x, self._trajectory,
                            grid_shape=self._image_shape_dynamic,
                            transform_type='type_1',
                            fft_direction='backward')
          if self._fft_norm is not None:
            x *= self._fft_norm_factor

      else:  # Cartesian imaging, use FFT.
        if self._mask is not None:
          x *= self._mask_linop_dtype  # Undersampling.
        x = fft_ops.ifftn(x, axes=self._image_axes,
                          norm=self._fft_norm or 'forward', shift=True)

      # Apply coil combination.
      if self.is_multicoil:
        x *= tf.math.conj(self._sensitivities)
        x = tf.math.reduce_sum(x, axis=-(self._rank + 1))

      # Maybe remove phase from image.
      if self.is_phase_constrained:
        x *= tf.math.conj(self._phase_rotator)
        x = tf.cast(tf.math.real(x), self.dtype)

      # Apply intensity correction.
      if self.is_multicoil and self._intensity_correction:
        x *= self._intensity_weights_sqrt

      # Apply FFT along dynamic axis, if necessary.
      if self.is_dynamic and self.dynamic_domain == 'frequency':
        x = fft_ops.fftn(x, axes=[self.dynamic_axis],
                         norm='ortho', shift=True)

    else:  # Forward operator.

      # Apply IFFT along dynamic axis, if necessary.
      if self.is_dynamic and self.dynamic_domain == 'frequency':
        x = fft_ops.ifftn(x, axes=[self.dynamic_axis],
                          norm='ortho', shift=True)

      # Apply intensity correction.
      if self.is_multicoil and self._intensity_correction:
        x *= self._intensity_weights_sqrt

      # Add phase to real-valued image if reconstruction is phase-constrained.
      if self.is_phase_constrained:
        x = tf.cast(tf.math.real(x), self.dtype)
        x *= self._phase_rotator

      # Apply sensitivity modulation.
      if self.is_multicoil:
        x = tf.expand_dims(x, axis=-(self._rank + 1))
        x *= self._sensitivities

      # Apply Fourier operator.
      if self.is_non_cartesian:  # Non-Cartesian imaging, use NUFFT.
        if not self._skip_nufft:
          x = fft_ops.nufft(x, self._trajectory,
                            transform_type='type_2',
                            fft_direction='forward')
          if self._fft_norm is not None:
            x *= self._fft_norm_factor

      else:  # Cartesian imaging, use FFT.
        x = fft_ops.fftn(x, axes=self._image_axes,
                         norm=self._fft_norm or 'backward', shift=True)
        if self._mask is not None:
          x *= self._mask_linop_dtype  # Undersampling.

      # Apply density compensation.
      if self._density is not None and not self._skip_nufft:
        x *= self._dens_weights_sqrt

    return x

  def _preprocess(self, x, adjoint=False):
    if adjoint:
      if self._density is not None:
        x *= self._dens_weights_sqrt
    else:
      raise NotImplementedError(
          "`_preprocess` not implemented for forward transform.")
    return x

  def _postprocess(self, x, adjoint=False):
    if adjoint:
      # Apply temporal Fourier operator, if necessary.
      if self.is_dynamic and self.dynamic_domain == 'frequency':
        x = fft_ops.ifftn(x, axes=[self.dynamic_axis],
                          norm='ortho', shift=True)

      # Apply intensity correction, if necessary.
      if self.is_multicoil and self._intensity_correction:
        x *= self._intensity_weights_sqrt
    else:
      raise NotImplementedError(
          "`_postprocess` not implemented for forward transform.")
    return x

  def _domain_shape(self):
    """Returns the static shape of the domain space of this operator."""
    return self._extra_shape_static.concatenate(self._image_shape_static)

  def _domain_shape_tensor(self):
    """Returns the dynamic shape of the domain space of this operator."""
    return tf.concat([self._extra_shape_dynamic, self._image_shape_dynamic], 0)

  def _range_shape(self):
    """Returns the shape of the range space of this operator."""
    if self.is_cartesian:
      range_shape = self._image_shape_static.as_list()
    else:
      range_shape = [self._trajectory.shape[-2]]
    if self.is_multicoil:
      range_shape = [self.num_coils] + range_shape
    return self._extra_shape_static.concatenate(range_shape)

  def _range_shape_tensor(self):
    if self.is_cartesian:
      range_shape = self._image_shape_dynamic
    else:
      range_shape = [tf.shape(self._trajectory)[-2]]
    if self.is_multicoil:
      range_shape = tf.concat([[self.num_coils_tensor()], range_shape], 0)
    return tf.concat([self._extra_shape_dynamic, range_shape], 0)

  def _batch_shape(self):
    """Returns the static batch shape of this operator."""
    return self._batch_shape_static

  def _batch_shape_tensor(self):
    """Returns the dynamic batch shape of this operator."""
    return self._batch_shape_dynamic

  @property
  def image_shape(self):
    """The image shape."""
    return self._image_shape_static

  def image_shape_tensor(self):
    """The image shape as a tensor."""
    return self._image_shape_dynamic

  @property
  def rank(self):
    """The number of spatial dimensions."""
    return self._rank

  @property
  def trajectory(self):
    """The k-space trajectory.

    Returns `None` for Cartesian imaging.
    """
    return self._trajectory

  @property
  def is_cartesian(self):
    """Whether this is a Cartesian MRI operator."""
    return self._trajectory is None

  @property
  def is_non_cartesian(self):
    """Whether this is a non-Cartesian MRI operator."""
    return self._trajectory is not None

  @property
  def is_multicoil(self):
    """Whether this is a multicoil MRI operator."""
    return self._sensitivities is not None

  @property
  def is_phase_constrained(self):
    """Whether this is a phase-constrained MRI operator."""
    return self._phase is not None

  @property
  def is_dynamic(self):
    """Whether this is a dynamic MRI operator."""
    return self._dynamic_domain is not None

  @property
  def dynamic_domain(self):
    """The dynamic domain of this operator."""
    return self._dynamic_domain

  @property
  def dynamic_axis(self):
    """The dynamic axis of this operator."""
    return -(self._rank + 1) if self.is_dynamic else None

  @property
  def num_coils(self):
    """The number of coils, computed statically."""
    if self._sensitivities is None:
      return None
    return self._sensitivities.shape[-(self._rank + 1)]

  def num_coils_tensor(self):
    """The number of coils, computed dynamically."""
    if self._sensitivities is None:
      return tf.convert_to_tensor(-1, dtype=tf.int32)
    return tf.shape(self._sensitivities)[-(self._rank + 1)]

  def _ignore_batch_dims_in_shape(self, shape, argname):
    if shape is None:
      return None
    shape = tf.convert_to_tensor(shape, dtype=tf.int32)
    if shape.shape.rank == 2:
      warned = _WARNED_IGNORED_BATCH_DIMENSIONS.get(argname, False)
      if not warned:
        _WARNED_IGNORED_BATCH_DIMENSIONS[argname] = True
        warnings.warn(
            f"Operator {self.name} got a batched `{argname}` argument. "
            f"It is not possible to process images with "
            f"different shapes in the same batch. "
            f"If the input batch has more than one element, "
            f"only the first shape will be used. "
            f"It is up to you to verify if this behavior is correct.")
      return tf.ensure_shape(shape[0], shape.shape[1:])
    return shape

  @property
  def _composite_tensor_fields(self):
    return ("image_shape",
            "extra_shape",
            "mask",
            "trajectory",
            "density",
            "sensitivities",
            "phase")

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("image_shape", "extra_shape")


@api_util.export("linalg.LinearOperatorGramMRI")
class LinearOperatorGramMRI(LinearOperatorMRI):  # pylint: disable=abstract-method
  """Linear operator representing the Gram matrix of an MRI measurement system.

  If $A$ is a `tfmri.linalg.LinearOperatorMRI`, then this ooperator
  represents the matrix $G = A^H A$.

  In certain circumstances, this operator may be able to apply the matrix
  $G$ more efficiently than the composition $G = A^H A$ using
  `tfmri.linalg.LinearOperatorMRI` objects.

  Args:
    image_shape: A 1D integer `tf.Tensor`. The shape of the images
      that this operator acts on. Must have length 2 or 3.
    extra_shape: An optional 1D integer `tf.Tensor`. Additional
      dimensions that should be included within the operator domain. Note that
      `extra_shape` is not needed to reconstruct independent batches of images.
      However, it is useful when this operator is used as part of a
      reconstruction that performs computation along non-spatial dimensions,
      e.g. for temporal regularization. Defaults to `None`.
    mask: An optional `tf.Tensor` of type `tf.bool`. The sampling mask. Must
      have shape `[..., *S]`, where `S` is the `image_shape` and `...` is
      the batch shape, which can have any number of dimensions. If `mask` is
      passed, this operator represents an undersampled MRI operator.
    trajectory: An optional `tf.Tensor` of type `float32` or `float64`. Must
      have shape `[..., M, N]`, where `N` is the rank (number of spatial
      dimensions), `M` is the number of samples in the encoded space and `...`
      is the batch shape, which can have any number of dimensions. If
      `trajectory` is passed, this operator represents a non-Cartesian MRI
      operator.
    density: An optional `tf.Tensor` of type `float32` or `float64`. The
      sampling densities. Must have shape `[..., M]`, where `M` is the number of
      samples and `...` is the batch shape, which can have any number of
      dimensions. This input is only relevant for non-Cartesian MRI operators.
      If passed, the non-Cartesian operator will include sampling density
      compensation. If `None`, the operator will not perform sampling density
      compensation.
    sensitivities: An optional `tf.Tensor` of type `complex64` or `complex128`.
      The coil sensitivity maps. Must have shape `[..., C, *S]`, where `S`
      is the `image_shape`, `C` is the number of coils and `...` is the batch
      shape, which can have any number of dimensions.
    phase: An optional `tf.Tensor` of type `float32` or `float64`. A phase
      estimate for the image. If provided, this operator will be
      phase-constrained.
    fft_norm: FFT normalization mode. Must be `None` (no normalization)
      or `'ortho'`. Defaults to `'ortho'`.
    sens_norm: A `boolean`. Whether to normalize coil sensitivities. Defaults to
      `True`.
    dynamic_domain: A `str`. The domain of the dynamic dimension, if present.
      Must be one of `'time'` or `'frequency'`. May only be provided together
      with a non-scalar `extra_shape`. The dynamic dimension is the last
      dimension of `extra_shape`. The `'time'` mode (default) should be
      used for regular dynamic reconstruction. The `'frequency'` mode should be
      used for reconstruction in x-f space.
    toeplitz_nufft: A `boolean`. If `True`, uses the Toeplitz approach [5]
      to compute :math:`F^H F x`, where :math:`F` is the non-uniform Fourier
      operator. If `False`, the same operation is performed using the standard
      NUFFT operation. The Toeplitz approach might be faster than the direct
      approach but is slightly less accurate. This argument is only relevant
      for non-Cartesian reconstruction and will be ignored for Cartesian
      problems.
    dtype: A `tf.dtypes.DType`. The dtype of this operator. Must be `complex64`
      or `complex128`. Defaults to `complex64`.
    name: An optional `str`. The name of this operator.
  """
  def __init__(self,
               image_shape,
               extra_shape=None,
               mask=None,
               trajectory=None,
               density=None,
               sensitivities=None,
               phase=None,
               fft_norm='ortho',
               sens_norm=True,
               dynamic_domain=None,
               toeplitz_nufft=False,
               dtype=tf.complex64,
               name="LinearOperatorGramMRI"):
    super().__init__(
        image_shape,
        extra_shape=extra_shape,
        mask=mask,
        trajectory=trajectory,
        density=density,
        sensitivities=sensitivities,
        phase=phase,
        fft_norm=fft_norm,
        sens_norm=sens_norm,
        dynamic_domain=dynamic_domain,
        dtype=dtype,
        name=name
    )

    self.toeplitz_nufft = toeplitz_nufft
    if self.toeplitz_nufft and self.is_non_cartesian:
      # Create a Gram NUFFT operator with Toeplitz embedding.
      self._linop_gram_nufft = linear_operator_nufft.LinearOperatorGramNUFFT(
          image_shape, trajectory=self._trajectory, density=self._density,
          norm=fft_norm, toeplitz=True)
      # Disable NUFFT computation on base class. The NUFFT will instead be
      # performed by the Gram NUFFT operator.
      self._skip_nufft = True

  def _transform(self, x, adjoint=False):
    x = super()._transform(x)
    if self.toeplitz_nufft:
      x = self._linop_gram_nufft.transform(x)
    x = super()._transform(x, adjoint=True)
    return x

  def _range_shape(self):
    return self._domain_shape()

  def _range_shape_tensor(self):
    return self._domain_shape_tensor()
