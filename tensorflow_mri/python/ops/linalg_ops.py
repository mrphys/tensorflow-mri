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
"""Linear algebra operations.

This module contains linear operators and solvers.
"""

import collections
import functools

import tensorflow as tf
import tensorflow_nufft as tfft

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.ops import wavelet_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import linalg_imaging
from tensorflow_mri.python.util import tensor_util


@api_util.export("linalg.LinearOperatorNUFFT")
class LinearOperatorNUFFT(linalg_imaging.LinearOperator):  # pylint: disable=abstract-method
  """Linear operator acting like a nonuniform DFT matrix.

  Args:
    domain_shape: A 1D integer `tf.Tensor`. The domain shape of this
      operator. This is usually the shape of the image but may include
      additional dimensions.
    trajectory: A `tf.Tensor` of type `float32` or `float64`. Contains the
      sampling locations or *k*-space trajectory. Must have shape
      `[..., M, N]`, where `N` is the rank (number of dimensions), `M` is
      the number of samples and `...` is the batch shape, which can have any
      number of dimensions.
    density: A `tf.Tensor` of type `float32` or `float64`. Contains the
      sampling density at each point in `trajectory`. Must have shape
      `[..., M]`, where `M` is the number of samples and `...` is the batch
      shape, which can have any number of dimensions. Defaults to `None`, in
      which case the density is assumed to be 1.0 in all locations.
    norm: A `str`. The FFT normalization mode. Must be `None` (no normalization)
      or `'ortho'`.
    name: An optional `str`. The name of this operator.

  Notes:
    In MRI, sampling density compensation is typically performed during the
    adjoint transform. However, in order to maintain the validity of the linear
    operator, this operator applies the compensation orthogonally, i.e.,
    it scales the data by the square root of `density` in both forward and
    adjoint transforms. If you are using this operator to compute the adjoint
    and wish to apply the full compensation, you can do so via the
    `precompensate` method.

    >>> import tensorflow as tf
    >>> import tensorflow_mri as tfmri
    >>> # Create some data.
    >>> image_shape = (128, 128)
    >>> image = image_ops.phantom(shape=image_shape, dtype=tf.complex64)
    >>> trajectory = tfmri.sampling.radial_trajectory(
    >>>     128, 128, flatten_encoding_dims=True)
    >>> density = tfmri.sampling.radial_density(
    >>>     128, 128, flatten_encoding_dims=True)
    >>> # Create a NUFFT operator.
    >>> linop = tfmri.linalg.LinearOperatorNUFFT(
    >>>     image_shape, trajectory=trajectory, density=density)
    >>> # Create k-space.
    >>> kspace = tfmri.signal.nufft(image, trajectory)
    >>> # This reconstructs the image applying only partial compensation
    >>> # (square root of weights).
    >>> image = linop.transform(kspace, adjoint=True)
    >>> # This reconstructs the image with full compensation.
    >>> image = linop.transform(linop.precompensate(kspace), adjoint=True)
  """
  def __init__(self,
               domain_shape,
               trajectory,
               density=None,
               norm='ortho',
               name="LinearOperatorNUFFT"):

    parameters = dict(
        domain_shape=domain_shape,
        trajectory=trajectory,
        norm=norm,
        name=name
    )

    # Get domain shapes.
    self._domain_shape_static, self._domain_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))

    # Validate the remaining inputs.
    self.trajectory = check_util.validate_tensor_dtype(
        tf.convert_to_tensor(trajectory), 'floating', 'trajectory')
    self.norm = check_util.validate_enum(norm, {None, 'ortho'}, 'norm')

    # We infer the operation's rank from the trajectory.
    self._rank_static = self.trajectory.shape[-1]
    self._rank_dynamic = tf.shape(self.trajectory)[-1]
    # The domain rank is >= the operation rank.
    domain_rank_static = self._domain_shape_static.rank
    domain_rank_dynamic = tf.shape(self._domain_shape_dynamic)[0]
    # The difference between this operation's rank and the domain rank is the
    # number of extra dims.
    extra_dims_static = domain_rank_static - self._rank_static
    extra_dims_dynamic = domain_rank_dynamic - self._rank_dynamic

    # The grid shape are the last `rank` dimensions of domain_shape. We don't
    # need the static grid shape.
    self._grid_shape = self._domain_shape_dynamic[-self._rank_dynamic:]

    # We need to do some work to figure out the batch shapes. This operator
    # could have a batch shape, if the trajectory has a batch shape. However,
    # we allow the user to include one or more batch dimensions in the domain
    # shape, if they so wish. Therefore, not all batch dimensions in the
    # trajectory are necessarily part of the batch shape.

    # The total number of dimensions in `trajectory` is equal to
    # `batch_dims + extra_dims + 2`.
    # Compute the true batch shape (i.e., the batch dimensions that are
    # NOT included in the domain shape).
    batch_dims_dynamic = tf.rank(self.trajectory) - extra_dims_dynamic - 2
    if (self.trajectory.shape.rank is not None and
        extra_dims_static is not None):
      # We know the total number of dimensions in `trajectory` and we know
      # the number of extra dims, so we can compute the number of batch dims
      # statically.
      batch_dims_static = self.trajectory.shape.rank - extra_dims_static - 2
    else:
      # We are missing at least some information, so the number of batch
      # dimensions is unknown.
      batch_dims_static = None

    self._batch_shape_dynamic = tf.shape(self.trajectory)[:batch_dims_dynamic]
    if batch_dims_static is not None:
      self._batch_shape_static = self.trajectory.shape[:batch_dims_static]
    else:
      self._batch_shape_static = tf.TensorShape(None)

    # Compute the "extra" shape. This shape includes those dimensions which
    # are not part of the NUFFT (e.g., they are effectively batch dimensions),
    # but which are included in the domain shape rather than in the batch shape.
    extra_shape_dynamic = self._domain_shape_dynamic[:-self._rank_dynamic]
    if self._rank_static is not None:
      extra_shape_static = self._domain_shape_static[:-self._rank_static]
    else:
      extra_shape_static = tf.TensorShape(None)

    # Check that the "extra" shape in `domain_shape` and `trajectory` are
    # compatible for broadcasting.
    shape1, shape2 = extra_shape_static, self.trajectory.shape[:-2]
    try:
      tf.broadcast_static_shape(shape1, shape2)
    except ValueError as err:
      raise ValueError(
          f"The \"batch\" shapes in `domain_shape` and `trajectory` are not "
          f"compatible for broadcasting: {shape1} vs {shape2}") from err

    # Compute the range shape.
    self._range_shape_dynamic = tf.concat(
        [extra_shape_dynamic, tf.shape(self.trajectory)[-2:-1]], 0)
    self._range_shape_static = extra_shape_static.concatenate(
        self.trajectory.shape[-2:-1])

    # Statically check that density can be broadcasted with trajectory.
    if density is not None:
      try:
        tf.broadcast_static_shape(self.trajectory.shape[:-1], density.shape)
      except ValueError as err:
        raise ValueError(
            f"The \"batch\" shapes in `trajectory` and `density` are not "
            f"compatible for broadcasting: {self.trajectory.shape[:-1]} vs "
            f"{density.shape}") from err
      self.density = tf.convert_to_tensor(density)
      self.weights = tf.math.reciprocal_no_nan(self.density)
      self._weights_sqrt = tf.cast(
          tf.math.sqrt(self.weights),
          tensor_util.get_complex_dtype(self.trajectory.dtype))
    else:
      self.density = None
      self.weights = None

    super().__init__(tensor_util.get_complex_dtype(self.trajectory.dtype),
                     is_non_singular=None,
                     is_self_adjoint=None,
                     is_positive_definite=None,
                     is_square=None,
                     name=name,
                     parameters=parameters)

    # Compute normalization factors.
    if self.norm == 'ortho':
      norm_factor = tf.math.reciprocal(
          tf.math.sqrt(tf.cast(tf.math.reduce_prod(self._grid_shape),
          self.dtype)))
      self._norm_factor_forward = norm_factor
      self._norm_factor_adjoint = norm_factor

  def _transform(self, x, adjoint=False):
    if adjoint:
      if self.density is not None:
        x *= self._weights_sqrt
      x = fft_ops.nufft(x, self.trajectory,
                        grid_shape=self._grid_shape,
                        transform_type='type_1',
                        fft_direction='backward')
      if self.norm is not None:
        x *= self._norm_factor_adjoint
    else:
      x = fft_ops.nufft(x, self.trajectory,
                        transform_type='type_2',
                        fft_direction='forward')
      if self.norm is not None:
        x *= self._norm_factor_forward
      if self.density is not None:
        x *= self._weights_sqrt
    return x

  def precompensate(self, x):
    if self.density is not None:
      return x * self._weights_sqrt
    return x

  def _domain_shape(self):
    return self._domain_shape_static

  def _domain_shape_tensor(self):
    return self._domain_shape_dynamic

  def _range_shape(self):
    return self._range_shape_static

  def _range_shape_tensor(self):
    return self._range_shape_dynamic

  def _batch_shape(self):
    return self._batch_shape_static

  def _batch_shape_tensor(self):
    return self._batch_shape_dynamic

  @property
  def rank(self):
    return self._rank_static

  def rank_tensor(self):
    return self._rank_dynamic


@api_util.export("linalg.LinearOperatorGramNUFFT")
class LinearOperatorGramNUFFT(LinearOperatorNUFFT):  # pylint: disable=abstract-method
  """Linear operator acting like the Gram matrix of an NUFFT operator.

  If :math:`F` is a `tfmri.linalg.LinearOperatorNUFFT`, then this operator
  applies :math:`F^H F`. This operator is self-adjoint.

  Args:
    domain_shape: A 1D integer `tf.Tensor`. The domain shape of this
      operator. This is usually the shape of the image but may include
      additional dimensions.
    trajectory: A `tf.Tensor` of type `float32` or `float64`. Contains the
      sampling locations or *k*-space trajectory. Must have shape
      `[..., M, N]`, where `N` is the rank (number of dimensions), `M` is
      the number of samples and `...` is the batch shape, which can have any
      number of dimensions.
    density: A `tf.Tensor` of type `float32` or `float64`. Contains the
      sampling density at each point in `trajectory`. Must have shape
      `[..., M]`, where `M` is the number of samples and `...` is the batch
      shape, which can have any number of dimensions. Defaults to `None`, in
      which case the density is assumed to be 1.0 in all locations.
    norm: A `str`. The FFT normalization mode. Must be `None` (no normalization)
      or `'ortho'`.
    toeplitz: A `boolean`. If `True`, uses the Toeplitz approach [1]
      to compute :math:`F^H F x`, where :math:`F` is the NUFFT operator.
      If `False`, the same operation is performed using the standard
      NUFFT operation. The Toeplitz approach might be faster than the direct
      approach but is slightly less accurate. This argument is only relevant
      for non-Cartesian reconstruction and will be ignored for Cartesian
      problems.
    name: An optional `str`. The name of this operator.

  References:
    [1] Fessler, J. A., Lee, S., Olafsson, V. T., Shi, H. R., & Noll, D. C.
      (2005). Toeplitz-based iterative image reconstruction for MRI with
      correction for magnetic field inhomogeneity. IEEE Transactions on Signal
      Processing, 53(9), 3393-3402.
  """
  def __init__(self,
               domain_shape,
               trajectory,
               density=None,
               norm='ortho',
               toeplitz=False,
               name="LinearOperatorNUFFT"):
    super().__init__(
        domain_shape=domain_shape,
        trajectory=trajectory,
        density=density,
        norm=norm,
        name=name
    )

    self.toeplitz = toeplitz
    if self.toeplitz:
      # Compute the FFT shift for adjoint NUFFT computation.
      self._fft_shift = tf.cast(self._grid_shape // 2, self.dtype.real_dtype)
      # Compute the Toeplitz kernel.
      self._toeplitz_kernel = self._compute_toeplitz_kernel()
      # Kernel shape (without batch dimensions).
      self._kernel_shape = tf.shape(self._toeplitz_kernel)[-self.rank_tensor():]

  def _transform(self, x, adjoint=False):  # pylint: disable=unused-argument
    """Applies this linear operator."""
    # This operator is self-adjoint, so `adjoint` arg is unused.
    if self.toeplitz:
      # Using specialized Toeplitz implementation.
      return self._transform_toeplitz(x)
    # Using standard NUFFT implementation.
    return super()._transform(super()._transform(x), adjoint=True)

  def _transform_toeplitz(self, x):
    """Applies this linear operator using the Toeplitz approach."""
    input_shape = tf.shape(x)
    fft_axes = tf.range(-self.rank_tensor(), 0)
    x = fft_ops.fftn(x, axes=fft_axes, shape=self._kernel_shape)
    x *= self._toeplitz_kernel
    x = fft_ops.ifftn(x, axes=fft_axes)
    x = tf.slice(x, tf.zeros([tf.rank(x)], dtype=tf.int32), input_shape)
    return x

  def _compute_toeplitz_kernel(self):
    """Computes the kernel for the Toeplitz approach."""
    trajectory = self.trajectory
    weights = self.weights
    if self.rank is None:
      raise NotImplementedError(
          f"The rank of {self.name} must be known statically.")

    if weights is None:
      # If no weights were passed, use ones.
      weights = tf.ones(tf.shape(trajectory)[:-1], dtype=self.dtype.real_dtype)
    # Cast weights to complex dtype.
    weights = tf.cast(tf.math.sqrt(weights), self.dtype)

    # Compute N-D kernel recursively. Begin with last axis.
    last_axis = self.rank - 1
    kernel = self._compute_kernel_recursive(trajectory, weights, last_axis)

    # Make sure that the kernel is symmetric/Hermitian/self-adjoint.
    kernel = self._enforce_kernel_symmetry(kernel)

    # Additional normalization by sqrt(2 ** rank). This is required because
    # we are using FFTs with twice the length of the original image.
    if self.norm == 'ortho':
      kernel *= tf.cast(tf.math.sqrt(2.0 ** self.rank), kernel.dtype)

    # Put the kernel in Fourier space.
    fft_axes = list(range(-self.rank, 0))
    fft_norm = self.norm or "backward"
    return fft_ops.fftn(kernel, axes=fft_axes, norm=fft_norm)

  def _compute_kernel_recursive(self, trajectory, weights, axis):
    """Recursively computes the kernel for the Toeplitz approach.

    This function works by computing the two halves of the kernel along each
    axis. The "left" half is computed using the input trajectory. The "right"
    half is computed using the trajectory flipped along the current axis, and
    then reversed. Then the two halves are concatenated, with a block of zeros
    inserted in between. If there is more than one axis, the process is repeated
    recursively for each axis.

    This function calls the adjoint NUFFT 2 ** N times, where N is the number
    of dimensions. NOTE: this could be optimized to 2 ** (N - 1) calls.

    Args:
      trajectory: A `tf.Tensor` containing the current *k*-space trajectory.
      weights: A `tf.Tensor` containing the current density compensation
        weights.
      axis: An `int` denoting the current axis.

    Returns:
      A `tf.Tensor` containing the kernel.

    Raises:
      NotImplementedError: If the rank of the operator is not known statically.
    """
    # Account for the batch dimensions. We do not need to do the recursion
    # for these.
    batch_dims = self.batch_shape.rank
    if batch_dims is None:
      raise NotImplementedError(
          f"The number of batch dimensions of {self.name} must be known "
          f"statically.")
    # The current axis without the batch dimensions.
    image_axis = axis + batch_dims
    if axis == 0:
      # Outer-most axis. Compute left half, then use Hermitian symmetry to
      # compute right half.
      # TODO(jmontalt): there should be a way to compute the NUFFT only once.
      kernel_left = self._nufft_adjoint(weights, trajectory)
      flippings = tf.tensor_scatter_nd_update(
          tf.ones([self.rank_tensor()]), [[axis]], [-1])
      kernel_right = self._nufft_adjoint(weights, trajectory * flippings)
    else:
      # We still have two or more axes to process. Compute left and right kernels
      # by calling this function recursively. We call ourselves twice, first
      # with current frequencies, then with negated frequencies along current
      # axes.
      kernel_left = self._compute_kernel_recursive(
          trajectory, weights, axis - 1)
      flippings = tf.tensor_scatter_nd_update(
          tf.ones([self.rank_tensor()]), [[axis]], [-1])
      kernel_right = self._compute_kernel_recursive(
          trajectory * flippings, weights, axis - 1)

    # Remove zero frequency and reverse.
    kernel_right = tf.reverse(array_ops.slice_along_axis(
        kernel_right, image_axis, 1, tf.shape(kernel_right)[image_axis] - 1),
        [image_axis])

    # Create block of zeros to be inserted between the left and right halves of
    # the kernel.
    zeros_shape = tf.concat([
        tf.shape(kernel_left)[:image_axis], [1],
        tf.shape(kernel_left)[(image_axis + 1):]], 0)
    zeros = tf.zeros(zeros_shape, dtype=kernel_left.dtype)

    # Concatenate the left and right halves of kernel, with a block of zeros in
    # the middle.
    kernel = tf.concat([kernel_left, zeros, kernel_right], image_axis)
    return kernel

  def _nufft_adjoint(self, x, trajectory=None):
    """Applies the adjoint NUFFT operator.

    We use this instead of `super()._transform(x, adjoint=True)` because we
    need to be able to change the trajectory and to apply an FFT shift.

    Args:
      x: A `tf.Tensor` containing the input data (typically the weights or
        ones).
      trajectory: A `tf.Tensor` containing the *k*-space trajectory, which
        may have been flipped and therefore different from the original. If
        `None`, the original trajectory is used.

    Returns:
      A `tf.Tensor` containing the result of the adjoint NUFFT.
    """
    # Apply FFT shift.
    x *= tf.math.exp(tf.dtypes.complex(
        tf.constant(0, dtype=self.dtype.real_dtype),
        tf.math.reduce_sum(trajectory * self._fft_shift, -1)))
    # Temporarily update trajectory.
    if trajectory is not None:
      temp = self.trajectory
      self.trajectory = trajectory
    x = super()._transform(x, adjoint=True)
    if trajectory is not None:
      self.trajectory = temp
    return x

  def _enforce_kernel_symmetry(self, kernel):
    """Enforces Hermitian symmetry on an input kernel.

    Args:
      kernel: A `tf.Tensor`. An approximately Hermitian kernel.

    Returns:
      A Hermitian-symmetric kernel.
    """
    kernel_axes = list(range(-self.rank, 0))
    reversed_kernel = tf.roll(
        tf.reverse(kernel, kernel_axes),
        shift=tf.ones([tf.size(kernel_axes)], dtype=tf.int32),
        axis=kernel_axes)
    return (kernel + tf.math.conj(reversed_kernel)) / 2

  def _range_shape(self):
    # Override the NUFFT operator's range shape. The range shape for this
    # operator is the same as the domain shape.
    return self._domain_shape()

  def _range_shape_tensor(self):
    return self._domain_shape_tensor()


@api_util.export("linalg.LinearOperatorFiniteDifference")
class LinearOperatorFiniteDifference(linalg_imaging.LinearOperator):  # pylint: disable=abstract-method
  """Linear operator representing a finite difference matrix.

  Args:
    domain_shape: A 1D `tf.Tensor` or a `list` of `int`. The domain shape of
      this linear operator.
    axis: An `int`. The axis along which the finite difference is taken.
      Defaults to -1.
    dtype: A `tf.dtypes.DType`. The data type for this operator. Defaults to
      `float32`.
    name: A `str`. A name for this operator.
  """
  def __init__(self,
               domain_shape,
               axis=-1,
               dtype=tf.dtypes.float32,
               name="LinearOperatorFiniteDifference"):

    parameters = dict(
        domain_shape=domain_shape,
        axis=axis,
        dtype=dtype,
        name=name
    )

    # Compute the static and dynamic shapes and save them for later use.
    self._domain_shape_static, self._domain_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))

    # Validate axis and canonicalize to negative. This ensures the correct
    # axis is selected in the presence of batch dimensions.
    self.axis = check_util.validate_static_axes(
        axis, self._domain_shape_static.rank,
        min_length=1,
        max_length=1,
        canonicalize="negative",
        scalar_to_list=False)

    # Compute range shape statically. The range has one less element along
    # the difference axis than the domain.
    range_shape_static = self._domain_shape_static.as_list()
    if range_shape_static[self.axis] is not None:
      range_shape_static[self.axis] -= 1
    range_shape_static = tf.TensorShape(range_shape_static)
    self._range_shape_static = range_shape_static

    # Now compute dynamic range shape. First concatenate the leading axes with
    # the updated difference dimension. Then, iff the difference axis is not
    # the last one, concatenate the trailing axes.
    range_shape_dynamic = self._domain_shape_dynamic
    range_shape_dynamic = tf.concat([
        range_shape_dynamic[:self.axis],
        [range_shape_dynamic[self.axis] - 1]], 0)
    if self.axis != -1:
      range_shape_dynamic = tf.concat([
          range_shape_dynamic,
          range_shape_dynamic[self.axis + 1:]], 0)
    self._range_shape_dynamic = range_shape_dynamic

    super().__init__(dtype,
                     is_non_singular=None,
                     is_self_adjoint=None,
                     is_positive_definite=None,
                     is_square=None,
                     name=name,
                     parameters=parameters)

  def _transform(self, x, adjoint=False):

    if adjoint:
      paddings1 = [[0, 0]] * x.shape.rank
      paddings2 = [[0, 0]] * x.shape.rank
      paddings1[self.axis] = [1, 0]
      paddings2[self.axis] = [0, 1]
      x1 = tf.pad(x, paddings1) # pylint: disable=no-value-for-parameter
      x2 = tf.pad(x, paddings2) # pylint: disable=no-value-for-parameter
      x = x1 - x2
    else:
      slice1 = [slice(None)] * x.shape.rank
      slice2 = [slice(None)] * x.shape.rank
      slice1[self.axis] = slice(1, None)
      slice2[self.axis] = slice(None, -1)
      x1 = x[tuple(slice1)]
      x2 = x[tuple(slice2)]
      x = x1 - x2

    return x

  def _domain_shape(self):
    return self._domain_shape_static

  def _range_shape(self):
    return self._range_shape_static

  def _domain_shape_tensor(self):
    return self._domain_shape_dynamic

  def _range_shape_tensor(self):
    return self._range_shape_dynamic


@api_util.export("linalg.LinearOperatorWavelet")
class LinearOperatorWavelet(linalg_imaging.LinearOperator):  # pylint: disable=abstract-method
  """Linear operator representing a wavelet decomposition matrix.

  Args:
    domain_shape: A 1D `tf.Tensor` or a `list` of `int`. The domain shape of
      this linear operator.
    wavelet: A `str` or a `pywt.Wavelet`_, or a `list` thereof. When passed a
      `list`, different wavelets are applied along each axis in `axes`.
    mode: A `str`. The padding or signal extension mode. Must be one of the
      values supported by `tfmri.signal.wavedec`. Defaults to `'symmetric'`.
    level: An `int` >= 0. The decomposition level. If `None` (default),
      the maximum useful level of decomposition will be used (see
      `tfmri.signal.max_wavelet_level`).
    axes: A `list` of `int`. The axes over which the DWT is computed. Axes refer
      only to domain dimensions without regard for the batch dimensions.
      Defaults to `None` (all domain dimensions).
    dtype: A `tf.dtypes.DType`. The data type for this operator. Defaults to
      `float32`.
    name: A `str`. A name for this operator.
  """
  def __init__(self,
               domain_shape,
               wavelet,
               mode='symmetric',
               level=None,
               axes=None,
               dtype=tf.dtypes.float32,
               name="LinearOperatorWavelet"):
    # Set parameters.
    parameters = dict(
        domain_shape=domain_shape,
        wavelet=wavelet,
        mode=mode,
        level=level,
        axes=axes,
        dtype=dtype,
        name=name
    )

    # Get the static and dynamic shapes and save them for later use.
    self._domain_shape_static, self._domain_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))
    # At the moment, the wavelet implementation relies on shapes being
    # statically known.
    if not self._domain_shape_static.is_fully_defined():
      raise ValueError(f"static `domain_shape` must be fully defined, "
                       f"but got {self._domain_shape_static}")
    static_rank = self._domain_shape_static.rank

    # Set arguments.
    self.wavelet = wavelet
    self.mode = mode
    self.level = level
    self.axes = check_util.validate_static_axes(axes,
                                                rank=static_rank,
                                                min_length=1,
                                                canonicalize="negative",
                                                must_be_unique=True,
                                                scalar_to_list=True,
                                                none_means_all=True)

    # Compute the coefficient slices needed for adjoint (wavelet
    # reconstruction).
    x = tf.ensure_shape(tf.zeros(self._domain_shape_dynamic, dtype=dtype),
                        self._domain_shape_static)
    x = wavelet_ops.wavedec(x, wavelet=self.wavelet, mode=self.mode,
                            level=self.level, axes=self.axes)
    y, self._coeff_slices = wavelet_ops.coeffs_to_tensor(x, axes=self.axes)

    # Get the range shape.
    self._range_shape_static = y.shape
    self._range_shape_dynamic = tf.shape(y)

    # Call base class.
    super().__init__(dtype,
                     is_non_singular=None,
                     is_self_adjoint=None,
                     is_positive_definite=None,
                     is_square=None,
                     name=name,
                     parameters=parameters)

  def _transform(self, x, adjoint=False):
    # While `wavedec` and `waverec` can transform only a subset of axes (and
    # thus theoretically support batches), there is a caveat due to the
    # `coeff_slices` object required by `waverec`. This object contains
    # information relevant to a specific batch shape. While we could recompute
    # this object for every input batch shape, it is easier to just process
    # each batch independently.
    if x.shape.rank is not None and self._domain_shape_static.rank is not None:
      # Rank of input and this operator are known statically, so we can infer
      # the number of batch dimensions statically too.
      batch_dims = x.shape.rank - self._domain_shape_static.rank
    else:
      # We need to obtain the number of batch dimensions dynamically.
      batch_dims = tf.rank(x) - tf.shape(self._domain_shape_dynamic)[0]
    # Transform each batch.
    x = array_ops.map_fn(
        functools.partial(self._transform_batch, adjoint=adjoint),
        x, batch_dims=batch_dims)
    return x

  def _transform_batch(self, x, adjoint=False):
    if adjoint:
      x = wavelet_ops.tensor_to_coeffs(x, self._coeff_slices)
      x = wavelet_ops.waverec(x, wavelet=self.wavelet, mode=self.mode,
                              axes=self.axes)
    else:
      x = wavelet_ops.wavedec(x, wavelet=self.wavelet, mode=self.mode,
                              level=self.level, axes=self.axes)
      x, _ = wavelet_ops.coeffs_to_tensor(x, axes=self.axes)
    return x

  def _domain_shape(self):
    return self._domain_shape_static

  def _range_shape(self):
    return self._range_shape_static

  def _domain_shape_tensor(self):
    return self._domain_shape_dynamic

  def _range_shape_tensor(self):
    return self._range_shape_dynamic


@api_util.export("linalg.LinearOperatorMRI")
class LinearOperatorMRI(linalg_imaging.LinearOperator):  # pylint: disable=abstract-method
  """Linear operator representing an MRI encoding matrix.

  The MRI operator, :math:`A`, maps a [batch of] images, :math:`x` to a
  [batch of] measurement data (*k*-space), :math:`b`.

  .. math::
    A x = b

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
    image_shape: A `tf.TensorShape` or a list of `ints`. The shape of the images
      that this operator acts on. Must have length 2 or 3.
    extra_shape: An optional `tf.TensorShape` or list of `ints`. Additional
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
        dynamic_domain=dynamic_domain,
        dtype=dtype,
        name=name)

    # Set dtype.
    dtype = tf.as_dtype(dtype)
    if dtype not in (tf.complex64, tf.complex128):
      raise ValueError(
          f"`dtype` must be `complex64` or `complex128`, but got: {str(dtype)}")

    # Set image shape, rank and extra shape.
    image_shape = tf.TensorShape(image_shape)
    rank = image_shape.rank
    if rank not in (2, 3):
      raise ValueError(
          f"Rank must be 2 or 3, but got: {rank}")
    if not image_shape.is_fully_defined():
      raise ValueError(
          f"`image_shape` must be fully defined, but got {image_shape}")
    self._rank = rank
    self._image_shape = image_shape
    self._image_axes = list(range(-self._rank, 0))  # pylint: disable=invalid-unary-operand-type
    self._extra_shape = tf.TensorShape(extra_shape or [])

    # Set initial batch shape, then update according to inputs.
    batch_shape = self._extra_shape
    batch_shape_tensor = tensor_util.convert_shape_to_tensor(batch_shape)

    # Set sampling mask after checking dtype and static shape.
    if mask is not None:
      mask = tf.convert_to_tensor(mask)
      if mask.dtype != tf.bool:
        raise TypeError(
            f"`mask` must have dtype `bool`, but got: {str(mask.dtype)}")
      if not mask.shape[-self._rank:].is_compatible_with(self._image_shape):
        raise ValueError(
            f"Expected the last dimensions of `mask` to be compatible with "
            f"{self._image_shape}], but got: {mask.shape[-self._rank:]}")
      batch_shape = tf.broadcast_static_shape(
          batch_shape, mask.shape[:-self._rank])
      batch_shape_tensor = tf.broadcast_dynamic_shape(
          batch_shape_tensor, tf.shape(mask)[:-self._rank])
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
      batch_shape = tf.broadcast_static_shape(
          batch_shape, trajectory.shape[:-2])
      batch_shape_tensor = tf.broadcast_dynamic_shape(
          batch_shape_tensor, tf.shape(trajectory)[:-2])
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
      batch_shape = tf.broadcast_static_shape(
          batch_shape, density.shape[:-1])
      batch_shape_tensor = tf.broadcast_dynamic_shape(
        batch_shape_tensor, tf.shape(density)[:-1])
    self._density = density

    # Set sensitivity maps after checking dtype and static shape.
    if sensitivities is not None:
      sensitivities = tf.convert_to_tensor(sensitivities)
      if sensitivities.dtype != dtype:
        raise TypeError(
            f"Expected `sensitivities` to have dtype `{str(dtype)}`, but got: "
            f"{str(sensitivities.dtype)}")
      if not sensitivities.shape[-self._rank:].is_compatible_with(
          self._image_shape):
        raise ValueError(
            f"Expected the last dimensions of `sensitivities` to be "
            f"compatible with {self._image_shape}, but got: "
            f"{sensitivities.shape[-self._rank:]}")
      batch_shape = tf.broadcast_static_shape(
          batch_shape, sensitivities.shape[:-(self._rank + 1)])
      batch_shape_tensor = tf.broadcast_dynamic_shape(
          batch_shape_tensor, tf.shape(sensitivities)[:-(self._rank + 1)])
    self._sensitivities = sensitivities

    if phase is not None:
      phase = tf.convert_to_tensor(phase)
      if phase.dtype != dtype.real_dtype:
        raise TypeError(
            f"Expected `phase` to have dtype `{str(dtype.real_dtype)}`, "
            f"but got: {str(phase.dtype)}")
      if not phase.shape[-self._rank:].is_compatible_with(
          self._image_shape):
        raise ValueError(
            f"Expected the last dimensions of `phase` to be "
            f"compatible with {self._image_shape}, but got: "
            f"{phase.shape[-self._rank:]}")
      batch_shape = tf.broadcast_static_shape(
          batch_shape, phase.shape[:-self._rank])
      batch_shape_tensor = tf.broadcast_dynamic_shape(
          batch_shape_tensor, tf.shape(phase)[:-self._rank])
    self._phase = phase

    # Set batch shapes.
    self._batch_shape_value = batch_shape
    self._batch_shape_tensor_value = batch_shape_tensor

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
          tf.math.sqrt(tf.cast(self._image_shape.num_elements(), dtype)))

    # Normalize coil sensitivities.
    self._sens_norm = sens_norm
    if self._sensitivities is not None and self._sens_norm:
      self._sensitivities = math_ops.normalize_no_nan(
          self._sensitivities, axis=-(self._rank + 1))

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

    super().__init__(dtype, name=name, parameters=parameters)

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
          x = tfft.nufft(x, self._trajectory,
                         grid_shape=self._image_shape,
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

      # Apply FFT along dynamic axis, if necessary.
      if self.is_dynamic and self.dynamic_domain == 'frequency':
        x = fft_ops.fftn(x, axes=[self.dynamic_axis],
                         norm='ortho', shift=True)

    else:  # Forward operator.

      # Apply FFT along dynamic axis, if necessary.
      if self.is_dynamic and self.dynamic_domain == 'frequency':
        x = fft_ops.ifftn(x, axes=[self.dynamic_axis],
                          norm='ortho', shift=True)

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
          x = tfft.nufft(x, self._trajectory,
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

  def _domain_shape(self):
    """Returns the shape of the domain space of this operator."""
    return self._extra_shape.concatenate(self._image_shape)

  def _range_shape(self):
    """Returns the shape of the range space of this operator."""
    if self.is_cartesian:
      range_shape = self._image_shape.as_list()
    else:
      range_shape = [self._trajectory.shape[-2]]
    if self.is_multicoil:
      range_shape = [self.num_coils] + range_shape
    return self._extra_shape.concatenate(range_shape)

  def _batch_shape(self):
    """Returns the static batch shape of this operator."""
    return self._batch_shape_value[:-self._extra_shape.rank or None]  # pylint: disable=invalid-unary-operand-type

  def _batch_shape_tensor(self):
    """Returns the dynamic batch shape of this operator."""
    return self._batch_shape_tensor_value[:-self._extra_shape.rank or None]  # pylint: disable=invalid-unary-operand-type

  @property
  def image_shape(self):
    """The image shape."""
    return self._image_shape

  @property
  def rank(self):
    """The number of spatial dimensions."""
    return self._rank

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
    """The number of coils."""
    if self._sensitivities is None:
      return None
    return self._sensitivities.shape[-(self._rank + 1)]

  @property
  def _composite_tensor_fields(self):
    return ("image_shape", "mask", "trajectory", "density", "sensitivities",
            "fft_norm")


@api_util.export("linalg.LinearOperatorGramMRI")
class LinearOperatorGramMRI(LinearOperatorMRI):  # pylint: disable=abstract-method
  """Linear operator representing an MRI encoding matrix.

  If :math:`A` is a `tfmri.linalg.LinearOperatorMRI`, then this ooperator
  represents the matrix :math:`G = A^H A`.

  In certain circumstances, this operator may be able to apply the matrix
  :math:`G` more efficiently than the composition :math:`G = A^H A` using
  `tfmri.linalg.LinearOperatorMRI` objects.

  Args:
    image_shape: A `tf.TensorShape` or a list of `ints`. The shape of the images
      that this operator acts on. Must have length 2 or 3.
    extra_shape: An optional `tf.TensorShape` or list of `ints`. Additional
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
      self._linop_gram_nufft = LinearOperatorGramNUFFT(
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


# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

@api_util.export("linalg.conjugate_gradient")
def conjugate_gradient(operator,
                       rhs,
                       preconditioner=None,
                       x=None,
                       tol=1e-5,
                       max_iterations=20,
                       bypass_gradient=False,
                       name=None):
  r"""Conjugate gradient solver.

  Solves a linear system of equations :math:`Ax = b` for self-adjoint, positive
  definite matrix :math:`A` and right-hand side vector :math:`b`, using an
  iterative, matrix-free algorithm where the action of the matrix :math:`A` is
  represented by `operator`. The iteration terminates when either the number of
  iterations exceeds `max_iterations` or when the residual norm has been reduced
  to `tol` times its initial value, i.e.
  :math:`(\left\| b - A x_k \right\| <= \mathrm{tol} \left\| b \right\|\\)`.

  .. note::
    This function is similar to
    `tf.linalg.experimental.conjugate_gradient`, except it adds support for
    complex-valued linear systems and for imaging operators.

  Args:
    operator: A `LinearOperator` that is self-adjoint and positive definite.
    rhs: A `tf.Tensor` of shape `[..., N]`. The right hand-side of the linear
      system.
    preconditioner: A `LinearOperator` that approximates the inverse of `A`.
      An efficient preconditioner could dramatically improve the rate of
      convergence. If `preconditioner` represents matrix `M`(`M` approximates
      `A^{-1}`), the algorithm uses `preconditioner.apply(x)` to estimate
      `A^{-1}x`. For this to be useful, the cost of applying `M` should be
      much lower than computing `A^{-1}` directly.
    x: A `tf.Tensor` of shape `[..., N]`. The initial guess for the solution.
    tol: A float scalar convergence tolerance.
    max_iterations: An `int` giving the maximum number of iterations.
    bypass_gradient: A `boolean`. If `True`, the gradient with respect to `rhs`
      will be computed by applying the inverse of `operator` to the upstream
      gradient with respect to `x` (through CG iteration), instead of relying
      on TensorFlow's automatic differentiation. This may reduce memory usage
      when training neural networks, but `operator` must not have any trainable
      parameters. If `False`, gradients are computed normally. For more details,
      see ref. [1].
    name: A name scope for the operation.

  Returns:
    A `namedtuple` representing the final state with fields

    - i: A scalar `int32` `tf.Tensor`. Number of iterations executed.
    - x: A rank-1 `tf.Tensor` of shape `[..., N]` containing the computed
        solution.
    - r: A rank-1 `tf.Tensor` of shape `[.., M]` containing the residual vector.
    - p: A rank-1 `tf.Tensor` of shape `[..., N]`. `A`-conjugate basis vector.
    - gamma: \\(r \dot M \dot r\\), equivalent to  \\(||r||_2^2\\) when
      `preconditioner=None`.

  Raises:
    ValueError: If `operator` is not self-adjoint and positive definite.

  References:
    .. [1] Aggarwal, H. K., Mani, M. P., & Jacob, M. (2018). MoDL: Model-based
      deep learning architecture for inverse problems. IEEE transactions on
      medical imaging, 38(2), 394-405.
  """
  if bypass_gradient:
    if preconditioner is not None:
      raise ValueError(
          "preconditioner is not supported when bypass_gradient is True.")
    if x is not None:
      raise ValueError("x is not supported when bypass_gradient is True.")

    def _conjugate_gradient_simple(rhs):
      return _conjugate_gradient_internal(operator, rhs,
                                          tol=tol,
                                          max_iterations=max_iterations,
                                          name=name)

    @tf.custom_gradient
    def _conjugate_gradient_internal_grad(rhs):
      result = _conjugate_gradient_simple(rhs)

      def grad(*upstream_grads):
        # upstream_grads has the upstream gradient for each element of the
        # output tuple (i, x, r, p, gamma).
        _, dx, _, _, _ = upstream_grads
        return _conjugate_gradient_simple(dx).x

      return result, grad

    return _conjugate_gradient_internal_grad(rhs)

  return _conjugate_gradient_internal(operator, rhs,
                                      preconditioner=preconditioner,
                                      x=x,
                                      tol=tol,
                                      max_iterations=max_iterations,
                                      name=name)


def _conjugate_gradient_internal(operator,
                                 rhs,
                                 preconditioner=None,
                                 x=None,
                                 tol=1e-5,
                                 max_iterations=20,
                                 name=None):
  """Implementation of `conjugate_gradient`.

  For the parameters, see `conjugate_gradient`.
  """
  if isinstance(operator, linalg_imaging.LinalgImagingMixin):
    rhs = operator.flatten_domain_shape(rhs)

  if not (operator.is_self_adjoint and operator.is_positive_definite):
    raise ValueError('Expected a self-adjoint, positive definite operator.')

  cg_state = collections.namedtuple('CGState', ['i', 'x', 'r', 'p', 'gamma'])

  def stopping_criterion(i, state):
    return tf.math.logical_and(
        i < max_iterations,
        tf.math.reduce_any(
            tf.math.real(tf.norm(state.r, axis=-1)) > tf.math.real(tol)))

  def dot(x, y):
    return tf.squeeze(
        tf.linalg.matvec(
            x[..., tf.newaxis],
            y, adjoint_a=True), axis=-1)

  def cg_step(i, state):  # pylint: disable=missing-docstring
    z = tf.linalg.matvec(operator, state.p)
    alpha = state.gamma / dot(state.p, z)
    x = state.x + alpha[..., tf.newaxis] * state.p
    r = state.r - alpha[..., tf.newaxis] * z
    if preconditioner is None:
      q = r
    else:
      q = preconditioner.matvec(r)
    gamma = dot(r, q)
    beta = gamma / state.gamma
    p = q + beta[..., tf.newaxis] * state.p
    return i + 1, cg_state(i + 1, x, r, p, gamma)

  # We now broadcast initial shapes so that we have fixed shapes per iteration.

  with tf.name_scope(name or 'conjugate_gradient'):
    broadcast_shape = tf.broadcast_dynamic_shape(
        tf.shape(rhs)[:-1],
        operator.batch_shape_tensor())
    static_broadcast_shape = tf.broadcast_static_shape(
        rhs.shape[:-1],
        operator.batch_shape)
    if preconditioner is not None:
      broadcast_shape = tf.broadcast_dynamic_shape(
          broadcast_shape,
          preconditioner.batch_shape_tensor())
      static_broadcast_shape = tf.broadcast_static_shape(
          static_broadcast_shape,
          preconditioner.batch_shape)
    broadcast_rhs_shape = tf.concat([broadcast_shape, [tf.shape(rhs)[-1]]], -1)
    static_broadcast_rhs_shape = static_broadcast_shape.concatenate(
        [rhs.shape[-1]])
    r0 = tf.broadcast_to(rhs, broadcast_rhs_shape)
    tol *= tf.norm(r0, axis=-1)

    if x is None:
      x = tf.zeros(
          broadcast_rhs_shape, dtype=rhs.dtype.base_dtype)
      x = tf.ensure_shape(x, static_broadcast_rhs_shape)
    else:
      r0 = rhs - tf.linalg.matvec(operator, x)
    if preconditioner is None:
      p0 = r0
    else:
      p0 = tf.linalg.matvec(preconditioner, r0)
    gamma0 = dot(r0, p0)
    i = tf.constant(0, dtype=tf.int32)
    state = cg_state(i=i, x=x, r=r0, p=p0, gamma=gamma0)
    _, state = tf.while_loop(
        stopping_criterion, cg_step, [i, state])

    if isinstance(operator, linalg_imaging.LinalgImagingMixin):
      x = operator.expand_range_dimension(state.x)
    else:
      x = state.x

    return cg_state(
        state.i,
        x=x,
        r=state.r,
        p=state.p,
        gamma=state.gamma)
