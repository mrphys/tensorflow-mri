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
"""Linear algebra operations.

This module contains linear operators and solvers.
"""

import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import tensor_util


@api_util.export("linalg.LinearOperatorNUFFT")
class LinearOperatorNUFFT(linear_operator.LinearOperator):  # pylint: disable=abstract-method
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
    adjoint transform. However, in order to maintain certain properties of the
    linear operator, this operator applies the compensation orthogonally, i.e.,
    it scales the data by the square root of `density` in both forward and
    adjoint transforms. If you are using this operator to compute the adjoint
    and wish to apply the full compensation, you can do so via the
    `preprocess` method.

  Example:
    >>> # Create some data.
    >>> image_shape = (128, 128)
    >>> image = tfmri.image.phantom(shape=image_shape, dtype=tf.complex64)
    >>> trajectory = tfmri.sampling.radial_trajectory(
    ...     base_resolution=128, views=129, flatten_encoding_dims=True)
    >>> density = tfmri.sampling.radial_density(
    ...     base_resolution=128, views=129, flatten_encoding_dims=True)
    >>> # Create a NUFFT operator.
    >>> linop = tfmri.linalg.LinearOperatorNUFFT(
    ...     image_shape, trajectory=trajectory, density=density)
    >>> # Create k-space.
    >>> kspace = tfmri.signal.nufft(image, trajectory)
    >>> # This reconstructs the image applying only partial compensation
    >>> # (square root of weights).
    >>> image = linop.transform(kspace, adjoint=True)
    >>> # This reconstructs the image with full compensation.
    >>> image = linop.transform(
    ...     linop.preprocess(kspace, adjoint=True), adjoint=True)

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

  def _preprocess(self, x, adjoint=False):
    if adjoint:
      if self.density is not None:
        x *= self._weights_sqrt
    else:
      raise NotImplementedError(
          "_preprocess not implemented for forward transform.")
    return x

  def _postprocess(self, x, adjoint=False):
    if adjoint:
      pass  # nothing to do
    else:
      raise NotImplementedError(
          "_postprocess not implemented for forward transform.")
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

  If $F$ is a `tfmri.linalg.LinearOperatorNUFFT`, then this operator
  applies $F^H F$. This operator is self-adjoint.

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
      to compute $F^H F x$, where $F$ is the NUFFT operator.
      If `False`, the same operation is performed using the standard
      NUFFT operation. The Toeplitz approach might be faster than the direct
      approach but is slightly less accurate. This argument is only relevant
      for non-Cartesian reconstruction and will be ignored for Cartesian
      problems.
    name: An optional `str`. The name of this operator.

  References:
    1. Fessler, J. A., Lee, S., Olafsson, V. T., Shi, H. R., & Noll, D. C.
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
