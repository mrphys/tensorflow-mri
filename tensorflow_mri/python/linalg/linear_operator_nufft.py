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

import tensorflow as tf

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
    >>> image = linop.transform(linop.preprocess(kspace, adjoint=True), adjoint=True)
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
