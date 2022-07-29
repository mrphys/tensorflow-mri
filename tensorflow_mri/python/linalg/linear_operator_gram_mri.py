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

import tensorflow as tf

from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.linalg import linear_operator_gram_nufft
from tensorflow_mri.python.linalg import linear_operator_mri


@api_util.export("linalg.LinearOperatorGramMRI")
class LinearOperatorGramMRI(linear_operator_mri.LinearOperatorMRI):  # pylint: disable=abstract-method
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
      self._linop_gram_nufft = linear_operator_gram_nufft.LinearOperatorGramNUFFT(
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
