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
"""Operators for MR image reconstruction.

Image reconstruction operators accept *k*-space data and additional
application-dependent inputs and return an image.
"""

import collections

import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import coil_ops
from tensorflow_mri.python.ops import convex_ops
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.ops import linalg_ops
from tensorflow_mri.python.ops import math_ops
from tensorflow_mri.python.ops import optimizer_ops
from tensorflow_mri.python.ops import signal_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import deprecation
from tensorflow_mri.python.util import linalg_imaging


@api_util.export("recon.adjoint", "recon.adj")
def reconstruct_adj(kspace,
                    image_shape,
                    mask=None,
                    trajectory=None,
                    density=None,
                    sensitivities=None,
                    phase=None,
                    sens_norm=True):
  r"""Reconstructs an MR image using the adjoint MRI operator.

  Given *k*-space data :math:`b`, this function estimates the corresponding
  image as :math:`x = A^H b`, where :math:`A` is the MRI linear operator.

  This operator supports Cartesian and non-Cartesian *k*-space data.

  Additional density compensation and intensity correction steps are applied
  depending on the input arguments.

  This operator supports batched inputs. All batch shapes should be
  broadcastable with each other.

  This operator supports multicoil imaging. Coil combination is triggered
  when `sensitivities` is not `None`. If you have multiple coils but wish to
  reconstruct each coil separately, simply set `sensitivities` to `None`. The
  coil dimension will then be treated as a standard batch dimension (i.e., it
  becomes part of `...`).

  Args:
    kspace: A `Tensor`. The *k*-space samples. Must have type `complex64` or
      `complex128`. `kspace` can be either Cartesian or non-Cartesian. A
      Cartesian `kspace` must have shape
      `[..., num_coils, *image_shape]`, where `...` are batch dimensions. A
      non-Cartesian `kspace` must have shape `[..., num_coils, num_samples]`.
      If not multicoil (`sensitivities` is `None`), then the `num_coils` axis
      must be omitted.
    image_shape: A `TensorShape` or a list of `ints`. Must have length 2 or 3.
      The shape of the reconstructed image[s].
    mask: An optional `Tensor` of type `bool`. The sampling mask. Must have
      shape `[..., image_shape]`. `mask` should be passed for reconstruction
      from undersampled Cartesian *k*-space. For each point, `mask` should be
      `True` if the corresponding *k*-space sample was measured and `False`
      otherwise.
    trajectory: An optional `Tensor` of type `float32` or `float64`. Must have
      shape `[..., num_samples, rank]`. `trajectory` should be passed for
      reconstruction from non-Cartesian *k*-space.
    density: An optional `Tensor` of type `float32` or `float64`. The sampling
      densities. Must have shape `[..., num_samples]`. This input is only
      relevant for non-Cartesian MRI reconstruction. If passed, the MRI linear
      operator will include sampling density compensation. If `None`, the MRI
      operator will not perform sampling density compensation.
    sensitivities: An optional `Tensor` of type `complex64` or `complex128`.
      The coil sensitivity maps. Must have shape
      `[..., num_coils, *image_shape]`. If provided, a multi-coil parallel
      imaging reconstruction will be performed.
    phase: An optional `Tensor` of type `float32` or `float64`. Must have shape
      `[..., *image_shape]`. A phase estimate for the reconstructed image. If
      provided, a phase-constrained reconstruction will be performed. This
      improves the conditioning of the reconstruction problem in applications
      where there is no interest in the phase data. However, artefacts may
      appear if an inaccurate phase estimate is passed.
    sens_norm: A `boolean`. Whether to normalize coil sensitivities. Defaults to
      `True`.

  Returns:
    A `Tensor`. The reconstructed image. Has the same type as `kspace` and
    shape `[..., *image_shape]`, where `...` is the broadcasted batch shape of
    all inputs.

  Notes:
    Reconstructs an image by applying the adjoint MRI operator to the *k*-space
    data. This typically involves an inverse FFT or a (density-compensated)
    NUFFT, and coil combination for multicoil inputs. This type of
    reconstruction is often called zero-filled reconstruction, because missing
    *k*-space samples are assumed to be zero. Therefore, the resulting image is
    likely to display aliasing artefacts if *k*-space is not sufficiently
    sampled according to the Nyquist criterion.
  """
  kspace = tf.convert_to_tensor(kspace)

  # Create the linear operator.
  operator = linalg_ops.LinearOperatorMRI(image_shape,
                                          mask=mask,
                                          trajectory=trajectory,
                                          density=density,
                                          sensitivities=sensitivities,
                                          phase=phase,
                                          fft_norm='ortho',
                                          sens_norm=sens_norm)
  rank = operator.rank

  # Apply density compensation, if provided.
  if density is not None:
    dens_weights_sqrt = tf.math.sqrt(tf.math.reciprocal_no_nan(density))
    dens_weights_sqrt = tf.cast(dens_weights_sqrt, kspace.dtype)
    if operator.is_multicoil:
      dens_weights_sqrt = tf.expand_dims(dens_weights_sqrt, axis=-2)
    kspace *= dens_weights_sqrt

  # Compute zero-filled image using the adjoint operator.
  image = operator.H.transform(kspace)

  # Apply intensity correction, if requested.
  if operator.is_multicoil and sens_norm:
    sens_weights_sqrt = tf.math.reciprocal_no_nan(
        tf.norm(sensitivities, axis=-(rank + 1), keepdims=False))
    image *= sens_weights_sqrt

  return image


@api_util.export("recon.least_squares", "recon.lstsq")
def reconstruct_lstsq(kspace,
                      image_shape,
                      extra_shape=None,
                      mask=None,
                      trajectory=None,
                      density=None,
                      sensitivities=None,
                      phase=None,
                      sens_norm=True,
                      dynamic_domain=None,
                      regularizer=None,
                      optimizer=None,
                      optimizer_kwargs=None,
                      filter_corners=False,
                      return_optimizer_state=False,
                      toeplitz_nufft=False):
  r"""Reconstructs an MR image using a least-squares formulation.

  This is an iterative reconstruction method which formulates the image
  reconstruction problem as follows:

  .. math::
    \hat{x} = {\mathop{\mathrm{argmin}}_x} \left (\left\| Ax - y \right\|_2^2 + g(x) \right )

  where :math:`A` is the MRI `LinearOperator`, :math:`x` is the solution, `y` is
  the measured *k*-space data, and :math:`g(x)` is an optional `ConvexFunction`
  used for regularization.

  This operator supports Cartesian and non-Cartesian *k*-space data.

  This operator supports linear and non-linear reconstruction, depending on the
  selected regularizer. The MRI operator is constructed internally and does not
  need to be provided.

  This operator supports batched inputs. All batch shapes should be
  broadcastable with each other.

  Args:
    kspace: A `Tensor`. The *k*-space samples. Must have type `complex64` or
      `complex128`. `kspace` can be either Cartesian or non-Cartesian. A
      Cartesian `kspace` must have shape
      `[..., *extra_shape, num_coils, *image_shape]`, where `...` are batch
      dimensions. A non-Cartesian `kspace` must have shape
      `[..., *extra_shape, num_coils, num_samples]`.
    image_shape: A `TensorShape` or a list of `ints`. Must have length 2 or 3.
      The shape of the reconstructed image[s].
    extra_shape: An optional `TensorShape` or list of `ints`. Additional
      dimensions that should be included within the solution domain. Note
      that `extra_shape` is not needed to reconstruct independent batches of
      images. However, it should be provided when performing a reconstruction
      that operates along non-spatial dimensions, e.g. for temporal
      regularization. Defaults to `[]`.
    mask: An optional `Tensor` of type `bool`. The sampling mask. Must have
      shape `[..., image_shape]`. `mask` should be passed for reconstruction
      from undersampled Cartesian *k*-space. For each point, `mask` should be
      `True` if the corresponding *k*-space sample was measured and `False`
      otherwise.
    trajectory: An optional `Tensor` of type `float32` or `float64`. Must have
      shape `[..., num_samples, rank]`. `trajectory` should be passed for
      reconstruction from non-Cartesian *k*-space.
    density: An optional `Tensor` of type `float32` or `float64`. The sampling
      densities. Must have shape `[..., num_samples]`. This input is only
      relevant for non-Cartesian MRI reconstruction. If passed, the MRI linear
      operator will include sampling density compensation. If `None`, the MRI
      operator will not perform sampling density compensation.
    sensitivities: An optional `Tensor` of type `complex64` or `complex128`.
      The coil sensitivity maps. Must have shape
      `[..., num_coils, *image_shape]`. If provided, a multi-coil parallel
      imaging reconstruction will be performed.
    phase: An optional `Tensor` of type `float32` or `float64`. Must have shape
      `[..., *image_shape]`. A phase estimate for the reconstructed image. If
      provided, a phase-constrained reconstruction will be performed. This
      improves the conditioning of the reconstruction problem in applications
      where there is no interest in the phase data. However, artefacts may
      appear if an inaccurate phase estimate is passed.
    sens_norm: A `boolean`. Whether to normalize coil sensitivities. Defaults to
      `True`.
    dynamic_domain: A `str`. The domain of the dynamic dimension, if present.
      Must be one of `'time'` or `'frequency'`. May only be provided together
      with a non-scalar `extra_shape`. The dynamic dimension is the last
      dimension of `extra_shape`. The `'time'` mode (default) should be
      used for regular dynamic reconstruction. The `'frequency'` mode should be
      used for reconstruction in x-f space.
    regularizer: A `ConvexFunction`. The regularization term added to
      least-squares objective.
    optimizer: A `str`. One of `'cg'` (conjugate gradient), `'admm'`
      (alternating direction method of multipliers) of `'lbfgs'`
      (limited-memory Broyden-Fletcher-Goldfarb-Shanno). If `None`, the
      optimizer is selected heuristically depending on other inputs. Note that
      this heuristic may change in the future, so specify an optimizer if you
      wish to ensure it will always be used in future versions. Not all
      optimizers are compatible with all configurations.
    optimizer_kwargs: An optional `dict`. Additional arguments to pass to the
      optimizer.
    filter_corners: A `boolean`. Whether to filter out the *k*-space corners in
      reconstructed image. This may be done for trajectories with a circular
      *k*-space coverage. Defaults to `False`.
    return_optimizer_state: A `boolean`. If `True`, returns the optimizer
      state along with the reconstructed image.
    toeplitz_nufft: A `boolean`. If `True`, uses the Toeplitz approach [5]
      to compute :math:`F^H F x`, where :math:`F` is the non-uniform Fourier
      operator. If `False`, the same operation is performed using the standard
      NUFFT operation. The Toeplitz approach might be faster than the direct
      approach but is slightly less accurate. This argument is only relevant
      for non-Cartesian reconstruction and will be ignored for Cartesian
      problems.

  Returns:
    A `Tensor`. The reconstructed image. Has the same type as `kspace` and
    shape `[..., *extra_shape, *image_shape]`, where `...` is the broadcasted
    batch shape of all inputs.

    If `return_optimizer_state` is `True`, returns a tuple containing the
    reconstructed image and the optimizer state.

  Raises:
    ValueError: If passed incompatible inputs.

  Notes:
    Reconstructs an image by formulating a (possibly regularized) least squares
    problem, which is solved iteratively. Since the problem may be ill-posed,
    different types of regularizers may be used to incorporate prior knowledge.
    Depending on the regularizer, the optimization problem may be linear or
    nonlinear. For sparsity-based regularizers, this is also called a compressed
    sensing reconstruction. This is a powerful operator which can often produce
    high-quality images even from highly undersampled *k*-space data. However,
    it may be time-consuming, depending on the characteristics of the problem.

  References:
    .. [1] Pruessmann, K.P., Weiger, M., Börnert, P. and Boesiger, P. (2001),
      Advances in sensitivity encoding with arbitrary k-space trajectories.
      Magn. Reson. Med., 46: 638-651. https://doi.org/10.1002/mrm.1241

    .. [2] Block, K.T., Uecker, M. and Frahm, J. (2007), Undersampled radial MRI
      with multiple coils. Iterative image reconstruction using a total
      variation constraint. Magn. Reson. Med., 57: 1086-1098.
      https://doi.org/10.1002/mrm.21236

    .. [3] Feng, L., Grimm, R., Block, K.T., Chandarana, H., Kim, S., Xu, J.,
      Axel, L., Sodickson, D.K. and Otazo, R. (2014), Golden-angle radial sparse
      parallel MRI: Combination of compressed sensing, parallel imaging, and
      golden-angle radial sampling for fast and flexible dynamic volumetric MRI.
      Magn. Reson. Med., 72: 707-717. https://doi.org/10.1002/mrm.24980

    .. [4] Tsao, J., Boesiger, P., & Pruessmann, K. P. (2003). k-t BLAST and
      k-t SENSE: dynamic MRI with high frame rate exploiting spatiotemporal
      correlations. Magnetic Resonance in Medicine: An Official Journal of the
      International Society for Magnetic Resonance in Medicine, 50(5),
      1031-1042.

    .. [5] Fessler, J. A., Lee, S., Olafsson, V. T., Shi, H. R., & Noll, D. C.
      (2005). Toeplitz-based iterative image reconstruction for MRI with
      correction for magnetic field inhomogeneity. IEEE Transactions on Signal
      Processing, 53(9), 3393-3402.
  """  # pylint: disable=line-too-long
  # Choose a default optimizer.
  if optimizer is None:
    if regularizer is None or isinstance(regularizer,
                                         convex_ops.ConvexFunctionTikhonov):
      optimizer = 'cg'
    else:
      optimizer = 'admm'
  # Check optimizer.
    optimizer = check_util.validate_enum(
        optimizer, {'cg', 'admm', 'lbfgs'}, name='optimizer')
  optimizer_kwargs = optimizer_kwargs or {}

  # We don't do a lot of input checking here, since it will be done by the
  # operator.
  kspace = tf.convert_to_tensor(kspace)

  # Create the linear operator.
  operator = linalg_ops.LinearOperatorMRI(image_shape,
                                          extra_shape=extra_shape,
                                          mask=mask,
                                          trajectory=trajectory,
                                          density=density,
                                          sensitivities=sensitivities,
                                          phase=phase,
                                          fft_norm='ortho',
                                          sens_norm=sens_norm,
                                          dynamic_domain=dynamic_domain)
  rank = operator.rank

  # If using Toeplitz NUFFT, we need to use the specialized Gram MRI operator.
  if toeplitz_nufft and operator.is_non_cartesian:
    gram_operator = linalg_ops.LinearOperatorGramMRI(
        image_shape,
        extra_shape=extra_shape,
        mask=mask,
        trajectory=trajectory,
        density=density,
        sensitivities=sensitivities,
        phase=phase,
        fft_norm='ortho',
        sens_norm=sens_norm,
        dynamic_domain=dynamic_domain,
        toeplitz_nufft=toeplitz_nufft)
  else:
    # No Toeplitz NUFFT. In this case don't bother defining the Gram operator.
    gram_operator = None

  # Apply density compensation, if provided.
  if density is not None:
    kspace *= operator._dens_weights_sqrt  # pylint: disable=protected-access

  initial_image = operator.H.transform(kspace)

  # Optimizer-specific logic.
  if optimizer == 'cg':
    if regularizer is not None:
      if not isinstance(regularizer, convex_ops.ConvexFunctionTikhonov):
        raise ValueError(
            f"Regularizer {regularizer.name} is incompatible with "
            f"CG optimizer.")
      reg_parameter = regularizer.function.scale
      reg_operator = regularizer.transform
      reg_prior = regularizer.prior
    else:
      reg_parameter = None
      reg_operator = None
      reg_prior = None

    operator_gm = linalg_imaging.LinearOperatorGramMatrix(
        operator, reg_parameter=reg_parameter, reg_operator=reg_operator,
        gram_operator=gram_operator)
    rhs = initial_image
    # Update the rhs with the a priori estimate, if provided.
    if reg_prior is not None:
      if reg_operator is not None:
        reg_prior = reg_operator.transform(
            reg_operator.transform(reg_prior), adjoint=True)
      rhs += tf.cast(reg_parameter, reg_prior.dtype) * reg_prior
    # Solve the (maybe regularized) linear system.
    result = linalg_ops.conjugate_gradient(operator_gm, rhs, **optimizer_kwargs)
    image = result.x

  elif optimizer == 'admm':
    if regularizer is None:
      raise ValueError("optimizer 'admm' requires a regularizer")
    # Create the least-squares objective.
    function_f = convex_ops.ConvexFunctionLeastSquares(
        operator, kspace, gram_operator=gram_operator)
    # Configure ADMM formulation depending on regularizer.
    if isinstance(regularizer,
                  convex_ops.ConvexFunctionLinearOperatorComposition):
      function_g = regularizer.function
      operator_a = regularizer.operator
    else:
      function_g = regularizer
      operator_a = None
    # Run ADMM minimization.
    result = optimizer_ops.admm_minimize(function_f, function_g,
                                         operator_a=operator_a,
                                         **optimizer_kwargs)
    image = operator.expand_domain_dimension(result.f_primal_variable)

  elif optimizer == 'lbfgs':
    # Flatten k-space and initial estimate.
    initial_image = operator.flatten_domain_shape(initial_image)
    y = operator.flatten_range_shape(kspace)

    # Currently L-BFGS implementation only supports real numbers, so reinterpret
    # complex image as real (C^N -> R^2*N).
    initial_image = math_ops.view_as_real(initial_image, stacked=False)

    # Define the objective function and its gradient.
    @tf.function
    @math_ops.make_val_and_grad_fn
    def _objective(x):
      # Reinterpret real input as complex.
      x = math_ops.view_as_complex(x, stacked=False)
      # Compute objective.
      obj = tf.math.abs(tf.norm(y - operator.matvec(x), ord=2))
      if regularizer is not None:
        obj += regularizer(x)
      return obj

    # Do minimization.
    result = optimizer_ops.lbfgs_minimize(_objective, initial_image,
                                          **optimizer_kwargs)

    # Reinterpret real result as complex and reshape image.
    image = operator.expand_domain_dimension(
        math_ops.view_as_complex(result.position, stacked=False))

  else:
    raise ValueError(f"Unknown optimizer: {optimizer}")

  # Apply temporal Fourier operator, if necessary.
  if operator.is_dynamic and operator.dynamic_domain == 'frequency':
    image = fft_ops.ifftn(image, axes=[operator.dynamic_axis],
                          norm='ortho', shift=True)

  # Apply intensity correction, if requested.
  if operator.is_multicoil and sens_norm:
    sens_weights_sqrt = tf.math.reciprocal_no_nan(
        tf.norm(sensitivities, axis=-(rank + 1), keepdims=False))
    image *= sens_weights_sqrt

  # If necessary, filter the image to remove k-space corners. This can be
  # done if the trajectory has circular coverage and does not cover the k-space
  # corners. If the user has not specified whether to apply the filter, we do it
  # only for non-Cartesian trajectories, under the assumption that non-Cartesian
  # trajectories are likely to have circular coverage of k-space while Cartesian
  # trajectories are likely to have rectangular coverage.
  if filter_corners is None:
    is_probably_circular = operator.is_non_cartesian
    filter_corners = is_probably_circular
  if filter_corners:
    fft_axes = list(range(-rank, 0))  # pylint: disable=invalid-unary-operand-type
    kspace = fft_ops.fftn(image, axes=fft_axes, norm='ortho', shift=True)
    kspace = signal_ops.filter_kspace(kspace, filter_fn='atanfilt',
                                      filter_rank=rank)
    image = fft_ops.ifftn(kspace, axes=fft_axes, norm='ortho', shift=True)

  if return_optimizer_state:
    return image, result

  return image


@api_util.export("recon.sense")
def reconstruct_sense(kspace,
                      sensitivities,
                      reduction_axis,
                      reduction_factor,
                      rank=None,
                      l2_regularizer=0.0,
                      fast=True):
  r"""Reconstructs an MR image using SENSE.

  Args:
    kspace: A `Tensor`. The *k*-space samples. Must have type `complex64` or
      `complex128`. Must have shape `[..., C, *K]`, where `K` is the shape
      of the spatial frequency dimensions, `C` is the number of coils and `...`
      is the batch shape, which can have any rank. Note that `K` is the
      reduced or undersampled shape.
    sensitivities: A `Tensor`. The coil sensitivity maps. Must have type
      `complex64` or `complex128`. Must have shape `[..., C, *S]`, where `S` is
      shape of the spatial dimensions, `C` is the number of coils and `...` is
      the batch shape, which can have any rank and must be broadcastable to the
      batch shape of `kspace`.
    reduction_axis: An `int` or a list of `ints`. The reduced axes. This
        parameter must be provided.
    reduction_factor: An `int` or a list of `ints`. The reduction
      factors corresponding to each reduction axis. The output image will have
      dimension `kspace.shape[ax] * r` for each pair `ax` and `r` in
      `reduction_axis` and `reduction_factor`. This parameter must be
      provided.
    rank: An optional `int`. The rank (in the sense of spatial
      dimensionality) of this operation. Defaults to `kspace.shape.rank - 1`.
      Therefore, if `rank` is not specified, axis 0 is interpreted to be the
      coil axis and the remaining dimensions are interpreted to be spatial
      dimensions. You must specify `rank` if you intend to provide any batch
      dimensions in `kspace` and/or `sensitivities`.
    l2_regularizer: An optional `float`. The L2 regularization factor
      used when solving the linear least-squares problem. Ignored if
      `fast=False`. Defaults to 0.0.
    fast: An optional `bool`. Defaults to `True`. If `False`, use a
      numerically robust orthogonal decomposition method to solve the linear
      least-squares. This algorithm finds the solution even for rank deficient
      matrices, but is significantly slower. For more details, see
      `tf.linalg.lstsq`.

  Returns:
    A `Tensor`. The reconstructed images. Has the same type as `kspace`. Has
    shape `[..., S]`, where `...` is the reconstruction batch shape and `S` is
    the spatial shape.

  Raises:
    ValueError: If `kspace` and `sensitivities` have incompatible batch shapes.

  References:
    .. [1] Pruessmann, K.P., Weiger, M., Scheidegger, M.B. and Boesiger, P.
      (1999), SENSE: Sensitivity encoding for fast MRI. Magn. Reson. Med.,
      42: 952-962.
      https://doi.org/10.1002/(SICI)1522-2594(199911)42:5<952::AID-MRM16>3.0.CO;2-S
  """
  # Parse inputs.
  kspace = tf.convert_to_tensor(kspace)
  sensitivities = tf.convert_to_tensor(sensitivities)

  # Rank or spatial dimensionality.
  rank = rank or kspace.shape.rank - 1

  reduced_shape = kspace.shape[-rank:]
  reduction_axis = check_util.validate_list(
    reduction_axis, element_type=int, name='reduction_axis')
  reduction_factor = check_util.validate_list(
    reduction_factor, element_type=int, length=len(reduction_axis),
    name='reduction_factor')
  reduction_axis = [ax + rank if ax < 0 else ax for ax in reduction_axis]
  canonical_reduction = [1] * rank
  for ax, r in zip(reduction_axis, reduction_factor):
    canonical_reduction[ax] = r
  image_shape = tf.TensorShape(
    [s * r for s, r in zip(reduced_shape.as_list(), canonical_reduction)])

  # Compute batch shapes. `batch_shape` is the output batch shape.
  kspace_rank = kspace.shape.rank
  kspace_batch_shape = kspace.shape[:-rank-1]
  sens_rank = sensitivities.shape.rank
  sens_batch_shape = sensitivities.shape[:-rank-1]
  batch_shape = tf.broadcast_static_shape(kspace_batch_shape, sens_batch_shape)
  # We do not broadcast the k-space, by design.
  if batch_shape != kspace_batch_shape:
    raise ValueError(
      f"`kspace` and `sensitivities` have incompatible batch shapes: "
      f"{kspace_batch_shape}, {sens_batch_shape}")

  # Rearrange dimensions. Put spatial dimensions first, then coil dimension,
  # then batch dimensions.
  kspace_perm = list(range(-rank, 0)) + [-rank-1]
  kspace_perm = [ax + kspace_rank for ax in kspace_perm]
  kspace_perm += list(range(0, kspace_rank - rank - 1))
  sens_perm = list(range(-rank, 0)) + [-rank-1]
  sens_perm = [ax + sens_rank for ax in sens_perm]
  sens_perm += list(range(0, sens_rank - rank - 1))
  kspace = tf.transpose(kspace, kspace_perm)
  sensitivities = tf.transpose(sensitivities, sens_perm)

  # Compute aliased images and shift along the reduced dimensions.
  aliased_images = fft_ops.ifftn(kspace, axes=list(range(rank)), shift=True)
  aliased_images = tf.signal.ifftshift(aliased_images, axes=reduction_axis)

  # Create a grid of indices into the reduced FOV image.
  reduced_indices = tf.stack(tf.meshgrid(*[tf.range(s) for s in reduced_shape]))
  reduced_indices = tf.transpose(tf.reshape(reduced_indices, [rank, -1]))

  # Compute corresponding indices into the full FOV image.
  offsets = [tf.range(r) * s for s, r in zip(
    reduced_shape.as_list(), canonical_reduction)]
  offsets = tf.transpose(tf.reshape(
    tf.stack(tf.meshgrid(*offsets)), [rank, -1]))
  indices = tf.expand_dims(reduced_indices, -2) + offsets

  # Compute the system matrices, ie, pixel-wise sensitivity matrices folding the
  # full FOV image into a reduced FOV image.
  sens_matrix = tf.gather_nd(sensitivities, indices)
  sens_matrix = tf.transpose(
    sens_matrix, [0, 2, 1] + list(range(3, 3 + sens_batch_shape.rank)))

  # Compute the right hand sides for the set of linear systems.
  rhs = tf.gather_nd(aliased_images, reduced_indices)

  # Remove any pixels known to have zero signal, with no contributions from any
  # of the aliases. Currently we can't do this for batched sensitivities, so it
  # is disabled in that case.
  if sens_batch_shape.rank == 0:
    mask = tf.reduce_sum(tf.math.square(tf.math.abs(sens_matrix)), -2) > 0
    mask = tf.math.reduce_any(mask, axis=-1)
    sens_matrix = tf.boolean_mask(sens_matrix, mask, axis=0)
    rhs = tf.boolean_mask(rhs, mask, axis=0)
    indices = tf.boolean_mask(indices, mask, axis=0)

  # Move batch dimensions to the beginning.
  sens_matrix = tf.transpose(
    sens_matrix, list(range(3, sens_matrix.shape.rank)) + [0, 1, 2])
  rhs = tf.transpose(rhs, list(range(2, rhs.shape.rank)) + [0, 1])
  rhs = tf.expand_dims(rhs, -1)

  # Broadcast the sensitivity matrix as necessary.
  sens_matrix = tf.broadcast_to(
    sens_matrix, batch_shape + sens_matrix.shape[-3:])

  # Solve the pixel-wise linear least-squares problems.
  unfolded_values = tf.linalg.lstsq(sens_matrix, rhs,
                                    l2_regularizer=l2_regularizer,
                                    fast=fast)

  unfolded_values = tf.reshape(unfolded_values, [-1])
  output_indices = tf.reshape(indices, [-1, rank])

  # For batch mode we need to do some additional indexing calculations.
  if batch_shape.rank > 0:
    batch_size = batch_shape.num_elements()
    element_size = unfolded_values.shape[0] // batch_size

    batch_indices = tf.stack(tf.meshgrid(*[tf.range(s) for s in batch_shape]))
    batch_indices = tf.transpose(
      tf.reshape(batch_indices, [batch_shape.rank, -1]))
    batch_indices = tf.expand_dims(batch_indices, -2)
    batch_indices = tf.tile(
      batch_indices, [1] * batch_shape.rank + [element_size, 1])
    batch_indices = tf.reshape(batch_indices, [-1, batch_shape.rank])

    output_indices = tf.tile(output_indices, [batch_size, 1])
    output_indices = tf.concat([batch_indices, output_indices], -1)

  # Scatter the unfolded values into the reconstructed image.
  image = tf.scatter_nd(output_indices, unfolded_values,
                        batch_shape + image_shape)

  return image


@api_util.export("recon.grappa")
def reconstruct_grappa(kspace,
                       mask,
                       calib,
                       kernel_size=5,
                       weights_l2_regularizer=0.0,
                       combine_coils=True,
                       sensitivities=None,
                       return_kspace=False):
  """Reconstructs an MR image using GRAPPA.

  Args:
    kspace: A `Tensor`. The *k*-space samples. Must have type `complex64` or
      `complex128`. Must have shape `[..., C, *K]`, where `K` is the shape
      of the spatial frequency dimensions, `C` is the number of coils and `...`
      is the batch shape, which can have any rank. Note that `K` is the
      reduced or undersampled shape.
    mask: A `Tensor`. The sampling mask. Must have type `bool`. Must have shape
      `S`, where `S` is the shape of the spatial dimensions. In other words,
      `mask` should have the shape of a fully sampled *k*-space. For each point,
      `mask` should be `True` if the corresponding *k*-space sample was measured
      and `False` otherwise. `True` entries should correspond to the data in
      `kspace`, and the result of dropping all `False` entries from `mask`
      should have shape `K`.
    calib: A `Tensor`. The calibration data. Must have type `complex64` or
      `complex128`. Must have shape `[..., C, *R]`, where `R` is the shape of
      the calibration region, `C` is the number of coils and `...` is the batch
      shape, which can have any rank and must be broadcastable to the batch
      shape of `kspace`. `calib` is required when `method` is `"grappa"`. For
      other methods, this parameter is not relevant.
    kernel_size: An `int` or list of `ints`. The size of the GRAPPA
      kernel. Must have length equal to the image rank or number of spatial
      dimensions. If a scalar `int` is provided, the same size is used in all
      dimensions.
    weights_l2_regularizer: An optional `float`. The regularization
      factor for the L2 regularization term used to fit the GRAPPA weights.
      If 0.0, no regularization is applied.
    combine_coils: An optional `bool`. If `True`, multi-coil images
      are combined. Otherwise, the uncombined images are returned. Defaults to
      `True`.
    sensitivities: A `Tensor`. The coil sensitivity maps. Must have type
      `complex64` or `complex128`. Must have shape `[..., C, *S]`, where `S` is
      shape of the spatial dimensions, `C` is the number of coils and `...` is
      the batch shape, which can have any rank and must be broadcastable to the
      batch shape of `kspace`. Note that `sensitivities` are not used for the
      GRAPPA computation, but they are used for adaptive coil combination. If
      `sensitivities` are not provided, coil combination will be performed using
      the sum of squares method.
    return_kspace: An optional `bool`. If `True`, returns the filled
      *k*-space without performing the Fourier transform. In this case, coils
      are not combined regardless of the value of `combine_coils`.

  Returns:
    A `Tensor`. The reconstructed images. Has the same type as `kspace`. Has
    shape `[..., S]`, where `...` is the reconstruction batch shape and `S` is
    the spatial shape.

  References:
    .. [1] Griswold, M.A., Jakob, P.M., Heidemann, R.M., Nittka, M., Jellus, V.,
      Wang, J., Kiefer, B. and Haase, A. (2002), Generalized autocalibrating
      partially parallel acquisitions (GRAPPA). Magn. Reson. Med., 47:
      1202-1210. https://doi.org/10.1002/mrm.10171
  """
  kspace = tf.convert_to_tensor(kspace)
  calib = tf.convert_to_tensor(calib)
  mask = tf.convert_to_tensor(mask)

  # If mask has no holes, there is nothing to do.
  if tf.math.count_nonzero(tf.math.logical_not(mask)) == 0:
    return kspace

  # Use `mask` to infer rank.
  rank = mask.shape.rank

  # If an `int` was given for the kernel size, use isotropic kernel in all
  # dimensions.
  if isinstance(kernel_size, int):
    kernel_size = [kernel_size] * rank

  # Get multi-dimensional and flat indices for kernel center, e.g. [2, 2]
  # (multi), 12 (flat) for [5, 5] kernel. `kernel_center` is also used as half
  # the size of the kernel.
  kernel_center = [ks // 2 for ks in kernel_size]
  kernel_center_index = array_ops.ravel_multi_index(kernel_center, kernel_size)

  # Save batch shape for later, broadcast `calib` to match `kspace` and reshape
  # inputs to a single batch axis (except `mask`, which should have no batch
  # dimensions).
  kspace_shape = tf.shape(kspace)[-rank-1:] # No batch dims.
  calib_shape = tf.shape(calib)[-rank-1:] # No batch dims.
  batch_shape = tf.shape(kspace)[:-rank-1]
  if tf.math.reduce_prod(tf.shape(calib)[:-rank-1]) == 1:
    # Shared calibration. Do not broadcast, but maybe add batch dimension.
    calib = tf.reshape(calib, tf.concat([[1], calib_shape], 0))
  else:
    # General case. Calibration may not be shared for all inputs.
    calib = tf.broadcast_to(calib, tf.concat([batch_shape, calib_shape], 0))
  kspace = tf.reshape(kspace, tf.concat([[-1], kspace_shape], 0))
  calib = tf.reshape(calib, tf.concat([[-1], calib_shape], 0))
  batch_size = tf.shape(kspace)[0]
  num_coils = tf.shape(kspace)[1]

  # Move coil axis to the end, i.e. [batch, coil, *dims] -> [batch, *dims, coil]
  perm = [0, *list(range(2, rank + 2)), 1]
  kspace = tf.transpose(kspace, perm)
  calib = tf.transpose(calib, perm)

  # Initialize output tensor and fill with the measured values.
  full_shape = tf.concat([[batch_size], tf.shape(mask), [num_coils]], 0)
  measured_indices = tf.cast(tf.where(mask), tf.int32)
  measured_indices = _insert_batch_indices(measured_indices, batch_size)
  full_kspace = tf.scatter_nd(measured_indices,
                              tf.reshape(kspace, [-1, num_coils]),
                              full_shape)

  # Pad arrays so we can slide the kernel in the edges.
  paddings = tf.concat([[0], kernel_center, [0]], 0)
  paddings = tf.expand_dims(paddings, -1)
  paddings = tf.tile(paddings, [1, 2])
  full_kspace = tf.pad(full_kspace, paddings) # pylint:disable=no-value-for-parameter
  calib = tf.pad(calib, paddings) # pylint:disable=no-value-for-parameter
  mask = tf.pad(mask, paddings[1:-1, :], constant_values=False)

  # Extract all patches from the mask. We cast to `float32` because `bool` is
  # not currently supported in all devices for `_extract_patches` (TF v2.6).
  mask_patches = _extract_patches(
      tf.cast(mask[tf.newaxis, ..., tf.newaxis], tf.float32), kernel_size) > 0.5

  # Find the unique patterns among all the mask patches. `unique_inverse` are
  # the indices that reconstruct `mask_patches` from `unique_patches`.
  patch_array_shape = tf.shape(mask_patches, out_type=tf.int64)[1:-1]
  mask_patches = tf.reshape(
      mask_patches, [-1, tf.math.reduce_prod(kernel_size)])
  unique_patches, unique_inverse = tf.raw_ops.UniqueV2(x=mask_patches, axis=[0])
  unique_inverse = tf.cast(unique_inverse, tf.int64)
  unique_inverse = tf.reshape(unique_inverse, patch_array_shape)

  # Select only patches that:
  # - Have a hole in the center. Otherwise job is done!
  # - Are not empty. Otherwise there is nothing we can do!
  valid_patch_indices = tf.where(tf.math.logical_and(
      tf.math.logical_not(unique_patches[:, kernel_center_index]),
      tf.math.reduce_any(unique_patches, axis=-1)))
  valid_patch_indices = tf.squeeze(valid_patch_indices, axis=-1)

  # Get all overlapping patches of ACS.
  calib_patches = _extract_patches(calib, kernel_size)
  calib_patches = _flatten_spatial_axes(calib_patches)
  calib_patches = _split_last_dimension(calib_patches, num_coils)

  # For each geometry.
  for patch_index in valid_patch_indices:

    # Estimate the GRAPPA weights for current geometry. Get all possible
    # calibration patches with current geometry: sources (available data) and
    # targets (holes to fill). Given known sources and targets, estimate weights
    # using (possibly regularized) least squares.
    sources = tf.boolean_mask(calib_patches,
                              unique_patches[patch_index, :], axis=-2)
    sources = _flatten_last_dimensions(sources)
    targets = calib_patches[..., kernel_center_index, :]
    weights = tf.linalg.lstsq(sources, targets,
                              l2_regularizer=weights_l2_regularizer)

    # Now find all patch offsets (upper-left corners) and centers for current
    # geometry.
    patch_offsets = tf.where(unique_inverse == patch_index)
    patch_centers = tf.cast(patch_offsets + kernel_center, tf.int32)
    patch_centers = _insert_batch_indices(patch_centers, batch_size)

    # Collect all sources from partially measured `kspace` (all patches with
    # current geometry are pulled at the same time here).
    sources = image_ops.extract_glimpses(
        full_kspace, kernel_size, patch_offsets)
    sources = _split_last_dimension(sources, num_coils)
    sources = tf.boolean_mask(sources, unique_patches[patch_index, :], axis=-2)
    sources = _flatten_last_dimensions(sources)

    # Compute targets using the previously estimated weights.
    targets = tf.linalg.matmul(sources, weights)
    targets = tf.reshape(targets, [-1, num_coils])

    # Fill the holes.
    full_kspace = tf.tensor_scatter_nd_update(full_kspace,
                                              patch_centers,
                                              targets)

  # `full_kspace` was zero-padded at the beginning. Crop it to correct shape.
  full_kspace = array_ops.central_crop(
      full_kspace, tf.concat([[-1], full_shape[1:-1], [-1]], 0))

  # Move coil axis back. [batch, *dims, coil] -> [batch, coil, *dims]
  inv_perm = tf.math.invert_permutation(perm)
  full_kspace = tf.transpose(full_kspace, inv_perm)

  # Restore batch shape.
  result = tf.reshape(
      full_kspace, tf.concat([batch_shape, tf.shape(full_kspace)[1:]], 0))

  if return_kspace:
    return result

  # Inverse FFT to image domain.
  result = fft_ops.ifftn(result, axes=list(range(-rank, 0)), shift=True)

  # Combine coils if requested.
  if combine_coils:
    result = coil_ops.combine_coils(result,
                                    maps=sensitivities,
                                    coil_axis=-rank-1)

  return result


def _extract_patches(images, sizes):
  """Extract patches from N-D image.

  Args:
    images: A `Tensor` of shape `[batch_size, *spatial_dims, channels]`.
      `spatial_dims` must have rank 2 or 3.
    sizes: A list of `ints`. The size of the patches. Must have the same length
      as `spatial_dims`.

  Returns:
    A `Tensor` containing the extracted patches.

  Raises:
    ValueError: If rank is not 2 or 3.
  """
  rank = len(sizes)
  if rank == 2:
    patches = tf.image.extract_patches(
        images,
        sizes=[1, *sizes, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID')
  elif rank == 3:
    # `tf.extract_volume_patches` does not support complex tensors, so we do the
    # extraction for real and imaginary separately and then combine.
    if images.dtype.is_complex:
      patches_real = tf.extract_volume_patches(
          tf.math.real(images),
          ksizes=[1, *sizes, 1],
          strides=[1, 1, 1, 1, 1],
          padding='VALID')
      patches_imag = tf.extract_volume_patches(
          tf.math.imag(images),
          ksizes=[1, *sizes, 1],
          strides=[1, 1, 1, 1, 1],
          padding='VALID')
      patches = tf.dtypes.complex(patches_real, patches_imag)
    else:
      patches = tf.extract_volume_patches(
          images,
          ksizes=[1, *sizes, 1],
          strides=[1, 1, 1, 1, 1],
          padding='VALID')
  else:
    raise ValueError(f"Unsupported rank: {rank}")
  return patches


def _insert_batch_indices(indices, batch_size): # pylint: disable=missing-param-doc
  """Inserts batch indices into an array of indices.

  Given an array of indices with shape `[M, N]` which indexes into a tensor `x`,
  returns a new array with shape `[batch_size * M, N + 1]` which indexes into a
  tensor of shape `[batch_size] + x.shape`.
  """
  batch_indices = tf.expand_dims(tf.repeat(
      tf.range(batch_size), tf.shape(indices)[0]), -1)
  indices = tf.tile(indices, [batch_size, 1])
  indices = tf.concat([batch_indices, indices], -1)
  return indices


def _flatten_spatial_axes(images): # pylint: disable=missing-param-doc
  """Flatten the spatial axes of an image.

  If `images` has shape `[batch_size, *spatial_dims, channels]`, returns a
  `Tensor` with shape `[batch_size, prod(spatial_dims), channels]`.
  """
  shape = tf.shape(images)
  return tf.reshape(images, [shape[0], -1, shape[-1]])


def _split_last_dimension(x, size):
  """Splits the last dimension into two dimensions.

  Returns an array of rank `tf.rank(x) + 1` whose last dimension has size
  `size`.
  """
  return tf.reshape(x, tf.concat([tf.shape(x)[:-1], [-1, size]], 0))


def _flatten_last_dimensions(x):
  """Flattens the last two dimensions.

  Returns an array of rank `tf.rank(x) - 1`.
  """
  return tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))


@api_util.export("recon.partial_fourier", "recon.pf")
@deprecation.deprecated_args(
    deprecation.REMOVAL_DATE['0.19.0'],
    'Use argument `preserve_phase` instead.',
    ('return_complex', None))
def reconstruct_pf(kspace,
                   factors,
                   preserve_phase=None,
                   return_kspace=False,
                   return_complex=None,
                   method='zerofill',
                   **kwargs):
  """Reconstructs an MR image using partial Fourier methods.

  Args:
    kspace: A `Tensor`. The *k*-space data. Must have type `complex64` or
      `complex128`. Must have shape `[..., *K]`, where `K` are the spatial
      frequency dimensions. `kspace` should only contain the observed data,
      without zero-filling of any kind.
    factors: A list of `floats`. The partial Fourier factors. There must be a
      factor for each spatial frequency dimension. Each factor must be between
      0.5 and 1.0 and indicates the proportion of observed *k*-space values
      along the specified dimensions.
    preserve_phase: A `boolean`. If `True`, keeps the phase information in the
      reconstructed images. Although it is not possible to reconstruct
      high-frequency phase details from an incomplete k-space, a low resolution
      phase map can still be recovered. If `True`, the output images will
      be complex-valued.
    return_kspace: A `boolean`. If `True`, returns the filled *k*-space instead
      of the reconstructed images. This is always complex-valued.
    return_complex: A `boolean`. If `True`, returns complex instead of
      real-valued images.
    method: A `string`. The partial Fourier reconstruction algorithm. Must be
      one of `"zerofill"`, `"homodyne"` (homodyne detection method) or `"pocs"`
      (projection onto convex sets method).
    **kwargs: Additional method-specific keyword arguments. See Notes for
    details.

  Returns:
    A `Tensor` with shape `[..., *S]` where `S = K / factors`. Has type
    `kspace.dtype` if either `preserve_phase` or `return_kspace` is `True`, and
    type `kspace.dtype.real_dtype` otherwise.

  Notes:
    This function accepts some method-specific arguments:

    * `method="zerofill"` accepts no additional arguments.

    * `method="homodyne"` accepts the following additional keyword arguments:

      * **weighting_fn**: An optional `string`. The weighting function. Must be
        one of `"step"`, `"ramp"`. Defaults to `"ramp"`. `"ramp"` helps
        mitigate Gibbs artifact, while `"step"` has better SNR properties.

    * `method="pocs"` accepts the following additional keyword arguments:

      * **tol**: An optional `float`. The convergence tolerance. Defaults to
        `1e-5`.
      * **max_iter**: An optional `int`. The maximum number of iterations of the
        POCS algorithm. Defaults to `10`.

  References:
    .. [1] Noll, D. C., Nishimura, D. G., & Macovski, A. (1991). Homodyne
      detection in magnetic resonance imaging. IEEE transactions on medical
      imaging, 10(2), 154-163.
    .. [2] Haacke, E. M., Lindskogj, E. D., & Lin, W. (1991). A fast, iterative,
      partial-Fourier technique capable of local phase recovery. Journal of
      Magnetic Resonance (1969), 92(1), 126-145.
  """
  kspace = tf.convert_to_tensor(kspace)
  factors = tf.convert_to_tensor(factors)

  # Validate inputs.
  method = check_util.validate_enum(method, {'zerofill', 'homodyne', 'pocs'})
  tf.debugging.assert_greater_equal(factors, 0.5, message=(
    f"`factors` must be greater than or equal to 0.5, but got: {factors}"))
  tf.debugging.assert_less_equal(factors, 1.0, message=(
    f"`factors` must be less than or equal to 1.0, but got: {factors}"))
  preserve_phase = deprecation.deprecated_argument_lookup(
      'preserve_phase', preserve_phase, 'return_complex', return_complex)
  if preserve_phase is None:
    preserve_phase = False

  func = {'zerofill': _pf_zerofill,
          'homodyne': _pf_homodyne,
          'pocs': _pf_pocs}

  return func[method](kspace, factors,
                      preserve_phase=preserve_phase,
                      return_kspace=return_kspace,
                      **kwargs)


def _pf_zerofill(kspace, factors,
                 preserve_phase=False,
                 return_kspace=False):
  """Partial Fourier reconstruction using zero-filling.

  For the parameters, see `reconstruct_pf`.
  """
  dtype = kspace.dtype
  rank = tf.size(factors)
  _, (_, _, right_dims) = _compute_dimensions(
      tf.shape(kspace), factors)

  # Pad right part of k-space with zeros.
  paddings = tf.stack([tf.zeros_like(right_dims), right_dims], axis=-1)
  full_kspace = tf.pad(kspace, paddings)  # pylint: disable=no-value-for-parameter

  # Reconstruct the image and take its magnitude.
  image = _ifftn(full_kspace, rank)
  image = tf.math.abs(image)

  # If `preserve_phase` is `True`, estimate phase and put it back in image.
  if preserve_phase:
    phase_modulator = _estimate_phase_modulator(kspace, factors)
    image = tf.cast(image, dtype) * phase_modulator

  if return_kspace:
    return _fftn(tf.cast(image, dtype), rank)

  return image


def _pf_homodyne(kspace,
                 factors,
                 preserve_phase=False,
                 return_kspace=False,
                 weighting_fn='ramp'):
  """Partial Fourier reconstruction using homodyne detection.

  For the parameters, see `reconstruct_pf`.
  """
  # Data type of this operation.
  input_shape = tf.shape(kspace)
  dtype = kspace.dtype
  rank = tf.size(factors)

  _, (left_dims, centre_dims, right_dims) = _compute_dimensions(
      input_shape, factors)

  # Create zero-filled k-space.
  paddings = tf.stack([tf.zeros_like(right_dims), right_dims], axis=-1)
  full_kspace = tf.pad(kspace, paddings)  # pylint: disable=no-value-for-parameter

  # Compute weighting function. Weighting function is:
  # - 2.0 for the asymmetric part of the measured k-space.
  # - A ramp from 2.0 to 0.0 for the symmetric part of the measured k-space.
  # - 0.0 for the part of k-space that was not measured.
  weights = tf.constant(1.0, dtype=kspace.dtype)
  for index in range(factors.shape[0]):
    axis = -(index + 1)

    # Weighting for symmetric part of k-space.
    if weighting_fn == 'step':
      weights_centre = tf.ones([centre_dims[axis]], dtype=dtype)
    elif weighting_fn == 'ramp':
      if factors[-index-1] == 1.0:
        weights_centre = tf.ones([centre_dims[axis]], dtype=dtype)
      else:
        weights_centre = tf.cast(
            tf.linspace(2.0, 0.0, centre_dims[axis] + 2)[1:-1], dtype)
    else:
      raise ValueError(f"Unknown `weighting_fn`: {weighting_fn}")
    # Weighting for asymmetric part of k-space.
    weights_left = 2.0 * tf.ones([left_dims[axis]], dtype=dtype)
    weights_left = tf.cond(tf.math.greater(left_dims[axis], right_dims[axis]),
                           lambda: tf.concat([[1], weights_left[1:]], 0),  # pylint: disable=cell-var-from-loop
                           lambda: weights_left)  # pylint: disable=cell-var-from-loop
    weights_right = tf.zeros([right_dims[axis]], dtype=dtype)
    # Combine weights.
    weights_1d = tf.concat([weights_left, weights_centre, weights_right], 0)
    weights *= tf.reshape(weights_1d, [-1] + [1] * index)

  # Phase correction. Estimate a phase modulator from low resolution image using
  # symmetric part of k-space.
  phase_modulator = _estimate_phase_modulator(kspace, factors)

  # Compute image with following steps.
  # 1. Apply weighting function.
  # 2. Convert to image domain.
  # 3. Apply phase correction.
  # 4. Keep real part.
  full_kspace *= weights
  image = _ifftn(full_kspace, tf.size(factors))
  image *= tf.math.conj(phase_modulator)
  image = _real_non_negative(image)

  if preserve_phase:
    image = tf.cast(image, dtype) * phase_modulator

  if return_kspace:
    return _fftn(tf.cast(image, dtype), rank)

  return image


def _pf_pocs(kspace,
             factors,
             preserve_phase=False,
             return_kspace=False,
             max_iter=10,
             tol=1e-5):
  """Partial Fourier reconstruction using projection onto convex sets (POCS).

  For the parameters, see `reconstruct_pf`.
  """
  # Data type of this operation.
  input_shape = tf.shape(kspace)
  dtype = kspace.dtype
  rank = tf.size(factors)

  _, (_, _, right_dims) = _compute_dimensions(
      input_shape, factors)

  # Create zero-filled k-space.
  paddings = tf.stack([tf.zeros_like(right_dims), right_dims], axis=-1)
  full_kspace = tf.pad(kspace, paddings)  # pylint: disable=no-value-for-parameter

  # Generate a k-space mask which is True for measured samples, False otherwise.
  kspace_mask = tf.constant(True)
  # for i, factor in enumerate(tf.reverse(factors, [0])):
  for i in tf.range(tf.size(factors)):
    dim_partial = kspace.shape[-i-1]
    dim_full = full_kspace.shape[-i-1]
    kspace_mask = tf.math.logical_and(kspace_mask, tf.reshape(tf.concat(
        [tf.fill([dim_partial], True),
         tf.fill([dim_full - dim_partial], False)], 0),
            tf.concat([[-1], tf.repeat([1], [i])], 0)))

  # Estimate the phase modulator from central symmetric region of k-space.
  phase_modulator = _estimate_phase_modulator(kspace, factors)

  # Initial estimate of the solution.
  image = tf.zeros_like(full_kspace)

  # Type to hold state of the iteration.
  pocs_state = collections.namedtuple('pocs_state', ['i', 'x', 'r'])

  def stopping_criterion(i, state):
    return tf.math.logical_and(i < max_iter, state.r > tol)

  def pocs_step(i, state):
    prev = state.x
    # Project onto real non-negative set.
    image = _real_non_negative(prev)
    # Set the estimated phase.
    image = tf.cast(image, prev.dtype) * phase_modulator
    # Project onto data consistency set by replacing estimated k-space values
    # by measured ones.
    kspace = _fftn(image, tf.size(factors))
    kspace = tf.where(kspace_mask, full_kspace, kspace)
    image = _ifftn(kspace, tf.size(factors))
    # Phase demodulation.
    image *= tf.math.conj(phase_modulator)
    # Calculate the relative difference.
    diff = tf.math.abs(tf.norm(image - prev) / tf.norm(prev))
    return i + 1, pocs_state(i=i + 1, x=image, r=diff)

  i = tf.constant(0, dtype=tf.int32)
  state = pocs_state(i=0, x=image, r=1.0)
  _, state = tf.while_loop(stopping_criterion, pocs_step, [i, state])

  # Get real part.
  image = _real_non_negative(state.x)

  # Add phase, if requested.
  if preserve_phase:
    image = tf.cast(image, dtype) * phase_modulator

  # Convert to k-space, if requested.
  if return_kspace:
    return _fftn(tf.cast(image, dtype), rank)

  return image


def _compute_dimensions(input_shape, factors):
  """Returns the sizes of the left, centre and right k-space parts.

  * The right part is the unobserved part of k-space.
  * The left part is the observed part of k-space without symmetric lines and
    thus without phase data.
  * The centre part is the observed part of k-space with symmetric lines and
    thus with phase data.

  Note that, for `factor = 1.0`, the centre part is equal to `input_shape - 1`,
  the left part is equal to 1 (as the left-most element does not have a
  symmetric counterpart) and the right part is 0.

  Args:
    input_shape: A tensor containing the shape of the input tensor.
    factors: A tensor containing the partial Fourier factors.

  Returns:
    A tuple of:

    * output shape tensor
    * a tuple of three tensors containing the dimensions of the left, centre
      and right k-space parts along each dimension.
  """
  # Pad factors with one to have the same size as input shape.
  factors = tf.pad(factors, [[tf.size(input_shape) - tf.size(factors), 0]],
                   constant_values=1.0)
  # Compute output shape.
  output_shape = tf.cast(tf.cast(input_shape, tf.float32) / factors + 0.5,
                         tf.int32)
  right_dims = output_shape - input_shape
  left_dims = tf.where(tf.math.equal(right_dims, 0),
                       0,
                       right_dims + tf.math.mod(output_shape + 1, 2))
  centre_dims = output_shape - (left_dims + right_dims)

  return output_shape, (left_dims, centre_dims, right_dims)


def _estimate_phase_modulator(kspace, factors):  # pylint: disable=missing-param-doc
  """Estimate a phase modulator from central region of k-space."""
  _, (left_dims, centre_dims, right_dims) = _compute_dimensions(
      tf.shape(kspace), factors)

  # Central part of k-space.
  centre_kspace = tf.slice(kspace, begin=left_dims, size=centre_dims)
  # Pad to left and right.
  paddings = tf.stack([left_dims, right_dims], axis=-1)
  lowres_kspace = tf.pad(centre_kspace, paddings)  # pylint: disable=no-value-for-parameter
  lowres_image = _ifftn(lowres_kspace, tf.size(factors))

  # Compute the phase modulator.
  phase_modulator = tf.math.exp(tf.dtypes.complex(
      tf.constant(0.0, dtype=lowres_image.dtype.real_dtype),
      tf.math.angle(lowres_image)))
  return phase_modulator


_real_non_negative = lambda x: tf.math.maximum(0.0, tf.math.real(x))


_fftn = lambda x, rank: fft_ops.fftn(x, axes=tf.range(-rank, 0), shift=True)
_ifftn = lambda x, rank: fft_ops.ifftn(x, axes=tf.range(-rank, 0), shift=True)
