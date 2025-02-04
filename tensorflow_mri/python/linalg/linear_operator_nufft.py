# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Non-uniform Fourier linear operators."""

import warnings

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_nd
from tensorflow_mri.python.linalg import linear_operator_util
from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import traj_ops
from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import tensor_util
from tensorflow_mri.python.util import types_util


@api_util.export("linalg.LinearOperatorNUFFT")
@linear_operator_nd.make_linear_operator_nd
class LinearOperatorNUFFT(linear_operator_nd.LinearOperatorND):
  r"""Linear operator acting like a [batch] nonuniform Fourier matrix.

  Performs an N-dimensional discrete Fourier transform via the nonuniform fast
  Fourier transform (NUFFT) algorithm. Let $A$ represent this linear operator,
  then:

  - The forward operator $A$ evaluates the forward, type-2 NUFFT (signal domain
    to frequency domain, uniform to nonuniform).
  - The adjoint operator $A^H$ evaluates the backward, type-1 NUFFT
    (frequency domain to signal domain, nonuniform to uniform).

  The dimensionality of the grid $n = n_0 \times ... \times n_d$ is determined
  by `domain_shape`. The $m$ non-uniform sampling locations in the frequency
  domain are defined by `points`.

  ```{rubric} Inverse NUFFT
  ```
  ```{note}
  The NUFFT operator is not generally invertible, so calling `inverse` or
  `solve` (or the related `solvevec` and `solvevec_nd`) will raise an error.
  ```

  However, you can solve $Ax = b$ in the least-squares sense.

  One approach is to use this operator's `lstsq` method (or one of the related
  methods `lstsqvec` and `lstsqvec_nd`).

  ```{attention}
  If you intend to use `lstsq`, `lstsqvec`, or `lstsqvec_nd`, you should
  consider providing `crosstalk_inverse` (see below). If you do not provide
  this argument, the solution will be computed using a potentially very slow
  algorithm which requires conversion to a dense matrix.
  ```

  Alternatively, you could use `tfmri.linalg.lsqr` or
  `tfmri.linalg.conjugate_gradient` to solve the least-squares problem
  iteratively.

  ```{rubric} Fourier crosstalk matrix
  ```
  The Fourier crosstalk matrix is the matrix $D = A A^H$ (if $m < n$) or the
  matrix $D = A^H A$ (if $m > n$). The solution to the least-squares problem
  can be written in terms of $D$ as $x = A^H D^{-1} b$ (if $m < n$) or
  $x = D^{-1} A b$ (if $m$ > $n$).

  Hence, if $D{-1}$ is known, the least-squares problem can be solved without
  performing an explicit inversion. The argument `crosstalk_inverse` allows
  you to provide $D^{-1}$.

  The matrix $D$ (and hence, $D^{-1}$) is dependent on the sampling locations
  `points`. For arbitrary sampling patterns, this matrix is full and requires
  $O(l^2)$ storage, with a runtime complexity of $O(l^3)$ for matrix-matrix
  multiplication (where $l = \min{(m, n)}$). This is clearly impractical for
  large $l$. Furthermore, in this case one might as well just store and apply
  $A^{-1}$ directly.

  A more interesting use of `crosstalk_inverse` is to provide an approximation
  to $D^{-1}$ with a more favorable structure. A common choice is to use a
  diagonal matrix, which requires only $O(l)$ storage and whose matrix-matrix
  product runs in $O(l^2)$ time. In MRI, this is often referred to as
  **sampling density compensation**.

  ```{tip}
  If `weights` are your density compensation weights, use
  `crosstalk_inverse=tfmri.linalg.LinearOperatorDiag(weights)`.
  ```

  ```{rubric} TLDR: how do I invert the NUFFT?
  ```
  Essentially, you have three options:

  1. **Direct solve** (not recommmended). Simply call `lstsq` (or one of the
     related methods `lstsqvec` and `lstsqvec_nd`). This is the most
     straightforward approach, but it is likely to be very slow for large $l$.
     If you can't or don't want to provide $D^{-1}$ (or an approximation
     thereof), you're probably better off using method 3.
  2. **Direct solve with crosstalk approximation** (in MRI, sometimes called
     the **conjugate phase** method): If the inverse of the Fourier crosstalk
     matrix $D^{-1}$ has favorable structure (i.e., it does not have large
     storage requirements and it can be applied quickly), or you can use an
     approximation which does, specify `crosstalk_inverse` and then use `lstsq`
     (or one of the related methods `lstsqvec` and `lstsqvec_nd`). In MRI, a
     common choice of approximation is a diagonal matrix containing whose
     diagonal elements are the density compensation weights. Under these
     conditions, this method is probably the fastest, but might compromise
     accuracy depending on your choice of $D^{-1}$.
  3. **Iterative solve**: If you do not know `D{-1}`, or if accuracy is
     paramount, use `tfmri.linalg.lsqr` or `tfmri.linalg.conjugate_gradient`
     to solve the least-squares problem iteratively. This method is likely
     to be slower than method 2 but faster than method 3 due to its iterative
     nature.
  ```

  ```{seealso}
  `tfmri.linalg.LinearOperatorFFT` for uniformly sampled Fourier transforms.
  ```

  Args:
    domain_shape: A 1D integer `tf.Tensor`. The domain shape of this
      operator. This is usually the shape of the image but may include
      additional leading dimensions. The trailing `d` dimensions (inferred
      from `points`) are the signal dimensions, and any additional leading
      dimensions are technically batch dimensions which are included in the
      domain rather than the batch.
    points: A `tf.Tensor` of type `float32` or `float64`. Contains the
      non-uniform sampling locations in the frequency domain. Must have
      shape `[..., m, d]`, where `d` is the dimension of the Fourier transform
      (must be 1, 2 or 3), `m` is the number of samples and `...` is the
      batch shape, which can have any number of dimensions. Must be in the
      range $[-\pi, \pi]$.
      ```{tip}
      In MRI, this is the *k*-space trajectory.
      ```
    crosstalk_inverse: A `tf.Tensor` or `tf.linalg.LinearOperator` of shape
      `[..., l, l]` representing the inverse of the Fourier crosstalk
      matrix [2], where $l = \min{(m, n)}$. This matrix is used to simplify the
      computation of the pseudo-inverse $A^{+}$ and/or to solve the
      least-squares problem defined by this operator. Ideally this matrix
      should be equal to $(A A^H)^{-1}$ (if $m < n$) or $(A^H A)^{-1}$
      (if $m > n$), where $A$ is this operator, but you can also provide a
      suitable approximation with a more favorable structure. Defaults to
      `None`.
      ```{attention}
      If you intend to use `lstsq`, `lstsqvec`, or `lstsqvec_nd`, you are
      strongly encouraged to provide `crosstalk_inverse`. If you do not,
      these methods will use a potentially very slow algorithm which requires
      conversion to a dense matrix.
      ```
      ```{warning}
      This operator will not check `crosstalk_inverse` for correctness. It is
      your responsibility to ensure that it is reasonable your purposes.
      ```
      ```{tip}
      In MRI, you can use `crosstalk_inverse` for density compensation by
      specifying a diagonal operator whose diagonal elements are the density
      compensation weights.
      ```
      ```{tip}
      If you do not need to invert this operator, you can safely ignore this
      argument.
      ```
    is_non_singular: A boolean, or `None`. Whether this operator is expected
      to be non-singular. Defaults to `None`.
    is_self_adjoint: A boolean, or `None`. Whether this operator is expected
      to be equal to its Hermitian transpose. If `dtype` is real, this is
      equivalent to being symmetric. Defaults to `False`.
    is_positive_definite: A boolean, or `None`. Whether this operator is
      expected to be positive definite, meaning the quadratic form $x^H A x$
      has positive real part for all nonzero $x$. Note that an operator [does
      not need to be self-adjoint to be positive definite](https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices)
      Defaults to `None`.
    is_square: A boolean, or `None`. Expect that this operator acts like a
      square matrix (or a batch of square matrices). Defaults to `False`.
    name: An optional `str`. The name of this operator.

  Example:
    >>> # Create some data.
    >>> image_shape = (128, 128)
    >>> image = tfmri.image.phantom(shape=image_shape, dtype=tf.complex64)
    >>> trajectory = tfmri.sampling.radial_trajectory(
    ...     base_resolution=128, views=129, flatten_encoding_dims=True)
    >>> density = tfmri.sampling.radial_density(
    ...     base_resolution=128, views=129, flatten_encoding_dims=True)
    >>> # Create a density compensation matrix. This will be used to invert
    >>> # the operator more efficiently.
    >>> weights = tf.math.reciprocal(density)
    >>> linop_density = tf.linalg.LinearOperatorDiag(weights)
    >>> # Create a NUFFT operator.
    >>> linop = tfmri.linalg.LinearOperatorNUFFT(
    ...     image_shape, points=trajectory, crosstalk_inverse=linop_density)
    >>> # Compute k-space by applying the forward operator.
    >>> kspace = linop.matvec_nd(image)
    >>> # Reconstruct the image by solving the corresponding least-squares
    >>> # problem.
    >>> recon = linop.lstsqvec_nd(kspace)

  References:
    1. A. H. Barnett, J. Magland, and L. af Klinteberg, "A Parallel Nonuniform
       Fast Fourier Transform Library Based on an "Exponential of Semicircle"
       Kernel", *SIAM Journal on Scientific Computing*, vol. 41, no. 5,
       pp. C479-C504, 2019,
       doi: [10.1137/18M120885X](https://doi.org/10.1137/18M120885X)
    2. Y. Shih, G. Wright, J. Anden, J. Blaschke, and A. H. Barnett,
       "cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform
       FFTs,” in *2021 IEEE International Parallel and Distributed Processing
       Symposium Workshops (IPDPSW)*, 2021, pp. 688-697.
       doi: [10.1109/IPDPSW52791.2021.00105](https://doi.org/10.1109/IPDPSW52791.2021.00105)
    3. J. A. Fessler and B. P. Sutton, "Nonuniform fast Fourier transforms
       using min-max interpolation", *IEEE Transactions on Signal Processing*,
       vol. 51, no. 2, pp. 560-574, 2003,
       doi: [10.1109/TSP.2002.807005](https://doi.org/10.1109/TSP.2002.807005)
    4. H. H. Barrett, J. L. Denny, R. F. Wagner, and K. J. Myers, "Objective
       assessment of image quality. II. Fisher information, Fourier crosstalk,
       and figures of merit for task performance", *J. Opt. Soc. Am. A*,
       vol. 12, no. 5, pp. 834-852, May 1995,
       doi: [10.1364/JOSAA.12.000834](https://doi.org/10.1364/josaa.12.000834)
  """
  def __init__(self,
               domain_shape,
               points,
               crosstalk_inverse=None,
               is_non_singular=None,
               is_self_adjoint=False,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorNUFFT"):

    parameters = dict(
        domain_shape=domain_shape,
        points=points,
        crosstalk_inverse=crosstalk_inverse,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    # Check non-reference types.
    types_util.assert_not_ref_type(domain_shape, "domain_shape")

    # Get domain shapes.
    self._domain_shape_static, self._domain_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))

    # Validate the remaining inputs.
    self._points = tf.convert_to_tensor(points, name="points")
    if self._points.dtype not in (tf.float32, tf.float64):
      raise TypeError(
          f"points must be a float32 or float64 tensor, "
          f"not {str(self._points.dtype)}")

    # Get dtype for this operator.
    dtype = tensor_util.get_complex_dtype(self._points.dtype)

    # We infer the operation's rank from the points.
    self._ndim_static = self._points.shape[-1]
    self._rank_dynamic = tf.shape(self._points)[-1]
    # The domain rank is >= the operation rank.
    domain_rank_static = self._domain_shape_static.rank
    domain_rank_dynamic = tf.shape(self._domain_shape_dynamic)[0]
    # The difference between this operation's rank and the domain rank is the
    # number of extra dims.
    extra_dims_static = domain_rank_static - self._ndim_static
    extra_dims_dynamic = domain_rank_dynamic - self._rank_dynamic

    # The grid shape are the last `rank` dimensions of domain_shape. We don't
    # need the static grid shape.
    self._grid_shape = self._domain_shape_dynamic[-self._rank_dynamic:]

    # We need to do some work to figure out the batch shapes. This operator
    # could have a batch shape, if the points have a batch shape. However,
    # we allow the user to include one or more batch dimensions in the domain
    # shape, if they so wish. Therefore, not all batch dimensions in the
    # points are necessarily part of the batch shape.

    # The total number of dimensions in `points` is equal to
    # `batch_dims + extra_dims + 2`.
    # Compute the true batch shape (i.e., the batch dimensions that are
    # NOT included in the domain shape).
    batch_dims_dynamic = tf.rank(self._points) - extra_dims_dynamic - 2
    if (self._points.shape.rank is not None and
        extra_dims_static is not None):
      # We know the total number of dimensions in `points` and we know
      # the number of extra dims, so we can compute the number of batch dims
      # statically.
      batch_dims_static = self._points.shape.rank - extra_dims_static - 2
    else:
      # We are missing at least some information, so the number of batch
      # dimensions is unknown.
      batch_dims_static = None

    self._batch_shape_dynamic = tf.shape(self._points)[:batch_dims_dynamic]
    if batch_dims_static is not None:
      self._batch_shape_static = self._points.shape[:batch_dims_static]
    else:
      self._batch_shape_static = tf.TensorShape(None)

    # Compute the "extra" shape. This shape includes those dimensions which
    # are not part of the NUFFT (e.g., they are effectively batch dimensions),
    # but which are included in the domain shape rather than in the batch shape.
    extra_shape_dynamic = self._domain_shape_dynamic[:-self._rank_dynamic]
    if self._ndim_static is not None:
      extra_shape_static = self._domain_shape_static[:-self._ndim_static]
    else:
      extra_shape_static = tf.TensorShape(None)

    # Check that the "extra" shape in `domain_shape` and `points` are
    # compatible for broadcasting.
    shape1, shape2 = extra_shape_static, self._points.shape[:-2]
    try:
      tf.broadcast_static_shape(shape1, shape2)
    except ValueError as err:
      raise ValueError(
          f"The \"batch\" shapes in `domain_shape` and `points` are not "
          f"compatible for broadcasting: {shape1} vs {shape2}") from err

    # Compute the range shape.
    self._range_shape_dynamic = tf.concat(
        [extra_shape_dynamic, tf.shape(self._points)[-2:-1]], 0)
    self._range_shape_static = extra_shape_static.concatenate(
        self._points.shape[-2:-1])

    # Set inverse of Fourier crosstalk matrix.
    # This needs to be done after setting all the shapes, as `self.shape`
    # must be valid at this point.
    self._crosstalk_inverse = crosstalk_inverse
    if self._crosstalk_inverse is not None:
      if not isinstance(self._crosstalk_inverse, tf.linalg.LinearOperator):
        # Not a linear operator, assume a full matrix was passed.
        self._crosstalk_inverse = tf.linalg.LinearOperatorFullMatrix(
            self._crosstalk_inverse)
      if not self._crosstalk_inverse.shape[-2:-1].is_compatible_with(
          self._crosstalk_inverse.shape[-1:]):
        raise ValueError(
            f"The crosstalk matrix must be square. Got shape "
            f"{self._crosstalk_inverse.shape}.")
      if self.shape[-2:].is_fully_defined():
        if self.shape[-2] < self.shape[-1]:
          if not self._crosstalk_inverse.shape[-2:-1].is_compatible_with(
              self.shape[-2:-1]):
            raise ValueError(
                f"The crosstalk matrix must have the same number of rows as "
                f"this operator. Got shape {self._crosstalk_inverse.shape} for "
                f"operator shape {self.shape}.")
        else:
          if not self._crosstalk_inverse.shape[-1:].is_compatible_with(
              self.shape[-1:]):
            raise ValueError(
                f"The crosstalk matrix must have the same number of columns as "
                f"this operator. Got shape {self._crosstalk_inverse.shape} for "
                f"operator shape {self.shape}.")

    # Compute normalization factors.
    self._norm_factor = tf.math.reciprocal(
        tf.math.sqrt(tf.cast(tf.math.reduce_prod(self._grid_shape), dtype)))

    # Default tolerance for NUFFT.
    self._tol = {tf.complex64: 1e-6, tf.complex128: 1e-12}[dtype]

    super().__init__(dtype,
                     is_non_singular=is_non_singular,
                     is_self_adjoint=is_self_adjoint,
                     is_positive_definite=is_positive_definite,
                     is_square=is_square,
                     parameters=parameters,
                     name=name)

  def _matvec_nd(self, x, adjoint=False):
    if adjoint:
      x = fft_ops.nufft(x, self._points,
                        grid_shape=self._grid_shape,
                        transform_type='type_1',
                        fft_direction='backward',
                        tol=self._tol)
      x *= self._norm_factor
    else:
      x = fft_ops.nufft(x, self._points,
                        transform_type='type_2',
                        fft_direction='forward',
                        tol=self._tol)
      x *= self._norm_factor
    return x

  def _solvevec_nd(self, rhs, adjoint=False):
    raise ValueError(
        f"{self.name} is not invertible. If you intend to solve the "
        f"associated least-squares problem, use `lstsq`, `lstsqvec` or "
        f"`lstsqvec_nd`.")

  def _lstsqvec_nd(self, rhs, adjoint=False):
    if self._crosstalk_inverse is None:
      warnings.warn(
          f"{self.name}: Using (possibly slow) implementation of lstsq which "
          f"requires conversion to a dense matrix and O(n^3) operations. "
          f"For a more efficient computation, consider specifying the "
          f"`crosstalk_inverse` argument (see class documentation) or using "
          f"an iterative solver such as `tfmri.linalg.lsqr` or "
          f"`tfmri.linalg.conjugate_gradient`.")
      rhs = tf.expand_dims(rhs, -1)
      x = linear_operator_util.matrix_solve_ls_with_broadcast(
          self.to_dense(), rhs, adjoint=adjoint)
      x = tf.squeeze(x, -1)
      return x
    if self.shape[-2:].is_fully_defined():
      # We know the static shape of the operator, so we can select the
      # appropriate code path when building the graph.
      if (self.shape[-2] < self.shape[-1]) ^ adjoint:  # pylint: disable=no-else-return
        return self._lstsqvec_nd_underdetermined(rhs, adjoint=adjoint)
      else:
        return self._lstsqvec_nd_overdetermined(rhs, adjoint=adjoint)
    else:
      # We don't know the static shape of the operator, so we need to
      # defer the selection of the code path until runtime.
      return tf.cond(
          tf.math.logical_xor(
              tf.math.less(self.shape_tensor()[-2], self.shape_tensor()[-1]),
              adjoint),
          lambda: self._lstsqvec_nd_underdetermined(rhs, adjoint=adjoint),
          lambda: self._lstsqvec_nd_overdetermined(rhs, adjoint=adjoint))

  def _lstsqvec_nd_underdetermined(self, rhs, adjoint=False):
    # Solve A x = b as A^H (A A^H)^-1 b, where (A A^H)^-1 is the inverse of
    # the Fourier crosstalk matrix.
    if isinstance(self._crosstalk_inverse,
                  linear_operator_nd.LinearOperatorND):
      rhs = self._crosstalk_inverse.matvec_nd(rhs)
    else:
      if adjoint:
        rhs = self.flatten_domain_shape(rhs)
      else:
        rhs = self.flatten_range_shape(rhs)
      rhs = self._crosstalk_inverse.matvec(rhs)
      if adjoint:
        rhs = self.expand_domain_dimension(rhs)
      else:
        rhs = self.expand_range_dimension(rhs)
    x = self._matvec_nd(rhs, adjoint=(not adjoint))
    return x

  def _lstsqvec_nd_overdetermined(self, rhs, adjoint=False):
    # Solve A x = b as (A^H A)^-1 A^H b, where (A^H A)^-1 is the inverse of
    # the Fourier crosstalk matrix.
    x = self._matvec_nd(rhs, adjoint=(not adjoint))
    if isinstance(self._crosstalk_inverse,
                  linear_operator_nd.LinearOperatorND):
      x = self._crosstalk_inverse.matvec_nd(x)
    else:
      if adjoint:
        x = self.flatten_range_shape(x)
      else:
        x = self.flatten_domain_shape(x)
      x = self._crosstalk_inverse.matvec(x)
      if adjoint:
        x = self.expand_range_dimension(x)
      else:
        x = self.expand_domain_dimension(x)
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

  def _ndim(self):
    return self._ndim_static

  @property
  def points(self):
    return self._points

  @property
  def crosstalk_inverse(self):
    return self._crosstalk_inverse

  @property
  def _composite_tensor_fields(self):
    return ('domain_shape', 'points', 'crosstalk_inverse')

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ('domain_shape',)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {'points': 2, 'crosstalk_inverse': 0}


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
      `[..., m, n]`, where `n` is the rank (number of dimensions), `m` is
      the number of samples and `...` is the batch shape, which can have any
      number of dimensions.
    density: A `tf.Tensor` of type `float32` or `float64`. Contains the
      sampling density at each point in `trajectory`. Must have shape
      `[..., m]`, where `m` is the number of samples and `...` is the batch
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
    trajectory = self._trajectory
    weights = self._weights
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
    if self._norm == 'ortho':
      kernel *= tf.cast(tf.math.sqrt(2.0 ** self.rank), kernel.dtype)

    # Put the kernel in Fourier space.
    fft_axes = list(range(-self.rank, 0))
    fft_norm = self._norm or "backward"
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
      temp = self._trajectory
      self._trajectory = trajectory
    x = super()._transform(x, adjoint=True)
    if trajectory is not None:
      self._trajectory = temp
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


@api_util.export("linalg.nudft_matrix")
def nudft_matrix(domain_shape, points):
  """Constructs a non-uniform discrete Fourier transform (NUDFT) matrix."""
  domain_shape_static, domain_shape_dynamic = (
      tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))
  if domain_shape_static.is_fully_defined():
    domain_shape = domain_shape_static.as_list()
  else:
    domain_shape = domain_shape_dynamic

  # For reshape.
  if domain_shape_static.rank is not None:
    grid_shape = [-1, domain_shape_static.rank]
  else:
    grid_shape = tf.concat([[-1], tf.size(domain_shape)], 0)

  grid = traj_ops.frequency_grid(
      domain_shape, max_val=tf.constant(0.5, dtype=points.dtype))
  grid = tf.reshape(grid, grid_shape)
  grid *= tf.cast(domain_shape, dtype=points.dtype)

  m = tf.linalg.matmul(points, tf.linalg.matrix_transpose(grid))
  m = tf.math.exp(tf.dtypes.complex(
      tf.constant(0.0, dtype=points.dtype), tf.math.negative(m)))
  m /= tf.math.sqrt(tf.cast(tf.math.reduce_prod(domain_shape), m.dtype))

  return m
