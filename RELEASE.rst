Release 0.21.0
==============

This release contains new functionality for wavelet decomposition and
reconstruction and optimized Gram matrices for some linear operators. It also
redesigns the convex optimization module and contains some improvements to the
documentation.


Breaking Changes
----------------

* ``tfmri.convex``:

  * Argument ``ndim`` has been removed from all functions.
  * All functions will now require the domain dimension to be
    specified. Therefore, `domain_dimension` is now the first positional
    argument in several functions including ``ConvexFunctionIndicatorBall``,
    ``ConvexFunctionNorm`` and ``ConvexFunctionTotalVariation``. However, while
    this parameter is no longer optional, it is now possible to pass dynamic
    or static information as opposed to static only (at least in the general
    case, but specific operators may have additional restrictions).
  * For consistency and accuracy, argument ``axis`` of
    ``ConvexFunctionTotalVariation`` has been renamed to ``axes``.


Major Features and Improvements
-------------------------------

* ``tfmri.convex``:

  * Added new class ``ConvexFunctionL1Wavelet``, which enables image/signal
    reconstruction with L1-wavelet regularization.
  * Added new argument ``gram_operator`` to ``ConvexFunctionLeastSquares``,
    which allows the user to specify a custom, potentially more efficient Gram
    matrix.

* ``tfmri.linalg``:

  * Added new classes ``LinearOperatorNUFFT`` and ``LinearOperatorGramNUFFT``
    to enable the use of NUFFT as a linear operator.
  * Added new class ``LinearOperatorWavelet`` to enable the use of wavelets
    as a linear operator.

* ``tfmri.sampling``:

  * Added new ordering type ``sorted_half`` to ``radial_trajectory``.

* ``tfmri.signal``:

  * Added new functions ``wavedec`` and ``waverec`` for wavelet decomposition
    and reconstruction, as well as utilities ``wavelet_coeffs_to_tensor``,
    ``tensor_to_wavelet_coeffs``, and ``max_wavelet_level``.


Bug Fixes and Other Changes
---------------------------

* ``tfmri.recon``:

  * Improved error reporting for ``least_squares``.
