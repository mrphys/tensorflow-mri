Release 0.10.0
==============

This release focuses on MRI reconstruction, with new functionality and major
refactoring of image reconstruction module as well as the associated linear
algebra and convex optimization modules.

This release also bumps the supported TensorFlow version to 2.8.

Breaking Changes
----------------

* ``tfmri``:

  * ``compress_coils`` no longer accepts a ``matrix`` argument. Use
    ``SVDCoilCompressor`` instead.
  * ``compress_coils`` no longer accepts a ``tol`` argument.
  * ``coil_compression_matrix`` has been removed. Use ``SVDCoilCompressor``
    instead.
  * Keyword argument ``num_output_coils`` of ``compress_coils`` has been renamed
    to ``out_coils``.
  * Removed ``LinearOperatorFFT``, ``LinearOperatorNUFFT``,
    ``LinearOperatorParallelMRI``, ``LinearOperatorSensitivityModulation``,
    ``LinearOperatorInterp``, ``LinearOperatorRealWeighting``.
    Use ``LinearOperatorMRI`` instead.
  * Removed ``LinearOperatorDifference``. Use ``LinearOperatorFiniteDifference``
    instead.
  * ``ConvexOperator`` abstract base class and all its subclasses have been
    removed. Use ``ConvexFunction`` and/or one of its subclasses instead.
  * Removed ``reconstruct`` op. Use one of the new reconstruction ops instead
    (see below).
  * Removed ``reconstruct_partial_kspace`` op. Use ``reconstruct_pf`` instead.


Major Features and Improvements
-------------------------------

* ``tfmri``:

  * Added new ops ``broadcast_dynamic_shapes`` and ``broadcast_static_shapes``
    to broadcast multiple shapes.
  * ``estimate_coil_sensitivities`` with ``method='espirit'`` can now be called
    with a statically unknown number of coils (graph mode).
  * ``estimate_coil_sensitivities`` with ``method='espirit'`` will now set the
    static number of maps, if possible (graph mode).
  * Added new class ``SVDCoilCompressor`` for flexible coil compression
    functionality.
  * Added new ops ``ConvexFunction``,
    ``ConvexFunctionAffineMappingComposition``,
    ``ConvexFunctionLinearOperatorComposition``,
    ``ConvexFunctionL1Norm``, ``ConvexFunctionL2Norm``,
    ``ConvexFunctionL2NormSquared``, ``ConvexFunctionTikhonov``,
    ``ConvexFunctionTotalVariation``, ``ConvexFunctionQuadratic`` and
    ``ConvexFunctionLeastSquares``.
  * Added new image ops ``image_gradients`` and ``gradient_magnitude``.
  * Added new image ops ``gmsd``, ``gmsd2d`` and ``gmsd3d``.
  * Added new image op ``extract_and_scale_complex_part``.
  * Added new ops ``LinearOperatorMRI`` and ``LinearOperatorFiniteDifference``.
  * ``conjugate_gradient`` now has seamless support for linear operators with
    the imaging mixin.
  * Added new math op ``normalize_no_nan``.
  * Added new math ops ``block_soft_threshold``, ``shrinkage`` and
    ``soft_threshold``.
  * Added new op ``admm_minimize`` implementing the ADMM algorithm.
  * Added new reconstruction ops ``reconstruct_adj``, ``reconstruct_lstsq``,
    ``reconstruct_grappa``, ``reconstruct_sense`` and ``reconstruct_pf``.
  * Added new signal ops ``hann`` and ``atanfilt``.
  * ``filter_kspace`` now supports Cartesian inputs and additional keyword
    arguments for filtering function via ``filter_kwargs``.
  * Added new ops ``density_grid`` and ``random_sampling_mask`` for generation
    of Cartesian masks.
  * ``radial_density``: added new parameter ``tiny_number``, mirroring the one
    in ``radial_trajectory``.

* ``tfmri.callbacks``:

  * ``TensorBoardImages``: added new parameters ``display_fn``, ``concat_axis``,
    ``feature_keys``, ``label_keys``, ``prediction_keys`` and ``complex_part``.
    These enable different levels of customization of the display function.

* ``tfmri.metrics``:

  * Added new argument ``multichannel`` to ``PeakSignalToNoiseRatio``,
    ``StructuralSimilarity`` and ``MultiscaleStructuralSimilarity``. This
    enables these metrics to accept inputs that do not have a channel axis,
    by setting ``multichannel=False``.
  * Added new argument ``complex_part`` to ``PeakSignalToNoiseRatio``,
    ``StructuralSimilarity`` and ``MultiscaleStructuralSimilarity``. This
    enables these metrics to accept complex inputs and calculate the metric
    from the specified part.
