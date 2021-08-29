Release 0.5.0
=============

This release focuses on the new `metrics` module and implements several
Keras metrics. It also adds some image reconstruction functionality.

Major Features and Improvements
-------------------------------

* `tfmr`:

  * Added new method `"grappa"` to `reconstruct` operation, implementing
    generalized autocalibrating partially parallel acquisitions (GRAPPA).
  * Added new operation `reconstruct_partial_kspace` for partial Fourier (PF)
    reconstruction. Supported PF methods are zero-filling, homodyne detection
    and projection onto convex sets.
  * Added new operation `ravel_multi_index` to convert arrays of
    multi-dimensionalindices to arrays of flat indices.
  * Added new operation `extract_glimpses` to extract patches or windows at the
    specified locations from N-dimensional images.

* `tfmr.metrics`:

  * Added new confusion metrics module with multiple binary, multiclass and
    multilabel metrics: `Accuracy`, `TruePositiveRate`, `TrueNegativeRate`,
    `PositivePredictiveValue`, `NegativePredictiveValue`, `Precision`, `Recall`,
    `Sensitivity`, `Specificity`, `Selectivity`, `TverskyIndex`, `FBetaScore`,
    `F1Score` and `IoU`. This module also exposes the abstract base class
    `ConfusionMetric`.
  * Added new image quality assessment metrics module with 2D/3D
    `PeakSignalToNoiseRatio`, `StructuralSimilarity` and
    `MultiscaleStructuralSimilarity`.


Bug Fixes and Other Changes
---------------------------

* `tfmr`:

  * Added new keyword argument `coil_axis` to `coil_compression_matrix`
    operation.
