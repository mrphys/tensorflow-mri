Release 0.5.0
=============

This release focuses on the new `metrics` module and implements several
Keras metrics.

Major Features and Improvements
-------------------------------

* `tfmr.metrics`:

  * Added new confusion metrics module with binary/multiclass/multilabel metrics
    `Accuracy`, `Precision`, `Recall`, `IoU`, `F1Score`, `FBetaScore` and
    `TverskyIndex`. This module also exposes abstract base class
    `ConfusionMetric`, which does most of the work.
  * Added new image quality assessment metrics `PeakSignalToNoiseRatio`,
    `StructuralSimilarity` and `MultiscaleStructuralSimilarity`.
