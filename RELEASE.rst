Release 0.5.0
=============

This release focuses on the new `metrics` module and implementing several
Keras metrics.

Major Features and Improvements
-------------------------------

* `tfmr.metrics`:

  * Added new confusion metrics module with classes `IoU`, `F1Score`,
    `FBetaScore`, `TverskyIndex` and `FocalTverskyIndex`. This module also
    exposes abstract base class `ConfusionMetric`, which does most of the work.
  * Added new image quality assessment metrics `PeakSignalToNoiseRatio`,
    `StructuralSimilarity` and `MultiscaleStructuralSimilarity`.
